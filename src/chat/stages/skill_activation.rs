//! Skill Auto-Activation Stage for the Chat Enrichment Pipeline.
//!
//! Matches the user message against skill trigger patterns (Regex, FileGlob)
//! and injects activated skills' context_templates into the enrichment context.
//!
//! This stage reuses the core trigger evaluation logic from `skills::activation`
//! but adapts it for the chat enrichment pipeline (free-text message input
//! instead of structured tool_input).
//!
//! # Activation flow
//!
//! 1. Load all matchable skills (Active/Emerging) for the project
//! 2. Evaluate each skill's trigger patterns against the user message
//! 3. Filter by confidence threshold, sort descending, take top N
//! 4. Inject context_template for each activated skill
//! 5. Async boost energy of activated skills (Hebbian reinforcement)

use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, ParallelEnrichmentStage, StageOutput,
};
use crate::neo4j::traits::GraphStore;
use crate::neurons::intent::{IntentDetector, QueryIntentMode};
use crate::skills::activation::evaluate_skill_match;
use crate::skills::models::SkillNode;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the skill activation stage.
#[derive(Debug, Clone)]
pub struct SkillActivationConfig {
    /// Minimum confidence score for activation (default: 0.7).
    pub confidence_threshold: f64,
    /// Maximum number of skills to activate simultaneously (default: 3).
    pub max_activated_skills: usize,
    /// Energy boost applied to activated skills (default: 0.1).
    pub energy_boost: f64,
}

impl Default for SkillActivationConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            max_activated_skills: 3,
            energy_boost: 0.1,
        }
    }
}

// ============================================================================
// Stage implementation
// ============================================================================

/// Enrichment stage that auto-activates skills based on message content.
pub struct SkillActivationStage {
    graph: Arc<dyn GraphStore>,
    config: SkillActivationConfig,
    /// Trajectory collector for decision capture (fire-and-forget).
    collector: Option<Arc<neural_routing_runtime::TrajectoryCollector>>,
}

impl SkillActivationStage {
    /// Create a new skill activation stage.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            graph,
            config: SkillActivationConfig::default(),
            collector: None,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(graph: Arc<dyn GraphStore>, config: SkillActivationConfig) -> Self {
        Self {
            graph,
            config,
            collector: None,
        }
    }

    /// Attach a trajectory collector for decision capture.
    pub fn with_collector(
        mut self,
        collector: Arc<neural_routing_runtime::TrajectoryCollector>,
    ) -> Self {
        self.collector = Some(collector);
        self
    }

    /// Extract file paths mentioned in the user message.
    ///
    /// Looks for patterns like `src/foo/bar.rs`, `./path/to/file`, etc.
    /// Used for FileGlob trigger matching.
    fn extract_file_paths(message: &str) -> Vec<String> {
        crate::utils::file_path_extractor::extract_file_paths(message)
    }

    /// Match skills against the user message.
    ///
    /// Uses the existing `evaluate_skill_match` function with:
    /// - `pattern`: the full user message (for regex matching)
    /// - `file_context`: extracted file paths concatenated (for file_glob matching)
    async fn match_skills(&self, message: &str, project_id: Uuid) -> Result<Vec<(SkillNode, f64)>> {
        // Load matchable skills
        let skills = self.graph.get_skills_for_project(project_id).await?;
        let matchable: Vec<_> = skills.into_iter().filter(|s| s.is_matchable()).collect();

        if matchable.is_empty() {
            return Ok(Vec::new());
        }

        // Extract file paths for FileGlob matching
        let file_paths = Self::extract_file_paths(message);
        let file_context = if file_paths.is_empty() {
            None
        } else {
            Some(file_paths.join(" "))
        };

        // Evaluate each skill
        let mut matches: Vec<(SkillNode, f64)> = Vec::new();
        for skill in matchable {
            let confidence = evaluate_skill_match(&skill, Some(message), file_context.as_deref());
            if confidence >= self.config.confidence_threshold {
                matches.push((skill, confidence));
            }
        }

        // Sort by confidence descending
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to max_activated_skills
        matches.truncate(self.config.max_activated_skills);

        Ok(matches)
    }

    /// Resolve project_id from project_slug via the graph.
    async fn resolve_project_id(&self, slug: &str) -> Result<Option<Uuid>> {
        match self.graph.get_project_by_slug(slug).await {
            Ok(Some(project)) => Ok(Some(project.id)),
            Ok(None) => Ok(None),
            Err(e) => {
                warn!(
                    "[skill_activation] Failed to resolve project slug '{}': {}",
                    slug, e
                );
                Ok(None)
            }
        }
    }
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for SkillActivationStage {
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
        let mut output = StageOutput::new(self.name());

        // Need a project scope for skill matching
        let project_id = if let Some(id) = input.project_id {
            id
        } else if let Some(ref slug) = input.project_slug {
            match self.resolve_project_id(slug).await? {
                Some(id) => id,
                None => return Ok(output), // No project found, skip
            }
        } else {
            return Ok(output); // No project scope, skip
        };

        // Match skills against the message
        let matches = self.match_skills(&input.message, project_id).await?;

        if matches.is_empty() {
            // No skill match — detect intent from message keywords as fallback
            let intent = detect_intent_from_message(&input.message);
            output.set_hint("intent", intent);
            return Ok(output);
        }

        // Emit intent hint based on matched skill tags/names
        let intent = detect_intent_from_skills(&matches, &input.message);
        output.set_hint("intent", intent);

        // Build enrichment content from activated skills
        let mut content_parts: Vec<String> = Vec::new();
        let mut skill_ids_to_boost: Vec<Uuid> = Vec::new();

        for (skill, confidence) in &matches {
            let mut section = format!(
                "### {} (confidence: {:.0}%)\n",
                skill.name,
                confidence * 100.0
            );

            // Add context_template if available
            if let Some(ref template) = skill.context_template {
                if !template.is_empty() {
                    section.push_str(template);
                    section.push('\n');
                }
            } else {
                // Fallback: add skill description
                if !skill.description.is_empty() {
                    section.push_str(&skill.description);
                    section.push('\n');
                }
            }

            content_parts.push(section);
            skill_ids_to_boost.push(skill.id);
        }

        if !content_parts.is_empty() {
            output.add_section("Activated Skills", content_parts.join("\n"), self.name());
        }

        // ── Trajectory collection: record skill activation decision ────────
        if let Some(ref collector) = self.collector {
            let alternatives_count = matches.len();
            let (chosen_name, chosen_confidence) = matches
                .first()
                .map(|(s, c)| (s.name.clone(), *c))
                .unwrap_or_else(|| ("none".to_string(), 0.0));

            let touched: Vec<neural_routing_runtime::TouchedEntity> = matches
                .iter()
                .map(|(s, _)| neural_routing_runtime::TouchedEntity {
                    entity_type: "Skill".to_string(),
                    entity_id: s.id.to_string(),
                    access_mode: "activate".to_string(),
                    relevance: Some(chosen_confidence),
                })
                .collect();

            collector.record_decision(neural_routing_runtime::DecisionRecord {
                session_id: input.session_id.to_string(),
                context_embedding: vec![],
                action_type: "skill.activate".to_string(),
                action_params: serde_json::json!({
                    "chosen_skill": chosen_name,
                    "activated_count": alternatives_count,
                }),
                alternatives_count,
                chosen_index: 0,
                confidence: chosen_confidence,
                tool_usages: vec![neural_routing_runtime::ToolUsage {
                    tool_name: "skill_activation".to_string(),
                    action: "match".to_string(),
                    params_hash: format!("project:{}", project_id),
                    duration_ms: None,
                    success: true,
                }],
                touched_entities: touched,
                timestamp_ms: 0,
                query_embedding: vec![],
                node_features: vec![],
                protocol_run_id: input.protocol_run_id,
                protocol_state: input.protocol_state.clone(),
            });
        }

        // Async boost energy for activated skills (Hebbian reinforcement)
        if !skill_ids_to_boost.is_empty() {
            let graph = self.graph.clone();
            let boost = self.config.energy_boost;
            tokio::spawn(async move {
                for skill_id in skill_ids_to_boost {
                    if let Err(e) = graph.increment_skill_activation(skill_id).await {
                        debug!(
                            skill_id = %skill_id,
                            error = %e,
                            "Failed to increment skill activation count"
                        );
                    }
                    // Boost energy of member notes
                    match graph.get_skill_members(skill_id).await {
                        Ok((notes, _decisions)) => {
                            let note_ids: Vec<Uuid> = notes.iter().map(|n| n.id).collect();
                            for note_id in &note_ids {
                                let _ = graph.boost_energy(*note_id, boost).await;
                            }
                        }
                        Err(e) => {
                            debug!(
                                skill_id = %skill_id,
                                error = %e,
                                "Failed to load skill members for energy boost"
                            );
                        }
                    }
                }
            });
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "skill_activation"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.skill_activation
    }
}

// ============================================================================
// Intent detection helpers
// ============================================================================

/// Detect intent from matched skills' tags and names.
///
/// Returns one of: planning, code, debug, review, general
fn detect_intent_from_skills(matches: &[(SkillNode, f64)], message: &str) -> &'static str {
    // Check skill tags first (most specific signal)
    for (skill, _) in matches {
        let tags_lower: Vec<String> = skill.tags.iter().map(|t| t.to_lowercase()).collect();
        let name_lower = skill.name.to_lowercase();

        if tags_lower
            .iter()
            .any(|t| t.contains("plan") || t.contains("roadmap"))
            || name_lower.contains("plan")
        {
            return "planning";
        }
        if tags_lower
            .iter()
            .any(|t| t.contains("debug") || t.contains("fix") || t.contains("error"))
            || name_lower.contains("debug")
        {
            return "debug";
        }
        if tags_lower
            .iter()
            .any(|t| t.contains("review") || t.contains("quality"))
            || name_lower.contains("review")
        {
            return "review";
        }
        if tags_lower
            .iter()
            .any(|t| t.contains("code") || t.contains("implement") || t.contains("refactor"))
            || name_lower.contains("code")
            || name_lower.contains("implement")
        {
            return "code";
        }
    }

    // Fallback to message-based detection
    detect_intent_from_message(message)
}

/// Map a [`QueryIntentMode`] from the canonical `IntentDetector` to the
/// enrichment hint string used by `KnowledgeInjectionStage` / `IntentWeightMap`.
///
/// Mapping:
/// - `Debug`   → `"debug"`
/// - `Explore` → `"explore"`
/// - `Impact`  → `"review"`
/// - `Plan`    → `"planning"`
/// - `Default` → `"general"`
fn map_intent_mode(mode: QueryIntentMode) -> &'static str {
    match mode {
        QueryIntentMode::Debug => "debug",
        QueryIntentMode::Explore => "explore",
        QueryIntentMode::Impact => "review",
        QueryIntentMode::Plan => "planning",
        QueryIntentMode::Default => "general",
    }
}

/// Detect intent from the raw message using the canonical [`IntentDetector`].
///
/// Returns one of: planning, code, debug, review, explore, general.
/// Delegates to `IntentDetector::detect()` for unified keyword matching
/// (bilingual FR/EN support), then maps the result to enrichment hint values.
fn detect_intent_from_message(message: &str) -> &'static str {
    map_intent_mode(IntentDetector::detect(message))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::enrichment::EnrichmentContext;

    #[test]
    fn test_extract_file_paths_basic() {
        let paths = SkillActivationStage::extract_file_paths(
            "Look at src/chat/manager.rs and src/skills/activation.rs",
        );
        assert!(paths.contains(&"src/chat/manager.rs".to_string()));
        assert!(paths.contains(&"src/skills/activation.rs".to_string()));
    }

    #[test]
    fn test_extract_file_paths_with_backticks() {
        let paths = SkillActivationStage::extract_file_paths(
            "Check `src/neo4j/client.rs` for the implementation",
        );
        assert!(paths.contains(&"src/neo4j/client.rs".to_string()));
    }

    #[test]
    fn test_extract_file_paths_no_paths() {
        let paths = SkillActivationStage::extract_file_paths("How do I create a new endpoint?");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_extract_file_paths_dotfile() {
        let paths = SkillActivationStage::extract_file_paths("Edit the .env file and Cargo.toml");
        // .env is too short (3 chars), Cargo.toml should match
        assert!(paths.contains(&"Cargo.toml".to_string()));
    }

    #[test]
    fn test_extract_file_paths_with_quotes() {
        let paths = SkillActivationStage::extract_file_paths(
            r#"Open "src/lib.rs" and 'tests/api_tests.rs'"#,
        );
        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(paths.contains(&"tests/api_tests.rs".to_string()));
    }

    #[test]
    fn test_skill_activation_config_default() {
        let config = SkillActivationConfig::default();
        assert!((config.confidence_threshold - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.max_activated_skills, 3);
        assert!((config.energy_boost - 0.1).abs() < f64::EPSILON);
    }

    // ── Intent detection tests ──────────────────────────────────────

    #[test]
    fn test_detect_intent_planning() {
        assert_eq!(
            detect_intent_from_message("crée un plan pour X"),
            "planning"
        );
        assert_eq!(detect_intent_from_message("check the roadmap"), "planning");
        assert_eq!(
            detect_intent_from_message("update the milestone"),
            "planning"
        );
    }

    #[test]
    fn test_detect_intent_debug() {
        assert_eq!(detect_intent_from_message("debug this error"), "debug");
        assert_eq!(detect_intent_from_message("fix this bug"), "debug");
        assert_eq!(detect_intent_from_message("the app crashes"), "debug");
    }

    #[test]
    fn test_detect_intent_review() {
        assert_eq!(detect_intent_from_message("review the code"), "review");
        assert_eq!(detect_intent_from_message("check code coverage"), "review");
        assert_eq!(detect_intent_from_message("audit the API"), "review");
    }

    #[test]
    fn test_detect_intent_code_mapped_to_planning_or_review() {
        // After unification with IntentDetector, "implement" maps to Plan → "planning"
        // and "refactor" maps to Impact → "review".
        assert_eq!(
            detect_intent_from_message("implement the feature"),
            "planning"
        );
        assert_eq!(detect_intent_from_message("refactor this module"), "review");
    }

    #[test]
    fn test_detect_intent_general() {
        assert_eq!(detect_intent_from_message("hello"), "general");
    }

    #[test]
    fn test_detect_intent_explore() {
        assert_eq!(detect_intent_from_message("how does this work?"), "explore");
    }

    #[test]
    fn test_hints_set_and_get() {
        let mut ctx = EnrichmentContext::default();
        assert!(ctx.get_hint("intent").is_none());
        ctx.set_hint("intent", "planning");
        assert_eq!(ctx.get_hint("intent"), Some("planning"));
        // Overwrite
        ctx.set_hint("intent", "debug");
        assert_eq!(ctx.get_hint("intent"), Some("debug"));
    }
}
