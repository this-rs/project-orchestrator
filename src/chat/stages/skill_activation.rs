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
    EnrichmentConfig, EnrichmentContext, EnrichmentInput, EnrichmentStage,
};
use crate::neo4j::traits::GraphStore;
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
        let mut paths = Vec::new();
        // Match file-like patterns: word chars, slashes, dots, hyphens
        // Must contain at least one slash or dot+extension to be a file path
        for word in message.split_whitespace() {
            // Strip common wrapping chars (quotes, backticks, parens)
            let cleaned = word.trim_matches(|c: char| {
                c == '`' || c == '\'' || c == '"' || c == '(' || c == ')' || c == ','
            });
            if cleaned.is_empty() {
                continue;
            }
            // Must contain a slash (path separator) or look like a file extension
            let has_slash = cleaned.contains('/');
            let has_extension = cleaned.contains('.') && {
                let parts: Vec<&str> = cleaned.rsplit('.').collect();
                parts
                    .first()
                    .map(|ext| ext.len() <= 10 && ext.chars().all(|c| c.is_alphanumeric()))
                    .unwrap_or(false)
            };
            if has_slash || (has_extension && cleaned.len() > 3) {
                // Additional check: must have path-like characters only
                if cleaned
                    .chars()
                    .all(|c| c.is_alphanumeric() || "/_.-+@".contains(c))
                {
                    paths.push(cleaned.to_string());
                }
            }
        }
        paths
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
impl EnrichmentStage for SkillActivationStage {
    async fn execute(&self, input: &EnrichmentInput, ctx: &mut EnrichmentContext) -> Result<()> {
        // Need a project scope for skill matching
        let project_id = if let Some(id) = input.project_id {
            id
        } else if let Some(ref slug) = input.project_slug {
            match self.resolve_project_id(slug).await? {
                Some(id) => id,
                None => return Ok(()), // No project found, skip
            }
        } else {
            return Ok(()); // No project scope, skip
        };

        // Match skills against the message
        let matches = self.match_skills(&input.message, project_id).await?;

        if matches.is_empty() {
            return Ok(());
        }

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
            ctx.add_section("Activated Skills", content_parts.join("\n"), self.name());
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

        Ok(())
    }

    fn name(&self) -> &str {
        "skill_activation"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.skill_activation
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
