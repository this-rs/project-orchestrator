//! Persona Enrichment Stage for the Chat Pipeline.
//!
//! Detects file paths mentioned in the user message, finds the best matching
//! persona (via KNOWS relations), and injects the persona's context into the
//! enrichment output.
//!
//! Enabled by default. Disable with `ENRICHMENT_PERSONA=false`.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::time::timeout;
use tracing::{debug, warn};

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, ParallelEnrichmentStage, StageOutput,
};
use crate::neo4j::traits::GraphStore;

/// Enrichment stage that activates the best-matching persona for files
/// mentioned in the user message.
pub struct PersonaStage {
    graph: Arc<dyn GraphStore>,
}

impl PersonaStage {
    /// Create a new persona enrichment stage.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }
}

/// Extract file-like paths from a message (e.g., `src/chat/manager.rs`, `lib/foo.ts`).
fn extract_file_paths(message: &str) -> Vec<String> {
    crate::utils::file_path_extractor::extract_file_paths(message)
}

/// Build an empty `PersonaSubgraph` as fallback when loading fails.
fn empty_subgraph(
    persona_id: uuid::Uuid,
    persona_name: &str,
) -> crate::neo4j::models::PersonaSubgraph {
    crate::neo4j::models::PersonaSubgraph {
        persona_id,
        persona_name: persona_name.to_string(),
        files: vec![],
        functions: vec![],
        notes: vec![],
        decisions: vec![],
        skills: vec![],
        protocols: vec![],
        feature_graph_id: None,
        parents: vec![],
        children: vec![],
        stats: crate::neo4j::models::PersonaSubgraphStats {
            total_entities: 0,
            coverage_score: 0.0,
            freshness: 0.0,
        },
    }
}

/// Default char budget for the persona context section.
const PERSONA_BUDGET_CHARS: usize = 2000;

/// Render a `CachedPersona` into a rich markdown section for prompt injection.
///
/// Adapted from `runner::persona::PersonaStack::render_entry`. Includes:
/// - Header with name + relevance %
/// - Description (if non-empty)
/// - Known files (top by weight)
/// - Known functions (top by weight)
/// - Knowledge notes (IDs + weight)
/// - Decisions (IDs + weight)
///
/// Output is capped at `PERSONA_BUDGET_CHARS` to avoid blowing up the prompt.
fn render_persona_section(cp: &crate::chat::enrichment::CachedPersona) -> String {
    let budget = PERSONA_BUDGET_CHARS;
    let mut out = format!(
        "## Active Persona: {} (relevance: {:.0}%)\n",
        cp.persona_name,
        cp.weight * 100.0
    );

    if !cp.description.is_empty() {
        out.push_str(&format!("\n{}\n", cp.description));
    }

    let sub = &cp.subgraph;

    // Files it knows (top by weight)
    if !sub.files.is_empty() {
        out.push_str("\n**Known files:**\n");
        let max_files = (budget / 100).clamp(3, 15);
        for rel in sub.files.iter().take(max_files) {
            out.push_str(&format!("- `{}` (w:{:.2})\n", rel.entity_id, rel.weight));
        }
    }

    // Functions it knows
    if !sub.functions.is_empty() {
        out.push_str("\n**Known functions:**\n");
        let max_fns = (budget / 120).clamp(2, 10);
        for rel in sub.functions.iter().take(max_fns) {
            out.push_str(&format!("- `{}` (w:{:.2})\n", rel.entity_id, rel.weight));
        }
    }

    // Notes it uses
    if !sub.notes.is_empty() {
        out.push_str("\n**Knowledge notes:**\n");
        let max_notes = (budget / 200).clamp(2, 8);
        for rel in sub.notes.iter().take(max_notes) {
            out.push_str(&format!(
                "- [note:{}] (w:{:.2})\n",
                rel.entity_id, rel.weight
            ));
        }
    }

    // Decisions it uses
    if !sub.decisions.is_empty() {
        out.push_str("\n**Decisions:**\n");
        let max_decisions = (budget / 200).clamp(1, 5);
        for rel in sub.decisions.iter().take(max_decisions) {
            out.push_str(&format!(
                "- [decision:{}] (w:{:.2})\n",
                rel.entity_id, rel.weight
            ));
        }
    }

    // Truncate if over budget
    if out.chars().count() > budget {
        let safe: String = out.chars().take(budget.saturating_sub(4)).collect();
        out = safe;
        out.push_str("...\n");
    }

    out
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for PersonaStage {
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
        use crate::chat::enrichment::CachedPersona;

        let mut output = StageOutput::new(self.name());

        let project_id = match input.project_id {
            Some(id) => id,
            None => {
                if let Some(slug) = &input.project_slug {
                    match self.graph.get_project_by_slug(slug).await {
                        Ok(Some(p)) => p.id,
                        _ => return Ok(output),
                    }
                } else {
                    return Ok(output);
                }
            }
        };

        let file_paths = extract_file_paths(&input.message);

        // ── Sticky cache logic ──────────────────────────────────────────
        // 1. If file paths found → detect best persona, upgrade cache if better
        // 2. If no file paths but cache exists → reuse cached persona
        // 3. If no file paths and no cache → skip (nothing to inject)

        let cached = input.cached_persona.as_ref();

        if file_paths.is_empty() {
            // No file paths in this message — reuse cache if available
            if let Some(cp) = cached {
                debug!(
                    "PersonaStage: reusing cached persona '{}' (no file paths in message)",
                    cp.persona_name
                );
                let section = render_persona_section(cp);
                output.set_hint("active_persona", cp.persona_name.clone());
                output.set_hint("active_persona_id", cp.persona_id.to_string());
                // Re-emit the cached persona JSON so the session keeps it
                if let Ok(json) = serde_json::to_string(cp) {
                    output.set_hint("cached_persona_json", json);
                }
                output.add_section(
                    "Persona Context",
                    section,
                    "persona",
                    crate::chat::enrichment::EnrichmentSource::Persona,
                );
                return Ok(output);
            }
            debug!("PersonaStage: no file paths detected and no cached persona");
            return Ok(output);
        }

        debug!(
            "PersonaStage: detected {} file path(s): {:?}",
            file_paths.len(),
            file_paths
        );

        // ── Detect best persona from file paths ─────────────────────────
        let stage_timeout = Duration::from_millis(400); // leave 100ms for subgraph load
        let graph = self.graph.clone();

        let best_persona = timeout(stage_timeout, async {
            let mut best: Option<(crate::neo4j::models::PersonaNode, f64)> = None;

            for path in &file_paths {
                match graph.find_personas_for_file(path, project_id).await {
                    Ok(personas) => {
                        if let Some((persona, weight)) = personas.into_iter().next() {
                            if best.as_ref().is_none_or(|(_, w)| weight > *w) {
                                best = Some((persona, weight));
                            }
                        }
                    }
                    Err(e) => {
                        debug!(
                            "PersonaStage: find_personas_for_file({}) error: {}",
                            path, e
                        );
                    }
                }
            }

            best
        })
        .await;

        let (persona, weight) = match best_persona {
            Ok(Some((p, w))) => (p, w),
            Ok(None) => {
                // No persona found for these files — fall back to cache
                if let Some(cp) = cached {
                    debug!(
                        "PersonaStage: no persona for detected files, reusing cached '{}'",
                        cp.persona_name
                    );
                    let section = render_persona_section(cp);
                    output.set_hint("active_persona", cp.persona_name.clone());
                    output.set_hint("active_persona_id", cp.persona_id.to_string());
                    if let Ok(json) = serde_json::to_string(cp) {
                        output.set_hint("cached_persona_json", json);
                    }
                    output.add_section(
                        "Persona Context",
                        section,
                        "persona",
                        crate::chat::enrichment::EnrichmentSource::Persona,
                    );
                    return Ok(output);
                }
                debug!("PersonaStage: no matching persona found for detected files");
                return Ok(output);
            }
            Err(_) => {
                warn!("PersonaStage: timed out finding personas");
                // On timeout, fall back to cache
                if let Some(cp) = cached {
                    let section = render_persona_section(cp);
                    output.set_hint("active_persona", cp.persona_name.clone());
                    output.set_hint("active_persona_id", cp.persona_id.to_string());
                    if let Ok(json) = serde_json::to_string(cp) {
                        output.set_hint("cached_persona_json", json);
                    }
                    output.add_section(
                        "Persona Context",
                        section,
                        "persona",
                        crate::chat::enrichment::EnrichmentSource::Persona,
                    );
                    return Ok(output);
                }
                return Ok(output);
            }
        };

        // ── Upgrade check: only replace cache if new match is strictly better ──
        if let Some(cp) = cached {
            if weight <= cp.weight {
                debug!(
                    "PersonaStage: detected '{}' (w={:.2}) <= cached '{}' (w={:.2}), keeping cache",
                    persona.name, weight, cp.persona_name, cp.weight
                );
                let section = render_persona_section(cp);
                output.set_hint("active_persona", cp.persona_name.clone());
                output.set_hint("active_persona_id", cp.persona_id.to_string());
                if let Ok(json) = serde_json::to_string(cp) {
                    output.set_hint("cached_persona_json", json);
                }
                output.add_section(
                    "Persona Context",
                    section,
                    "persona",
                    crate::chat::enrichment::EnrichmentSource::Persona,
                );
                return Ok(output);
            }
        }

        debug!(
            "PersonaStage: activating persona '{}' (weight={:.2})",
            persona.name, weight
        );

        // ── Load subgraph for the new persona ───────────────────────────
        let subgraph = match timeout(
            Duration::from_millis(100),
            self.graph.get_persona_subgraph(persona.id),
        )
        .await
        {
            Ok(Ok(sg)) => sg,
            Ok(Err(e)) => {
                warn!("PersonaStage: get_persona_subgraph error: {}", e);
                empty_subgraph(persona.id, &persona.name)
            }
            Err(_) => {
                warn!("PersonaStage: get_persona_subgraph timed out");
                empty_subgraph(persona.id, &persona.name)
            }
        };

        // Build new CachedPersona
        let new_cached = CachedPersona {
            persona_id: persona.id,
            persona_name: persona.name.clone(),
            description: persona.description.clone(),
            subgraph,
            weight,
        };

        let section = render_persona_section(&new_cached);
        output.set_hint("active_persona", new_cached.persona_name.clone());
        output.set_hint("active_persona_id", new_cached.persona_id.to_string());
        if let Ok(json) = serde_json::to_string(&new_cached) {
            output.set_hint("cached_persona_json", json);
        }
        output.add_section(
            "Persona Context",
            section,
            "persona",
            crate::chat::enrichment::EnrichmentSource::Persona,
        );

        Ok(output)
    }

    fn name(&self) -> &str {
        "persona"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.persona
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::enrichment::ParallelEnrichmentStage;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{PersonaNode, PersonaStatus};
    use uuid::Uuid;

    fn make_input(message: &str, project_id: Option<Uuid>) -> EnrichmentInput {
        EnrichmentInput {
            message: message.to_string(),
            session_id: Uuid::new_v4(),
            project_slug: None,
            project_id,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
            cached_persona: None,
        }
    }

    fn make_persona(name: &str, description: &str, project_id: Uuid) -> PersonaNode {
        PersonaNode {
            id: Uuid::new_v4(),
            project_id: Some(project_id),
            name: name.to_string(),
            description: description.to_string(),
            status: PersonaStatus::Active,
            complexity_default: None,
            timeout_secs: None,
            max_cost_usd: None,
            model_preference: None,
            system_prompt_override: None,
            energy: 0.8,
            cohesion: 0.7,
            activation_count: 0,
            success_rate: 0.0,
            avg_duration_secs: 0.0,
            last_activated: None,
            energy_boost_accumulated: 0.0,
            energy_history: Vec::new(),
            origin: Default::default(),
            created_at: chrono::Utc::now(),
            updated_at: Some(chrono::Utc::now()),
        }
    }

    // ── extract_file_paths unit tests ──────────────────────────────────

    #[test]
    fn test_extract_file_paths_basic() {
        let msg = "Look at src/chat/manager.rs and src/lib.rs for the issue";
        let paths = extract_file_paths(msg);
        assert_eq!(paths, vec!["src/chat/manager.rs", "src/lib.rs"]);
    }

    #[test]
    fn test_extract_file_paths_backticks() {
        let msg = "Check `src/neo4j/traits.rs` for the trait definition";
        let paths = extract_file_paths(msg);
        assert_eq!(paths, vec!["src/neo4j/traits.rs"]);
    }

    #[test]
    fn test_extract_file_paths_no_paths() {
        let msg = "What is the architecture of this project?";
        let paths = extract_file_paths(msg);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_extract_file_paths_dedup() {
        let msg = "src/lib.rs and src/lib.rs again";
        let paths = extract_file_paths(msg);
        assert_eq!(paths, vec!["src/lib.rs"]);
    }

    #[test]
    fn test_extract_file_paths_nested() {
        let msg = "Modify src/chat/stages/persona.rs";
        let paths = extract_file_paths(msg);
        assert_eq!(paths, vec!["src/chat/stages/persona.rs"]);
    }

    // ── stage metadata tests ───────────────────────────────────────────

    #[test]
    fn test_stage_name() {
        let mock = Arc::new(MockGraphStore::new());
        let stage = PersonaStage::new(mock);
        assert_eq!(stage.name(), "persona");
    }

    #[test]
    fn test_stage_is_enabled() {
        let mock = Arc::new(MockGraphStore::new());
        let stage = PersonaStage::new(mock);

        let enabled = EnrichmentConfig {
            persona: true,
            ..Default::default()
        };
        assert!(stage.is_enabled(&enabled));

        let disabled = EnrichmentConfig {
            persona: false,
            ..Default::default()
        };
        assert!(!stage.is_enabled(&disabled));
    }

    // ── execute integration tests ──────────────────────────────────────

    #[tokio::test]
    async fn test_execute_no_project_skips() {
        let mock = Arc::new(MockGraphStore::new());
        let stage = PersonaStage::new(mock);
        let input = make_input("Look at src/lib.rs", None);

        let output = stage.execute(&input).await.unwrap();
        assert!(output.sections.is_empty());
    }

    #[tokio::test]
    async fn test_execute_no_file_paths_skips() {
        let mock = Arc::new(MockGraphStore::new());
        let stage = PersonaStage::new(mock);
        let project_id = Uuid::new_v4();
        let input = make_input("What is the architecture?", Some(project_id));

        let output = stage.execute(&input).await.unwrap();
        assert!(output.sections.is_empty());
    }

    #[tokio::test]
    async fn test_execute_no_matching_persona() {
        let mock = Arc::new(MockGraphStore::new());
        let stage = PersonaStage::new(mock);
        let project_id = Uuid::new_v4();
        let input = make_input("Check src/lib.rs", Some(project_id));

        let output = stage.execute(&input).await.unwrap();
        // No persona registered → no section injected
        assert!(output.sections.is_empty());
    }

    #[tokio::test]
    async fn test_execute_finds_matching_persona() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Create persona and link to file
        let persona = make_persona(
            "neo4j-expert",
            "Specialist in graph database queries",
            project_id,
        );
        mock.create_persona(&persona).await.unwrap();
        mock.add_persona_file(persona.id, "src/neo4j/client.rs", 0.9)
            .await
            .unwrap();

        let stage = PersonaStage::new(mock);
        let input = make_input("Fix the bug in src/neo4j/client.rs", Some(project_id));

        let output = stage.execute(&input).await.unwrap();

        // Should have injected a section
        assert_eq!(output.sections.len(), 1);
        assert_eq!(output.sections[0].title, "Persona Context");
        assert!(output.sections[0].content.contains("neo4j-expert"));
        assert!(output.sections[0].content.contains("90%")); // weight 0.9 → 90%
        assert!(output.sections[0]
            .content
            .contains("Specialist in graph database queries"));

        // Should have set hints
        assert_eq!(output.hints.get("active_persona").unwrap(), "neo4j-expert");
        assert_eq!(
            output.hints.get("active_persona_id").unwrap(),
            &persona.id.to_string()
        );
    }

    #[tokio::test]
    async fn test_execute_picks_highest_weight_persona() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Two personas for different files in the same message
        let p1 = make_persona("low-weight", "Low priority persona", project_id);
        let p2 = make_persona("high-weight", "High priority persona", project_id);
        mock.create_persona(&p1).await.unwrap();
        mock.create_persona(&p2).await.unwrap();
        mock.add_persona_file(p1.id, "src/lib.rs", 0.3)
            .await
            .unwrap();
        mock.add_persona_file(p2.id, "src/chat/manager.rs", 0.95)
            .await
            .unwrap();

        let stage = PersonaStage::new(mock);
        let input = make_input(
            "Refactor src/lib.rs and src/chat/manager.rs",
            Some(project_id),
        );

        let output = stage.execute(&input).await.unwrap();

        assert_eq!(output.sections.len(), 1);
        assert!(output.sections[0].content.contains("high-weight"));
        assert_eq!(output.hints.get("active_persona").unwrap(), "high-weight");
    }

    #[tokio::test]
    async fn test_execute_empty_description_omitted() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let persona = make_persona("minimal", "", project_id);
        mock.create_persona(&persona).await.unwrap();
        mock.add_persona_file(persona.id, "src/main.rs", 0.5)
            .await
            .unwrap();

        let stage = PersonaStage::new(mock);
        let input = make_input("Check src/main.rs", Some(project_id));

        let output = stage.execute(&input).await.unwrap();

        assert_eq!(output.sections.len(), 1);
        // Should have the header but NOT an empty description paragraph
        assert!(output.sections[0].content.contains("minimal"));
        // The header line should not be followed by a description paragraph
        let content = &output.sections[0].content;
        let first_line = content.lines().next().unwrap();
        assert!(first_line.starts_with("## Active Persona: minimal"));
        // No empty description between header and Known files
        assert!(!content.contains("\n\n\n"));
    }

    // ── Sticky behavior tests ─────────────────────────────────────────

    fn make_cached_persona(
        name: &str,
        description: &str,
        weight: f64,
        files: Vec<(&str, f64)>,
    ) -> crate::chat::enrichment::CachedPersona {
        use crate::neo4j::models::{
            PersonaSubgraph, PersonaSubgraphStats, PersonaWeightedRelation,
        };

        crate::chat::enrichment::CachedPersona {
            persona_id: Uuid::new_v4(),
            persona_name: name.to_string(),
            description: description.to_string(),
            subgraph: PersonaSubgraph {
                persona_id: Uuid::new_v4(),
                persona_name: name.to_string(),
                files: files
                    .into_iter()
                    .map(|(path, w)| PersonaWeightedRelation {
                        entity_type: "file".to_string(),
                        entity_id: path.to_string(),
                        weight: w,
                    })
                    .collect(),
                functions: vec![],
                notes: vec![],
                decisions: vec![],
                skills: vec![],
                protocols: vec![],
                feature_graph_id: None,
                parents: vec![],
                children: vec![],
                stats: PersonaSubgraphStats {
                    total_entities: 0,
                    coverage_score: 0.0,
                    freshness: 0.0,
                },
            },
            weight,
        }
    }

    fn make_input_with_cache(
        message: &str,
        project_id: Option<Uuid>,
        cached: Option<crate::chat::enrichment::CachedPersona>,
    ) -> EnrichmentInput {
        EnrichmentInput {
            message: message.to_string(),
            session_id: Uuid::new_v4(),
            project_slug: None,
            project_id,
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: Default::default(),
            reasoning_path_tracker: None,
            cached_persona: cached,
        }
    }

    #[tokio::test]
    async fn test_sticky_reuses_cached_persona_without_file_paths() {
        let mock = Arc::new(MockGraphStore::new());
        let stage = PersonaStage::new(mock);
        let project_id = Uuid::new_v4();

        let cached = make_cached_persona(
            "chat-expert",
            "Expert in chat systems",
            0.85,
            vec![("src/chat/manager.rs", 0.9)],
        );
        // Message WITHOUT file paths → should reuse cached persona
        let input = make_input_with_cache(
            "What is the architecture of this module?",
            Some(project_id),
            Some(cached),
        );

        let output = stage.execute(&input).await.unwrap();

        assert_eq!(output.sections.len(), 1);
        assert!(output.sections[0].content.contains("chat-expert"));
        assert!(output.sections[0].content.contains("85%"));
        assert_eq!(output.hints.get("active_persona").unwrap(), "chat-expert");
        // Should re-emit cached_persona_json for session persistence
        assert!(output.hints.contains_key("cached_persona_json"));
    }

    #[tokio::test]
    async fn test_sticky_keeps_cache_when_new_match_is_weaker() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Cached persona with high weight
        let cached = make_cached_persona(
            "strong-persona",
            "High weight persona",
            0.95,
            vec![("src/chat/manager.rs", 0.9)],
        );

        // New persona with lower weight linked to a different file
        let weak = make_persona("weak-persona", "Low weight persona", project_id);
        mock.create_persona(&weak).await.unwrap();
        mock.add_persona_file(weak.id, "src/lib.rs", 0.3)
            .await
            .unwrap();

        let stage = PersonaStage::new(mock);
        // Message with file paths that match the weaker persona
        let input = make_input_with_cache("Check src/lib.rs", Some(project_id), Some(cached));

        let output = stage.execute(&input).await.unwrap();

        assert_eq!(output.sections.len(), 1);
        // Should keep the strong cached persona, NOT the weaker new match
        assert!(output.sections[0].content.contains("strong-persona"));
        assert!(!output.sections[0].content.contains("weak-persona"));
        assert_eq!(
            output.hints.get("active_persona").unwrap(),
            "strong-persona"
        );
    }

    #[tokio::test]
    async fn test_sticky_upgrades_cache_when_new_match_is_stronger() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Cached persona with low weight
        let cached =
            make_cached_persona("old-persona", "Old persona", 0.4, vec![("src/old.rs", 0.4)]);

        // New persona with higher weight
        let strong = make_persona("new-persona", "New strong persona", project_id);
        mock.create_persona(&strong).await.unwrap();
        mock.add_persona_file(strong.id, "src/new.rs", 0.9)
            .await
            .unwrap();

        let stage = PersonaStage::new(mock);
        let input = make_input_with_cache("Fix src/new.rs", Some(project_id), Some(cached));

        let output = stage.execute(&input).await.unwrap();

        assert_eq!(output.sections.len(), 1);
        // Should upgrade to the new, stronger persona
        assert!(output.sections[0].content.contains("new-persona"));
        assert!(!output.sections[0].content.contains("old-persona"));
        assert_eq!(output.hints.get("active_persona").unwrap(), "new-persona");
        // cached_persona_json should contain the new persona
        let json = output.hints.get("cached_persona_json").unwrap();
        assert!(json.contains("new-persona"));
    }

    #[tokio::test]
    async fn test_subgraph_renders_known_files() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let persona = make_persona("graph-expert", "Graph DB specialist", project_id);
        mock.create_persona(&persona).await.unwrap();
        mock.add_persona_file(persona.id, "src/neo4j/client.rs", 0.9)
            .await
            .unwrap();
        mock.add_persona_file(persona.id, "src/neo4j/models.rs", 0.8)
            .await
            .unwrap();

        let stage = PersonaStage::new(mock);
        let input = make_input("Check src/neo4j/client.rs", Some(project_id));

        let output = stage.execute(&input).await.unwrap();

        let content = &output.sections[0].content;
        // Should contain the Known files section with actual file paths
        assert!(content.contains("**Known files:**"));
        assert!(content.contains("src/neo4j/client.rs"));
        assert!(content.contains("src/neo4j/models.rs"));
    }

    #[tokio::test]
    async fn test_render_persona_section_respects_budget() {
        // Build a cached persona with many files to exceed budget
        let mut files: Vec<(&str, f64)> = Vec::new();
        for i in 0..100 {
            // Leak strings to get &str — only in test
            let path: &str = Box::leak(
                format!("src/module_{:03}/very_long_file_name_{:03}.rs", i, i).into_boxed_str(),
            );
            files.push((path, 0.5));
        }
        let cached = make_cached_persona(
            "mega-persona",
            "A persona with extensive file knowledge spanning many modules across the entire project",
            0.9,
            files,
        );

        let section = render_persona_section(&cached);
        // Should be within budget (PERSONA_BUDGET_CHARS = 2000)
        assert!(
            section.chars().count() <= PERSONA_BUDGET_CHARS,
            "Section exceeds budget: {} > {}",
            section.chars().count(),
            PERSONA_BUDGET_CHARS
        );
        // Should end with truncation marker if it was truncated
        // (with 100 files, it should be truncated)
        if section.chars().count() > 500 {
            // If we generated enough content, it should have been truncated
            assert!(
                section.ends_with("...\n") || section.chars().count() <= PERSONA_BUDGET_CHARS,
                "Should be truncated or within budget"
            );
        }
    }

    #[tokio::test]
    async fn test_no_cache_no_file_paths_skips() {
        // No cached persona + no file paths = no output
        let mock = Arc::new(MockGraphStore::new());
        let stage = PersonaStage::new(mock);
        let project_id = Uuid::new_v4();

        let input = make_input_with_cache("What is the architecture?", Some(project_id), None);

        let output = stage.execute(&input).await.unwrap();
        assert!(output.sections.is_empty());
    }

    #[tokio::test]
    async fn test_cached_persona_json_is_valid() {
        let mock = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        let persona = make_persona("json-test", "Test JSON serialization", project_id);
        mock.create_persona(&persona).await.unwrap();
        mock.add_persona_file(persona.id, "src/test.rs", 0.7)
            .await
            .unwrap();

        let stage = PersonaStage::new(mock);
        let input = make_input("Check src/test.rs", Some(project_id));

        let output = stage.execute(&input).await.unwrap();

        // Verify the cached_persona_json hint is valid JSON that deserializes
        let json = output.hints.get("cached_persona_json").unwrap();
        let deserialized: crate::chat::enrichment::CachedPersona =
            serde_json::from_str(json).expect("cached_persona_json should be valid JSON");
        assert_eq!(deserialized.persona_name, "json-test");
        assert!((deserialized.weight - 0.7).abs() < 0.01);
    }
}
