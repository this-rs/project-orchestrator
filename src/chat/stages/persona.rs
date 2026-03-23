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

#[async_trait::async_trait]
impl ParallelEnrichmentStage for PersonaStage {
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
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
        if file_paths.is_empty() {
            debug!("PersonaStage: no file paths detected in message");
            return Ok(output);
        }

        debug!(
            "PersonaStage: detected {} file path(s): {:?}",
            file_paths.len(),
            file_paths
        );

        // Find the best persona across all mentioned files
        let stage_timeout = Duration::from_millis(500);
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
                debug!("PersonaStage: no matching persona found for detected files");
                return Ok(output);
            }
            Err(_) => {
                warn!("PersonaStage: timed out finding personas");
                return Ok(output);
            }
        };

        debug!(
            "PersonaStage: activated persona '{}' (weight={:.2})",
            persona.name, weight
        );

        // Build the persona context section
        let mut section = format!(
            "## Active Persona: {} (relevance: {:.0}%)\n",
            persona.name,
            weight * 100.0
        );

        if !persona.description.is_empty() {
            section.push_str(&format!("\n{}\n", persona.description));
        }

        // Set hint so downstream stages know which persona is active
        output.set_hint("active_persona", persona.name.clone());
        output.set_hint("active_persona_id", persona.id.to_string());

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
        // Should have the header but NOT an empty description line
        assert!(output.sections[0].content.contains("minimal"));
        // Content should only have the header line (no extra newlines from empty description)
        let lines: Vec<&str> = output.sections[0].content.trim().lines().collect();
        assert_eq!(lines.len(), 1); // Just the "## Active Persona: ..." line
    }
}
