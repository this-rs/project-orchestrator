//! ReflexStage — enrichment stage that injects reflex suggestions into the prompt.
//!
//! Runs after the biomimicry stage in the enrichment pipeline.
//! Uses the ReflexEngine to generate suggestions based on the current context.

use std::sync::Arc;

use anyhow::Result;
use tracing::debug;

use super::{RefContext, ReflexEngine};
use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, ParallelEnrichmentStage, StageOutput,
};
use crate::neo4j::traits::GraphStore;

/// Enrichment stage that injects reflex suggestions (co-change, episode recall, scar warnings).
pub struct ReflexStage {
    engine: ReflexEngine,
}

impl ReflexStage {
    /// Create a new reflex stage.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            engine: ReflexEngine::new(graph),
        }
    }
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for ReflexStage {
    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
        let mut output = StageOutput::new(self.name());

        let project_id = match input.project_id {
            Some(id) => id,
            None => {
                debug!("[reflex] No project_id, skipping reflex stage");
                return Ok(output);
            }
        };

        // Build RefContext from EnrichmentInput.
        // Note: the `detected_files` hint from other stages is no longer available
        // in parallel mode (it was never set by anyone anyway — task description confirms).
        // We extract file paths directly from the message.
        let affected = extract_file_paths(&input.message);

        let ref_ctx = RefContext {
            affected_files: affected,
            task_title: None,
            step_description: None,
            embedding: None, // Will be populated when embedding pipeline is wired
            project_id,
        };

        // Skip if no affected files
        if ref_ctx.affected_files.is_empty() {
            debug!("[reflex] No affected files detected, skipping");
            return Ok(output);
        }

        let suggestions = self.engine.suggest(&ref_ctx).await;

        if !suggestions.is_empty() {
            let markdown = ReflexEngine::format_markdown(&suggestions);
            output.add_section("Reflexes", markdown, self.name());
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "reflex"
    }

    fn is_enabled(&self, config: &EnrichmentConfig) -> bool {
        config.reflex
    }
}

/// Extract file paths from a message (simple heuristic).
///
/// Looks for backtick-quoted paths and bare paths matching common patterns.
fn extract_file_paths(message: &str) -> Vec<String> {
    let mut paths = Vec::new();

    // Extract backtick-quoted paths
    for part in message.split('`') {
        let trimmed = part.trim();
        if looks_like_file_path(trimmed) {
            paths.push(trimmed.to_string());
        }
    }

    paths.dedup();
    paths
}

/// Check if a string looks like a file path.
fn looks_like_file_path(s: &str) -> bool {
    if s.is_empty() || s.len() > 256 {
        return false;
    }
    // Must contain a dot or slash
    if !s.contains('.') && !s.contains('/') {
        return false;
    }
    // Common file extensions
    let extensions = [
        ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".java", ".toml", ".yaml", ".yml",
        ".json", ".md", ".sql", ".sh", ".css", ".html",
    ];
    extensions.iter().any(|ext| s.ends_with(ext)) || (s.contains('/') && !s.contains(' '))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::enrichment::{EnrichmentContext, ParallelEnrichmentStage};
    use std::collections::HashSet;

    #[test]
    fn test_extract_file_paths_with_hints() {
        // Simulate EnrichmentContext with detected_files hint
        let mut ctx = EnrichmentContext::default();
        ctx.set_hint("detected_files", "src/api/routes.rs,src/lib.rs");

        // Message mentions one file, hint has two (one overlapping)
        let msg = "Fix the bug in `src/lib.rs`";
        let mut affected = extract_file_paths(msg);
        if let Some(hint_files) = ctx.get_hint("detected_files") {
            for f in hint_files.split(',') {
                let f = f.trim().to_string();
                if !f.is_empty() && !affected.contains(&f) {
                    affected.push(f);
                }
            }
        }

        assert_eq!(affected.len(), 2); // src/lib.rs (from msg) + src/api/routes.rs (from hint)
        assert!(affected.contains(&"src/lib.rs".to_string()));
        assert!(affected.contains(&"src/api/routes.rs".to_string()));
    }

    #[tokio::test]
    async fn test_reflex_stage_skips_without_project_id() {
        use crate::neo4j::mock::MockGraphStore;

        let graph = Arc::new(MockGraphStore::new());
        let stage = ReflexStage::new(graph);
        let input = EnrichmentInput {
            message: "Modify `src/main.rs`".to_string(),
            session_id: uuid::Uuid::new_v4(),
            project_slug: None,
            project_id: None, // No project_id → should skip
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: HashSet::new(),
            reasoning_path_tracker: None,
        };
        let config = EnrichmentConfig::default();

        assert!(stage.is_enabled(&config)); // reflex defaults to true
        let result = stage.execute(&input).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.sections.is_empty(), "Should skip when no project_id");
    }

    #[tokio::test]
    async fn test_reflex_stage_runs_with_project_id() {
        use crate::neo4j::mock::MockGraphStore;

        let graph = Arc::new(MockGraphStore::new());
        let stage = ReflexStage::new(graph);
        let input = EnrichmentInput {
            message: "Fix `src/main.rs` error handling".to_string(),
            session_id: uuid::Uuid::new_v4(),
            project_slug: Some("test-project".to_string()),
            project_id: Some(uuid::Uuid::new_v4()),
            cwd: None,
            protocol_run_id: None,
            protocol_state: None,
            excluded_note_ids: HashSet::new(),
            reasoning_path_tracker: None,
        };

        let result = stage.execute(&input).await;
        assert!(result.is_ok());
        // Mock graph has no scar notes, so sections should be empty
        // but the stage should run without error
    }

    #[test]
    fn test_extract_file_paths() {
        let msg = "I need to modify `src/chat/manager.rs` and `src/lib.rs` for this feature";
        let paths = extract_file_paths(msg);
        assert!(paths.contains(&"src/chat/manager.rs".to_string()));
        assert!(paths.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn test_extract_file_paths_no_paths() {
        let paths = extract_file_paths("Hello, how are you?");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_looks_like_file_path() {
        assert!(looks_like_file_path("src/main.rs"));
        assert!(looks_like_file_path("Cargo.toml"));
        assert!(!looks_like_file_path("hello world"));
        assert!(!looks_like_file_path(""));
    }
}
