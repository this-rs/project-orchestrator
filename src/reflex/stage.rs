//! ReflexStage — enrichment stage that injects reflex suggestions into the prompt.
//!
//! Runs after the biomimicry stage in the enrichment pipeline.
//! Uses the ReflexEngine to generate suggestions based on the current context.

use std::sync::Arc;

use anyhow::Result;
use tracing::debug;

use super::{RefContext, ReflexEngine};
use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentContext, EnrichmentInput, EnrichmentStage,
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
impl EnrichmentStage for ReflexStage {
    async fn execute(&self, input: &EnrichmentInput, ctx: &mut EnrichmentContext) -> Result<()> {
        let project_id = match input.project_id {
            Some(id) => id,
            None => {
                debug!("[reflex] No project_id, skipping reflex stage");
                return Ok(());
            }
        };

        // Build RefContext from EnrichmentInput
        // For now, affected_files and embedding come from the message context.
        // In a full integration, these would be populated from the task/step context.
        let ref_ctx = RefContext {
            affected_files: extract_file_paths(&input.message),
            task_title: None,
            step_description: None,
            embedding: None, // Will be populated when embedding pipeline is wired
            project_id,
        };

        // Skip if no affected files
        if ref_ctx.affected_files.is_empty() {
            debug!("[reflex] No affected files detected, skipping");
            return Ok(());
        }

        let suggestions = self.engine.suggest(&ref_ctx).await;

        if !suggestions.is_empty() {
            let markdown = ReflexEngine::format_markdown(&suggestions);
            ctx.add_section("Reflexes", markdown, self.name());
        }

        Ok(())
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
