//! FileContextStage — Multi-dimensional file intelligence injection
//!
//! Injects ContextCard-based intelligence for recently mentioned files
//! into the enrichment context. For each file (max 3), produces a compact
//! profile with community, risk, co-changers, bridge status, and notes.
//!
//! # Budget
//! - Max 3 files per invocation
//! - Max 250 tokens per file (~800 tokens total)
//! - Cache: 30s per file (avoids repeated Neo4j fetches)
//! - Total stage budget: 100ms

use crate::chat::enrichment::{
    EnrichmentConfig, EnrichmentInput, ParallelEnrichmentStage, StageOutput,
};
use crate::graph::models::ContextCard;
use crate::neo4j::traits::GraphStore;
use crate::neurons::AutoReinforcementConfig;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::debug;

/// Cache TTL for file profiles (seconds).
const CACHE_TTL_SECS: u64 = 30;

/// Maximum number of files to profile per stage execution.
const MAX_FILES: usize = 3;

/// FileContextStage injects ContextCard-based intelligence for recently mentioned files.
pub struct FileContextStage {
    graph_store: Arc<dyn GraphStore>,
    /// Per-file cache: path → (timestamp, formatted profile text)
    cache: Mutex<HashMap<String, (Instant, String)>>,
}

impl FileContextStage {
    pub fn new(graph_store: Arc<dyn GraphStore>) -> Self {
        Self {
            graph_store,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Extract file paths from the user message (look for paths like src/foo/bar.rs).
    fn extract_file_paths(message: &str) -> Vec<String> {
        let mut paths = Vec::new();
        for word in message.split_whitespace() {
            let cleaned =
                word.trim_matches(|c: char| c == '`' || c == '\'' || c == '"' || c == ',');
            // Match file-like patterns (contains / and a dot extension)
            if cleaned.contains('/')
                && cleaned.contains('.')
                && !cleaned.starts_with("http")
                && !cleaned.starts_with("//")
            {
                paths.push(cleaned.to_string());
            }
        }
        paths.truncate(MAX_FILES);
        paths
    }

    /// Format a ContextCard into a compact, informative text block.
    fn format_card(card: &ContextCard) -> String {
        let mut lines = Vec::new();

        let short_path = card.path.rsplit('/').next().unwrap_or(&card.path);
        lines.push(format!("📄 {}", card.path));

        // Community
        if !card.cc_community_label.is_empty() {
            lines.push(format!(
                "  🏛️ Community: \"{}\" (id: {})",
                card.cc_community_label, card.cc_community_id
            ));
        }

        // Risk assessment
        let risk_score = card.cc_pagerank * 0.4
            + card.cc_betweenness * 0.3
            + (card.cc_imports_in as f64 / (card.cc_imports_in as f64 + 1.0).max(1.0)) * 0.3;
        let risk_level = if risk_score > 0.6 {
            "HIGH"
        } else if risk_score > 0.3 {
            "MEDIUM"
        } else {
            "LOW"
        };
        if risk_score > 0.2 {
            lines.push(format!(
                "  📊 Risk: {} (pagerank: {:.2}, betweenness: {:.2}, imports_in: {})",
                risk_level, card.cc_pagerank, card.cc_betweenness, card.cc_imports_in
            ));
        }

        // Bridge
        if card.cc_betweenness > 0.5 {
            lines.push(format!(
                "  🌉 Bridge: YES — bottleneck between clusters (betweenness: {:.2}). Use analyze_impact before modifying.",
                card.cc_betweenness
            ));
        }

        // Co-changers
        if !card.cc_co_changers_top5.is_empty() {
            let changers: Vec<&str> = card
                .cc_co_changers_top5
                .iter()
                .take(3)
                .map(|s| s.rsplit('/').next().unwrap_or(s))
                .collect();
            lines.push(format!("  🔗 Co-changers: {}", changers.join(", ")));
        }

        // Connectivity
        if card.cc_imports_in > 0 || card.cc_imports_out > 0 {
            lines.push(format!(
                "  📦 Imports: {} in / {} out | Calls: {} in / {} out",
                card.cc_imports_in, card.cc_imports_out, card.cc_calls_in, card.cc_calls_out
            ));
        }

        let _ = short_path; // used above via card.path
        lines.join("\n")
    }

    /// Check if a file path is cached and still valid.
    fn get_cached(&self, path: &str) -> Option<String> {
        let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((ts, text)) = cache.get(path) {
            if ts.elapsed().as_secs() < CACHE_TTL_SECS {
                return Some(text.clone());
            }
        }
        None
    }

    /// Store a formatted profile in cache.
    fn set_cache(&self, path: String, text: String) {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        // Evict stale entries
        cache.retain(|_, (ts, _)| ts.elapsed().as_secs() < CACHE_TTL_SECS * 2);
        cache.insert(path, (Instant::now(), text));
    }

    /// Fire-and-forget: boost energy for notes linked to a profiled file.
    /// This closes the neural feedback loop — files that are profiled (surfaced
    /// in context) see their notes gain energy, making them more likely to
    /// survive synapse decay and emerge again in future sessions.
    fn spawn_energy_boost(&self, file_path: &str) {
        let graph = self.graph_store.clone();
        let path = file_path.to_string();
        let boost = AutoReinforcementConfig::default().hook_energy_boost;
        tokio::spawn(async move {
            match graph
                .get_notes_for_entity(&crate::notes::EntityType::File, &path)
                .await
            {
                Ok(notes) if !notes.is_empty() => {
                    for note in &notes {
                        let _ = graph.boost_energy(note.id, boost).await;
                    }
                    // If 2+ notes co-surfaced, reinforce their synapses
                    if notes.len() >= 2 {
                        let note_ids: Vec<uuid::Uuid> = notes.iter().map(|n| n.id).collect();
                        let _ = graph.reinforce_synapses(&note_ids, 0.03).await;
                    }
                    debug!(
                        path = %path,
                        notes = notes.len(),
                        "FileContextStage: energy boost for profiled file notes"
                    );
                }
                _ => {} // No notes or error → skip silently
            }
        });
    }
}

#[async_trait::async_trait]
impl ParallelEnrichmentStage for FileContextStage {
    fn name(&self) -> &str {
        "file_context"
    }

    fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
        // Always enabled — ContextCards are pre-computed, fetches are cheap
        true
    }

    async fn execute(&self, input: &EnrichmentInput) -> Result<StageOutput> {
        let mut output = StageOutput::new(self.name());

        let project_id = match input.project_id {
            Some(id) => id,
            None => return Ok(output), // No project context → skip
        };

        let file_paths = Self::extract_file_paths(&input.message);
        if file_paths.is_empty() {
            return Ok(output);
        }

        let project_id_str = project_id.to_string();
        let mut profiles = Vec::new();

        for path in &file_paths {
            // Check cache first
            if let Some(cached) = self.get_cached(path) {
                profiles.push(cached);
                continue;
            }

            // Fetch ContextCard (single Neo4j query, pre-computed data)
            match self
                .graph_store
                .get_context_card(path, &project_id_str)
                .await
            {
                Ok(Some(card)) => {
                    let formatted = Self::format_card(&card);
                    self.set_cache(path.clone(), formatted.clone());
                    profiles.push(formatted);
                    // Neural feedback: boost energy for notes linked to this file
                    self.spawn_energy_boost(path);
                }
                Ok(None) => {
                    debug!(path = %path, "No ContextCard found for file");
                }
                Err(e) => {
                    debug!(path = %path, error = %e, "Failed to fetch ContextCard");
                }
            }
        }

        if !profiles.is_empty() {
            let content = profiles.join("\n\n");
            output.add_section(
                "File Intelligence".to_string(),
                content,
                "file_context".to_string(),
            );
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_file_paths() {
        let msg = "Can you check src/chat/manager.rs and also src/neo4j/client.rs please?";
        let paths = FileContextStage::extract_file_paths(msg);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&"src/chat/manager.rs".to_string()));
        assert!(paths.contains(&"src/neo4j/client.rs".to_string()));
    }

    #[test]
    fn test_extract_file_paths_with_backticks() {
        let msg = "Look at `src/api/handlers.rs`";
        let paths = FileContextStage::extract_file_paths(msg);
        assert_eq!(paths, vec!["src/api/handlers.rs"]);
    }

    #[test]
    fn test_extract_file_paths_no_match() {
        let msg = "Please fix the bug in the login system";
        let paths = FileContextStage::extract_file_paths(msg);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_extract_file_paths_ignores_urls() {
        let msg = "See http://localhost:8080/api/foo.json";
        let paths = FileContextStage::extract_file_paths(msg);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_extract_file_paths_max_3() {
        let msg = "a/b.rs c/d.rs e/f.rs g/h.rs i/j.rs";
        let paths = FileContextStage::extract_file_paths(msg);
        assert_eq!(paths.len(), 3);
    }

    #[test]
    fn test_format_card_minimal() {
        let card = ContextCard {
            path: "src/main.rs".to_string(),
            ..Default::default()
        };
        let text = FileContextStage::format_card(&card);
        assert!(text.contains("src/main.rs"));
        // Low risk → no risk line
        assert!(!text.contains("Risk:"));
    }

    #[test]
    fn test_format_card_full() {
        let card = ContextCard {
            path: "src/chat/manager.rs".to_string(),
            cc_pagerank: 0.85,
            cc_betweenness: 0.72,
            cc_community_id: 3,
            cc_community_label: "Chat & Session".to_string(),
            cc_imports_in: 15,
            cc_imports_out: 8,
            cc_calls_in: 20,
            cc_calls_out: 12,
            cc_co_changers_top5: vec![
                "src/chat/types.rs".to_string(),
                "src/chat/enrichment.rs".to_string(),
            ],
            ..Default::default()
        };
        let text = FileContextStage::format_card(&card);
        assert!(text.contains("Chat & Session"));
        assert!(text.contains("HIGH"));
        assert!(text.contains("Bridge: YES"));
        assert!(text.contains("types.rs"));
        assert!(text.contains("enrichment.rs"));
    }

    #[test]
    fn test_cache_operations() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let stage = FileContextStage::new(mock);

        // Empty cache → None
        assert!(stage.get_cached("test/path.rs").is_none());

        // Set → get returns value
        stage.set_cache("test/path.rs".to_string(), "profile text".to_string());
        assert_eq!(stage.get_cached("test/path.rs").unwrap(), "profile text");
    }
}
