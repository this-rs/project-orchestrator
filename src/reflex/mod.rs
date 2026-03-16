//! ReflexEngine — contextual suggestion engine for the enrichment pipeline.
//!
//! Provides three types of reflexive suggestions:
//! - **CoChangeReminder** — files that frequently co-change with the affected files
//! - **EpisodeRecall** — similar past episodes from episodic memory
//! - **ScarWarning** — notes with high scar intensity on affected files
//!
//! The engine runs all three providers concurrently with a 300ms total budget.
//! On empty graphs or failures, each provider gracefully returns an empty vec.

pub mod co_change;
pub mod episode_recall;
pub mod scar_warning;
pub mod stage;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Suggestion model
// ============================================================================

/// A contextual suggestion produced by the ReflexEngine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Suggestion {
    /// A file that frequently co-changes with one of the affected files.
    CoChangeReminder {
        /// The co-changing file path.
        file_path: String,
        /// The source file that triggered this suggestion.
        source_file: String,
        /// Co-change coupling score (0.0 - 1.0).
        coupling: f64,
    },
    /// A recalled episode that is semantically similar to the current context.
    EpisodeRecall {
        /// The episode ID.
        episode_id: Uuid,
        /// Short summary of the episode stimulus.
        summary: String,
        /// Cosine similarity score (0.0 - 1.0).
        similarity: f64,
    },
    /// A warning from a scarred note on one of the affected files.
    ScarWarning {
        /// The note ID.
        note_id: Uuid,
        /// Truncated note content (max ~100 chars).
        content: String,
        /// Scar intensity (0.0 - 1.0).
        scar_intensity: f64,
        /// The file the note is attached to.
        file_path: String,
    },
}

impl Suggestion {
    /// Returns a priority weight for sorting (higher = more important).
    /// ScarWarning > EpisodeRecall > CoChangeReminder.
    pub fn priority(&self) -> u8 {
        match self {
            Suggestion::ScarWarning { .. } => 3,
            Suggestion::EpisodeRecall { .. } => 2,
            Suggestion::CoChangeReminder { .. } => 1,
        }
    }

    /// Format this suggestion as a single markdown line.
    pub fn to_markdown(&self) -> String {
        match self {
            Suggestion::ScarWarning {
                content,
                file_path,
                scar_intensity,
                ..
            } => {
                format!(
                    "- **SCAR** `{}` (intensity {:.0}%): {}",
                    file_path,
                    scar_intensity * 100.0,
                    content,
                )
            }
            Suggestion::EpisodeRecall {
                summary,
                similarity,
                ..
            } => {
                format!("- **RECALL** (sim {:.0}%): {}", similarity * 100.0, summary,)
            }
            Suggestion::CoChangeReminder {
                file_path,
                source_file,
                coupling,
                ..
            } => {
                format!(
                    "- **CO-CHANGE** `{}` (coupling {:.0}%, from `{}`)",
                    file_path,
                    coupling * 100.0,
                    source_file,
                )
            }
        }
    }
}

// ============================================================================
// RefContext — input context for the ReflexEngine
// ============================================================================

/// Context provided to the ReflexEngine for generating suggestions.
#[derive(Debug, Clone)]
pub struct RefContext {
    /// Files affected by the current task/step.
    pub affected_files: Vec<String>,
    /// Title of the current task (if available).
    pub task_title: Option<String>,
    /// Description of the current step (if available).
    pub step_description: Option<String>,
    /// Embedding vector for semantic search (if available).
    pub embedding: Option<Vec<f32>>,
    /// Project ID.
    pub project_id: Uuid,
}

// ============================================================================
// ReflexEngine — orchestrates all suggestion providers
// ============================================================================

use std::sync::Arc;
use std::time::Duration;

use tokio::time::timeout;
use tracing::{debug, warn};

use crate::neo4j::traits::GraphStore;

/// Maximum number of suggestions returned by the engine.
const MAX_SUGGESTIONS: usize = 5;

/// Total time budget for all providers (ms).
const REFLEX_TIMEOUT_MS: u64 = 300;

/// The ReflexEngine runs co-change, episode recall, and scar warning providers
/// concurrently and merges their results into a sorted, deduplicated list.
pub struct ReflexEngine {
    graph: Arc<dyn GraphStore>,
}

impl ReflexEngine {
    /// Create a new ReflexEngine.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    /// Generate suggestions for the given context.
    ///
    /// Runs all three providers concurrently with a 300ms total budget.
    /// Returns at most `MAX_SUGGESTIONS` suggestions, sorted by priority
    /// (ScarWarning > EpisodeRecall > CoChangeReminder).
    pub async fn suggest(&self, ctx: &RefContext) -> Vec<Suggestion> {
        let deadline = Duration::from_millis(REFLEX_TIMEOUT_MS);

        let graph_co = self.graph.clone();
        let graph_ep = self.graph.clone();
        let graph_sc = self.graph.clone();

        let ctx_co = ctx.clone();
        let ctx_ep = ctx.clone();
        let ctx_sc = ctx.clone();

        // Run all three providers concurrently under a shared timeout
        let combined = timeout(deadline, async move {
            let (co_changes, episodes, scars) = tokio::join!(
                co_change::fetch_co_change_suggestions(graph_co.as_ref(), &ctx_co),
                episode_recall::fetch_episode_suggestions(graph_ep.as_ref(), &ctx_ep),
                scar_warning::fetch_scar_suggestions(graph_sc.as_ref(), &ctx_sc),
            );
            (co_changes, episodes, scars)
        })
        .await;

        let mut suggestions = Vec::new();

        match combined {
            Ok((co_changes, episodes, scars)) => {
                suggestions.extend(co_changes);
                suggestions.extend(episodes);
                suggestions.extend(scars);
            }
            Err(_) => {
                warn!(
                    "[reflex] Total timeout ({}ms) exceeded, returning partial results",
                    REFLEX_TIMEOUT_MS
                );
            }
        }

        // Sort by priority descending, then by score within each category
        suggestions.sort_by(|a, b| {
            b.priority().cmp(&a.priority()).then_with(|| {
                let score_b = suggestion_score(b);
                let score_a = suggestion_score(a);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        suggestions.truncate(MAX_SUGGESTIONS);

        debug!(
            "[reflex] Generated {} suggestions for {} affected files",
            suggestions.len(),
            ctx.affected_files.len(),
        );

        suggestions
    }

    /// Format suggestions as a markdown block for prompt injection.
    ///
    /// Returns an empty string if there are no suggestions.
    /// Output is capped at ~200 tokens.
    pub fn format_markdown(suggestions: &[Suggestion]) -> String {
        if suggestions.is_empty() {
            return String::new();
        }

        let mut lines = Vec::new();
        let mut char_count = 0;
        // Rough estimate: 1 token ~= 4 chars, 200 tokens ~= 800 chars
        let max_chars = 800;

        for s in suggestions {
            let line = s.to_markdown();
            char_count += line.len();
            if char_count > max_chars {
                break;
            }
            lines.push(line);
        }

        format!("Reflexes:\n{}", lines.join("\n"))
    }
}

/// Extract the numeric score from a suggestion for secondary sorting.
fn suggestion_score(s: &Suggestion) -> f64 {
    match s {
        Suggestion::CoChangeReminder { coupling, .. } => *coupling,
        Suggestion::EpisodeRecall { similarity, .. } => *similarity,
        Suggestion::ScarWarning { scar_intensity, .. } => *scar_intensity,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggestion_priority_ordering() {
        let scar = Suggestion::ScarWarning {
            note_id: Uuid::new_v4(),
            content: "Don't use unwrap here".to_string(),
            scar_intensity: 0.8,
            file_path: "src/main.rs".to_string(),
        };
        let episode = Suggestion::EpisodeRecall {
            episode_id: Uuid::new_v4(),
            summary: "Similar refactoring".to_string(),
            similarity: 0.9,
        };
        let co_change = Suggestion::CoChangeReminder {
            file_path: "src/lib.rs".to_string(),
            source_file: "src/main.rs".to_string(),
            coupling: 0.7,
        };

        assert!(scar.priority() > episode.priority());
        assert!(episode.priority() > co_change.priority());
    }

    #[test]
    fn test_suggestion_markdown() {
        let scar = Suggestion::ScarWarning {
            note_id: Uuid::new_v4(),
            content: "Avoid unwrap".to_string(),
            scar_intensity: 0.8,
            file_path: "src/main.rs".to_string(),
        };
        let md = scar.to_markdown();
        assert!(md.contains("SCAR"));
        assert!(md.contains("src/main.rs"));
        assert!(md.contains("80%"));
    }

    #[test]
    fn test_format_markdown_empty() {
        assert_eq!(ReflexEngine::format_markdown(&[]), "");
    }

    #[test]
    fn test_format_markdown_capped() {
        let suggestions: Vec<Suggestion> = (0..20)
            .map(|i| Suggestion::CoChangeReminder {
                file_path: format!("src/very/long/path/to/file_{}.rs", i),
                source_file: "src/origin.rs".to_string(),
                coupling: 0.6,
            })
            .collect();
        let md = ReflexEngine::format_markdown(&suggestions);
        // Should be capped at ~800 chars
        assert!(md.len() <= 1000); // some slack for header
    }

    #[test]
    fn test_ref_context_clone() {
        let ctx = RefContext {
            affected_files: vec!["a.rs".to_string()],
            task_title: Some("Test".to_string()),
            step_description: None,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            project_id: Uuid::new_v4(),
        };
        let cloned = ctx.clone();
        assert_eq!(cloned.affected_files, ctx.affected_files);
        assert_eq!(cloned.project_id, ctx.project_id);
    }

    #[test]
    fn test_suggestion_score() {
        let co = Suggestion::CoChangeReminder {
            file_path: "a.rs".into(),
            source_file: "b.rs".into(),
            coupling: 0.75,
        };
        assert!((suggestion_score(&co) - 0.75).abs() < f64::EPSILON);

        let ep = Suggestion::EpisodeRecall {
            episode_id: Uuid::new_v4(),
            summary: "test".into(),
            similarity: 0.9,
        };
        assert!((suggestion_score(&ep) - 0.9).abs() < f64::EPSILON);

        let sc = Suggestion::ScarWarning {
            note_id: Uuid::new_v4(),
            content: "danger".into(),
            scar_intensity: 0.85,
            file_path: "c.rs".into(),
        };
        assert!((suggestion_score(&sc) - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_episode_recall_markdown() {
        let ep = Suggestion::EpisodeRecall {
            episode_id: Uuid::new_v4(),
            summary: "Fixed null pointer".to_string(),
            similarity: 0.92,
        };
        let md = ep.to_markdown();
        assert!(md.contains("RECALL"));
        assert!(md.contains("92%"));
        assert!(md.contains("Fixed null pointer"));
    }

    #[test]
    fn test_co_change_markdown() {
        let co = Suggestion::CoChangeReminder {
            file_path: "src/api.rs".to_string(),
            source_file: "src/routes.rs".to_string(),
            coupling: 0.65,
        };
        let md = co.to_markdown();
        assert!(md.contains("CO-CHANGE"));
        assert!(md.contains("src/api.rs"));
        assert!(md.contains("src/routes.rs"));
        assert!(md.contains("65%"));
    }

    #[test]
    fn test_format_markdown_single() {
        let suggestions = vec![Suggestion::ScarWarning {
            note_id: Uuid::new_v4(),
            content: "Watch out".to_string(),
            scar_intensity: 0.7,
            file_path: "src/danger.rs".to_string(),
        }];
        let md = ReflexEngine::format_markdown(&suggestions);
        assert!(md.starts_with("Reflexes:"));
        assert!(md.contains("SCAR"));
    }

    #[test]
    fn test_suggestion_sorting_by_priority_then_score() {
        let mut suggestions = [
            Suggestion::CoChangeReminder {
                file_path: "a.rs".into(),
                source_file: "b.rs".into(),
                coupling: 0.9,
            },
            Suggestion::ScarWarning {
                note_id: Uuid::new_v4(),
                content: "low scar".into(),
                scar_intensity: 0.6,
                file_path: "c.rs".into(),
            },
            Suggestion::ScarWarning {
                note_id: Uuid::new_v4(),
                content: "high scar".into(),
                scar_intensity: 0.95,
                file_path: "d.rs".into(),
            },
            Suggestion::EpisodeRecall {
                episode_id: Uuid::new_v4(),
                summary: "past event".into(),
                similarity: 0.8,
            },
        ];

        suggestions.sort_by(|a, b| {
            b.priority().cmp(&a.priority()).then_with(|| {
                let score_b = suggestion_score(b);
                let score_a = suggestion_score(a);
                score_b
                    .partial_cmp(&score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        // ScarWarnings first (priority 3), then Episode (2), then CoChange (1)
        assert_eq!(suggestions[0].priority(), 3);
        assert_eq!(suggestions[1].priority(), 3);
        assert_eq!(suggestions[2].priority(), 2);
        assert_eq!(suggestions[3].priority(), 1);

        // Within same priority, higher score first
        assert!((suggestion_score(&suggestions[0]) - 0.95).abs() < f64::EPSILON);
        assert!((suggestion_score(&suggestions[1]) - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ref_context_defaults() {
        let ctx = RefContext {
            affected_files: vec![],
            task_title: None,
            step_description: None,
            embedding: None,
            project_id: Uuid::nil(),
        };
        assert!(ctx.affected_files.is_empty());
        assert!(ctx.task_title.is_none());
        assert!(ctx.embedding.is_none());
    }
}
