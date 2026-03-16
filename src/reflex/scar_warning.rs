//! Scar Warning Reflex Provider
//!
//! Loads notes attached to affected files and filters by scar_intensity > 0.6.
//! Produces ScarWarning suggestions with truncated content.

use tracing::debug;

use super::{RefContext, Suggestion};
use crate::neo4j::traits::GraphStore;
use crate::notes::models::EntityType;

/// Minimum scar intensity to trigger a warning.
const MIN_SCAR_INTENSITY: f64 = 0.6;

/// Maximum content length for scar warning suggestions.
const MAX_CONTENT_LEN: usize = 100;

/// Fetch scar warning suggestions for the given context.
///
/// For each affected file, loads notes via get_notes_for_entity and filters
/// those with scar_intensity > 0.6. Truncates content and produces ScarWarning
/// suggestions.
///
/// Gracefully returns an empty vec on any error.
pub async fn fetch_scar_suggestions(graph: &dyn GraphStore, ctx: &RefContext) -> Vec<Suggestion> {
    let mut suggestions = Vec::new();

    for file_path in &ctx.affected_files {
        match graph
            .get_notes_for_entity(&EntityType::File, file_path)
            .await
        {
            Ok(notes) => {
                for note in notes {
                    if note.scar_intensity > MIN_SCAR_INTENSITY {
                        let content = truncate_content(&note.content, MAX_CONTENT_LEN);
                        suggestions.push(Suggestion::ScarWarning {
                            note_id: note.id,
                            content,
                            scar_intensity: note.scar_intensity,
                            file_path: file_path.clone(),
                        });
                    }
                }
            }
            Err(e) => {
                debug!("[reflex:scar] Error loading notes for {}: {}", file_path, e);
            }
        }
    }

    debug!(
        "[reflex:scar] Found {} scar warning suggestions",
        suggestions.len()
    );
    suggestions
}

/// Truncate content to a maximum length, preserving word boundaries.
fn truncate_content(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }

    // Find last space before max_len
    let truncated = &s[..max_len];
    match truncated.rfind(' ') {
        Some(pos) if pos > max_len / 2 => format!("{}...", &s[..pos]),
        _ => format!("{}...", truncated),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_content_short() {
        assert_eq!(truncate_content("short", 100), "short");
    }

    #[test]
    fn test_truncate_content_word_boundary() {
        let result = truncate_content("hello world this is a test", 15);
        assert!(result.ends_with("..."));
        // Should break at "hello world" (11 chars), not mid-word
        assert!(result.starts_with("hello world"));
    }

    #[test]
    fn test_truncate_content_exact_boundary() {
        assert_eq!(truncate_content("12345", 5), "12345");
    }

    #[test]
    fn test_truncate_content_no_space() {
        // Very long word with no spaces — should truncate at max_len
        let result = truncate_content("abcdefghijklmnopqrstuvwxyz", 10);
        assert!(result.ends_with("..."));
        assert_eq!(result, "abcdefghij...");
    }

    #[test]
    fn test_truncate_content_empty() {
        assert_eq!(truncate_content("", 100), "");
    }

    #[test]
    fn test_min_scar_intensity_threshold() {
        assert!((MIN_SCAR_INTENSITY - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_content_len() {
        assert_eq!(MAX_CONTENT_LEN, 100);
    }

    #[tokio::test]
    async fn test_fetch_scar_empty_context() {
        use crate::neo4j::mock::MockGraphStore;
        let graph = MockGraphStore::new();
        let ctx = super::RefContext {
            affected_files: vec![],
            task_title: None,
            step_description: None,
            embedding: None,
            project_id: uuid::Uuid::new_v4(),
        };
        let suggestions = fetch_scar_suggestions(&graph, &ctx).await;
        assert!(suggestions.is_empty());
    }
}
