//! Episode Recall Reflex Provider
//!
//! Searches episodic memory for past episodes semantically similar to
//! the current context. Uses cosine similarity on embeddings when available,
//! falls back gracefully to empty results.

use tracing::debug;

use super::{RefContext, Suggestion};
use crate::neo4j::traits::GraphStore;

/// Maximum episode recall suggestions to return.
const MAX_EPISODES: usize = 2;

/// Fetch episode recall suggestions for the given context.
///
/// If an embedding is provided, searches for similar episodes using cosine
/// similarity. Otherwise returns an empty vec (graceful fallback).
///
/// Gracefully returns an empty vec on any error.
pub async fn fetch_episode_suggestions(
    graph: &dyn GraphStore,
    ctx: &RefContext,
) -> Vec<Suggestion> {
    let embedding = match &ctx.embedding {
        Some(e) if !e.is_empty() => e,
        _ => {
            debug!("[reflex:episode] No embedding provided, skipping episode recall");
            return Vec::new();
        }
    };

    // Search for episodes with similar embeddings
    match graph
        .search_episodes_by_embedding(ctx.project_id, embedding, MAX_EPISODES)
        .await
    {
        Ok(episodes) => {
            let suggestions: Vec<Suggestion> = episodes
                .into_iter()
                .map(|(episode, similarity)| Suggestion::EpisodeRecall {
                    episode_id: episode.id,
                    summary: truncate_str(&episode.stimulus.request, 120),
                    similarity,
                })
                .collect();

            debug!(
                "[reflex:episode] Found {} episode recall suggestions",
                suggestions.len()
            );
            suggestions
        }
        Err(e) => {
            debug!("[reflex:episode] Error searching episodes: {}", e);
            Vec::new()
        }
    }
}

/// Truncate a string to a maximum number of characters, adding "..." if truncated.
fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_string()
    } else {
        let mut end = max_chars;
        // Don't cut in the middle of a UTF-8 char
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        let result = truncate_str("hello world this is a long string", 10);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 13); // 10 + "..."
    }

    #[test]
    fn test_truncate_str_exact_boundary() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_empty() {
        assert_eq!(truncate_str("", 10), "");
    }

    #[test]
    fn test_truncate_str_unicode() {
        // Should not panic on multi-byte chars
        let result = truncate_str("héllo wörld café résumé", 10);
        assert!(result.ends_with("..."));
    }

    #[tokio::test]
    async fn test_fetch_episode_no_embedding() {
        use crate::neo4j::mock::MockGraphStore;
        let graph = MockGraphStore::new();
        let ctx = super::RefContext {
            affected_files: vec!["src/main.rs".to_string()],
            task_title: None,
            step_description: None,
            embedding: None,
            project_id: uuid::Uuid::new_v4(),
        };
        let suggestions = fetch_episode_suggestions(&graph, &ctx).await;
        assert!(suggestions.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_episode_empty_embedding() {
        use crate::neo4j::mock::MockGraphStore;
        let graph = MockGraphStore::new();
        let ctx = super::RefContext {
            affected_files: vec![],
            task_title: None,
            step_description: None,
            embedding: Some(vec![]),
            project_id: uuid::Uuid::new_v4(),
        };
        let suggestions = fetch_episode_suggestions(&graph, &ctx).await;
        assert!(suggestions.is_empty());
    }
}
