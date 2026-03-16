//! CoChange Reflex Provider
//!
//! Queries the knowledge graph for files that frequently co-change with
//! the affected files. Filters by coupling > 0.5, excludes files already
//! in the affected set, and limits to 3 suggestions.

use std::time::Duration;

use tokio::time::timeout;
use tracing::{debug, warn};

use super::{RefContext, Suggestion};
use crate::neo4j::traits::GraphStore;

/// Timeout for co-change queries per file.
const CO_CHANGE_TIMEOUT_MS: u64 = 200;

/// Minimum coupling score to include a co-changer.
const MIN_COUPLING: f64 = 0.5;

/// Maximum co-change suggestions to return.
const MAX_CO_CHANGE: usize = 3;

/// Fetch co-change suggestions for the given context.
///
/// For each affected file, queries the graph for co-changers with coupling > 0.5.
/// Excludes files already in the affected set. Sorts by coupling descending and
/// limits to 3 suggestions total.
///
/// Gracefully returns an empty vec on any error or timeout.
pub async fn fetch_co_change_suggestions(
    graph: &dyn GraphStore,
    ctx: &RefContext,
) -> Vec<Suggestion> {
    let mut suggestions = Vec::new();
    let deadline = Duration::from_millis(CO_CHANGE_TIMEOUT_MS);

    for source_file in &ctx.affected_files {
        let result = timeout(deadline, graph.get_file_co_changers(source_file, 1, 10)).await;

        match result {
            Ok(Ok(co_changers)) => {
                for cc in co_changers {
                    // Compute coupling as count normalized (rough heuristic)
                    // The co-changer count is raw — we normalize to 0..1 range
                    // using a sigmoid-like formula: coupling = count / (count + 5)
                    let coupling = cc.count as f64 / (cc.count as f64 + 5.0);

                    if coupling < MIN_COUPLING {
                        continue;
                    }

                    // Skip files already in the affected set
                    if ctx.affected_files.iter().any(|f| f == &cc.path) {
                        continue;
                    }

                    suggestions.push(Suggestion::CoChangeReminder {
                        file_path: cc.path,
                        source_file: source_file.clone(),
                        coupling,
                    });
                }
            }
            Ok(Err(e)) => {
                debug!(
                    "[reflex:co_change] Error querying co-changers for {}: {}",
                    source_file, e
                );
            }
            Err(_) => {
                warn!(
                    "[reflex:co_change] Timeout querying co-changers for {}",
                    source_file
                );
            }
        }
    }

    // Sort by coupling descending and limit
    suggestions.sort_by(|a, b| {
        let ca = if let Suggestion::CoChangeReminder { coupling, .. } = a {
            *coupling
        } else {
            0.0
        };
        let cb = if let Suggestion::CoChangeReminder { coupling, .. } = b {
            *coupling
        } else {
            0.0
        };
        cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
    });
    suggestions.truncate(MAX_CO_CHANGE);

    debug!(
        "[reflex:co_change] Found {} co-change suggestions",
        suggestions.len()
    );
    suggestions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(CO_CHANGE_TIMEOUT_MS, 200);
        assert!((MIN_COUPLING - 0.5).abs() < f64::EPSILON);
        assert_eq!(MAX_CO_CHANGE, 3);
    }

    #[test]
    fn test_coupling_formula() {
        // coupling = count / (count + 5)
        // count=5 → 5/10 = 0.5 (exactly at threshold)
        let coupling_5: f64 = 5.0 / (5.0 + 5.0);
        assert!((coupling_5 - 0.5).abs() < f64::EPSILON);

        // count=10 → 10/15 ≈ 0.667 (above threshold)
        let coupling_10 = 10.0 / (10.0 + 5.0);
        assert!(coupling_10 > MIN_COUPLING);

        // count=1 → 1/6 ≈ 0.167 (below threshold)
        let coupling_1 = 1.0 / (1.0 + 5.0);
        assert!(coupling_1 < MIN_COUPLING);
    }

    #[tokio::test]
    async fn test_fetch_co_change_empty_context() {
        use crate::neo4j::mock::MockGraphStore;
        let graph = MockGraphStore::new();
        let ctx = RefContext {
            affected_files: vec![],
            task_title: None,
            step_description: None,
            embedding: None,
            project_id: uuid::Uuid::new_v4(),
        };
        let suggestions = fetch_co_change_suggestions(&graph, &ctx).await;
        assert!(suggestions.is_empty());
    }
}
