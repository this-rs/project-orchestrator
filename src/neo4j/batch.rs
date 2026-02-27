//! Batch utilities for Neo4j operations.
//!
//! Provides helpers to run UNWIND queries in chunks of [`BATCH_SIZE`] items,
//! preventing OOM and timeout on large batch operations.
//!
//! # Pattern reference
//!
//! ## UNWIND batch (create/upsert)
//! ```ignore
//! use crate::neo4j::batch::{run_unwind_in_chunks, BoltMap, BATCH_SIZE};
//!
//! let items: Vec<BoltMap> = build_items(&data);
//! run_unwind_in_chunks(&self.graph, items, r#"
//!     UNWIND $items AS item
//!     MERGE (n:MyNode {id: item.id})
//!     SET n.name = item.name
//! "#).await?;
//! ```
//!
//! ## Cleanup batch (delete with LIMIT loop)
//! ```ignore
//! use crate::neo4j::batch::cleanup_in_batches;
//!
//! let deleted = cleanup_in_batches(&self.graph, r#"
//!     MATCH ()-[r:MY_REL]->()
//!     WITH r LIMIT $batch_size
//!     DELETE r
//!     RETURN count(r) AS cnt
//! "#).await?;
//! ```

use anyhow::{Context, Result};
use neo4rs::query;

/// Maximum items per UNWIND batch to prevent OOM/timeout.
///
/// Based on production experience: Neo4j transactions with >10K UNWIND items
/// can cause memory pressure and timeouts on standard configurations.
/// The cleanup_sync_data pattern (which handles 300K+ CALLS rels) uses this limit.
pub const BATCH_SIZE: usize = 10_000;

/// Type alias for the HashMap used in UNWIND parameters.
pub type BoltMap = std::collections::HashMap<String, neo4rs::BoltType>;

/// Run a Cypher UNWIND query in chunks of [`BATCH_SIZE`].
///
/// The query MUST contain an `$items` parameter for the UNWIND clause.
/// Items are automatically split into chunks and each chunk is executed
/// as a separate transaction.
///
/// # Arguments
/// * `graph` - Neo4j graph connection
/// * `items` - Full list of items to process
/// * `cypher` - Cypher query using `UNWIND $items AS ...`
///
/// # Returns
/// Number of chunks executed. Errors on first failed chunk.
///
/// # Example
/// ```ignore
/// let items: Vec<BoltMap> = functions.iter().map(|f| {
///     let mut m = BoltMap::new();
///     m.insert("id".into(), f.id.clone().into());
///     m.insert("name".into(), f.name.clone().into());
///     m
/// }).collect();
///
/// run_unwind_in_chunks(&graph, items, r#"
///     UNWIND $items AS func
///     MERGE (f:Function {id: func.id})
///     SET f.name = func.name
/// "#).await?;
/// ```
pub async fn run_unwind_in_chunks(
    graph: &neo4rs::Graph,
    items: Vec<BoltMap>,
    cypher: &str,
) -> Result<usize> {
    if items.is_empty() {
        return Ok(0);
    }

    let total_items = items.len();
    let mut chunks_executed = 0;

    for chunk in items.chunks(BATCH_SIZE) {
        let q = query(cypher).param("items", chunk.to_vec());
        graph.run(q).await.with_context(|| {
            format!(
                "UNWIND batch chunk {} failed ({}/{} items)",
                chunks_executed,
                chunk.len(),
                total_items
            )
        })?;
        chunks_executed += 1;
    }

    if chunks_executed > 1 {
        tracing::debug!(
            "run_unwind_in_chunks: processed {} items in {} chunks",
            total_items,
            chunks_executed
        );
    }

    Ok(chunks_executed)
}

/// Run a Cypher UNWIND query in chunks with additional static parameters.
///
/// Like [`run_unwind_in_chunks`] but allows passing extra parameters
/// (e.g., `$project_id`) that are the same for every chunk.
///
/// # Arguments
/// * `graph` - Neo4j graph connection
/// * `items` - Full list of items to process
/// * `cypher` - Cypher query using `UNWIND $items AS ...`
/// * `build_query` - Closure that builds the query for each chunk.
///   Receives a pre-built `Query` with `$items` already set.
///   The closure should add any extra `.param(...)` calls.
///
/// # Example
/// ```ignore
/// run_unwind_in_chunks_with(
///     &graph,
///     items,
///     "UNWIND $items AS call ...",
///     |q| q.param("project_id", pid.to_string()),
/// ).await?;
/// ```
#[allow(dead_code)]
pub async fn run_unwind_in_chunks_with<F>(
    graph: &neo4rs::Graph,
    items: Vec<BoltMap>,
    cypher: &str,
    add_params: F,
) -> Result<usize>
where
    F: Fn(neo4rs::Query) -> neo4rs::Query,
{
    if items.is_empty() {
        return Ok(0);
    }

    let total_items = items.len();
    let mut chunks_executed = 0;

    for chunk in items.chunks(BATCH_SIZE) {
        let q = query(cypher).param("items", chunk.to_vec());
        let q = add_params(q);
        graph.run(q).await.with_context(|| {
            format!(
                "UNWIND batch chunk {} failed ({}/{} items)",
                chunks_executed,
                chunk.len(),
                total_items
            )
        })?;
        chunks_executed += 1;
    }

    if chunks_executed > 1 {
        tracing::debug!(
            "run_unwind_in_chunks_with: processed {} items in {} chunks",
            total_items,
            chunks_executed
        );
    }

    Ok(chunks_executed)
}

/// Run a cleanup query in batches using the LIMIT loop pattern.
///
/// The query MUST:
/// 1. Contain `$batch_size` parameter (injected automatically as [`BATCH_SIZE`])
/// 2. Return a column named `cnt` with the count of deleted items
///
/// The loop continues until `cnt` returns 0.
///
/// # Arguments
/// * `graph` - Neo4j graph connection
/// * `cypher` - Cleanup query with `LIMIT $batch_size` and `RETURN count(...) AS cnt`
/// * `label` - Human-readable label for logging (e.g., "CALLS rels")
///
/// # Returns
/// Total number of items deleted across all batches.
///
/// # Example
/// ```ignore
/// let deleted = cleanup_in_batches(&graph, r#"
///     MATCH ()-[r:EXTENDS]->()
///     WITH r LIMIT $batch_size
///     DELETE r
///     RETURN count(r) AS cnt
/// "#, "EXTENDS rels").await?;
/// ```
#[allow(dead_code)]
pub async fn cleanup_in_batches(graph: &neo4rs::Graph, cypher: &str, label: &str) -> Result<i64> {
    let mut total_deleted: i64 = 0;

    loop {
        let q = query(cypher).param("batch_size", BATCH_SIZE as i64);
        match graph.execute(q).await {
            Ok(mut result) => {
                if let Ok(Some(row)) = result.next().await {
                    let cnt: i64 = row.get("cnt").unwrap_or(0);
                    if cnt == 0 {
                        break;
                    }
                    total_deleted += cnt;
                    tracing::info!(
                        "cleanup_in_batches({}): deleted batch of {} (total: {})",
                        label,
                        cnt,
                        total_deleted
                    );
                } else {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!("cleanup_in_batches({}) failed: {}", label, e);
                break;
            }
        }
    }

    if total_deleted > 0 {
        tracing::info!(
            "cleanup_in_batches({}): finished — deleted {} total",
            label,
            total_deleted
        );
    }

    Ok(total_deleted)
}

/// Build a [`BoltMap`] from key-value pairs.
///
/// Convenience macro-like helper for constructing UNWIND items.
///
/// # Example
/// ```ignore
/// let m = bolt_map(&[
///     ("id", func.id.clone().into()),
///     ("name", func.name.clone().into()),
/// ]);
/// ```
#[allow(dead_code)]
pub fn bolt_map(pairs: &[(&str, neo4rs::BoltType)]) -> BoltMap {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_size_is_10k() {
        assert_eq!(BATCH_SIZE, 10_000);
    }

    #[test]
    fn test_bolt_map_builds_correctly() {
        let m = bolt_map(&[
            ("id", "test-123".to_string().into()),
            ("name", "my_function".to_string().into()),
        ]);

        assert_eq!(m.len(), 2);
        assert!(m.contains_key("id"));
        assert!(m.contains_key("name"));
    }

    #[test]
    fn test_bolt_map_empty() {
        let m = bolt_map(&[]);
        assert!(m.is_empty());
    }

    #[test]
    fn test_chunking_math() {
        // Verify that chunks(BATCH_SIZE) produces expected number of chunks
        let items: Vec<i32> = (0..25_001).collect();
        let chunks: Vec<&[i32]> = items.chunks(BATCH_SIZE).collect();
        assert_eq!(chunks.len(), 3); // 10K + 10K + 5001
        assert_eq!(chunks[0].len(), 10_000);
        assert_eq!(chunks[1].len(), 10_000);
        assert_eq!(chunks[2].len(), 5_001);
    }

    #[test]
    fn test_chunking_exact_boundary() {
        let items: Vec<i32> = (0..10_000).collect();
        let chunks: Vec<&[i32]> = items.chunks(BATCH_SIZE).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 10_000);
    }

    #[test]
    fn test_chunking_single_item() {
        let items: Vec<i32> = vec![1];
        let chunks: Vec<&[i32]> = items.chunks(BATCH_SIZE).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1);
    }

    #[test]
    fn test_bolt_map_with_various_types() {
        let m = bolt_map(&[
            ("string_val", "hello".to_string().into()),
            ("int_val", (42i64).into()),
            ("bool_val", true.into()),
        ]);

        assert_eq!(m.len(), 3);
    }
}
