//! Analytics results writer.
//!
//! Batch-updates computed analytics scores back to Neo4j nodes.
//! Uses new `GraphStore` trait methods to write PageRank, betweenness,
//! clustering coefficient, community ID, and component ID as node properties.
//!
//! The write is idempotent: re-running analytics overwrites previous scores.

use crate::neo4j::GraphStore;
use anyhow::Result;
use std::sync::Arc;

use super::models::{
    CodeGraph, CodeNodeType, FileAnalyticsUpdate, FunctionAnalyticsUpdate, GraphAnalytics,
};

/// Writes analytics results back to Neo4j via the `GraphStore` trait.
///
/// Separates metrics by node type (File vs Function) and performs two
/// batch UNWIND queries for maximum efficiency.
pub struct AnalyticsWriter {
    store: Arc<dyn GraphStore>,
}

impl AnalyticsWriter {
    /// Create a new writer backed by the given GraphStore.
    pub fn new(store: Arc<dyn GraphStore>) -> Self {
        Self { store }
    }

    /// Write all analytics results back to Neo4j.
    ///
    /// 1. Separates metrics into File and Function updates
    /// 2. Resolves community labels from `CommunityInfo` for File nodes
    /// 3. Calls `batch_update_file_analytics` + `batch_update_function_analytics`
    ///
    /// Nodes that no longer exist in Neo4j are silently ignored (MATCH finds nothing).
    pub async fn write_analytics(
        &self,
        analytics: &GraphAnalytics,
        graph: &CodeGraph,
    ) -> Result<()> {
        // Build a community_id → label lookup
        let community_labels: std::collections::HashMap<u32, &str> = analytics
            .communities
            .iter()
            .map(|c| (c.id, c.label.as_str()))
            .collect();

        let mut file_updates: Vec<FileAnalyticsUpdate> = Vec::new();
        let mut function_updates: Vec<FunctionAnalyticsUpdate> = Vec::new();

        for (node_id, metrics) in &analytics.metrics {
            // Determine node type from the graph
            let node = match graph.get_node(node_id) {
                Some(n) => n,
                None => continue, // Node not in graph (shouldn't happen)
            };

            match node.node_type {
                CodeNodeType::File => {
                    let label = community_labels
                        .get(&metrics.community_id)
                        .unwrap_or(&"unknown");
                    file_updates.push(FileAnalyticsUpdate {
                        path: node_id.clone(),
                        pagerank: metrics.pagerank,
                        betweenness: metrics.betweenness,
                        community_id: metrics.community_id,
                        community_label: label.to_string(),
                        clustering_coefficient: metrics.clustering_coefficient,
                        component_id: metrics.component_id,
                    });
                }
                CodeNodeType::Function => {
                    function_updates.push(FunctionAnalyticsUpdate {
                        name: node_id.clone(),
                        pagerank: metrics.pagerank,
                        betweenness: metrics.betweenness,
                        community_id: metrics.community_id,
                        clustering_coefficient: metrics.clustering_coefficient,
                        component_id: metrics.component_id,
                    });
                }
                // Struct/Trait/Enum nodes don't get analytics scores (yet)
                _ => {}
            }
        }

        // Batch write both types
        if !file_updates.is_empty() {
            self.store
                .batch_update_file_analytics(&file_updates)
                .await?;
        }
        if !function_updates.is_empty() {
            self.store
                .batch_update_function_analytics(&function_updates)
                .await?;
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::algorithms::compute_all;
    use crate::graph::models::{
        AnalyticsConfig, CodeEdge, CodeEdgeType, CodeGraph, CodeNode, CodeNodeType,
    };
    use crate::neo4j::mock::MockGraphStore;

    /// Build a small file graph for testing the writer.
    fn make_file_graph() -> CodeGraph {
        let mut g = CodeGraph::new();
        let files = [
            "src/main.rs",
            "src/lib.rs",
            "src/api/mod.rs",
            "src/api/handlers.rs",
            "src/api/routes.rs",
        ];
        for path in &files {
            g.add_node(CodeNode {
                id: path.to_string(),
                node_type: CodeNodeType::File,
                path: Some(path.to_string()),
                name: path.rsplit('/').next().unwrap_or(path).to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "src/main.rs",
            "src/lib.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/main.rs",
            "src/api/mod.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/api/mod.rs",
            "src/api/handlers.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/api/mod.rs",
            "src/api/routes.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g.add_edge(
            "src/api/routes.rs",
            "src/api/handlers.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        g
    }

    /// Build a function graph for testing.
    fn make_function_graph() -> CodeGraph {
        let mut g = CodeGraph::new();
        for name in &["main", "setup", "handle_request", "respond"] {
            g.add_node(CodeNode {
                id: name.to_string(),
                node_type: CodeNodeType::Function,
                path: None,
                name: name.to_string(),
                project_id: None,
            });
        }
        g.add_edge(
            "main",
            "setup",
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );
        g.add_edge(
            "main",
            "handle_request",
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );
        g.add_edge(
            "handle_request",
            "respond",
            CodeEdge {
                edge_type: CodeEdgeType::Calls,
                weight: 1.0,
            },
        );
        g
    }

    #[tokio::test]
    async fn test_write_file_analytics() {
        let store = Arc::new(MockGraphStore::new());
        let graph = make_file_graph();
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&graph, &config);

        assert_eq!(analytics.metrics.len(), 5);

        let writer = AnalyticsWriter::new(store.clone());
        writer.write_analytics(&analytics, &graph).await.unwrap();

        // Verify file analytics were persisted
        let fa = store.file_analytics.read().await;
        assert_eq!(fa.len(), 5, "Should have 5 file analytics entries");

        // Check specific values
        let main_rs = &fa["src/main.rs"];
        assert!(main_rs.pagerank > 0.0, "PageRank should be positive");
        assert!(
            main_rs.betweenness >= 0.0,
            "Betweenness should be non-negative"
        );
        assert!(
            !main_rs.community_label.is_empty(),
            "Community label should be set"
        );

        // Function analytics should be empty (no function nodes)
        let func_a = store.function_analytics.read().await;
        assert!(func_a.is_empty());
    }

    #[tokio::test]
    async fn test_write_function_analytics() {
        let store = Arc::new(MockGraphStore::new());
        let graph = make_function_graph();
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&graph, &config);

        let writer = AnalyticsWriter::new(store.clone());
        writer.write_analytics(&analytics, &graph).await.unwrap();

        // Verify function analytics were persisted
        let func_a = store.function_analytics.read().await;
        assert_eq!(
            func_a.len(),
            4,
            "Should have 4 function analytics entries"
        );

        let main_fn = &func_a["main"];
        assert!(main_fn.pagerank > 0.0);

        // File analytics should be empty
        let fa = store.file_analytics.read().await;
        assert!(fa.is_empty());
    }

    #[tokio::test]
    async fn test_write_empty_analytics() {
        let store = Arc::new(MockGraphStore::new());
        let graph = CodeGraph::new();
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&graph, &config);

        let writer = AnalyticsWriter::new(store.clone());
        writer.write_analytics(&analytics, &graph).await.unwrap();

        // Both should be empty
        let fa = store.file_analytics.read().await;
        let func_a = store.function_analytics.read().await;
        assert!(fa.is_empty());
        assert!(func_a.is_empty());
    }

    #[tokio::test]
    async fn test_write_idempotent() {
        let store = Arc::new(MockGraphStore::new());
        let graph = make_file_graph();
        let config = AnalyticsConfig::default();
        let analytics = compute_all(&graph, &config);

        let writer = AnalyticsWriter::new(store.clone());

        // Write twice
        writer.write_analytics(&analytics, &graph).await.unwrap();
        writer.write_analytics(&analytics, &graph).await.unwrap();

        // Should still have 5 entries (overwritten, not duplicated)
        let fa = store.file_analytics.read().await;
        assert_eq!(fa.len(), 5);
    }

    #[tokio::test]
    async fn test_write_mixed_graph() {
        // Graph with both File and Function nodes
        let mut graph = CodeGraph::new();
        graph.add_node(CodeNode {
            id: "src/main.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/main.rs".to_string()),
            name: "main.rs".to_string(),
            project_id: None,
        });
        graph.add_node(CodeNode {
            id: "src/lib.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/lib.rs".to_string()),
            name: "lib.rs".to_string(),
            project_id: None,
        });
        graph.add_node(CodeNode {
            id: "main".to_string(),
            node_type: CodeNodeType::Function,
            path: None,
            name: "main".to_string(),
            project_id: None,
        });
        graph.add_edge(
            "src/main.rs",
            "src/lib.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        graph.add_edge(
            "src/main.rs",
            "main",
            CodeEdge {
                edge_type: CodeEdgeType::Defines,
                weight: 1.0,
            },
        );

        let config = AnalyticsConfig::default();
        let analytics = compute_all(&graph, &config);

        let store = Arc::new(MockGraphStore::new());
        let writer = AnalyticsWriter::new(store.clone());
        writer.write_analytics(&analytics, &graph).await.unwrap();

        let fa = store.file_analytics.read().await;
        let func_a = store.function_analytics.read().await;
        assert_eq!(fa.len(), 2, "Should have 2 file analytics");
        assert_eq!(func_a.len(), 1, "Should have 1 function analytics");
    }
}
