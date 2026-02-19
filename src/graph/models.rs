//! Graph analytics data models.
//!
//! Defines the structs used to represent graph analytics results:
//! - [`NodeMetrics`] — per-node scores (PageRank, betweenness, clustering, community)
//! - [`CommunityInfo`] — metadata about a detected community (Louvain)
//! - [`GraphAnalytics`] — aggregated result of a full analytics run
//! - [`AnalyticsConfig`] — tuning parameters for the analytics algorithms

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Tuning parameters for graph analytics algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// PageRank damping factor (default: 0.85)
    pub pagerank_damping: f64,
    /// PageRank convergence tolerance (default: 1e-6)
    pub pagerank_tolerance: f64,
    /// PageRank maximum iterations (default: 100)
    pub pagerank_max_iterations: usize,
    /// Louvain resolution parameter (default: 1.0, higher = smaller communities)
    pub louvain_resolution: f64,
    /// Louvain maximum iterations (default: 100)
    pub louvain_max_iterations: usize,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            pagerank_damping: 0.85,
            pagerank_tolerance: 1e-6,
            pagerank_max_iterations: 100,
            louvain_resolution: 1.0,
            louvain_max_iterations: 100,
        }
    }
}

// ============================================================================
// Per-node metrics
// ============================================================================

/// Aggregated metrics for a single node in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// Node identifier (file path or symbol name in the knowledge graph)
    pub node_id: String,
    /// Node label/type (e.g. "File", "Function", "Struct")
    pub node_type: String,
    /// PageRank score (0.0–1.0, higher = more important)
    pub pagerank: f64,
    /// Betweenness centrality (0.0–1.0, higher = more bridge-like)
    pub betweenness: f64,
    /// Local clustering coefficient (0.0–1.0, higher = more clustered neighbors)
    pub clustering_coefficient: f64,
    /// Community ID assigned by Louvain algorithm
    pub community_id: usize,
    /// Weakly connected component ID
    pub component_id: usize,
    /// In-degree (number of incoming edges)
    pub in_degree: usize,
    /// Out-degree (number of outgoing edges)
    pub out_degree: usize,
}

// ============================================================================
// Community info
// ============================================================================

/// Metadata about a community detected by the Louvain algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityInfo {
    /// Community identifier
    pub id: usize,
    /// Number of nodes in this community
    pub size: usize,
    /// Sum of PageRank scores of all nodes in this community
    pub total_pagerank: f64,
    /// Internal edge density (edges within / possible edges within)
    pub density: f64,
    /// Representative node IDs (top-3 by PageRank within the community)
    pub top_nodes: Vec<String>,
}

// ============================================================================
// Aggregated analytics result
// ============================================================================

/// Complete result of a graph analytics computation.
///
/// Contains per-node metrics, community summaries, component count,
/// and the global modularity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalytics {
    /// Project identifier (UUID) these analytics belong to
    pub project_id: String,
    /// Per-node metrics, sorted by PageRank descending
    pub nodes: Vec<NodeMetrics>,
    /// Community summaries from Louvain detection
    pub communities: Vec<CommunityInfo>,
    /// Number of weakly connected components
    pub component_count: usize,
    /// Global modularity score from Louvain (0.0–1.0)
    pub modularity: f64,
    /// Total number of nodes analyzed
    pub node_count: usize,
    /// Total number of edges analyzed
    pub edge_count: usize,
    /// Computation time in milliseconds
    pub computation_ms: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_config_defaults() {
        let config = AnalyticsConfig::default();
        assert!((config.pagerank_damping - 0.85).abs() < f64::EPSILON);
        assert!((config.pagerank_tolerance - 1e-6).abs() < f64::EPSILON);
        assert_eq!(config.pagerank_max_iterations, 100);
        assert!((config.louvain_resolution - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.louvain_max_iterations, 100);
    }

    #[test]
    fn test_node_metrics_serialization() {
        let metrics = NodeMetrics {
            node_id: "src/main.rs".to_string(),
            node_type: "File".to_string(),
            pagerank: 0.05,
            betweenness: 0.12,
            clustering_coefficient: 0.33,
            community_id: 0,
            component_id: 0,
            in_degree: 5,
            out_degree: 3,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: NodeMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.node_id, "src/main.rs");
        assert!((deserialized.pagerank - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_graph_analytics_serialization() {
        let analytics = GraphAnalytics {
            project_id: "test-project".to_string(),
            nodes: vec![],
            communities: vec![],
            component_count: 1,
            modularity: 0.45,
            node_count: 0,
            edge_count: 0,
            computation_ms: 42,
        };
        let json = serde_json::to_string(&analytics).unwrap();
        let deserialized: GraphAnalytics = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.project_id, "test-project");
        assert!((deserialized.modularity - 0.45).abs() < f64::EPSILON);
    }
}
