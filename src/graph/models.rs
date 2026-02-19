//! Graph analytics data models.
//!
//! Defines the complete type system for graph analytics:
//!
//! ## Input types (extraction)
//! - [`CodeNodeType`] / [`CodeNode`] — nodes extracted from the knowledge graph
//! - [`CodeEdgeType`] / [`CodeEdge`] — relationships extracted from the knowledge graph
//! - [`CodeGraph`] — petgraph wrapper with ID ↔ NodeIndex mapping
//!
//! ## Output types (analytics)
//! - [`NodeMetrics`] — per-node scores (PageRank, betweenness, clustering, community)
//! - [`CommunityInfo`] — metadata about a Louvain community
//! - [`ComponentInfo`] — metadata about a weakly connected component
//! - [`CodeHealthReport`] — global health metrics (god functions, circular deps, orphans)
//! - [`GraphAnalytics`] — aggregated result of a full analytics run
//!
//! ## Configuration
//! - [`AnalyticsConfig`] — tuning parameters for the analytics algorithms

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Input types — Graph structure (extraction → algorithms)
// ============================================================================

/// Type of code entity extracted from the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeNodeType {
    File,
    Function,
    Struct,
    Trait,
    Enum,
}

impl std::fmt::Display for CodeNodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File => write!(f, "File"),
            Self::Function => write!(f, "Function"),
            Self::Struct => write!(f, "Struct"),
            Self::Trait => write!(f, "Trait"),
            Self::Enum => write!(f, "Enum"),
        }
    }
}

/// A code entity node extracted from the Neo4j knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeNode {
    /// Unique identifier (file path or qualified symbol name)
    pub id: String,
    /// Type of code entity
    pub node_type: CodeNodeType,
    /// File path (for File nodes, or the containing file for symbols)
    pub path: Option<String>,
    /// Display name (filename or symbol name)
    pub name: String,
    /// Project UUID this node belongs to
    pub project_id: Option<String>,
}

/// Type of relationship between code entities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodeEdgeType {
    Imports,
    Calls,
    Defines,
    ImplementsTrait,
    ImplementsFor,
}

impl std::fmt::Display for CodeEdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Imports => write!(f, "IMPORTS"),
            Self::Calls => write!(f, "CALLS"),
            Self::Defines => write!(f, "DEFINES"),
            Self::ImplementsTrait => write!(f, "IMPLEMENTS_TRAIT"),
            Self::ImplementsFor => write!(f, "IMPLEMENTS_FOR"),
        }
    }
}

/// A relationship (edge) between two code entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEdge {
    /// Type of relationship
    pub edge_type: CodeEdgeType,
    /// Edge weight (default: 1.0, can be adjusted for weighted algorithms)
    pub weight: f64,
}

impl Default for CodeEdge {
    fn default() -> Self {
        Self {
            edge_type: CodeEdgeType::Calls,
            weight: 1.0,
        }
    }
}

// ============================================================================
// CodeGraph — petgraph wrapper with ID mapping
// ============================================================================

/// Wrapper around `petgraph::DiGraph` with bidirectional ID ↔ NodeIndex mapping.
///
/// This is the intermediate representation between Neo4j extraction and
/// algorithm computation. The `id_to_index` HashMap enables O(1) lookups
/// by node ID (file path or symbol name).
#[derive(Debug, Clone)]
pub struct CodeGraph {
    /// The underlying directed graph
    pub graph: DiGraph<CodeNode, CodeEdge>,
    /// Mapping from node ID (String) to petgraph NodeIndex
    pub id_to_index: HashMap<String, NodeIndex>,
}

impl CodeGraph {
    /// Create a new empty CodeGraph.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_index: HashMap::new(),
        }
    }

    /// Create a CodeGraph with pre-allocated capacity.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: DiGraph::with_capacity(nodes, edges),
            id_to_index: HashMap::with_capacity(nodes),
        }
    }

    /// Add a node to the graph. Returns the NodeIndex.
    /// If a node with the same ID already exists, returns its existing index.
    pub fn add_node(&mut self, node: CodeNode) -> NodeIndex {
        if let Some(&idx) = self.id_to_index.get(&node.id) {
            return idx;
        }
        let id = node.id.clone();
        let idx = self.graph.add_node(node);
        self.id_to_index.insert(id, idx);
        idx
    }

    /// Add an edge between two nodes identified by their IDs.
    /// Returns `Some(EdgeIndex)` if both nodes exist, `None` otherwise.
    pub fn add_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge: CodeEdge,
    ) -> Option<petgraph::graph::EdgeIndex> {
        let from_idx = self.id_to_index.get(from_id)?;
        let to_idx = self.id_to_index.get(to_id)?;
        Some(self.graph.add_edge(*from_idx, *to_idx, edge))
    }

    /// Get a reference to a node by its ID.
    pub fn get_node(&self, id: &str) -> Option<&CodeNode> {
        let idx = self.id_to_index.get(id)?;
        self.graph.node_weight(*idx)
    }

    /// Get the NodeIndex for a given ID.
    pub fn get_index(&self, id: &str) -> Option<NodeIndex> {
        self.id_to_index.get(id).copied()
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Output types — Analytics results
// ============================================================================

/// Per-node analytics metrics computed by the graph algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// PageRank score (0.0–1.0, higher = more important)
    pub pagerank: f64,
    /// Betweenness centrality (0.0–1.0, higher = more bridge-like)
    pub betweenness: f64,
    /// Community ID assigned by Louvain algorithm
    pub community_id: u32,
    /// Local clustering coefficient (0.0–1.0, higher = more clustered neighbors)
    pub clustering_coefficient: f64,
    /// Weakly connected component ID
    pub component_id: u32,
    /// In-degree (number of incoming edges)
    pub in_degree: usize,
    /// Out-degree (number of outgoing edges)
    pub out_degree: usize,
}

impl Default for NodeMetrics {
    fn default() -> Self {
        Self {
            pagerank: 0.0,
            betweenness: 0.0,
            community_id: 0,
            clustering_coefficient: 0.0,
            component_id: 0,
            in_degree: 0,
            out_degree: 0,
        }
    }
}

/// Metadata about a community detected by the Louvain algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityInfo {
    /// Community identifier
    pub id: u32,
    /// Number of nodes in this community
    pub size: usize,
    /// Node IDs belonging to this community
    pub members: Vec<String>,
    /// Auto-generated label (e.g. top file or function names)
    pub label: String,
}

/// Metadata about a weakly connected component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInfo {
    /// Component identifier
    pub id: u32,
    /// Number of nodes in this component
    pub size: usize,
    /// Node IDs belonging to this component
    pub members: Vec<String>,
    /// Whether this is the largest (main) component
    pub is_main: bool,
}

/// Global code health metrics derived from the graph structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeHealthReport {
    /// Functions with degree (in + out) above the 95th percentile
    pub god_functions: Vec<String>,
    /// Cycles detected in the dependency graph (each cycle is a list of node IDs)
    pub circular_dependencies: Vec<Vec<String>>,
    /// Files with no incoming or outgoing edges (isolated)
    pub orphan_files: Vec<String>,
    /// Average coupling score across all nodes (mean degree / max possible)
    pub avg_coupling: f64,
    /// Maximum coupling score (highest-degree node / total nodes)
    pub max_coupling: f64,
}

impl Default for CodeHealthReport {
    fn default() -> Self {
        Self {
            god_functions: vec![],
            circular_dependencies: vec![],
            orphan_files: vec![],
            avg_coupling: 0.0,
            max_coupling: 0.0,
        }
    }
}

// ============================================================================
// Aggregated analytics result
// ============================================================================

/// Complete result of a graph analytics computation.
///
/// Contains per-node metrics indexed by node ID, community summaries,
/// component summaries, and global health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalytics {
    /// Per-node metrics keyed by node ID
    pub metrics: HashMap<String, NodeMetrics>,
    /// Community summaries from Louvain detection
    pub communities: Vec<CommunityInfo>,
    /// Weakly connected component summaries
    pub components: Vec<ComponentInfo>,
    /// Code health report (god functions, circular deps, orphans, coupling)
    pub health: CodeHealthReport,
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
    /// Percentile threshold for "god function" detection (default: 0.95)
    pub god_function_percentile: f64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            pagerank_damping: 0.85,
            pagerank_tolerance: 1e-6,
            pagerank_max_iterations: 100,
            louvain_resolution: 1.0,
            louvain_max_iterations: 100,
            god_function_percentile: 0.95,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- AnalyticsConfig ---

    #[test]
    fn test_analytics_config_defaults() {
        let config = AnalyticsConfig::default();
        assert!((config.pagerank_damping - 0.85).abs() < f64::EPSILON);
        assert!((config.pagerank_tolerance - 1e-6).abs() < f64::EPSILON);
        assert_eq!(config.pagerank_max_iterations, 100);
        assert!((config.louvain_resolution - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.louvain_max_iterations, 100);
        assert!((config.god_function_percentile - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analytics_config_serde_roundtrip() {
        let config = AnalyticsConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AnalyticsConfig = serde_json::from_str(&json).unwrap();
        assert!((deserialized.pagerank_damping - config.pagerank_damping).abs() < f64::EPSILON);
        assert_eq!(
            deserialized.pagerank_max_iterations,
            config.pagerank_max_iterations
        );
    }

    // --- CodeNode ---

    #[test]
    fn test_code_node_serde_roundtrip() {
        let node = CodeNode {
            id: "src/main.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/main.rs".to_string()),
            name: "main.rs".to_string(),
            project_id: Some("abc-123".to_string()),
        };
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: CodeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "src/main.rs");
        assert_eq!(deserialized.node_type, CodeNodeType::File);
        assert_eq!(deserialized.project_id, Some("abc-123".to_string()));
    }

    #[test]
    fn test_code_node_type_display() {
        assert_eq!(CodeNodeType::File.to_string(), "File");
        assert_eq!(CodeNodeType::Function.to_string(), "Function");
        assert_eq!(CodeNodeType::Struct.to_string(), "Struct");
        assert_eq!(CodeNodeType::Trait.to_string(), "Trait");
        assert_eq!(CodeNodeType::Enum.to_string(), "Enum");
    }

    // --- CodeEdge ---

    #[test]
    fn test_code_edge_default() {
        let edge = CodeEdge::default();
        assert_eq!(edge.edge_type, CodeEdgeType::Calls);
        assert!((edge.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_code_edge_type_display() {
        assert_eq!(CodeEdgeType::Imports.to_string(), "IMPORTS");
        assert_eq!(CodeEdgeType::Calls.to_string(), "CALLS");
        assert_eq!(CodeEdgeType::Defines.to_string(), "DEFINES");
        assert_eq!(CodeEdgeType::ImplementsTrait.to_string(), "IMPLEMENTS_TRAIT");
        assert_eq!(CodeEdgeType::ImplementsFor.to_string(), "IMPLEMENTS_FOR");
    }

    // --- NodeMetrics ---

    #[test]
    fn test_node_metrics_default() {
        let m = NodeMetrics::default();
        assert!((m.pagerank - 0.0).abs() < f64::EPSILON);
        assert!((m.betweenness - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.community_id, 0);
        assert!((m.clustering_coefficient - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.component_id, 0);
        assert_eq!(m.in_degree, 0);
        assert_eq!(m.out_degree, 0);
    }

    #[test]
    fn test_node_metrics_serde_roundtrip() {
        let metrics = NodeMetrics {
            pagerank: 0.05,
            betweenness: 0.12,
            community_id: 3,
            clustering_coefficient: 0.33,
            component_id: 1,
            in_degree: 5,
            out_degree: 3,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: NodeMetrics = serde_json::from_str(&json).unwrap();
        assert!((deserialized.pagerank - 0.05).abs() < f64::EPSILON);
        assert_eq!(deserialized.community_id, 3);
        assert_eq!(deserialized.in_degree, 5);
    }

    // --- CodeGraph ---

    #[test]
    fn test_code_graph_add_node() {
        let mut g = CodeGraph::new();
        let node = CodeNode {
            id: "src/main.rs".to_string(),
            node_type: CodeNodeType::File,
            path: Some("src/main.rs".to_string()),
            name: "main.rs".to_string(),
            project_id: None,
        };
        let idx = g.add_node(node);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
        assert!(g.get_node("src/main.rs").is_some());
        assert_eq!(g.get_index("src/main.rs"), Some(idx));
    }

    #[test]
    fn test_code_graph_add_node_idempotent() {
        let mut g = CodeGraph::new();
        let node1 = CodeNode {
            id: "src/main.rs".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "main.rs".to_string(),
            project_id: None,
        };
        let node2 = CodeNode {
            id: "src/main.rs".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "main.rs".to_string(),
            project_id: None,
        };
        let idx1 = g.add_node(node1);
        let idx2 = g.add_node(node2);
        assert_eq!(idx1, idx2);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_code_graph_add_edge() {
        let mut g = CodeGraph::new();
        g.add_node(CodeNode {
            id: "a.rs".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "a.rs".to_string(),
            project_id: None,
        });
        g.add_node(CodeNode {
            id: "b.rs".to_string(),
            node_type: CodeNodeType::File,
            path: None,
            name: "b.rs".to_string(),
            project_id: None,
        });

        let edge_idx = g.add_edge(
            "a.rs",
            "b.rs",
            CodeEdge {
                edge_type: CodeEdgeType::Imports,
                weight: 1.0,
            },
        );
        assert!(edge_idx.is_some());
        assert_eq!(g.edge_count(), 1);

        // Non-existent source → None
        assert!(g.add_edge("missing.rs", "b.rs", CodeEdge::default()).is_none());
    }

    #[test]
    fn test_code_graph_get_nonexistent() {
        let g = CodeGraph::new();
        assert!(g.get_node("doesnt_exist").is_none());
        assert!(g.get_index("doesnt_exist").is_none());
    }

    // --- GraphAnalytics ---

    #[test]
    fn test_graph_analytics_serde_roundtrip() {
        let analytics = GraphAnalytics {
            metrics: HashMap::new(),
            communities: vec![],
            components: vec![],
            health: CodeHealthReport::default(),
            modularity: 0.45,
            node_count: 100,
            edge_count: 250,
            computation_ms: 42,
        };
        let json = serde_json::to_string(&analytics).unwrap();
        let deserialized: GraphAnalytics = serde_json::from_str(&json).unwrap();
        assert!((deserialized.modularity - 0.45).abs() < f64::EPSILON);
        assert_eq!(deserialized.node_count, 100);
        assert_eq!(deserialized.edge_count, 250);
    }

    // --- CodeHealthReport ---

    #[test]
    fn test_code_health_report_default() {
        let report = CodeHealthReport::default();
        assert!(report.god_functions.is_empty());
        assert!(report.circular_dependencies.is_empty());
        assert!(report.orphan_files.is_empty());
        assert!((report.avg_coupling - 0.0).abs() < f64::EPSILON);
        assert!((report.max_coupling - 0.0).abs() < f64::EPSILON);
    }
}
