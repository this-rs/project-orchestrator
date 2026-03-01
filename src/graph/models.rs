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
    /// Temporal coupling — files changed together in commits (Knowledge Fabric P1)
    CoChanged,
    /// Commit touches — file was modified by a commit (Knowledge Fabric P1)
    Touches,
    /// Chat discussed — entity was discussed in a chat session (Knowledge Fabric P4)
    Discussed,
    /// Decision affects — decision affects this entity (Knowledge Fabric P3)
    Affects,
    /// Neural synapse — weighted connection between notes (Knowledge Fabric P3)
    Synapse,
    /// Class inheritance — child class extends parent class (Heritage Plan 5)
    Extends,
    /// Interface implementation — class implements interface/protocol (Heritage Plan 5)
    Implements,
}

impl std::fmt::Display for CodeEdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Imports => write!(f, "IMPORTS"),
            Self::Calls => write!(f, "CALLS"),
            Self::Defines => write!(f, "DEFINES"),
            Self::ImplementsTrait => write!(f, "IMPLEMENTS_TRAIT"),
            Self::ImplementsFor => write!(f, "IMPLEMENTS_FOR"),
            Self::CoChanged => write!(f, "CO_CHANGED"),
            Self::Touches => write!(f, "TOUCHES"),
            Self::Discussed => write!(f, "DISCUSSED"),
            Self::Affects => write!(f, "AFFECTS"),
            Self::Synapse => write!(f, "SYNAPSE"),
            Self::Extends => write!(f, "EXTENDS"),
            Self::Implements => write!(f, "IMPLEMENTS"),
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
    /// Cohesion ratio: internal_edges / (internal_edges + external_edges).
    /// 1.0 = perfectly isolated community, 0.0 = no internal edges.
    #[serde(default)]
    pub cohesion: f64,
    /// How the label was generated: "heuristic" (path prefix) or "llm" (LLM-enriched).
    #[serde(default)]
    pub enriched_by: Option<String>,
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
    /// Name of the analysis profile used (None = default/unweighted)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_name: Option<String>,
}

// ============================================================================
// Writer types — Analytics updates for Neo4j persistence
// ============================================================================

/// Batch update payload for File nodes' analytics properties.
///
/// Used by the `AnalyticsWriter` to persist computed scores back to Neo4j
/// in a single UNWIND query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAnalyticsUpdate {
    /// File path (matches the `path` property on File nodes)
    pub path: String,
    /// PageRank score
    pub pagerank: f64,
    /// Betweenness centrality score
    pub betweenness: f64,
    /// Louvain community ID
    pub community_id: u32,
    /// Human-readable community label
    pub community_label: String,
    /// Local clustering coefficient
    pub clustering_coefficient: f64,
    /// Weakly connected component ID
    pub component_id: u32,
}

/// Batch update payload for Function nodes' analytics properties.
///
/// Used by the `AnalyticsWriter` to persist computed scores back to Neo4j
/// in a single UNWIND query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionAnalyticsUpdate {
    /// Function ID (matches the `id` property on Function nodes, format: "file_path:name:line_start")
    pub id: String,
    /// PageRank score
    pub pagerank: f64,
    /// Betweenness centrality score
    pub betweenness: f64,
    /// Louvain community ID
    pub community_id: u32,
    /// Local clustering coefficient
    pub clustering_coefficient: f64,
    /// Weakly connected component ID
    pub component_id: u32,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for large-graph adaptive mode in Louvain community detection.
///
/// When enabled AND the graph exceeds `max_nodes_full` nodes, the algorithm
/// applies optimizations:
/// - **Edge filtering**: edges with weight < `min_confidence` are excluded
/// - **Degree-1 pre-assignment**: leaf nodes are assigned to their sole neighbor's
///   community without participating in Louvain iterations
/// - **Timeout**: the iterative loop aborts after `max_duration_ms` and returns
///   a partial (but valid) result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeGraphConfig {
    /// Node count threshold to activate large-graph mode (default: 10_000)
    pub max_nodes_full: usize,
    /// Minimum edge weight to keep in the adjacency list (default: 0.5)
    pub min_confidence: f64,
    /// Pre-assign degree-1 nodes to their neighbor's community (default: true)
    pub skip_degree_one: bool,
    /// Maximum wall-clock time for the Louvain loop in ms (default: 60_000)
    pub max_duration_ms: u64,
}

impl Default for LargeGraphConfig {
    fn default() -> Self {
        Self {
            max_nodes_full: 10_000,
            min_confidence: 0.5,
            skip_degree_one: true,
            max_duration_ms: 60_000,
        }
    }
}

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
    /// Optional large-graph mode for Louvain (None = classic mode, Some = adaptive)
    #[serde(default)]
    pub large_graph: Option<LargeGraphConfig>,
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
            large_graph: None,
        }
    }
}

/// Batch update payload for File nodes' **fabric** analytics properties.
///
/// Used by the `AnalyticsWriter` to persist fabric-specific scores
/// (from the multi-layer graph) separately from the code-only scores.
/// Written to `fabric_pagerank`, `fabric_betweenness`, `fabric_community_id`, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricFileAnalyticsUpdate {
    /// File path (matches the `path` property on File nodes)
    pub path: String,
    /// PageRank from fabric graph (multi-layer)
    pub fabric_pagerank: f64,
    /// Betweenness centrality from fabric graph
    pub fabric_betweenness: f64,
    /// Louvain community ID from fabric graph
    pub fabric_community_id: u32,
    /// Human-readable community label from fabric graph
    pub fabric_community_label: String,
    /// Local clustering coefficient from fabric graph
    pub fabric_clustering_coefficient: f64,
}

/// Batch update payload for structural DNA persistence.
///
/// Each entry maps a node path to its DNA vector (K-dimensional distance vector
/// to anchor nodes, normalized [0,1]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralDnaUpdate {
    /// File path (matches the `path` property on File nodes)
    pub path: String,
    /// Structural DNA vector — distances to K anchor nodes, normalized [0,1]
    pub dna: Vec<f64>,
}

/// A predicted missing link between two nodes in the code graph.
///
/// Each prediction includes a plausibility score (0-1) and the individual
/// signals that contributed to the score (Jaccard, co-change, proximity,
/// Adamic-Adar, DNA similarity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkPrediction {
    /// Source node ID (file path or function name)
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Combined plausibility score (0.0 - 1.0)
    pub plausibility: f64,
    /// Individual signal scores: (signal_name, score)
    pub signals: Vec<(String, f64)>,
    /// Suggested relationship type (IMPORTS, CALLS, etc.)
    pub suggested_relation: String,
}

/// Mode of stress test simulation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StressTestMode {
    /// Remove a single node and measure impact
    NodeRemoval,
    /// Remove a single edge and measure impact
    EdgeRemoval,
    /// Cascade removal: iteratively remove orphaned dependents
    Cascade,
}

/// Result of a stress test simulation.
///
/// Measures the impact of removing a node or edge from the graph:
/// blast radius, orphaned nodes, resilience score, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Target node/edge that was removed
    pub target: String,
    /// Type of stress test performed
    pub mode: StressTestMode,
    /// Resilience score (0.0 = catastrophic, 1.0 = no impact)
    pub resilience_score: f64,
    /// Number of nodes that became orphaned (unreachable)
    pub orphaned_nodes: usize,
    /// Total blast radius (nodes affected by removal)
    pub blast_radius: usize,
    /// Depth of cascade (for cascade mode)
    pub cascade_depth: usize,
    /// Number of connected components before removal
    pub components_before: usize,
    /// Number of connected components after removal
    pub components_after: usize,
    /// List of critical edges (bridges) found near the target
    pub critical_edges: Vec<(String, String)>,
}

/// A group of files sharing the same WL subgraph hash (isomorphic neighborhoods).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsomorphicGroup {
    /// The shared WL hash value
    pub wl_hash: u64,
    /// File paths in this group
    pub members: Vec<String>,
    /// Group size
    pub size: usize,
}

/// A structural template derived from isomorphic groups.
/// Files with the same WL hash share the same structural neighborhood topology,
/// making them candidates for reusable patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralTemplate {
    /// The shared WL hash fingerprint
    pub wl_hash: u64,
    /// Number of files following this pattern
    pub occurrences: usize,
    /// Example file paths (up to 5)
    pub exemplars: Vec<String>,
    /// Human-readable description of the pattern (e.g. "Handler → Service → Repository chain")
    pub description: String,
    /// Common structural DNA prefix shared by exemplars (if any)
    pub common_dna_prefix: Option<String>,
}

/// Pre-computed structural profile for a file node (Context Card).
///
/// Aggregates all analytics metrics into a single, self-contained summary
/// that can be stored as Neo4j node properties (cc_* prefix) and served
/// instantly without recomputation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCard {
    /// File path (matches the File node's path property)
    pub path: String,
    /// PageRank centrality score
    #[serde(default)]
    pub cc_pagerank: f64,
    /// Betweenness centrality score
    #[serde(default)]
    pub cc_betweenness: f64,
    /// Clustering coefficient
    #[serde(default)]
    pub cc_clustering: f64,
    /// Community ID from Louvain clustering
    #[serde(default)]
    pub cc_community_id: u32,
    /// Community label (human-readable)
    #[serde(default)]
    pub cc_community_label: String,
    /// Number of files this file imports
    #[serde(default)]
    pub cc_imports_out: usize,
    /// Number of files that import this file
    #[serde(default)]
    pub cc_imports_in: usize,
    /// Number of outgoing function calls from this file
    #[serde(default)]
    pub cc_calls_out: usize,
    /// Number of incoming function calls to this file
    #[serde(default)]
    pub cc_calls_in: usize,
    /// Structural DNA vector (Plan 2)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cc_structural_dna: Vec<f64>,
    /// Weisfeiler-Leman hash (Plan 7)
    #[serde(default)]
    pub cc_wl_hash: u64,
    /// Top-5 co-changers (file paths that frequently change together)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cc_co_changers_top5: Vec<String>,
    /// Card version (incremented on recompute, -1 = invalidated)
    #[serde(default)]
    pub cc_version: i32,
    /// Timestamp of last computation (ISO 8601)
    #[serde(default)]
    pub cc_computed_at: String,
}

impl Default for ContextCard {
    fn default() -> Self {
        Self {
            path: String::new(),
            cc_pagerank: 0.0,
            cc_betweenness: 0.0,
            cc_clustering: 0.0,
            cc_community_id: 0,
            cc_community_label: String::new(),
            cc_imports_out: 0,
            cc_imports_in: 0,
            cc_calls_out: 0,
            cc_calls_in: 0,
            cc_structural_dna: Vec::new(),
            cc_wl_hash: 0,
            cc_co_changers_top5: Vec::new(),
            cc_version: 0,
            cc_computed_at: String::new(),
        }
    }
}

/// Result of K-means clustering on structural DNA vectors.
///
/// Each cluster represents a group of structurally similar files that play
/// a similar architectural role (e.g., handlers, models, utilities).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DnaCluster {
    /// Cluster index (0-based)
    pub id: usize,
    /// Centroid vector (average DNA of all members)
    pub centroid: Vec<f64>,
    /// File paths in this cluster
    pub members: Vec<String>,
    /// Auto-inferred label based on dominant file patterns
    pub label: String,
    /// Intra-cluster cohesion: average cosine similarity between members
    pub cohesion: f64,
}

// ============================================================================
// Fabric layer weights — configurable per-relation weights for multi-layer graph
// ============================================================================

/// Configurable weights for each relationship type in the fabric graph.
///
/// The fabric graph overlays multiple relationship types into a single
/// petgraph, each with a distinct weight reflecting its coupling strength.
/// Higher weight = stronger coupling signal for PageRank/Louvain/Betweenness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricWeights {
    /// Weight for IMPORTS edges (code dependency, default: 0.8)
    pub imports: f64,
    /// Weight for CALLS edges (function calls, default: 0.9)
    pub calls: f64,
    /// Weight for CO_CHANGED edges (temporal coupling, default: 0.4)
    pub co_changed: f64,
    /// Weight for AFFECTS edges (decision impact, default: 0.7)
    pub affects: f64,
    /// Weight for TOUCHES edges (commit coupling, default: 0.5)
    pub touches: f64,
    /// Weight for DISCUSSED edges (chat mentions, default: 0.3)
    pub discussed: f64,
    /// Weight for SYNAPSE edges (neural connections, default: 0.6)
    pub synapse: f64,
    /// Weight for DEFINES/CONTAINS edges (structural, default: 1.0)
    pub defines: f64,
    /// Weight for EXTENDS edges (class inheritance, default: 0.95)
    pub extends: f64,
    /// Weight for IMPLEMENTS edges (interface implementation, default: 0.85)
    pub implements: f64,
    /// Minimum CO_CHANGED count to include an edge (filters noise)
    pub co_changed_min_count: i64,
}

impl Default for FabricWeights {
    fn default() -> Self {
        Self {
            imports: 0.8,
            calls: 0.9,
            co_changed: 0.4,
            affects: 0.7,
            touches: 0.5,
            discussed: 0.3,
            synapse: 0.6,
            defines: 1.0,
            extends: 0.95,
            implements: 0.85,
            co_changed_min_count: 2,
        }
    }
}

// ============================================================================
// GraIL — Extended pipeline types
// ============================================================================

/// Configuration for the extended `compute_all` pipeline (GraIL algorithms).
///
/// When `grail` is `Some`, the pipeline runs the additional GraIL stages
/// after the base analytics (PageRank, Betweenness, Louvain, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrailConfig {
    /// Optional analysis profile to weight edges before computation.
    /// When set, `apply_profile_weights()` runs BEFORE PageRank.
    #[serde(default)]
    pub profile: Option<AnalysisProfile>,
    /// Number of top-PageRank nodes to stress-test (default: 20)
    #[serde(default = "default_stress_top_n")]
    pub stress_top_n: usize,
    /// Number of missing link predictions to keep (default: 50)
    #[serde(default = "default_missing_links_top_n")]
    pub missing_links_top_n: usize,
    /// Minimum plausibility score for predicted links (default: 0.3)
    #[serde(default = "default_min_plausibility")]
    pub min_plausibility: f64,
    /// Number of K anchors for structural DNA (default: 10)
    #[serde(default = "default_dna_k")]
    pub dna_k: usize,
    /// WL hash neighborhood radius (default: 2)
    #[serde(default = "default_wl_radius")]
    pub wl_radius: usize,
    /// WL hash iterations (default: 3)
    #[serde(default = "default_wl_iterations")]
    pub wl_iterations: usize,
}

fn default_stress_top_n() -> usize {
    20
}
fn default_missing_links_top_n() -> usize {
    50
}
fn default_min_plausibility() -> f64 {
    0.3
}
fn default_dna_k() -> usize {
    10
}
fn default_wl_radius() -> usize {
    2
}
fn default_wl_iterations() -> usize {
    3
}

impl Default for GrailConfig {
    fn default() -> Self {
        Self {
            profile: None,
            stress_top_n: default_stress_top_n(),
            missing_links_top_n: default_missing_links_top_n(),
            min_plausibility: default_min_plausibility(),
            dna_k: default_dna_k(),
            wl_radius: default_wl_radius(),
            wl_iterations: default_wl_iterations(),
        }
    }
}

/// An analysis profile that weights edge types for contextual analytics.
///
/// Inspired by R-GCN basis decomposition from GraIL — different contexts
/// (refactoring, security, onboarding) weight relationships differently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisProfile {
    /// Unique identifier (UUID)
    pub id: String,
    /// Project scope — None = global profile, Some = project-specific
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Profile name (e.g., "default", "refactoring", "security", "onboarding")
    pub name: String,
    /// Optional description
    #[serde(default)]
    pub description: Option<String>,
    /// Edge type weights — multiplied against edge weights before analytics.
    /// Missing edge types use weight 1.0 (unchanged).
    pub edge_weights: HashMap<String, f64>,
    /// Fusion weights for multi-signal impact analysis.
    #[serde(default)]
    pub fusion_weights: FusionWeights,
    /// Whether this is a built-in profile (cannot be deleted)
    #[serde(default)]
    pub is_builtin: bool,
}

impl Default for AnalysisProfile {
    fn default() -> Self {
        profile_default()
    }
}

// ============================================================================
// Built-in Analysis Profiles (Plan 6 / GraIL R-GCN inspired)
// ============================================================================

/// Default profile — balanced weights for general-purpose analysis.
pub fn profile_default() -> AnalysisProfile {
    AnalysisProfile {
        id: "00000000-0000-0000-0000-000000000001".to_string(),
        project_id: None,
        name: "default".to_string(),
        description: Some("Balanced weights for general-purpose analysis".to_string()),
        edge_weights: HashMap::from([
            ("IMPORTS".to_string(), 0.5),
            ("CALLS".to_string(), 0.5),
            ("EXTENDS".to_string(), 0.5),
            ("IMPLEMENTS".to_string(), 0.5),
            ("CO_CHANGED".to_string(), 0.5),
            ("TOUCHES".to_string(), 0.3),
            ("AFFECTS".to_string(), 0.3),
            ("SYNAPSE".to_string(), 0.2),
        ]),
        fusion_weights: FusionWeights {
            structural: 0.30,
            co_change: 0.25,
            knowledge: 0.15,
            pagerank: 0.15,
            bridge: 0.15,
        },
        is_builtin: true,
    }
}

/// Refactoring profile — emphasizes imports and co-change for safe restructuring.
pub fn profile_refactoring() -> AnalysisProfile {
    AnalysisProfile {
        id: "00000000-0000-0000-0000-000000000002".to_string(),
        project_id: None,
        name: "refactoring".to_string(),
        description: Some(
            "Optimized for safe code restructuring — imports and co-change are emphasized"
                .to_string(),
        ),
        edge_weights: HashMap::from([
            ("IMPORTS".to_string(), 0.8),
            ("CALLS".to_string(), 0.5),
            ("EXTENDS".to_string(), 0.3),
            ("IMPLEMENTS".to_string(), 0.3),
            ("CO_CHANGED".to_string(), 0.7),
            ("TOUCHES".to_string(), 0.4),
            ("AFFECTS".to_string(), 0.2),
            ("SYNAPSE".to_string(), 0.1),
        ]),
        fusion_weights: FusionWeights {
            structural: 0.40,
            co_change: 0.30,
            knowledge: 0.05,
            pagerank: 0.15,
            bridge: 0.10,
        },
        is_builtin: true,
    }
}

/// Security profile — emphasizes call chains and inheritance for vulnerability tracing.
pub fn profile_security() -> AnalysisProfile {
    AnalysisProfile {
        id: "00000000-0000-0000-0000-000000000003".to_string(),
        project_id: None,
        name: "security".to_string(),
        description: Some(
            "Optimized for security review — call chains and inheritance are prioritized"
                .to_string(),
        ),
        edge_weights: HashMap::from([
            ("IMPORTS".to_string(), 0.4),
            ("CALLS".to_string(), 0.9),
            ("EXTENDS".to_string(), 0.8),
            ("IMPLEMENTS".to_string(), 0.8),
            ("CO_CHANGED".to_string(), 0.2),
            ("TOUCHES".to_string(), 0.1),
            ("AFFECTS".to_string(), 0.6),
            ("SYNAPSE".to_string(), 0.1),
        ]),
        fusion_weights: FusionWeights {
            structural: 0.45,
            co_change: 0.10,
            knowledge: 0.20,
            pagerank: 0.15,
            bridge: 0.10,
        },
        is_builtin: true,
    }
}

/// Onboarding profile — emphasizes definitions and knowledge for new developer orientation.
pub fn profile_onboarding() -> AnalysisProfile {
    AnalysisProfile {
        id: "00000000-0000-0000-0000-000000000004".to_string(),
        project_id: None,
        name: "onboarding".to_string(),
        description: Some(
            "Optimized for onboarding — definitions and knowledge documentation are emphasized"
                .to_string(),
        ),
        edge_weights: HashMap::from([
            ("IMPORTS".to_string(), 0.5),
            ("CALLS".to_string(), 0.3),
            ("EXTENDS".to_string(), 0.4),
            ("IMPLEMENTS".to_string(), 0.4),
            ("CO_CHANGED".to_string(), 0.2),
            ("TOUCHES".to_string(), 0.2),
            ("AFFECTS".to_string(), 0.3),
            ("SYNAPSE".to_string(), 0.7),
        ]),
        fusion_weights: FusionWeights {
            structural: 0.20,
            co_change: 0.10,
            knowledge: 0.40,
            pagerank: 0.15,
            bridge: 0.15,
        },
        is_builtin: true,
    }
}

/// Architect profile — emphasizes structure, hierarchies, and bridge nodes for system design.
pub fn profile_architect() -> AnalysisProfile {
    AnalysisProfile {
        id: "00000000-0000-0000-0000-000000000005".to_string(),
        project_id: None,
        name: "architect".to_string(),
        description: Some(
            "Optimized for system design — type hierarchies, module dependencies, \
             and bridge nodes are prioritized"
                .to_string(),
        ),
        edge_weights: HashMap::from([
            ("IMPORTS".to_string(), 0.7),
            ("CALLS".to_string(), 0.5),
            ("EXTENDS".to_string(), 0.9),
            ("IMPLEMENTS".to_string(), 0.9),
            ("CO_CHANGED".to_string(), 0.3),
            ("TOUCHES".to_string(), 0.2),
            ("AFFECTS".to_string(), 0.7),
            ("SYNAPSE".to_string(), 0.4),
        ]),
        fusion_weights: FusionWeights {
            structural: 0.35,
            co_change: 0.10,
            knowledge: 0.15,
            pagerank: 0.15,
            bridge: 0.25,
        },
        is_builtin: true,
    }
}

/// Returns all built-in analysis profiles.
pub fn builtin_profiles() -> Vec<AnalysisProfile> {
    vec![
        profile_default(),
        profile_refactoring(),
        profile_security(),
        profile_onboarding(),
        profile_architect(),
    ]
}

/// Weights for multi-signal impact fusion (Plan 4).
///
/// Each weight controls the contribution of one signal in the combined
/// impact score. They should sum to ~1.0 for normalized results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionWeights {
    /// Structural impact (imports/calls graph traversal)
    pub structural: f64,
    /// Co-change signal (temporal coupling from TOUCHES/CO_CHANGED)
    pub co_change: f64,
    /// Knowledge density signal (linked notes + decisions)
    pub knowledge: f64,
    /// PageRank importance signal
    pub pagerank: f64,
    /// Bridge proximity signal (betweenness / bridge detection)
    pub bridge: f64,
}

impl FusionWeights {
    /// Returns the sum of all weights.
    pub fn sum(&self) -> f64 {
        self.structural + self.co_change + self.knowledge + self.pagerank + self.bridge
    }

    /// Validates that all weights sum to approximately 1.0 (±0.01 tolerance)
    /// and that no individual weight is negative.
    pub fn validate(&self) -> Result<(), String> {
        // Check for negative weights
        if self.structural < 0.0 {
            return Err(format!(
                "structural weight must be >= 0, got {}",
                self.structural
            ));
        }
        if self.co_change < 0.0 {
            return Err(format!(
                "co_change weight must be >= 0, got {}",
                self.co_change
            ));
        }
        if self.knowledge < 0.0 {
            return Err(format!(
                "knowledge weight must be >= 0, got {}",
                self.knowledge
            ));
        }
        if self.pagerank < 0.0 {
            return Err(format!(
                "pagerank weight must be >= 0, got {}",
                self.pagerank
            ));
        }
        if self.bridge < 0.0 {
            return Err(format!("bridge weight must be >= 0, got {}", self.bridge));
        }

        // Check sum ≈ 1.0
        let sum = self.sum();
        if (sum - 1.0).abs() > 0.01 {
            Err(format!(
                "FusionWeights must sum to ~1.0 (±0.01), got {:.4} \
                 (structural={}, co_change={}, knowledge={}, pagerank={}, bridge={})",
                sum, self.structural, self.co_change, self.knowledge, self.pagerank, self.bridge
            ))
        } else {
            Ok(())
        }
    }
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            structural: 0.30,
            co_change: 0.25,
            knowledge: 0.15,
            pagerank: 0.15,
            bridge: 0.15,
        }
    }
}

/// Timing information for a single pipeline step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepTiming {
    /// Step name (e.g., "pagerank", "structural_dna")
    pub name: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether the step succeeded
    pub success: bool,
    /// Error message if the step failed (None if success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Statistics from the extended compute_all pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GrailStats {
    /// Number of nodes with structural DNA computed
    pub dna_computed: usize,
    /// Number of nodes with WL hash computed
    pub wl_computed: usize,
    /// Number of context cards generated
    pub cards_computed: usize,
    /// Number of missing links predicted
    pub links_predicted: usize,
    /// Number of topology violations found
    pub violations_found: usize,
    /// Number of nodes stress-tested
    pub stress_tested: usize,
}

/// Result of the extended `compute_all` pipeline with GraIL algorithms.
///
/// Extends `GraphAnalytics` with per-step timing, error collection,
/// and GraIL-specific data (DNA vectors, WL hashes, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeAllResult {
    /// Base analytics (PageRank, Betweenness, Louvain, Health, etc.)
    pub analytics: GraphAnalytics,
    /// Per-step timing information
    pub timings: Vec<StepTiming>,
    /// GraIL-specific statistics (only populated if grail config was provided)
    pub grail_stats: GrailStats,
    /// Structural DNA vectors per node ID (Plan 2)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub structural_dna: HashMap<String, Vec<f64>>,
    /// WL hashes per node ID (Plan 7)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub wl_hashes: HashMap<String, u64>,
    /// Top predicted missing links (Plan 9)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub predicted_links: Vec<LinkPrediction>,
    /// Pre-computed context cards for file nodes (Plan 8)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context_cards: Vec<ContextCard>,
    /// Whether incremental mode was used (vs full recompute)
    #[serde(default)]
    pub mode: ComputeMode,
    /// Total duration in milliseconds (wall clock)
    pub total_ms: u64,
}

/// Computation mode: full recompute or incremental (stale nodes only).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeMode {
    /// Full recompute of all nodes
    #[default]
    Full,
    /// Incremental recompute of stale nodes only
    Incremental,
}

// ============================================================================
// Margin Ranking — Universal relative ranking (Plan 10 / GraIL)
// ============================================================================

/// Confidence level of a ranking position, based on the margin to the
/// next/previous item. Inspired by GraIL's MarginRankingLoss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankConfidence {
    /// Margin > 0.2 — strong separation from neighbors
    High,
    /// 0.05 < margin ≤ 0.2 — moderate separation
    Medium,
    /// 0.0 < margin ≤ 0.05 — nearly tied
    Low,
    /// Margin = 0.0 — exact tie
    Tied,
}

impl RankConfidence {
    /// Determine confidence from a margin value.
    pub fn from_margin(margin: f64) -> Self {
        if margin <= 0.0 {
            Self::Tied
        } else if margin <= 0.05 {
            Self::Low
        } else if margin <= 0.2 {
            Self::Medium
        } else {
            Self::High
        }
    }
}

/// A single ranked result with score, position, and margin information.
///
/// Generic over T so it can wrap any result type (ImpactedFile, Reference, etc.).
/// Uses `#[serde(flatten)]` on `item` to maintain JSON retro-compatibility:
/// existing fields stay at top-level, new ranking fields are added alongside.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult<T: Serialize> {
    /// The ranked item
    #[serde(flatten)]
    pub item: T,
    /// 1-based rank position
    pub rank: usize,
    /// Absolute score (higher = more relevant)
    pub score: f64,
    /// Score difference to the next-ranked item (None for last)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin_to_next: Option<f64>,
    /// Score difference to the previous-ranked item (None for first)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin_to_prev: Option<f64>,
    /// Confidence in this ranking position
    pub confidence: RankConfidence,
    /// Breakdown of contributing signals (signal_name, signal_value)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub signals: Vec<(String, f64)>,
}

/// A natural cluster detected by gap analysis in a ranked list.
///
/// When scores have significant gaps, items naturally group into clusters
/// (e.g., "critical", "high", "moderate", "low", "peripheral").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankCluster {
    /// Start rank (1-based, inclusive)
    pub start_rank: usize,
    /// End rank (1-based, inclusive)
    pub end_rank: usize,
    /// Average score within this cluster
    pub avg_score: f64,
    /// Human-readable label (e.g., "critical", "high", "moderate")
    pub label: String,
}

/// A ranked list with metadata: score range, total candidates, and natural clusters.
///
/// Wraps `Vec<RankedResult<T>>` with ranking-level metadata. Inspired by
/// GraIL's MRR/Hits@K metrics and MarginRankingLoss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedList<T: Serialize> {
    /// Ranked items, sorted by score descending
    pub items: Vec<RankedResult<T>>,
    /// Total number of candidates before any filtering/truncation
    pub total_candidates: usize,
    /// Score range: (min_score, max_score)
    pub score_range: (f64, f64),
    /// Natural clusters detected by gap analysis
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub natural_clusters: Vec<RankCluster>,
}

// ============================================================================
// Topological Firewall — topology rules & violations (Plan 3 GraIL)
// ============================================================================

/// Type of topological rule, inspired by GraIL negative triples.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyRuleType {
    /// Source files must not import target files (layer separation)
    MustNotImport,
    /// Source functions must not call target functions
    MustNotCall,
    /// Shortest path between source and target must be >= threshold
    MaxDistance,
    /// No file matching the pattern may have more than threshold imports
    MaxFanOut,
    /// No circular imports within the matching files
    NoCircular,
}

impl std::fmt::Display for TopologyRuleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MustNotImport => write!(f, "must_not_import"),
            Self::MustNotCall => write!(f, "must_not_call"),
            Self::MaxDistance => write!(f, "max_distance"),
            Self::MaxFanOut => write!(f, "max_fan_out"),
            Self::NoCircular => write!(f, "no_circular"),
        }
    }
}

impl TopologyRuleType {
    /// Parse from string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('-', "_").as_str() {
            "must_not_import" | "mustnotimport" => Some(Self::MustNotImport),
            "must_not_call" | "mustnotcall" => Some(Self::MustNotCall),
            "max_distance" | "maxdistance" => Some(Self::MaxDistance),
            "max_fan_out" | "maxfanout" => Some(Self::MaxFanOut),
            "no_circular" | "nocircular" => Some(Self::NoCircular),
            _ => None,
        }
    }
}

/// Severity level of a topology rule violation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologySeverity {
    /// Hard constraint — must be fixed
    Error,
    /// Soft constraint — should be reviewed
    Warning,
}

impl std::fmt::Display for TopologySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
        }
    }
}

impl TopologySeverity {
    /// Parse from string (case-insensitive).
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "error" => Some(Self::Error),
            "warning" => Some(Self::Warning),
            _ => None,
        }
    }
}

/// A topological constraint rule stored in the knowledge graph.
///
/// Inspired by GraIL's negative triples — encodes relationships that
/// SHOULD NOT exist in the graph (architectural boundaries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyRule {
    /// Unique identifier (UUID)
    pub id: String,
    /// Project scope
    pub project_id: String,
    /// Type of rule
    pub rule_type: TopologyRuleType,
    /// Glob pattern for source files/functions (e.g. "src/neo4j/**")
    pub source_pattern: String,
    /// Glob pattern for target files/functions (e.g. "src/api/**")
    /// Not used for MaxFanOut and NoCircular
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_pattern: Option<String>,
    /// Numeric threshold (for MaxDistance, MaxFanOut)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<u32>,
    /// Severity level
    pub severity: TopologySeverity,
    /// Human-readable description of the rule
    pub description: String,
}

/// A violation detected by the topological firewall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyViolation {
    /// The rule that was violated
    pub rule_id: String,
    /// Human-readable description of the rule
    pub rule_description: String,
    /// Rule type
    pub rule_type: TopologyRuleType,
    /// The file/function that violates the rule
    pub violator_path: String,
    /// The target involved (if applicable)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_path: Option<String>,
    /// Severity level
    pub severity: TopologySeverity,
    /// Additional details about the violation
    pub details: String,
    /// Violation score: structural_plausibility × forbidden_weight
    /// High score = dangerous (structurally tempting but architecturally forbidden)
    #[serde(default)]
    pub violation_score: f64,
}

/// Result of a topology check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyCheckResult {
    /// Project ID checked
    pub project_id: String,
    /// Number of rules checked
    pub rules_checked: usize,
    /// All violations found, sorted by violation_score descending
    pub violations: Vec<TopologyViolation>,
    /// Count of error-level violations
    pub error_count: usize,
    /// Count of warning-level violations
    pub warning_count: usize,
    /// Execution time in milliseconds
    pub timing_ms: u64,
}

/// Convert a glob pattern to a regex pattern suitable for Neo4j.
///
/// Supports:
/// - `**` → `.*` (match any depth)
/// - `*` → `[^/]*` (match within one directory level)
/// - `.` → `\\.` (literal dot)
/// - `?` → `.` (single character)
///
/// # Examples
/// ```
/// use project_orchestrator::graph::models::glob_to_regex;
/// assert_eq!(glob_to_regex("src/neo4j/**"), "^src/neo4j/.*$");
/// assert_eq!(glob_to_regex("src/*.rs"), "^src/[^/]*\\.rs$");
/// assert_eq!(glob_to_regex("src/api/?.rs"), "^src/api/.\\.rs$");
/// ```
pub fn glob_to_regex(pattern: &str) -> String {
    let mut regex = String::with_capacity(pattern.len() * 2 + 2);
    regex.push('^');

    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    // ** → match any depth
                    regex.push_str(".*");
                    i += 2;
                    // Skip trailing slash after **
                    if i < chars.len() && chars[i] == '/' {
                        // .* already matches the slash
                        i += 1;
                    }
                } else {
                    // * → match within one directory level
                    regex.push_str("[^/]*");
                    i += 1;
                }
            }
            '?' => {
                regex.push('.');
                i += 1;
            }
            '.' => {
                regex.push_str("\\.");
                i += 1;
            }
            '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                regex.push('\\');
                regex.push(chars[i]);
                i += 1;
            }
            c => {
                regex.push(c);
                i += 1;
            }
        }
    }

    regex.push('$');
    regex
}

// ============================================================================
// Bridge Subgraph — raw Neo4j results (Plan 1 GraIL)
// ============================================================================

/// Raw node from the bridge subgraph extraction (pre-labeling).
/// The double-radius labeling (distance_to_source, distance_to_target) is computed
/// in Rust after extraction (see Task 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRawNode {
    /// File path (unique identifier within the bridge)
    pub path: String,
    /// Node type: "File" or "Function"
    pub node_type: String,
}

/// Raw edge from the bridge subgraph extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRawEdge {
    /// Source node path
    pub from_path: String,
    /// Target node path
    pub to_path: String,
    /// Relationship type (IMPORTS, CALLS, etc.)
    pub rel_type: String,
}

/// A node in the enriched bridge subgraph (post double-radius labeling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeNode {
    /// File or function path
    pub path: String,
    /// Node type: "File" or "Function"
    pub node_type: String,
    /// BFS distance from source
    pub distance_to_source: u32,
    /// BFS distance from target
    pub distance_to_target: u32,
}

/// An edge in the enriched bridge subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeEdge {
    /// Source node path
    pub from_path: String,
    /// Target node path
    pub to_path: String,
    /// Relationship type (IMPORTS, CALLS, etc.)
    pub rel_type: String,
}

/// Complete bridge subgraph between two nodes, enriched with GraIL-style
/// double-radius labeling, density, and bottleneck detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeSubgraph {
    /// Source node path
    pub source: String,
    /// Target node path
    pub target: String,
    /// All nodes in the bridge subgraph (with double-radius labels)
    pub nodes: Vec<BridgeNode>,
    /// All edges in the bridge subgraph
    pub edges: Vec<BridgeEdge>,
    /// Graph density: edges / (nodes * (nodes - 1))
    pub density: f64,
    /// Top bottleneck nodes (highest betweenness centrality in the subgraph)
    pub bottleneck_nodes: Vec<String>,
}

// ============================================================================
// Multi-signal Impact Fusion (Plan 4)
// ============================================================================

/// Individual file score with 5 signal components for multi-signal impact fusion.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MultiSignalScore {
    /// File path
    pub path: String,
    /// Structural impact score (from IMPORTS + CALLS traversal, 1.0/distance)
    pub structural_score: f64,
    /// Co-change score (temporal coupling frequency, normalized)
    pub co_change_score: f64,
    /// Knowledge density score (notes + decisions linked to this file, [0,1])
    pub knowledge_score: f64,
    /// PageRank importance score (from GDS projection)
    pub pagerank_score: f64,
    /// Bridge proximity score (1.0/shortest_path_distance to co-changers)
    pub bridge_score: f64,
    /// Weighted combined score (using FusionWeights from profile)
    pub combined_score: f64,
    /// Which signals contributed (non-zero) to this score
    pub signals: Vec<String>,
}

/// Full multi-signal impact analysis response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSignalImpact {
    /// The target file/function being analyzed
    pub target: String,
    /// Analysis profile used for weighting
    pub profile_used: String,
    /// Fusion weights applied
    pub weights: FusionWeights,
    /// Ranked list of impacted files with per-signal scores
    pub ranked: RankedList<MultiSignalScore>,
    /// Execution timing in milliseconds
    pub timing_ms: u64,
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
        assert_eq!(
            CodeEdgeType::ImplementsTrait.to_string(),
            "IMPLEMENTS_TRAIT"
        );
        assert_eq!(CodeEdgeType::ImplementsFor.to_string(), "IMPLEMENTS_FOR");
        assert_eq!(CodeEdgeType::CoChanged.to_string(), "CO_CHANGED");
        assert_eq!(CodeEdgeType::Touches.to_string(), "TOUCHES");
        assert_eq!(CodeEdgeType::Discussed.to_string(), "DISCUSSED");
        assert_eq!(CodeEdgeType::Affects.to_string(), "AFFECTS");
        assert_eq!(CodeEdgeType::Synapse.to_string(), "SYNAPSE");
        assert_eq!(CodeEdgeType::Extends.to_string(), "EXTENDS");
        assert_eq!(CodeEdgeType::Implements.to_string(), "IMPLEMENTS");
    }

    #[test]
    fn test_fabric_weights_default() {
        let w = FabricWeights::default();
        assert!((w.imports - 0.8).abs() < f64::EPSILON);
        assert!((w.calls - 0.9).abs() < f64::EPSILON);
        assert!((w.co_changed - 0.4).abs() < f64::EPSILON);
        assert!((w.affects - 0.7).abs() < f64::EPSILON);
        assert!((w.touches - 0.5).abs() < f64::EPSILON);
        assert!((w.discussed - 0.3).abs() < f64::EPSILON);
        assert!((w.synapse - 0.6).abs() < f64::EPSILON);
        assert!((w.defines - 1.0).abs() < f64::EPSILON);
        assert!((w.extends - 0.95).abs() < f64::EPSILON);
        assert!((w.implements - 0.85).abs() < f64::EPSILON);
        assert_eq!(w.co_changed_min_count, 2);
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
        assert!(g
            .add_edge("missing.rs", "b.rs", CodeEdge::default())
            .is_none());
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
            profile_name: None,
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

    // --- FusionWeights ---

    #[test]
    fn test_fusion_weights_default_validates() {
        let w = FusionWeights::default();
        assert!(w.validate().is_ok());
        assert!((w.sum() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fusion_weights_validate_bad_sum() {
        let w = FusionWeights {
            structural: 0.5,
            co_change: 0.5,
            knowledge: 0.5,
            pagerank: 0.5,
            bridge: 0.5,
        };
        let err = w.validate().unwrap_err();
        assert!(err.contains("must sum to ~1.0"));
        assert!(err.contains("2.5"));
    }

    #[test]
    fn test_fusion_weights_validate_negative() {
        let w = FusionWeights {
            structural: -0.1,
            co_change: 0.4,
            knowledge: 0.3,
            pagerank: 0.2,
            bridge: 0.2,
        };
        let err = w.validate().unwrap_err();
        assert!(err.contains("structural weight must be >= 0"));
    }

    #[test]
    fn test_fusion_weights_validate_tolerance() {
        // Sum = 1.005 — within ±0.01 tolerance
        let w = FusionWeights {
            structural: 0.305,
            co_change: 0.25,
            knowledge: 0.15,
            pagerank: 0.15,
            bridge: 0.15,
        };
        assert!(w.validate().is_ok());

        // Sum = 1.02 — outside tolerance
        let w2 = FusionWeights {
            structural: 0.32,
            co_change: 0.25,
            knowledge: 0.15,
            pagerank: 0.15,
            bridge: 0.15,
        };
        assert!(w2.validate().is_err());
    }

    // --- AnalysisProfile ---

    #[test]
    fn test_analysis_profile_default_is_default_profile() {
        let p = AnalysisProfile::default();
        assert_eq!(p.name, "default");
        assert!(p.is_builtin);
        assert!(p.project_id.is_none());
        assert!(p.fusion_weights.validate().is_ok());
    }

    #[test]
    fn test_builtin_profiles_count_and_names() {
        let profiles = builtin_profiles();
        assert_eq!(profiles.len(), 5);

        let names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"default"));
        assert!(names.contains(&"refactoring"));
        assert!(names.contains(&"security"));
        assert!(names.contains(&"onboarding"));
        assert!(names.contains(&"architect"));
    }

    #[test]
    fn test_builtin_profiles_all_validate() {
        for profile in builtin_profiles() {
            assert!(
                profile.fusion_weights.validate().is_ok(),
                "Profile '{}' has invalid fusion_weights: {:?}",
                profile.name,
                profile.fusion_weights.validate().err()
            );
            assert!(
                profile.is_builtin,
                "Profile '{}' should be builtin",
                profile.name
            );
            assert!(
                profile.project_id.is_none(),
                "Built-in profile '{}' should be global",
                profile.name
            );
        }
    }

    #[test]
    fn test_builtin_profiles_unique_ids() {
        let profiles = builtin_profiles();
        let ids: Vec<&str> = profiles.iter().map(|p| p.id.as_str()).collect();
        let mut unique_ids = ids.clone();
        unique_ids.sort();
        unique_ids.dedup();
        assert_eq!(ids.len(), unique_ids.len(), "Profile IDs must be unique");
    }

    #[test]
    fn test_analysis_profile_serde_roundtrip() {
        let profile = profile_refactoring();
        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: AnalysisProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "refactoring");
        assert_eq!(deserialized.id, profile.id);
        assert!(deserialized.is_builtin);
        assert_eq!(deserialized.fusion_weights, profile.fusion_weights);
        assert_eq!(deserialized.edge_weights.len(), profile.edge_weights.len());
    }

    #[test]
    fn test_profile_security_weights() {
        let p = profile_security();
        // CALLS should be highest edge weight for security
        assert!(p.edge_weights["CALLS"] > p.edge_weights["IMPORTS"]);
        // EXTENDS/IMPLEMENTS should be high
        assert!(p.edge_weights["EXTENDS"] >= 0.8);
        assert!(p.edge_weights["IMPLEMENTS"] >= 0.8);
        // structural fusion should be highest
        assert!(p.fusion_weights.structural > p.fusion_weights.co_change);
    }

    #[test]
    fn test_profile_onboarding_weights() {
        let p = profile_onboarding();
        // Knowledge fusion should be highest for onboarding
        assert!(p.fusion_weights.knowledge > p.fusion_weights.structural);
        assert!(p.fusion_weights.knowledge > p.fusion_weights.co_change);
        // SYNAPSE should be high (knowledge connections)
        assert!(p.edge_weights["SYNAPSE"] >= 0.7);
    }

    #[test]
    fn test_profile_architect_weights() {
        let p = profile_architect();
        // Hierarchies should be highest edge weights
        assert!(p.edge_weights["EXTENDS"] >= 0.9);
        assert!(p.edge_weights["IMPLEMENTS"] >= 0.9);
        // Bridge fusion should be highest non-structural
        assert!(p.fusion_weights.bridge > p.fusion_weights.co_change);
        assert!(p.fusion_weights.bridge > p.fusion_weights.knowledge);
        // AFFECTS should be high (architectural decisions)
        assert!(p.edge_weights["AFFECTS"] >= 0.7);
    }

    // --- Topology Firewall ---

    #[test]
    fn test_glob_to_regex_double_star() {
        assert_eq!(glob_to_regex("src/neo4j/**"), "^src/neo4j/.*$");
    }

    #[test]
    fn test_glob_to_regex_single_star() {
        assert_eq!(glob_to_regex("src/*.rs"), "^src/[^/]*\\.rs$");
    }

    #[test]
    fn test_glob_to_regex_question_mark() {
        assert_eq!(glob_to_regex("src/api/?.rs"), "^src/api/.\\.rs$");
    }

    #[test]
    fn test_glob_to_regex_nested_double_star() {
        assert_eq!(glob_to_regex("**/test_*.rs"), "^.*test_[^/]*\\.rs$");
    }

    #[test]
    fn test_glob_to_regex_literal_dots() {
        assert_eq!(glob_to_regex("src/main.rs"), "^src/main\\.rs$");
    }

    #[test]
    fn test_glob_to_regex_special_chars() {
        assert_eq!(glob_to_regex("src/foo+bar.rs"), "^src/foo\\+bar\\.rs$");
    }

    #[test]
    fn test_topology_rule_type_display() {
        assert_eq!(
            TopologyRuleType::MustNotImport.to_string(),
            "must_not_import"
        );
        assert_eq!(TopologyRuleType::MustNotCall.to_string(), "must_not_call");
        assert_eq!(TopologyRuleType::MaxDistance.to_string(), "max_distance");
        assert_eq!(TopologyRuleType::MaxFanOut.to_string(), "max_fan_out");
        assert_eq!(TopologyRuleType::NoCircular.to_string(), "no_circular");
    }

    #[test]
    fn test_topology_rule_type_parse() {
        assert_eq!(
            TopologyRuleType::from_str_loose("must_not_import"),
            Some(TopologyRuleType::MustNotImport)
        );
        assert_eq!(
            TopologyRuleType::from_str_loose("MUST_NOT_IMPORT"),
            Some(TopologyRuleType::MustNotImport)
        );
        assert_eq!(
            TopologyRuleType::from_str_loose("no_circular"),
            Some(TopologyRuleType::NoCircular)
        );
        assert_eq!(TopologyRuleType::from_str_loose("invalid"), None);
    }

    #[test]
    fn test_topology_severity_display() {
        assert_eq!(TopologySeverity::Error.to_string(), "error");
        assert_eq!(TopologySeverity::Warning.to_string(), "warning");
    }

    #[test]
    fn test_topology_severity_parse() {
        assert_eq!(
            TopologySeverity::from_str_loose("error"),
            Some(TopologySeverity::Error)
        );
        assert_eq!(
            TopologySeverity::from_str_loose("WARNING"),
            Some(TopologySeverity::Warning)
        );
        assert_eq!(TopologySeverity::from_str_loose("critical"), None);
    }

    #[test]
    fn test_topology_rule_serde_roundtrip() {
        let rule = TopologyRule {
            id: "test-id".to_string(),
            project_id: "proj-1".to_string(),
            rule_type: TopologyRuleType::MustNotImport,
            source_pattern: "src/neo4j/**".to_string(),
            target_pattern: Some("src/api/**".to_string()),
            threshold: None,
            severity: TopologySeverity::Error,
            description: "Neo4j must not import API".to_string(),
        };
        let json = serde_json::to_string(&rule).unwrap();
        let deserialized: TopologyRule = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-id");
        assert_eq!(deserialized.rule_type, TopologyRuleType::MustNotImport);
        assert_eq!(deserialized.target_pattern, Some("src/api/**".to_string()));
        assert!(deserialized.threshold.is_none());
    }

    #[test]
    fn test_topology_violation_serde() {
        let violation = TopologyViolation {
            rule_id: "r1".to_string(),
            rule_description: "No cross-layer imports".to_string(),
            rule_type: TopologyRuleType::MustNotImport,
            violator_path: "src/neo4j/client.rs".to_string(),
            target_path: Some("src/api/routes.rs".to_string()),
            severity: TopologySeverity::Error,
            details: "Direct import detected".to_string(),
            violation_score: 0.85,
        };
        let json = serde_json::to_string(&violation).unwrap();
        let deserialized: TopologyViolation = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.violator_path, "src/neo4j/client.rs");
        assert!((deserialized.violation_score - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_topology_check_result_serde() {
        let result = TopologyCheckResult {
            project_id: "proj-1".to_string(),
            rules_checked: 5,
            violations: vec![],
            error_count: 0,
            warning_count: 0,
            timing_ms: 42,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: TopologyCheckResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.rules_checked, 5);
        assert_eq!(deserialized.timing_ms, 42);
    }
}
