//! Graph analytics engine.
//!
//! Provides in-process graph data science capabilities using petgraph and
//! rustworkx-core. Computes PageRank, betweenness centrality, community
//! detection (Louvain), clustering coefficients, and weakly connected
//! components on the code knowledge graph.
//!
//! ## Architecture
//!
//! ```text
//! Neo4j (GraphStore) ──► extraction ──► petgraph::DiGraph
//!                                            │
//!                                       algorithms
//!                                            │
//!                                    GraphAnalytics result
//!                                            │
//!                                        writer ──► Neo4j (batch update)
//! ```
//!
//! ## Modules
//!
//! - [`models`] — Data structures (NodeMetrics, CommunityInfo, GraphAnalytics, AnalyticsConfig)
//! - [`algorithms`] — Algorithm implementations (PageRank, Betweenness, Louvain, Clustering, WCC)
//! - [`extraction`] — Neo4j → petgraph conversion via GraphStore trait
//! - [`writer`] — Batch-write results back to Neo4j via GraphStore trait

pub mod algorithms;
pub mod extraction;
pub mod models;
pub mod writer;

// Re-export primary types for convenience
pub use models::{
    AnalyticsConfig, CodeEdge, CodeEdgeType, CodeGraph, CodeHealthReport, CodeNode, CodeNodeType,
    CommunityInfo, ComponentInfo, FileAnalyticsUpdate, FunctionAnalyticsUpdate, GraphAnalytics,
    NodeMetrics,
};
