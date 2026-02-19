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
//!                                            │
//!                              AnalyticsEngine (orchestrator)
//! ```
//!
//! ## Modules
//!
//! - [`models`] — Data structures (NodeMetrics, CommunityInfo, GraphAnalytics, AnalyticsConfig)
//! - [`algorithms`] — Algorithm implementations (PageRank, Betweenness, Louvain, Clustering, WCC)
//! - [`extraction`] — Neo4j → petgraph conversion via GraphStore trait
//! - [`writer`] — Batch-write results back to Neo4j via GraphStore trait
//! - [`engine`] — `AnalyticsEngine` trait and `GraphAnalyticsEngine` orchestrator
//! - [`mock`] — `MockAnalyticsEngine` for testing (cfg(test) only)

pub mod algorithms;
pub mod engine;
pub mod extraction;
pub mod models;
pub mod writer;

#[cfg(test)]
pub mod mock;

// Re-export primary types for convenience
pub use engine::{AnalyticsEngine, GraphAnalyticsEngine, ProjectAnalytics};
pub use models::{
    AnalyticsConfig, CodeEdge, CodeEdgeType, CodeGraph, CodeHealthReport, CodeNode, CodeNodeType,
    CommunityInfo, ComponentInfo, FileAnalyticsUpdate, FunctionAnalyticsUpdate, GraphAnalytics,
    NodeMetrics,
};
