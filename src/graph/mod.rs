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
//! - [`debouncer`] — `AnalyticsDebouncer` for coalescing rapid-fire triggers
//! - [`mock`] — `MockAnalyticsEngine` for testing (cfg(test) only)
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod algorithms;
pub mod confidence;
pub mod cpu;
pub mod debouncer;
pub mod engine;
pub mod enrichment;
pub mod extraction;
pub mod guard;
pub mod models;
pub mod neighborhood;
pub mod process;
pub mod rules;
pub mod schema;
pub mod writer;

#[cfg(test)]
pub mod mock;

#[cfg(test)]
mod self_model_tests;

// Re-export primary types for convenience
pub use confidence::{
    confidence_from_graph_density, confidence_from_signal_convergence, ConfidenceBasis,
    ConfidenceFeedback, ConfidenceScore, ConfidenceTracker, PredictionOutcome, SystemConfidence,
};
pub use debouncer::{
    AnalyticsDebouncer, CoChangeDebouncer, NeuralReinforcementDebouncer, ReinforcementPayload,
};
pub use engine::{AnalyticsEngine, GraphAnalyticsEngine, ProjectAnalytics};
pub use models::{
    AnalyticsConfig, CodeEdge, CodeEdgeType, CodeGraph, CodeHealthReport, CodeNode, CodeNodeType,
    CommunityInfo, ComponentInfo, FabricFileAnalyticsUpdate, FabricWeights, FileAnalyticsUpdate,
    FunctionAnalyticsUpdate, GraphAnalytics, NodeMetrics,
};
