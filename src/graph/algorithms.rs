//! Graph analytics algorithms.
//!
//! Implements core graph data science algorithms on petgraph graphs:
//! - **PageRank** — via `rustworkx_core::centrality::pagerank`
//! - **Betweenness centrality** — via `rustworkx_core::centrality::betweenness_centrality`
//! - **Community detection (Louvain)** — custom implementation (~200 lines)
//! - **Clustering coefficient** — local clustering per node
//! - **Weakly connected components** — via petgraph's `connected_components` on undirected view
//!
//! All algorithms operate on `petgraph::DiGraph<String, String>` and return
//! results as `Vec<f64>` or `Vec<usize>` indexed by `NodeIndex`.
//!
//! The Louvain algorithm is implemented from scratch because the `graphina` crate
//! requires Rust 1.86+ (our MSRV target is 1.70+).
