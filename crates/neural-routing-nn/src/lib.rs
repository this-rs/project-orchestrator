//! Nearest Neighbor Router — the permanent fallback for neural route learning.
//!
//! Given a query embedding, finds the K most similar past trajectories and
//! extracts the best route. Zero ML dependencies, works from ~100 trajectories.

pub mod router;
pub mod scoring;
pub mod metrics;

pub use router::{NNRouter, NNConfig};
pub use metrics::NNMetrics;
