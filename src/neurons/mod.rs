//! Neural Network module for Knowledge Notes
//!
//! Implements spreading activation over the knowledge graph:
//! notes act as neurons, SYNAPSE relations as weighted connections,
//! and the `energy` field gates activation propagation.
//!
//! This module is **independent** from the existing PageRank-weighted
//! propagation in `get_context_notes`. Both systems coexist during
//! Phase 3 (dual-run comparison).

pub mod activation;
pub mod config;

pub use activation::{ActivatedNote, ActivationSource, SpreadingActivationEngine};
pub use config::{AutoReinforcementConfig, SpreadingActivationConfig};
