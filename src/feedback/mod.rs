//! OutcomeTracker — Closed-loop learning system.
//!
//! This module implements the feedback loop that closes the learning cycle:
//! 1. **Explicit feedback** (API): Users rate decisions, notes, plans via POST /api/feedback
//! 2. **Implicit signals**: Automatic detection of CommitReverted, TaskRestarted, etc.
//! 3. **Score propagation**: Boost/penalize notes, synapses, profiles based on signals
//!
//! The OutcomeTracker works standalone (explicit feedback API) even if other
//! subsystems (HeartbeatEngine, NeuralFeedback) are not deployed.

pub mod handlers;
pub mod models;
pub mod propagator;
pub mod signals;
pub mod tracker;

pub use models::*;
pub use tracker::OutcomeTracker;
