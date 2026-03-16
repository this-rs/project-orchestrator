//! Concrete HeartbeatCheck implementations.
//!
//! Each check runs at its own interval and operates on the shared GraphStore.
//! Checks MUST NOT block the Axum server — they run in a separate tokio::spawn.
//! Any check exceeding 5s is automatically skipped by the engine.

pub mod convention_guard;
pub mod git_drift;
pub mod maintenance;
pub mod staleness;
pub mod synapse_decay;
