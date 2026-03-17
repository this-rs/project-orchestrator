//! HeartbeatEngine — background daemon for periodic health checks.
//!
//! Runs permanently on the server (not tied to chat sessions).
//! Each check implements `HeartbeatCheck` with its own interval.
//! The engine evaluates all checks in a single tokio loop,
//! respecting individual intervals and a 5s timeout per check.

pub mod checks;
pub mod engine;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;

use crate::events::EventEmitter;
use crate::meilisearch::SearchStore;
use crate::neo4j::traits::GraphStore;

// ============================================================================
// HeartbeatCheck trait
// ============================================================================

/// Context provided to each heartbeat check on every tick.
pub struct HeartbeatContext {
    /// Shared graph store for Neo4j operations.
    pub graph: Arc<dyn GraphStore>,
    /// Optional search store for MeiliSearch operations (e.g., backfill_synapses).
    pub search: Option<Arc<dyn SearchStore>>,
    /// Optional event emitter for broadcasting CrudEvents (e.g., AlertCreated).
    pub emitter: Option<Arc<dyn EventEmitter>>,
}

/// A periodic health check executed by the HeartbeatEngine.
///
/// Each check defines its own interval and logic. The engine
/// calls `run()` at most once per interval, with a default 5s timeout.
/// Checks can override the timeout via `timeout_override()` for
/// operations that need more time (e.g., paginated backfill).
/// If a check exceeds its timeout, it is skipped (not retried).
#[async_trait]
pub trait HeartbeatCheck: Send + Sync {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// How often this check should run.
    fn interval(&self) -> Duration;

    /// Optional per-check timeout override.
    /// Returns `None` to use the engine's default (5s).
    /// Override this for checks that need more time (e.g., paginated backfill).
    fn timeout_override(&self) -> Option<Duration> {
        None
    }

    /// Execute the check. Returns Ok(()) on success, or an error
    /// that will be logged but won't crash the engine.
    async fn run(&self, ctx: &HeartbeatContext) -> Result<()>;
}
