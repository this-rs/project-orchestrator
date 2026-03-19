//! Protocol execution engine — drives FSM runs to completion.
//!
//! Provides the [`Executor`] trait for executing protocol state actions,
//! and two concrete implementations:
//! - [`SystemExecutor`]: Executes system protocol actions natively via GraphStore (no LLM)
//! - [`AgentExecutor`]: Executes business protocol actions via Claude agent sessions.
//!   Uses a fast-path (MCP-style actions → SystemExecutor) and a slow-path
//!   (spawns a Claude session via ChatManager, streams response, extracts trigger).
//!
//! The runner loop ([`super::runner`]) selects the appropriate executor
//! based on the protocol's `protocol_category`.

pub mod agent;
pub mod system;

use crate::neo4j::traits::GraphStore;
use crate::protocol::{Protocol, ProtocolRun, ProtocolState};
use anyhow::Result;
use async_trait::async_trait;

/// Result of executing a protocol state action.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// The trigger to fire for the next transition.
    pub trigger: String,
    /// Optional notes or log messages produced during execution.
    pub notes: Vec<String>,
    /// Whether this execution should be retried (transient failure).
    pub should_retry: bool,
}

/// Trait for executing protocol state actions.
///
/// Implementors drive a single state's action to completion and return
/// the trigger that should be fired on the FSM to advance to the next state.
#[async_trait]
pub trait Executor: Send + Sync {
    /// Execute the action defined on a protocol state.
    ///
    /// Returns an [`ExecutionResult`] containing the trigger to fire
    /// for the next transition, or an error if execution fails.
    async fn execute_state(
        &self,
        state: &ProtocolState,
        run: &ProtocolRun,
        protocol: &Protocol,
        store: &dyn GraphStore,
    ) -> Result<ExecutionResult>;
}
