//! AgentExecutor — executes business protocol actions via Claude agent (V1 stub).
//!
//! For V1, this executor provides:
//! - **Fast path**: if the action is parseable as an MCP-style call (e.g., `admin(...)`),
//!   delegates to the [`SystemExecutor`] logic.
//! - **Slow path**: logs that LLM execution would be needed and returns a "completed" trigger.
//!
//! The real LLM integration (prompt composition, tool use, streaming) comes in a later version.

use super::system::SystemExecutor;
use super::{ExecutionResult, Executor};
use crate::neo4j::traits::GraphStore;
use crate::protocol::{Protocol, ProtocolRun, ProtocolState};
use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, info};

/// Executes business protocol state actions.
///
/// V1 implementation: delegates MCP-style actions to [`SystemExecutor`],
/// and stubs LLM-driven actions with a log + auto-advance.
pub struct AgentExecutor {
    system: SystemExecutor,
}

impl AgentExecutor {
    /// Create a new AgentExecutor.
    pub fn new() -> Self {
        Self {
            system: SystemExecutor::new(),
        }
    }
}

impl Default for AgentExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if an action string looks like an MCP-style call: `tool(method, ...)`.
fn is_mcp_style(action: &str) -> bool {
    let trimmed = action.trim();
    // Check for pattern: word(...)  or  word(...) + word(...)
    trimmed.contains('(') && trimmed.contains(')')
}

#[async_trait]
impl Executor for AgentExecutor {
    async fn execute_state(
        &self,
        state: &ProtocolState,
        run: &ProtocolRun,
        protocol: &Protocol,
        store: &dyn GraphStore,
    ) -> Result<ExecutionResult> {
        let action_str = match &state.action {
            Some(a) if !a.is_empty() => a.clone(),
            _ => {
                debug!(
                    state_name = %state.name,
                    run_id = %run.id,
                    "AgentExecutor: no action defined — auto-advancing"
                );
                // Delegate to system executor for trigger determination
                return self.system.execute_state(state, run, protocol, store).await;
            }
        };

        // Fast path: MCP-style action → delegate to SystemExecutor
        if is_mcp_style(&action_str) {
            debug!(
                state_name = %state.name,
                action = %action_str,
                run_id = %run.id,
                "AgentExecutor: MCP-style action — delegating to SystemExecutor"
            );
            return self.system.execute_state(state, run, protocol, store).await;
        }

        // Slow path: would need LLM execution — stub for V1
        info!(
            state_name = %state.name,
            action = %action_str,
            run_id = %run.id,
            prompt_fragment = ?state.prompt_fragment,
            "AgentExecutor: LLM execution required (V1 stub — auto-completing)"
        );

        // Build a description of what would happen
        let mut notes = vec![format!(
            "V1 stub: action '{}' would require LLM execution",
            action_str
        )];
        if let Some(ref pf) = state.prompt_fragment {
            notes.push(format!("Prompt fragment: {}", pf));
        }
        if let Some(ref tools) = state.available_tools {
            notes.push(format!("Available tools: {}", tools.join(", ")));
        }

        // Auto-complete with "done" trigger — the system executor's
        // determine_trigger logic will find the right outgoing transition
        Ok(ExecutionResult {
            trigger: "done".to_string(),
            notes,
            should_retry: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mcp_style() {
        assert!(is_mcp_style("admin(update_staleness_scores)"));
        assert!(is_mcp_style("admin(a) + admin(b)"));
        assert!(is_mcp_style("note(list, project_id)"));
        assert!(!is_mcp_style("Run all tests"));
        assert!(!is_mcp_style("analyze code quality"));
        assert!(!is_mcp_style(""));
    }
}
