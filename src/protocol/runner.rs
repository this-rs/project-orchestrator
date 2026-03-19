//! Protocol Runner — autonomous FSM execution loop.
//!
//! Drives runner-managed protocol runs to completion by repeatedly:
//! 1. Loading the current state
//! 2. Selecting the appropriate executor (System vs Agent)
//! 3. Executing the state action
//! 4. Firing the resulting transition
//! 5. Looping until a terminal state is reached or an error occurs
//!
//! Each run is driven by a spawned tokio task with a CancellationToken
//! for graceful shutdown.

use crate::chat::manager::ChatManager;
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
use crate::neo4j::traits::GraphStore;
use crate::protocol::engine;
use crate::protocol::executor::agent::AgentExecutor;
use crate::protocol::executor::system::SystemExecutor;
use crate::protocol::executor::Executor;
use crate::protocol::{ProtocolCategory, ProtocolRun, RunStatus, StateType};
use anyhow::{bail, Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Base backoff delay for retries (1 second).
const BASE_BACKOFF_MS: u64 = 1000;

/// Default timeout for system protocol states (5 minutes).
const DEFAULT_SYSTEM_TIMEOUT_SECS: u64 = 300;

/// Default timeout for business protocol states (30 minutes).
const DEFAULT_BUSINESS_TIMEOUT_SECS: u64 = 1800;

/// Maximum number of retries for transient failures.
const MAX_RETRIES: u32 = 3;

/// Run a protocol to completion.
///
/// This function is the core loop of the ProtocolRunner. It:
/// 1. Loads the run and its protocol
/// 2. Marks the run as `runner_managed = true`
/// 3. Selects the appropriate executor (System vs Agent) based on protocol category
/// 4. Loops through states, executing actions and firing transitions
/// 5. Handles timeouts, cancellation, retries, and errors
///
/// # Arguments
/// - `store` — shared GraphStore for persistence
/// - `run_id` — the protocol run to drive
/// - `cancel` — cancellation token for graceful shutdown
/// - `emitter` — event emitter for CRUD notifications
/// - `chat_manager` — optional ChatManager for LLM-driven business protocols.
///   When None, business protocol states that require LLM will use stub behavior.
///
/// # Returns
/// The final state of the protocol run (completed, failed, or cancelled).
pub async fn run_protocol(
    store: Arc<dyn GraphStore>,
    run_id: Uuid,
    cancel: CancellationToken,
    emitter: Arc<dyn EventEmitter>,
    chat_manager: Option<Arc<ChatManager>>,
) -> Result<ProtocolRun> {
    // 1. Load run + protocol
    let mut run = store
        .get_protocol_run(run_id)
        .await?
        .with_context(|| format!("ProtocolRun not found: {run_id}"))?;

    if run.status != RunStatus::Running {
        bail!(
            "Run {} is not active (status: {}), cannot drive",
            run_id,
            run.status
        );
    }

    let protocol = store
        .get_protocol(run.protocol_id)
        .await?
        .with_context(|| format!("Protocol not found: {}", run.protocol_id))?;

    // 2. Mark runner_managed = true
    if !run.runner_managed {
        run.runner_managed = true;
        store.update_protocol_run(&mut run).await?;
    }

    // 3. Select executor based on protocol category
    let executor: Box<dyn Executor> = match protocol.protocol_category {
        ProtocolCategory::System => Box::new(SystemExecutor::new()),
        ProtocolCategory::Business => {
            Box::new(AgentExecutor::new(chat_manager, cancel.clone()))
        }
    };

    info!(
        run_id = %run_id,
        protocol_name = %protocol.name,
        category = %protocol.protocol_category,
        "ProtocolRunner: starting execution loop"
    );

    // 4. Main execution loop
    loop {
        // Reload run to get latest state (may have been updated by fire_transition)
        run = store
            .get_protocol_run(run_id)
            .await?
            .with_context(|| format!("ProtocolRun disappeared: {run_id}"))?;

        // Check if run is still active
        if run.status != RunStatus::Running {
            info!(
                run_id = %run_id,
                status = %run.status,
                "ProtocolRunner: run is no longer active — exiting loop"
            );
            break;
        }

        // Get current state
        let states = store.get_protocol_states(run.protocol_id).await?;
        let current_state = states
            .iter()
            .find(|s| s.id == run.current_state)
            .with_context(|| {
                format!(
                    "Current state {} not found in protocol {}",
                    run.current_state, run.protocol_id
                )
            })?
            .clone();

        // Check if terminal → break
        if current_state.state_type == StateType::Terminal {
            debug!(
                run_id = %run_id,
                state_name = %current_state.name,
                "ProtocolRunner: reached terminal state"
            );
            break;
        }

        // Check if state has no action → break (wait for external trigger)
        if current_state.action.as_ref().is_none_or(|a| a.is_empty()) {
            debug!(
                run_id = %run_id,
                state_name = %current_state.name,
                "ProtocolRunner: state has no action — waiting for external trigger"
            );
            break;
        }

        // Emit progress event: starting state execution
        emit_progress_event(
            &emitter,
            run_id,
            &current_state.name,
            "executing",
            None,
            None,
            protocol.project_id,
        );

        // Determine timeout
        let timeout_secs =
            current_state
                .state_timeout_secs
                .unwrap_or(match protocol.protocol_category {
                    ProtocolCategory::System => DEFAULT_SYSTEM_TIMEOUT_SECS,
                    ProtocolCategory::Business => DEFAULT_BUSINESS_TIMEOUT_SECS,
                });
        let timeout = Duration::from_secs(timeout_secs);

        // Execute with retry loop
        let mut retries = 0u32;
        let execution_result = loop {
            let exec_future = executor.execute_state(&current_state, &run, &protocol, &*store);

            // Execute with timeout and cancellation
            let result = tokio::select! {
                _ = cancel.cancelled() => {
                    info!(
                        run_id = %run_id,
                        state_name = %current_state.name,
                        "ProtocolRunner: cancelled"
                    );
                    // Cancel the run
                    if let Err(e) = engine::cancel_run(&*store, run_id).await {
                        warn!(run_id = %run_id, "Failed to cancel run: {}", e);
                    }
                    // Reload and return
                    let final_run = store.get_protocol_run(run_id).await?.unwrap_or(run);
                    return Ok(final_run);
                }
                result = tokio::time::timeout(timeout, exec_future) => {
                    match result {
                        Ok(Ok(exec_result)) => Ok(exec_result),
                        Ok(Err(e)) => Err(e),
                        Err(_) => Err(anyhow::anyhow!(
                            "State '{}' execution timed out after {}s",
                            current_state.name,
                            timeout_secs
                        )),
                    }
                }
            };

            match result {
                Ok(exec_result) => {
                    if exec_result.should_retry && retries < MAX_RETRIES {
                        retries += 1;
                        let backoff = Duration::from_millis(BASE_BACKOFF_MS * 2u64.pow(retries - 1));
                        warn!(
                            run_id = %run_id,
                            state_name = %current_state.name,
                            retry = retries,
                            "ProtocolRunner: retrying after backoff ({:?})",
                            backoff
                        );
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    break Ok(exec_result);
                }
                Err(e) => {
                    if retries < MAX_RETRIES {
                        retries += 1;
                        let backoff = Duration::from_millis(BASE_BACKOFF_MS * 2u64.pow(retries - 1));
                        warn!(
                            run_id = %run_id,
                            state_name = %current_state.name,
                            retry = retries,
                            error = %e,
                            "ProtocolRunner: transient error, retrying after backoff ({:?})",
                            backoff
                        );
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    break Err(e);
                }
            }
        };

        match execution_result {
            Ok(exec_result) => {
                info!(
                    run_id = %run_id,
                    state_name = %current_state.name,
                    trigger = %exec_result.trigger,
                    notes = ?exec_result.notes,
                    "ProtocolRunner: firing transition"
                );

                // Fire the transition
                let transition_result =
                    engine::fire_transition(&*store, run_id, &exec_result.trigger).await?;

                if !transition_result.success {
                    let error_msg = transition_result.error.unwrap_or_else(|| {
                        format!(
                            "Transition '{}' from state '{}' failed",
                            exec_result.trigger, current_state.name
                        )
                    });
                    warn!(
                        run_id = %run_id,
                        "ProtocolRunner: transition failed — failing run: {}",
                        error_msg
                    );
                    fail_run_with_error(&*store, run_id, &error_msg).await?;
                    let final_run = store.get_protocol_run(run_id).await?.unwrap_or(run);
                    return Ok(final_run);
                }

                // Emit CRUD event for the transition
                emit_transition_event(
                    &emitter,
                    run_id,
                    &transition_result.current_state_name,
                    transition_result.status,
                    protocol.project_id,
                );
            }
            Err(e) => {
                warn!(
                    run_id = %run_id,
                    state_name = %current_state.name,
                    error = %e,
                    "ProtocolRunner: execution failed after retries — failing run"
                );
                fail_run_with_error(
                    &*store,
                    run_id,
                    &format!("Execution failed in state '{}': {}", current_state.name, e),
                )
                .await?;
                let final_run = store.get_protocol_run(run_id).await?.unwrap_or(run);
                return Ok(final_run);
            }
        }
    }

    // 5. Return final run state
    let final_run = store.get_protocol_run(run_id).await?.unwrap_or(run);

    info!(
        run_id = %run_id,
        status = %final_run.status,
        states_visited = final_run.states_visited.len(),
        "ProtocolRunner: execution loop finished"
    );

    Ok(final_run)
}

/// Mark a run as failed with an error message.
async fn fail_run_with_error(store: &dyn GraphStore, run_id: Uuid, error: &str) -> Result<()> {
    if let Some(mut run) = store.get_protocol_run(run_id).await? {
        if run.is_active() {
            run.fail(error);
            store.update_protocol_run(&mut run).await?;
        }
    }
    Ok(())
}

/// Emit a CrudEvent::Progress for in-flight state execution updates.
///
/// Used to notify the frontend (via WebSocket) that the runner is executing
/// a state, including optional processed/total counters for long-running states.
fn emit_progress_event(
    emitter: &Arc<dyn EventEmitter>,
    run_id: Uuid,
    state_name: &str,
    sub_action: &str,
    processed: Option<u64>,
    total: Option<u64>,
    project_id: Uuid,
) {
    let mut payload = serde_json::json!({
        "current_state": state_name,
        "sub_action": sub_action,
    });
    if let Some(p) = processed {
        payload["processed"] = serde_json::json!(p);
    }
    if let Some(t) = total {
        payload["total"] = serde_json::json!(t);
    }
    emitter.emit(CrudEvent {
        entity_type: EntityType::ProtocolRun,
        action: CrudAction::Progress,
        entity_id: run_id.to_string(),
        related: None,
        payload,
        timestamp: chrono::Utc::now().to_rfc3339(),
        project_id: Some(project_id.to_string()),
    });
}

/// Emit a CrudEvent after a protocol run state transition.
fn emit_transition_event(
    emitter: &Arc<dyn EventEmitter>,
    run_id: Uuid,
    state_name: &str,
    status: RunStatus,
    project_id: Uuid,
) {
    let action = match status {
        RunStatus::Completed => CrudAction::StatusChanged,
        RunStatus::Failed => CrudAction::StatusChanged,
        RunStatus::Cancelled => CrudAction::StatusChanged,
        RunStatus::Running => CrudAction::Updated,
    };

    emitter.emit(CrudEvent {
        entity_type: EntityType::ProtocolRun,
        action,
        entity_id: run_id.to_string(),
        related: None,
        payload: serde_json::json!({
            "current_state": state_name,
            "status": status.to_string(),
        }),
        timestamp: chrono::Utc::now().to_rfc3339(),
        project_id: Some(project_id.to_string()),
    });
}
