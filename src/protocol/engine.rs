//! FSM Runtime Engine
//!
//! Provides the core state machine execution logic for Protocol runs.
//! These functions operate on the [`GraphStore`] trait, making them
//! testable with both real Neo4j and the mock implementation.
//!
//! # Key Operations
//!
//! - [`start_run`]: Create a new protocol run at the entry state
//! - [`fire_transition`]: Advance a run by firing a trigger
//! - [`cancel_run`]: Cancel an active run
//!
//! # State Machine Rules
//!
//! 1. A run starts at the protocol's `entry_state`
//! 2. Transitions are matched by `(current_state, trigger)` pair
//! 3. Guards are evaluated as simple string conditions (future: expression engine)
//! 4. Reaching a terminal state automatically completes the run
//! 5. Only `Running` runs can accept transitions

use crate::neo4j::traits::GraphStore;
use crate::protocol::{
    ChildCompletionInfo, CompletionStrategy, OnFailureStrategy, ProtocolRun, RunStatus, StateType,
    TransitionResult,
};
use anyhow::{bail, Context, Result};
use uuid::Uuid;

/// Maximum nesting depth for hierarchical protocol runs.
/// Prevents infinite recursion when protocols reference each other.
const MAX_HIERARCHY_DEPTH: u32 = 5;

/// Start a new protocol run.
///
/// Creates a `ProtocolRun` at the protocol's entry state and persists it.
/// Returns the newly created run.
///
/// # Concurrency
///
/// Only one `Running` run is allowed per protocol at a time (mutual exclusion).
/// If a run is already `Running`, this function returns an error with message
/// "Skipped: concurrent run already running". The store-level `create_protocol_run`
/// also enforces this atomically (conditional CREATE in Neo4j, check-and-insert
/// within the same write lock in the mock).
///
/// # Arguments
/// - `triggered_by` — How this run was started. `None` defaults to `"manual"`.
///   Event-driven hooks pass `"event:post_sync"`, `"event:post_import"`, etc.
///   Scheduler passes `"schedule:daily"`, `"schedule:hourly"`, etc.
///
/// # Errors
/// - Protocol not found
/// - Entry state not found in the protocol's states
/// - Concurrent run already running for this protocol
pub async fn start_run(
    store: &dyn GraphStore,
    protocol_id: Uuid,
    plan_id: Option<Uuid>,
    task_id: Option<Uuid>,
    triggered_by: Option<&str>,
) -> Result<ProtocolRun> {
    // 1. Load the protocol
    let protocol = store
        .get_protocol(protocol_id)
        .await?
        .with_context(|| format!("Protocol not found: {protocol_id}"))?;

    // 2. Resolve the entry state name
    let states = store.get_protocol_states(protocol_id).await?;
    let entry_state = states
        .iter()
        .find(|s| s.id == protocol.entry_state)
        .with_context(|| {
            format!(
                "Entry state {} not found in protocol {}",
                protocol.entry_state, protocol_id
            )
        })?;

    // 3. Concurrency check: reject if a Running run already exists.
    //    This is the fast-path check. The store also enforces atomically
    //    via conditional CREATE (Neo4j) or check-within-lock (mock) to
    //    prevent TOCTOU races.
    let (_, running_count) = store
        .list_protocol_runs(protocol_id, Some(RunStatus::Running), 1, 0)
        .await?;

    if running_count > 0 {
        bail!(
            "Skipped: concurrent run already running for protocol {}",
            protocol_id
        );
    }

    // 4. Create the run
    let mut run = ProtocolRun::new(protocol_id, entry_state.id, &entry_state.name);
    run.plan_id = plan_id;
    run.task_id = task_id;
    if let Some(trigger) = triggered_by {
        run.triggered_by = trigger.to_string();
    }

    // 5. Persist (store enforces atomicity as a second line of defense)
    store
        .create_protocol_run(&run)
        .await
        .context("Failed to create protocol run")?;

    Ok(run)
}

/// Start a child run for hierarchical protocol execution.
///
/// Creates a `ProtocolRun` linked to a parent run via `parent_run_id`.
/// The child run inherits the parent's `plan_id` and `task_id`.
///
/// # Hierarchy Safety
/// - Depth is `parent.depth + 1`; bails if `depth >= MAX_HIERARCHY_DEPTH` (5)
/// - Cycle detection: walks ancestors to ensure `sub_protocol_id` is not already
///   in the ancestor chain (prevents A→B→A infinite loops)
///
/// # Errors
/// - Max depth exceeded
/// - Cycle detected in protocol hierarchy
/// - Sub-protocol not found or has no entry state
pub async fn start_child_run(
    store: &dyn GraphStore,
    parent_run: &ProtocolRun,
    sub_protocol_id: Uuid,
) -> Result<ProtocolRun> {
    let child_depth = parent_run.depth + 1;

    // Guard: max depth
    if child_depth >= MAX_HIERARCHY_DEPTH {
        bail!(
            "Max hierarchy depth ({}) exceeded: parent run {} is at depth {}, \
             cannot spawn child at depth {}",
            MAX_HIERARCHY_DEPTH,
            parent_run.id,
            parent_run.depth,
            child_depth,
        );
    }

    // Guard: cycle detection — walk ancestors to check no protocol_id repeats
    let mut ancestor_protocol_ids = vec![parent_run.protocol_id];
    let mut current_ancestor_id = parent_run.parent_run_id;
    while let Some(ancestor_id) = current_ancestor_id {
        if let Some(ancestor) = store.get_protocol_run(ancestor_id).await? {
            ancestor_protocol_ids.push(ancestor.protocol_id);
            current_ancestor_id = ancestor.parent_run_id;
        } else {
            break;
        }
    }
    if ancestor_protocol_ids.contains(&sub_protocol_id) {
        bail!(
            "Cycle detected in protocol hierarchy: sub_protocol {} is already \
             an ancestor of run {}. Chain: {:?}",
            sub_protocol_id,
            parent_run.id,
            ancestor_protocol_ids,
        );
    }

    // Load sub-protocol and its entry state
    let sub_protocol = store
        .get_protocol(sub_protocol_id)
        .await?
        .with_context(|| format!("Sub-protocol not found: {sub_protocol_id}"))?;

    let states = store.get_protocol_states(sub_protocol_id).await?;
    let entry_state = states
        .iter()
        .find(|s| s.id == sub_protocol.entry_state)
        .with_context(|| {
            format!(
                "Entry state {} not found in sub-protocol {}",
                sub_protocol.entry_state, sub_protocol_id
            )
        })?;

    // Create the child run
    let mut child = ProtocolRun::new(sub_protocol_id, entry_state.id, &entry_state.name);
    child.parent_run_id = Some(parent_run.id);
    child.depth = child_depth;
    child.plan_id = parent_run.plan_id;
    child.task_id = parent_run.task_id;
    child.triggered_by = format!("hierarchy:parent_{}", parent_run.id);

    // Persist (no concurrency guard for child runs — parent manages exclusion)
    store
        .create_protocol_run(&child)
        .await
        .context("Failed to create child protocol run")?;

    Ok(child)
}

/// Fire a transition on a running protocol.
///
/// Finds a matching transition from the current state with the given trigger,
/// advances the run to the target state, and auto-completes if the target
/// is a terminal state.
///
/// **Hierarchical FSM**: If the target state has a `sub_protocol_id`, a child
/// run is automatically spawned. The parent stays `Running` at the macro-state
/// until the child completes (see [`handle_child_completion`]).
///
/// # Returns
/// A [`TransitionResult`] indicating success/failure and the new state.
/// If a child run was spawned, `child_run_id` will be set.
///
/// # Errors
/// - Run not found
/// - Run is not active (already completed/failed/cancelled)
/// - No matching transition for the given trigger from current state
/// - Max hierarchy depth exceeded (if target has sub_protocol_id)
/// - Cycle detected in protocol hierarchy
pub fn fire_transition<'a>(
    store: &'a dyn GraphStore,
    run_id: Uuid,
    trigger: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TransitionResult>> + Send + 'a>> {
    Box::pin(fire_transition_inner(store, run_id, trigger))
}

async fn fire_transition_inner(
    store: &dyn GraphStore,
    run_id: Uuid,
    trigger: &str,
) -> Result<TransitionResult> {
    // 1. Load the run
    let mut run = store
        .get_protocol_run(run_id)
        .await?
        .with_context(|| format!("ProtocolRun not found: {run_id}"))?;

    // 2. Check run is active
    if !run.is_active() {
        return Ok(TransitionResult {
            success: false,
            current_state: run.current_state,
            current_state_name: last_state_name(&run),
            run_completed: run.is_finished(),
            status: run.status,
            error: Some(format!(
                "Run is not active (status: {}). Cannot fire transition.",
                run.status
            )),
            child_run_id: None,
            child_completion: None,
        });
    }

    // 3. Load transitions and find a match
    let transitions = store.get_protocol_transitions(run.protocol_id).await?;
    let matching = transitions
        .iter()
        .find(|t| t.from_state == run.current_state && t.trigger == trigger);

    let transition = match matching {
        Some(t) => t,
        None => {
            return Ok(TransitionResult {
                success: false,
                current_state: run.current_state,
                current_state_name: last_state_name(&run),
                run_completed: false,
                status: run.status,
                error: Some(format!(
                    "No transition found from state {} with trigger '{trigger}'",
                    run.current_state
                )),
                child_run_id: None,
                child_completion: None,
            });
        }
    };

    // 4. Check guard (if any) — for now, guards are informational only
    // Future: evaluate guard expressions against run context
    // if let Some(guard) = &transition.guard { ... }

    // 5. Resolve target state name
    let states = store.get_protocol_states(run.protocol_id).await?;
    let target_state = states
        .iter()
        .find(|s| s.id == transition.to_state)
        .with_context(|| {
            format!(
                "Target state {} not found in protocol {}",
                transition.to_state, run.protocol_id
            )
        })?;

    // 6. Advance the run
    run.visit_state(target_state.id, &target_state.name, trigger);

    // 7. Auto-complete if terminal state
    let protocol = store.get_protocol(run.protocol_id).await?;
    let is_terminal = protocol
        .as_ref()
        .map(|p| p.terminal_states.contains(&target_state.id))
        .unwrap_or(false)
        || target_state.is_terminal();

    if is_terminal {
        run.complete();
    }

    // 8. Persist the updated run
    store
        .update_protocol_run(&mut run)
        .await
        .context("Failed to update protocol run after transition")?;

    // 9. If this run just completed and has a parent, handle child completion
    let child_completion = if is_terminal {
        if let Some(parent_run_id) = run.parent_run_id {
            // Auto-transition parent if completion strategy allows
            let _ = handle_child_completion(store, &run).await;
            // Populate child completion info for event emission at handler layer
            Some(ChildCompletionInfo {
                parent_run_id,
                child_run_id: run.id,
                protocol_id: run.protocol_id,
                status: run.status,
            })
        } else {
            None
        }
    } else {
        None
    };

    // 10. Auto-spawn child run if target state has sub_protocol_id (hierarchical FSM)
    let child_run_id = if !is_terminal {
        if let Some(sub_protocol_id) = target_state.sub_protocol_id {
            let child = start_child_run(store, &run, sub_protocol_id).await?;
            Some(child.id)
        } else {
            None
        }
    } else {
        None
    };

    // 11. Generator state: create RuntimeStates if target is a Generator
    if !is_terminal && target_state.state_type == StateType::Generator {
        // generate() is idempotent — skips if RuntimeStates already exist
        let _ = crate::protocol::generator::generate(target_state, &run, store).await;
    }

    Ok(TransitionResult {
        success: true,
        current_state: run.current_state,
        current_state_name: target_state.name.clone(),
        run_completed: is_terminal,
        status: run.status,
        error: None,
        child_run_id,
        child_completion,
    })
}

/// Handle completion of a child run — auto-transition parent if completion strategy allows.
///
/// Called after a run reaches a terminal state. If the run has a `parent_run_id`:
/// 1. Load parent run and find its current macro-state
/// 2. Check `completion_strategy` on the macro-state
/// 3. If strategy allows (`AllComplete`: all siblings done, `AnyComplete`: immediate),
///    fire `child_completed` trigger on parent
/// 4. `Manual`: do nothing
///
/// Returns `Some(TransitionResult)` if the parent was auto-transitioned, `None` otherwise.
pub async fn handle_child_completion(
    store: &dyn GraphStore,
    completed_run: &ProtocolRun,
) -> Result<Option<TransitionResult>> {
    // Only process if this run has a parent
    let parent_run_id = match completed_run.parent_run_id {
        Some(id) => id,
        None => return Ok(None),
    };

    let parent_run = store
        .get_protocol_run(parent_run_id)
        .await?
        .with_context(|| format!("Parent run not found: {parent_run_id}"))?;

    if !parent_run.is_active() {
        return Ok(None); // Parent already done
    }

    // Find the macro-state (parent's current state)
    let states = store.get_protocol_states(parent_run.protocol_id).await?;
    let macro_state = states.iter().find(|s| s.id == parent_run.current_state);

    // Handle child failure before checking completion strategy
    if completed_run.status == RunStatus::Failed {
        let failure_strategy = macro_state
            .and_then(|s| s.on_failure_strategy)
            .unwrap_or(OnFailureStrategy::Abort);

        return match failure_strategy {
            OnFailureStrategy::Abort => {
                let _failed_parent = fail_run(
                    store,
                    parent_run_id,
                    &format!("Child run failed: {}", completed_run.id),
                )
                .await?;
                Ok(None)
            }
            OnFailureStrategy::Skip => {
                let result =
                    Box::pin(fire_transition(store, parent_run_id, "child_skipped")).await?;
                Ok(Some(result))
            }
            OnFailureStrategy::Retry { max } => {
                // Count how many FAILED child runs exist for this parent
                let siblings = store.list_child_runs(parent_run_id).await?;
                let failed_count = siblings
                    .iter()
                    .filter(|r| r.status == RunStatus::Failed)
                    .count();

                if (failed_count as u8) <= max {
                    // Retry: spawn a new child run with the same sub-protocol
                    let parent_run_fresh = store
                        .get_protocol_run(parent_run_id)
                        .await?
                        .with_context(|| {
                            format!("Parent run not found for retry: {parent_run_id}")
                        })?;
                    let sub_protocol_id = completed_run.protocol_id;
                    let _child = start_child_run(store, &parent_run_fresh, sub_protocol_id).await?;
                    Ok(None)
                } else {
                    // Max retries exceeded — fall back to Abort
                    let _failed_parent = fail_run(
                        store,
                        parent_run_id,
                        &format!(
                            "Child run failed after {} retries: {}",
                            max, completed_run.id
                        ),
                    )
                    .await?;
                    Ok(None)
                }
            }
        };
    }

    let strategy = macro_state
        .and_then(|s| s.completion_strategy)
        .unwrap_or(CompletionStrategy::AllComplete);

    match strategy {
        CompletionStrategy::Manual => Ok(None),
        CompletionStrategy::AnyComplete => {
            // Transition parent immediately
            let result = fire_transition(store, parent_run_id, "child_completed").await?;
            Ok(Some(result))
        }
        CompletionStrategy::AllComplete => {
            // Check if ALL sibling child runs are completed
            let siblings = store.list_child_runs(parent_run_id).await?;
            let all_done = siblings.iter().all(|r| r.is_finished());
            if all_done {
                let result = fire_transition(store, parent_run_id, "child_completed").await?;
                Ok(Some(result))
            } else {
                Ok(None)
            }
        }
    }
}

/// Cancel an active protocol run.
///
/// # Errors
/// - Run not found
/// - Run is not active
pub async fn cancel_run(store: &dyn GraphStore, run_id: Uuid) -> Result<ProtocolRun> {
    let mut run = store
        .get_protocol_run(run_id)
        .await?
        .with_context(|| format!("ProtocolRun not found: {run_id}"))?;

    if !run.is_active() {
        bail!(
            "Cannot cancel run {} — status is already: {}",
            run_id,
            run.status
        );
    }

    // Signal the runner loop to stop gracefully (if active)
    crate::protocol::hooks::cancel_active_runner(run_id);

    run.cancel();

    store
        .update_protocol_run(&mut run)
        .await
        .context("Failed to update protocol run after cancel")?;

    Ok(run)
}

/// Fail an active protocol run with an error message.
///
/// Uses `Box::pin` internally to support recursive failure propagation
/// (child failure → parent failure via `handle_child_completion`).
///
/// # Errors
/// - Run not found
/// - Run is not active
pub fn fail_run<'a>(
    store: &'a dyn GraphStore,
    run_id: Uuid,
    error: &'a str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ProtocolRun>> + Send + 'a>> {
    Box::pin(fail_run_inner(store, run_id, error))
}

async fn fail_run_inner(store: &dyn GraphStore, run_id: Uuid, error: &str) -> Result<ProtocolRun> {
    let mut run = store
        .get_protocol_run(run_id)
        .await?
        .with_context(|| format!("ProtocolRun not found: {run_id}"))?;

    if !run.is_active() {
        bail!(
            "Cannot fail run {} — status is already: {}",
            run_id,
            run.status
        );
    }

    run.fail(error);

    store
        .update_protocol_run(&mut run)
        .await
        .context("Failed to update protocol run after fail")?;

    // If this failed run has a parent, handle child failure compensation
    if run.parent_run_id.is_some() {
        let _ = handle_child_completion(store, &run).await;
    }

    Ok(run)
}

/// Extract the last visited state name from a run.
fn last_state_name(run: &ProtocolRun) -> String {
    run.states_visited
        .last()
        .map(|sv| sv.state_name.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;
    use crate::protocol::{
        Protocol, ProtocolCategory, ProtocolState, ProtocolTransition, RunStatus,
    };

    /// Helper to set up a simple 3-state protocol in the mock:
    /// Start -> Processing -> Done (terminal)
    /// Triggers: "begin" (Start->Processing), "finish" (Processing->Done)
    async fn setup_protocol(store: &MockGraphStore) -> (Uuid, Protocol) {
        let project_id = Uuid::new_v4();
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-project".to_string(),
            slug: "test-project".to_string(),
            description: None,
            root_path: "/tmp/test".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        let protocol_id = Uuid::new_v4();
        let start_state = ProtocolState::start(protocol_id, "Start");
        let processing_state = ProtocolState::new(protocol_id, "Processing");
        let done_state = ProtocolState::terminal(protocol_id, "Done");

        let protocol = Protocol::new_full(
            project_id,
            "Test Protocol",
            "A simple test protocol",
            start_state.id,
            vec![done_state.id],
            ProtocolCategory::Business,
        );
        // Override ID
        let mut protocol = protocol;
        protocol.id = protocol_id;

        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&start_state).await.unwrap();
        store
            .upsert_protocol_state(&processing_state)
            .await
            .unwrap();
        store.upsert_protocol_state(&done_state).await.unwrap();

        // Transitions
        let t1 = ProtocolTransition::new(protocol_id, start_state.id, processing_state.id, "begin");
        let t2 = ProtocolTransition::new(protocol_id, processing_state.id, done_state.id, "finish");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        (project_id, protocol)
    }

    #[tokio::test]
    async fn test_start_run() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        assert_eq!(run.protocol_id, protocol.id);
        assert_eq!(run.current_state, protocol.entry_state);
        assert_eq!(run.status, RunStatus::Running);
        assert!(run.plan_id.is_none());
        assert!(run.task_id.is_none());
        assert_eq!(run.states_visited.len(), 1);
        assert_eq!(run.states_visited[0].state_name, "Start");
        assert!(run.states_visited[0].trigger.is_none());

        // Verify it was persisted
        let loaded = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(loaded.id, run.id);
    }

    #[tokio::test]
    async fn test_start_run_with_context() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let plan_id = Some(Uuid::new_v4());
        let task_id = Some(Uuid::new_v4());

        let run = start_run(&store, protocol.id, plan_id, task_id, None)
            .await
            .unwrap();

        assert_eq!(run.plan_id, plan_id);
        assert_eq!(run.task_id, task_id);
    }

    #[tokio::test]
    async fn test_start_run_protocol_not_found() {
        let store = MockGraphStore::new();
        let result = start_run(&store, Uuid::new_v4(), None, None, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Protocol not found"));
    }

    #[tokio::test]
    async fn test_fire_transition_success() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        assert_eq!(run.states_visited.len(), 1);

        // Fire "begin" → should move to Processing
        let result = fire_transition(&store, run.id, "begin").await.unwrap();
        assert!(result.success);
        assert_eq!(result.current_state_name, "Processing");
        assert!(!result.run_completed);
        assert_eq!(result.status, RunStatus::Running);
        assert!(result.error.is_none());

        // Verify the run was updated in the store
        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.states_visited.len(), 2);
        assert_eq!(updated.states_visited[1].state_name, "Processing");
        assert_eq!(updated.states_visited[1].trigger, Some("begin".to_string()));
    }

    #[tokio::test]
    async fn test_fire_transition_to_terminal_completes_run() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Fire "begin" → Processing
        fire_transition(&store, run.id, "begin").await.unwrap();

        // Fire "finish" → Done (terminal) → should auto-complete
        let result = fire_transition(&store, run.id, "finish").await.unwrap();
        assert!(result.success);
        assert_eq!(result.current_state_name, "Done");
        assert!(result.run_completed);
        assert_eq!(result.status, RunStatus::Completed);

        // Verify the run is completed
        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, RunStatus::Completed);
        assert!(updated.completed_at.is_some());
        assert_eq!(updated.states_visited.len(), 3);
    }

    #[tokio::test]
    async fn test_fire_transition_invalid_trigger() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Fire "nonexistent" trigger from Start → should fail gracefully
        let result = fire_transition(&store, run.id, "nonexistent")
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("No transition found"));
    }

    #[tokio::test]
    async fn test_fire_transition_on_completed_run() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        fire_transition(&store, run.id, "begin").await.unwrap();
        fire_transition(&store, run.id, "finish").await.unwrap();

        // Run is now completed — trying to fire should fail
        let result = fire_transition(&store, run.id, "begin").await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not active"));
    }

    #[tokio::test]
    async fn test_cancel_run() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        let cancelled = cancel_run(&store, run.id).await.unwrap();

        assert_eq!(cancelled.status, RunStatus::Cancelled);
        assert!(cancelled.completed_at.is_some());

        // Verify persistence
        let loaded = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(loaded.status, RunStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_cancel_already_completed() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        fire_transition(&store, run.id, "begin").await.unwrap();
        fire_transition(&store, run.id, "finish").await.unwrap();

        // Already completed — cancel should fail
        let result = cancel_run(&store, run.id).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cannot cancel"));
    }

    #[tokio::test]
    async fn test_fail_run() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        let failed = fail_run(&store, run.id, "something went wrong")
            .await
            .unwrap();

        assert_eq!(failed.status, RunStatus::Failed);
        assert_eq!(failed.error, Some("something went wrong".to_string()));
        assert!(failed.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_list_protocol_runs() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        // Create 3 runs sequentially (complete/cancel the previous before starting next,
        // because the concurrency guard prevents multiple Running runs).
        let run1 = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        // Complete run1 → Done
        fire_transition(&store, run1.id, "begin").await.unwrap();
        fire_transition(&store, run1.id, "finish").await.unwrap();

        let run2 = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        // Cancel run2
        cancel_run(&store, run2.id).await.unwrap();

        let run3 = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        // run3 stays Running

        // List all
        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 3);
        assert_eq!(runs.len(), 3);

        // List only running
        let (running, running_total) = store
            .list_protocol_runs(protocol.id, Some(RunStatus::Running), 10, 0)
            .await
            .unwrap();
        assert_eq!(running_total, 1);
        assert_eq!(running[0].id, run3.id);

        // List only completed
        let (completed, completed_total) = store
            .list_protocol_runs(protocol.id, Some(RunStatus::Completed), 10, 0)
            .await
            .unwrap();
        assert_eq!(completed_total, 1);
        assert_eq!(completed[0].id, run1.id);

        // List only cancelled
        let (cancelled, cancelled_total) = store
            .list_protocol_runs(protocol.id, Some(RunStatus::Cancelled), 10, 0)
            .await
            .unwrap();
        assert_eq!(cancelled_total, 1);
        assert_eq!(cancelled[0].id, run2.id);
    }

    #[tokio::test]
    async fn test_concurrent_run_rejected() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        // Start a run — succeeds
        let _run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Try to start another while the first is Running — should fail
        let result = start_run(&store, protocol.id, None, None, None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("concurrent run already running"));
    }

    #[tokio::test]
    async fn test_concurrent_run_allowed_after_completion() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        // Start and complete a run
        let run1 = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        fire_transition(&store, run1.id, "begin").await.unwrap();
        fire_transition(&store, run1.id, "finish").await.unwrap();
        assert_eq!(
            store
                .get_protocol_run(run1.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            RunStatus::Completed
        );

        // Now starting another run should succeed
        let run2 = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        assert_eq!(run2.status, RunStatus::Running);
    }

    // ================================================================
    // Hierarchical FSM tests (T4.1)
    // ================================================================

    /// Helper: set up a parent protocol with a macro-state pointing to a child protocol.
    /// Parent: Start -> Macro(sub=child_protocol) -> Done
    /// Child: ChildStart -> ChildDone (terminal)
    async fn setup_hierarchical(store: &MockGraphStore) -> (Uuid, Protocol, Protocol) {
        let project_id = Uuid::new_v4();
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "test-hierarchy".to_string(),
            slug: "test-hierarchy".to_string(),
            description: None,
            root_path: "/tmp/test-h".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: true,
        };
        store.create_project(&project).await.unwrap();

        // Child protocol: ChildStart -> ChildDone
        let child_id = Uuid::new_v4();
        let child_start = ProtocolState::start(child_id, "ChildStart");
        let child_done = ProtocolState::terminal(child_id, "ChildDone");
        let mut child_proto = Protocol::new_full(
            project_id,
            "Child Protocol",
            "A child sub-protocol",
            child_start.id,
            vec![child_done.id],
            ProtocolCategory::Business,
        );
        child_proto.id = child_id;
        store.upsert_protocol(&child_proto).await.unwrap();
        store.upsert_protocol_state(&child_start).await.unwrap();
        store.upsert_protocol_state(&child_done).await.unwrap();
        let ct = ProtocolTransition::new(child_id, child_start.id, child_done.id, "child_finish");
        store.upsert_protocol_transition(&ct).await.unwrap();

        // Parent protocol: Start -> Macro(sub=child) -> Done
        let parent_id = Uuid::new_v4();
        let p_start = ProtocolState::start(parent_id, "Start");
        let mut p_macro = ProtocolState::new(parent_id, "Macro");
        p_macro.sub_protocol_id = Some(child_id);
        p_macro.completion_strategy = Some(crate::protocol::CompletionStrategy::AllComplete);
        let p_done = ProtocolState::terminal(parent_id, "Done");

        let mut parent_proto = Protocol::new_full(
            project_id,
            "Parent Protocol",
            "A parent with macro-state",
            p_start.id,
            vec![p_done.id],
            ProtocolCategory::Business,
        );
        parent_proto.id = parent_id;
        store.upsert_protocol(&parent_proto).await.unwrap();
        store.upsert_protocol_state(&p_start).await.unwrap();
        store.upsert_protocol_state(&p_macro).await.unwrap();
        store.upsert_protocol_state(&p_done).await.unwrap();

        let t1 = ProtocolTransition::new(parent_id, p_start.id, p_macro.id, "enter_macro");
        let t2 = ProtocolTransition::new(parent_id, p_macro.id, p_done.id, "child_completed");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        (project_id, parent_proto, child_proto)
    }

    #[tokio::test]
    async fn test_auto_spawn_child_run() {
        let store = MockGraphStore::new();
        let (_, parent_proto, child_proto) = setup_hierarchical(&store).await;

        // Start parent run
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();

        // Transition to macro-state → should auto-spawn child run
        let result = fire_transition(&store, parent_run.id, "enter_macro")
            .await
            .unwrap();
        assert!(result.success);
        assert_eq!(result.current_state_name, "Macro");
        assert!(!result.run_completed);

        // Verify child_run_id is set
        assert!(result.child_run_id.is_some());
        let child_run_id = result.child_run_id.unwrap();

        // Verify child run exists in store
        let child_run = store.get_protocol_run(child_run_id).await.unwrap().unwrap();
        assert_eq!(child_run.protocol_id, child_proto.id);
        assert_eq!(child_run.parent_run_id, Some(parent_run.id));
        assert_eq!(child_run.depth, 1);
        assert_eq!(child_run.status, RunStatus::Running);
        assert!(child_run.triggered_by.starts_with("hierarchy:parent_"));
    }

    #[tokio::test]
    async fn test_no_spawn_without_sub_protocol() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Transition to Processing (no sub_protocol_id) → no child spawn
        let result = fire_transition(&store, run.id, "begin").await.unwrap();
        assert!(result.success);
        assert!(result.child_run_id.is_none());
    }

    #[tokio::test]
    async fn test_max_depth_rejected() {
        let store = MockGraphStore::new();
        let (_, parent_proto, _) = setup_hierarchical(&store).await;

        // Start a root run
        let root_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();

        // Simulate a deeply nested parent by creating a fake parent run at depth 4
        let mut deep_parent = ProtocolRun::new(parent_proto.id, root_run.current_state, "Start");
        deep_parent.depth = MAX_HIERARCHY_DEPTH - 1; // depth 4
        deep_parent.parent_run_id = Some(root_run.id);

        let result = start_child_run(&store, &deep_parent, parent_proto.id).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Max hierarchy depth"), "Error: {err}");
    }

    #[tokio::test]
    async fn test_cycle_detection() {
        let store = MockGraphStore::new();
        let (_, parent_proto, _) = setup_hierarchical(&store).await;

        // Start a run for the parent protocol
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();

        // Try to spawn a child with the SAME protocol_id as parent → cycle!
        let result = start_child_run(&store, &parent_run, parent_proto.id).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Cycle detected"), "Error: {err}");
    }

    #[tokio::test]
    async fn test_full_lifecycle() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        // Start
        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        assert_eq!(run.status, RunStatus::Running);
        assert_eq!(run.states_visited.len(), 1);

        // Transition: Start -> Processing
        let r1 = fire_transition(&store, run.id, "begin").await.unwrap();
        assert!(r1.success);
        assert_eq!(r1.current_state_name, "Processing");
        assert!(!r1.run_completed);

        // Transition: Processing -> Done
        let r2 = fire_transition(&store, run.id, "finish").await.unwrap();
        assert!(r2.success);
        assert_eq!(r2.current_state_name, "Done");
        assert!(r2.run_completed);
        assert_eq!(r2.status, RunStatus::Completed);

        // Verify final state
        let final_run = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(final_run.status, RunStatus::Completed);
        assert!(final_run.completed_at.is_some());
        assert_eq!(final_run.states_visited.len(), 3);

        // Verify state visit history
        let names: Vec<_> = final_run
            .states_visited
            .iter()
            .map(|sv| sv.state_name.as_str())
            .collect();
        assert_eq!(names, vec!["Start", "Processing", "Done"]);
    }

    // ================================================================
    // Child completion tests (T4.2)
    // ================================================================

    #[tokio::test]
    async fn test_child_completion_no_parent() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        // Start and complete a root run (no parent)
        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        fire_transition(&store, run.id, "begin").await.unwrap();
        fire_transition(&store, run.id, "finish").await.unwrap();

        let completed = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(completed.status, RunStatus::Completed);

        // handle_child_completion should return None for a root run
        let result = handle_child_completion(&store, &completed).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_child_completion_all_complete() {
        let store = MockGraphStore::new();
        let (project_id, parent_proto, _child_proto) = setup_hierarchical(&store).await;

        // Create a second child protocol (different from the first to avoid concurrency guard)
        let child2_id = Uuid::new_v4();
        let c2_start = ProtocolState::start(child2_id, "C2Start");
        let c2_done = ProtocolState::terminal(child2_id, "C2Done");
        let mut child2_proto = Protocol::new_full(
            project_id,
            "Child Protocol 2",
            "Second child sub-protocol",
            c2_start.id,
            vec![c2_done.id],
            ProtocolCategory::Business,
        );
        child2_proto.id = child2_id;
        store.upsert_protocol(&child2_proto).await.unwrap();
        store.upsert_protocol_state(&c2_start).await.unwrap();
        store.upsert_protocol_state(&c2_done).await.unwrap();
        let c2t = ProtocolTransition::new(child2_id, c2_start.id, c2_done.id, "c2_finish");
        store.upsert_protocol_transition(&c2t).await.unwrap();

        // Start parent and transition to macro-state (spawns child 1)
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();
        let result = fire_transition(&store, parent_run.id, "enter_macro")
            .await
            .unwrap();
        assert!(result.success);
        let child1_id = result.child_run_id.unwrap();

        // Manually spawn a second child run (different protocol) to test AllComplete
        let parent_run_now = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        let child2 = start_child_run(&store, &parent_run_now, child2_proto.id)
            .await
            .unwrap();

        // Complete child 1 — fire_transition auto-calls handle_child_completion internally.
        // Parent should NOT transition yet (child2 still running).
        fire_transition(&store, child1_id, "child_finish")
            .await
            .unwrap();
        let child1_done = store.get_protocol_run(child1_id).await.unwrap().unwrap();
        assert_eq!(child1_done.status, RunStatus::Completed);

        // Verify parent is still Running at Macro
        let parent_check = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            parent_check.status,
            RunStatus::Running,
            "Parent should NOT transition when only 1 of 2 children is done"
        );

        // Complete child 2 — now ALL children done, fire_transition auto-transitions parent
        fire_transition(&store, child2.id, "c2_finish")
            .await
            .unwrap();
        let child2_done = store.get_protocol_run(child2.id).await.unwrap().unwrap();
        assert_eq!(child2_done.status, RunStatus::Completed);

        // Verify parent is now Completed (auto-transitioned by handle_child_completion)
        let parent_final = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            parent_final.status,
            RunStatus::Completed,
            "Parent should auto-transition when all children are done"
        );
        assert_eq!(
            parent_final.states_visited.last().unwrap().state_name,
            "Done"
        );
    }

    #[tokio::test]
    async fn test_child_completion_any_complete() {
        let store = MockGraphStore::new();
        let (project_id, _, child_proto) = setup_hierarchical(&store).await;

        // Set up a parent protocol with AnyComplete strategy
        let parent_id = Uuid::new_v4();
        let p_start = ProtocolState::start(parent_id, "Start");
        let mut p_macro = ProtocolState::new(parent_id, "Macro");
        p_macro.sub_protocol_id = Some(child_proto.id);
        p_macro.completion_strategy = Some(crate::protocol::CompletionStrategy::AnyComplete);
        let p_done = ProtocolState::terminal(parent_id, "Done");

        let mut parent_proto = Protocol::new_full(
            project_id,
            "AnyComplete Parent",
            "Parent with AnyComplete strategy",
            p_start.id,
            vec![p_done.id],
            ProtocolCategory::Business,
        );
        parent_proto.id = parent_id;
        store.upsert_protocol(&parent_proto).await.unwrap();
        store.upsert_protocol_state(&p_start).await.unwrap();
        store.upsert_protocol_state(&p_macro).await.unwrap();
        store.upsert_protocol_state(&p_done).await.unwrap();

        let t1 = ProtocolTransition::new(parent_id, p_start.id, p_macro.id, "enter_macro");
        let t2 = ProtocolTransition::new(parent_id, p_macro.id, p_done.id, "child_completed");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        // Start parent and transition to macro-state (spawns child 1)
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();
        let result = fire_transition(&store, parent_run.id, "enter_macro")
            .await
            .unwrap();
        assert!(result.success);
        let child1_id = result.child_run_id.unwrap();

        // Create a second child protocol to avoid concurrency guard
        let child2_id = Uuid::new_v4();
        let c2_start = ProtocolState::start(child2_id, "C2Start");
        let c2_done = ProtocolState::terminal(child2_id, "C2Done");
        let mut child2_proto = Protocol::new_full(
            project_id,
            "Child Protocol 2",
            "Second child",
            c2_start.id,
            vec![c2_done.id],
            ProtocolCategory::Business,
        );
        child2_proto.id = child2_id;
        store.upsert_protocol(&child2_proto).await.unwrap();
        store.upsert_protocol_state(&c2_start).await.unwrap();
        store.upsert_protocol_state(&c2_done).await.unwrap();
        let c2t = ProtocolTransition::new(child2_id, c2_start.id, c2_done.id, "c2_finish");
        store.upsert_protocol_transition(&c2t).await.unwrap();

        // Spawn a second child (different protocol)
        let parent_run_now = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        let _child2 = start_child_run(&store, &parent_run_now, child2_proto.id)
            .await
            .unwrap();

        // Complete only child 1 — with AnyComplete, fire_transition auto-transitions parent
        fire_transition(&store, child1_id, "child_finish")
            .await
            .unwrap();
        let child1_done = store.get_protocol_run(child1_id).await.unwrap().unwrap();
        assert_eq!(child1_done.status, RunStatus::Completed);

        // Verify parent is now Completed (AnyComplete transitions on first child)
        let parent_final = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            parent_final.status,
            RunStatus::Completed,
            "AnyComplete: parent should auto-transition on first child completion"
        );
        assert_eq!(
            parent_final.states_visited.last().unwrap().state_name,
            "Done"
        );
    }

    #[tokio::test]
    async fn test_child_completion_manual() {
        let store = MockGraphStore::new();
        let (project_id, _, child_proto) = setup_hierarchical(&store).await;

        // Set up a parent protocol with Manual strategy
        let parent_id = Uuid::new_v4();
        let p_start = ProtocolState::start(parent_id, "Start");
        let mut p_macro = ProtocolState::new(parent_id, "Macro");
        p_macro.sub_protocol_id = Some(child_proto.id);
        p_macro.completion_strategy = Some(crate::protocol::CompletionStrategy::Manual);
        let p_done = ProtocolState::terminal(parent_id, "Done");

        let mut parent_proto = Protocol::new_full(
            project_id,
            "Manual Parent",
            "Parent with Manual strategy",
            p_start.id,
            vec![p_done.id],
            ProtocolCategory::Business,
        );
        parent_proto.id = parent_id;
        store.upsert_protocol(&parent_proto).await.unwrap();
        store.upsert_protocol_state(&p_start).await.unwrap();
        store.upsert_protocol_state(&p_macro).await.unwrap();
        store.upsert_protocol_state(&p_done).await.unwrap();

        let t1 = ProtocolTransition::new(parent_id, p_start.id, p_macro.id, "enter_macro");
        let t2 = ProtocolTransition::new(parent_id, p_macro.id, p_done.id, "child_completed");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        // Start parent and transition to macro-state (spawns child)
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();
        let result = fire_transition(&store, parent_run.id, "enter_macro")
            .await
            .unwrap();
        assert!(result.success);
        let child_id = result.child_run_id.unwrap();

        // Complete the child — fire_transition auto-calls handle_child_completion
        fire_transition(&store, child_id, "child_finish")
            .await
            .unwrap();
        let child_done = store.get_protocol_run(child_id).await.unwrap().unwrap();
        assert_eq!(child_done.status, RunStatus::Completed);

        // Manual strategy: parent should NOT auto-transition.
        // Verify parent is still Running
        let parent_check = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(parent_check.status, RunStatus::Running);
    }

    #[tokio::test]
    async fn test_child_failure_abort() {
        let store = MockGraphStore::new();
        let (_, parent_proto, _child_proto) = setup_hierarchical(&store).await;

        // Start parent and transition to macro-state (spawns child)
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();
        let result = fire_transition(&store, parent_run.id, "enter_macro")
            .await
            .unwrap();
        assert!(result.success);
        let child_run_id = result.child_run_id.unwrap();

        // Fail the child run — default on_failure_strategy is Abort
        fail_run(&store, child_run_id, "child error").await.unwrap();

        // Verify parent is now Failed
        let parent_final = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            parent_final.status,
            RunStatus::Failed,
            "Abort strategy: parent should fail when child fails"
        );
        assert!(parent_final
            .error
            .as_ref()
            .unwrap()
            .contains("Child run failed"));
    }

    #[tokio::test]
    async fn test_child_failure_skip() {
        let store = MockGraphStore::new();
        let (project_id, _, child_proto) = setup_hierarchical(&store).await;

        // Set up a parent protocol with Skip on_failure_strategy
        // and a "child_skipped" transition from Macro to Done
        let parent_id = Uuid::new_v4();
        let p_start = ProtocolState::start(parent_id, "Start");
        let mut p_macro = ProtocolState::new(parent_id, "Macro");
        p_macro.sub_protocol_id = Some(child_proto.id);
        p_macro.completion_strategy = Some(crate::protocol::CompletionStrategy::AllComplete);
        p_macro.on_failure_strategy = Some(OnFailureStrategy::Skip);
        let p_done = ProtocolState::terminal(parent_id, "Done");

        let mut parent_proto = Protocol::new_full(
            project_id,
            "Skip Parent",
            "Parent with Skip failure strategy",
            p_start.id,
            vec![p_done.id],
            ProtocolCategory::Business,
        );
        parent_proto.id = parent_id;
        store.upsert_protocol(&parent_proto).await.unwrap();
        store.upsert_protocol_state(&p_start).await.unwrap();
        store.upsert_protocol_state(&p_macro).await.unwrap();
        store.upsert_protocol_state(&p_done).await.unwrap();

        let t1 = ProtocolTransition::new(parent_id, p_start.id, p_macro.id, "enter_macro");
        let t2 = ProtocolTransition::new(parent_id, p_macro.id, p_done.id, "child_completed");
        let t3 = ProtocolTransition::new(parent_id, p_macro.id, p_done.id, "child_skipped");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();
        store.upsert_protocol_transition(&t3).await.unwrap();

        // Start parent and transition to macro-state (spawns child)
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();
        let result = fire_transition(&store, parent_run.id, "enter_macro")
            .await
            .unwrap();
        assert!(result.success);
        let child_run_id = result.child_run_id.unwrap();

        // Fail the child run — Skip strategy should fire "child_skipped" on parent
        fail_run(&store, child_run_id, "child error").await.unwrap();

        // Verify parent has transitioned to Done via child_skipped
        let parent_final = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            parent_final.status,
            RunStatus::Completed,
            "Skip strategy: parent should transition to Done via child_skipped"
        );
        assert_eq!(
            parent_final.states_visited.last().unwrap().state_name,
            "Done"
        );
    }

    #[tokio::test]
    async fn test_child_failure_retry() {
        let store = MockGraphStore::new();
        let (project_id, _, child_proto) = setup_hierarchical(&store).await;

        // Set up a parent protocol with Retry{max:2} on_failure_strategy
        let parent_id = Uuid::new_v4();
        let p_start = ProtocolState::start(parent_id, "Start");
        let mut p_macro = ProtocolState::new(parent_id, "Macro");
        p_macro.sub_protocol_id = Some(child_proto.id);
        p_macro.completion_strategy = Some(crate::protocol::CompletionStrategy::AllComplete);
        p_macro.on_failure_strategy = Some(OnFailureStrategy::Retry { max: 2 });
        let p_done = ProtocolState::terminal(parent_id, "Done");

        let mut parent_proto = Protocol::new_full(
            project_id,
            "Retry Parent",
            "Parent with Retry failure strategy",
            p_start.id,
            vec![p_done.id],
            ProtocolCategory::Business,
        );
        parent_proto.id = parent_id;
        store.upsert_protocol(&parent_proto).await.unwrap();
        store.upsert_protocol_state(&p_start).await.unwrap();
        store.upsert_protocol_state(&p_macro).await.unwrap();
        store.upsert_protocol_state(&p_done).await.unwrap();

        let t1 = ProtocolTransition::new(parent_id, p_start.id, p_macro.id, "enter_macro");
        let t2 = ProtocolTransition::new(parent_id, p_macro.id, p_done.id, "child_completed");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        // Start parent and transition to macro-state (spawns child 1)
        let parent_run = start_run(&store, parent_proto.id, None, None, None)
            .await
            .unwrap();
        let result = fire_transition(&store, parent_run.id, "enter_macro")
            .await
            .unwrap();
        assert!(result.success);
        let child1_id = result.child_run_id.unwrap();

        // Fail child 1 — should spawn a retry (child 2)
        fail_run(&store, child1_id, "attempt 1 failed")
            .await
            .unwrap();

        // Parent should still be running
        let parent_check = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(parent_check.status, RunStatus::Running);

        // There should be a new child run (the retry)
        let children = store.list_child_runs(parent_run.id).await.unwrap();
        assert_eq!(
            children.len(),
            2,
            "Should have 2 children after first retry"
        );
        let child2 = children
            .iter()
            .find(|r| r.status == RunStatus::Running)
            .unwrap();

        // Fail child 2 — should spawn another retry (child 3)
        fail_run(&store, child2.id, "attempt 2 failed")
            .await
            .unwrap();

        // Parent should still be running
        let parent_check = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(parent_check.status, RunStatus::Running);

        let children = store.list_child_runs(parent_run.id).await.unwrap();
        assert_eq!(
            children.len(),
            3,
            "Should have 3 children after second retry"
        );
        let child3 = children
            .iter()
            .find(|r| r.status == RunStatus::Running)
            .unwrap();

        // Fail child 3 — max retries (2) exceeded, parent should abort
        fail_run(&store, child3.id, "attempt 3 failed")
            .await
            .unwrap();

        // Parent should now be Failed
        let parent_final = store
            .get_protocol_run(parent_run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            parent_final.status,
            RunStatus::Failed,
            "Retry strategy: parent should fail after max retries exceeded"
        );
        assert!(parent_final
            .error
            .as_ref()
            .unwrap()
            .contains("after 2 retries"));
    }

    #[tokio::test]
    async fn test_fire_transition_closes_previous_state_visit() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Fire "begin" → Processing
        fire_transition(&store, run.id, "begin").await.unwrap();

        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        // First state (Start) should have exited_at and duration_ms filled
        assert!(
            updated.states_visited[0].exited_at.is_some(),
            "Start state should have exited_at after transition"
        );
        assert!(
            updated.states_visited[0].duration_ms.is_some(),
            "Start state should have duration_ms after transition"
        );
        // Second state (Processing) should still be open
        assert!(updated.states_visited[1].exited_at.is_none());
    }

    #[tokio::test]
    async fn test_fire_transition_to_terminal_closes_all_states() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        fire_transition(&store, run.id, "begin").await.unwrap();
        fire_transition(&store, run.id, "finish").await.unwrap();

        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, RunStatus::Completed);
        // ALL state visits should be closed
        for (i, sv) in updated.states_visited.iter().enumerate() {
            assert!(
                sv.exited_at.is_some(),
                "State visit {} ({}) should have exited_at",
                i,
                sv.state_name
            );
        }
    }

    #[tokio::test]
    async fn test_cas_version_increments_on_transition() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        assert_eq!(run.version, 0);

        fire_transition(&store, run.id, "begin").await.unwrap();

        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert!(
            updated.version > 0,
            "Version should have incremented after transition"
        );
    }

    #[tokio::test]
    async fn test_cas_rejects_stale_version() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        let run = start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Manually create a stale copy with wrong version
        let mut stale_run = run.clone();
        // First, do a real transition to bump the version in the store
        fire_transition(&store, run.id, "begin").await.unwrap();

        // Now try to update with the stale version (0)
        stale_run.status = RunStatus::Failed;
        let result = store.update_protocol_run(&mut stale_run).await;
        assert!(result.is_err(), "Should reject stale version");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("OptimisticLockError"),
            "Error should mention OptimisticLockError"
        );
    }

    #[tokio::test]
    async fn test_get_protocol_by_name_and_project_found() {
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_protocol(&store).await;

        let result = store
            .get_protocol_by_name_and_project(&protocol.name, project_id)
            .await
            .unwrap();
        assert_eq!(result, Some(protocol.id));
    }

    #[tokio::test]
    async fn test_get_protocol_by_name_and_project_not_found() {
        let store = MockGraphStore::new();
        let (project_id, _) = setup_protocol(&store).await;

        let result = store
            .get_protocol_by_name_and_project("nonexistent", project_id)
            .await
            .unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_get_protocol_by_name_wrong_project() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_protocol(&store).await;

        // Different project ID
        let result = store
            .get_protocol_by_name_and_project(&protocol.name, Uuid::new_v4())
            .await
            .unwrap();
        assert_eq!(result, None);
    }
}
