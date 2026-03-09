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
use crate::protocol::{ProtocolRun, RunStatus, TransitionResult};
use anyhow::{bail, Context, Result};
use uuid::Uuid;

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

/// Fire a transition on a running protocol.
///
/// Finds a matching transition from the current state with the given trigger,
/// advances the run to the target state, and auto-completes if the target
/// is a terminal state.
///
/// # Returns
/// A [`TransitionResult`] indicating success/failure and the new state.
///
/// # Errors
/// - Run not found
/// - Run is not active (already completed/failed/cancelled)
/// - No matching transition for the given trigger from current state
pub async fn fire_transition(
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
        .update_protocol_run(&run)
        .await
        .context("Failed to update protocol run after transition")?;

    Ok(TransitionResult {
        success: true,
        current_state: run.current_state,
        current_state_name: target_state.name.clone(),
        run_completed: is_terminal,
        status: run.status,
        error: None,
    })
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

    run.cancel();

    store
        .update_protocol_run(&run)
        .await
        .context("Failed to update protocol run after cancel")?;

    Ok(run)
}

/// Fail an active protocol run with an error message.
///
/// # Errors
/// - Run not found
/// - Run is not active
pub async fn fail_run(store: &dyn GraphStore, run_id: Uuid, error: &str) -> Result<ProtocolRun> {
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
        .update_protocol_run(&run)
        .await
        .context("Failed to update protocol run after fail")?;

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
            scaffolding_override: None,
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
}
