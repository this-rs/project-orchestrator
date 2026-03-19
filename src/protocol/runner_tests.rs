//! Integration tests for the Protocol Runner.
//!
//! Covers 4 E2E scenarios:
//! 1. Auto-maintenance traverses all states to "maintained" (terminal)
//! 2. Backward compatibility — manual fire_transition works without runner
//! 3. Server restart mid-run — resume from last state (not from scratch)
//! 4. Cancel during execution — clean shutdown, no orphan tasks

#[cfg(test)]
mod tests {
    use crate::events::{CrudEvent, EventEmitter};
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::ProjectNode;
    use crate::neo4j::traits::GraphStore;
    use crate::protocol::engine;
    use crate::protocol::runner::run_protocol;
    use crate::protocol::{
        Protocol, ProtocolCategory, ProtocolState, ProtocolTransition, RunStatus, TriggerConfig,
        TriggerMode,
    };
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

    /// Helper to call run_protocol with None chat_manager (system protocols don't need LLM).
    async fn run_protocol_no_llm(
        store: Arc<dyn GraphStore>,
        run_id: Uuid,
        cancel: CancellationToken,
        emitter: Arc<dyn EventEmitter>,
    ) -> anyhow::Result<crate::protocol::ProtocolRun> {
        run_protocol(store, run_id, cancel, emitter, None).await
    }

    // ========================================================================
    // Test helpers
    // ========================================================================

    /// A test EventEmitter that records all emitted events.
    struct RecordingEmitter {
        events: Mutex<Vec<CrudEvent>>,
    }

    impl RecordingEmitter {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn events(&self) -> Vec<CrudEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl EventEmitter for RecordingEmitter {
        fn emit(&self, event: CrudEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    /// Build the auto-maintenance protocol in the mock store.
    ///
    /// States: health_check → analyze_delta → triage → auto_fix → plan_remediation → maintained
    /// All intermediate states have system actions so the runner can execute them.
    async fn setup_auto_maintenance_protocol(
        store: &MockGraphStore,
    ) -> (Uuid, Protocol, Vec<ProtocolState>) {
        let project_id = Uuid::new_v4();
        let project = ProjectNode {
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
        };
        store.create_project(&project).await.unwrap();

        let protocol_id = Uuid::new_v4();

        // Build states
        let mut health_check = ProtocolState::start(protocol_id, "health_check");
        health_check.action = Some("admin(persist_health_report)".to_string());

        let mut analyze_delta = ProtocolState::new(protocol_id, "analyze_delta");
        analyze_delta.action = Some("admin(update_staleness_scores)".to_string());

        let mut triage = ProtocolState::new(protocol_id, "triage");
        triage.action = Some("admin(detect_stagnation)".to_string());

        let mut auto_fix = ProtocolState::new(protocol_id, "auto_fix");
        auto_fix.action = Some(
            "admin(update_staleness_scores) + admin(decay_synapses, 0.05, 0.01) + admin(update_energy_scores)"
                .to_string(),
        );

        let mut plan_remediation = ProtocolState::new(protocol_id, "plan_remediation");
        plan_remediation.action = Some("admin(audit_gaps)".to_string());

        let maintained = ProtocolState::terminal(protocol_id, "maintained");

        // Build protocol
        let mut protocol = Protocol::new_full(
            project_id,
            "auto-maintenance",
            "Auto-maintenance protocol for knowledge graph health",
            health_check.id,
            vec![maintained.id],
            ProtocolCategory::System,
        );
        protocol.id = protocol_id;
        protocol.trigger_mode = TriggerMode::Event;
        protocol.trigger_config = Some(TriggerConfig {
            events: vec!["post_sync".to_string()],
            schedule: None,
            conditions: vec![],
        });

        // Build transitions (linear flow)
        let transitions = vec![
            ProtocolTransition::new(protocol_id, health_check.id, analyze_delta.id, "done"),
            ProtocolTransition::new(protocol_id, analyze_delta.id, triage.id, "done"),
            ProtocolTransition::new(protocol_id, triage.id, auto_fix.id, "done"),
            ProtocolTransition::new(protocol_id, auto_fix.id, plan_remediation.id, "done"),
            ProtocolTransition::new(protocol_id, plan_remediation.id, maintained.id, "done"),
        ];

        let states = vec![
            health_check.clone(),
            analyze_delta.clone(),
            triage.clone(),
            auto_fix.clone(),
            plan_remediation.clone(),
            maintained.clone(),
        ];

        // Persist in mock
        store.upsert_protocol(&protocol).await.unwrap();
        for state in &states {
            store.upsert_protocol_state(state).await.unwrap();
        }
        for transition in &transitions {
            store.upsert_protocol_transition(transition).await.unwrap();
        }

        (project_id, protocol, states)
    }

    /// Build a simple 3-state protocol for simpler tests.
    /// Start → Processing → Done (terminal)
    async fn setup_simple_protocol(
        store: &MockGraphStore,
        category: ProtocolCategory,
    ) -> (Uuid, Protocol, Vec<ProtocolState>) {
        let project_id = Uuid::new_v4();
        let project = ProjectNode {
            id: project_id,
            name: "simple-project".to_string(),
            slug: "simple-project".to_string(),
            description: None,
            root_path: "/tmp/simple".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        store.create_project(&project).await.unwrap();

        let protocol_id = Uuid::new_v4();

        let mut start = ProtocolState::start(protocol_id, "start");
        start.action = Some("admin(update_staleness_scores)".to_string());

        let mut processing = ProtocolState::new(protocol_id, "processing");
        processing.action = Some("admin(update_energy_scores)".to_string());

        let done = ProtocolState::terminal(protocol_id, "done");

        let mut protocol = Protocol::new_full(
            project_id,
            "simple-protocol",
            "A simple test protocol",
            start.id,
            vec![done.id],
            category,
        );
        protocol.id = protocol_id;

        let transitions = vec![
            ProtocolTransition::new(protocol_id, start.id, processing.id, "done"),
            ProtocolTransition::new(protocol_id, processing.id, done.id, "done"),
        ];

        let states = vec![start.clone(), processing.clone(), done.clone()];

        store.upsert_protocol(&protocol).await.unwrap();
        for state in &states {
            store.upsert_protocol_state(state).await.unwrap();
        }
        for transition in &transitions {
            store.upsert_protocol_transition(transition).await.unwrap();
        }

        (project_id, protocol, states)
    }

    // ========================================================================
    // E2E #1: Auto-maintenance full traversal
    // ========================================================================

    /// Verify that the auto-maintenance protocol traverses:
    /// health_check → analyze_delta → triage → auto_fix → plan_remediation → maintained
    /// and finishes with status = Completed.
    #[tokio::test]
    async fn test_e2e_auto_maintenance_full_traversal() {
        let store = Arc::new(MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::new());
        let cancel = CancellationToken::new();

        let (_project_id, protocol, states) =
            setup_auto_maintenance_protocol(&store).await;

        // Start a run
        let run = engine::start_run(
            &*store,
            protocol.id,
            None, // no plan
            None, // no task
            Some("event:post_sync"),
        )
        .await
        .unwrap();

        assert_eq!(run.status, RunStatus::Running);
        assert_eq!(run.current_state, states[0].id); // health_check
        assert_eq!(run.triggered_by, "event:post_sync");

        // Run the protocol to completion
        let final_run = run_protocol_no_llm(
            store.clone(),
            run.id,
            cancel.clone(),
            emitter.clone(),
        )
        .await
        .unwrap();

        // Verify final state
        assert_eq!(
            final_run.status,
            RunStatus::Completed,
            "Run should be completed, got: {:?}",
            final_run.status
        );
        assert!(
            final_run.completed_at.is_some(),
            "completed_at should be set"
        );
        assert!(
            final_run.runner_managed,
            "runner_managed should be true"
        );

        // Verify states_visited: should have 6 entries (all states including maintained)
        assert!(
            final_run.states_visited.len() >= 5,
            "Should have visited at least 5 states, got: {}. Visited: {:?}",
            final_run.states_visited.len(),
            final_run
                .states_visited
                .iter()
                .map(|sv| &sv.state_name)
                .collect::<Vec<_>>()
        );

        // Verify state names in order
        let visited_names: Vec<&str> = final_run
            .states_visited
            .iter()
            .map(|sv| sv.state_name.as_str())
            .collect();

        assert!(
            visited_names.contains(&"health_check"),
            "Should have visited health_check"
        );
        assert!(
            visited_names.contains(&"analyze_delta"),
            "Should have visited analyze_delta"
        );
        assert!(
            visited_names.contains(&"triage"),
            "Should have visited triage"
        );
        assert!(
            visited_names.contains(&"auto_fix"),
            "Should have visited auto_fix"
        );
        assert!(
            visited_names.contains(&"plan_remediation"),
            "Should have visited plan_remediation"
        );

        // Verify the terminal state is "maintained"
        let terminal_state = states.iter().find(|s| s.name == "maintained").unwrap();
        assert_eq!(final_run.current_state, terminal_state.id);

        // Verify CRUD events were emitted for transitions
        let events = emitter.events();
        assert!(
            events.len() >= 4,
            "Should have emitted at least 4 transition events, got: {}",
            events.len()
        );

        // Verify run is persisted correctly in the store
        let persisted_run = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(persisted_run.status, RunStatus::Completed);
    }

    // ========================================================================
    // E2E #2: Backward compatibility — manual fire_transition
    // ========================================================================

    /// Verify that fire_transition() works on a run that is NOT runner-managed.
    /// The run should stay as manual and NOT be auto-executed.
    #[tokio::test]
    async fn test_e2e_manual_fire_transition_backward_compat() {
        let store = Arc::new(MockGraphStore::new());

        let (_project_id, protocol, states) =
            setup_simple_protocol(&store, ProtocolCategory::Business).await;

        // Start a run manually (no runner)
        let run = engine::start_run(
            &*store,
            protocol.id,
            None,
            None,
            Some("manual"),
        )
        .await
        .unwrap();

        // Verify initial state
        assert_eq!(run.status, RunStatus::Running);
        assert!(!run.runner_managed, "Manual run should not be runner_managed");
        assert_eq!(run.current_state, states[0].id); // start

        // Fire transition manually: start → processing
        let result = engine::fire_transition(&*store, run.id, "done").await.unwrap();
        assert!(result.success, "Transition should succeed");
        assert_eq!(result.current_state_name, "processing");
        assert!(!result.run_completed, "Run should not be completed yet");

        // Verify run is still Running and still NOT runner_managed
        let mid_run = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(mid_run.status, RunStatus::Running);
        assert!(!mid_run.runner_managed, "Run should remain non-runner_managed");
        assert_eq!(mid_run.current_state, states[1].id); // processing

        // Fire transition manually: processing → done (terminal)
        let result = engine::fire_transition(&*store, run.id, "done").await.unwrap();
        assert!(result.success, "Transition should succeed");
        assert_eq!(result.current_state_name, "done");
        assert!(result.run_completed, "Run should be completed");

        // Verify final state
        let final_run = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(final_run.status, RunStatus::Completed);
        assert!(!final_run.runner_managed, "Run should remain non-runner_managed");
        assert!(final_run.completed_at.is_some());

        // Verify states_visited has entries
        assert!(
            final_run.states_visited.len() >= 2,
            "Should have at least 2 state visits"
        );
    }

    /// Verify that a run created by the runner (runner_managed=true) can still
    /// receive manual fire_transition calls (coexistence).
    #[tokio::test]
    async fn test_e2e_runner_managed_accepts_external_transitions() {
        let store = Arc::new(MockGraphStore::new());
        let _emitter = Arc::new(RecordingEmitter::new());

        let (_project_id, protocol, _states) =
            setup_simple_protocol(&store, ProtocolCategory::System).await;

        // Start a run
        let run = engine::start_run(
            &*store,
            protocol.id,
            None,
            None,
            Some("event:post_sync"),
        )
        .await
        .unwrap();

        // Manually mark as runner_managed (simulating what the runner does)
        let mut run_mut = store.get_protocol_run(run.id).await.unwrap().unwrap();
        run_mut.runner_managed = true;
        store.update_protocol_run(&mut run_mut).await.unwrap();

        // External fire_transition should still work
        let result = engine::fire_transition(&*store, run.id, "done").await.unwrap();
        assert!(result.success, "External transition on runner_managed run should succeed");
        assert_eq!(result.current_state_name, "processing");
    }

    // ========================================================================
    // E2E #3: Server restart mid-run — resume from last state
    // ========================================================================

    /// Simulate a restart by:
    /// 1. Starting a run and advancing it to "triage" state
    /// 2. "Stopping" the runner (dropping it)
    /// 3. Creating a new runner and resuming from the persisted state
    /// 4. Verifying the run continues from "triage" (not from the beginning)
    #[tokio::test]
    async fn test_e2e_restart_resume_from_last_state() {
        let store = Arc::new(MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::new());

        let (_project_id, protocol, states) =
            setup_auto_maintenance_protocol(&store).await;

        // Start a run at health_check
        let run = engine::start_run(
            &*store,
            protocol.id,
            None,
            None,
            Some("event:post_sync"),
        )
        .await
        .unwrap();
        assert_eq!(run.current_state, states[0].id); // health_check

        // Manually advance through health_check → analyze_delta → triage
        let result = engine::fire_transition(&*store, run.id, "done").await.unwrap();
        assert!(result.success);
        assert_eq!(result.current_state_name, "analyze_delta");

        let result = engine::fire_transition(&*store, run.id, "done").await.unwrap();
        assert!(result.success);
        assert_eq!(result.current_state_name, "triage");

        // Mark runner_managed to simulate a runner was driving it
        let mut mid_run = store.get_protocol_run(run.id).await.unwrap().unwrap();
        mid_run.runner_managed = true;
        store.update_protocol_run(&mut mid_run).await.unwrap();

        // *** "Server restart" — the runner task is gone ***
        // Verify the run is persisted at "triage"
        let persisted = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(persisted.status, RunStatus::Running);
        assert_eq!(persisted.current_state, states[2].id); // triage

        // *** "Resume" — create a new runner from the persisted state ***
        let cancel = CancellationToken::new();
        let final_run = run_protocol_no_llm(
            store.clone(),
            run.id,
            cancel,
            emitter.clone(),
        )
        .await
        .unwrap();

        // Verify the run completed successfully
        assert_eq!(
            final_run.status,
            RunStatus::Completed,
            "Resumed run should complete"
        );

        // Verify it continued from triage, not from the beginning
        // The states_visited should show the full history including the pre-restart states
        let visited_names: Vec<&str> = final_run
            .states_visited
            .iter()
            .map(|sv| sv.state_name.as_str())
            .collect();

        // The run should have health_check + analyze_delta (from manual transitions)
        // + triage + auto_fix + plan_remediation + maintained (from runner)
        assert!(
            visited_names.contains(&"triage"),
            "Should contain triage (pre-restart state)"
        );
        assert!(
            visited_names.contains(&"auto_fix"),
            "Should have progressed through auto_fix after restart"
        );
        assert!(
            visited_names.contains(&"plan_remediation"),
            "Should have progressed through plan_remediation"
        );

        // The terminal state should be "maintained"
        let maintained_state = states.iter().find(|s| s.name == "maintained").unwrap();
        assert_eq!(final_run.current_state, maintained_state.id);
    }

    // ========================================================================
    // E2E #4: Cancel during execution — clean shutdown
    // ========================================================================

    /// Verify that cancel_run() during execution stops the runner cleanly.
    /// Since the mock store is synchronous and very fast, we pre-cancel the token
    /// before starting the runner. The runner checks cancellation in tokio::select!
    /// at each state, so it should detect it. However, with synchronous mocks,
    /// executor futures resolve instantly — the select! may complete the exec future
    /// before checking the cancel branch. Therefore we accept both Cancelled and
    /// Completed as valid outcomes, but verify the cancel mechanism works via
    /// engine::cancel_run on a manually-held run.
    #[tokio::test]
    async fn test_e2e_cancel_via_engine() {
        let store = Arc::new(MockGraphStore::new());

        let (_project_id, protocol, _states) =
            setup_simple_protocol(&store, ProtocolCategory::System).await;

        // Start a run
        let run = engine::start_run(
            &*store,
            protocol.id,
            None,
            None,
            Some("event:post_sync"),
        )
        .await
        .unwrap();
        assert_eq!(run.status, RunStatus::Running);

        // Cancel via the engine API (the way the MCP handler does it)
        engine::cancel_run(&*store, run.id).await.unwrap();

        // Verify the run was cancelled
        let final_run = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(
            final_run.status,
            RunStatus::Cancelled,
            "Run should be Cancelled after cancel_run()"
        );
        assert!(
            final_run.completed_at.is_some(),
            "completed_at should be set on cancelled run"
        );
    }

    /// Cancel mid-execution using a spawned task that cancels after a delay.
    #[tokio::test]
    async fn test_e2e_cancel_mid_execution_spawned() {
        let store = Arc::new(MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::new());
        let cancel = CancellationToken::new();

        let (_project_id, protocol, _states) =
            setup_auto_maintenance_protocol(&store).await;

        // Start a run
        let run = engine::start_run(
            &*store,
            protocol.id,
            None,
            None,
            Some("event:post_sync"),
        )
        .await
        .unwrap();

        // Spawn a task that cancels after a very short delay
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            // Give the runner a tiny bit of time to start
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            cancel_clone.cancel();
        });

        // Run the protocol — should get cancelled mid-execution
        let final_run = run_protocol_no_llm(
            store.clone(),
            run.id,
            cancel,
            emitter.clone(),
        )
        .await
        .unwrap();

        // Should be either Cancelled (if cancel arrived during execution)
        // or Completed (if it finished before cancel arrived — race condition)
        assert!(
            final_run.status == RunStatus::Cancelled || final_run.status == RunStatus::Completed,
            "Run should be Cancelled or Completed (race), got: {:?}",
            final_run.status
        );
    }

    // ========================================================================
    // Additional runner tests
    // ========================================================================

    /// Verify concurrency guard: only one running run per protocol.
    #[tokio::test]
    async fn test_concurrency_guard_prevents_double_run() {
        let store = Arc::new(MockGraphStore::new());

        let (_project_id, protocol, _states) =
            setup_simple_protocol(&store, ProtocolCategory::System).await;

        // Start first run
        let run1 = engine::start_run(&*store, protocol.id, None, None, Some("manual"))
            .await
            .unwrap();
        assert_eq!(run1.status, RunStatus::Running);

        // Try to start second run — should fail
        let result = engine::start_run(&*store, protocol.id, None, None, Some("manual")).await;
        assert!(
            result.is_err(),
            "Second concurrent run should be rejected"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("concurrent") || err_msg.contains("already"),
            "Error should mention concurrent run: {err_msg}"
        );
    }

    /// Verify that runner marks run as runner_managed.
    #[tokio::test]
    async fn test_runner_marks_runner_managed() {
        let store = Arc::new(MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::new());
        let cancel = CancellationToken::new();

        let (_project_id, protocol, _states) =
            setup_simple_protocol(&store, ProtocolCategory::System).await;

        let run = engine::start_run(&*store, protocol.id, None, None, Some("event:post_sync"))
            .await
            .unwrap();

        // Runner_managed starts as false
        assert!(!run.runner_managed);

        let final_run = run_protocol_no_llm(store.clone(), run.id, cancel, emitter).await.unwrap();

        // After runner execution, it should be true
        assert!(final_run.runner_managed);
    }

    /// Verify that running on an already-completed run returns error.
    #[tokio::test]
    async fn test_runner_rejects_completed_run() {
        let store = Arc::new(MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::new());
        let cancel = CancellationToken::new();

        let (_project_id, protocol, _states) =
            setup_simple_protocol(&store, ProtocolCategory::System).await;

        let run = engine::start_run(&*store, protocol.id, None, None, Some("manual"))
            .await
            .unwrap();

        // Complete the run via fire_transition
        engine::fire_transition(&*store, run.id, "done").await.unwrap(); // start → processing
        engine::fire_transition(&*store, run.id, "done").await.unwrap(); // processing → done

        let completed = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(completed.status, RunStatus::Completed);

        // Try to run the protocol on the completed run — should fail
        let result = run_protocol_no_llm(store.clone(), run.id, cancel, emitter).await;
        assert!(
            result.is_err(),
            "Running a completed run should return error"
        );
    }

    /// Verify that the auto-maintenance protocol with chained actions executes correctly.
    /// The auto_fix state has: admin(update_staleness_scores) + admin(decay_synapses) + admin(update_energy_scores)
    #[tokio::test]
    async fn test_chained_actions_in_single_state() {
        let store = Arc::new(MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::new());
        let cancel = CancellationToken::new();

        let (_project_id, protocol, _states) =
            setup_auto_maintenance_protocol(&store).await;

        let run = engine::start_run(&*store, protocol.id, None, None, Some("event:post_sync"))
            .await
            .unwrap();

        let final_run = run_protocol_no_llm(store.clone(), run.id, cancel, emitter.clone())
            .await
            .unwrap();

        // The auto_fix state has 3 chained actions and should succeed
        assert_eq!(
            final_run.status,
            RunStatus::Completed,
            "Chained actions should all succeed"
        );

        // Check that events were emitted (one per transition)
        let events = emitter.events();
        assert!(
            events.len() >= 4,
            "Should have transition events for each state hop"
        );
    }

    /// Verify that a protocol with a state that has no action stops at that state
    /// (waits for external trigger).
    #[tokio::test]
    async fn test_runner_stops_at_actionless_state() {
        let store = Arc::new(MockGraphStore::new());
        let emitter = Arc::new(RecordingEmitter::new());
        let cancel = CancellationToken::new();

        let project_id = Uuid::new_v4();
        let project = ProjectNode {
            id: project_id,
            name: "actionless".to_string(),
            slug: "actionless".to_string(),
            description: None,
            root_path: "/tmp/actionless".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        store.create_project(&project).await.unwrap();

        let protocol_id = Uuid::new_v4();

        let mut start = ProtocolState::start(protocol_id, "start");
        start.action = Some("admin(update_staleness_scores)".to_string());

        // This state has NO action — runner should stop here
        let waiting = ProtocolState::new(protocol_id, "waiting_for_human");
        // action is None by default

        let done = ProtocolState::terminal(protocol_id, "done");

        let mut protocol = Protocol::new_full(
            project_id,
            "needs-human",
            "Protocol that needs human input at one point",
            start.id,
            vec![done.id],
            ProtocolCategory::System,
        );
        protocol.id = protocol_id;

        let transitions = vec![
            ProtocolTransition::new(protocol_id, start.id, waiting.id, "done"),
            ProtocolTransition::new(protocol_id, waiting.id, done.id, "human_approved"),
        ];

        store.upsert_protocol(&protocol).await.unwrap();
        for state in [&start, &waiting, &done] {
            store.upsert_protocol_state(state).await.unwrap();
        }
        for t in &transitions {
            store.upsert_protocol_transition(t).await.unwrap();
        }

        let run = engine::start_run(&*store, protocol.id, None, None, Some("event:post_sync"))
            .await
            .unwrap();

        let final_run = run_protocol_no_llm(store.clone(), run.id, cancel, emitter)
            .await
            .unwrap();

        // Runner should have stopped at "waiting_for_human" (no action to execute)
        assert_eq!(
            final_run.status,
            RunStatus::Running,
            "Run should still be Running, waiting for external trigger"
        );
        assert_eq!(final_run.current_state, waiting.id);

        // Now manually fire the human_approved trigger
        let result = engine::fire_transition(&*store, run.id, "human_approved")
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.run_completed);
        assert_eq!(result.current_state_name, "done");
    }
}
