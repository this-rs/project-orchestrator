//! Event-driven and scheduled protocol triggers
//!
//! Provides two auto-triggering mechanisms for protocols:
//!
//! ## Event-driven hooks
//!
//! Called from REST handlers as fire-and-forget background tasks via `tokio::spawn`.
//!
//! - `post_sync` — after `admin(sync_directory)`, `project(sync)`, or `commit(create)` with files
//! - `post_import` — after `skill(import)`
//! - `post_plan_complete` — after a plan reaches `completed` status (future)
//!
//! ## Periodic scheduler
//!
//! Spawned at server startup via [`spawn_protocol_scheduler`]. Evaluates every hour
//! which protocols have `trigger_mode = Scheduled | Auto` and a `schedule` field
//! in their `trigger_config`. If the schedule interval has elapsed since
//! `last_triggered_at`, a new run is started automatically.
//!
//! ## Debounce
//!
//! Each protocol has a `last_triggered_at` timestamp. Auto-triggers are skipped
//! if the protocol was triggered less than [`MIN_TRIGGER_INTERVAL_SECS`] seconds ago.

use crate::chat::manager::ChatManager;
use crate::events::EventEmitter;
use crate::neo4j::traits::GraphStore;
use crate::protocol::engine;
use dashmap::DashMap;
use std::sync::{Arc, LazyLock};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Global registry of active protocol runner tasks.
///
/// Maps run_id -> CancellationToken. Used by `cancel_run()` to signal
/// graceful shutdown of the runner loop, and by `spawn_protocol_runner()`
/// to register new runners.
static ACTIVE_RUNNERS: LazyLock<DashMap<Uuid, CancellationToken>> = LazyLock::new(DashMap::new);

/// Cancel an active runner by run_id (if one is registered).
///
/// Called from `engine::cancel_run()` to signal the runner loop
/// to stop gracefully via its CancellationToken.
pub fn cancel_active_runner(run_id: Uuid) {
    if let Some((_, token)) = ACTIVE_RUNNERS.remove(&run_id) {
        tracing::info!(%run_id, "Cancelling active protocol runner");
        token.cancel();
    }
}

/// Spawn a protocol runner task for the given run.
///
/// Registers the run in `ACTIVE_RUNNERS` and spawns a tokio task
/// that drives the FSM to completion via `runner::run_protocol()`.
///
/// When `chat_manager` is provided, business protocols can spawn Claude
/// agent sessions for LLM-driven state execution. Without it, business
/// states that require LLM will use stub/auto-advance behavior.
pub fn spawn_protocol_runner(
    store: Arc<dyn GraphStore>,
    run_id: Uuid,
    emitter: Arc<dyn EventEmitter>,
    chat_manager: Option<Arc<ChatManager>>,
) {
    let cancel = CancellationToken::new();
    ACTIVE_RUNNERS.insert(run_id, cancel.clone());
    tokio::spawn(async move {
        let result =
            super::runner::run_protocol(store, run_id, cancel, emitter, chat_manager).await;
        ACTIVE_RUNNERS.remove(&run_id);
        match &result {
            Ok(run) => tracing::info!(run_id = %run_id, "Protocol run completed: {:?}", run.status),
            Err(e) => tracing::warn!(run_id = %run_id, "Protocol run failed: {}", e),
        }
    });
}

/// Minimum interval between auto-triggered runs of the same protocol (5 minutes).
const MIN_TRIGGER_INTERVAL_SECS: i64 = 300;

/// Maximum time a run can stay in a single state before being auto-timed-out (8 hours).
/// Applies to runs created by auto-triggers (event/scheduled). Manual runs are not affected.
/// Set high enough to allow agent-driven protocols to complete across long sessions,
/// but low enough to prevent zombie runs from blocking triggers indefinitely.
const AUTO_RUN_TIMEOUT_SECS: i64 = 28800;

/// Spawn event-triggered protocol runs in the background.
///
/// Finds all protocols for the given project that listen to the specified event
/// (trigger_mode = Event or Auto, with the event in trigger_config.events),
/// and starts a run for each one.
///
/// This is a fire-and-forget operation — errors are logged, never propagated.
pub fn spawn_event_triggered_protocols(
    store: Arc<dyn GraphStore>,
    project_id: Uuid,
    event: &str,
    emitter: Option<Arc<dyn EventEmitter>>,
) {
    let event = event.to_string();
    tokio::spawn(async move {
        if let Err(e) =
            trigger_protocols_for_event(store.clone(), project_id, &event, emitter).await
        {
            tracing::warn!(
                %project_id,
                event = %event,
                "Event-triggered protocol hook failed: {}", e
            );
        }
    });
}

/// Core logic: find matching protocols and start runs.
async fn trigger_protocols_for_event(
    store: Arc<dyn GraphStore>,
    project_id: Uuid,
    event: &str,
    emitter: Option<Arc<dyn EventEmitter>>,
) -> anyhow::Result<()> {
    // List all protocols for the project (reasonable upper bound — few protocols per project)
    let (protocols, _) = store.list_protocols(project_id, None, 100, 0).await?;

    let now = chrono::Utc::now();
    let mut triggered_count = 0u32;

    for protocol in &protocols {
        // 1. Check trigger_mode listens to events
        if !protocol.trigger_mode.listens_to_events() {
            continue;
        }

        // 2. Check trigger_config contains the event
        let has_event = protocol
            .trigger_config
            .as_ref()
            .map(|c| c.events.iter().any(|e| e == event))
            .unwrap_or(false);

        if !has_event {
            continue;
        }

        // 3. Debounce: skip if last triggered less than MIN_TRIGGER_INTERVAL ago
        if let Some(last) = protocol.last_triggered_at {
            let elapsed_secs = (now - last).num_seconds();
            if elapsed_secs < MIN_TRIGGER_INTERVAL_SECS {
                tracing::debug!(
                    protocol_id = %protocol.id,
                    protocol_name = %protocol.name,
                    event = %event,
                    elapsed_secs,
                    "Skipping event trigger: debounce ({elapsed_secs}s < {MIN_TRIGGER_INTERVAL_SECS}s)"
                );
                continue;
            }
        }

        // 4. Start the run with triggered_by set
        let triggered_by = format!("event:{event}");
        match engine::start_run(&*store, protocol.id, None, None, Some(&triggered_by)).await {
            Ok(run) => {
                tracing::info!(
                    protocol_id = %protocol.id,
                    protocol_name = %protocol.name,
                    run_id = %run.id,
                    event = %event,
                    "Auto-triggered protocol run"
                );

                // 5. Update last_triggered_at on the protocol
                let mut updated_protocol = protocol.clone();
                updated_protocol.last_triggered_at = Some(now);
                updated_protocol.updated_at = now;
                if let Err(e) = store.upsert_protocol(&updated_protocol).await {
                    tracing::warn!(
                        protocol_id = %protocol.id,
                        "Failed to update last_triggered_at: {}", e
                    );
                }

                // 6. Spawn the protocol runner to drive the FSM
                if let Some(ref emitter) = emitter {
                    spawn_protocol_runner(store.clone(), run.id, emitter.clone(), None);
                }

                triggered_count += 1;
            }
            Err(e) => {
                tracing::warn!(
                    protocol_id = %protocol.id,
                    protocol_name = %protocol.name,
                    event = %event,
                    "Failed to auto-trigger protocol: {}", e
                );
            }
        }
    }

    if triggered_count > 0 {
        tracing::info!(
            %project_id,
            event = %event,
            count = triggered_count,
            "Event-triggered {triggered_count} protocol run(s)"
        );
    }

    Ok(())
}

// ============================================================================
// Orphan Run Recovery (server startup)
// ============================================================================

/// Maximum age for a Running run before it's considered orphaned (1 hour).
const ORPHAN_RUN_MAX_AGE_SECS: i64 = 3600;

/// Recover orphaned protocol runs at server startup.
///
/// Scans all projects for `ProtocolRun` nodes with `status = running` and
/// `started_at` older than [`ORPHAN_RUN_MAX_AGE_SECS`]. These runs were
/// likely interrupted by a server crash or restart and will never complete
/// on their own.
///
/// Each orphaned run is marked as `Failed` with the error message
/// "Recovered: server restarted during execution".
///
/// This function should be called once at server startup, before spawning
/// the scheduler or handling any requests.
pub async fn recover_orphaned_runs(
    store: Arc<dyn GraphStore>,
    emitter: Option<Arc<dyn EventEmitter>>,
) -> anyhow::Result<u32> {
    let projects = store.list_projects().await?;
    let now = chrono::Utc::now();
    let mut total_recovered = 0u32;

    for project in &projects {
        // List all protocols for this project
        let (protocols, _) = store.list_protocols(project.id, None, 100, 0).await?;

        for protocol in &protocols {
            // List running runs for this protocol
            let (running_runs, _) = store
                .list_protocol_runs(
                    protocol.id,
                    Some(crate::protocol::RunStatus::Running),
                    100,
                    0,
                )
                .await?;

            for run in &running_runs {
                let age_secs = (now - run.started_at).num_seconds();
                if age_secs > ORPHAN_RUN_MAX_AGE_SECS {
                    if run.runner_managed {
                        // Re-spawn the runner instead of failing
                        if let Some(ref emitter) = emitter {
                            tracing::info!(
                                run_id = %run.id,
                                protocol_id = %protocol.id,
                                protocol_name = %protocol.name,
                                age_secs,
                                "Re-spawning runner for orphaned runner-managed run"
                            );
                            spawn_protocol_runner(store.clone(), run.id, emitter.clone(), None);
                            total_recovered += 1;
                        } else {
                            let mut recovered_run = run.clone();
                            recovered_run
                                .fail("Recovered: server restarted (no emitter for re-spawn)");
                            if let Err(e) = store.update_protocol_run(&mut recovered_run).await {
                                tracing::warn!(
                                    run_id = %run.id,
                                    "Failed to recover orphaned run: {}", e
                                );
                            } else {
                                total_recovered += 1;
                            }
                        }
                    } else {
                        // Not runner-managed — mark as failed
                        let mut recovered_run = run.clone();
                        recovered_run.fail("Recovered: server restarted during execution");
                        if let Err(e) = store.update_protocol_run(&mut recovered_run).await {
                            tracing::warn!(
                                run_id = %run.id,
                                protocol_id = %protocol.id,
                                "Failed to recover orphaned run: {}", e
                            );
                        } else {
                            tracing::info!(
                                run_id = %run.id,
                                protocol_id = %protocol.id,
                                protocol_name = %protocol.name,
                                age_secs,
                                "Recovered orphaned protocol run"
                            );
                            total_recovered += 1;
                        }
                    }
                }
            }
        }
    }

    if total_recovered > 0 {
        tracing::info!(
            count = total_recovered,
            "Recovered {total_recovered} orphaned protocol run(s)"
        );
    }

    Ok(total_recovered)
}

// ============================================================================
// Stale Run Timeout
// ============================================================================

/// Timeout stale protocol runs that have been stuck on a single state too long.
///
/// Scans all projects for `ProtocolRun` nodes with `status = running` whose
/// last state entry timestamp is older than [`AUTO_RUN_TIMEOUT_SECS`].
/// Only auto-triggered runs (event:* or schedule:*) are affected — manual runs
/// are left alone since they may legitimately wait for human input.
///
/// Each timed-out run is marked as `Failed` with a descriptive error message.
pub async fn timeout_stale_runs(store: &dyn GraphStore) -> anyhow::Result<u32> {
    let projects = store.list_projects().await?;
    let now = chrono::Utc::now();
    let mut total_timed_out = 0u32;

    for project in &projects {
        let (protocols, _) = store.list_protocols(project.id, None, 100, 0).await?;

        for protocol in &protocols {
            let (running_runs, _) = store
                .list_protocol_runs(
                    protocol.id,
                    Some(crate::protocol::RunStatus::Running),
                    100,
                    0,
                )
                .await?;

            for run in &running_runs {
                // Only timeout auto-triggered runs (event:* or schedule:*)
                if run.triggered_by == "manual" {
                    continue;
                }

                // Check how long the run has been in its current state
                let last_state_entered = run
                    .states_visited
                    .last()
                    .map(|sv| sv.entered_at)
                    .unwrap_or(run.started_at);

                let state_age_secs = (now - last_state_entered).num_seconds();
                if state_age_secs <= AUTO_RUN_TIMEOUT_SECS {
                    continue;
                }

                let state_name = run
                    .states_visited
                    .last()
                    .map(|sv| sv.state_name.as_str())
                    .unwrap_or("unknown");

                let error_msg = format!(
                    "Auto-timeout: state '{}' exceeded {}s (stuck for {}s)",
                    state_name, AUTO_RUN_TIMEOUT_SECS, state_age_secs
                );

                let mut timed_out_run = run.clone();
                timed_out_run.fail(&error_msg);

                if let Err(e) = store.update_protocol_run(&mut timed_out_run).await {
                    tracing::warn!(
                        run_id = %run.id,
                        protocol_id = %protocol.id,
                        "Failed to timeout stale run: {}", e
                    );
                } else {
                    tracing::warn!(
                        run_id = %run.id,
                        protocol_id = %protocol.id,
                        protocol_name = %protocol.name,
                        state_name,
                        state_age_secs,
                        "Timed out stale protocol run"
                    );
                    total_timed_out += 1;
                }
            }
        }
    }

    if total_timed_out > 0 {
        tracing::info!(
            count = total_timed_out,
            "Timed out {total_timed_out} stale protocol run(s)"
        );
    }

    Ok(total_timed_out)
}

// ============================================================================
// Periodic Scheduler
// ============================================================================

/// Evaluation interval for the protocol scheduler (1 hour).
const SCHEDULER_INTERVAL_SECS: u64 = 3600;

/// Schedule thresholds: how long since `last_triggered_at` before re-triggering.
fn schedule_interval_secs(schedule: &str) -> Option<i64> {
    match schedule {
        "hourly" => Some(3600),   // 1 hour
        "daily" => Some(86400),   // 24 hours
        "weekly" => Some(604800), // 7 days
        _ => {
            tracing::warn!(schedule, "Unknown schedule value — ignoring");
            None
        }
    }
}

/// Spawn the periodic protocol scheduler as a background task.
///
/// Runs every [`SCHEDULER_INTERVAL_SECS`] seconds (1 hour). On each tick,
/// evaluates all protocols across all projects and starts runs for those
/// with `trigger_mode = Scheduled | Auto` whose schedule interval has elapsed.
///
/// # Arguments
/// - `store` — shared GraphStore for querying protocols and starting runs
///
/// This task runs for the lifetime of the server and never returns.
pub fn spawn_protocol_scheduler(
    store: Arc<dyn GraphStore>,
    emitter: Option<Arc<dyn EventEmitter>>,
) {
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(SCHEDULER_INTERVAL_SECS));

        // The first tick fires immediately — skip it to avoid triggering
        // right at server startup (let the system settle first).
        interval.tick().await;

        loop {
            interval.tick().await;

            tracing::debug!("Protocol scheduler: tick — orphan recovery + timeout + scheduling");

            // 1. Recover orphaned runs (same logic as startup, now periodic)
            if let Err(e) = recover_orphaned_runs(store.clone(), emitter.clone()).await {
                tracing::warn!("Protocol scheduler: orphan recovery failed: {}", e);
            }

            // 2. Timeout stale auto-triggered runs stuck on a single state
            if let Err(e) = timeout_stale_runs(&*store).await {
                tracing::warn!("Protocol scheduler: stale run timeout failed: {}", e);
            }

            // 3. Evaluate scheduled protocols and start new runs
            if let Err(e) = run_scheduled_protocols(store.clone(), emitter.clone()).await {
                tracing::warn!("Protocol scheduler: scheduled trigger failed: {}", e);
            }
        }
    });
}

/// Core logic: find all scheduled protocols across all projects and trigger them.
async fn run_scheduled_protocols(
    store: Arc<dyn GraphStore>,
    emitter: Option<Arc<dyn EventEmitter>>,
) -> anyhow::Result<()> {
    let projects = store.list_projects().await?;
    let now = chrono::Utc::now();
    let mut total_triggered = 0u32;

    for project in &projects {
        // List all protocols for this project
        let (protocols, _) = store.list_protocols(project.id, None, 100, 0).await?;

        for protocol in &protocols {
            // Check trigger_mode supports scheduling
            if !protocol.trigger_mode.is_scheduled() {
                continue;
            }

            // Check trigger_config has a schedule
            let schedule = match protocol
                .trigger_config
                .as_ref()
                .and_then(|c| c.schedule.as_deref())
            {
                Some(s) => s,
                None => continue,
            };

            // Resolve the required interval
            let required_secs = match schedule_interval_secs(schedule) {
                Some(s) => s,
                None => continue,
            };

            // Check if enough time has passed since last trigger
            let elapsed_secs = protocol
                .last_triggered_at
                .map(|last| (now - last).num_seconds())
                .unwrap_or(i64::MAX); // Never triggered = always due

            if elapsed_secs < required_secs {
                continue;
            }

            // Start the run
            let triggered_by = format!("schedule:{schedule}");
            match engine::start_run(&*store, protocol.id, None, None, Some(&triggered_by)).await {
                Ok(run) => {
                    tracing::info!(
                        protocol_id = %protocol.id,
                        protocol_name = %protocol.name,
                        schedule,
                        run_id = %run.id,
                        "Scheduled protocol run started"
                    );

                    // Update last_triggered_at
                    let mut updated_protocol = protocol.clone();
                    updated_protocol.last_triggered_at = Some(now);
                    updated_protocol.updated_at = now;
                    if let Err(e) = store.upsert_protocol(&updated_protocol).await {
                        tracing::warn!(
                            protocol_id = %protocol.id,
                            "Failed to update last_triggered_at: {}", e
                        );
                    }

                    // Spawn the protocol runner to drive the FSM
                    if let Some(ref emitter) = emitter {
                        spawn_protocol_runner(store.clone(), run.id, emitter.clone(), None);
                    }

                    total_triggered += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        protocol_id = %protocol.id,
                        protocol_name = %protocol.name,
                        schedule,
                        "Failed to start scheduled protocol run: {}", e
                    );
                }
            }
        }
    }

    if total_triggered > 0 {
        tracing::info!(
            count = total_triggered,
            "Scheduler triggered {total_triggered} protocol run(s)"
        );
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::protocol::{
        Protocol, ProtocolCategory, ProtocolState, ProtocolTransition, TriggerConfig, TriggerMode,
    };
    use std::sync::Arc;

    /// Helper to set up a 3-state protocol with event trigger config.
    async fn setup_event_triggered_protocol(
        store: &MockGraphStore,
        trigger_mode: TriggerMode,
        events: Vec<String>,
    ) -> (Uuid, Protocol) {
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
        };
        store.create_project(&project).await.unwrap();

        let protocol_id = Uuid::new_v4();
        let start_state = ProtocolState::start(protocol_id, "Start");
        let done_state = ProtocolState::terminal(protocol_id, "Done");

        let mut protocol = Protocol::new_full(
            project_id,
            "Test Event Protocol",
            "Protocol for event-trigger testing",
            start_state.id,
            vec![done_state.id],
            ProtocolCategory::System,
        );
        protocol.id = protocol_id;
        protocol.trigger_mode = trigger_mode;
        protocol.trigger_config = Some(TriggerConfig {
            events,
            schedule: None,
            conditions: vec![],
        });

        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&start_state).await.unwrap();
        store.upsert_protocol_state(&done_state).await.unwrap();

        let t1 = ProtocolTransition::new(protocol_id, start_state.id, done_state.id, "complete");
        store.upsert_protocol_transition(&t1).await.unwrap();

        (project_id, protocol)
    }

    #[tokio::test]
    async fn test_event_trigger_starts_run() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Verify a run was created
        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "event:post_sync");
    }

    #[tokio::test]
    async fn test_event_trigger_auto_mode() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Auto,
            vec!["post_sync".to_string(), "post_import".to_string()],
        )
        .await;

        // post_sync should trigger
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        let (runs, _) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].triggered_by, "event:post_sync");
    }

    #[tokio::test]
    async fn test_event_trigger_ignores_manual_mode() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Manual,
            vec!["post_sync".to_string()],
        )
        .await;

        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // No run should be created
        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
        assert!(runs.is_empty());
    }

    #[tokio::test]
    async fn test_event_trigger_ignores_scheduled_only() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Scheduled,
            vec!["post_sync".to_string()],
        )
        .await;

        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_event_trigger_wrong_event_ignored() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_import".to_string()],
        )
        .await;

        // post_sync should NOT trigger a protocol listening only to post_import
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_event_trigger_debounce() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, mut protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Set last_triggered_at to 1 minute ago (within debounce window)
        protocol.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::seconds(60));
        store.upsert_protocol(&protocol).await.unwrap();

        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Should be debounced — no run created
        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_event_trigger_after_debounce_window() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, mut protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Set last_triggered_at to 10 minutes ago (outside debounce window)
        protocol.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::seconds(600));
        store.upsert_protocol(&protocol).await.unwrap();

        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Should trigger — debounce window has passed
        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "event:post_sync");
    }

    #[tokio::test]
    async fn test_event_trigger_no_protocols() {
        let store = Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();

        // Create only the project (no protocols)
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "empty-project".to_string(),
            slug: "empty-project".to_string(),
            description: None,
            root_path: "/tmp/empty".to_string(),
            created_at: chrono::Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
        };
        store.create_project(&project).await.unwrap();

        // Should succeed silently (no protocols to trigger)
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_event_trigger_updates_last_triggered_at() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        assert!(protocol.last_triggered_at.is_none());

        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Verify last_triggered_at was updated
        let updated = store.get_protocol(protocol.id).await.unwrap().unwrap();
        assert!(updated.last_triggered_at.is_some());
    }

    // ====================================================================
    // Scheduler tests
    // ====================================================================

    /// Helper to set up a protocol with schedule config.
    async fn setup_scheduled_protocol(
        store: &MockGraphStore,
        trigger_mode: TriggerMode,
        schedule: Option<&str>,
    ) -> (Uuid, Protocol) {
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
        };
        store.create_project(&project).await.unwrap();

        let protocol_id = Uuid::new_v4();
        let start_state = ProtocolState::start(protocol_id, "Start");
        let done_state = ProtocolState::terminal(protocol_id, "Done");

        let mut protocol = Protocol::new_full(
            project_id,
            "Test Scheduled Protocol",
            "Protocol for scheduler testing",
            start_state.id,
            vec![done_state.id],
            ProtocolCategory::System,
        );
        protocol.id = protocol_id;
        protocol.trigger_mode = trigger_mode;
        protocol.trigger_config = Some(TriggerConfig {
            events: vec![],
            schedule: schedule.map(|s| s.to_string()),
            conditions: vec![],
        });

        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&start_state).await.unwrap();
        store.upsert_protocol_state(&done_state).await.unwrap();

        let t1 = ProtocolTransition::new(protocol_id, start_state.id, done_state.id, "complete");
        store.upsert_protocol_transition(&t1).await.unwrap();

        (project_id, protocol)
    }

    #[test]
    fn test_schedule_interval_secs_values() {
        assert_eq!(schedule_interval_secs("hourly"), Some(3600));
        assert_eq!(schedule_interval_secs("daily"), Some(86400));
        assert_eq!(schedule_interval_secs("weekly"), Some(604800));
        assert_eq!(schedule_interval_secs("unknown"), None);
    }

    // ====================================================================
    // Recovery tests
    // ====================================================================

    #[tokio::test]
    async fn test_recover_orphaned_run() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Create a run that's been "running" for 2 hours (orphaned)
        let mut run = crate::protocol::ProtocolRun::new(protocol.id, protocol.entry_state, "Start");
        run.started_at = chrono::Utc::now() - chrono::Duration::hours(2);
        store.create_protocol_run(&run).await.unwrap();

        let count = recover_orphaned_runs(store.clone() as Arc<dyn GraphStore>, None)
            .await
            .unwrap();
        assert_eq!(count, 1);

        // Verify run is now Failed
        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, crate::protocol::RunStatus::Failed);
        assert_eq!(
            updated.error.as_deref(),
            Some("Recovered: server restarted during execution")
        );
        assert!(updated.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_recover_skips_recent_runs() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Create a run that's been running for 5 minutes (not orphaned)
        let mut run = crate::protocol::ProtocolRun::new(protocol.id, protocol.entry_state, "Start");
        run.started_at = chrono::Utc::now() - chrono::Duration::minutes(5);
        store.create_protocol_run(&run).await.unwrap();

        let count = recover_orphaned_runs(store.clone() as Arc<dyn GraphStore>, None)
            .await
            .unwrap();
        assert_eq!(count, 0);

        // Run should still be Running
        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, crate::protocol::RunStatus::Running);
    }

    #[tokio::test]
    async fn test_recover_skips_completed_runs() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Create an old but already-completed run
        let mut run = crate::protocol::ProtocolRun::new(protocol.id, protocol.entry_state, "Start");
        run.started_at = chrono::Utc::now() - chrono::Duration::hours(5);
        run.complete();
        store.create_protocol_run(&run).await.unwrap();

        let count = recover_orphaned_runs(store.clone() as Arc<dyn GraphStore>, None)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_recover_no_projects() {
        let store: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
        let count = recover_orphaned_runs(store, None).await.unwrap();
        assert_eq!(count, 0);
    }

    // ====================================================================
    // Scheduler tests
    // ====================================================================

    #[tokio::test]
    async fn test_scheduled_trigger_daily_due() {
        let store = Arc::new(MockGraphStore::new());
        let (_, mut protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("daily")).await;

        // Set last_triggered_at to 25 hours ago (> 24h daily threshold)
        protocol.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::hours(25));
        store.upsert_protocol(&protocol).await.unwrap();

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:daily");
    }

    #[tokio::test]
    async fn test_scheduled_trigger_never_triggered() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("hourly")).await;

        // last_triggered_at is None — should trigger (never triggered = always due)
        assert!(protocol.last_triggered_at.is_none());

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:hourly");
    }

    #[tokio::test]
    async fn test_scheduled_not_due_yet() {
        let store = Arc::new(MockGraphStore::new());
        let (_, mut protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("daily")).await;

        // Set last_triggered_at to 1 hour ago (< 24h daily threshold)
        protocol.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        store.upsert_protocol(&protocol).await.unwrap();

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_scheduled_ignores_event_only() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Event, Some("daily")).await;

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        // Event-only protocols should not be triggered by scheduler
        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_scheduled_ignores_manual() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Manual, Some("daily")).await;

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_scheduled_auto_mode() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Auto, Some("weekly")).await;

        // Auto = Event + Scheduled, should be picked up by scheduler
        // last_triggered_at is None → always due
        run_scheduled_protocols(store.clone(), None).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:weekly");
    }

    #[tokio::test]
    async fn test_scheduled_updates_last_triggered_at() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("hourly")).await;

        assert!(protocol.last_triggered_at.is_none());

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        let updated = store.get_protocol(protocol.id).await.unwrap().unwrap();
        assert!(updated.last_triggered_at.is_some());
    }

    #[tokio::test]
    async fn test_scheduled_no_schedule_config() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) = setup_scheduled_protocol(&store, TriggerMode::Scheduled, None).await;

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        // No schedule in config → not triggered
        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    // ====================================================================
    // Integration tests: Auto mode (Event + Scheduled combined)
    // ====================================================================

    /// Helper: create a protocol mimicking the "inference" protocol with
    /// trigger_mode=Auto, events=["post_sync", "post_import"], schedule="daily".
    async fn setup_inference_protocol(store: &MockGraphStore) -> (Uuid, Protocol) {
        let project_id = Uuid::new_v4();
        let project = crate::neo4j::models::ProjectNode {
            id: project_id,
            name: "inference-project".to_string(),
            slug: "inference-project".to_string(),
            description: None,
            root_path: "/tmp/inference".to_string(),
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
        let start_state = ProtocolState::start(protocol_id, "Start");
        let mid_state = ProtocolState::new(protocol_id, "Processing");
        let done_state = ProtocolState::terminal(protocol_id, "Done");

        let mut protocol = Protocol::new_full(
            project_id,
            "inference",
            "Code intelligence inference protocol",
            start_state.id,
            vec![done_state.id],
            ProtocolCategory::System,
        );
        protocol.id = protocol_id;
        protocol.trigger_mode = TriggerMode::Auto;
        protocol.trigger_config = Some(TriggerConfig {
            events: vec!["post_sync".to_string(), "post_import".to_string()],
            schedule: Some("daily".to_string()),
            conditions: vec![],
        });

        store.upsert_protocol(&protocol).await.unwrap();
        store.upsert_protocol_state(&start_state).await.unwrap();
        store.upsert_protocol_state(&mid_state).await.unwrap();
        store.upsert_protocol_state(&done_state).await.unwrap();

        let t1 = ProtocolTransition::new(protocol_id, start_state.id, mid_state.id, "process");
        let t2 = ProtocolTransition::new(protocol_id, mid_state.id, done_state.id, "complete");
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        (project_id, protocol)
    }

    /// Integration test 1: sync → auto-triggered event run
    #[tokio::test]
    async fn test_integration_sync_triggers_inference() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // Simulate a post_sync event (like after admin(sync_directory))
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "event:post_sync");

        // Complete the first run so the next event can start (concurrency guard)
        engine::fire_transition(&*store, runs[0].id, "process")
            .await
            .unwrap();
        engine::fire_transition(&*store, runs[0].id, "complete")
            .await
            .unwrap();

        // Also verify post_import triggers
        // Advance time past debounce window by updating last_triggered_at to 10min ago
        let mut p = store.get_protocol(protocol.id).await.unwrap().unwrap();
        p.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::seconds(600));
        store.upsert_protocol(&p).await.unwrap();

        trigger_protocols_for_event(store.clone(), project_id, "post_import", None)
            .await
            .unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 2);
        // Most recent run should be the post_import one
        let import_run = runs.iter().find(|r| r.triggered_by == "event:post_import");
        assert!(
            import_run.is_some(),
            "Should have a post_import triggered run"
        );
    }

    /// Integration test 2: schedule → auto-triggered scheduled run
    #[tokio::test]
    async fn test_integration_schedule_triggers_inference() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) = setup_inference_protocol(&store).await;

        // Protocol has never been triggered → scheduler should trigger it
        run_scheduled_protocols(store.clone(), None).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:daily");
    }

    /// Integration test 3: manual, event, and scheduled runs coexist sequentially.
    /// Each must complete before the next can start (concurrency guard).
    #[tokio::test]
    async fn test_integration_manual_then_event_then_schedule() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // 1. Manual start (via MCP/REST — triggered_by defaults to "manual")
        let manual_run = engine::start_run(&*store, protocol.id, None, None, None)
            .await
            .unwrap();
        assert_eq!(manual_run.triggered_by, "manual");

        // Complete the manual run so event-triggered can start
        engine::fire_transition(&*store, manual_run.id, "process")
            .await
            .unwrap();
        engine::fire_transition(&*store, manual_run.id, "complete")
            .await
            .unwrap();

        // 2. Event-triggered start (simulating post_sync)
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Complete the event run so schedule-triggered can start
        let (event_runs, _) = store
            .list_protocol_runs(protocol.id, Some(crate::protocol::RunStatus::Running), 1, 0)
            .await
            .unwrap();
        assert_eq!(event_runs.len(), 1);
        engine::fire_transition(&*store, event_runs[0].id, "process")
            .await
            .unwrap();
        engine::fire_transition(&*store, event_runs[0].id, "complete")
            .await
            .unwrap();

        // 3. Advance past debounce, then schedule-triggered start
        let mut p = store.get_protocol(protocol.id).await.unwrap().unwrap();
        p.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::hours(25));
        store.upsert_protocol(&p).await.unwrap();

        run_scheduled_protocols(store.clone(), None).await.unwrap();

        // All 3 runs should exist with different triggered_by values
        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 3, "Expected 3 runs (manual + event + schedule)");

        let triggers: Vec<&str> = runs.iter().map(|r| r.triggered_by.as_str()).collect();
        assert!(triggers.contains(&"manual"), "Should have manual run");
        assert!(
            triggers.contains(&"event:post_sync"),
            "Should have event run"
        );
        assert!(
            triggers.contains(&"schedule:daily"),
            "Should have schedule run"
        );
    }

    /// Integration test: concurrent run guard rejects event trigger when run is active
    #[tokio::test]
    async fn test_integration_concurrent_guard_rejects_event() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // Start a manual run (still Running)
        let _manual_run = engine::start_run(&*store, protocol.id, None, None, None)
            .await
            .unwrap();

        // Event trigger should be silently rejected (concurrent run guard)
        // trigger_protocols_for_event catches the error and logs it, so it returns Ok(())
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Only 1 run should exist (the manual one)
        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(
            total, 1,
            "Event trigger should be rejected when a run is active"
        );
    }

    /// Integration test: event debounce prevents double-trigger on rapid syncs
    #[tokio::test]
    async fn test_integration_debounce_prevents_rapid_triggers() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // First sync triggers
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Complete the run so next events can attempt to start (not blocked by concurrency guard)
        let (runs, _) = store
            .list_protocol_runs(protocol.id, Some(crate::protocol::RunStatus::Running), 1, 0)
            .await
            .unwrap();
        engine::fire_transition(&*store, runs[0].id, "process")
            .await
            .unwrap();
        engine::fire_transition(&*store, runs[0].id, "complete")
            .await
            .unwrap();

        // Second sync should be debounced (last_triggered_at was just set)
        trigger_protocols_for_event(store.clone(), project_id, "post_sync", None)
            .await
            .unwrap();

        // Third sync with different event should also be debounced
        // (debounce is per-protocol, not per-event)
        trigger_protocols_for_event(store.clone(), project_id, "post_import", None)
            .await
            .unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(
            total, 1,
            "Only the first event should trigger, rest debounced"
        );
    }

    // ====================================================================
    // Stress test: concurrent triggers via tokio::join!
    // ====================================================================

    /// Stress test: 10 concurrent event triggers → exactly 1 run created.
    /// The mock's atomic check-and-create within a single write lock ensures
    /// that only 1 `start_run` succeeds, even under high concurrency.
    #[tokio::test]
    async fn test_stress_10_concurrent_triggers() {
        let store = Arc::new(MockGraphStore::new());
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // Launch 10 concurrent event triggers
        let mut handles = Vec::new();
        for _ in 0..10 {
            let store_clone: Arc<dyn GraphStore> = store.clone();
            let handle = tokio::spawn(async move {
                trigger_protocols_for_event(store_clone, project_id, "post_sync", None).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        let results: Vec<_> = futures::future::join_all(handles).await;

        // All should return Ok (errors are caught internally by trigger_protocols_for_event)
        for result in &results {
            assert!(result.is_ok(), "Spawned task should not panic");
            assert!(
                result.as_ref().unwrap().is_ok(),
                "trigger_protocols_for_event should not propagate errors"
            );
        }

        // Exactly 1 run should have been created
        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(
            total, 1,
            "Expected exactly 1 run from 10 concurrent triggers, got {}",
            total
        );
        assert_eq!(runs[0].triggered_by, "event:post_sync");
    }

    // ====================================================================
    // Stale run timeout tests
    // ====================================================================

    #[tokio::test]
    async fn test_timeout_stale_auto_triggered_run() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Create a run triggered by event, stuck for 9 hours (> 8h timeout)
        let mut run = crate::protocol::ProtocolRun::new(protocol.id, protocol.entry_state, "Start");
        run.triggered_by = "event:post_sync".to_string();
        run.started_at = chrono::Utc::now() - chrono::Duration::hours(9);
        // Also backdate the state visit entry
        if let Some(sv) = run.states_visited.first_mut() {
            sv.entered_at = run.started_at;
        }
        store.create_protocol_run(&run).await.unwrap();

        let count = timeout_stale_runs(&*store).await.unwrap();
        assert_eq!(count, 1);

        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, crate::protocol::RunStatus::Failed);
        assert!(updated.error.as_ref().unwrap().contains("Auto-timeout"));
        assert!(updated.error.as_ref().unwrap().contains("Start"));
        assert!(updated.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_timeout_skips_recent_auto_run() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Create a run triggered by event, only 1 hour old (< 8h timeout)
        let mut run = crate::protocol::ProtocolRun::new(protocol.id, protocol.entry_state, "Start");
        run.triggered_by = "event:post_sync".to_string();
        run.started_at = chrono::Utc::now() - chrono::Duration::hours(1);
        if let Some(sv) = run.states_visited.first_mut() {
            sv.entered_at = run.started_at;
        }
        store.create_protocol_run(&run).await.unwrap();

        let count = timeout_stale_runs(&*store).await.unwrap();
        assert_eq!(count, 0);

        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, crate::protocol::RunStatus::Running);
    }

    #[tokio::test]
    async fn test_timeout_skips_manual_runs() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) =
            setup_event_triggered_protocol(&store, TriggerMode::Manual, vec![]).await;

        // Create a manual run stuck for 10 hours — should NOT be timed out
        let mut run = crate::protocol::ProtocolRun::new(protocol.id, protocol.entry_state, "Start");
        run.triggered_by = "manual".to_string();
        run.started_at = chrono::Utc::now() - chrono::Duration::hours(10);
        if let Some(sv) = run.states_visited.first_mut() {
            sv.entered_at = run.started_at;
        }
        store.create_protocol_run(&run).await.unwrap();

        let count = timeout_stale_runs(&*store).await.unwrap();
        assert_eq!(count, 0);

        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, crate::protocol::RunStatus::Running);
    }

    #[tokio::test]
    async fn test_timeout_scheduled_run() {
        let store = Arc::new(MockGraphStore::new());
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("daily")).await;

        // Create a scheduled run stuck for 9 hours
        let mut run = crate::protocol::ProtocolRun::new(protocol.id, protocol.entry_state, "Start");
        run.triggered_by = "schedule:daily".to_string();
        run.started_at = chrono::Utc::now() - chrono::Duration::hours(9);
        if let Some(sv) = run.states_visited.first_mut() {
            sv.entered_at = run.started_at;
        }
        store.create_protocol_run(&run).await.unwrap();

        let count = timeout_stale_runs(&*store).await.unwrap();
        assert_eq!(count, 1);

        let updated = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(updated.status, crate::protocol::RunStatus::Failed);
    }
}
