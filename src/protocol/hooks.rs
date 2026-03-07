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

use crate::neo4j::traits::GraphStore;
use crate::protocol::engine;
use std::sync::Arc;
use uuid::Uuid;

/// Minimum interval between auto-triggered runs of the same protocol (5 minutes).
const MIN_TRIGGER_INTERVAL_SECS: i64 = 300;

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
) {
    let event = event.to_string();
    tokio::spawn(async move {
        if let Err(e) = trigger_protocols_for_event(&*store, project_id, &event).await {
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
    store: &dyn GraphStore,
    project_id: Uuid,
    event: &str,
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
        match engine::start_run(store, protocol.id, None, None, Some(&triggered_by)).await {
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
// Periodic Scheduler
// ============================================================================

/// Evaluation interval for the protocol scheduler (1 hour).
const SCHEDULER_INTERVAL_SECS: u64 = 3600;

/// Schedule thresholds: how long since `last_triggered_at` before re-triggering.
fn schedule_interval_secs(schedule: &str) -> Option<i64> {
    match schedule {
        "hourly" => Some(3600),        // 1 hour
        "daily" => Some(86400),        // 24 hours
        "weekly" => Some(604800),      // 7 days
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
pub fn spawn_protocol_scheduler(store: Arc<dyn GraphStore>) {
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(SCHEDULER_INTERVAL_SECS));

        // The first tick fires immediately — skip it to avoid triggering
        // right at server startup (let the system settle first).
        interval.tick().await;

        loop {
            interval.tick().await;

            tracing::debug!("Protocol scheduler: evaluating scheduled protocols");

            if let Err(e) = run_scheduled_protocols(&*store).await {
                tracing::warn!("Protocol scheduler tick failed: {}", e);
            }
        }
    });
}

/// Core logic: find all scheduled protocols across all projects and trigger them.
async fn run_scheduled_protocols(store: &dyn GraphStore) -> anyhow::Result<()> {
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
            match engine::start_run(store, protocol.id, None, None, Some(&triggered_by)).await {
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

        let t1 =
            ProtocolTransition::new(protocol_id, start_state.id, done_state.id, "complete");
        store.upsert_protocol_transition(&t1).await.unwrap();

        (project_id, protocol)
    }

    #[tokio::test]
    async fn test_event_trigger_starts_run() {
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        trigger_protocols_for_event(&store, project_id, "post_sync")
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
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Auto,
            vec!["post_sync".to_string(), "post_import".to_string()],
        )
        .await;

        // post_sync should trigger
        trigger_protocols_for_event(&store, project_id, "post_sync")
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
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Manual,
            vec!["post_sync".to_string()],
        )
        .await;

        trigger_protocols_for_event(&store, project_id, "post_sync")
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
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Scheduled,
            vec!["post_sync".to_string()],
        )
        .await;

        trigger_protocols_for_event(&store, project_id, "post_sync")
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
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_import".to_string()],
        )
        .await;

        // post_sync should NOT trigger a protocol listening only to post_import
        trigger_protocols_for_event(&store, project_id, "post_sync")
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
        let store = MockGraphStore::new();
        let (project_id, mut protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Set last_triggered_at to 1 minute ago (within debounce window)
        protocol.last_triggered_at =
            Some(chrono::Utc::now() - chrono::Duration::seconds(60));
        store.upsert_protocol(&protocol).await.unwrap();

        trigger_protocols_for_event(&store, project_id, "post_sync")
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
        let store = MockGraphStore::new();
        let (project_id, mut protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        // Set last_triggered_at to 10 minutes ago (outside debounce window)
        protocol.last_triggered_at =
            Some(chrono::Utc::now() - chrono::Duration::seconds(600));
        store.upsert_protocol(&protocol).await.unwrap();

        trigger_protocols_for_event(&store, project_id, "post_sync")
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
        let store = MockGraphStore::new();
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
        };
        store.create_project(&project).await.unwrap();

        // Should succeed silently (no protocols to trigger)
        trigger_protocols_for_event(&store, project_id, "post_sync")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_event_trigger_updates_last_triggered_at() {
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_event_triggered_protocol(
            &store,
            TriggerMode::Event,
            vec!["post_sync".to_string()],
        )
        .await;

        assert!(protocol.last_triggered_at.is_none());

        trigger_protocols_for_event(&store, project_id, "post_sync")
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

        let t1 =
            ProtocolTransition::new(protocol_id, start_state.id, done_state.id, "complete");
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

    #[tokio::test]
    async fn test_scheduled_trigger_daily_due() {
        let store = MockGraphStore::new();
        let (_, mut protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("daily")).await;

        // Set last_triggered_at to 25 hours ago (> 24h daily threshold)
        protocol.last_triggered_at =
            Some(chrono::Utc::now() - chrono::Duration::hours(25));
        store.upsert_protocol(&protocol).await.unwrap();

        run_scheduled_protocols(&store).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:daily");
    }

    #[tokio::test]
    async fn test_scheduled_trigger_never_triggered() {
        let store = MockGraphStore::new();
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("hourly")).await;

        // last_triggered_at is None — should trigger (never triggered = always due)
        assert!(protocol.last_triggered_at.is_none());

        run_scheduled_protocols(&store).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:hourly");
    }

    #[tokio::test]
    async fn test_scheduled_not_due_yet() {
        let store = MockGraphStore::new();
        let (_, mut protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("daily")).await;

        // Set last_triggered_at to 1 hour ago (< 24h daily threshold)
        protocol.last_triggered_at =
            Some(chrono::Utc::now() - chrono::Duration::hours(1));
        store.upsert_protocol(&protocol).await.unwrap();

        run_scheduled_protocols(&store).await.unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_scheduled_ignores_event_only() {
        let store = MockGraphStore::new();
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Event, Some("daily")).await;

        run_scheduled_protocols(&store).await.unwrap();

        // Event-only protocols should not be triggered by scheduler
        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_scheduled_ignores_manual() {
        let store = MockGraphStore::new();
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Manual, Some("daily")).await;

        run_scheduled_protocols(&store).await.unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_scheduled_auto_mode() {
        let store = MockGraphStore::new();
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Auto, Some("weekly")).await;

        // Auto = Event + Scheduled, should be picked up by scheduler
        // last_triggered_at is None → always due
        run_scheduled_protocols(&store).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:weekly");
    }

    #[tokio::test]
    async fn test_scheduled_updates_last_triggered_at() {
        let store = MockGraphStore::new();
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, Some("hourly")).await;

        assert!(protocol.last_triggered_at.is_none());

        run_scheduled_protocols(&store).await.unwrap();

        let updated = store.get_protocol(protocol.id).await.unwrap().unwrap();
        assert!(updated.last_triggered_at.is_some());
    }

    #[tokio::test]
    async fn test_scheduled_no_schedule_config() {
        let store = MockGraphStore::new();
        let (_, protocol) =
            setup_scheduled_protocol(&store, TriggerMode::Scheduled, None).await;

        run_scheduled_protocols(&store).await.unwrap();

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

        let t1 = ProtocolTransition::new(
            protocol_id, start_state.id, mid_state.id, "process",
        );
        let t2 = ProtocolTransition::new(
            protocol_id, mid_state.id, done_state.id, "complete",
        );
        store.upsert_protocol_transition(&t1).await.unwrap();
        store.upsert_protocol_transition(&t2).await.unwrap();

        (project_id, protocol)
    }

    /// Integration test 1: sync → auto-triggered event run
    #[tokio::test]
    async fn test_integration_sync_triggers_inference() {
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // Simulate a post_sync event (like after admin(sync_directory))
        trigger_protocols_for_event(&store, project_id, "post_sync")
            .await
            .unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "event:post_sync");

        // Also verify post_import triggers
        // First, advance time past debounce window by updating last_triggered_at to 10min ago
        let mut p = store.get_protocol(protocol.id).await.unwrap().unwrap();
        p.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::seconds(600));
        store.upsert_protocol(&p).await.unwrap();

        trigger_protocols_for_event(&store, project_id, "post_import")
            .await
            .unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 2);
        // Most recent run should be the post_import one
        let import_run = runs.iter().find(|r| r.triggered_by == "event:post_import");
        assert!(import_run.is_some(), "Should have a post_import triggered run");
    }

    /// Integration test 2: schedule → auto-triggered scheduled run
    #[tokio::test]
    async fn test_integration_schedule_triggers_inference() {
        let store = MockGraphStore::new();
        let (_, protocol) = setup_inference_protocol(&store).await;

        // Protocol has never been triggered → scheduler should trigger it
        run_scheduled_protocols(&store).await.unwrap();

        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(runs[0].triggered_by, "schedule:daily");
    }

    /// Integration test 3: manual start coexists with auto-triggered runs
    #[tokio::test]
    async fn test_integration_manual_coexists_with_auto() {
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // 1. Manual start (via MCP/REST — triggered_by defaults to "manual")
        let manual_run = engine::start_run(&store, protocol.id, None, None, None)
            .await
            .unwrap();
        assert_eq!(manual_run.triggered_by, "manual");

        // 2. Event-triggered start (simulating post_sync)
        trigger_protocols_for_event(&store, project_id, "post_sync")
            .await
            .unwrap();

        // 3. Advance past debounce, then schedule-triggered start
        let mut p = store.get_protocol(protocol.id).await.unwrap().unwrap();
        p.last_triggered_at = Some(chrono::Utc::now() - chrono::Duration::hours(25));
        store.upsert_protocol(&p).await.unwrap();

        run_scheduled_protocols(&store).await.unwrap();

        // All 3 runs should coexist
        let (runs, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 3, "Expected 3 runs (manual + event + schedule)");

        let triggers: Vec<&str> = runs.iter().map(|r| r.triggered_by.as_str()).collect();
        assert!(triggers.contains(&"manual"), "Should have manual run");
        assert!(triggers.contains(&"event:post_sync"), "Should have event run");
        assert!(triggers.contains(&"schedule:daily"), "Should have schedule run");
    }

    /// Integration test: event debounce prevents double-trigger on rapid syncs
    #[tokio::test]
    async fn test_integration_debounce_prevents_rapid_triggers() {
        let store = MockGraphStore::new();
        let (project_id, protocol) = setup_inference_protocol(&store).await;

        // First sync triggers
        trigger_protocols_for_event(&store, project_id, "post_sync")
            .await
            .unwrap();

        // Immediate second sync should be debounced
        trigger_protocols_for_event(&store, project_id, "post_sync")
            .await
            .unwrap();

        // Third sync with different event should also be debounced
        // (debounce is per-protocol, not per-event)
        trigger_protocols_for_event(&store, project_id, "post_import")
            .await
            .unwrap();

        let (_, total) = store
            .list_protocol_runs(protocol.id, None, 10, 0)
            .await
            .unwrap();
        assert_eq!(total, 1, "Only the first event should trigger, rest debounced");
    }
}
