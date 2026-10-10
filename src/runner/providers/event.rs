//! EventProvider — internal event bus subscriber for trigger chaining.
//!
//! Subscribes to the CrudEvent broadcast channel and evaluates Event-type
//! triggers when matching events are received (e.g., plan_completed → start plan B).

use super::TriggerProvider;
use crate::events::{CrudEvent, EntityType};
use crate::neo4j::traits::GraphStore;
use crate::runner::dispatch::{DispatchOutcome, FireRequest, TriggerDispatcher};
use crate::runner::models::TriggerType;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{broadcast, watch};
use tracing::{debug, error, info, warn};

/// Event-based trigger provider for plan chaining.
///
/// Subscribes to the internal CrudEvent broadcast and matches events
/// against triggers with `trigger_type = Event`. Config format:
/// ```json
/// { "event_type": "plan_completed", "entity_id": "optional-uuid" }
/// ```
pub struct EventProvider {
    graph: Arc<dyn GraphStore>,
    dispatcher: Arc<TriggerDispatcher>,
    event_rx: broadcast::Receiver<CrudEvent>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

impl std::fmt::Debug for EventProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventProvider").finish()
    }
}

impl EventProvider {
    pub fn new(
        graph: Arc<dyn GraphStore>,
        dispatcher: Arc<TriggerDispatcher>,
        event_rx: broadcast::Receiver<CrudEvent>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            graph,
            dispatcher,
            event_rx,
            shutdown_tx,
            shutdown_rx,
        }
    }
}

#[async_trait]
impl TriggerProvider for EventProvider {
    async fn setup(&self) -> Result<()> {
        let graph = self.graph.clone();
        let dispatcher = self.dispatcher.clone();
        let mut event_rx = self.event_rx.resubscribe();
        let mut shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            info!("EventProvider started — listening for CrudEvents");

            loop {
                tokio::select! {
                    result = event_rx.recv() => {
                        match result {
                            Ok(event) => {
                                if let Err(e) = handle_event(&graph, &dispatcher, &event).await {
                                    error!("EventProvider error handling event: {}", e);
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                tracing::warn!("EventProvider lagged {} events", n);
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                info!("EventProvider: broadcast channel closed, shutting down");
                                break;
                            }
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        info!("EventProvider shutting down");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    async fn teardown(&self) -> Result<()> {
        let _ = self.shutdown_tx.send(true);
        info!("EventProvider teardown complete");
        Ok(())
    }

    fn provider_type(&self) -> TriggerType {
        TriggerType::Event
    }
}

/// Handle a single CrudEvent: check all Event-type triggers for matches.
async fn handle_event(
    graph: &Arc<dyn GraphStore>,
    dispatcher: &TriggerDispatcher,
    event: &CrudEvent,
) -> Result<()> {
    // Build the event type string for matching (e.g., "plan_updated", "task_created")
    let event_type = format!("{:?}_{:?}", event.entity_type, event.action).to_lowercase();

    // Also build a status-aware event type from payload if available
    // (e.g., "plan_completed" when payload contains {"status": "completed"})
    let status_event_type = event
        .payload
        .get("status")
        .and_then(|s| s.as_str())
        .map(|status| format!("{:?}_{}", event.entity_type, status).to_lowercase());

    let entity_id = event.entity_id.clone();

    // Get all Event-type triggers
    let triggers = graph.list_all_triggers(Some("event")).await?;
    if triggers.is_empty() {
        return Ok(());
    }
    // The plan the event belongs to (resolved once, only when a trigger may use it).
    let mut event_plan: Option<Option<uuid::Uuid>> = None;
    // The chain depth of the run behind the event (resolved once, likewise).
    let mut origin_depth: Option<u32> = None;
    // Same signal on every instance (NATS bridge): one dispatch wins it.
    let dedupe_key = format!(
        "event:{:?}:{:?}:{}:{}",
        event.entity_type, event.action, event.entity_id, event.timestamp
    );

    for trigger in &triggers {
        // Match config.event_type against both raw event type and status-aware type
        let config_event_type = trigger
            .config
            .get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let matches = config_event_type == event_type
            || (status_event_type.as_deref() == Some(config_event_type));
        if !matches {
            continue;
        }

        // Match config.entity_id (optional filter)
        if let Some(config_entity_id) = trigger.config.get("entity_id").and_then(|v| v.as_str()) {
            if config_entity_id != entity_id {
                debug!(
                    "EventProvider: trigger {} entity_id filter mismatch ({} != {})",
                    trigger.id, config_entity_id, entity_id
                );
                continue;
            }
        }

        // An event of the trigger's own plan (its tasks, its runs) never
        // relaunches that plan: a run would trigger itself.
        if event_plan.is_none() {
            event_plan = Some(plan_of_event(graph.as_ref(), event).await);
        }
        if event_plan.flatten() == Some(trigger.plan_id) {
            debug!(
                "EventProvider: trigger {} ignores event {} of its own plan {}",
                trigger.id, event_type, trigger.plan_id
            );
            continue;
        }

        // How deep the chain of event-triggered runs this one would extend is
        // (A done → B, B done → A...): the dispatcher refuses it beyond
        // MAX_EVENT_CHAIN_DEPTH.
        if origin_depth.is_none() {
            origin_depth =
                Some(chain_depth_of_event(graph.as_ref(), event, event_plan.flatten()).await);
        }

        // Evaluate the guards, start the run (as the trigger's author: no
        // caller is behind an event), record the firing.
        let request = FireRequest {
            dedupe_key: dedupe_key.clone(),
            payload: serde_json::to_value(event).ok(),
            claims: None,
            chain_depth: Some(origin_depth.unwrap_or(0) + 1),
        };
        match dispatcher.dispatch(trigger, request).await {
            Ok(DispatchOutcome::Started { start, .. }) => {
                info!(
                    "Event trigger {} started run {} of plan {} on event {}",
                    trigger.id, start.run_id, trigger.plan_id, event_type
                );
            }
            Ok(DispatchOutcome::StartFailed { error, .. }) => {
                warn!(
                    "Event trigger {} fired on event {}, plan {} not started: {}",
                    trigger.id, event_type, trigger.plan_id, error
                );
            }
            Ok(DispatchOutcome::Duplicate) => {
                debug!(
                    "Event trigger {}: event {} already dispatched",
                    trigger.id, event_type
                );
            }
            Ok(DispatchOutcome::Skipped) => {
                debug!(
                    "Event trigger {} for plan {} matched but guards not met",
                    trigger.id, trigger.plan_id
                );
            }
            // One trigger that cannot be evaluated does not stop the others.
            Err(e) => error!("Event trigger {} dispatch error: {:#}", trigger.id, e),
        }
    }

    Ok(())
}

/// How long after its end a run still counts as the origin of its plan's
/// events (the completion of its tasks and of the plan arrive right after).
const CHAIN_ORIGIN_WINDOW_SECS: i64 = 600;

/// The chain depth of the event-triggered run `event` comes from: that run's
/// own `chain_depth`, 0 when the event comes from no run, or from one that no
/// event started (a person, a schedule, a webhook).
///
/// The run is the one the event names (a runner event), else its plan's
/// latest run if it is still running or ended within
/// [`CHAIN_ORIGIN_WINDOW_SECS`]. A graph error counts as no run: the chain is
/// then bounded by the cooldown alone for that hop.
async fn chain_depth_of_event(
    graph: &dyn GraphStore,
    event: &CrudEvent,
    event_plan: Option<uuid::Uuid>,
) -> u32 {
    let named = match event.entity_type {
        EntityType::Runner => match event.entity_id.parse::<uuid::Uuid>() {
            Ok(run_id) => graph.get_plan_run(run_id).await.ok().flatten(),
            Err(_) => None,
        },
        _ => None,
    };
    let run = match (named, event_plan) {
        (Some(run), _) => Some(run),
        (None, Some(plan_id)) => graph
            .list_plan_runs(plan_id, 1)
            .await
            .ok()
            .and_then(|runs| runs.into_iter().next())
            .filter(|run| {
                run.completed_at.is_none_or(|end| {
                    (chrono::Utc::now() - end).num_seconds() <= CHAIN_ORIGIN_WINDOW_SECS
                })
            }),
        (None, None) => None,
    };
    match run.map(|r| r.triggered_by) {
        Some(crate::runner::TriggerSource::Event { chain_depth, .. }) => chain_depth,
        _ => 0,
    }
}

/// The plan `event` belongs to: the plan itself, a task's plan, a run's plan.
/// `None` when it belongs to none (or it cannot be read).
async fn plan_of_event(graph: &dyn GraphStore, event: &CrudEvent) -> Option<uuid::Uuid> {
    let id = event.entity_id.parse::<uuid::Uuid>().ok();
    match event.entity_type {
        EntityType::Plan => id,
        EntityType::Task => graph.get_task_plan_id(id?).await.ok().flatten(),
        EntityType::Runner => match event
            .payload
            .get("plan_id")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
        {
            Some(plan_id) => Some(plan_id),
            None => graph
                .get_plan_run(id?)
                .await
                .ok()
                .flatten()
                .map(|r| r.plan_id),
        },
        _ => None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, EntityType};
    use crate::neo4j::mock::MockGraphStore;
    use crate::runner::dispatch::tests::{runnable_plan, trigger_of, RecordingStarter};
    use crate::runner::models::Trigger;
    use crate::runner::trigger::TriggerEngine;
    use chrono::Utc;
    use uuid::Uuid;

    fn dispatcher_on(
        mock: &Arc<MockGraphStore>,
    ) -> (Arc<TriggerDispatcher>, Arc<RecordingStarter>) {
        let starter = Arc::new(RecordingStarter::on(mock.clone()));
        let dispatcher = Arc::new(TriggerDispatcher::new(
            mock.clone(),
            Arc::new(TriggerEngine::new(mock.clone())),
            starter.clone(),
        ));
        (dispatcher, starter)
    }

    fn event_trigger(plan_id: Uuid, config: serde_json::Value) -> Trigger {
        let mut trigger = trigger_of(plan_id, TriggerType::Event);
        trigger.config = config;
        trigger
    }

    fn plan_completed(plan_id: Uuid) -> CrudEvent {
        CrudEvent {
            entity_type: EntityType::Plan,
            action: CrudAction::Updated,
            entity_id: plan_id.to_string(),
            related: None,
            payload: serde_json::json!({"status": "completed"}),
            timestamp: Utc::now().to_rfc3339(),
            project_id: None,
        }
    }

    #[tokio::test]
    async fn test_event_provider_setup_teardown() {
        let mock = Arc::new(MockGraphStore::new());
        let (dispatcher, _) = dispatcher_on(&mock);
        let (tx, rx) = broadcast::channel::<CrudEvent>(16);
        let provider = EventProvider::new(mock, dispatcher, rx);

        assert_eq!(provider.provider_type(), TriggerType::Event);
        provider.setup().await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        drop(tx);
        provider.teardown().await.unwrap();
    }

    #[tokio::test]
    async fn a_matching_event_starts_a_run() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let source_plan_id = Uuid::new_v4();
        let trigger = event_trigger(
            plan_id,
            serde_json::json!({
                "event_type": "plan_completed",
                "entity_id": source_plan_id.to_string()
            }),
        );
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);

        handle_event(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            &plan_completed(source_plan_id),
        )
        .await
        .unwrap();

        let calls = starter.calls.lock().await;
        assert_eq!(calls.len(), 1, "the run was started");
        assert_eq!(calls[0].0, plan_id);
        assert_eq!(
            calls[0].1,
            crate::runner::TriggerSource::Event {
                trigger_id: trigger.id,
                source_event: "plan_completed".to_string(),
                chain_depth: 1,
            }
        );
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
        let run_id = firings[0].plan_run_id.expect("plan_run_id in the firing");
        assert!(mock.get_plan_run(run_id).await.unwrap().is_some());
        assert!(firings[0].source_payload.is_some());
    }

    #[tokio::test]
    async fn an_event_start_failure_is_recorded_in_the_firing() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let trigger = event_trigger(plan_id, serde_json::json!({"event_type": "plan_completed"}));
        mock.create_trigger(&trigger).await.unwrap();
        let dispatcher = TriggerDispatcher::new(
            mock.clone(),
            Arc::new(TriggerEngine::new(mock.clone())),
            Arc::new(crate::runner::dispatch::NoPlanRunner),
        );

        handle_event(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            &plan_completed(Uuid::new_v4()),
        )
        .await
        .unwrap();

        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
        assert!(firings[0].plan_run_id.is_none());
        assert!(firings[0].start_error.is_some());
    }

    /// The tasks and the runs of the trigger's own plan never relaunch it.
    #[tokio::test]
    async fn events_of_the_target_plan_do_not_relaunch_it() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let task_id = mock.get_plan_tasks(plan_id).await.unwrap()[0].id;
        let trigger = event_trigger(plan_id, serde_json::json!({"event_type": "task_updated"}));
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);

        let own_task = CrudEvent {
            entity_type: EntityType::Task,
            action: CrudAction::Updated,
            entity_id: task_id.to_string(),
            related: None,
            payload: serde_json::json!({"status": "completed"}),
            timestamp: Utc::now().to_rfc3339(),
            project_id: None,
        };
        handle_event(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            &own_task,
        )
        .await
        .unwrap();
        assert!(starter.calls.lock().await.is_empty());
        assert!(mock
            .list_trigger_firings(trigger.id, 10)
            .await
            .unwrap()
            .is_empty());

        // The same event on a task of another plan does start it.
        let other_plan = crate::test_helpers::test_plan();
        mock.create_plan(&other_plan).await.unwrap();
        let other_task = crate::test_helpers::test_task();
        mock.create_task(other_plan.id, &other_task).await.unwrap();
        let foreign = CrudEvent {
            entity_id: other_task.id.to_string(),
            ..own_task
        };
        handle_event(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            &foreign,
        )
        .await
        .unwrap();
        assert_eq!(starter.calls.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn test_handle_event_skips_non_matching() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let trigger = event_trigger(plan_id, serde_json::json!({"event_type": "task_completed"}));
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);

        // A plan_completed event does NOT match a task_completed trigger
        handle_event(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            &plan_completed(Uuid::new_v4()),
        )
        .await
        .unwrap();

        assert!(starter.calls.lock().await.is_empty());
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 0);
    }

    /// (D) Plans whose event triggers answer each other (A done → B, B done
    /// → A) stop: the run started by an event carries the depth of the run the
    /// event came from, plus one, and the dispatcher refuses it beyond the limit.
    #[tokio::test]
    async fn a_ping_pong_of_event_triggers_stops_at_the_chain_limit() {
        use crate::runner::dispatch::MAX_EVENT_CHAIN_DEPTH;
        use crate::runner::{PlanRunStatus, RunnerState, TriggerSource};
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_b = runnable_plan(&mock, dir.path()).await;
        let (dispatcher, starter) = dispatcher_on(&mock);

        // Plan A's last run was itself started by an event, at depth `depth`;
        // it has just completed.
        let a_completes_at = |depth: u32| {
            let mock = mock.clone();
            async move {
                let plan_a = crate::test_helpers::test_plan();
                mock.create_plan(&plan_a).await.unwrap();
                let mut run = RunnerState::new(
                    Uuid::new_v4(),
                    plan_a.id,
                    1,
                    TriggerSource::Event {
                        trigger_id: Uuid::new_v4(),
                        source_event: "plan_completed".into(),
                        chain_depth: depth,
                    },
                );
                run.finalize(PlanRunStatus::Completed);
                mock.create_plan_run(&run).await.unwrap();
                plan_a.id
            }
        };

        // Within the limit: B starts, one hop deeper.
        let plan_a = a_completes_at(2).await;
        let trigger = event_trigger(
            plan_b,
            serde_json::json!({"event_type": "plan_completed", "entity_id": plan_a.to_string()}),
        );
        mock.create_trigger(&trigger).await.unwrap();
        handle_event(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            &plan_completed(plan_a),
        )
        .await
        .unwrap();
        assert!(matches!(
            starter.calls.lock().await[0].1,
            TriggerSource::Event { chain_depth: 3, .. }
        ));
        crate::runner::dispatch::tests::finish_all_runs(&mock).await;

        // At the limit: B is refused, the firing says why.
        let plan_a = a_completes_at(MAX_EVENT_CHAIN_DEPTH).await;
        let trigger = event_trigger(
            plan_b,
            serde_json::json!({"event_type": "plan_completed", "entity_id": plan_a.to_string()}),
        );
        mock.create_trigger(&trigger).await.unwrap();
        handle_event(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            &plan_completed(plan_a),
        )
        .await
        .unwrap();
        assert_eq!(starter.calls.lock().await.len(), 1, "no second run");
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert!(firings[0]
            .start_error
            .as_deref()
            .is_some_and(|e| e.contains("event chain too deep")));
    }
}
