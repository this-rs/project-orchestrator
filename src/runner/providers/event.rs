//! EventProvider — internal event bus subscriber for trigger chaining.
//!
//! Subscribes to the CrudEvent broadcast channel and evaluates Event-type
//! triggers when matching events are received (e.g., plan_completed → start plan B).

use super::TriggerProvider;
use crate::events::CrudEvent;
use crate::neo4j::traits::GraphStore;
use crate::runner::models::TriggerType;
use crate::runner::trigger::TriggerEngine;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{broadcast, watch};
use tracing::{debug, error, info};

/// Event-based trigger provider for plan chaining.
///
/// Subscribes to the internal CrudEvent broadcast and matches events
/// against triggers with `trigger_type = Event`. Config format:
/// ```json
/// { "event_type": "plan_completed", "entity_id": "optional-uuid" }
/// ```
pub struct EventProvider {
    graph: Arc<dyn GraphStore>,
    engine: Arc<TriggerEngine>,
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
        engine: Arc<TriggerEngine>,
        event_rx: broadcast::Receiver<CrudEvent>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            graph,
            engine,
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
        let engine = self.engine.clone();
        let mut event_rx = self.event_rx.resubscribe();
        let mut shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            info!("EventProvider started — listening for CrudEvents");

            loop {
                tokio::select! {
                    result = event_rx.recv() => {
                        match result {
                            Ok(event) => {
                                if let Err(e) = handle_event(&graph, &engine, &event).await {
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
    engine: &TriggerEngine,
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

        // Evaluate the trigger
        match engine.evaluate_and_prepare(trigger).await? {
            Some(source) => {
                let payload = serde_json::to_value(event).ok();
                engine.record_fire(trigger, None, payload).await?;
                info!(
                    "Event trigger {} fired for plan {} on event {} (source: {:?})",
                    trigger.id, trigger.plan_id, event_type, source
                );
            }
            None => {
                debug!(
                    "Event trigger {} for plan {} matched but guards not met",
                    trigger.id, trigger.plan_id
                );
            }
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, EntityType};
    use crate::neo4j::mock::MockGraphStore;
    use crate::runner::models::Trigger;
    use chrono::Utc;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_event_provider_setup_teardown() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = Arc::new(TriggerEngine::new(mock.clone()));
        let (tx, rx) = broadcast::channel::<CrudEvent>(16);
        let provider = EventProvider::new(mock, engine, rx);

        assert_eq!(provider.provider_type(), TriggerType::Event);
        provider.setup().await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        drop(tx);
        provider.teardown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_event_fires_matching_trigger() {
        let mock = Arc::new(MockGraphStore::new());

        let plan_id = Uuid::new_v4();
        let source_plan_id = Uuid::new_v4();
        let trigger = Trigger {
            id: Uuid::new_v4(),
            plan_id,
            trigger_type: TriggerType::Event,
            config: serde_json::json!({
                "event_type": "plan_completed",
                "entity_id": source_plan_id.to_string()
            }),
            enabled: true,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
        };
        mock.create_trigger(&trigger).await.unwrap();

        let engine = Arc::new(TriggerEngine::new(mock.clone()));

        // Simulate a plan_completed event
        let event = CrudEvent {
            entity_type: EntityType::Plan,
            action: CrudAction::Updated,
            entity_id: source_plan_id.to_string(),
            related: None,
            payload: serde_json::json!({"status": "completed"}),
            timestamp: Utc::now().to_rfc3339(),
            project_id: None,
        };

        handle_event(&(mock.clone() as Arc<dyn GraphStore>), &engine, &event)
            .await
            .unwrap();

        // Verify firing was recorded
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
    }

    #[tokio::test]
    async fn test_handle_event_skips_non_matching() {
        let mock = Arc::new(MockGraphStore::new());

        let plan_id = Uuid::new_v4();
        let trigger = Trigger {
            id: Uuid::new_v4(),
            plan_id,
            trigger_type: TriggerType::Event,
            config: serde_json::json!({
                "event_type": "task_completed"
            }),
            enabled: true,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
        };
        mock.create_trigger(&trigger).await.unwrap();

        let engine = Arc::new(TriggerEngine::new(mock.clone()));

        // Send a plan_completed event — should NOT match task_completed trigger
        let event = CrudEvent {
            entity_type: EntityType::Plan,
            action: CrudAction::Updated,
            entity_id: Uuid::new_v4().to_string(),
            related: None,
            payload: serde_json::json!({"status": "completed"}),
            timestamp: Utc::now().to_rfc3339(),
            project_id: None,
        };

        handle_event(&(mock.clone() as Arc<dyn GraphStore>), &engine, &event)
            .await
            .unwrap();

        // No firing recorded
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 0);
    }
}
