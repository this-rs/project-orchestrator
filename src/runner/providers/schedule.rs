//! ScheduleProvider — cron-based trigger activation.
//!
//! Uses a simple tokio interval loop to evaluate schedule triggers.
//! Each trigger's `config.cron` contains a cron expression that is parsed
//! and checked against the current time at each tick.

use super::TriggerProvider;
use crate::neo4j::traits::GraphStore;
use crate::runner::models::TriggerType;
use crate::runner::trigger::TriggerEngine;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::watch;
use tracing::{debug, error, info};

/// Schedule-based trigger provider.
///
/// On `setup()`, spawns a background task that periodically evaluates
/// all enabled Schedule triggers. The tick interval defaults to 60s.
pub struct ScheduleProvider {
    graph: Arc<dyn GraphStore>,
    engine: Arc<TriggerEngine>,
    tick_interval_secs: u64,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

impl std::fmt::Debug for ScheduleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScheduleProvider")
            .field("tick_interval_secs", &self.tick_interval_secs)
            .finish()
    }
}

impl ScheduleProvider {
    pub fn new(
        graph: Arc<dyn GraphStore>,
        engine: Arc<TriggerEngine>,
        tick_interval_secs: Option<u64>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            graph,
            engine,
            tick_interval_secs: tick_interval_secs.unwrap_or(60),
            shutdown_tx,
            shutdown_rx,
        }
    }
}

#[async_trait]
impl TriggerProvider for ScheduleProvider {
    async fn setup(&self) -> Result<()> {
        let graph = self.graph.clone();
        let engine = self.engine.clone();
        let tick = self.tick_interval_secs;
        let mut shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(tick));
            info!("ScheduleProvider started (tick interval: {}s)", tick);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = evaluate_schedule_triggers(&graph, &engine).await {
                            error!("ScheduleProvider tick error: {}", e);
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        info!("ScheduleProvider shutting down");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    async fn teardown(&self) -> Result<()> {
        let _ = self.shutdown_tx.send(true);
        info!("ScheduleProvider teardown complete");
        Ok(())
    }

    fn provider_type(&self) -> TriggerType {
        TriggerType::Schedule
    }
}

/// Evaluate all enabled Schedule triggers.
///
/// For each trigger, checks if it should fire via TriggerEngine::evaluate_and_prepare().
/// If it fires, records the firing. The actual plan run is NOT started here —
/// that's the responsibility of a higher-level coordinator.
async fn evaluate_schedule_triggers(
    graph: &Arc<dyn GraphStore>,
    engine: &TriggerEngine,
) -> Result<()> {
    let triggers = graph.list_all_triggers(Some("schedule")).await?;

    let mut fired_count = 0;
    for trigger in &triggers {
        match engine.evaluate_and_prepare(trigger).await? {
            Some(source) => {
                engine.record_fire(trigger, None, None).await?;
                fired_count += 1;
                info!(
                    "Schedule trigger {} fired for plan {} (source: {:?})",
                    trigger.id, trigger.plan_id, source
                );
            }
            None => {
                debug!(
                    "Schedule trigger {} for plan {} skipped (guards not met)",
                    trigger.id, trigger.plan_id
                );
            }
        }
    }

    if fired_count > 0 {
        info!("ScheduleProvider tick: {} trigger(s) fired", fired_count);
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
    use crate::runner::models::Trigger;
    use chrono::Utc;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_schedule_provider_setup_teardown() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = Arc::new(TriggerEngine::new(mock.clone()));
        let provider = ScheduleProvider::new(mock, engine, Some(1));

        assert_eq!(provider.provider_type(), TriggerType::Schedule);
        provider.setup().await.unwrap();
        // Give the background task a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        provider.teardown().await.unwrap();
    }

    #[tokio::test]
    async fn test_evaluate_schedule_triggers_fires() {
        let mock = Arc::new(MockGraphStore::new());

        let trigger = Trigger {
            id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            trigger_type: TriggerType::Schedule,
            config: serde_json::json!({"cron": "* * * * *"}),
            enabled: true,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
        };
        mock.create_trigger(&trigger).await.unwrap();

        let engine = Arc::new(TriggerEngine::new(mock.clone()));

        evaluate_schedule_triggers(&(mock.clone() as Arc<dyn GraphStore>), &engine)
            .await
            .unwrap();

        // Verify firing was recorded
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
    }

    #[tokio::test]
    async fn test_evaluate_schedule_triggers_disabled() {
        let mock = Arc::new(MockGraphStore::new());

        let trigger = Trigger {
            id: Uuid::new_v4(),
            plan_id: Uuid::new_v4(),
            trigger_type: TriggerType::Schedule,
            config: serde_json::json!({"cron": "* * * * *"}),
            enabled: false,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
        };
        mock.create_trigger(&trigger).await.unwrap();

        let engine = Arc::new(TriggerEngine::new(mock.clone()));

        evaluate_schedule_triggers(&(mock.clone() as Arc<dyn GraphStore>), &engine)
            .await
            .unwrap();

        // No firing for disabled trigger
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 0);
    }
}
