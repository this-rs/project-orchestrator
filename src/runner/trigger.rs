//! TriggerEngine — evaluates trigger guards and fires plan runs.
//!
//! The engine checks the guards (enabled, cooldown, no active run) and records
//! firings in Neo4j. `runner::dispatch::TriggerDispatcher` starts the run.

use crate::neo4j::traits::GraphStore;
use crate::runner::models::{Trigger, TriggerFiring, TriggerSource, TriggerType};
use anyhow::Result;
use chrono::Utc;
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

/// Result of evaluating a trigger.
#[derive(Debug, Clone)]
pub enum EvalResult {
    /// Trigger should fire.
    Fire,
    /// Trigger is disabled.
    Disabled,
    /// Cooldown has not elapsed.
    CooldownActive { remaining_secs: i64 },
    /// A run is already active for this plan.
    ActiveRunExists { run_id: Uuid },
}

/// Floor of the cooldown of event and webhook triggers, whatever their
/// `cooldown_secs` (0 by default): two event triggers that answer each other's
/// plans, or a burst of deliveries, start at most one run a minute each.
/// A schedule trigger is already held to one firing per matching minute.
pub const MIN_AUTOMATIC_COOLDOWN_SECS: u64 = 60;

/// The cooldown `evaluate` applies to `trigger`.
pub fn effective_cooldown_secs(trigger: &Trigger) -> u64 {
    match trigger.trigger_type {
        TriggerType::Event | TriggerType::Webhook => {
            trigger.cooldown_secs.max(MIN_AUTOMATIC_COOLDOWN_SECS)
        }
        TriggerType::Schedule | TriggerType::Chat => trigger.cooldown_secs,
    }
}

/// Trigger evaluation engine.
///
/// Stateless — all state is in Neo4j. Each `evaluate()` call checks
/// guards and returns whether the trigger should fire.
pub struct TriggerEngine {
    graph: Arc<dyn GraphStore>,
}

impl std::fmt::Debug for TriggerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriggerEngine").finish()
    }
}

impl TriggerEngine {
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    /// Evaluate whether a trigger should fire.
    ///
    /// Checks in order:
    /// 1. Is the trigger enabled?
    /// 2. Has the cooldown elapsed since last_fired?
    /// 3. Is there no active run for this plan?
    pub async fn evaluate(&self, trigger: &Trigger) -> Result<EvalResult> {
        // 1. Check enabled
        if !trigger.enabled {
            return Ok(EvalResult::Disabled);
        }

        // 2. Check cooldown
        let cooldown_secs = effective_cooldown_secs(trigger);
        if cooldown_secs > 0 {
            if let Some(last_fired) = trigger.last_fired {
                let elapsed = (Utc::now() - last_fired).num_seconds();
                let cooldown = cooldown_secs as i64;
                if elapsed < cooldown {
                    return Ok(EvalResult::CooldownActive {
                        remaining_secs: cooldown - elapsed,
                    });
                }
            }
        }

        // 3. Check no active run exists for this plan
        let active_runs = self.graph.list_active_plan_runs().await?;
        if let Some(existing) = active_runs.iter().find(|r| r.plan_id == trigger.plan_id) {
            return Ok(EvalResult::ActiveRunExists {
                run_id: existing.run_id,
            });
        }

        Ok(EvalResult::Fire)
    }

    /// Record a firing of `trigger`: the run it started (`plan_run_id`) or
    /// why none started (`start_error`).
    ///
    /// Starting the run is the job of `runner::dispatch::TriggerDispatcher`,
    /// which calls this once the start has succeeded or failed.
    pub async fn record_fire(
        &self,
        trigger: &Trigger,
        plan_run_id: Option<Uuid>,
        source_payload: Option<serde_json::Value>,
        start_error: Option<String>,
    ) -> Result<TriggerFiring> {
        let firing = TriggerFiring {
            id: Uuid::new_v4(),
            trigger_id: trigger.id,
            plan_run_id,
            fired_at: Utc::now(),
            source_payload,
            start_error,
        };

        self.graph.record_trigger_firing(&firing).await?;

        info!(
            "Trigger {} fired for plan {} (firing: {})",
            trigger.id, trigger.plan_id, firing.id
        );

        Ok(firing)
    }

    /// Evaluate and fire a trigger if conditions are met.
    ///
    /// Returns the TriggerSource to pass to PlanRunner::start().
    pub async fn evaluate_and_prepare(&self, trigger: &Trigger) -> Result<Option<TriggerSource>> {
        match self.evaluate(trigger).await? {
            EvalResult::Fire => {
                let source = match trigger.trigger_type {
                    TriggerType::Schedule => TriggerSource::Schedule {
                        trigger_id: trigger.id,
                    },
                    TriggerType::Webhook => TriggerSource::Webhook {
                        trigger_id: trigger.id,
                        payload_hash: None,
                    },
                    TriggerType::Event => TriggerSource::Event {
                        trigger_id: trigger.id,
                        source_event: trigger
                            .config
                            .get("event_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                    },
                    TriggerType::Chat => TriggerSource::Chat { session_id: None },
                };
                Ok(Some(source))
            }
            EvalResult::Disabled => {
                warn!("Trigger {} is disabled, skipping", trigger.id);
                Ok(None)
            }
            EvalResult::CooldownActive { remaining_secs } => {
                info!(
                    "Trigger {} in cooldown ({} secs remaining), skipping",
                    trigger.id, remaining_secs
                );
                Ok(None)
            }
            EvalResult::ActiveRunExists { run_id } => {
                info!(
                    "Trigger {} skipped: active run {} exists for plan {}",
                    trigger.id, run_id, trigger.plan_id
                );
                Ok(None)
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;

    fn make_trigger(plan_id: Uuid, enabled: bool, cooldown_secs: u64) -> Trigger {
        Trigger {
            id: Uuid::new_v4(),
            plan_id,
            trigger_type: TriggerType::Schedule,
            config: serde_json::json!({"cron": "0 * * * *"}),
            enabled,
            cooldown_secs,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_evaluate_enabled_no_cooldown() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = TriggerEngine::new(mock);
        let trigger = make_trigger(Uuid::new_v4(), true, 0);

        let result = engine.evaluate(&trigger).await.unwrap();
        assert!(matches!(result, EvalResult::Fire));
    }

    #[tokio::test]
    async fn test_evaluate_disabled() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = TriggerEngine::new(mock);
        let trigger = make_trigger(Uuid::new_v4(), false, 0);

        let result = engine.evaluate(&trigger).await.unwrap();
        assert!(matches!(result, EvalResult::Disabled));
    }

    #[tokio::test]
    async fn test_evaluate_cooldown_active() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = TriggerEngine::new(mock);
        let mut trigger = make_trigger(Uuid::new_v4(), true, 3600);
        trigger.last_fired = Some(Utc::now()); // just fired

        let result = engine.evaluate(&trigger).await.unwrap();
        assert!(matches!(result, EvalResult::CooldownActive { .. }));
    }

    #[tokio::test]
    async fn test_evaluate_cooldown_expired() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = TriggerEngine::new(mock);
        let mut trigger = make_trigger(Uuid::new_v4(), true, 60);
        trigger.last_fired = Some(Utc::now() - chrono::Duration::seconds(120)); // 2 min ago

        let result = engine.evaluate(&trigger).await.unwrap();
        assert!(matches!(result, EvalResult::Fire));
    }

    #[tokio::test]
    async fn test_evaluate_active_run_exists() {
        let mock = Arc::new(MockGraphStore::new());
        let plan_id = Uuid::new_v4();

        // Create an active run for this plan
        let state =
            crate::runner::RunnerState::new(Uuid::new_v4(), plan_id, 3, TriggerSource::Manual);
        mock.create_plan_run(&state).await.unwrap();

        let engine = TriggerEngine::new(mock);
        let trigger = make_trigger(plan_id, true, 0);

        let result = engine.evaluate(&trigger).await.unwrap();
        assert!(matches!(result, EvalResult::ActiveRunExists { .. }));
    }

    #[tokio::test]
    async fn event_and_webhook_triggers_have_a_minimum_cooldown() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = TriggerEngine::new(mock);
        for trigger_type in [TriggerType::Event, TriggerType::Webhook] {
            let mut trigger = make_trigger(Uuid::new_v4(), true, 0);
            trigger.trigger_type = trigger_type.clone();
            trigger.last_fired = Some(Utc::now() - chrono::Duration::seconds(10));
            assert!(
                matches!(
                    engine.evaluate(&trigger).await.unwrap(),
                    EvalResult::CooldownActive { .. }
                ),
                "{trigger_type:?} with cooldown_secs 0 still waits {MIN_AUTOMATIC_COOLDOWN_SECS}s"
            );
            trigger.last_fired = Some(Utc::now() - chrono::Duration::seconds(61));
            assert!(matches!(
                engine.evaluate(&trigger).await.unwrap(),
                EvalResult::Fire
            ));
        }
    }

    #[tokio::test]
    async fn test_record_fire() {
        let mock = Arc::new(MockGraphStore::new());
        let trigger = make_trigger(Uuid::new_v4(), true, 0);
        mock.create_trigger(&trigger).await.unwrap();

        let engine = TriggerEngine::new(mock.clone());
        let firing = engine
            .record_fire(&trigger, None, None, None)
            .await
            .unwrap();

        assert_eq!(firing.trigger_id, trigger.id);
        assert!(firing.plan_run_id.is_none());

        // Verify firing was recorded
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);

        // Verify trigger fire_count was updated
        let updated = mock.get_trigger(trigger.id).await.unwrap().unwrap();
        assert_eq!(updated.fire_count, 1);
    }

    #[tokio::test]
    async fn test_evaluate_and_prepare_fire() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = TriggerEngine::new(mock);
        let trigger = make_trigger(Uuid::new_v4(), true, 0);

        let source = engine.evaluate_and_prepare(&trigger).await.unwrap();
        assert!(source.is_some());
        assert!(matches!(source.unwrap(), TriggerSource::Schedule { .. }));
    }

    #[tokio::test]
    async fn test_evaluate_and_prepare_skip() {
        let mock = Arc::new(MockGraphStore::new());
        let engine = TriggerEngine::new(mock);
        let trigger = make_trigger(Uuid::new_v4(), false, 0);

        let source = engine.evaluate_and_prepare(&trigger).await.unwrap();
        assert!(source.is_none());
    }
}
