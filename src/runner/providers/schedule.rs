//! ScheduleProvider — cron-based trigger activation.
//!
//! Uses a simple tokio interval loop to evaluate schedule triggers.
//! Each trigger's `config.cron` contains a cron expression (5 fields, local
//! time) that is checked against the current minute at each tick; a trigger
//! whose expression matches, and that has not fired in this minute yet, goes
//! to the `TriggerDispatcher`, which starts the plan run.

use super::TriggerProvider;
use crate::neo4j::traits::GraphStore;
use crate::runner::dispatch::{DispatchOutcome, TriggerDispatcher};
use crate::runner::models::{Trigger, TriggerType};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Datelike, Local, Timelike};
use std::sync::Arc;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Schedule-based trigger provider.
///
/// On `setup()`, spawns a background task that periodically evaluates
/// all enabled Schedule triggers. The tick interval defaults to 60s.
pub struct ScheduleProvider {
    graph: Arc<dyn GraphStore>,
    dispatcher: Arc<TriggerDispatcher>,
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
        dispatcher: Arc<TriggerDispatcher>,
        tick_interval_secs: Option<u64>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            graph,
            dispatcher,
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
        let dispatcher = self.dispatcher.clone();
        let tick = self.tick_interval_secs;
        let mut shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(tick));
            info!("ScheduleProvider started (tick interval: {}s)", tick);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = evaluate_schedule_triggers(&graph, &dispatcher, Local::now()).await {
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

/// Evaluate all Schedule triggers at `now`.
///
/// A trigger whose `config.cron` matches the minute of `now` and that has not
/// fired in that minute yet goes to the dispatcher, which checks the guards,
/// starts the plan run and records the firing.
async fn evaluate_schedule_triggers(
    graph: &Arc<dyn GraphStore>,
    dispatcher: &TriggerDispatcher,
    now: DateTime<Local>,
) -> Result<()> {
    let triggers = graph.list_all_triggers(Some("schedule")).await?;

    let mut fired_count = 0;
    for trigger in &triggers {
        if !is_due(trigger, now) {
            continue;
        }
        match dispatcher.dispatch(trigger, None).await {
            Ok(DispatchOutcome::Started { start, .. }) => {
                fired_count += 1;
                info!(
                    "Schedule trigger {} started run {} of plan {}",
                    trigger.id, start.run_id, trigger.plan_id
                );
            }
            Ok(DispatchOutcome::StartFailed { error, .. }) => {
                fired_count += 1;
                warn!(
                    "Schedule trigger {} fired, plan {} not started: {}",
                    trigger.id, trigger.plan_id, error
                );
            }
            Ok(DispatchOutcome::Skipped) => {
                debug!(
                    "Schedule trigger {} for plan {} skipped (guards not met)",
                    trigger.id, trigger.plan_id
                );
            }
            // One trigger that cannot be evaluated does not stop the others.
            Err(e) => error!("Schedule trigger {} dispatch error: {:#}", trigger.id, e),
        }
    }

    if fired_count > 0 {
        info!("ScheduleProvider tick: {} trigger(s) fired", fired_count);
    }

    Ok(())
}

/// Whether `trigger` is due at `now`: its cron matches the minute of `now` and
/// it has not fired in that minute (a tick may land twice in one minute).
/// A missing or invalid cron is never due.
fn is_due(trigger: &Trigger, now: DateTime<Local>) -> bool {
    let Some(expr) = trigger.config.get("cron").and_then(|v| v.as_str()) else {
        warn!(
            "Schedule trigger {} has no config.cron, never due",
            trigger.id
        );
        return false;
    };
    match cron_matches(expr, now) {
        Some(true) => {}
        Some(false) => return false,
        None => {
            warn!(
                "Schedule trigger {}: invalid cron '{}', never due",
                trigger.id, expr
            );
            return false;
        }
    }
    let minute = |t: DateTime<Local>| t.with_second(0).and_then(|t| t.with_nanosecond(0));
    match trigger.last_fired {
        Some(last) => minute(last.with_timezone(&Local)) != minute(now),
        None => true,
    }
}

/// Whether the 5-field cron `expr` (minute hour day-of-month month
/// day-of-week) matches `now`. `None` when `expr` is not a valid expression.
///
/// Each field takes `*`, `N`, `A-B`, `*/S`, `A-B/S` and comma lists of these;
/// day-of-week is 0-7 (0 and 7 are Sunday). As in cron, when both
/// day-of-month and day-of-week are restricted, either one matching suffices.
fn cron_matches(expr: &str, now: DateTime<Local>) -> Option<bool> {
    let fields: Vec<&str> = expr.split_whitespace().collect();
    let [minute, hour, dom, month, dow] = fields.as_slice() else {
        return None;
    };
    let minute_ok = field_matches(minute, now.minute(), 0, 59)?;
    let hour_ok = field_matches(hour, now.hour(), 0, 23)?;
    let month_ok = field_matches(month, now.month(), 1, 12)?;
    let dom_ok = field_matches(dom, now.day(), 1, 31)?;
    let weekday = now.weekday().num_days_from_sunday();
    let dow_ok =
        field_matches(dow, weekday, 0, 7)? || (weekday == 0 && field_matches(dow, 7, 0, 7)?);
    let day_ok = match (*dom == "*", *dow == "*") {
        (false, false) => dom_ok || dow_ok,
        _ => dom_ok && dow_ok,
    };
    Some(minute_ok && hour_ok && month_ok && day_ok)
}

/// Whether `value` is in the cron field `field` (bounds `min..=max`).
fn field_matches(field: &str, value: u32, min: u32, max: u32) -> Option<bool> {
    let mut matched = false;
    for part in field.split(',') {
        let (range, step) = match part.split_once('/') {
            Some((range, step)) => (range, step.parse::<u32>().ok().filter(|s| *s > 0)?),
            None => (part, 1),
        };
        let (lo, hi) = if range == "*" {
            (min, max)
        } else if let Some((a, b)) = range.split_once('-') {
            (a.parse().ok()?, b.parse().ok()?)
        } else {
            let n: u32 = range.parse().ok()?;
            // `N/S` means from N to the end, stepping S.
            (n, if part.contains('/') { max } else { n })
        };
        if lo < min || hi > max || lo > hi {
            return None;
        }
        if value >= lo && value <= hi && (value - lo).is_multiple_of(step) {
            matched = true;
        }
    }
    Some(matched)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::runner::dispatch::tests::{runnable_plan, trigger_of, RecordingStarter};
    use crate::runner::trigger::TriggerEngine;
    use chrono::{TimeZone, Utc};
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

    fn schedule_trigger(plan_id: Uuid, cron: &str) -> Trigger {
        let mut trigger = trigger_of(plan_id, TriggerType::Schedule);
        trigger.config = serde_json::json!({ "cron": cron });
        trigger
    }

    #[tokio::test]
    async fn test_schedule_provider_setup_teardown() {
        let mock = Arc::new(MockGraphStore::new());
        let (dispatcher, _) = dispatcher_on(&mock);
        let provider = ScheduleProvider::new(mock, dispatcher, Some(1));

        assert_eq!(provider.provider_type(), TriggerType::Schedule);
        provider.setup().await.unwrap();
        // Give the background task a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        provider.teardown().await.unwrap();
    }

    #[tokio::test]
    async fn a_due_schedule_trigger_starts_a_run() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let trigger = schedule_trigger(plan_id, "* * * * *");
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);

        evaluate_schedule_triggers(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            Local::now(),
        )
        .await
        .unwrap();

        let calls = starter.calls.lock().await;
        assert_eq!(calls.len(), 1, "the run was started");
        assert_eq!(calls[0].0, plan_id);
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
        let run_id = firings[0].plan_run_id.expect("plan_run_id in the firing");
        let run = mock.get_plan_run(run_id).await.unwrap().unwrap();
        assert_eq!(
            run.triggered_by,
            crate::runner::TriggerSource::Schedule {
                trigger_id: trigger.id
            }
        );
    }

    #[tokio::test]
    async fn a_schedule_start_failure_is_recorded_in_the_firing() {
        let mock = Arc::new(MockGraphStore::new());
        // A plan without a project: nowhere to run.
        let plan = crate::test_helpers::test_plan();
        mock.create_plan(&plan).await.unwrap();
        let trigger = schedule_trigger(plan.id, "* * * * *");
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);

        evaluate_schedule_triggers(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            Local::now(),
        )
        .await
        .unwrap();

        assert!(starter.calls.lock().await.is_empty());
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
        assert!(firings[0].plan_run_id.is_none());
        assert!(firings[0]
            .start_error
            .as_deref()
            .is_some_and(|e| e.contains("has no project")));
    }

    #[tokio::test]
    async fn test_evaluate_schedule_triggers_disabled() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let mut trigger = schedule_trigger(plan_id, "* * * * *");
        trigger.enabled = false;
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);

        evaluate_schedule_triggers(
            &(mock.clone() as Arc<dyn GraphStore>),
            &dispatcher,
            Local::now(),
        )
        .await
        .unwrap();

        // No firing, no run for disabled trigger
        assert!(starter.calls.lock().await.is_empty());
        let firings = mock.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 0);
    }

    #[tokio::test]
    async fn a_schedule_trigger_off_its_cron_does_not_fire() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let trigger = schedule_trigger(plan_id, "30 3 * * *");
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);

        let at = Local.with_ymd_and_hms(2026, 10, 10, 3, 31, 5).unwrap();
        evaluate_schedule_triggers(&(mock.clone() as Arc<dyn GraphStore>), &dispatcher, at)
            .await
            .unwrap();
        assert!(starter.calls.lock().await.is_empty());
        assert!(mock
            .list_trigger_firings(trigger.id, 10)
            .await
            .unwrap()
            .is_empty());
    }

    #[test]
    fn is_due_once_per_matching_minute() {
        let at = Local.with_ymd_and_hms(2026, 10, 10, 3, 30, 40).unwrap();
        let mut trigger = schedule_trigger(Uuid::new_v4(), "30 3 * * *");
        assert!(is_due(&trigger, at));
        trigger.last_fired = Some(
            Local
                .with_ymd_and_hms(2026, 10, 10, 3, 30, 1)
                .unwrap()
                .with_timezone(&Utc),
        );
        assert!(!is_due(&trigger, at), "already fired in this minute");
        trigger.last_fired = Some(Utc::now() - chrono::Duration::days(1));
        assert!(is_due(&trigger, at));

        trigger.config = serde_json::json!({ "cron": "not a cron" });
        trigger.last_fired = None;
        assert!(!is_due(&trigger, at));
        trigger.config = serde_json::json!({});
        assert!(!is_due(&trigger, at));
    }

    #[test]
    fn cron_expressions() {
        // Saturday 10 October 2026, 03:30.
        let at = Local.with_ymd_and_hms(2026, 10, 10, 3, 30, 0).unwrap();
        assert_eq!(cron_matches("* * * * *", at), Some(true));
        assert_eq!(cron_matches("30 3 * * *", at), Some(true));
        assert_eq!(cron_matches("0 3 * * *", at), Some(false));
        assert_eq!(cron_matches("*/15 * * * *", at), Some(true));
        assert_eq!(cron_matches("*/20 * * * *", at), Some(false));
        assert_eq!(cron_matches("0,30 1-5 * * *", at), Some(true));
        assert_eq!(cron_matches("30 3 * 10 6", at), Some(true));
        assert_eq!(cron_matches("30 3 * * 1-5", at), Some(false));
        // dom and dow both restricted: either one suffices.
        assert_eq!(cron_matches("30 3 1 * 6", at), Some(true));
        assert_eq!(cron_matches("30 3 10 * 1", at), Some(true));
        assert_eq!(cron_matches("30 3 1 * 1", at), Some(false));
        // Sunday is 0 or 7.
        let sunday = Local.with_ymd_and_hms(2026, 10, 11, 3, 30, 0).unwrap();
        assert_eq!(cron_matches("30 3 * * 7", sunday), Some(true));
        assert_eq!(cron_matches("30 3 * * 0", sunday), Some(true));
        // Invalid.
        assert_eq!(cron_matches("* * * *", at), None);
        assert_eq!(cron_matches("60 * * * *", at), None);
        assert_eq!(cron_matches("*/0 * * * *", at), None);
        assert_eq!(cron_matches("5-1 * * * *", at), None);
        assert_eq!(cron_matches("@daily", at), None);
    }
}
