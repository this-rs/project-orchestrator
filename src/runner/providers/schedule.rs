//! ScheduleProvider — cron-based trigger activation.
//!
//! Uses a simple tokio interval loop to evaluate schedule triggers.
//! Each trigger's `config.cron` contains a cron expression (5 fields, **UTC**)
//! checked, at each tick, against every minute elapsed since the previous tick
//! (so a tick that drifts or lands late misses no minute). A trigger due in one
//! of those minutes goes to the `TriggerDispatcher` with that minute as its
//! signal key: every instance reserves the same key, one run starts.

use super::TriggerProvider;
use crate::neo4j::traits::GraphStore;
use crate::runner::dispatch::{DispatchOutcome, FireRequest, TriggerDispatcher};
use crate::runner::models::{Trigger, TriggerType};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
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

            // End of the window evaluated by the previous tick.
            let mut since: Option<DateTime<Utc>> = None;
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let now = Utc::now();
                        if let Err(e) = evaluate_schedule_triggers(&graph, &dispatcher, since, now).await {
                            error!("ScheduleProvider tick error: {}", e);
                        }
                        since = Some(now);
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

/// Longest catch-up of a tick, in minutes (after a sleep or a stall, the
/// latest due minute in this window fires once; older ones are not replayed).
const MAX_CATCH_UP_MINUTES: i64 = 10;

/// Evaluate all Schedule triggers over the minutes in `(since, now]`.
///
/// A trigger due in one of these minutes goes to the dispatcher with the
/// minute as its signal key; the dispatcher checks the guards, reserves the
/// key (one start per minute, across instances), starts the plan run and
/// records the firing.
async fn evaluate_schedule_triggers(
    graph: &Arc<dyn GraphStore>,
    dispatcher: &TriggerDispatcher,
    since: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> Result<()> {
    let triggers = graph.list_all_triggers(Some("schedule")).await?;

    let mut fired_count = 0;
    for trigger in &triggers {
        let Some(minute) = due_minute(trigger, since, now) else {
            continue;
        };
        let request = FireRequest {
            dedupe_key: format!("schedule:{}", minute.to_rfc3339()),
            payload: None,
            claims: None,
        };
        match dispatcher.dispatch(trigger, request).await {
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
            Ok(DispatchOutcome::Duplicate) => {
                debug!(
                    "Schedule trigger {}: minute {} already dispatched",
                    trigger.id, minute
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

/// The minute (UTC, seconds zeroed) in which `trigger` is due within the
/// window `(since, now]`: the latest one its cron matches. The first tick
/// (`since` = `None`) looks at the minute of `now` only; a tick in the same
/// minute as the previous one finds no new minute. A missing or invalid cron
/// is never due.
fn due_minute(
    trigger: &Trigger,
    since: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> Option<DateTime<Utc>> {
    let Some(expr) = trigger.config.get("cron").and_then(|v| v.as_str()) else {
        warn!(
            "Schedule trigger {} has no config.cron, never due",
            trigger.id
        );
        return None;
    };
    if validate_cron(expr).is_err() {
        warn!(
            "Schedule trigger {}: invalid cron '{}', never due",
            trigger.id, expr
        );
        return None;
    }
    let end = floor_minute(now);
    let start = since
        .map(|s| floor_minute(s) + Duration::minutes(1))
        .unwrap_or(end)
        .max(end - Duration::minutes(MAX_CATCH_UP_MINUTES - 1));
    let mut minute = end;
    while minute >= start {
        if cron_matches(expr, minute) == Some(true) {
            return Some(minute);
        }
        minute -= Duration::minutes(1);
    }
    None
}

fn floor_minute(t: DateTime<Utc>) -> DateTime<Utc> {
    t.with_second(0)
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(t)
}

/// `Ok` when `expr` is a cron expression this provider understands (see
/// [`cron_matches`]); the error says what is expected. Checked when a schedule
/// trigger is created.
pub fn validate_cron(expr: &str) -> std::result::Result<(), String> {
    match cron_matches(expr, DateTime::<Utc>::UNIX_EPOCH) {
        Some(_) => Ok(()),
        None => Err(format!(
            "invalid cron '{expr}': expected 5 fields (minute hour day-of-month month \
             day-of-week, UTC), each '*', 'N', 'A-B', '*/S', 'A-B/S' or a comma list"
        )),
    }
}

/// Whether the 5-field cron `expr` (minute hour day-of-month month
/// day-of-week) matches `now`. `None` when `expr` is not a valid expression.
///
/// Each field takes `*`, `N`, `A-B`, `*/S`, `A-B/S` and comma lists of these;
/// day-of-week is 0-7 (0 and 7 are Sunday). As in cron, when both
/// day-of-month and day-of-week are restricted, either one matching suffices.
fn cron_matches(expr: &str, now: DateTime<Utc>) -> Option<bool> {
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
    use chrono::TimeZone;
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

    fn utc(h: u32, m: u32, s: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 10, h, m, s).unwrap()
    }

    async fn tick(
        mock: &Arc<MockGraphStore>,
        dispatcher: &TriggerDispatcher,
        since: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) {
        evaluate_schedule_triggers(
            &(mock.clone() as Arc<dyn GraphStore>),
            dispatcher,
            since,
            now,
        )
        .await
        .unwrap();
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

        tick(&mock, &dispatcher, None, Utc::now()).await;

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

    /// Two instances (or two ticks) evaluating the same minute: one run.
    #[tokio::test]
    async fn the_same_minute_dispatched_twice_starts_one_run() {
        let mock = Arc::new(MockGraphStore::new());
        let dir = tempfile::tempdir().unwrap();
        let plan_id = runnable_plan(&mock, dir.path()).await;
        let trigger = schedule_trigger(plan_id, "30 3 * * *");
        mock.create_trigger(&trigger).await.unwrap();
        let (dispatcher, starter) = dispatcher_on(&mock);
        let (other_instance, other_starter) = dispatcher_on(&mock);

        tick(&mock, &dispatcher, None, utc(3, 30, 1)).await;
        assert_eq!(starter.calls.lock().await.len(), 1);
        // The run is over: only the reservation of 03:30 stands in the way now.
        crate::runner::dispatch::tests::finish_all_runs(&mock).await;
        tick(&mock, &other_instance, None, utc(3, 30, 2)).await;

        let started = starter.calls.lock().await.len() + other_starter.calls.lock().await.len();
        assert_eq!(started, 1, "one run for one minute");
        assert_eq!(
            mock.list_trigger_firings(trigger.id, 10)
                .await
                .unwrap()
                .len(),
            1
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

        tick(&mock, &dispatcher, None, Utc::now()).await;

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

        tick(&mock, &dispatcher, None, Utc::now()).await;

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

        tick(&mock, &dispatcher, Some(utc(3, 30, 5)), utc(3, 31, 5)).await;
        assert!(starter.calls.lock().await.is_empty());
        assert!(mock
            .list_trigger_firings(trigger.id, 10)
            .await
            .unwrap()
            .is_empty());
    }

    #[test]
    fn due_minute_covers_every_minute_since_the_last_tick_once() {
        let trigger = schedule_trigger(Uuid::new_v4(), "30 3 * * *");
        // First tick: the current minute only.
        assert_eq!(
            due_minute(&trigger, None, utc(3, 30, 40)),
            Some(utc(3, 30, 0))
        );
        assert_eq!(due_minute(&trigger, None, utc(3, 31, 0)), None);
        // A late tick (03:29:59.9 → 03:31:00.1) does not skip 03:30.
        assert_eq!(
            due_minute(&trigger, Some(utc(3, 29, 59)), utc(3, 31, 0)),
            Some(utc(3, 30, 0))
        );
        // A second tick in the same minute finds no new minute.
        assert_eq!(
            due_minute(&trigger, Some(utc(3, 30, 1)), utc(3, 30, 59)),
            None
        );
        // The next tick does not see 03:30 again.
        assert_eq!(
            due_minute(&trigger, Some(utc(3, 30, 40)), utc(3, 31, 40)),
            None
        );
        // After a long stall, only the last MAX_CATCH_UP_MINUTES minutes count.
        assert_eq!(
            due_minute(&trigger, Some(utc(3, 0, 0)), utc(3, 45, 0)),
            None
        );
        assert_eq!(
            due_minute(&trigger, Some(utc(3, 0, 0)), utc(3, 39, 0)),
            Some(utc(3, 30, 0))
        );

        let mut invalid = schedule_trigger(Uuid::new_v4(), "not a cron");
        assert_eq!(due_minute(&invalid, None, utc(3, 30, 0)), None);
        invalid.config = serde_json::json!({});
        assert_eq!(due_minute(&invalid, None, utc(3, 30, 0)), None);
    }

    #[test]
    fn cron_expressions() {
        // Saturday 10 October 2026, 03:30 UTC.
        let at = utc(3, 30, 0);
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
        let sunday = Utc.with_ymd_and_hms(2026, 10, 11, 3, 30, 0).unwrap();
        assert_eq!(cron_matches("30 3 * * 7", sunday), Some(true));
        assert_eq!(cron_matches("30 3 * * 0", sunday), Some(true));
        // Invalid.
        for invalid in [
            "* * * *",
            "60 * * * *",
            "*/0 * * * *",
            "5-1 * * * *",
            "@daily",
        ] {
            assert_eq!(cron_matches(invalid, at), None, "{invalid}");
            assert!(validate_cron(invalid).is_err(), "{invalid}");
        }
        assert!(validate_cron("*/5 1-3 * * 1-5").is_ok());
    }
}
