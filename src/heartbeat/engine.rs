//! HeartbeatEngine — scheduler that runs HeartbeatChecks at their configured intervals.
//!
//! Pattern follows ScheduleProvider: watch::channel shutdown, Arc<dyn GraphStore>, tracing logs.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::events::EventEmitter;
use crate::meilisearch::SearchStore;
use crate::neo4j::traits::GraphStore;

use super::{HeartbeatCheck, HeartbeatContext};

/// Upper bound on how long a timed-out check waits before its next attempt.
///
/// A timed-out check used to keep its old `last_run` and was retried on the
/// very next tick. For a check that cannot finish within its timeout that is
/// an endless loop: deep_maintenance ran back-to-back ~every 12s in prod
/// (165 timeouts for 1 completion over two days), re-running Louvain skill
/// evolution thousands of times a day and starving every other check, since
/// the engine runs checks sequentially.
const TIMEOUT_RETRY_BACKOFF: Duration = Duration::from_secs(10 * 60);

/// Delay before retrying a check that timed out: its own interval, capped at
/// [`TIMEOUT_RETRY_BACKOFF`] so a daily check still retries the same day.
fn timeout_retry_delay(check_interval: Duration) -> Duration {
    check_interval.min(TIMEOUT_RETRY_BACKOFF)
}

/// Whether a check is due at `now`.
///
/// `retry_at` (set after a timeout) takes precedence over the interval.
fn is_due(
    last_run: Option<Instant>,
    retry_at: Option<Instant>,
    interval: Duration,
    now: Instant,
) -> bool {
    if let Some(retry_at) = retry_at {
        return now >= retry_at;
    }
    match last_run {
        None => true,
        Some(last) => now.duration_since(last) >= interval,
    }
}

/// Background engine that periodically evaluates all registered heartbeat checks.
///
/// Each check has its own interval. The engine ticks every `tick_interval` (default 30s)
/// and evaluates any check whose interval has elapsed since its last run.
/// Checks that exceed their timeout are skipped. The default timeout is 5s,
/// but checks can override it via `HeartbeatCheck::timeout_override()`.
pub struct HeartbeatEngine {
    graph: Arc<dyn GraphStore>,
    search: Option<Arc<dyn SearchStore>>,
    emitter: Option<Arc<dyn EventEmitter>>,
    checks: Vec<Box<dyn HeartbeatCheck>>,
    tick_interval: Duration,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

impl HeartbeatEngine {
    /// Create a new engine with the given graph store and checks.
    pub fn new(
        graph: Arc<dyn GraphStore>,
        search: Option<Arc<dyn SearchStore>>,
        emitter: Option<Arc<dyn EventEmitter>>,
        checks: Vec<Box<dyn HeartbeatCheck>>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            graph,
            search,
            emitter,
            checks,
            tick_interval: Duration::from_secs(30),
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Start the engine, consuming it. Spawns a tokio task and returns
    /// a shutdown handle. The engine is leaked (like ScheduleProvider)
    /// so it runs for the lifetime of the process.
    pub fn start_owned(self) -> HeartbeatHandle {
        let graph = self.graph;
        let search = self.search;
        let emitter = self.emitter;
        let checks = self.checks;
        let tick_interval = self.tick_interval;
        let shutdown_tx = self.shutdown_tx;
        let mut shutdown_rx = self.shutdown_rx;

        let check_count = checks.len();

        tokio::spawn(async move {
            let ctx = HeartbeatContext {
                graph,
                search,
                emitter,
            };

            // Track last-run time per check
            let mut last_run: Vec<Option<Instant>> = vec![None; checks.len()];
            // Earliest retry for a check that timed out (see TIMEOUT_RETRY_BACKOFF).
            let mut retry_at: Vec<Option<Instant>> = vec![None; checks.len()];
            let mut interval = tokio::time::interval(tick_interval);
            // A tick can take minutes (checks run sequentially). Burst mode
            // would then fire the missed ticks back-to-back; delay instead.
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

            info!(
                "HeartbeatEngine started ({} checks, tick interval: {:?})",
                check_count, tick_interval
            );

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let now = Instant::now();

                        for (i, check) in checks.iter().enumerate() {
                            if !is_due(last_run[i], retry_at[i], check.interval(), now) {
                                continue;
                            }

                            let default_timeout = Duration::from_secs(5);
                            let check_timeout = check.timeout_override().unwrap_or(default_timeout);
                            debug!(
                                "HeartbeatEngine: running check '{}' (timeout: {:?})",
                                check.name(),
                                check_timeout
                            );

                            match tokio::time::timeout(check_timeout, check.run(&ctx)).await {
                                Ok(Ok(())) => {
                                    debug!("HeartbeatEngine: check '{}' completed OK", check.name());
                                    last_run[i] = Some(Instant::now());
                                    retry_at[i] = None;
                                }
                                Ok(Err(e)) => {
                                    warn!(
                                        "HeartbeatEngine: check '{}' failed: {}",
                                        check.name(),
                                        e
                                    );
                                    last_run[i] = Some(Instant::now());
                                    retry_at[i] = None;
                                }
                                Err(_) => {
                                    let delay = timeout_retry_delay(check.interval());
                                    warn!(
                                        "HeartbeatEngine: check '{}' timed out (>{:?}), retrying in {:?}",
                                        check.name(),
                                        check_timeout,
                                        delay
                                    );
                                    // last_run stays untouched (the interval
                                    // did not complete), but the retry waits.
                                    retry_at[i] = Some(Instant::now() + delay);
                                }
                            }
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        info!("HeartbeatEngine shutting down");
                        break;
                    }
                }
            }
        });

        HeartbeatHandle { shutdown_tx }
    }
}

/// Handle returned by `HeartbeatEngine::start_owned()` for graceful shutdown.
pub struct HeartbeatHandle {
    #[allow(dead_code)]
    shutdown_tx: watch::Sender<bool>,
}

impl HeartbeatHandle {
    /// Signal the engine to stop.
    #[allow(dead_code)]
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
        info!("HeartbeatEngine: shutdown signal sent");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use anyhow::Result;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct CountingCheck {
        count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl HeartbeatCheck for CountingCheck {
        fn name(&self) -> &str {
            "counting_check"
        }
        fn interval(&self) -> Duration {
            Duration::from_millis(10)
        }
        async fn run(&self, _ctx: &HeartbeatContext) -> Result<()> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn test_timeout_retry_delay_is_capped() {
        assert_eq!(
            timeout_retry_delay(Duration::from_secs(24 * 3600)),
            TIMEOUT_RETRY_BACKOFF
        );
        assert_eq!(
            timeout_retry_delay(Duration::from_secs(60)),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn test_is_due_waits_for_retry_after_timeout() {
        let now = Instant::now();
        let day = Duration::from_secs(24 * 3600);
        // Never ran, no timeout: due immediately.
        assert!(is_due(None, None, day, now));
        // Never completed, but timed out: NOT due until the backoff elapses —
        // this is the regression (it used to be retried on the next tick).
        let retry = now + Duration::from_secs(600);
        assert!(!is_due(None, Some(retry), day, now));
        assert!(!is_due(
            None,
            Some(retry),
            day,
            now + Duration::from_secs(599)
        ));
        assert!(is_due(
            None,
            Some(retry),
            day,
            now + Duration::from_secs(600)
        ));
        // Completed recently: not due until the interval elapses.
        assert!(!is_due(
            Some(now),
            None,
            day,
            now + Duration::from_secs(3600)
        ));
        assert!(is_due(Some(now), None, day, now + day));
    }

    struct NeverFinishingCheck {
        count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl HeartbeatCheck for NeverFinishingCheck {
        fn name(&self) -> &str {
            "slow_check"
        }
        fn interval(&self) -> Duration {
            Duration::from_secs(3600)
        }
        fn timeout_override(&self) -> Option<Duration> {
            Some(Duration::from_millis(5))
        }
        async fn run(&self, _ctx: &HeartbeatContext) -> Result<()> {
            self.count.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_engine_does_not_hot_loop_a_timed_out_check() {
        let count = Arc::new(AtomicU32::new(0));
        let graph = Arc::new(MockGraphStore::new());
        let mut engine = HeartbeatEngine::new(
            graph,
            None,
            None,
            vec![Box::new(NeverFinishingCheck {
                count: count.clone(),
            })],
        );
        engine.tick_interval = Duration::from_millis(10);
        let handle = engine.start_owned();
        // Wait for the first run (bounded: slow CI / coverage builds), then
        // let ~20 more ticks elapse.
        for _ in 0..500 {
            if count.load(Ordering::SeqCst) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        handle.shutdown();
        // The timed-out check must have run exactly once: its retry waits
        // for the backoff instead of the next tick.
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    struct FailingCheck;

    #[async_trait]
    impl HeartbeatCheck for FailingCheck {
        fn name(&self) -> &str {
            "failing_check"
        }
        fn interval(&self) -> Duration {
            Duration::from_millis(10)
        }
        async fn run(&self, _ctx: &HeartbeatContext) -> Result<()> {
            anyhow::bail!("intentional failure")
        }
    }

    #[allow(dead_code)]
    struct SlowCheck;

    #[async_trait]
    impl HeartbeatCheck for SlowCheck {
        fn name(&self) -> &str {
            "slow_check"
        }
        fn interval(&self) -> Duration {
            Duration::from_millis(10)
        }
        async fn run(&self, _ctx: &HeartbeatContext) -> Result<()> {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(())
        }
    }

    #[test]
    fn test_engine_new() {
        let graph = Arc::new(MockGraphStore::new());
        let engine = HeartbeatEngine::new(graph, None, None, vec![]);
        assert_eq!(engine.checks.len(), 0);
        assert_eq!(engine.tick_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_engine_with_checks() {
        let graph = Arc::new(MockGraphStore::new());
        let count = Arc::new(AtomicU32::new(0));
        let checks: Vec<Box<dyn HeartbeatCheck>> = vec![Box::new(CountingCheck {
            count: count.clone(),
        })];
        let engine = HeartbeatEngine::new(graph, None, None, checks);
        assert_eq!(engine.checks.len(), 1);
    }

    #[tokio::test]
    async fn test_engine_shutdown() {
        let graph = Arc::new(MockGraphStore::new());
        let count = Arc::new(AtomicU32::new(0));
        let checks: Vec<Box<dyn HeartbeatCheck>> = vec![Box::new(CountingCheck {
            count: count.clone(),
        })];

        let mut engine = HeartbeatEngine::new(graph, None, None, checks);
        engine.tick_interval = Duration::from_millis(10);
        let handle = engine.start_owned();

        // Let it run a few ticks
        tokio::time::sleep(Duration::from_millis(100)).await;
        handle.shutdown();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Should have run at least once
        assert!(count.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_engine_failing_check_doesnt_crash() {
        let graph = Arc::new(MockGraphStore::new());
        let checks: Vec<Box<dyn HeartbeatCheck>> = vec![Box::new(FailingCheck)];

        let mut engine = HeartbeatEngine::new(graph, None, None, checks);
        engine.tick_interval = Duration::from_millis(10);
        let handle = engine.start_owned();

        // Engine should survive failing checks
        tokio::time::sleep(Duration::from_millis(100)).await;
        handle.shutdown();
    }

    #[tokio::test]
    async fn test_handle_shutdown_idempotent() {
        let graph = Arc::new(MockGraphStore::new());
        let engine = HeartbeatEngine::new(graph, None, None, vec![]);
        let handle = engine.start_owned();

        handle.shutdown();
        handle.shutdown(); // Should not panic
    }
}
