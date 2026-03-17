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
            let mut interval = tokio::time::interval(tick_interval);

            info!(
                "HeartbeatEngine started ({} checks, tick interval: {:?})",
                check_count, tick_interval
            );

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let now = Instant::now();

                        for (i, check) in checks.iter().enumerate() {
                            let should_run = match last_run[i] {
                                None => true,
                                Some(last) => now.duration_since(last) >= check.interval(),
                            };

                            if !should_run {
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
                                }
                                Ok(Err(e)) => {
                                    warn!(
                                        "HeartbeatEngine: check '{}' failed: {}",
                                        check.name(),
                                        e
                                    );
                                    last_run[i] = Some(Instant::now());
                                }
                                Err(_) => {
                                    warn!(
                                        "HeartbeatEngine: check '{}' timed out (>{:?}), skipping",
                                        check.name(),
                                        check_timeout
                                    );
                                    // Don't update last_run — retry next tick
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
