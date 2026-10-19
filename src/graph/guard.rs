//! One gate in front of every analytics pass.
//!
//! Before it, a pass on a large project could be started by five different paths (the
//! debouncer, the post-sync background task, the startup refresh, the file watcher, the
//! post-commit hook), several for the same project at once, each holding the CPU for
//! minutes. [`GuardedAnalyticsEngine`] wraps the real engine where it is built, so every
//! path goes through it and none can bypass it:
//!
//! * **One pass per project at a time, with a single catch-up.** A call that arrives while
//!   a pass runs waits for it; when it ends, one new pass starts for the callers that
//!   arrived during the previous one, and they all share its result.
//! * **A bounded number of passes at once** across projects (`ANALYTICS_MAX_CONCURRENT_RUNS`,
//!   default 2).
//! * **A deadline** (`ANALYTICS_RUN_DEADLINE_SECS`, default 30 min): a pass that exceeds
//!   it is abandoned with an error, its slot freed.

use super::engine::{AnalyticsEngine, ProjectAnalytics};
use super::models::{FabricWeights, GraphAnalytics};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use uuid::Uuid;

pub const MAX_CONCURRENT_ENV: &str = "ANALYTICS_MAX_CONCURRENT_RUNS";
pub const RUN_DEADLINE_ENV: &str = "ANALYTICS_RUN_DEADLINE_SECS";
pub const MIN_INTERVAL_ENV: &str = "ANALYTICS_MIN_INTERVAL_SECS";

const DEFAULT_MAX_CONCURRENT: usize = 2;
const DEFAULT_DEADLINE: Duration = Duration::from_secs(30 * 60);
/// Minimum delay between two AUTOMATIC passes of the same project.
const DEFAULT_MIN_INTERVAL: Duration = Duration::from_secs(15 * 60);

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// Minimum delay between two automatic (debounced) passes of one project.
pub fn min_interval_from_env() -> Duration {
    env_u64(MIN_INTERVAL_ENV)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_MIN_INTERVAL)
}

struct Done {
    started: Instant,
    result: ProjectAnalytics,
}

/// The last finished pass of a project; its mutex is the project's turn.
type Slot = Arc<tokio::sync::Mutex<Option<Done>>>;

pub struct GuardedAnalyticsEngine {
    inner: Arc<dyn AnalyticsEngine>,
    slots: Mutex<HashMap<Uuid, Slot>>,
    permits: Arc<Semaphore>,
    deadline: Duration,
}

impl GuardedAnalyticsEngine {
    pub fn new(inner: Arc<dyn AnalyticsEngine>, max_concurrent: usize, deadline: Duration) -> Self {
        Self {
            inner,
            slots: Mutex::new(HashMap::new()),
            permits: Arc::new(Semaphore::new(max_concurrent.max(1))),
            deadline,
        }
    }

    /// Limits from the environment, with the defaults above.
    pub fn from_env(inner: Arc<dyn AnalyticsEngine>) -> Self {
        let max = env_u64(MAX_CONCURRENT_ENV)
            .map(|n| n as usize)
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_MAX_CONCURRENT);
        let deadline = env_u64(RUN_DEADLINE_ENV)
            .filter(|n| *n > 0)
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_DEADLINE);
        Self::new(inner, max, deadline)
    }

    fn slot(&self, project_id: Uuid) -> Slot {
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(project_id)
            .or_default()
            .clone()
    }

    async fn bounded<T>(&self, work: impl std::future::Future<Output = Result<T>>) -> Result<T> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| anyhow!("analytics gate closed"))?;
        tokio::time::timeout(self.deadline, work)
            .await
            .map_err(|_| anyhow!("analytics abandoned after {:?}", self.deadline))?
    }
}

#[async_trait]
impl AnalyticsEngine for GuardedAnalyticsEngine {
    async fn analyze_file_graph(&self, project_id: Uuid) -> Result<GraphAnalytics> {
        self.bounded(self.inner.analyze_file_graph(project_id))
            .await
    }

    async fn analyze_function_graph(&self, project_id: Uuid) -> Result<GraphAnalytics> {
        self.bounded(self.inner.analyze_function_graph(project_id))
            .await
    }

    async fn analyze_project(&self, project_id: Uuid) -> Result<ProjectAnalytics> {
        let arrived = Instant::now();
        let slot = self.slot(project_id);
        let mut last = slot.lock().await;
        // A pass that STARTED after this call arrived saw everything this caller wanted.
        if let Some(done) = last.as_ref() {
            if done.started >= arrived {
                return Ok(done.result.clone());
            }
        }
        let started = Instant::now();
        let result = self.bounded(self.inner.analyze_project(project_id)).await?;
        *last = Some(Done {
            started,
            result: result.clone(),
        });
        Ok(result)
    }

    async fn analyze_fabric_graph(
        &self,
        project_id: Uuid,
        weights: &FabricWeights,
    ) -> Result<GraphAnalytics> {
        self.bounded(self.inner.analyze_fabric_graph(project_id, weights))
            .await
    }

    async fn detect_processes(
        &self,
        project_id: Uuid,
    ) -> Result<Vec<crate::graph::process::Process>> {
        self.bounded(self.inner.detect_processes(project_id)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts passes, tracks how many overlap, and takes `delay` per pass.
    struct Probe {
        calls: AtomicUsize,
        running: AtomicUsize,
        peak: AtomicUsize,
        delay: Duration,
    }

    impl Probe {
        fn new(delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                running: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                delay,
            })
        }
    }

    fn empty_project() -> ProjectAnalytics {
        let empty = || {
            crate::graph::algorithms::compute_all(
                &crate::graph::models::CodeGraph::new(),
                &crate::graph::models::AnalyticsConfig::default(),
            )
        };
        ProjectAnalytics {
            file_analytics: empty(),
            function_analytics: empty(),
            computed_at: chrono::Utc::now(),
        }
    }

    #[async_trait]
    impl AnalyticsEngine for Probe {
        async fn analyze_file_graph(&self, _: Uuid) -> Result<GraphAnalytics> {
            unimplemented!()
        }
        async fn analyze_function_graph(&self, _: Uuid) -> Result<GraphAnalytics> {
            unimplemented!()
        }
        async fn analyze_project(&self, _: Uuid) -> Result<ProjectAnalytics> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            self.running.fetch_sub(1, Ordering::SeqCst);
            Ok(empty_project())
        }
        async fn analyze_fabric_graph(&self, _: Uuid, _: &FabricWeights) -> Result<GraphAnalytics> {
            unimplemented!()
        }
        async fn detect_processes(&self, _: Uuid) -> Result<Vec<crate::graph::process::Process>> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn a_burst_on_one_project_is_one_pass_and_one_catch_up() {
        let probe = Probe::new(Duration::from_millis(100));
        let guard = Arc::new(GuardedAnalyticsEngine::new(
            probe.clone(),
            4,
            Duration::from_secs(5),
        ));
        let pid = Uuid::new_v4();
        let first = tokio::spawn({
            let g = guard.clone();
            async move { g.analyze_project(pid).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        // Eight more calls arrive while the first pass runs.
        let burst: Vec<_> = (0..8)
            .map(|_| {
                let g = guard.clone();
                tokio::spawn(async move { g.analyze_project(pid).await })
            })
            .collect();
        first.await.unwrap().unwrap();
        for b in burst {
            b.await.unwrap().unwrap();
        }
        assert_eq!(
            probe.calls.load(Ordering::SeqCst),
            2,
            "the pass in flight, then ONE catch-up for the whole burst"
        );
        assert_eq!(probe.peak.load(Ordering::SeqCst), 1, "never two at once");
    }

    #[tokio::test]
    async fn passes_of_different_projects_are_capped_by_the_gate() {
        let probe = Probe::new(Duration::from_millis(80));
        let guard = Arc::new(GuardedAnalyticsEngine::new(
            probe.clone(),
            2,
            Duration::from_secs(5),
        ));
        let runs: Vec<_> = (0..6)
            .map(|_| {
                let g = guard.clone();
                tokio::spawn(async move { g.analyze_project(Uuid::new_v4()).await })
            })
            .collect();
        for r in runs {
            r.await.unwrap().unwrap();
        }
        assert_eq!(probe.calls.load(Ordering::SeqCst), 6);
        assert_eq!(probe.peak.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_pass_over_the_deadline_is_abandoned_and_frees_its_slot() {
        let probe = Probe::new(Duration::from_secs(10));
        let guard = GuardedAnalyticsEngine::new(probe, 1, Duration::from_millis(50));
        let err = guard
            .analyze_project(Uuid::new_v4())
            .await
            .expect_err("abandoned");
        assert!(err.to_string().contains("abandoned"), "{err}");
        // The permit is back: another project can run (and also hits the deadline, not a hang).
        assert!(guard.analyze_project(Uuid::new_v4()).await.is_err());
    }
}
