//! Analytics debouncer for incremental sync.
//!
//! Coalesces rapid-fire analytics triggers (e.g., from file watcher events)
//! into a single `analyze_project` call after a configurable quiet period.
//!
//! ## Design
//!
//! Uses an mpsc channel + timeout loop:
//! 1. `trigger(project_id)` sends a non-blocking message
//! 2. Background task waits for the first trigger, then keeps consuming
//!    triggers until `debounce_ms` of silence (no new triggers)
//! 3. After the quiet period, runs `analyze_project` synchronously
//! 4. The loop is sequential — no concurrent analytics computations
//!
//! ## Usage
//!
//! ```ignore
//! let debouncer = AnalyticsDebouncer::new(analytics_engine, 2000);
//! debouncer.trigger(project_id); // non-blocking
//! ```

use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::engine::AnalyticsEngine;
use crate::neo4j::traits::GraphStore;
use std::sync::Arc;

/// Debounced analytics trigger for incremental sync scenarios.
///
/// Coalesces multiple trigger calls into a single `analyze_project` invocation
/// after a configurable quiet period. Thread-safe and non-blocking.
pub struct AnalyticsDebouncer {
    trigger_tx: mpsc::Sender<Uuid>,
}

impl AnalyticsDebouncer {
    /// Create a new debouncer that waits `debounce_ms` of silence before computing.
    ///
    /// Spawns a background tokio task that lives until the debouncer is dropped.
    /// Optionally accepts a `GraphStore` to update `analytics_computed_at` after success.
    pub fn new(analytics: Arc<dyn AnalyticsEngine>, debounce_ms: u64) -> Self {
        Self::with_graph_store(analytics, debounce_ms, None)
    }

    /// Create a debouncer that also updates `analytics_computed_at` on the project
    /// after a successful analytics computation.
    pub fn with_graph_store(
        analytics: Arc<dyn AnalyticsEngine>,
        debounce_ms: u64,
        graph_store: Option<Arc<dyn GraphStore>>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Uuid>(64);
        tokio::spawn(Self::run_loop(analytics, rx, debounce_ms, graph_store));
        Self { trigger_tx: tx }
    }

    /// Trigger an analytics recomputation for the given project.
    ///
    /// Non-blocking: returns immediately. If the channel is full, the trigger
    /// is silently dropped (the pending trigger will still fire).
    pub fn trigger(&self, project_id: Uuid) {
        let _ = self.trigger_tx.try_send(project_id);
    }

    /// Background loop: debounce triggers and compute analytics.
    async fn run_loop(
        analytics: Arc<dyn AnalyticsEngine>,
        mut rx: mpsc::Receiver<Uuid>,
        debounce_ms: u64,
        graph_store: Option<Arc<dyn GraphStore>>,
    ) {
        let debounce = Duration::from_millis(debounce_ms);

        loop {
            // Wait for the first trigger
            let pid = match rx.recv().await {
                Some(pid) => pid,
                None => break, // channel closed, debouncer dropped
            };
            let mut last_pid = pid;

            // Debounce: keep consuming triggers until quiet period
            loop {
                match tokio::time::timeout(debounce, rx.recv()).await {
                    Ok(Some(pid)) => {
                        last_pid = pid; // new trigger arrived, reset timer
                    }
                    Ok(None) => return, // channel closed
                    Err(_) => break,    // timeout = quiet period elapsed
                }
            }

            // Compute analytics (sequential — no concurrent computation)
            let start = std::time::Instant::now();
            match analytics.analyze_project(last_pid).await {
                Ok(result) => {
                    tracing::info!(
                        "Debounced analytics computed for project {} in {:?} (files: {} nodes, functions: {} nodes)",
                        last_pid,
                        start.elapsed(),
                        result.file_analytics.node_count,
                        result.function_analytics.node_count,
                    );
                    // Update analytics_computed_at timestamp on the project
                    if let Some(ref store) = graph_store {
                        if let Err(e) = store.update_project_analytics_timestamp(last_pid).await {
                            tracing::warn!(
                                "Failed to update analytics_computed_at for project {}: {}",
                                last_pid,
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Debounced analytics failed for project {}: {}", last_pid, e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::mock::MockAnalyticsEngine;
    use crate::graph::models::{CodeHealthReport, GraphAnalytics};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// MockAnalyticsEngine with a call counter
    struct CountingAnalyticsEngine {
        call_count: Arc<AtomicU32>,
        inner: MockAnalyticsEngine,
    }

    impl CountingAnalyticsEngine {
        fn new(call_count: Arc<AtomicU32>) -> Self {
            let analytics = GraphAnalytics {
                metrics: HashMap::new(),
                communities: vec![],
                components: vec![],
                health: CodeHealthReport::default(),
                modularity: 0.0,
                node_count: 5,
                edge_count: 3,
                computation_ms: 1,
            };
            Self {
                call_count,
                inner: MockAnalyticsEngine::with_results(analytics.clone(), analytics),
            }
        }
    }

    #[async_trait::async_trait]
    impl AnalyticsEngine for CountingAnalyticsEngine {
        async fn analyze_file_graph(&self, project_id: Uuid) -> anyhow::Result<GraphAnalytics> {
            self.inner.analyze_file_graph(project_id).await
        }

        async fn analyze_function_graph(&self, project_id: Uuid) -> anyhow::Result<GraphAnalytics> {
            self.inner.analyze_function_graph(project_id).await
        }

        async fn analyze_project(
            &self,
            project_id: Uuid,
        ) -> anyhow::Result<crate::graph::engine::ProjectAnalytics> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            self.inner.analyze_project(project_id).await
        }
    }

    #[tokio::test]
    async fn test_debounce_coalesces_rapid_triggers() {
        let count = Arc::new(AtomicU32::new(0));
        let engine: Arc<dyn AnalyticsEngine> =
            Arc::new(CountingAnalyticsEngine::new(count.clone()));

        let debouncer = AnalyticsDebouncer::new(engine, 100); // 100ms debounce
        let pid = Uuid::new_v4();

        // Fire 10 triggers in rapid succession
        for _ in 0..10 {
            debouncer.trigger(pid);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Wait for debounce period + processing
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Should have coalesced into exactly 1 call
        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "10 rapid triggers should coalesce into 1 analytics call"
        );
    }

    #[tokio::test]
    async fn test_debounce_separate_bursts() {
        let count = Arc::new(AtomicU32::new(0));
        let engine: Arc<dyn AnalyticsEngine> =
            Arc::new(CountingAnalyticsEngine::new(count.clone()));

        let debouncer = AnalyticsDebouncer::new(engine, 50); // 50ms debounce
        let pid = Uuid::new_v4();

        // First burst
        debouncer.trigger(pid);
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Second burst (after first completed)
        debouncer.trigger(pid);
        tokio::time::sleep(Duration::from_millis(150)).await;

        assert_eq!(
            count.load(Ordering::SeqCst),
            2,
            "Two separate bursts should produce 2 analytics calls"
        );
    }

    #[tokio::test]
    async fn test_debounce_no_trigger_no_call() {
        let count = Arc::new(AtomicU32::new(0));
        let engine: Arc<dyn AnalyticsEngine> =
            Arc::new(CountingAnalyticsEngine::new(count.clone()));

        let _debouncer = AnalyticsDebouncer::new(engine, 50);

        // Don't trigger anything
        tokio::time::sleep(Duration::from_millis(200)).await;

        assert_eq!(
            count.load(Ordering::SeqCst),
            0,
            "No triggers should produce 0 analytics calls"
        );
    }
}
