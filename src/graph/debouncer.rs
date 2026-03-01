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
//! 3. After the quiet period, runs `analyze_project` **sequentially** for
//!    each unique project collected during the debounce window
//! 4. The loop is sequential — no concurrent analytics computations
//!
//! ## Usage
//!
//! ```ignore
//! let debouncer = AnalyticsDebouncer::new(analytics_engine, 2000);
//! debouncer.trigger(project_id); // non-blocking
//! ```

use std::collections::HashSet;
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
    ///
    /// Collects ALL unique project IDs during the debounce window, then
    /// processes them **sequentially** (one at a time). This ensures:
    /// - No concurrent analytics computations (avoids 2× heap pressure)
    /// - All triggered projects are processed (not just the last one)
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

            // Collect all unique project IDs during debounce window
            let mut pending_projects = HashSet::new();
            pending_projects.insert(pid);

            // Debounce: keep consuming triggers until quiet period
            loop {
                match tokio::time::timeout(debounce, rx.recv()).await {
                    Ok(Some(pid)) => {
                        pending_projects.insert(pid); // collect all unique projects
                    }
                    Ok(None) => return, // channel closed
                    Err(_) => break,    // timeout = quiet period elapsed
                }
            }

            // Process each project sequentially (no concurrent computation)
            if pending_projects.len() > 1 {
                tracing::info!(
                    "Debounced analytics: processing {} projects sequentially",
                    pending_projects.len()
                );
            }

            for project_id in &pending_projects {
                let start = std::time::Instant::now();
                match analytics.analyze_project(*project_id).await {
                    Ok(result) => {
                        tracing::info!(
                            "Debounced analytics computed for project {} in {:?} (files: {} nodes, functions: {} nodes)",
                            project_id,
                            start.elapsed(),
                            result.file_analytics.node_count,
                            result.function_analytics.node_count,
                        );
                        // Update analytics_computed_at timestamp on the project
                        if let Some(ref store) = graph_store {
                            if let Err(e) =
                                store.update_project_analytics_timestamp(*project_id).await
                            {
                                tracing::warn!(
                                    "Failed to update analytics_computed_at for project {}: {}",
                                    project_id,
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Debounced analytics failed for project {}: {}",
                            project_id,
                            e
                        );
                    }
                }
            }
        }
    }
}

// ============================================================================
// CoChangeDebouncer — debounced CO_CHANGED recomputation
// ============================================================================

/// Debounced CO_CHANGED computation trigger.
///
/// Same pattern as `AnalyticsDebouncer` but with a longer debounce period (30s default)
/// and calls `compute_co_changed` instead of `analyze_project`.
///
/// ## Usage
///
/// ```ignore
/// let debouncer = CoChangeDebouncer::new(graph_store, 30_000);
/// debouncer.trigger(project_id); // non-blocking
/// ```
pub struct CoChangeDebouncer {
    trigger_tx: mpsc::Sender<Uuid>,
}

impl CoChangeDebouncer {
    /// Create a new CO_CHANGED debouncer.
    ///
    /// `debounce_ms` is the quiet period before triggering computation (default: 30_000ms).
    pub fn new(graph_store: Arc<dyn GraphStore>, debounce_ms: u64) -> Self {
        let (tx, rx) = mpsc::channel::<Uuid>(64);
        tokio::spawn(Self::run_loop(graph_store, rx, debounce_ms));
        Self { trigger_tx: tx }
    }

    /// Trigger a CO_CHANGED recomputation for the given project.
    ///
    /// Non-blocking: returns immediately. If the channel is full, the trigger
    /// is silently dropped (a pending trigger will still fire).
    pub fn trigger(&self, project_id: Uuid) {
        let _ = self.trigger_tx.try_send(project_id);
    }

    /// Background loop: debounce triggers and compute CO_CHANGED.
    async fn run_loop(
        graph_store: Arc<dyn GraphStore>,
        mut rx: mpsc::Receiver<Uuid>,
        debounce_ms: u64,
    ) {
        let debounce = Duration::from_millis(debounce_ms);

        loop {
            // Wait for the first trigger
            let pid = match rx.recv().await {
                Some(pid) => pid,
                None => break, // channel closed
            };

            // Collect all unique project IDs during debounce window
            let mut pending_projects = HashSet::new();
            pending_projects.insert(pid);

            // Debounce: keep consuming triggers until quiet period
            loop {
                match tokio::time::timeout(debounce, rx.recv()).await {
                    Ok(Some(pid)) => {
                        pending_projects.insert(pid);
                    }
                    Ok(None) => return, // channel closed
                    Err(_) => break,    // timeout = quiet period elapsed
                }
            }

            // Process each project sequentially
            if pending_projects.len() > 1 {
                tracing::info!(
                    "Debounced CO_CHANGED: processing {} projects sequentially",
                    pending_projects.len()
                );
            }

            for project_id in &pending_projects {
                let start = std::time::Instant::now();

                // Read last_co_change_computed_at for incremental computation
                let since = graph_store
                    .get_project(*project_id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|p| p.last_co_change_computed_at);

                match graph_store
                    .compute_co_changed(*project_id, since, 3, 500)
                    .await
                {
                    Ok(count) => {
                        tracing::info!(
                            "Debounced CO_CHANGED computed for project {} in {:?} ({} relations)",
                            project_id,
                            start.elapsed(),
                            count,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Debounced CO_CHANGED failed for project {}: {}",
                            project_id,
                            e
                        );
                    }
                }
            }
        }
    }
}

// ============================================================================
// NeuralReinforcementDebouncer — debounced Hebbian reinforcement on commits
// ============================================================================

/// Payload for a neural reinforcement trigger.
/// Groups file paths by project so we can batch energy boosts and synapse
/// reinforcement across rapid-fire commits.
#[derive(Debug, Clone)]
pub struct ReinforcementPayload {
    pub project_id: Uuid,
    pub file_paths: Vec<String>,
}

/// Debounced neural reinforcement trigger for commit hooks.
///
/// Prevents CPU spikes during git checkout / rebase storms where hundreds of
/// commits arrive in rapid succession. Collects all file paths from multiple
/// commits during the debounce window, deduplicates them, then performs a
/// single batch of energy boosts + synapse reinforcement.
///
/// ## Design
///
/// - **Debounce**: 5s quiet period (configurable) before processing
/// - **Dedup**: file paths are deduplicated within the window (per project)
/// - **Batch**: single `get_notes_for_entity` + `boost_energy` + `reinforce_synapses`
///   pass instead of per-commit processing
///
/// ## Usage
///
/// ```ignore
/// let debouncer = NeuralReinforcementDebouncer::new(graph_store, config, 5_000);
/// debouncer.trigger(ReinforcementPayload {
///     project_id: pid,
///     file_paths: vec!["src/main.rs".into()],
/// });
/// ```
pub struct NeuralReinforcementDebouncer {
    trigger_tx: mpsc::Sender<ReinforcementPayload>,
}

impl NeuralReinforcementDebouncer {
    /// Create a new neural reinforcement debouncer.
    ///
    /// `debounce_ms` is the quiet period before processing (default: 5_000ms).
    pub fn new(
        graph_store: Arc<dyn GraphStore>,
        config: crate::neurons::AutoReinforcementConfig,
        debounce_ms: u64,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<ReinforcementPayload>(128);
        tokio::spawn(Self::run_loop(graph_store, config, rx, debounce_ms));
        Self { trigger_tx: tx }
    }

    /// Trigger a neural reinforcement for files touched by a commit.
    ///
    /// Non-blocking: returns immediately. If the channel is full, the trigger
    /// is silently dropped (a pending trigger will still fire).
    pub fn trigger(&self, payload: ReinforcementPayload) {
        let _ = self.trigger_tx.try_send(payload);
    }

    /// Background loop: debounce triggers and perform batch reinforcement.
    async fn run_loop(
        graph_store: Arc<dyn GraphStore>,
        config: crate::neurons::AutoReinforcementConfig,
        mut rx: mpsc::Receiver<ReinforcementPayload>,
        debounce_ms: u64,
    ) {
        let debounce = Duration::from_millis(debounce_ms);

        loop {
            // Wait for the first trigger
            let payload = match rx.recv().await {
                Some(p) => p,
                None => break, // channel closed
            };

            // Collect all payloads during debounce window, grouped by project
            let mut pending: std::collections::HashMap<Uuid, HashSet<String>> =
                std::collections::HashMap::new();
            pending
                .entry(payload.project_id)
                .or_default()
                .extend(payload.file_paths);

            // Debounce: keep consuming triggers until quiet period
            loop {
                match tokio::time::timeout(debounce, rx.recv()).await {
                    Ok(Some(p)) => {
                        pending
                            .entry(p.project_id)
                            .or_default()
                            .extend(p.file_paths);
                    }
                    Ok(None) => return, // channel closed
                    Err(_) => break,    // timeout = quiet period elapsed
                }
            }

            // Process each project's accumulated file paths
            for (project_id, file_paths) in &pending {
                let total_files = file_paths.len();
                let mut all_note_ids: Vec<Uuid> = Vec::new();
                let mut boost_count = 0u64;

                for file_path in file_paths {
                    match graph_store
                        .get_notes_for_entity(&crate::notes::EntityType::File, file_path)
                        .await
                    {
                        Ok(notes) => {
                            for note in &notes {
                                if let Err(e) = graph_store
                                    .boost_energy(note.id, config.commit_energy_boost)
                                    .await
                                {
                                    tracing::warn!(
                                        note_id = %note.id, error = %e,
                                        "Neural reinforcement debouncer: energy boost failed"
                                    );
                                } else {
                                    boost_count += 1;
                                }
                                all_note_ids.push(note.id);
                            }
                        }
                        Err(e) => {
                            tracing::debug!(
                                file = %file_path, error = %e,
                                "Neural reinforcement debouncer: get_entity_notes failed"
                            );
                        }
                    }
                }

                // Hebbian synapse reinforcement
                let mut synapse_count = 0usize;
                if all_note_ids.len() >= 2 {
                    all_note_ids.sort();
                    all_note_ids.dedup();
                    if all_note_ids.len() >= 2 {
                        match graph_store
                            .reinforce_synapses(&all_note_ids, config.commit_synapse_boost)
                            .await
                        {
                            Ok(count) => synapse_count = count,
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "Neural reinforcement debouncer: synapse reinforcement failed"
                                );
                            }
                        }
                    }
                }

                tracing::info!(
                    project_id = %project_id,
                    files = total_files,
                    notes_boosted = boost_count,
                    synapses_reinforced = synapse_count,
                    "Neural reinforcement debouncer: batch complete"
                );
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
                profile_name: None,
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

        async fn analyze_fabric_graph(
            &self,
            project_id: Uuid,
            weights: &crate::graph::models::FabricWeights,
        ) -> anyhow::Result<GraphAnalytics> {
            self.inner.analyze_fabric_graph(project_id, weights).await
        }

        async fn detect_processes(
            &self,
            project_id: Uuid,
        ) -> anyhow::Result<Vec<crate::graph::process::Process>> {
            self.inner.detect_processes(project_id).await
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

    #[tokio::test]
    async fn test_debounce_multiple_projects_collected() {
        // Verifies that triggers for DIFFERENT projects during the same
        // debounce window ALL get processed (not just the last one).
        let count = Arc::new(AtomicU32::new(0));
        let engine: Arc<dyn AnalyticsEngine> =
            Arc::new(CountingAnalyticsEngine::new(count.clone()));

        let debouncer = AnalyticsDebouncer::new(engine, 100); // 100ms debounce

        let pid_a = Uuid::new_v4();
        let pid_b = Uuid::new_v4();
        let pid_c = Uuid::new_v4();

        // Fire triggers for 3 different projects in rapid succession
        debouncer.trigger(pid_a);
        tokio::time::sleep(Duration::from_millis(10)).await;
        debouncer.trigger(pid_b);
        tokio::time::sleep(Duration::from_millis(10)).await;
        debouncer.trigger(pid_c);

        // Wait for debounce period + processing
        tokio::time::sleep(Duration::from_millis(300)).await;

        // All 3 projects should have been processed
        assert_eq!(
            count.load(Ordering::SeqCst),
            3,
            "3 different projects should produce 3 analytics calls"
        );
    }

    #[tokio::test]
    async fn test_debounce_duplicate_projects_deduped() {
        // Same project triggered multiple times → should only be processed once
        let count = Arc::new(AtomicU32::new(0));
        let engine: Arc<dyn AnalyticsEngine> =
            Arc::new(CountingAnalyticsEngine::new(count.clone()));

        let debouncer = AnalyticsDebouncer::new(engine, 100);

        let pid = Uuid::new_v4();

        // Same project 5 times
        for _ in 0..5 {
            debouncer.trigger(pid);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        tokio::time::sleep(Duration::from_millis(300)).await;

        assert_eq!(
            count.load(Ordering::SeqCst),
            1,
            "5 triggers for the same project should coalesce into 1 call"
        );
    }

    #[tokio::test]
    async fn test_debounce_mixed_unique_and_duplicate() {
        let count = Arc::new(AtomicU32::new(0));
        let engine: Arc<dyn AnalyticsEngine> =
            Arc::new(CountingAnalyticsEngine::new(count.clone()));

        let debouncer = AnalyticsDebouncer::new(engine, 100);

        let pid_a = Uuid::new_v4();
        let pid_b = Uuid::new_v4();

        // A, B, A, B, A → should dedupe to {A, B}
        debouncer.trigger(pid_a);
        debouncer.trigger(pid_b);
        debouncer.trigger(pid_a);
        debouncer.trigger(pid_b);
        debouncer.trigger(pid_a);

        tokio::time::sleep(Duration::from_millis(300)).await;

        assert_eq!(
            count.load(Ordering::SeqCst),
            2,
            "Mixed A/B triggers should produce exactly 2 analytics calls"
        );
    }
}
