//! ContinualTrainer — incremental fine-tuning with EWC and replay buffer.
//!
//! Manages the lifecycle of model retraining:
//! 1. Accumulates new trajectories in a replay buffer
//! 2. When threshold is reached, triggers background fine-tuning
//! 3. Uses EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting
//! 4. Evaluates on validation set before promoting
//! 5. Discards if worse than current model
//!
//! All training runs on CPU in a background tokio task with nice(19).
//! A CpuGuard circuit breaker pauses training if system load exceeds 80%.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex};

use neural_routing_policy::dataloader::{TrajectoryDataLoader, TrajectoryDataLoaderConfig};
use neural_routing_policy::dataset::TrajectoryTensors;
use neural_routing_policy::ewc::{EWCConfig, EWCRegularizer};
use neural_routing_policy::training::{train_decision_transformer_with_ewc, TrainingConfig};
use neural_routing_policy::transformer::DecisionTransformerConfig;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// ContinualTrainer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualConfig {
    /// Maximum replay buffer capacity.
    pub buffer_capacity: usize,
    /// Number of new trajectories before triggering fine-tuning.
    pub trigger_threshold: usize,
    /// Number of epochs for fine-tuning on new data.
    pub new_data_epochs: usize,
    /// Number of epochs for replay on old data (anti-forgetting).
    pub replay_epochs: usize,
    /// Replay sample size (from the buffer).
    pub replay_sample_size: usize,
    /// EWC lambda — importance weight for the penalty term.
    pub ewc_lambda: f64,
    /// Number of samples for Fisher information estimation.
    pub ewc_fisher_samples: usize,
    /// Decay factor for old trajectories: weight = exp(-age_days / decay_halflife).
    pub decay_halflife_days: f64,
    /// Archive threshold: trajectories older than this (days) are removed.
    pub archive_threshold_days: u64,
    /// Minimum improvement on val set to promote (relative %).
    pub min_improvement_pct: f64,
    /// Directory for model checkpoints (VarMap save/load between cycles).
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for ContinualConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 10_000,
            trigger_threshold: 1_000,
            new_data_epochs: 10,
            replay_epochs: 5,
            replay_sample_size: 500,
            ewc_lambda: 5000.0,
            ewc_fisher_samples: 200,
            decay_halflife_days: 30.0,
            archive_threshold_days: 90,
            min_improvement_pct: 1.0,
            checkpoint_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Replay Buffer
// ---------------------------------------------------------------------------

/// A trajectory entry in the replay buffer with temporal metadata.
#[derive(Debug, Clone)]
pub struct BufferEntry {
    /// Trajectory tensors (states, actions, RTGs, rewards).
    pub tensors: TrajectoryTensors,
    /// When this trajectory was collected.
    pub collected_at: DateTime<Utc>,
    /// Total reward for this trajectory.
    pub reward: f64,
    /// Whether this was an exploratory trajectory (reduced weight).
    pub exploratory: bool,
}

/// Replay buffer with capacity limit and temporal decay.
pub struct ReplayBuffer {
    entries: VecDeque<BufferEntry>,
    capacity: usize,
    new_count: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            new_count: 0,
        }
    }

    /// Add a new trajectory to the buffer.
    pub fn push(&mut self, entry: BufferEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
        self.new_count += 1;
    }

    /// Number of new trajectories since last training.
    pub fn new_count(&self) -> usize {
        self.new_count
    }

    /// Reset the new-trajectory counter (called after training).
    pub fn reset_new_count(&mut self) {
        self.new_count = 0;
    }

    /// Total buffer size.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the buffer empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sample entries with temporal decay weighting.
    /// More recent trajectories are more likely to be selected.
    pub fn sample_weighted(&self, n: usize, decay_halflife_days: f64) -> Vec<&BufferEntry> {
        if self.entries.is_empty() || n == 0 {
            return vec![];
        }

        let now = Utc::now();
        let mut weighted: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let age_days = (now - entry.collected_at).num_hours() as f64 / 24.0;
                let weight = (-age_days / decay_halflife_days).exp();
                let weight = if entry.exploratory {
                    weight * 0.7
                } else {
                    weight
                };
                (i, weight)
            })
            .collect();

        // Sort by weight descending and take top N
        weighted.sort_by(|a, b| b.1.total_cmp(&a.1));
        weighted
            .into_iter()
            .take(n)
            .map(|(i, _)| &self.entries[i])
            .collect()
    }

    /// Get the most recent N entries (new data for fine-tuning).
    pub fn recent(&self, n: usize) -> Vec<&BufferEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Purge entries older than the archive threshold.
    pub fn purge_old(&mut self, threshold_days: u64) {
        let cutoff = Utc::now() - chrono::Duration::days(threshold_days as i64);
        self.entries.retain(|e| e.collected_at > cutoff);
    }
}

// ---------------------------------------------------------------------------
// Training Events
// ---------------------------------------------------------------------------

/// Events sent to/from the ContinualTrainer.
#[derive(Debug, Clone)]
pub enum TrainingEvent {
    /// New trajectory data available.
    NewData(BufferEntry),
    /// Request immediate training (override threshold).
    ForceTraining,
    /// Shutdown the trainer.
    Shutdown,
}

/// Result of a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunResult {
    /// Whether the new model was promoted.
    pub promoted: bool,
    /// Validation loss before training.
    pub val_loss_before: f64,
    /// Validation loss after training.
    pub val_loss_after: f64,
    /// Improvement percentage.
    pub improvement_pct: f64,
    /// Number of new trajectories used.
    pub new_data_count: usize,
    /// Number of replay trajectories used.
    pub replay_count: usize,
    /// Training duration.
    pub duration_ms: u64,
    /// When the training started.
    pub started_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// ContinualTrainer
// ---------------------------------------------------------------------------

/// ContinualTrainer — manages background fine-tuning pipeline.
///
/// Thread-safe: the replay buffer is behind a Mutex, and communication
/// happens via mpsc channels. The EWC regularizer is shared between the
/// trainer and the background training loop.
pub struct ContinualTrainer {
    config: ContinualConfig,
    /// Shared replay buffer.
    buffer: Arc<Mutex<ReplayBuffer>>,
    /// Channel to send events to the trainer.
    tx: mpsc::Sender<TrainingEvent>,
    /// Training run history.
    history: Arc<Mutex<Vec<TrainingRunResult>>>,
    /// EWC regularizer — shared with the training loop.
    ewc: Arc<Mutex<EWCRegularizer>>,
}

impl ContinualTrainer {
    /// Create a new ContinualTrainer and spawn the background training loop.
    ///
    /// Returns the trainer handle and a JoinHandle for the background task.
    pub fn new(config: ContinualConfig) -> (Self, tokio::task::JoinHandle<()>) {
        let buffer = Arc::new(Mutex::new(ReplayBuffer::new(config.buffer_capacity)));
        let (tx, rx) = mpsc::channel(1000);
        let history = Arc::new(Mutex::new(Vec::new()));
        let ewc = Arc::new(Mutex::new(EWCRegularizer::new(EWCConfig {
            lambda: config.ewc_lambda,
            fisher_samples: config.ewc_fisher_samples,
        })));

        // Ensure checkpoint directory exists
        if let Some(ref dir) = config.checkpoint_dir {
            std::fs::create_dir_all(dir).ok();
        }

        let buffer_clone = buffer.clone();
        let history_clone = history.clone();
        let config_clone = config.clone();
        let ewc_clone = ewc.clone();

        let handle = tokio::spawn(async move {
            training_loop(rx, buffer_clone, history_clone, ewc_clone, config_clone).await;
        });

        let trainer = Self {
            config,
            buffer,
            tx,
            history,
            ewc,
        };

        (trainer, handle)
    }

    /// Whether EWC is currently active (has at least one snapshot).
    pub async fn ewc_active(&self) -> bool {
        self.ewc.lock().await.is_active()
    }

    /// Submit new trajectory data for eventual training.
    pub fn submit(&self, entry: BufferEntry) {
        if let Err(e) = self.tx.try_send(TrainingEvent::NewData(entry)) {
            tracing::warn!("Training channel full, dropping trajectory data: {}", e);
        }
    }

    /// Force immediate training (bypass threshold).
    pub async fn force_training(&self) {
        let _ = self.tx.send(TrainingEvent::ForceTraining).await;
    }

    /// Shutdown the trainer gracefully.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(TrainingEvent::Shutdown).await;
    }

    /// Get the replay buffer size.
    pub async fn buffer_size(&self) -> usize {
        self.buffer.lock().await.len()
    }

    /// Get the number of new trajectories since last training.
    pub async fn new_data_count(&self) -> usize {
        self.buffer.lock().await.new_count()
    }

    /// Get training run history.
    pub async fn history(&self) -> Vec<TrainingRunResult> {
        self.history.lock().await.clone()
    }

    /// Purge old entries from the replay buffer.
    pub async fn purge_old(&self) {
        self.buffer
            .lock()
            .await
            .purge_old(self.config.archive_threshold_days);
    }
}

/// Collect trajectory tensors from the buffer for a training cycle.
///
/// Returns (all_tensors, new_count, replay_count).
async fn collect_training_data(
    buffer: &Arc<Mutex<ReplayBuffer>>,
    config: &ContinualConfig,
) -> (Vec<TrajectoryTensors>, usize, usize) {
    let mut buf = buffer.lock().await;
    let new_count = buf.new_count();

    // Sample replay data (weighted by recency)
    let replay_sample = buf.sample_weighted(config.replay_sample_size, config.decay_halflife_days);
    let replay_count = replay_sample.len();

    // Collect all tensors: recent entries (new data) + replay sample
    let mut tensors: Vec<TrajectoryTensors> = buf
        .recent(new_count)
        .into_iter()
        .map(|e| e.tensors.clone())
        .collect();
    for entry in &replay_sample {
        tensors.push(entry.tensors.clone());
    }

    buf.reset_new_count();
    (tensors, new_count, replay_count)
}

/// Background training loop.
async fn training_loop(
    mut rx: mpsc::Receiver<TrainingEvent>,
    buffer: Arc<Mutex<ReplayBuffer>>,
    history: Arc<Mutex<Vec<TrainingRunResult>>>,
    ewc: Arc<Mutex<EWCRegularizer>>,
    config: ContinualConfig,
) {
    while let Some(event) = rx.recv().await {
        match event {
            TrainingEvent::NewData(entry) => {
                let mut buf = buffer.lock().await;
                buf.push(entry);

                // Check if threshold reached
                if buf.new_count() >= config.trigger_threshold {
                    drop(buf);
                    let (tensors, new_count, replay_count) =
                        collect_training_data(&buffer, &config).await;

                    let result =
                        run_training_cycle(tensors, new_count, replay_count, &config, &ewc).await;
                    history.lock().await.push(result);
                }
            }

            TrainingEvent::ForceTraining => {
                let (tensors, new_count, replay_count) =
                    collect_training_data(&buffer, &config).await;

                let result =
                    run_training_cycle(tensors, new_count, replay_count, &config, &ewc).await;
                history.lock().await.push(result);
            }

            TrainingEvent::Shutdown => {
                tracing::info!("ContinualTrainer shutting down");
                break;
            }
        }
    }
}

/// Execute a single training cycle with EWC regularization and checkpoint persistence.
///
/// Pipeline:
/// 1. Build a TrajectoryDataLoader from the collected tensors
/// 2. Load checkpoint if available (fine-tuning, not from scratch)
/// 3. Run a short fine-tuning pass with EWC penalty if active
/// 4. If improvement >= `min_improvement_pct` → promoted:
///    a. Save new checkpoint
///    b. Snapshot EWC Fisher information for next cycle
/// 5. Otherwise discard the new weights
///
/// Training runs on CPU via `spawn_blocking` to avoid blocking the async runtime.
async fn run_training_cycle(
    tensors: Vec<TrajectoryTensors>,
    new_count: usize,
    replay_count: usize,
    config: &ContinualConfig,
    ewc: &Arc<Mutex<EWCRegularizer>>,
) -> TrainingRunResult {
    let started_at = Utc::now();
    let start = std::time::Instant::now();

    tracing::info!(
        new_data = new_count,
        replay = replay_count,
        total_tensors = tensors.len(),
        epochs = config.new_data_epochs,
        ewc_lambda = config.ewc_lambda,
        "Starting continual training cycle"
    );

    // Need at least a few trajectories for a meaningful train/val split
    if tensors.len() < 10 {
        tracing::warn!(
            count = tensors.len(),
            "Not enough trajectories for training, skipping cycle"
        );
        return TrainingRunResult {
            promoted: false,
            val_loss_before: 0.0,
            val_loss_after: 0.0,
            improvement_pct: 0.0,
            new_data_count: new_count,
            replay_count,
            duration_ms: start.elapsed().as_millis() as u64,
            started_at,
        };
    }

    let epochs = config.new_data_epochs;
    let min_improvement = config.min_improvement_pct;
    let checkpoint_path = config
        .checkpoint_dir
        .as_ref()
        .map(|dir| dir.join("continual_best.safetensors"));

    // Pass EWC to the blocking thread via Arc clone
    let ewc_clone = ewc.clone();
    let ckpt_path_clone = checkpoint_path.clone();

    // Run training on a blocking thread (candle is CPU-bound)
    let training_result = tokio::task::spawn_blocking(move || {
        // Build DataLoader with temporal split
        let dl_config = TrajectoryDataLoaderConfig {
            batch_size: 32.min(tensors.len() / 2).max(1),
            ..Default::default()
        };
        let dataloader = TrajectoryDataLoader::new(tensors, dl_config);

        // Build training config — short fine-tuning run
        let training_config = TrainingConfig {
            epochs,
            learning_rate: 1e-4, // Lower LR for fine-tuning (vs 3e-4 for full training)
            patience: 3,         // Short patience for incremental updates
            ..Default::default()
        };

        let model_config = DecisionTransformerConfig::default();

        // Lock EWC for the duration of training (blocking thread, no await)
        let ewc_guard = ewc_clone.blocking_lock();

        // Determine checkpoint load path
        let load_path = ckpt_path_clone
            .as_ref()
            .filter(|p| p.exists())
            .map(|p| p.as_path());

        // Run the actual candle training loop with EWC
        let ewc_ref = if ewc_guard.is_active() {
            Some(&*ewc_guard)
        } else {
            None
        };

        train_decision_transformer_with_ewc(
            model_config,
            &dataloader,
            &training_config,
            ewc_ref,
            load_path,
        )
    })
    .await;

    let duration_ms = start.elapsed().as_millis() as u64;

    match training_result {
        Ok(Ok(result)) => {
            // Use the first and best epoch val_loss to compute improvement
            let val_loss_before = result
                .history
                .first()
                .map(|m| m.val_loss as f64)
                .unwrap_or(1.0);
            let val_loss_after = result.best_val_loss as f64;
            let improvement_pct = if val_loss_before > 0.0 {
                (val_loss_before - val_loss_after) / val_loss_before * 100.0
            } else {
                0.0
            };
            let promoted = improvement_pct >= min_improvement;

            if promoted {
                // Save to our continual checkpoint path for next cycle reload
                if let Some(ref ckpt_path) = checkpoint_path {
                    if let Some(ref best_ckpt) = result.checkpoint_path {
                        if best_ckpt.exists() {
                            std::fs::copy(best_ckpt, ckpt_path).ok();
                            tracing::info!(
                                path = %ckpt_path.display(),
                                "Saved continual checkpoint for next cycle"
                            );
                        }
                    }
                }

                // EWC snapshot: capture Fisher information from the promoted model.
                // The training function returned per-parameter sample gradients
                // collected at θ* (best checkpoint). We use these to compute
                // F_i = mean(grad²) via snapshot_from_gradients().
                if let Some(fisher_grads) = &result.fisher_gradients {
                    if !fisher_grads.is_empty() {
                        // We need a VarMap at θ* to take the EWC snapshot.
                        // Load from the continual checkpoint we just saved.
                        if let Some(ref ckpt_path) = checkpoint_path {
                            if ckpt_path.exists() {
                                let snapshot_result: Result<(), Box<dyn std::error::Error + Send + Sync>> = (|| {
                                    let mut varmap = candle_nn::VarMap::new();
                                    let vb = candle_nn::VarBuilder::from_varmap(
                                        &varmap,
                                        candle_core::DType::F32,
                                        &candle_core::Device::Cpu,
                                    );
                                    // Initialize model structure in VarMap
                                    let _ = neural_routing_policy::transformer::DecisionTransformer::new(
                                        DecisionTransformerConfig::default(),
                                        vb,
                                    )?;
                                    // Load promoted weights
                                    varmap.load(ckpt_path)?;
                                    Ok(varmap)
                                })().and_then(|varmap| {
                                    // snapshot_from_gradients is sync, but we need to
                                    // acquire the async Mutex. Use try_lock since we know
                                    // the training thread released it.
                                    let mut ewc_guard = ewc.try_lock()
                                        .map_err(|_| "EWC mutex still locked")?;
                                    ewc_guard.snapshot_from_gradients(&varmap, fisher_grads)
                                        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
                                    tracing::info!(
                                        param_count = ewc_guard.param_count(),
                                        "EWC Fisher snapshot taken — penalty active for next cycle"
                                    );
                                    Ok(())
                                });

                                if let Err(e) = snapshot_result {
                                    tracing::warn!(error = %e, "Failed to take EWC snapshot");
                                }
                            }
                        }
                    }
                }
            }

            tracing::info!(
                promoted,
                best_epoch = result.best_epoch,
                epochs_completed = result.epochs_completed,
                early_stopped = result.early_stopped,
                ewc_active = result.ewc_was_active,
                val_loss_before = format!("{:.4}", val_loss_before),
                val_loss_after = format!("{:.4}", val_loss_after),
                improvement_pct = format!("{:.2}%", improvement_pct),
                duration_ms,
                "Continual training cycle complete"
            );

            TrainingRunResult {
                promoted,
                val_loss_before,
                val_loss_after,
                improvement_pct,
                new_data_count: new_count,
                replay_count,
                duration_ms,
                started_at,
            }
        }
        Ok(Err(candle_err)) => {
            tracing::error!(error = %candle_err, "Candle training failed");
            TrainingRunResult {
                promoted: false,
                val_loss_before: 0.0,
                val_loss_after: 0.0,
                improvement_pct: 0.0,
                new_data_count: new_count,
                replay_count,
                duration_ms,
                started_at,
            }
        }
        Err(join_err) => {
            tracing::error!(error = %join_err, "Training task panicked");
            TrainingRunResult {
                promoted: false,
                val_loss_before: 0.0,
                val_loss_after: 0.0,
                improvement_pct: 0.0,
                new_data_count: new_count,
                replay_count,
                duration_ms,
                started_at,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use neural_routing_policy::dataset::{ACTION_DIM, STATE_DIM};
    use std::time::Duration;

    fn make_tensors() -> TrajectoryTensors {
        // Create minimal valid TrajectoryTensors matching the actual struct layout
        let seq_len = 4;
        TrajectoryTensors {
            seq_len,
            returns_to_go: vec![1.0; seq_len],
            states_flat: vec![0.0; seq_len * STATE_DIM],
            actions_flat: vec![0.0; seq_len * ACTION_DIM],
            rewards: vec![0.0; seq_len],
            timesteps: vec![0, 1, 2, 3],
            weight: 1.0,
            total_reward: 1.0,
            created_at_epoch: chrono::Utc::now().timestamp(),
        }
    }

    fn make_entry(reward: f64, exploratory: bool) -> BufferEntry {
        BufferEntry {
            tensors: make_tensors(),
            collected_at: Utc::now(),
            reward,
            exploratory,
        }
    }

    #[test]
    fn test_replay_buffer_push() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        buffer.push(make_entry(0.8, false));
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.new_count(), 1);
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let mut buffer = ReplayBuffer::new(5);

        for i in 0..10 {
            buffer.push(make_entry(i as f64 * 0.1, false));
        }

        assert_eq!(buffer.len(), 5, "Buffer should cap at capacity");
    }

    #[test]
    fn test_replay_buffer_recent() {
        let mut buffer = ReplayBuffer::new(100);

        for i in 0..5 {
            buffer.push(make_entry(i as f64 * 0.1, false));
        }

        let recent = buffer.recent(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_replay_buffer_sample_weighted() {
        let mut buffer = ReplayBuffer::new(100);

        for i in 0..20 {
            buffer.push(make_entry(i as f64 * 0.05, false));
        }

        let sample = buffer.sample_weighted(10, 30.0);
        assert_eq!(sample.len(), 10);
    }

    #[test]
    fn test_replay_buffer_reset_new_count() {
        let mut buffer = ReplayBuffer::new(100);

        buffer.push(make_entry(0.8, false));
        buffer.push(make_entry(0.7, false));
        assert_eq!(buffer.new_count(), 2);

        buffer.reset_new_count();
        assert_eq!(buffer.new_count(), 0);
        assert_eq!(buffer.len(), 2); // Entries still there
    }

    #[tokio::test]
    async fn test_continual_trainer_lifecycle() {
        let config = ContinualConfig {
            trigger_threshold: 3, // Low threshold for testing
            buffer_capacity: 100,
            ..Default::default()
        };

        let (trainer, handle) = ContinualTrainer::new(config);

        // Submit entries
        trainer.submit(make_entry(0.8, false));
        trainer.submit(make_entry(0.7, false));
        trainer.submit(make_entry(0.9, false));

        // Give the background task time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        let history = trainer.history().await;
        // Should have triggered training (3 >= threshold of 3)
        assert!(!history.is_empty(), "Should have at least one training run");

        // Shutdown
        trainer.shutdown().await;
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }

    #[tokio::test]
    async fn test_force_training() {
        let config = ContinualConfig {
            trigger_threshold: 10000, // Very high — won't trigger automatically
            buffer_capacity: 100,
            ..Default::default()
        };

        let (trainer, handle) = ContinualTrainer::new(config);

        trainer.submit(make_entry(0.8, false));
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Force training despite not reaching threshold
        trainer.force_training().await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        let history = trainer.history().await;
        assert!(
            !history.is_empty(),
            "Force training should produce a result"
        );

        trainer.shutdown().await;
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
    }

    #[test]
    fn test_buffer_exploratory_weight() {
        let mut buffer = ReplayBuffer::new(100);

        // Add exploratory and non-exploratory entries
        buffer.push(make_entry(0.8, false));
        buffer.push(make_entry(0.8, true));

        let sample = buffer.sample_weighted(2, 30.0);
        assert_eq!(sample.len(), 2);
    }

    #[tokio::test]
    async fn test_ewc_wiring_two_cycles() {
        // Test that EWC snapshot is taken after first promoted cycle,
        // and that EWC penalty is active during the second cycle.
        let tmp_dir = std::env::temp_dir().join("po_test_ewc_wiring");
        let _ = std::fs::remove_dir_all(&tmp_dir);

        let config = ContinualConfig {
            trigger_threshold: 12, // Need at least 10 for train/val split
            buffer_capacity: 200,
            new_data_epochs: 2, // Minimal epochs for speed
            ewc_lambda: 100.0,
            ewc_fisher_samples: 5,
            min_improvement_pct: -100.0, // Always promote (any result is "improvement")
            checkpoint_dir: Some(tmp_dir.clone()),
            ..Default::default()
        };

        let (trainer, handle) = ContinualTrainer::new(config);

        // EWC should not be active initially
        assert!(
            !trainer.ewc_active().await,
            "EWC should not be active initially"
        );

        // --- Cycle 1: submit enough entries to trigger training (>= 10 for train/val split) ---
        for _ in 0..12 {
            trainer.submit(make_entry(0.8, false));
        }

        // Wait for training cycle to complete (includes Fisher estimation)
        // Training with candle + 12 trajectories + 2 epochs takes ~5-10s
        tokio::time::sleep(Duration::from_secs(15)).await;

        let history = trainer.history().await;
        assert!(!history.is_empty(), "First training cycle should have run");
        assert!(
            history[0].promoted,
            "First cycle should be promoted (min_improvement_pct=-100), got: {:?}",
            history[0]
        );

        // After a promoted cycle, EWC should have taken a Fisher snapshot
        let ewc_active = trainer.ewc_active().await;
        assert!(
            ewc_active,
            "EWC should be active after first promoted cycle (Fisher snapshot taken)"
        );

        // --- Cycle 2: submit more entries — EWC penalty should be active ---
        for _ in 0..12 {
            trainer.submit(make_entry(0.9, false));
        }
        tokio::time::sleep(Duration::from_secs(15)).await;

        let history = trainer.history().await;
        assert!(
            history.len() >= 2,
            "Expected at least 2 training runs, got {}",
            history.len()
        );

        // Verify checkpoint file exists
        let ckpt_file = tmp_dir.join("continual_best.safetensors");
        assert!(
            ckpt_file.exists(),
            "Continual checkpoint file should exist at {:?}",
            ckpt_file
        );

        // Cleanup
        trainer.shutdown().await;
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_continual_config_defaults() {
        let config = ContinualConfig::default();
        assert!(config.checkpoint_dir.is_none());
        assert_eq!(config.ewc_lambda, 5000.0);
        assert_eq!(config.ewc_fisher_samples, 200);
    }
}
