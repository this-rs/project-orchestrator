//! TrajectoryDataLoader — weighted sampling, temporal splits, and batch iteration.
//!
//! Provides a DataLoader that:
//! - Splits trajectories temporally (70/15/15 by creation date)
//! - Applies weighted sampling (real:1.0 / migrated:0.5 / simulated:0.3)
//! - Yields padded batches as candle tensors
//! - Supports Gaussian jitter for data augmentation

use candle_core::Result as CandleResult;
use serde::{Deserialize, Serialize};

use crate::dataset::{
    pad_and_batch, PolicyNormStats, TrajectoryBatch, TrajectoryTensors, ACTION_DIM, STATE_DIM,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// DataLoader configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryDataLoaderConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Maximum sequence length (trajectories longer than this are truncated).
    pub max_length: usize,
    /// Train fraction (default 0.70).
    pub train_fraction: f64,
    /// Validation fraction (default 0.15).
    pub val_fraction: f64,
    /// Gaussian jitter std for data augmentation (0.0 = disabled).
    pub jitter_std: f32,
    /// Random seed for shuffling.
    pub seed: u64,
}

impl Default for TrajectoryDataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_length: 32,
            train_fraction: 0.70,
            val_fraction: 0.15,
            jitter_std: 0.01,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Split type
// ---------------------------------------------------------------------------

/// Dataset split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Validation,
    Test,
}

// ---------------------------------------------------------------------------
// TrajectoryDataLoader
// ---------------------------------------------------------------------------

/// DataLoader for trajectory data.
///
/// Holds all trajectories in memory (pre-tensorized), provides temporal splits
/// and weighted batch iteration.
pub struct TrajectoryDataLoader {
    /// All trajectory tensors, sorted by creation time.
    trajectories: Vec<TrajectoryTensors>,
    /// Train/val/test split boundaries (indices into self.trajectories).
    train_end: usize,
    val_end: usize,
    /// Normalization stats computed on the training set.
    pub norm_stats: PolicyNormStats,
    /// Configuration.
    config: TrajectoryDataLoaderConfig,
}

impl TrajectoryDataLoader {
    /// Create a new DataLoader from pre-computed TrajectoryTensors.
    ///
    /// Sorts by creation time, splits temporally, and computes normalization stats.
    pub fn new(
        mut trajectories: Vec<TrajectoryTensors>,
        config: TrajectoryDataLoaderConfig,
    ) -> Self {
        // Sort by creation timestamp (oldest first → newest last)
        trajectories.sort_by_key(|t| t.created_at_epoch);

        let n = trajectories.len();
        let train_end = ((n as f64) * config.train_fraction).round() as usize;
        let val_end = train_end + ((n as f64) * config.val_fraction).round() as usize;
        let val_end = val_end.min(n);

        // Compute normalization stats on training set only
        let train_slice = &trajectories[..train_end];
        let norm_stats = PolicyNormStats::compute(train_slice);

        Self {
            trajectories,
            train_end,
            val_end,
            norm_stats,
            config,
        }
    }

    /// Get the number of trajectories in each split.
    pub fn split_sizes(&self) -> (usize, usize, usize) {
        let train = self.train_end;
        let val = self.val_end - self.train_end;
        let test = self.trajectories.len() - self.val_end;
        (train, val, test)
    }

    /// Get total number of trajectories.
    pub fn total_count(&self) -> usize {
        self.trajectories.len()
    }

    /// Get a reference to the normalization stats.
    pub fn norm_stats(&self) -> &PolicyNormStats {
        &self.norm_stats
    }

    /// Get the state dimension.
    pub fn state_dim(&self) -> usize {
        STATE_DIM
    }

    /// Get the action dimension.
    pub fn action_dim(&self) -> usize {
        ACTION_DIM
    }

    /// Get the slice of trajectories for a given split.
    fn split_slice(&self, split: Split) -> &[TrajectoryTensors] {
        match split {
            Split::Train => &self.trajectories[..self.train_end],
            Split::Validation => &self.trajectories[self.train_end..self.val_end],
            Split::Test => &self.trajectories[self.val_end..],
        }
    }

    /// Create a batch iterator for a given split.
    ///
    /// For training: uses weighted shuffling.
    /// For validation/test: sequential iteration (no shuffle).
    pub fn batches(&self, split: Split, epoch: usize) -> BatchIterator<'_> {
        let data = self.split_slice(split);
        let is_train = split == Split::Train;

        // Build index array with weighted shuffling for train
        let indices: Vec<usize> = if is_train {
            weighted_shuffle(data, self.config.seed.wrapping_add(epoch as u64))
        } else {
            (0..data.len()).collect()
        };

        BatchIterator {
            data,
            indices,
            pos: 0,
            batch_size: self.config.batch_size,
            max_length: self.config.max_length,
            norm_stats: &self.norm_stats,
            jitter_std: if is_train {
                self.config.jitter_std
            } else {
                0.0
            },
            seed: self.config.seed.wrapping_add(epoch as u64 * 1000),
        }
    }

    /// Get summary statistics for each split.
    pub fn summary(&self) -> DataLoaderSummary {
        let (train_n, val_n, test_n) = self.split_sizes();
        let train_slice = self.split_slice(Split::Train);
        let val_slice = self.split_slice(Split::Validation);
        let test_slice = self.split_slice(Split::Test);

        DataLoaderSummary {
            total: self.trajectories.len(),
            train_count: train_n,
            val_count: val_n,
            test_count: test_n,
            train_avg_len: avg_seq_len(train_slice),
            val_avg_len: avg_seq_len(val_slice),
            test_avg_len: avg_seq_len(test_slice),
            train_avg_reward: avg_reward(train_slice),
            val_avg_reward: avg_reward(val_slice),
            test_avg_reward: avg_reward(test_slice),
        }
    }
}

/// Summary statistics for the DataLoader.
#[derive(Debug, Clone, Serialize)]
pub struct DataLoaderSummary {
    pub total: usize,
    pub train_count: usize,
    pub val_count: usize,
    pub test_count: usize,
    pub train_avg_len: f32,
    pub val_avg_len: f32,
    pub test_avg_len: f32,
    pub train_avg_reward: f32,
    pub val_avg_reward: f32,
    pub test_avg_reward: f32,
}

// ---------------------------------------------------------------------------
// BatchIterator
// ---------------------------------------------------------------------------

/// Iterator over padded batches.
pub struct BatchIterator<'a> {
    data: &'a [TrajectoryTensors],
    indices: Vec<usize>,
    pos: usize,
    batch_size: usize,
    max_length: usize,
    norm_stats: &'a PolicyNormStats,
    jitter_std: f32,
    seed: u64,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = CandleResult<TrajectoryBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;

        // Gather references
        let batch_refs: Vec<&TrajectoryTensors> =
            batch_indices.iter().map(|&i| &self.data[i]).collect();

        let batch_seed = self.seed.wrapping_add(self.pos as u64);
        Some(pad_and_batch(
            &batch_refs,
            self.max_length,
            Some(self.norm_stats),
            self.jitter_std,
            batch_seed,
        ))
    }
}

impl<'a> BatchIterator<'a> {
    /// Get the number of batches remaining.
    pub fn remaining_batches(&self) -> usize {
        let remaining = self.indices.len().saturating_sub(self.pos);
        remaining.div_ceil(self.batch_size)
    }

    /// Get the total number of samples.
    pub fn total_samples(&self) -> usize {
        self.indices.len()
    }
}

// ---------------------------------------------------------------------------
// Weighted shuffle
// ---------------------------------------------------------------------------

/// Shuffle indices with weights — higher weight = more likely to appear earlier.
///
/// Uses reservoir-based weighted shuffle: assign score = u^(1/w) where u~Uniform(0,1),
/// then sort descending by score. This gives exact weighted sampling without replacement.
fn weighted_shuffle(data: &[TrajectoryTensors], seed: u64) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, t)| {
            // Deterministic pseudo-random uniform in (0, 1)
            let h = splitmix64(seed, i as u64);
            let u = (h as f64 + 1.0) / (u64::MAX as f64 + 2.0); // avoid 0 and 1
            let w = t.weight.max(0.01) as f64;
            // Efraimidis-Spirakis: key = u^(1/w)
            let key = u.powf(1.0 / w);
            (i, key)
        })
        .collect();

    // Sort descending by key
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored.into_iter().map(|(i, _)| i).collect()
}

/// Simple deterministic hash (splitmix64).
fn splitmix64(seed: u64, index: u64) -> u64 {
    let mut h = seed.wrapping_mul(6364136223846793005).wrapping_add(index);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn avg_seq_len(data: &[TrajectoryTensors]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().map(|t| t.seq_len as f32).sum::<f32>() / data.len() as f32
}

fn avg_reward(data: &[TrajectoryTensors]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().map(|t| t.total_reward).sum::<f32>() / data.len() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{trajectory_to_tensors, CONTEXT_DIM, QUERY_DIM};
    use chrono::{Duration, Utc};
    use neural_routing_core::models::{Trajectory, TrajectoryNode};
    use uuid::Uuid;

    fn make_trajectory_at(
        num_nodes: usize,
        reward: f64,
        prefix: &str,
        days_ago: i64,
    ) -> Trajectory {
        let mut nodes = Vec::with_capacity(num_nodes);
        let per_step = reward / num_nodes as f64;
        for i in 0..num_nodes {
            nodes.push(TrajectoryNode {
                id: Uuid::new_v4(),
                context_embedding: vec![0.1 * (i as f32 + 1.0); CONTEXT_DIM],
                action_type: format!("action_{}", i % 5),
                action_params: serde_json::json!({}),
                alternatives_count: 3,
                chosen_index: 0,
                confidence: 0.8,
                local_reward: per_step,
                cumulative_reward: per_step * (i + 1) as f64,
                delta_ms: 100,
                order: i,
            });
        }
        Trajectory {
            id: Uuid::new_v4(),
            session_id: format!("{}{}", prefix, Uuid::new_v4()),
            query_embedding: vec![0.5f32; QUERY_DIM],
            total_reward: reward,
            step_count: num_nodes,
            duration_ms: 1000,
            nodes,
            created_at: Utc::now() - Duration::days(days_ago),
        }
    }

    fn make_test_dataloader(n: usize) -> TrajectoryDataLoader {
        let trajs: Vec<Trajectory> = (0..n)
            .map(|i| make_trajectory_at(4, 8.0 + i as f64, "session-", (n - i) as i64))
            .collect();
        let tensors: Vec<TrajectoryTensors> =
            trajs.iter().filter_map(trajectory_to_tensors).collect();
        TrajectoryDataLoader::new(tensors, TrajectoryDataLoaderConfig::default())
    }

    #[test]
    fn test_temporal_split() {
        let dl = make_test_dataloader(100);
        let (train, val, test) = dl.split_sizes();

        assert_eq!(train, 70);
        assert_eq!(val, 15);
        assert_eq!(test, 15);
        assert_eq!(train + val + test, 100);
    }

    #[test]
    fn test_temporal_ordering() {
        // Verify that test set contains the most recent trajectories
        let dl = make_test_dataloader(100);

        let train_max_ts = dl
            .split_slice(Split::Train)
            .last()
            .unwrap()
            .created_at_epoch;
        let val_min_ts = dl
            .split_slice(Split::Validation)
            .first()
            .unwrap()
            .created_at_epoch;
        let val_max_ts = dl
            .split_slice(Split::Validation)
            .last()
            .unwrap()
            .created_at_epoch;
        let test_min_ts = dl
            .split_slice(Split::Test)
            .first()
            .unwrap()
            .created_at_epoch;

        assert!(
            train_max_ts <= val_min_ts,
            "Train max ({}) should be <= val min ({})",
            train_max_ts,
            val_min_ts
        );
        assert!(
            val_max_ts <= test_min_ts,
            "Val max ({}) should be <= test min ({})",
            val_max_ts,
            test_min_ts
        );
    }

    #[test]
    fn test_batch_iteration() -> CandleResult<()> {
        let dl = make_test_dataloader(100);
        let mut total_samples = 0;

        for batch_result in dl.batches(Split::Train, 0) {
            let batch = batch_result?;
            total_samples += batch.batch_size;
            assert!(batch.batch_size <= 32);
        }

        assert_eq!(total_samples, 70); // all train samples iterated
        Ok(())
    }

    #[test]
    fn test_val_no_jitter() -> CandleResult<()> {
        let dl = make_test_dataloader(50);

        // Two passes over val should produce identical results (no jitter, no shuffle)
        let batches_1: Vec<TrajectoryBatch> = dl
            .batches(Split::Validation, 0)
            .collect::<Result<Vec<_>, _>>()?;
        let batches_2: Vec<TrajectoryBatch> = dl
            .batches(Split::Validation, 1)
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(batches_1.len(), batches_2.len());

        // Compare first batch states
        if !batches_1.is_empty() {
            let s1 = batches_1[0].states.flatten_all()?.to_vec1::<f32>()?;
            let s2 = batches_2[0].states.flatten_all()?.to_vec1::<f32>()?;
            let diff: f32 = s1.iter().zip(s2.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(
                diff < 1e-6,
                "Validation batches should be identical across epochs"
            );
        }

        Ok(())
    }

    #[test]
    fn test_weighted_shuffle_favors_higher_weight() {
        // Create mix of real (1.0) and simulated (0.3) trajectories
        let mut trajs = Vec::new();
        for i in 0..50 {
            let prefix = if i < 25 { "session-" } else { "mcts-sim-" };
            let t = make_trajectory_at(3, 5.0, prefix, i);
            if let Some(tensors) = trajectory_to_tensors(&t) {
                trajs.push(tensors);
            }
        }

        let indices = weighted_shuffle(&trajs, 42);

        // Count how many of the top-25 positions are real trajectories
        let top_25_real = indices[..25]
            .iter()
            .filter(|&&i| (trajs[i].weight - 1.0).abs() < 0.01)
            .count();

        // Real trajectories should dominate the top positions
        assert!(
            top_25_real > 15,
            "Expected >15 real in top 25, got {}",
            top_25_real
        );
    }

    #[test]
    fn test_summary() {
        let dl = make_test_dataloader(100);
        let summary = dl.summary();

        assert_eq!(summary.total, 100);
        assert_eq!(summary.train_count, 70);
        assert_eq!(summary.val_count, 15);
        assert_eq!(summary.test_count, 15);
        assert!(summary.train_avg_len > 0.0);
        assert!(summary.train_avg_reward > 0.0);
    }
}
