//! TrajectoryDataset — converts trajectories from Neo4j into Decision Transformer tensors.
//!
//! Converts `Trajectory` + `TrajectoryNode` from neural-routing-core into
//! candle tensors suitable for training the Decision Transformer or CQL:
//! - returns_to_go [T] — sum of future rewards from timestep t
//! - states [T, state_dim] — concatenation of [query_embedding(256d) || context_embedding(256d)] = 512d
//! - actions [T, action_dim] — context_embedding at each step (256d action representation)
//! - rewards [T] — per-step decomposed reward
//! - timesteps [T] — position indices 0..T-1
//! - attention_mask [T] — 1.0 for real steps, 0.0 for padding
//!
//! Also provides z-score normalization with stats computed on the training set.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use neural_routing_core::models::{Trajectory, TrajectoryNode};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Dimension of the query_embedding (from Trajectory).
pub const QUERY_DIM: usize = 256;
/// Dimension of the context_embedding (from TrajectoryNode).
pub const CONTEXT_DIM: usize = 256;
/// State = [query_embedding || context_embedding] = 512d.
pub const STATE_DIM: usize = QUERY_DIM + CONTEXT_DIM;
/// Action = context_embedding = 256d.
pub const ACTION_DIM: usize = CONTEXT_DIM;

// ---------------------------------------------------------------------------
// NormStats — z-score normalization statistics
// ---------------------------------------------------------------------------

/// Z-score normalization statistics per dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyNormStats {
    /// Per-dimension mean for states [STATE_DIM].
    pub state_mean: Vec<f32>,
    /// Per-dimension std for states [STATE_DIM].
    pub state_std: Vec<f32>,
    /// Per-dimension mean for actions [ACTION_DIM].
    pub action_mean: Vec<f32>,
    /// Per-dimension std for actions [ACTION_DIM].
    pub action_std: Vec<f32>,
    /// Mean reward (scalar).
    pub reward_mean: f32,
    /// Std reward (scalar).
    pub reward_std: f32,
}

impl PolicyNormStats {
    /// Compute normalization stats from a set of trajectory tensors.
    pub fn compute(trajectories: &[TrajectoryTensors]) -> Self {
        let mut state_sum = vec![0.0f64; STATE_DIM];
        let mut state_sq_sum = vec![0.0f64; STATE_DIM];
        let mut action_sum = vec![0.0f64; ACTION_DIM];
        let mut action_sq_sum = vec![0.0f64; ACTION_DIM];
        let mut reward_sum = 0.0f64;
        let mut reward_sq_sum = 0.0f64;
        let mut count = 0usize;

        for traj in trajectories {
            for t in 0..traj.seq_len {
                // States
                let state_offset = t * STATE_DIM;
                for d in 0..STATE_DIM {
                    let v = traj.states_flat[state_offset + d] as f64;
                    state_sum[d] += v;
                    state_sq_sum[d] += v * v;
                }
                // Actions
                let action_offset = t * ACTION_DIM;
                for d in 0..ACTION_DIM {
                    let v = traj.actions_flat[action_offset + d] as f64;
                    action_sum[d] += v;
                    action_sq_sum[d] += v * v;
                }
                // Rewards
                let r = traj.rewards[t] as f64;
                reward_sum += r;
                reward_sq_sum += r * r;

                count += 1;
            }
        }

        let n = count.max(1) as f64;
        let eps = 1e-8;

        let state_mean: Vec<f32> = state_sum.iter().map(|s| (s / n) as f32).collect();
        let state_std: Vec<f32> = state_sum
            .iter()
            .zip(state_sq_sum.iter())
            .map(|(s, sq)| {
                let mean = s / n;
                let var = (sq / n) - mean * mean;
                var.max(0.0).sqrt().max(eps) as f32
            })
            .collect();

        let action_mean: Vec<f32> = action_sum.iter().map(|s| (s / n) as f32).collect();
        let action_std: Vec<f32> = action_sum
            .iter()
            .zip(action_sq_sum.iter())
            .map(|(s, sq)| {
                let mean = s / n;
                let var = (sq / n) - mean * mean;
                var.max(0.0).sqrt().max(eps) as f32
            })
            .collect();

        let reward_mean = (reward_sum / n) as f32;
        let reward_std = {
            let mean = reward_sum / n;
            let var = (reward_sq_sum / n) - mean * mean;
            var.max(0.0).sqrt().max(eps) as f32
        };

        Self {
            state_mean,
            state_std,
            action_mean,
            action_std,
            reward_mean,
            reward_std,
        }
    }
}

// ---------------------------------------------------------------------------
// TrajectoryTensors — flat CPU representation before conversion to candle
// ---------------------------------------------------------------------------

/// Pre-tensor representation of a single trajectory.
/// Kept as flat Vec<f32> to avoid candle overhead during batch construction.
#[derive(Debug, Clone)]
pub struct TrajectoryTensors {
    /// Actual sequence length (before padding).
    pub seq_len: usize,
    /// Returns-to-go: cumulative future reward from timestep t. [seq_len].
    pub returns_to_go: Vec<f32>,
    /// Flattened states: [seq_len * STATE_DIM]. Row-major.
    pub states_flat: Vec<f32>,
    /// Flattened actions: [seq_len * ACTION_DIM]. Row-major.
    pub actions_flat: Vec<f32>,
    /// Per-step rewards. [seq_len].
    pub rewards: Vec<f32>,
    /// Timestep indices. [seq_len].
    pub timesteps: Vec<u32>,
    /// Source weight for weighted sampling (1.0=real, 0.5=migrated, 0.3=simulated).
    pub weight: f32,
    /// Original trajectory total reward (for RTG conditioning).
    pub total_reward: f32,
    /// Created timestamp (for temporal split).
    pub created_at_epoch: i64,
}

/// Source type of a trajectory (determines weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectorySource {
    Real,
    Migrated,
    Simulated,
}

impl TrajectorySource {
    /// Determine source from session_id prefix.
    pub fn from_session_id(session_id: &str) -> Self {
        if session_id.starts_with("mcts-sim-") {
            Self::Simulated
        } else if session_id.starts_with("migrated-") {
            Self::Migrated
        } else {
            Self::Real
        }
    }

    /// Get the sampling weight.
    pub fn weight(self) -> f32 {
        match self {
            Self::Real => 1.0,
            Self::Migrated => 0.5,
            Self::Simulated => 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion: Trajectory → TrajectoryTensors
// ---------------------------------------------------------------------------

/// Convert a single Trajectory (from core) into pre-tensor form.
///
/// The state at each timestep is the concatenation of [query_embedding || node.context_embedding].
/// The action is the node's context_embedding (represents the decision taken).
/// Returns-to-go is the sum of future rewards from each timestep.
pub fn trajectory_to_tensors(traj: &Trajectory) -> Option<TrajectoryTensors> {
    if traj.nodes.is_empty() {
        return None;
    }

    let seq_len = traj.nodes.len();
    let source = TrajectorySource::from_session_id(&traj.session_id);

    // Ensure query_embedding has correct dimension (256d), else zero-pad/truncate
    let query_emb = pad_or_truncate(&traj.query_embedding, QUERY_DIM);

    let mut states_flat = Vec::with_capacity(seq_len * STATE_DIM);
    let mut actions_flat = Vec::with_capacity(seq_len * ACTION_DIM);
    let mut rewards = Vec::with_capacity(seq_len);
    let mut timesteps = Vec::with_capacity(seq_len);

    // Sort nodes by order to ensure correct sequence
    let mut nodes: Vec<&TrajectoryNode> = traj.nodes.iter().collect();
    nodes.sort_by_key(|n| n.order);

    for (t, node) in nodes.iter().enumerate() {
        // State = [query_embedding(256d) || context_embedding(256d)]
        let context_emb = pad_or_truncate(&node.context_embedding, CONTEXT_DIM);
        states_flat.extend_from_slice(&query_emb);
        states_flat.extend_from_slice(&context_emb);

        // Action = context_embedding (represents the decision taken)
        actions_flat.extend_from_slice(&context_emb);

        // Reward = local_reward (already decomposed by RewardDecomposer)
        rewards.push(node.local_reward as f32);

        // Timestep = position index
        timesteps.push(t as u32);
    }

    // Returns-to-go: RTG_t = Σ_{i=t}^{T-1} reward_i
    let mut returns_to_go = vec![0.0f32; seq_len];
    let mut cumsum = 0.0f32;
    for t in (0..seq_len).rev() {
        cumsum += rewards[t];
        returns_to_go[t] = cumsum;
    }

    let created_at_epoch = traj.created_at.timestamp();

    Some(TrajectoryTensors {
        seq_len,
        returns_to_go,
        states_flat,
        actions_flat,
        rewards,
        timesteps,
        weight: source.weight(),
        total_reward: traj.total_reward as f32,
        created_at_epoch,
    })
}

/// Convert a batch of Trajectories into TrajectoryTensors, filtering invalid ones.
pub fn trajectories_to_tensors(trajectories: &[Trajectory]) -> Vec<TrajectoryTensors> {
    trajectories
        .iter()
        .filter_map(trajectory_to_tensors)
        .collect()
}

// ---------------------------------------------------------------------------
// Padding and batching — conversion to candle tensors
// ---------------------------------------------------------------------------

/// A padded batch of trajectories as candle tensors, ready for the Decision Transformer.
#[derive(Debug)]
pub struct TrajectoryBatch {
    /// Returns-to-go [batch_size, max_len].
    pub returns_to_go: Tensor,
    /// States [batch_size, max_len, STATE_DIM].
    pub states: Tensor,
    /// Actions [batch_size, max_len, ACTION_DIM].
    pub actions: Tensor,
    /// Rewards [batch_size, max_len].
    pub rewards: Tensor,
    /// Timestep indices [batch_size, max_len] (u32).
    pub timesteps: Tensor,
    /// Attention mask [batch_size, max_len] — 1.0 for real, 0.0 for padding.
    pub attention_mask: Tensor,
    /// Per-sample weights [batch_size].
    pub weights: Tensor,
    /// Batch size.
    pub batch_size: usize,
    /// Max sequence length in this batch.
    pub max_len: usize,
}

/// Pad a batch of TrajectoryTensors to a uniform max_length and convert to candle tensors.
///
/// Optionally applies z-score normalization if stats are provided.
pub fn pad_and_batch(
    batch: &[&TrajectoryTensors],
    max_length: usize,
    norm_stats: Option<&PolicyNormStats>,
    jitter_std: f32,
    seed: u64,
) -> CandleResult<TrajectoryBatch> {
    let device = Device::Cpu;
    let batch_size = batch.len();

    // Find the actual max_len (clamped to max_length)
    let max_len = batch
        .iter()
        .map(|t| t.seq_len.min(max_length))
        .max()
        .unwrap_or(1);

    let mut rtg_flat = vec![0.0f32; batch_size * max_len];
    let mut states_flat = vec![0.0f32; batch_size * max_len * STATE_DIM];
    let mut actions_flat = vec![0.0f32; batch_size * max_len * ACTION_DIM];
    let mut rewards_flat = vec![0.0f32; batch_size * max_len];
    let mut timesteps_flat = vec![0u32; batch_size * max_len];
    let mut mask_flat = vec![0.0f32; batch_size * max_len];
    let mut weights_flat = Vec::with_capacity(batch_size);

    for (b, traj) in batch.iter().enumerate() {
        let effective_len = traj.seq_len.min(max_length);
        weights_flat.push(traj.weight);

        for t in 0..effective_len {
            let batch_t = b * max_len + t;

            // RTG
            rtg_flat[batch_t] = traj.returns_to_go[t];

            // States (with optional normalization)
            let state_src_offset = t * STATE_DIM;
            let state_dst_offset = batch_t * STATE_DIM;
            for d in 0..STATE_DIM {
                let mut v = traj.states_flat[state_src_offset + d];
                if let Some(stats) = norm_stats {
                    v = (v - stats.state_mean[d]) / stats.state_std[d];
                }
                states_flat[state_dst_offset + d] = v;
            }

            // Actions (with optional normalization)
            let action_src_offset = t * ACTION_DIM;
            let action_dst_offset = batch_t * ACTION_DIM;
            for d in 0..ACTION_DIM {
                let mut v = traj.actions_flat[action_src_offset + d];
                if let Some(stats) = norm_stats {
                    v = (v - stats.action_mean[d]) / stats.action_std[d];
                }
                actions_flat[action_dst_offset + d] = v;
            }

            // Rewards (with optional normalization)
            let mut r = traj.rewards[t];
            if let Some(stats) = norm_stats {
                r = (r - stats.reward_mean) / stats.reward_std;
            }
            rewards_flat[batch_t] = r;

            // Timesteps
            timesteps_flat[batch_t] = traj.timesteps[t];

            // Mask
            mask_flat[batch_t] = 1.0;
        }
    }

    // Optional jitter for data augmentation
    if jitter_std > 0.0 {
        apply_jitter(&mut states_flat, &mask_flat, max_len, STATE_DIM, jitter_std, seed);
        apply_jitter(
            &mut actions_flat,
            &mask_flat,
            max_len,
            ACTION_DIM,
            jitter_std,
            seed.wrapping_add(1),
        );
    }

    // Convert to tensors
    let rtg = Tensor::from_vec(rtg_flat, (batch_size, max_len), &device)?;
    let states = Tensor::from_vec(states_flat, (batch_size, max_len, STATE_DIM), &device)?;
    let actions = Tensor::from_vec(actions_flat, (batch_size, max_len, ACTION_DIM), &device)?;
    let rewards = Tensor::from_vec(rewards_flat, (batch_size, max_len), &device)?;
    let timesteps =
        Tensor::from_vec(timesteps_flat, (batch_size, max_len), &device)?.to_dtype(DType::U32)?;
    let mask = Tensor::from_vec(mask_flat, (batch_size, max_len), &device)?;
    let weights = Tensor::from_vec(weights_flat, batch_size, &device)?;

    Ok(TrajectoryBatch {
        returns_to_go: rtg,
        states,
        actions,
        rewards,
        timesteps,
        attention_mask: mask,
        weights,
        batch_size,
        max_len,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pad or truncate a vector to target_dim.
fn pad_or_truncate(v: &[f32], target_dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; target_dim];
    let copy_len = v.len().min(target_dim);
    result[..copy_len].copy_from_slice(&v[..copy_len]);
    result
}

/// Apply Gaussian jitter to flattened data, respecting the attention mask.
/// Uses a simple deterministic PRNG (splitmix64-inspired) for reproducibility.
fn apply_jitter(
    data: &mut [f32],
    mask: &[f32],
    max_len: usize,
    dim: usize,
    std: f32,
    seed: u64,
) {
    let batch_size = mask.len() / max_len;
    for b in 0..batch_size {
        for t in 0..max_len {
            let mask_idx = b * max_len + t;
            if mask[mask_idx] < 0.5 {
                continue; // skip padding
            }
            let data_offset = mask_idx * dim;
            for d in 0..dim {
                // Box-Muller approximate: simple hash-based pseudo-Gaussian
                let h = splitmix64(seed, (b * max_len * dim + t * dim + d) as u64);
                let u = (h as f64) / (u64::MAX as f64);
                // Approximate Gaussian via Irwin-Hall (sum of 3 uniforms - 1.5) / sqrt(0.25)
                let h2 = splitmix64(seed.wrapping_add(1), (b * max_len * dim + t * dim + d) as u64);
                let u2 = (h2 as f64) / (u64::MAX as f64);
                let h3 = splitmix64(seed.wrapping_add(2), (b * max_len * dim + t * dim + d) as u64);
                let u3 = (h3 as f64) / (u64::MAX as f64);
                let noise = ((u + u2 + u3 - 1.5) / 0.5) as f32 * std;
                data[data_offset + d] += noise;
            }
        }
    }
}

/// Simple deterministic hash (splitmix64-inspired).
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_test_trajectory(num_nodes: usize, total_reward: f64, session_prefix: &str) -> Trajectory {
        let mut nodes = Vec::with_capacity(num_nodes);
        let per_step_reward = total_reward / num_nodes as f64;

        for i in 0..num_nodes {
            nodes.push(TrajectoryNode {
                id: Uuid::new_v4(),
                context_embedding: vec![0.1 * (i as f32 + 1.0); CONTEXT_DIM],
                action_type: format!("action_{}", i % 5),
                action_params: serde_json::json!({"key": i}),
                alternatives_count: 3,
                chosen_index: 0,
                confidence: 0.8,
                local_reward: per_step_reward,
                cumulative_reward: per_step_reward * (i + 1) as f64,
                delta_ms: 100,
                order: i,
            });
        }

        Trajectory {
            id: Uuid::new_v4(),
            session_id: format!("{}{}", session_prefix, Uuid::new_v4()),
            query_embedding: vec![0.5f32; QUERY_DIM],
            total_reward,
            step_count: num_nodes,
            duration_ms: 1000,
            nodes,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_trajectory_to_tensors_basic() {
        let traj = make_test_trajectory(5, 10.0, "session-");
        let tensors = trajectory_to_tensors(&traj).unwrap();

        assert_eq!(tensors.seq_len, 5);
        assert_eq!(tensors.states_flat.len(), 5 * STATE_DIM);
        assert_eq!(tensors.actions_flat.len(), 5 * ACTION_DIM);
        assert_eq!(tensors.rewards.len(), 5);
        assert_eq!(tensors.timesteps.len(), 5);

        // Returns-to-go: first should be total, last should be per-step
        assert!((tensors.returns_to_go[0] - 10.0).abs() < 0.01);
        assert!((tensors.returns_to_go[4] - 2.0).abs() < 0.01);

        // Weight for real trajectory
        assert!((tensors.weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_trajectory_to_tensors_empty() {
        let traj = Trajectory {
            id: Uuid::new_v4(),
            session_id: "test".to_string(),
            query_embedding: vec![],
            total_reward: 0.0,
            step_count: 0,
            duration_ms: 0,
            nodes: vec![],
            created_at: Utc::now(),
        };
        assert!(trajectory_to_tensors(&traj).is_none());
    }

    #[test]
    fn test_source_weights() {
        let real = make_test_trajectory(3, 5.0, "session-");
        let migrated = make_test_trajectory(3, 5.0, "migrated-");
        let simulated = make_test_trajectory(3, 5.0, "mcts-sim-");

        let t_real = trajectory_to_tensors(&real).unwrap();
        let t_mig = trajectory_to_tensors(&migrated).unwrap();
        let t_sim = trajectory_to_tensors(&simulated).unwrap();

        assert!((t_real.weight - 1.0).abs() < 0.01);
        assert!((t_mig.weight - 0.5).abs() < 0.01);
        assert!((t_sim.weight - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_norm_stats_computation() {
        let trajs: Vec<Trajectory> = (0..10)
            .map(|i| make_test_trajectory(4, 8.0 + i as f64, "session-"))
            .collect();
        let tensors = trajectories_to_tensors(&trajs);
        let stats = PolicyNormStats::compute(&tensors);

        assert_eq!(stats.state_mean.len(), STATE_DIM);
        assert_eq!(stats.state_std.len(), STATE_DIM);
        assert_eq!(stats.action_mean.len(), ACTION_DIM);
        assert_eq!(stats.action_std.len(), ACTION_DIM);

        // Std should be > 0 (we ensured eps minimum)
        for s in &stats.state_std {
            assert!(*s > 0.0);
        }
    }

    #[test]
    fn test_pad_and_batch() -> CandleResult<()> {
        let trajs: Vec<Trajectory> = (0..4)
            .map(|i| make_test_trajectory(3 + i, 6.0, "session-"))
            .collect();
        let tensors = trajectories_to_tensors(&trajs);
        let refs: Vec<&TrajectoryTensors> = tensors.iter().collect();

        let batch = pad_and_batch(&refs, 32, None, 0.0, 42)?;

        assert_eq!(batch.batch_size, 4);
        // max_len should be 6 (longest trajectory has 3+3=6 nodes)
        assert_eq!(batch.max_len, 6);

        // Check shapes
        assert_eq!(batch.states.dims(), &[4, 6, STATE_DIM]);
        assert_eq!(batch.actions.dims(), &[4, 6, ACTION_DIM]);
        assert_eq!(batch.returns_to_go.dims(), &[4, 6]);
        assert_eq!(batch.rewards.dims(), &[4, 6]);
        assert_eq!(batch.attention_mask.dims(), &[4, 6]);
        assert_eq!(batch.weights.dims(), &[4]);

        // Check mask: first trajectory has 3 steps, rest padded
        let mask_vec = batch.attention_mask.to_vec2::<f32>()?;
        // traj 0 has 3 steps
        assert!((mask_vec[0][0] - 1.0).abs() < 0.01);
        assert!((mask_vec[0][2] - 1.0).abs() < 0.01);
        assert!((mask_vec[0][3]).abs() < 0.01); // padding

        Ok(())
    }

    #[test]
    fn test_pad_and_batch_with_normalization() -> CandleResult<()> {
        let trajs: Vec<Trajectory> = (0..8)
            .map(|i| make_test_trajectory(4, 8.0 + i as f64, "session-"))
            .collect();
        let tensors = trajectories_to_tensors(&trajs);
        let stats = PolicyNormStats::compute(&tensors);

        let refs: Vec<&TrajectoryTensors> = tensors.iter().collect();
        let batch = pad_and_batch(&refs, 32, Some(&stats), 0.0, 42)?;

        // After z-score normalization, values should be roughly centered around 0
        let states_flat = batch.states.flatten_all()?.to_vec1::<f32>()?;
        let sum: f32 = states_flat.iter().sum();
        let mean = sum / states_flat.len() as f32;
        // Mean should be close to 0 (within tolerance due to padding zeros)
        assert!(
            mean.abs() < 1.0,
            "Expected normalized mean near 0, got {}",
            mean
        );

        Ok(())
    }

    #[test]
    fn test_pad_and_batch_with_jitter() -> CandleResult<()> {
        let trajs: Vec<Trajectory> = (0..2)
            .map(|_| make_test_trajectory(3, 6.0, "session-"))
            .collect();
        let tensors = trajectories_to_tensors(&trajs);
        let refs: Vec<&TrajectoryTensors> = tensors.iter().collect();

        let batch_no_jitter = pad_and_batch(&refs, 32, None, 0.0, 42)?;
        let batch_jitter = pad_and_batch(&refs, 32, None, 0.01, 42)?;

        // With jitter, states should differ slightly
        let s1 = batch_no_jitter.states.flatten_all()?.to_vec1::<f32>()?;
        let s2 = batch_jitter.states.flatten_all()?.to_vec1::<f32>()?;

        let diff: f32 = s1.iter().zip(s2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "Jitter should change values");
        // But not by too much (std=0.01)
        let avg_diff = diff / s1.len() as f32;
        assert!(
            avg_diff < 0.1,
            "Jitter should be small, got avg diff {}",
            avg_diff
        );

        Ok(())
    }

    #[test]
    fn test_returns_to_go_monotone_decreasing() {
        let traj = make_test_trajectory(8, 16.0, "session-");
        let tensors = trajectory_to_tensors(&traj).unwrap();

        // RTG should be monotonically non-increasing
        for t in 1..tensors.seq_len {
            assert!(
                tensors.returns_to_go[t] <= tensors.returns_to_go[t - 1] + 0.001,
                "RTG should decrease: rtg[{}]={} > rtg[{}]={}",
                t,
                tensors.returns_to_go[t],
                t - 1,
                tensors.returns_to_go[t - 1]
            );
        }
    }
}
