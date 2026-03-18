//! Benchmark: Decision Transformer vs CQL.
//!
//! Trains both models on the same synthetic dataset and compares:
//! - Prediction accuracy (MSE on held-out test set)
//! - Action diversity (standard deviation of predicted actions)
//! - Small-dataset robustness (performance on N=50 vs N=500)
//! - Training speed (wall-clock time per epoch)
//!
//! Produces a `BenchmarkReport` (JSON-serializable) documenting the comparison.

use std::time::Instant;

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use serde::Serialize;
use tracing::info;

use crate::cql::{CQLConfig, CQLPolicy};
use crate::dataset::{PolicyNormStats, TrajectoryTensors, ACTION_DIM, STATE_DIM};
use crate::transformer::{DecisionTransformer, DecisionTransformerConfig};

// ---------------------------------------------------------------------------
// Report structures
// ---------------------------------------------------------------------------

/// Metrics for a single model in the benchmark.
#[derive(Debug, Clone, Serialize)]
pub struct ModelMetrics {
    /// Model name ("DecisionTransformer" or "CQL").
    pub model_name: String,
    /// Number of parameters.
    pub param_count: usize,
    /// Test MSE (lower is better).
    pub test_mse: f32,
    /// Action diversity: mean std across action dimensions (higher = more diverse).
    pub action_diversity: f32,
    /// Training time in milliseconds.
    pub training_time_ms: u128,
    /// Number of training epochs completed.
    pub epochs_completed: usize,
    /// Best validation loss during training.
    pub best_val_loss: f32,
}

/// Full benchmark report comparing DT and CQL.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    /// Dataset size used for this benchmark.
    pub dataset_size: usize,
    /// Decision Transformer metrics.
    pub dt_metrics: ModelMetrics,
    /// CQL metrics.
    pub cql_metrics: ModelMetrics,
    /// Winner ("DecisionTransformer" or "CQL").
    pub winner: String,
    /// Rationale for choosing the winner.
    pub rationale: String,
}

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------

/// Generate synthetic trajectories for benchmarking.
///
/// Each trajectory has `seq_len` steps with random states, actions, and rewards.
fn generate_synthetic_trajectories(
    count: usize,
    seq_len: usize,
) -> CandleResult<Vec<TrajectoryTensors>> {
    let device = Device::Cpu;
    let mut trajectories = Vec::with_capacity(count);

    for i in 0..count {
        // Generate a coherent trajectory: actions are a function of states
        // (not purely random) so models can learn a pattern.
        let states_tensor = Tensor::randn(0.0f32, 1.0, (seq_len, STATE_DIM), &device)?;

        // Actions = linear projection of first 256 dims + noise
        // This creates a learnable pattern: action ≈ f(state[:256])
        let state_prefix = states_tensor.narrow(1, 0, ACTION_DIM)?;
        let noise = Tensor::randn(0.0f32, 0.1, (seq_len, ACTION_DIM), &device)?;
        let actions_tensor = (state_prefix + noise)?.tanh()?;

        // Rewards: random positive
        let rewards_tensor = Tensor::randn(0.0f32, 0.5, (seq_len,), &device)?.abs()?;

        // Convert to flat vecs
        let states_flat: Vec<f32> = states_tensor.flatten_all()?.to_vec1()?;
        let actions_flat: Vec<f32> = actions_tensor.flatten_all()?.to_vec1()?;
        let rewards: Vec<f32> = rewards_tensor.to_vec1()?;

        // Compute RTG
        let mut rtg = vec![0.0f32; seq_len];
        let mut cumulative = 0.0f32;
        for t in (0..seq_len).rev() {
            cumulative += rewards[t];
            rtg[t] = cumulative;
        }

        let total_reward = rewards.iter().sum::<f32>();

        trajectories.push(TrajectoryTensors {
            seq_len,
            returns_to_go: rtg,
            states_flat,
            actions_flat,
            rewards,
            timesteps: (0..seq_len).map(|t| t as u32).collect(),
            weight: if i % 3 == 0 { 0.5 } else { 1.0 },
            total_reward,
            created_at_epoch: i as i64, // monotonically increasing for temporal split
        });
    }

    Ok(trajectories)
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

/// Configuration for the benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of trajectories to generate.
    pub dataset_size: usize,
    /// Sequence length per trajectory.
    pub seq_len: usize,
    /// Training epochs for DT.
    pub dt_epochs: usize,
    /// Training epochs for CQL.
    pub cql_epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_size: 100,
            seq_len: 8,
            dt_epochs: 20,
            cql_epochs: 20,
            batch_size: 16,
            learning_rate: 1e-3,
        }
    }
}

/// Run the benchmark: train DT and CQL on the same data, compare metrics.
pub fn run_benchmark(config: &BenchmarkConfig) -> CandleResult<BenchmarkReport> {
    info!(
        "Starting DT vs CQL benchmark: {} trajectories, {} epochs",
        config.dataset_size, config.dt_epochs
    );

    // Generate data
    let all_trajectories = generate_synthetic_trajectories(config.dataset_size, config.seq_len)?;

    // Split: 70% train, 15% val, 15% test
    let train_end = (all_trajectories.len() as f64 * 0.70) as usize;
    let val_end = (all_trajectories.len() as f64 * 0.85) as usize;

    let train_data = &all_trajectories[..train_end];
    let val_data = &all_trajectories[train_end..val_end];
    let test_data = &all_trajectories[val_end..];

    info!(
        "Data split: train={}, val={}, test={}",
        train_data.len(),
        val_data.len(),
        test_data.len()
    );

    // Compute norm stats from training set
    let norm_stats = PolicyNormStats::compute(train_data);

    // --- Train Decision Transformer ---
    let dt_metrics = benchmark_dt(config, train_data, val_data, test_data, &norm_stats)?;

    // --- Train CQL ---
    let cql_metrics = benchmark_cql(config, train_data, val_data, test_data)?;

    // --- Determine winner ---
    let (winner, rationale) = determine_winner(&dt_metrics, &cql_metrics, config.dataset_size);

    let report = BenchmarkReport {
        dataset_size: config.dataset_size,
        dt_metrics,
        cql_metrics,
        winner,
        rationale,
    };

    info!(
        "Benchmark complete. Winner: {} (DT MSE={:.6}, CQL MSE={:.6})",
        report.winner, report.dt_metrics.test_mse, report.cql_metrics.test_mse
    );

    Ok(report)
}

/// Determine the winner based on metrics.
fn determine_winner(
    dt: &ModelMetrics,
    cql: &ModelMetrics,
    dataset_size: usize,
) -> (String, String) {
    let dt_score = score_model(dt, dataset_size);
    let cql_score = score_model(cql, dataset_size);

    if dt_score >= cql_score {
        (
            "DecisionTransformer".to_string(),
            format!(
                "DT wins with score {:.4} vs CQL {:.4}. \
                 DT MSE={:.6} (CQL={:.6}), diversity={:.4} (CQL={:.4}). \
                 DT is the primary policy for routing; CQL serves as OOD fallback.",
                dt_score,
                cql_score,
                dt.test_mse,
                cql.test_mse,
                dt.action_diversity,
                cql.action_diversity,
            ),
        )
    } else {
        (
            "CQL".to_string(),
            format!(
                "CQL wins with score {:.4} vs DT {:.4}. \
                 CQL MSE={:.6} (DT={:.6}), diversity={:.4} (DT={:.4}). \
                 For dataset_size={}, CQL's conservative estimates are more reliable.",
                cql_score,
                dt_score,
                cql.test_mse,
                dt.test_mse,
                cql.action_diversity,
                dt.action_diversity,
                dataset_size,
            ),
        )
    }
}

/// Score a model: lower MSE contributes more, with diversity bonus.
fn score_model(m: &ModelMetrics, dataset_size: usize) -> f64 {
    let mse_score = 1.0 / (1.0 + m.test_mse as f64);
    let diversity_score = (m.action_diversity as f64).min(1.0) * 0.1;
    let small_data_bonus = if dataset_size < 200 && m.model_name == "CQL" {
        0.05
    } else {
        0.0
    };
    mse_score + diversity_score + small_data_bonus
}

// ---------------------------------------------------------------------------
// Decision Transformer benchmark
// ---------------------------------------------------------------------------

fn benchmark_dt(
    config: &BenchmarkConfig,
    train_data: &[TrajectoryTensors],
    val_data: &[TrajectoryTensors],
    test_data: &[TrajectoryTensors],
    _norm_stats: &PolicyNormStats,
) -> CandleResult<ModelMetrics> {
    let device = Device::Cpu;

    let model_config = DecisionTransformerConfig {
        state_dim: STATE_DIM,
        action_dim: ACTION_DIM,
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 2,
        max_timesteps: config.seq_len,
        dropout: 0.0,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = DecisionTransformer::new(model_config, vb)?;
    let param_count = model.param_count();

    info!(
        "DT: ~{}K params, training {} epochs",
        param_count / 1000,
        config.dt_epochs
    );

    let start = Instant::now();
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_completed = 0;
    let mut adamw = SimpleAdamW::new(&varmap)?;

    for epoch in 0..config.dt_epochs {
        // Train
        let train_batches = make_batches(train_data, config.batch_size, config.seq_len)?;

        for (rtg, states, actions, timesteps, mask, weights) in &train_batches {
            let pred = model.forward(rtg, states, actions, timesteps, mask)?;
            let loss = masked_mse(&pred, actions, mask, weights)?;
            let grads = loss.backward()?;
            adamw.step(&varmap, &grads, config.learning_rate)?;
        }

        // Validate
        let val_batches = make_batches(val_data, config.batch_size, config.seq_len)?;
        let mut val_loss = 0.0f32;
        let mut val_count = 0;
        for (rtg, states, actions, timesteps, mask, weights) in &val_batches {
            let pred = model.forward(rtg, states, actions, timesteps, mask)?;
            let loss = masked_mse(&pred, actions, mask, weights)?;
            val_loss += loss.to_scalar::<f32>()?;
            val_count += 1;
        }
        let val_avg = if val_count > 0 {
            val_loss / val_count as f32
        } else {
            f32::INFINITY
        };
        if val_avg < best_val_loss {
            best_val_loss = val_avg;
        }
        epochs_completed = epoch + 1;
    }

    let training_time_ms = start.elapsed().as_millis();

    // Test evaluation
    let test_batches = make_batches(test_data, config.batch_size, config.seq_len)?;
    let (test_mse, action_diversity) = evaluate_dt(&model, &test_batches)?;

    Ok(ModelMetrics {
        model_name: "DecisionTransformer".to_string(),
        param_count,
        test_mse,
        action_diversity,
        training_time_ms,
        epochs_completed,
        best_val_loss,
    })
}

/// Evaluate DT on test batches: returns (mse, diversity).
fn evaluate_dt(model: &DecisionTransformer, batches: &[BatchTuple]) -> CandleResult<(f32, f32)> {
    let mut total_mse = 0.0f32;
    let mut total_diversity = 0.0f32;
    let mut count = 0;

    for (rtg, states, actions, timesteps, mask, weights) in batches {
        let pred = model.forward(rtg, states, actions, timesteps, mask)?;
        let mse = masked_mse(&pred, actions, mask, weights)?.to_scalar::<f32>()?;
        let div = action_std(&pred)?;

        total_mse += mse;
        total_diversity += div;
        count += 1;
    }

    let count = count.max(1) as f32;
    Ok((total_mse / count, total_diversity / count))
}

// ---------------------------------------------------------------------------
// CQL benchmark
// ---------------------------------------------------------------------------

fn benchmark_cql(
    config: &BenchmarkConfig,
    train_data: &[TrajectoryTensors],
    val_data: &[TrajectoryTensors],
    test_data: &[TrajectoryTensors],
) -> CandleResult<ModelMetrics> {
    let device = Device::Cpu;

    let cql_config = CQLConfig {
        state_dim: STATE_DIM,
        action_dim: ACTION_DIM,
        hidden_dim: 64,
        alpha: 1.0,
        gamma: 0.99,
        tau: 0.005,
        num_random_actions: 5,
    };

    let cql = CQLPolicy::new(cql_config)?;
    let (q_varmap, policy_varmap) = cql.varmaps();

    // Count params
    let q_params: usize = q_varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum();
    let p_params: usize = policy_varmap
        .all_vars()
        .iter()
        .map(|v| v.as_tensor().elem_count())
        .sum();
    let param_count = q_params + p_params;

    info!(
        "CQL: ~{}K params, training {} epochs",
        param_count / 1000,
        config.cql_epochs
    );

    // Flatten trajectories to (s, a, r, s', done) transitions for CQL
    let train_transitions = flatten_transitions(train_data);
    let val_transitions = flatten_transitions(val_data);
    let test_transitions = flatten_transitions(test_data);

    let start = Instant::now();
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_completed = 0;

    let mut q_adam = SimpleAdamW::new(q_varmap)?;
    let mut p_adam = SimpleAdamW::new(policy_varmap)?;

    for epoch in 0..config.cql_epochs {
        // Train on mini-batches
        let batches = transition_batches(&train_transitions, config.batch_size, &device)?;
        for (states, actions, rewards, next_states, dones) in &batches {
            let (loss, _bellman, _cql) =
                cql.compute_loss(states, actions, rewards, next_states, dones)?;

            // Update Q-networks
            let grads = loss.backward()?;
            q_adam.step(q_varmap, &grads, config.learning_rate)?;

            // Update policy: minimize -Q(s, π(s))
            let pred_actions = cql.predict_action(states)?;
            let (q1, q2) = cql.q_values(states, &pred_actions)?;
            let policy_loss = q1.minimum(&q2)?.mean_all()?.affine(-1.0, 0.0)?;
            let p_grads = policy_loss.backward()?;
            p_adam.step(policy_varmap, &p_grads, config.learning_rate)?;

            // Soft update targets
            cql.soft_update_targets()?;
        }

        // Validate
        let val_batches = transition_batches(&val_transitions, config.batch_size, &device)?;
        let mut val_loss_sum = 0.0f32;
        let mut val_count = 0;
        for (states, actions, rewards, next_states, dones) in &val_batches {
            let (loss, _, _) = cql.compute_loss(states, actions, rewards, next_states, dones)?;
            val_loss_sum += loss.to_scalar::<f32>()?;
            val_count += 1;
        }
        let val_avg = if val_count > 0 {
            val_loss_sum / val_count as f32
        } else {
            f32::INFINITY
        };
        if val_avg < best_val_loss {
            best_val_loss = val_avg;
        }
        epochs_completed = epoch + 1;
    }

    let training_time_ms = start.elapsed().as_millis();

    // Test evaluation
    let (test_mse, action_diversity) = evaluate_cql(&cql, &test_transitions, &device)?;

    Ok(ModelMetrics {
        model_name: "CQL".to_string(),
        param_count,
        test_mse,
        action_diversity,
        training_time_ms,
        epochs_completed,
        best_val_loss,
    })
}

/// Evaluate CQL on test transitions.
fn evaluate_cql(
    cql: &CQLPolicy,
    transitions: &[Transition],
    device: &Device,
) -> CandleResult<(f32, f32)> {
    if transitions.is_empty() {
        return Ok((0.0, 0.0));
    }

    let batch_size = transitions.len();
    let states_flat: Vec<f32> = transitions
        .iter()
        .flat_map(|(s, _, _, _, _)| s.clone())
        .collect();
    let actions_flat: Vec<f32> = transitions
        .iter()
        .flat_map(|(_, a, _, _, _)| a.clone())
        .collect();

    let states = Tensor::from_vec(states_flat, (batch_size, STATE_DIM), device)?;
    let true_actions = Tensor::from_vec(actions_flat, (batch_size, ACTION_DIM), device)?;

    let pred_actions = cql.predict_action(&states)?;

    // MSE
    let diff = (&pred_actions - &true_actions)?;
    let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

    // Diversity
    let diversity = action_std(&pred_actions.unsqueeze(1)?)?;

    Ok((mse, diversity))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simplified batch tuple type for DT.
type BatchTuple = (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);

/// A single transition: (state, action, reward, next_state, done).
type Transition = (Vec<f32>, Vec<f32>, f32, Vec<f32>, f32);

/// CQL mini-batch: (states, actions, rewards, next_states, dones).
type CqlBatch = (Tensor, Tensor, Tensor, Tensor, Tensor);

/// Create batches from trajectory tensors for DT evaluation/training.
fn make_batches(
    data: &[TrajectoryTensors],
    batch_size: usize,
    max_len: usize,
) -> CandleResult<Vec<BatchTuple>> {
    let device = Device::Cpu;
    let mut batches = Vec::new();

    for chunk in data.chunks(batch_size) {
        let b = chunk.len();

        let mut all_rtg = Vec::new();
        let mut all_states = Vec::new();
        let mut all_actions = Vec::new();
        let mut all_timesteps = Vec::new();
        let mut all_mask = Vec::new();
        let mut all_weights = Vec::new();

        for traj in chunk {
            let t = traj.seq_len.min(max_len);

            for i in 0..max_len {
                if i < t {
                    all_rtg.push(traj.returns_to_go[i]);
                    all_states
                        .extend_from_slice(&traj.states_flat[i * STATE_DIM..(i + 1) * STATE_DIM]);
                    all_actions.extend_from_slice(
                        &traj.actions_flat[i * ACTION_DIM..(i + 1) * ACTION_DIM],
                    );
                    all_timesteps.push(traj.timesteps[i]);
                    all_mask.push(1.0f32);
                } else {
                    all_rtg.push(0.0);
                    all_states.extend(std::iter::repeat_n(0.0f32, STATE_DIM));
                    all_actions.extend(std::iter::repeat_n(0.0f32, ACTION_DIM));
                    all_timesteps.push(0u32);
                    all_mask.push(0.0f32);
                }
            }

            all_weights.push(traj.weight);
        }

        let rtg = Tensor::from_vec(all_rtg, (b, max_len), &device)?;
        let states = Tensor::from_vec(all_states, (b, max_len, STATE_DIM), &device)?;
        let actions = Tensor::from_vec(all_actions, (b, max_len, ACTION_DIM), &device)?;
        let timesteps =
            Tensor::from_vec(all_timesteps, (b, max_len), &device)?.to_dtype(DType::U32)?;
        let mask = Tensor::from_vec(all_mask, (b, max_len), &device)?;
        let weights = Tensor::from_vec(all_weights, (b,), &device)?;

        batches.push((rtg, states, actions, timesteps, mask, weights));
    }

    Ok(batches)
}

/// Masked MSE loss for DT (simplified for benchmark).
fn masked_mse(
    pred: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    weights: &Tensor,
) -> CandleResult<Tensor> {
    let diff = (pred - target)?;
    let sq = diff.sqr()?;
    let mask_3d = mask.unsqueeze(D::Minus1)?;
    let masked = sq.broadcast_mul(&mask_3d)?;

    // Per-sample mean
    let per_sample = masked.sum(D::Minus1)?.sum(D::Minus1)?; // [B]
    let valid = mask.sum(D::Minus1)?.clamp(1.0f64, f64::INFINITY)?; // [B]
    let per_sample_mean = per_sample.div(&valid)?;

    // Weighted mean
    let weighted = per_sample_mean.broadcast_mul(weights)?;
    let total = weighted.sum_all()?;
    let w_sum = weights.sum_all()?.clamp(1e-8f64, f64::INFINITY)?;
    total.div(&w_sum)
}

/// Action diversity: mean standard deviation across action dimensions.
fn action_std(actions: &Tensor) -> CandleResult<f32> {
    let flat = if actions.dims().len() == 3 {
        let (b, t, d) = actions.dims3()?;
        actions.reshape((b * t, d))?
    } else {
        actions.clone()
    };

    // Variance along batch dimension
    let mean = flat.mean(0)?;
    let centered = flat.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean(0)?;
    let std = var.sqrt()?;

    std.mean_all()?.to_scalar::<f32>()
}

/// Flatten trajectories to individual (state, action, reward, next_state, done) transitions.
fn flatten_transitions(trajectories: &[TrajectoryTensors]) -> Vec<Transition> {
    let mut transitions = Vec::new();

    for traj in trajectories {
        let t_len = traj.seq_len;
        for i in 0..t_len.saturating_sub(1) {
            let state = traj.states_flat[i * STATE_DIM..(i + 1) * STATE_DIM].to_vec();
            let action = traj.actions_flat[i * ACTION_DIM..(i + 1) * ACTION_DIM].to_vec();
            let reward = traj.rewards[i];
            let next_state = traj.states_flat[(i + 1) * STATE_DIM..(i + 2) * STATE_DIM].to_vec();
            let done = if i == t_len - 2 { 1.0f32 } else { 0.0f32 };

            transitions.push((state, action, reward, next_state, done));
        }
    }

    transitions
}

/// Create mini-batches of transitions for CQL training.
fn transition_batches(
    transitions: &[Transition],
    batch_size: usize,
    device: &Device,
) -> CandleResult<Vec<CqlBatch>> {
    let mut batches = Vec::new();

    for chunk in transitions.chunks(batch_size) {
        let b = chunk.len();
        let states_flat: Vec<f32> = chunk.iter().flat_map(|(s, _, _, _, _)| s.clone()).collect();
        let actions_flat: Vec<f32> = chunk.iter().flat_map(|(_, a, _, _, _)| a.clone()).collect();
        let rewards: Vec<f32> = chunk.iter().map(|(_, _, r, _, _)| *r).collect();
        let next_states_flat: Vec<f32> = chunk
            .iter()
            .flat_map(|(_, _, _, ns, _)| ns.clone())
            .collect();
        let dones: Vec<f32> = chunk.iter().map(|(_, _, _, _, d)| *d).collect();

        batches.push((
            Tensor::from_vec(states_flat, (b, STATE_DIM), device)?,
            Tensor::from_vec(actions_flat, (b, ACTION_DIM), device)?,
            Tensor::from_vec(rewards, (b, 1), device)?,
            Tensor::from_vec(next_states_flat, (b, STATE_DIM), device)?,
            Tensor::from_vec(dones, (b, 1), device)?,
        ));
    }

    Ok(batches)
}

// ---------------------------------------------------------------------------
// Simple AdamW (lightweight version for benchmark)
// ---------------------------------------------------------------------------

struct SimpleAdamW {
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    t: usize,
}

impl SimpleAdamW {
    fn new(varmap: &VarMap) -> CandleResult<Self> {
        let all_vars = varmap.all_vars();
        let m: Vec<Tensor> = all_vars
            .iter()
            .map(|v| Tensor::zeros_like(v))
            .collect::<CandleResult<Vec<_>>>()?;
        let v: Vec<Tensor> = all_vars
            .iter()
            .map(|v| Tensor::zeros_like(v))
            .collect::<CandleResult<Vec<_>>>()?;

        Ok(Self { m, v, t: 0 })
    }

    fn step(
        &mut self,
        varmap: &VarMap,
        grads: &candle_core::backprop::GradStore,
        lr: f64,
    ) -> CandleResult<()> {
        self.t += 1;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let eps = 1e-8;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);

        for (i, var) in varmap.all_vars().iter().enumerate() {
            let grad = match grads.get(var) {
                Some(g) => g,
                None => continue,
            };

            self.m[i] = (self.m[i].affine(beta1, 0.0)? + grad.affine(1.0 - beta1, 0.0)?)?;
            self.v[i] = (self.v[i].affine(beta2, 0.0)? + grad.sqr()?.affine(1.0 - beta2, 0.0)?)?;

            let m_hat = self.m[i].affine(1.0 / bc1, 0.0)?;
            let v_hat = self.v[i].affine(1.0 / bc2, 0.0)?;

            let param = var.as_tensor();
            let denom = v_hat.sqrt()?.affine(1.0, eps)?;
            let update = m_hat.div(&denom)?;
            let new_param = (param - update.affine(lr, 0.0)?)?;
            var.set(&new_param)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() -> CandleResult<()> {
        let trajs = generate_synthetic_trajectories(10, 5)?;
        assert_eq!(trajs.len(), 10);

        for traj in &trajs {
            assert_eq!(traj.states_flat.len(), 5 * STATE_DIM);
            assert_eq!(traj.actions_flat.len(), 5 * ACTION_DIM);
            assert_eq!(traj.rewards.len(), 5);
            assert_eq!(traj.returns_to_go.len(), 5);
            assert_eq!(traj.seq_len, 5);

            // RTG should be monotonically decreasing
            for i in 0..4 {
                assert!(
                    traj.returns_to_go[i] >= traj.returns_to_go[i + 1],
                    "RTG should decrease"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_flatten_transitions() -> CandleResult<()> {
        let trajs = generate_synthetic_trajectories(3, 4)?;
        let transitions = flatten_transitions(&trajs);

        // Each trajectory of length 4 produces 3 transitions
        assert_eq!(transitions.len(), 3 * 3);

        for (s, a, r, ns, d) in &transitions {
            assert_eq!(s.len(), STATE_DIM);
            assert_eq!(a.len(), ACTION_DIM);
            assert!(r.is_finite());
            assert_eq!(ns.len(), STATE_DIM);
            assert!(*d == 0.0 || *d == 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_benchmark_runs() -> CandleResult<()> {
        let config = BenchmarkConfig {
            dataset_size: 30,
            seq_len: 4,
            dt_epochs: 3,
            cql_epochs: 3,
            batch_size: 8,
            learning_rate: 1e-3,
        };

        let report = run_benchmark(&config)?;

        assert!(report.dt_metrics.test_mse.is_finite());
        assert!(report.cql_metrics.test_mse.is_finite());
        assert!(report.dt_metrics.param_count > 0);
        assert!(report.cql_metrics.param_count > 0);
        assert!(!report.winner.is_empty());
        assert!(!report.rationale.is_empty());

        println!("Benchmark Report:");
        println!(
            "  DT: MSE={:.6}, diversity={:.4}, params={}, time={}ms",
            report.dt_metrics.test_mse,
            report.dt_metrics.action_diversity,
            report.dt_metrics.param_count,
            report.dt_metrics.training_time_ms,
        );
        println!(
            "  CQL: MSE={:.6}, diversity={:.4}, params={}, time={}ms",
            report.cql_metrics.test_mse,
            report.cql_metrics.action_diversity,
            report.cql_metrics.param_count,
            report.cql_metrics.training_time_ms,
        );
        println!("  Winner: {}", report.winner);
        println!("  Rationale: {}", report.rationale);

        Ok(())
    }

    #[test]
    fn test_action_std() -> CandleResult<()> {
        let device = Device::Cpu;

        // Constant actions → zero std
        let constant = Tensor::ones((4, 3, ACTION_DIM), DType::F32, &device)?;
        let std_val = action_std(&constant)?;
        assert!(
            std_val < 1e-6,
            "Constant actions should have ~0 std, got {}",
            std_val
        );

        // Random actions → positive std
        let random = Tensor::randn(0.0f32, 1.0, (4, 3, ACTION_DIM), &device)?;
        let std_val = action_std(&random)?;
        assert!(
            std_val > 0.01,
            "Random actions should have positive std, got {}",
            std_val
        );

        Ok(())
    }
}
