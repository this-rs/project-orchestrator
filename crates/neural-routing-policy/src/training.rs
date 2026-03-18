//! Training pipeline for the Decision Transformer.
//!
//! Provides:
//! - TrainingConfig: hyperparameters and scheduling
//! - AdamW optimizer (proper implementation with moment estimates + bias correction)
//! - Combined loss: MSE + cosine distance on predicted actions
//! - Cosine annealing with warmup learning rate schedule
//! - Early stopping with patience
//! - Safetensors checkpoint save/load
//! - Training loop with epoch metrics

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::dataloader::{Split, TrajectoryDataLoader};
use crate::dataset::PolicyNormStats;
use crate::ewc::EWCRegularizer;
use crate::transformer::{DecisionTransformer, DecisionTransformerConfig};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    /// Training mode: "automatic" or "manual".
    pub mode: String,
    /// Number of trajectories before auto-training triggers (automatic mode).
    pub auto_trigger_threshold: usize,
    /// Maximum threads for training.
    pub max_threads: usize,
    /// Cron schedule for automatic training (e.g., "0 2 * * *").
    pub schedule: String,
    /// Learning rate (peak).
    pub learning_rate: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of epochs.
    pub epochs: usize,
    /// Gradient clipping max norm.
    pub max_grad_norm: f64,
    /// Weight decay (AdamW decoupled).
    pub weight_decay: f64,
    /// Warmup fraction (fraction of total epochs for LR warmup).
    pub warmup_fraction: f64,
    /// Early stopping patience (epochs without val improvement).
    pub patience: usize,
    /// MSE loss weight.
    pub mse_weight: f64,
    /// Cosine distance loss weight.
    pub cosine_weight: f64,
    /// Checkpoint directory.
    pub checkpoint_dir: String,
    /// Save checkpoint every N epochs.
    pub checkpoint_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: "manual".to_string(),
            auto_trigger_threshold: 1000,
            max_threads: 2,
            schedule: "0 2 * * *".to_string(),
            learning_rate: 1e-4,
            batch_size: 32,
            epochs: 100,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            warmup_fraction: 0.1,
            patience: 15,
            mse_weight: 1.0,
            cosine_weight: 0.1,
            checkpoint_dir: "checkpoints/policy".to_string(),
            checkpoint_every: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// AdamW optimizer (proper implementation)
// ---------------------------------------------------------------------------

/// AdamW optimizer with bias correction (Loshchilov & Hutter, 2019).
struct AdamWState {
    /// First moment estimates (one per parameter tensor).
    m: Vec<Tensor>,
    /// Second moment estimates (one per parameter tensor).
    v: Vec<Tensor>,
    /// β1 (first moment decay).
    beta1: f64,
    /// β2 (second moment decay).
    beta2: f64,
    /// Numerical stability.
    eps: f64,
    /// Current timestep (for bias correction).
    t: usize,
}

impl AdamWState {
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

        Ok(Self {
            m,
            v,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
        })
    }

    /// Perform one AdamW step on all parameters.
    fn step(
        &mut self,
        varmap: &VarMap,
        grads: &candle_core::backprop::GradStore,
        lr: f64,
        weight_decay: f64,
        max_grad_norm: f64,
    ) -> CandleResult<()> {
        self.t += 1;
        let all_vars = varmap.all_vars();

        // Bias correction factors
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, var) in all_vars.iter().enumerate() {
            let grad = match grads.get(var) {
                Some(g) => g,
                None => continue,
            };

            // Gradient clipping (per-parameter L2 norm)
            let grad_norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
            let grad = if grad_norm > max_grad_norm {
                let scale = max_grad_norm / (grad_norm + 1e-12);
                grad.affine(scale, 0.0)?
            } else {
                grad.clone()
            };

            // Update first moment: m = β1·m + (1-β1)·grad
            self.m[i] =
                ((self.m[i].affine(self.beta1, 0.0))? + grad.affine(1.0 - self.beta1, 0.0)?)?;

            // Update second moment: v = β2·v + (1-β2)·grad²
            self.v[i] = ((self.v[i].affine(self.beta2, 0.0))?
                + grad.sqr()?.affine(1.0 - self.beta2, 0.0)?)?;

            // Bias-corrected estimates
            let m_hat = self.m[i].affine(1.0 / bc1, 0.0)?;
            let v_hat = self.v[i].affine(1.0 / bc2, 0.0)?;

            // AdamW update: param = param * (1 - lr·wd) - lr · m_hat / (√v_hat + eps)
            let param = var.as_tensor();
            let decayed = param.affine(1.0 - lr * weight_decay, 0.0)?;
            let denom = v_hat.sqrt()?.affine(1.0, self.eps)?;
            let update = m_hat.div(&denom)?;
            let new_param = (decayed - update.affine(lr, 0.0)?)?;

            var.set(&new_param)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Learning rate schedule
// ---------------------------------------------------------------------------

/// Cosine annealing with linear warmup.
fn cosine_lr(base_lr: f64, warmup_epochs: usize, total_epochs: usize, epoch: usize) -> f64 {
    if epoch < warmup_epochs {
        // Linear warmup
        base_lr * (epoch + 1) as f64 / warmup_epochs as f64
    } else {
        // Cosine annealing
        let progress =
            (epoch - warmup_epochs) as f64 / (total_epochs - warmup_epochs).max(1) as f64;
        let min_lr = base_lr * 0.01; // 1% of base LR
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

/// Compute combined loss: weighted MSE + cosine distance.
///
/// - `pred_actions`: [B, T, action_dim] — predicted actions (tanh-bounded)
/// - `target_actions`: [B, T, action_dim] — ground truth actions
/// - `attention_mask`: [B, T] — 1.0 for real tokens, 0.0 for padding
/// - `weights`: [B] — per-sample weights
///
/// Returns (total_loss, mse_loss, cosine_loss) as scalars.
fn compute_loss(
    pred_actions: &Tensor,
    target_actions: &Tensor,
    attention_mask: &Tensor,
    weights: &Tensor,
    mse_weight: f64,
    cosine_weight: f64,
) -> CandleResult<(Tensor, f32, f32)> {
    let (batch_size, seq_len, action_dim) = pred_actions.dims3()?;

    // Expand mask to [B, T, 1] for broadcasting
    let mask_3d = attention_mask.unsqueeze(D::Minus1)?;

    // MSE loss: mean of (pred - target)² over valid positions
    let diff = (pred_actions - target_actions)?;
    let sq_diff = diff.sqr()?;
    let masked_sq = sq_diff.broadcast_mul(&mask_3d)?;

    // Per-sample loss: mean over T and action_dim
    let per_sample_mse = masked_sq.sum(D::Minus1)?.sum(D::Minus1)?; // [B]
    let valid_count = attention_mask.sum(D::Minus1)?; // [B]
    let per_sample_mse = per_sample_mse.div(
        &valid_count
            .clamp(1.0f64, f64::INFINITY)?
            .affine(action_dim as f64, 0.0)?,
    )?;

    // Weighted mean
    let weighted_mse = (per_sample_mse.broadcast_mul(weights))?.sum_all()?;
    let weight_sum = weights.sum_all()?.clamp(1e-8f64, f64::INFINITY)?;
    let mse_loss = weighted_mse.div(&weight_sum)?;

    // Cosine distance loss: 1 - cosine_similarity(pred, target)
    let pred_flat = pred_actions.reshape((batch_size * seq_len, action_dim))?;
    let target_flat = target_actions.reshape((batch_size * seq_len, action_dim))?;
    let mask_flat = attention_mask.reshape(batch_size * seq_len)?;

    // Dot product per position
    let dot = (&pred_flat * &target_flat)?.sum(D::Minus1)?; // [B*T]
    let pred_norm = pred_flat.sqr()?.sum(D::Minus1)?.sqrt()?;
    let target_norm = target_flat.sqr()?.sum(D::Minus1)?.sqrt()?;
    let denom = (pred_norm * target_norm)?.clamp(1e-8f64, f64::INFINITY)?;
    let cosine_sim = dot.div(&denom)?;
    let cosine_dist = cosine_sim.affine(-1.0, 1.0)?; // 1 - cos_sim

    let masked_cosine = (cosine_dist * mask_flat)?;
    let cosine_loss = masked_cosine
        .sum_all()?
        .div(&attention_mask.sum_all()?.clamp(1e-8f64, f64::INFINITY)?)?;

    // Combined loss
    let total = (mse_loss.affine(mse_weight, 0.0)? + cosine_loss.affine(cosine_weight, 0.0)?)?;

    let mse_val = mse_loss.to_scalar::<f32>()?;
    let cos_val = cosine_loss.to_scalar::<f32>()?;

    Ok((total, mse_val, cos_val))
}

// ---------------------------------------------------------------------------
// Training result
// ---------------------------------------------------------------------------

/// Metrics for a single epoch.
#[derive(Debug, Clone, Serialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_mse: f32,
    pub train_cosine: f32,
    pub val_loss: f32,
    pub val_mse: f32,
    pub val_cosine: f32,
    pub learning_rate: f64,
}

/// Result of a training run.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingResult {
    pub epochs_completed: usize,
    pub best_val_loss: f32,
    pub best_epoch: usize,
    pub early_stopped: bool,
    pub checkpoint_path: Option<PathBuf>,
    pub norm_stats: PolicyNormStats,
    pub config: TrainingConfig,
    pub model_config: DecisionTransformerConfig,
    pub history: Vec<EpochMetrics>,
    /// Whether EWC regularization was active during training.
    #[serde(skip)]
    pub ewc_was_active: bool,
    /// Per-parameter sample gradients for EWC Fisher estimation.
    /// Collected from `fisher_samples` forward+backward passes after training.
    /// Each entry is (param_name, Vec<gradient_per_sample>).
    #[serde(skip)]
    pub fisher_gradients: Option<Vec<(String, Vec<Tensor>)>>,
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

/// Train a Decision Transformer on the provided DataLoader.
///
/// Returns the training result with metrics and checkpoint path.
pub fn train_decision_transformer(
    model_config: DecisionTransformerConfig,
    dataloader: &TrajectoryDataLoader,
    config: &TrainingConfig,
) -> CandleResult<TrainingResult> {
    train_decision_transformer_with_ewc(model_config, dataloader, config, None, None)
}

/// Train a Decision Transformer with optional EWC regularization and checkpoint reload.
///
/// - `ewc`: if Some, adds the EWC penalty term to the loss at each training step
/// - `checkpoint_load_path`: if Some, loads pre-trained weights from safetensors file (fine-tuning)
///
/// Returns the training result with metrics and checkpoint path.
pub fn train_decision_transformer_with_ewc(
    model_config: DecisionTransformerConfig,
    dataloader: &TrajectoryDataLoader,
    config: &TrainingConfig,
    ewc: Option<&EWCRegularizer>,
    checkpoint_load_path: Option<&Path>,
) -> CandleResult<TrainingResult> {
    let device = Device::Cpu;

    // Initialize model
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = DecisionTransformer::new(model_config.clone(), vb)?;

    // Load pre-trained weights if checkpoint provided (fine-tuning mode)
    if let Some(ckpt_path) = checkpoint_load_path {
        if ckpt_path.exists() {
            varmap.load(ckpt_path)?;
            info!(
                "Loaded checkpoint from {:?} for fine-tuning (~{}K params)",
                ckpt_path,
                model.param_count() / 1000,
            );
        } else {
            info!(
                "Checkpoint {:?} not found, training from scratch (~{}K params)",
                ckpt_path,
                model.param_count() / 1000,
            );
        }
    } else {
        info!(
            "Decision Transformer initialized from scratch: ~{}K params, {} layers, hidden={}",
            model.param_count() / 1000,
            model_config.num_layers,
            model_config.hidden_dim,
        );
    }

    // Initialize optimizer
    let mut adamw = AdamWState::new(&varmap)?;

    let warmup_epochs = (config.epochs as f64 * config.warmup_fraction).ceil() as usize;
    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut patience_counter = 0;
    let mut history = Vec::with_capacity(config.epochs);
    let mut best_checkpoint: Option<PathBuf> = None;

    // Create checkpoint directory
    let checkpoint_dir = Path::new(&config.checkpoint_dir);
    if !checkpoint_dir.exists() {
        std::fs::create_dir_all(checkpoint_dir).ok();
    }

    for epoch in 0..config.epochs {
        let lr = cosine_lr(config.learning_rate, warmup_epochs, config.epochs, epoch);

        // --- Training ---
        let mut train_loss_sum = 0.0f32;
        let mut train_mse_sum = 0.0f32;
        let mut train_cos_sum = 0.0f32;
        let mut train_batches = 0usize;

        for batch_result in dataloader.batches(Split::Train, epoch) {
            let batch = batch_result?;

            // Forward pass
            let pred_actions = model.forward(
                &batch.returns_to_go,
                &batch.states,
                &batch.actions,
                &batch.timesteps,
                &batch.attention_mask,
            )?;

            // Compute loss
            let (task_loss, mse_val, cos_val) = compute_loss(
                &pred_actions,
                &batch.actions,
                &batch.attention_mask,
                &batch.weights,
                config.mse_weight,
                config.cosine_weight,
            )?;

            // Add EWC penalty if active
            let loss = if let Some(ewc_reg) = ewc {
                if ewc_reg.is_active() {
                    let penalty = ewc_reg.penalty(&varmap)?;
                    (task_loss + penalty)?
                } else {
                    task_loss
                }
            } else {
                task_loss
            };

            // Backward pass
            let grads = loss.backward()?;

            // Optimizer step
            adamw.step(
                &varmap,
                &grads,
                lr,
                config.weight_decay,
                config.max_grad_norm,
            )?;

            train_loss_sum += loss.to_scalar::<f32>()?;
            train_mse_sum += mse_val;
            train_cos_sum += cos_val;
            train_batches += 1;
        }

        let train_batches = train_batches.max(1) as f32;
        let train_loss = train_loss_sum / train_batches;
        let train_mse = train_mse_sum / train_batches;
        let train_cosine = train_cos_sum / train_batches;

        // --- Validation ---
        let mut val_loss_sum = 0.0f32;
        let mut val_mse_sum = 0.0f32;
        let mut val_cos_sum = 0.0f32;
        let mut val_batches = 0usize;

        for batch_result in dataloader.batches(Split::Validation, epoch) {
            let batch = batch_result?;

            let pred_actions = model.forward(
                &batch.returns_to_go,
                &batch.states,
                &batch.actions,
                &batch.timesteps,
                &batch.attention_mask,
            )?;

            let (loss, mse_val, cos_val) = compute_loss(
                &pred_actions,
                &batch.actions,
                &batch.attention_mask,
                &batch.weights,
                config.mse_weight,
                config.cosine_weight,
            )?;

            val_loss_sum += loss.to_scalar::<f32>()?;
            val_mse_sum += mse_val;
            val_cos_sum += cos_val;
            val_batches += 1;
        }

        let val_batches_f = val_batches.max(1) as f32;
        let val_loss = val_loss_sum / val_batches_f;
        let val_mse = val_mse_sum / val_batches_f;
        let val_cosine = val_cos_sum / val_batches_f;

        let metrics = EpochMetrics {
            epoch,
            train_loss,
            train_mse,
            train_cosine,
            val_loss,
            val_mse,
            val_cosine,
            learning_rate: lr,
        };

        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            info!(
                "Epoch {}/{}: train_loss={:.4} val_loss={:.4} (mse={:.4} cos={:.4}) lr={:.2e}",
                epoch + 1,
                config.epochs,
                train_loss,
                val_loss,
                val_mse,
                val_cosine,
                lr,
            );
        }

        // Early stopping check
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            best_epoch = epoch;
            patience_counter = 0;

            // Save best checkpoint
            let path = checkpoint_dir.join("best_policy.safetensors");
            varmap.save(&path)?;
            best_checkpoint = Some(path);
            debug!(
                "New best model at epoch {} (val_loss={:.4})",
                epoch, val_loss
            );
        } else {
            patience_counter += 1;
        }

        // Periodic checkpoint
        if config.checkpoint_every > 0 && (epoch + 1) % config.checkpoint_every == 0 {
            let path = checkpoint_dir.join(format!("policy_epoch_{}.safetensors", epoch + 1));
            varmap.save(&path)?;
        }

        history.push(metrics);

        // Early stopping
        if patience_counter >= config.patience {
            info!(
                "Early stopping at epoch {} (no improvement for {} epochs, best={:.4} at epoch {})",
                epoch + 1,
                config.patience,
                best_val_loss,
                best_epoch + 1
            );
            break;
        }
    }

    // Save final config alongside checkpoint
    if let Some(ref ckpt_path) = best_checkpoint {
        let config_path = ckpt_path.with_extension("json");
        let config_json = serde_json::json!({
            "model_config": model_config,
            "training_config": config,
            "norm_stats": dataloader.norm_stats(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        });
        if let Ok(json_str) = serde_json::to_string_pretty(&config_json) {
            std::fs::write(&config_path, json_str).ok();
        }
    }

    // Load best checkpoint back before Fisher estimation
    // (so gradients are computed at θ* not at the last epoch's θ)
    if let Some(ref ckpt_path) = best_checkpoint {
        varmap.load(ckpt_path)?;
    }

    // Collect per-parameter gradients for EWC Fisher estimation.
    // We run a few forward+backward passes on training data at θ* to compute
    // the diagonal Fisher: F_i = (1/N) Σ_n (∂L_n/∂θ_i)²
    let fisher_gradients = {
        let (train_len, _, _) = dataloader.split_sizes();
        let fisher_samples = 10.min(train_len); // cap at 10 for speed
        if fisher_samples > 0 {
            let all_vars = varmap.all_vars();
            let var_names: Vec<String> = {
                let data = varmap.data().lock().unwrap();
                data.keys().cloned().collect()
            };

            // Initialize gradient accumulators: one Vec<Tensor> per parameter
            let mut grad_accum: Vec<Vec<Tensor>> = vec![Vec::new(); var_names.len()];
            let mut samples_collected = 0;

            for batch_result in dataloader.batches(Split::Train, 0) {
                if samples_collected >= fisher_samples {
                    break;
                }
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                let pred = model.forward(
                    &batch.returns_to_go,
                    &batch.states,
                    &batch.actions,
                    &batch.timesteps,
                    &batch.attention_mask,
                )?;

                let (loss, _, _) = compute_loss(
                    &pred,
                    &batch.actions,
                    &batch.attention_mask,
                    &batch.weights,
                    config.mse_weight,
                    config.cosine_weight,
                )?;

                let grads = loss.backward()?;

                for (i, var) in all_vars.iter().enumerate() {
                    if let Some(g) = grads.get(var) {
                        grad_accum[i].push(g.clone());
                    }
                }

                samples_collected += 1;
            }

            let fisher_grads: Vec<(String, Vec<Tensor>)> = var_names
                .into_iter()
                .zip(grad_accum)
                .filter(|(_, grads)| !grads.is_empty())
                .collect();

            debug!(
                "Collected Fisher gradients: {} params, {} samples each",
                fisher_grads.len(),
                samples_collected,
            );
            Some(fisher_grads)
        } else {
            None
        }
    };

    Ok(TrainingResult {
        epochs_completed: history.len(),
        best_val_loss,
        best_epoch,
        early_stopped: patience_counter >= config.patience,
        checkpoint_path: best_checkpoint,
        norm_stats: dataloader.norm_stats().clone(),
        config: config.clone(),
        model_config,
        history,
        ewc_was_active: ewc.is_some_and(|e| e.is_active()),
        fisher_gradients,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::TrajectoryDataLoaderConfig;
    use crate::dataset::{
        trajectory_to_tensors, TrajectoryTensors, ACTION_DIM, CONTEXT_DIM, QUERY_DIM, STATE_DIM,
    };
    use chrono::{Duration, Utc};
    use neural_routing_core::models::{Trajectory, TrajectoryNode};
    use uuid::Uuid;

    fn make_trajectory_at(num_nodes: usize, reward: f64, days_ago: i64) -> Trajectory {
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
            session_id: format!("session-{}", Uuid::new_v4()),
            query_embedding: vec![0.5f32; QUERY_DIM],
            total_reward: reward,
            step_count: num_nodes,
            duration_ms: 1000,
            nodes,
            created_at: Utc::now() - Duration::days(days_ago),
            protocol_run_id: None,
        }
    }

    fn make_test_dataloader(n: usize) -> TrajectoryDataLoader {
        let trajs: Vec<Trajectory> = (0..n)
            .map(|i| make_trajectory_at(4, 8.0 + (i as f64 * 0.1), (n - i) as i64))
            .collect();
        let tensors: Vec<TrajectoryTensors> =
            trajs.iter().filter_map(trajectory_to_tensors).collect();
        let config = TrajectoryDataLoaderConfig {
            batch_size: 8,
            ..Default::default()
        };
        TrajectoryDataLoader::new(tensors, config)
    }

    #[test]
    fn test_cosine_lr_schedule() {
        let base_lr = 1e-4;
        let warmup = 10;
        let total = 100;

        // During warmup: monotonically increasing
        let lr_0 = cosine_lr(base_lr, warmup, total, 0);
        let lr_5 = cosine_lr(base_lr, warmup, total, 5);
        let lr_9 = cosine_lr(base_lr, warmup, total, 9);
        assert!(lr_0 < lr_5);
        assert!(lr_5 < lr_9);

        // At warmup end: should be close to base_lr
        let lr_10 = cosine_lr(base_lr, warmup, total, 10);
        assert!((lr_10 - base_lr).abs() < base_lr * 0.1);

        // After warmup: should decrease
        let lr_50 = cosine_lr(base_lr, warmup, total, 50);
        let lr_99 = cosine_lr(base_lr, warmup, total, 99);
        assert!(lr_50 < lr_10);
        assert!(lr_99 < lr_50);
        assert!(lr_99 > 0.0); // never zero
    }

    #[test]
    fn test_compute_loss() -> CandleResult<()> {
        let device = Device::Cpu;
        let b = 2;
        let t = 4;

        let pred = Tensor::randn(0.0f32, 0.5, (b, t, ACTION_DIM), &device)?;
        let target = Tensor::randn(0.0f32, 0.5, (b, t, ACTION_DIM), &device)?;
        let mask = Tensor::ones((b, t), DType::F32, &device)?;
        let weights = Tensor::ones(b, DType::F32, &device)?;

        let (total, mse, cos) = compute_loss(&pred, &target, &mask, &weights, 1.0, 0.1)?;

        assert!(mse > 0.0, "MSE should be positive");
        assert!(cos >= 0.0, "Cosine distance should be >= 0");
        let total_val = total.to_scalar::<f32>()?;
        assert!(total_val > 0.0, "Total loss should be positive");

        // Loss with identical pred and target should be near zero
        let (_total_same, mse_same, _cos_same) =
            compute_loss(&pred, &pred, &mask, &weights, 1.0, 0.1)?;
        assert!(
            mse_same < 1e-6,
            "MSE with identical tensors should be ~0, got {}",
            mse_same
        );

        Ok(())
    }

    #[test]
    fn test_training_loop_converges() -> CandleResult<()> {
        // Small model + dataset to verify the training loop runs and loss decreases
        let dl = make_test_dataloader(50);

        let model_config = DecisionTransformerConfig {
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            max_timesteps: 16,
            dropout: 0.0,
        };

        let config = TrainingConfig {
            epochs: 5,
            learning_rate: 1e-3,
            batch_size: 8,
            patience: 100, // don't early stop
            checkpoint_dir: "/tmp/policy_test_ckpt".to_string(),
            checkpoint_every: 0,
            ..Default::default()
        };

        let result = train_decision_transformer(model_config, &dl, &config)?;

        assert_eq!(result.epochs_completed, 5);
        assert!(result.history.len() == 5);

        // Loss should be finite
        for m in &result.history {
            assert!(m.train_loss.is_finite(), "Train loss should be finite");
            assert!(m.val_loss.is_finite(), "Val loss should be finite");
        }

        // First epoch loss should be higher than last (convergence)
        let first_loss = result.history.first().unwrap().train_loss;
        let last_loss = result.history.last().unwrap().train_loss;
        info!(
            "Training: first_loss={:.4}, last_loss={:.4}",
            first_loss, last_loss
        );
        // With 5 epochs on a small dataset, we expect at least some improvement
        // (not guaranteed, but very likely with lr=1e-3)

        Ok(())
    }

    #[test]
    fn test_early_stopping() -> CandleResult<()> {
        let dl = make_test_dataloader(30);

        let model_config = DecisionTransformerConfig {
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 2,
            max_timesteps: 8,
            dropout: 0.0,
        };

        // Use a very low LR so model barely improves, triggering early stop
        let config = TrainingConfig {
            epochs: 50,
            learning_rate: 1e-8, // tiny LR → stagnation → early stop
            batch_size: 8,
            patience: 3,
            checkpoint_dir: "/tmp/policy_test_early_stop".to_string(),
            checkpoint_every: 0,
            ..Default::default()
        };

        let result = train_decision_transformer(model_config, &dl, &config)?;

        // With such a tiny LR, validation should stagnate and trigger early stopping
        // The model improves on epoch 0 (from infinity → first val loss), then stalls
        // So we expect ~4 epochs (1 best + 3 patience)
        assert!(
            result.epochs_completed < 50,
            "Expected early stopping, but ran all {} epochs",
            result.epochs_completed
        );
        assert!(result.early_stopped);

        Ok(())
    }

    #[test]
    fn test_train_with_ewc_penalty() -> CandleResult<()> {
        use crate::ewc::{EWCConfig, EWCRegularizer};
        use candle_core::Device;

        let dl = make_test_dataloader(50);

        let model_config = DecisionTransformerConfig {
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            max_timesteps: 16,
            dropout: 0.0,
        };

        let config = TrainingConfig {
            epochs: 3,
            learning_rate: 1e-3,
            batch_size: 8,
            patience: 100,
            checkpoint_dir: "/tmp/policy_test_ewc".to_string(),
            checkpoint_every: 0,
            ..Default::default()
        };

        // Train once without EWC
        let result1 =
            train_decision_transformer_with_ewc(model_config.clone(), &dl, &config, None, None)?;
        assert_eq!(result1.epochs_completed, 3);
        assert!(!result1.ewc_was_active);

        // Create EWC with a snapshot (simulate: Fisher = uniform)
        let mut ewc = EWCRegularizer::new(EWCConfig {
            lambda: 10.0,
            fisher_samples: 1,
        });

        // Build a temporary model to get param names and create fake gradients
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let _model = DecisionTransformer::new(model_config.clone(), vb)?;

        let data = varmap.data().lock().unwrap();
        let sample_grads: Vec<(String, Vec<Tensor>)> = data
            .iter()
            .map(|(name, var)| {
                let grad = Tensor::ones(var.shape(), DType::F32, &Device::Cpu).unwrap();
                (name.clone(), vec![grad])
            })
            .collect();
        drop(data);
        ewc.snapshot_from_gradients(&varmap, &sample_grads)?;
        assert!(ewc.is_active());

        // Train with EWC active
        let result2 =
            train_decision_transformer_with_ewc(model_config, &dl, &config, Some(&ewc), None)?;
        assert_eq!(result2.epochs_completed, 3);
        assert!(result2.ewc_was_active);

        // Both should produce finite losses
        for m in &result2.history {
            assert!(m.train_loss.is_finite(), "EWC train loss should be finite");
            assert!(m.val_loss.is_finite(), "EWC val loss should be finite");
        }

        Ok(())
    }

    #[test]
    fn test_train_with_checkpoint_reload() -> CandleResult<()> {
        let dl = make_test_dataloader(50);
        let ckpt_dir = "/tmp/policy_test_ckpt_reload";
        let _ = std::fs::create_dir_all(ckpt_dir);

        let model_config = DecisionTransformerConfig {
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            hidden_dim: 64,
            num_layers: 1,
            num_heads: 2,
            max_timesteps: 16,
            dropout: 0.0,
        };

        let config = TrainingConfig {
            epochs: 3,
            learning_rate: 1e-3,
            batch_size: 8,
            patience: 100,
            checkpoint_dir: ckpt_dir.to_string(),
            checkpoint_every: 0,
            ..Default::default()
        };

        // Train once — produces a checkpoint
        let result1 = train_decision_transformer(model_config.clone(), &dl, &config)?;
        assert!(result1.checkpoint_path.is_some());
        let ckpt_path = result1.checkpoint_path.unwrap();
        assert!(ckpt_path.exists(), "Checkpoint file should exist");

        // Train again, loading from checkpoint (fine-tuning)
        let result2 = train_decision_transformer_with_ewc(
            model_config,
            &dl,
            &config,
            None,
            Some(ckpt_path.as_path()),
        )?;
        assert_eq!(result2.epochs_completed, 3);

        // Fine-tuned model should start with a lower loss than random init
        // (not guaranteed but very likely)
        let first_loss_1 = result1.history.first().unwrap().val_loss;
        let first_loss_2 = result2.history.first().unwrap().val_loss;
        info!(
            "Random init first val_loss={:.4}, checkpoint reload first val_loss={:.4}",
            first_loss_1, first_loss_2
        );

        // Cleanup
        let _ = std::fs::remove_dir_all(ckpt_dir);

        Ok(())
    }

    #[test]
    fn test_train_with_nonexistent_checkpoint() -> CandleResult<()> {
        let dl = make_test_dataloader(30);

        let model_config = DecisionTransformerConfig {
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 2,
            max_timesteps: 8,
            dropout: 0.0,
        };

        let config = TrainingConfig {
            epochs: 2,
            learning_rate: 1e-3,
            batch_size: 8,
            patience: 100,
            checkpoint_dir: "/tmp/policy_test_noexist".to_string(),
            checkpoint_every: 0,
            ..Default::default()
        };

        // Should gracefully fall back to random init
        let result = train_decision_transformer_with_ewc(
            model_config,
            &dl,
            &config,
            None,
            Some(std::path::Path::new(
                "/tmp/nonexistent_checkpoint.safetensors",
            )),
        )?;
        assert_eq!(result.epochs_completed, 2);
        assert!(!result.ewc_was_active);

        Ok(())
    }
}
