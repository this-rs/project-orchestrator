//! EWC (Elastic Weight Consolidation) — continual learning.
//!
//! Prevents catastrophic forgetting when re-training on new trajectories
//! by penalizing changes to parameters important for previous tasks.
//!
//! Algorithm (Kirkpatrick et al., 2017):
//! 1. After training on task A, compute diagonal Fisher Information Matrix F_i
//!    for each parameter θ_i (approximation: squared gradient of the loss).
//! 2. Store the optimal parameters θ*_i.
//! 3. When training on task B, add penalty: λ/2 · Σ_i F_i · (θ_i - θ*_i)²
//!
//! This penalizes changes to parameters that were important for task A.

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

/// Configuration for EWC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCConfig {
    /// Lambda — importance weighting for the EWC penalty.
    pub lambda: f64,
    /// Number of samples used to estimate Fisher information.
    pub fisher_samples: usize,
}

impl Default for EWCConfig {
    fn default() -> Self {
        Self {
            lambda: 5000.0,
            fisher_samples: 200,
        }
    }
}

/// A snapshot of parameters and their Fisher importance scores.
#[derive(Clone)]
struct ParamSnapshot {
    /// Parameter name (from VarMap).
    name: String,
    /// Optimal parameter values θ* (detached copy).
    optimal_values: Tensor,
    /// Diagonal Fisher information F_i (importance per parameter).
    fisher_diag: Tensor,
}

/// EWC regularizer — prevents catastrophic forgetting.
///
/// Usage:
/// 1. After training on old data: call `snapshot(varmap, loss_fn)` to capture θ* and F
/// 2. During new training: call `penalty(varmap)` to get the EWC loss term
/// 3. Add to total loss: `L_total = L_new + ewc.penalty(varmap)?`
pub struct EWCRegularizer {
    pub config: EWCConfig,
    /// Stored parameter snapshots with Fisher information.
    snapshots: Vec<ParamSnapshot>,
}

impl EWCRegularizer {
    pub fn new(config: EWCConfig) -> Self {
        Self {
            config,
            snapshots: Vec::new(),
        }
    }

    /// Whether a snapshot has been taken (EWC is active).
    pub fn is_active(&self) -> bool {
        !self.snapshots.is_empty()
    }

    /// Snapshot current parameters and compute diagonal Fisher Information.
    ///
    /// `grad_samples` is a list of per-sample gradient tensors for each parameter,
    /// computed by running forward+backward on `fisher_samples` data points.
    /// Each entry is (param_name, gradient_tensor) for one sample.
    ///
    /// For efficiency, we use the empirical Fisher approximation:
    /// F_i = (1/N) Σ_n (∂L_n/∂θ_i)²
    pub fn snapshot_from_gradients(
        &mut self,
        varmap: &VarMap,
        sample_gradients: &[(String, Vec<Tensor>)],
    ) -> CandleResult<()> {
        self.snapshots.clear();

        let all_vars = varmap.all_vars();
        let data = varmap.data().lock().unwrap();

        for (name, var) in data.iter() {
            // Find matching gradients for this parameter
            let grads_for_param: Option<&Vec<Tensor>> = sample_gradients
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, g)| g);

            let fisher_diag = if let Some(grads) = grads_for_param {
                if grads.is_empty() {
                    // No gradients → zero importance
                    Tensor::zeros_like(var)?
                } else {
                    // F_i = mean(grad²) across samples
                    let mut sum = grads[0].sqr()?;
                    for g in &grads[1..] {
                        sum = sum.add(&g.sqr()?)?;
                    }
                    sum.div(
                        &Tensor::new(grads.len() as f32, sum.device())?
                            .broadcast_as(sum.shape())?,
                    )?
                }
            } else {
                // Parameter not in gradient samples → zero importance
                Tensor::zeros_like(var)?
            };

            // Deep-copy current values as optimal θ* (independent storage from var)
            let optimal_values = var.copy()?;

            self.snapshots.push(ParamSnapshot {
                name: name.clone(),
                optimal_values,
                fisher_diag,
            });
        }

        let _ = all_vars; // keep borrow checker happy
        Ok(())
    }

    /// Compute the EWC penalty: λ/2 · Σ_i F_i · (θ_i - θ*_i)²
    ///
    /// Returns a scalar loss tensor that should be added to the main loss.
    pub fn penalty(&self, varmap: &VarMap) -> CandleResult<Tensor> {
        if self.snapshots.is_empty() {
            // No snapshot → zero penalty
            return Tensor::new(0.0f32, &candle_core::Device::Cpu);
        }

        let data = varmap.data().lock().unwrap();
        let lambda = self.config.lambda;

        let mut total_penalty: Option<Tensor> = None;

        for snapshot in &self.snapshots {
            if let Some(current_var) = data.get(&snapshot.name) {
                // (θ - θ*)²
                let diff = current_var.sub(&snapshot.optimal_values)?;
                let diff_sq = diff.sqr()?;

                // F_i · (θ - θ*)²
                let weighted = snapshot.fisher_diag.mul(&diff_sq)?;

                // Sum all elements → scalar
                let param_penalty = weighted.sum_all()?;

                total_penalty = Some(match total_penalty {
                    Some(p) => p.add(&param_penalty)?,
                    None => param_penalty,
                });
            }
        }

        match total_penalty {
            Some(p) => {
                // λ/2 · Σ
                let scale = Tensor::new((lambda / 2.0) as f32, p.device())?;
                p.mul(&scale)
            }
            None => Tensor::new(0.0f32, &candle_core::Device::Cpu),
        }
    }

    /// Number of parameters tracked.
    pub fn param_count(&self) -> usize {
        self.snapshots.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn test_ewc_default_config() {
        let config = EWCConfig::default();
        assert_eq!(config.lambda, 5000.0);
        assert_eq!(config.fisher_samples, 200);
    }

    #[test]
    fn test_ewc_not_active_initially() {
        let ewc = EWCRegularizer::new(EWCConfig::default());
        assert!(!ewc.is_active());
        assert_eq!(ewc.param_count(), 0);
    }

    #[test]
    fn test_ewc_penalty_zero_without_snapshot() {
        let varmap = VarMap::new();
        let ewc = EWCRegularizer::new(EWCConfig::default());
        let penalty = ewc.penalty(&varmap).unwrap();
        assert_eq!(penalty.to_scalar::<f32>().unwrap(), 0.0);
    }

    #[test]
    fn test_ewc_snapshot_and_penalty() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create a simple parameter
        let _w = vb
            .get_with_hints((3, 2), "weight", candle_nn::Init::Const(1.0))
            .unwrap();

        // Snapshot with uniform Fisher (all params equally important)
        let mut ewc = EWCRegularizer::new(EWCConfig {
            lambda: 1.0,
            fisher_samples: 1,
        });

        // Create fake gradients (all ones)
        let grad = Tensor::ones((3, 2), DType::F32, &device).unwrap();
        let sample_grads = vec![("weight".to_string(), vec![grad])];
        ewc.snapshot_from_gradients(&varmap, &sample_grads).unwrap();

        assert!(ewc.is_active());
        assert_eq!(ewc.param_count(), 1);

        // Penalty should be zero (params haven't changed)
        let penalty = ewc.penalty(&varmap).unwrap();
        let p = penalty.to_scalar::<f32>().unwrap();
        assert!(
            p.abs() < 1e-6,
            "Penalty should be ~0 when params unchanged, got {}",
            p
        );
    }

    #[test]
    fn test_ewc_penalty_increases_with_param_change() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let _w = vb
            .get_with_hints((4,), "param", candle_nn::Init::Const(0.0))
            .unwrap();

        let mut ewc = EWCRegularizer::new(EWCConfig {
            lambda: 2.0,
            fisher_samples: 1,
        });

        // Fisher = all ones
        let grad = Tensor::ones((4,), DType::F32, &device).unwrap();
        ewc.snapshot_from_gradients(&varmap, &vec![("param".to_string(), vec![grad])])
            .unwrap();

        // Now mutate the parameter via the Var stored in VarMap
        let new_val = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &device).unwrap();
        let data = varmap.data().lock().unwrap();
        let var = data.get("param").expect("param should exist in varmap");
        var.set(&new_val).unwrap();
        drop(data);

        // Penalty = λ/2 · Σ F_i · (θ - θ*)² = 2/2 · 4 · 1² = 4.0
        let penalty = ewc.penalty(&varmap).unwrap();
        let p = penalty.to_scalar::<f32>().unwrap();
        assert!((p - 4.0).abs() < 0.01, "Expected penalty ≈ 4.0, got {}", p);
    }
}
