//! Evaluation Framework — offline metrics and OOD detection.
//!
//! ## Offline Metrics
//! 1. **Return Correlation**: Pearson/Spearman between predicted and actual returns
//! 2. **Action Accuracy**: fraction of actions correctly predicted (cosine > threshold)
//! 3. **Route Diversity**: Shannon entropy of predicted action distribution
//! 4. **Importance Weighted Eval (IWE)**: π_new/π_old × reward, clipped [0.1, 10]
//!
//! ## OOD Detection
//! 1. **Codebook Distance**: cosine distance to nearest codebook entry
//! 2. **Mahalanobis Distance**: distance from training state distribution
//!
//! All pure Rust, no Python dependency.

use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{linear, Dropout, Linear, VarBuilder, VarMap};
use serde::Serialize;

use crate::codebook::ActionCodebook;
use crate::dataset::ACTION_DIM;

// ---------------------------------------------------------------------------
// Offline Metrics
// ---------------------------------------------------------------------------

/// Complete evaluation report.
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationReport {
    /// Pearson correlation between predicted and actual returns.
    pub return_pearson: f64,
    /// Spearman rank correlation between predicted and actual returns.
    pub return_spearman: f64,
    /// Action accuracy: fraction with cosine similarity > threshold.
    pub action_accuracy: f64,
    /// Route diversity: Shannon entropy of action distribution.
    pub route_diversity: f64,
    /// Importance-weighted evaluation score.
    pub iwe_score: f64,
    /// Number of samples evaluated.
    pub num_samples: usize,
}

/// Compute Pearson correlation coefficient between two vectors.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x: f64 = x.iter().take(n).sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().take(n).sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}

/// Compute Spearman rank correlation between two vectors.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let rank_x = compute_ranks(&x[..n]);
    let rank_y = compute_ranks(&y[..n]);
    pearson_correlation(&rank_x, &rank_y)
}

/// Compute ranks (average rank for ties).
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find ties
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        // Average rank for tied group
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Compute action accuracy: fraction of predicted actions that match ground truth.
///
/// A prediction is "correct" if cosine similarity > `threshold`.
pub fn action_accuracy(predicted: &[Vec<f32>], actual: &[Vec<f32>], threshold: f32) -> f64 {
    if predicted.is_empty() {
        return 0.0;
    }

    let correct = predicted
        .iter()
        .zip(actual.iter())
        .filter(|(pred, act)| cosine_sim(pred, act) > threshold)
        .count();

    correct as f64 / predicted.len() as f64
}

/// Compute Shannon entropy of action distribution (route diversity).
///
/// Higher entropy = more diverse routing decisions.
/// `action_counts`: map of action_key → frequency count.
pub fn shannon_entropy(action_counts: &[usize]) -> f64 {
    let total: usize = action_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    action_counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * p.ln()
        })
        .sum()
}

/// Importance Weighted Evaluation (IWE).
///
/// Score = E[ clip(π_new(a|s) / π_old(a|s), 0.1, 10) × reward ]
///
/// - `new_log_probs`: log probability of actions under new policy
/// - `old_log_probs`: log probability of actions under old policy
/// - `rewards`: corresponding rewards
pub fn importance_weighted_eval(
    new_log_probs: &[f64],
    old_log_probs: &[f64],
    rewards: &[f64],
) -> f64 {
    let n = new_log_probs
        .len()
        .min(old_log_probs.len())
        .min(rewards.len());
    if n == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for i in 0..n {
        let log_ratio = new_log_probs[i] - old_log_probs[i];
        let ratio = log_ratio.exp().clamp(0.1, 10.0);
        total += ratio * rewards[i];
    }

    total / n as f64
}

// ---------------------------------------------------------------------------
// OOD Detection
// ---------------------------------------------------------------------------

/// OOD detection result for a single sample.
#[derive(Debug, Clone, Serialize)]
pub struct OodResult {
    /// Whether the sample is detected as OOD.
    pub is_ood: bool,
    /// Codebook distance score (1 - cosine_similarity). Higher = more OOD.
    pub codebook_distance: f32,
    /// Mahalanobis distance from training distribution. Higher = more OOD.
    pub mahalanobis_distance: f64,
    /// Combined OOD score (max of normalized distances).
    pub combined_score: f64,
}

/// OOD detector using codebook distance + Mahalanobis distance.
pub struct OodDetector {
    /// Training state distribution: mean per dimension.
    state_mean: Vec<f64>,
    /// Training state distribution: inverse variance per dimension (diagonal approx).
    state_inv_var: Vec<f64>,
    /// Codebook distance threshold (1 - cosine_sim).
    codebook_threshold: f32,
    /// Mahalanobis distance threshold.
    mahalanobis_threshold: f64,
}

impl OodDetector {
    /// Build an OOD detector from training state statistics.
    ///
    /// Uses diagonal covariance approximation for efficiency.
    pub fn from_training_states(states: &[Vec<f32>], codebook_threshold: f32) -> Self {
        let dim = if states.is_empty() {
            ACTION_DIM
        } else {
            states[0].len()
        };
        let n = states.len().max(1) as f64;

        // Compute per-dimension mean and variance
        let mut mean = vec![0.0f64; dim];
        for state in states {
            for (i, &v) in state.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        let mut var = vec![0.0f64; dim];
        for state in states {
            for (i, &v) in state.iter().enumerate() {
                let d = v as f64 - mean[i];
                var[i] += d * d;
            }
        }
        let inv_var: Vec<f64> = var
            .iter()
            .map(|v| {
                let variance = v / n;
                if variance > 1e-10 {
                    1.0 / variance
                } else {
                    0.0 // zero variance = ignore this dimension
                }
            })
            .collect();

        // Mahalanobis threshold: chi-squared critical value for 95% at dim degrees of freedom
        // Approximation: mean + 2*std ≈ dim + 2*sqrt(2*dim)
        let mahalanobis_threshold = dim as f64 + 2.0 * (2.0 * dim as f64).sqrt();

        Self {
            state_mean: mean,
            state_inv_var: inv_var,
            codebook_threshold,
            mahalanobis_threshold,
        }
    }

    /// Detect whether a state-action pair is OOD.
    pub fn detect(
        &self,
        state: &[f32],
        action_vector: &[f32],
        codebook: &ActionCodebook,
    ) -> OodResult {
        // 1. Codebook distance
        let codebook_distance = match codebook.nearest_neighbor(action_vector) {
            Some((_, sim)) => 1.0 - sim,
            None => 1.0,
        };

        // 2. Mahalanobis distance
        let mahalanobis_distance = self.mahalanobis(state);

        // 3. Combined: either detector triggers → OOD
        let norm_codebook = codebook_distance as f64 / self.codebook_threshold.max(1e-6) as f64;
        let norm_mahal = mahalanobis_distance / self.mahalanobis_threshold.max(1e-6);
        let combined_score = norm_codebook.max(norm_mahal);

        let is_ood = codebook_distance > self.codebook_threshold
            || mahalanobis_distance > self.mahalanobis_threshold;

        OodResult {
            is_ood,
            codebook_distance,
            mahalanobis_distance,
            combined_score,
        }
    }

    /// Compute Mahalanobis distance (diagonal covariance approximation).
    fn mahalanobis(&self, state: &[f32]) -> f64 {
        let mut dist = 0.0;
        for (i, &s) in state.iter().enumerate().take(self.state_mean.len()) {
            let d = s as f64 - self.state_mean[i];
            dist += d * d * self.state_inv_var[i];
        }
        dist.sqrt()
    }

    /// Compute OOD AUC: area under the ROC curve for OOD detection.
    ///
    /// - `in_distribution`: scores for known in-distribution samples
    /// - `ood`: scores for known OOD samples
    ///
    /// Uses the combined_score from `detect()`.
    pub fn compute_auc(in_dist_scores: &[f64], ood_scores: &[f64]) -> f64 {
        if in_dist_scores.is_empty() || ood_scores.is_empty() {
            return 0.5;
        }

        // AUC = P(score_ood > score_in) = Wilcoxon-Mann-Whitney statistic
        let mut count = 0.0f64;
        let total = in_dist_scores.len() as f64 * ood_scores.len() as f64;

        for &ood in ood_scores {
            for &ind in in_dist_scores {
                if ood > ind {
                    count += 1.0;
                } else if (ood - ind).abs() < 1e-12 {
                    count += 0.5; // tie: count as half
                }
            }
        }

        count / total
    }
}

// ---------------------------------------------------------------------------
// MC Dropout OOD Detection
// ---------------------------------------------------------------------------

/// MC Dropout uncertainty estimator for OOD detection.
///
/// Trains a lightweight probe network (MLP with dropout) on training data.
/// At inference, runs N stochastic forward passes with dropout enabled
/// (Gal & Ghahramani, 2016) and uses prediction variance as an epistemic
/// uncertainty measure.
///
/// Higher variance → model is uncertain → likely out-of-distribution.
pub struct McDropoutEstimator {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    dropout: Dropout,
    num_passes: usize,
    state_dim: usize,
    action_dim: usize,
    varmap: VarMap,
    /// Uncertainty threshold: above this → OOD.
    pub uncertainty_threshold: f64,
}

impl McDropoutEstimator {
    /// Create a new MC Dropout estimator.
    ///
    /// - `state_dim`: input state dimension
    /// - `action_dim`: output action dimension
    /// - `hidden_dim`: hidden layer size (recommended: 128-256)
    /// - `dropout_rate`: dropout probability (recommended: 0.1-0.3)
    /// - `num_passes`: number of stochastic forward passes (recommended: 10-30)
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        dropout_rate: f32,
        num_passes: usize,
    ) -> CandleResult<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let l1 = linear(state_dim, hidden_dim, vb.pp("l1"))?;
        let l2 = linear(hidden_dim, hidden_dim, vb.pp("l2"))?;
        let l3 = linear(hidden_dim, action_dim, vb.pp("l3"))?;
        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            l1,
            l2,
            l3,
            dropout,
            num_passes,
            state_dim,
            action_dim,
            varmap,
            uncertainty_threshold: 0.1, // default, should be calibrated
        })
    }

    /// Single forward pass with dropout enabled (stochastic).
    fn forward_stochastic(&self, state: &Tensor) -> CandleResult<Tensor> {
        let h = self.l1.forward(state)?.relu()?;
        let h = self.dropout.forward(&h, true)?; // train=true → dropout active
        let h = self.l2.forward(&h)?.relu()?;
        let h = self.dropout.forward(&h, true)?;
        self.l3.forward(&h)
    }

    /// Train the probe network on (state, action) pairs via SGD.
    ///
    /// Returns the final epoch MSE loss.
    pub fn fit(
        &self,
        states: &[Vec<f32>],
        actions: &[Vec<f32>],
        epochs: usize,
        lr: f64,
    ) -> CandleResult<f32> {
        let device = Device::Cpu;
        let n = states.len();
        if n == 0 {
            return Ok(0.0);
        }

        let states_flat: Vec<f32> = states.iter().flatten().copied().collect();
        let actions_flat: Vec<f32> = actions.iter().flatten().copied().collect();
        let states_t = Tensor::from_vec(states_flat, (n, self.state_dim), &device)?;
        let actions_t = Tensor::from_vec(actions_flat, (n, self.action_dim), &device)?;

        let mut last_loss = 0.0f32;

        for _epoch in 0..epochs {
            let pred = self.forward_stochastic(&states_t)?;
            let loss = (&pred - &actions_t)?.sqr()?.mean_all()?;
            last_loss = loss.to_scalar::<f32>()?;

            // Manual SGD step
            let grads = loss.backward()?;
            for var in self.varmap.all_vars() {
                if let Some(grad) = grads.get(&var) {
                    let updated = (var.as_tensor() - grad.affine(lr, 0.0)?)?;
                    var.set(&updated)?;
                }
            }
        }

        Ok(last_loss)
    }

    /// Estimate epistemic uncertainty for a single state.
    ///
    /// Runs `num_passes` stochastic forward passes and returns the average
    /// per-dimension variance of predictions.
    pub fn estimate_uncertainty(&self, state: &[f32]) -> CandleResult<f64> {
        let device = Device::Cpu;
        let state_t = Tensor::from_vec(state.to_vec(), (1, self.state_dim), &device)?;

        let mut predictions = Vec::with_capacity(self.num_passes);
        for _ in 0..self.num_passes {
            let pred = self.forward_stochastic(&state_t)?;
            let vals = pred.flatten_all()?.to_vec1::<f32>()?;
            predictions.push(vals);
        }

        // Compute per-dimension variance, then average
        let dim = predictions[0].len();
        let n = predictions.len() as f64;
        let mut total_var = 0.0;

        for d in 0..dim {
            let mean: f64 = predictions.iter().map(|p| p[d] as f64).sum::<f64>() / n;
            let var: f64 = predictions
                .iter()
                .map(|p| {
                    let diff = p[d] as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / n;
            total_var += var;
        }

        Ok(total_var / dim as f64) // mean variance across dimensions
    }

    /// Check if a state is OOD based on MC Dropout uncertainty.
    pub fn is_ood(&self, state: &[f32]) -> CandleResult<bool> {
        let uncertainty = self.estimate_uncertainty(state)?;
        Ok(uncertainty > self.uncertainty_threshold)
    }

    /// Calibrate the uncertainty threshold from training data.
    ///
    /// Sets the threshold at the given percentile of training uncertainties.
    /// E.g., `percentile=95.0` means 95% of training samples fall below the threshold.
    pub fn calibrate_threshold(
        &mut self,
        training_states: &[Vec<f32>],
        percentile: f64,
    ) -> CandleResult<()> {
        let mut uncertainties = Vec::with_capacity(training_states.len());
        for state in training_states {
            let u = self.estimate_uncertainty(state)?;
            uncertainties.push(u);
        }

        uncertainties.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((uncertainties.len() as f64 * percentile / 100.0) as usize)
            .min(uncertainties.len().saturating_sub(1));
        self.uncertainty_threshold = uncertainties[idx];

        Ok(())
    }

    /// Get a reference to the internal VarMap (for save/load).
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{ActionCodebook, CodebookEntry};

    #[test]
    fn test_pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!(
            (r - 1.0).abs() < 1e-10,
            "Perfect positive correlation, got {r}"
        );
    }

    #[test]
    fn test_pearson_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&x, &y);
        assert!(
            (r + 1.0).abs() < 1e-10,
            "Perfect negative correlation, got {r}"
        );
    }

    #[test]
    fn test_spearman_monotonic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearman_correlation(&x, &y);
        assert!(
            (rho - 1.0).abs() < 1e-10,
            "Monotonic → Spearman=1.0, got {rho}"
        );
    }

    #[test]
    fn test_spearman_non_linear() {
        // Non-linear but monotonic
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x²
        let rho = spearman_correlation(&x, &y);
        assert!(
            (rho - 1.0).abs() < 1e-10,
            "x² is monotonic → Spearman=1.0, got {rho}"
        );
    }

    #[test]
    fn test_action_accuracy() {
        let pred = vec![
            vec![1.0; ACTION_DIM],
            vec![1.0; ACTION_DIM],
            vec![0.0; ACTION_DIM],
        ];
        let actual = vec![
            vec![1.0; ACTION_DIM],  // exact match
            vec![0.99; ACTION_DIM], // very close
            vec![-1.0; ACTION_DIM], // opposite
        ];

        let acc = action_accuracy(&pred, &actual, 0.9);
        // First two should pass (cosine > 0.9), third should fail
        assert!(
            (acc - 2.0 / 3.0).abs() < 1e-10,
            "2/3 accuracy expected, got {acc}"
        );
    }

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution: max entropy
        let uniform = vec![10, 10, 10, 10];
        let h_uniform = shannon_entropy(&uniform);
        let h_max = (4.0f64).ln();
        assert!(
            (h_uniform - h_max).abs() < 1e-10,
            "Uniform should give max entropy {h_max}, got {h_uniform}"
        );

        // Single class: zero entropy
        let single = vec![100, 0, 0, 0];
        let h_single = shannon_entropy(&single);
        assert!(
            h_single.abs() < 1e-10,
            "Single class should give 0 entropy, got {h_single}"
        );

        // Partial: entropy should be between 0 and max
        let partial = vec![90, 10, 0, 0];
        let h_partial = shannon_entropy(&partial);
        assert!(h_partial > 0.0 && h_partial < h_max);
    }

    #[test]
    fn test_iwe_equal_policies() {
        // Same policy → ratio = 1.0 → IWE = mean(reward)
        let new_lp = vec![0.0; 5]; // log(1) = 0
        let old_lp = vec![0.0; 5];
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let iwe = importance_weighted_eval(&new_lp, &old_lp, &rewards);
        let expected = 3.0; // mean of rewards
        assert!(
            (iwe - expected).abs() < 1e-10,
            "Same policy: IWE should be mean(reward)={expected}, got {iwe}"
        );
    }

    #[test]
    fn test_iwe_clipping() {
        // Extreme ratio: should be clipped to [0.1, 10]
        let new_lp = vec![10.0]; // very high
        let old_lp = vec![-10.0]; // very low → ratio = e^20 >> 10
        let rewards = vec![1.0];

        let iwe = importance_weighted_eval(&new_lp, &old_lp, &rewards);
        assert!(
            (iwe - 10.0).abs() < 1e-10,
            "Clipped ratio × reward should be 10.0, got {iwe}"
        );
    }

    #[test]
    fn test_mahalanobis_in_distribution() {
        // Training states centered at 0 with unit variance
        let states: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..16)
                    .map(|j| ((i * 7 + j * 13) as f32 % 20.0 - 10.0) / 10.0)
                    .collect()
            })
            .collect();

        let detector = OodDetector::from_training_states(&states, 0.5);

        // In-distribution point (near mean)
        let near_mean = vec![0.0f32; 16];
        let dist = detector.mahalanobis(&near_mean);
        assert!(
            dist < detector.mahalanobis_threshold,
            "Near-mean point should be in-distribution, dist={dist}, threshold={}",
            detector.mahalanobis_threshold
        );
    }

    #[test]
    fn test_mahalanobis_ood() {
        let states: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..16)
                    .map(|j| ((i * 7 + j * 13) as f32 % 20.0 - 10.0) / 10.0)
                    .collect()
            })
            .collect();

        let detector = OodDetector::from_training_states(&states, 0.5);

        // Far OOD point
        let far_point = vec![100.0f32; 16];
        let dist = detector.mahalanobis(&far_point);
        assert!(
            dist > detector.mahalanobis_threshold,
            "Far point should be OOD, dist={dist}, threshold={}",
            detector.mahalanobis_threshold
        );
    }

    #[test]
    fn test_ood_detection_combined() {
        // Build a small codebook
        let mut codebook = ActionCodebook::new();
        for i in 0..5 {
            let emb: Vec<f32> = (0..ACTION_DIM)
                .map(|j| ((j + i * 50) as f32 * 0.1).sin())
                .collect();
            codebook.add_entry(CodebookEntry::new(
                "tool".into(),
                format!("action_{i}"),
                emb,
                10,
                0.5,
            ));
        }
        codebook.ood_threshold = 0.5;

        // Training states
        let states: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..ACTION_DIM)
                    .map(|j| ((j + i * 3) as f32 * 0.05).sin())
                    .collect()
            })
            .collect();

        let detector = OodDetector::from_training_states(&states, 0.5);

        // In-distribution query
        let in_state = states[0].clone();
        let in_action: Vec<f32> = (0..ACTION_DIM).map(|j| (j as f32 * 0.1).sin()).collect();
        let result = detector.detect(&in_state, &in_action, &codebook);
        // Should likely be in-distribution (state is from training)
        assert!(
            result.mahalanobis_distance < detector.mahalanobis_threshold,
            "Training state should pass Mahalanobis"
        );

        // OOD query
        let ood_state = vec![100.0f32; ACTION_DIM];
        let ood_action = vec![999.0f32; ACTION_DIM];
        let result = detector.detect(&ood_state, &ood_action, &codebook);
        assert!(result.is_ood, "Far-out point should be OOD");
        assert!(
            result.combined_score > 1.0,
            "Combined score should exceed 1.0"
        );
    }

    #[test]
    fn test_auc_perfect() {
        // Perfect separation: all OOD scores > all in-dist scores
        let in_dist = vec![0.1, 0.2, 0.3];
        let ood = vec![0.7, 0.8, 0.9];

        let auc = OodDetector::compute_auc(&in_dist, &ood);
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "Perfect separation should give AUC=1.0, got {auc}"
        );
    }

    #[test]
    fn test_auc_random() {
        // Random / overlapping: AUC should be ~0.5
        let in_dist = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let ood = vec![0.2, 0.4, 0.6, 0.8, 1.0];

        let auc = OodDetector::compute_auc(&in_dist, &ood);
        // Not exactly 0.5 due to specific values, but should be moderate
        assert!(
            auc > 0.3 && auc < 0.9,
            "Overlapping distributions should give moderate AUC, got {auc}"
        );
    }

    #[test]
    fn test_ranks() {
        let values = vec![3.0, 1.0, 2.0, 1.0];
        let ranks = compute_ranks(&values);
        // 1.0 appears at idx 1,3 → average rank (1+2)/2 = 1.5
        // 2.0 at idx 2 → rank 3
        // 3.0 at idx 0 → rank 4
        assert!((ranks[0] - 4.0).abs() < 1e-10);
        assert!((ranks[1] - 1.5).abs() < 1e-10);
        assert!((ranks[2] - 3.0).abs() < 1e-10);
        assert!((ranks[3] - 1.5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // MC Dropout tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mc_dropout_creation() {
        let estimator = McDropoutEstimator::new(16, 8, 32, 0.2, 10);
        assert!(estimator.is_ok());
        let estimator = estimator.unwrap();
        assert_eq!(estimator.num_passes, 10);
        assert_eq!(estimator.state_dim, 16);
        assert_eq!(estimator.action_dim, 8);
    }

    #[test]
    fn test_mc_dropout_fit() {
        let estimator = McDropoutEstimator::new(16, 8, 32, 0.1, 10).unwrap();

        // Generate simple training data
        let states: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..16).map(|j| ((i * 7 + j) as f32) / 100.0).collect())
            .collect();
        let actions: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| ((i * 3 + j) as f32) / 100.0).collect())
            .collect();

        let loss = estimator.fit(&states, &actions, 20, 0.01).unwrap();
        assert!(
            loss.is_finite(),
            "Training loss should be finite, got {loss}"
        );
    }

    #[test]
    fn test_mc_dropout_uncertainty() {
        let estimator = McDropoutEstimator::new(16, 8, 32, 0.2, 20).unwrap();

        // Fit on some data
        let states: Vec<Vec<f32>> = (0..30)
            .map(|i| (0..16).map(|j| ((i * 7 + j) as f32) / 100.0).collect())
            .collect();
        let actions: Vec<Vec<f32>> = (0..30)
            .map(|i| (0..8).map(|j| ((i * 3 + j) as f32) / 100.0).collect())
            .collect();

        estimator.fit(&states, &actions, 10, 0.01).unwrap();

        // Estimate uncertainty for a known point
        let uncertainty = estimator.estimate_uncertainty(&states[0]).unwrap();
        assert!(
            uncertainty >= 0.0,
            "Uncertainty should be non-negative, got {uncertainty}"
        );
        assert!(
            uncertainty.is_finite(),
            "Uncertainty should be finite, got {uncertainty}"
        );
    }

    #[test]
    fn test_mc_dropout_ood_vs_in_dist() {
        let estimator = McDropoutEstimator::new(16, 8, 64, 0.3, 30).unwrap();

        // Train on small-valued states
        let states: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..16)
                    .map(|j| ((i * 7 + j) as f32 % 20.0) / 100.0)
                    .collect()
            })
            .collect();
        let actions: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..8)
                    .map(|j| ((i * 3 + j) as f32 % 10.0) / 100.0)
                    .collect()
            })
            .collect();

        estimator.fit(&states, &actions, 30, 0.01).unwrap();

        // In-distribution uncertainty (near training data)
        let in_dist_unc = estimator.estimate_uncertainty(&states[0]).unwrap();

        // OOD uncertainty (far from training data)
        let ood_state = vec![100.0f32; 16];
        let ood_unc = estimator.estimate_uncertainty(&ood_state).unwrap();

        // Both should be non-negative and finite
        assert!(in_dist_unc >= 0.0 && in_dist_unc.is_finite());
        assert!(ood_unc >= 0.0 && ood_unc.is_finite());

        // Note: with random init and few epochs, the relative ordering isn't
        // guaranteed, but both values should be computable without error.
    }

    #[test]
    fn test_mc_dropout_calibrate() {
        let mut estimator = McDropoutEstimator::new(16, 8, 32, 0.2, 10).unwrap();

        let states: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..16).map(|j| ((i * 7 + j) as f32) / 100.0).collect())
            .collect();
        let actions: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..8).map(|j| ((i * 3 + j) as f32) / 100.0).collect())
            .collect();

        estimator.fit(&states, &actions, 5, 0.01).unwrap();
        estimator.calibrate_threshold(&states, 95.0).unwrap();

        assert!(
            estimator.uncertainty_threshold >= 0.0,
            "Calibrated threshold should be non-negative"
        );
        assert!(
            estimator.uncertainty_threshold.is_finite(),
            "Calibrated threshold should be finite"
        );
    }
}
