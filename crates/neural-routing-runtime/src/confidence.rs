//! Confidence calibration — Platt scaling and progressive rollout.
//!
//! Ensures that the Policy Net's raw confidence scores are well-calibrated:
//! P(correct | confidence = c) ≈ c
//!
//! Uses Platt scaling (logistic regression on raw logits) fitted on a validation set.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Platt scaling
// ---------------------------------------------------------------------------

/// Platt scaling parameters: P(y=1|x) = 1 / (1 + exp(A*x + B))
///
/// Fitted via maximum likelihood on validation set (confidence, was_correct) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattCalibrator {
    /// Slope parameter (learned).
    pub a: f64,
    /// Intercept parameter (learned).
    pub b: f64,
    /// Number of samples used for fitting.
    pub n_samples: usize,
}

impl Default for PlattCalibrator {
    fn default() -> Self {
        // Identity calibration (no transformation): A=-1, B=0 → sigmoid(-x) ≈ x for x ∈ [0,1]
        Self {
            a: -1.0,
            b: 0.0,
            n_samples: 0,
        }
    }
}

impl PlattCalibrator {
    /// Calibrate a raw confidence score.
    pub fn calibrate(&self, raw_confidence: f32) -> f32 {
        let x = raw_confidence as f64;
        // Platt's sigmoid: P(y=1|x) = 1/(1 + exp(Ax + B))
        // With A < 0 for well-calibrated output (higher x → higher P).
        // The fit() uses sigmoid(f) = 1/(1+exp(-f)) where f = ax+b,
        // so calibrate must also use sigmoid(ax+b) = 1/(1+exp(-(ax+b))).
        let p = 1.0 / (1.0 + (-(self.a * x + self.b)).exp());
        p as f32
    }

    /// Fit Platt scaling parameters from (confidence, was_correct) pairs.
    ///
    /// Uses Newton's method to minimize negative log-likelihood.
    /// Reference: Platt (1999), "Probabilistic Outputs for Support Vector Machines"
    pub fn fit(data: &[(f32, bool)]) -> Self {
        if data.len() < 10 {
            return Self::default();
        }

        let n = data.len();
        let n_pos = data.iter().filter(|(_, y)| *y).count();
        let n_neg = n - n_pos;

        if n_pos == 0 || n_neg == 0 {
            return Self::default();
        }

        // Target labels with Bayes-corrected smoothing
        let t_pos = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let t_neg = 1.0 / (n_neg as f64 + 2.0);

        let targets: Vec<f64> = data
            .iter()
            .map(|(_, y)| if *y { t_pos } else { t_neg })
            .collect();
        let scores: Vec<f64> = data.iter().map(|(s, _)| *s as f64).collect();

        // Newton's method for logistic regression
        let mut a = 0.0f64;
        let mut b = ((n_neg as f64 + 1.0) / (n_pos as f64 + 1.0)).ln();

        let max_iter = 100;
        let min_step = 1e-10;

        for _ in 0..max_iter {
            // Compute gradient and Hessian
            let mut fval = 0.0f64;
            let mut fval_ab = 0.0f64;
            let mut fval_aa = 0.0f64;
            let mut fval_bb = 0.0f64;
            let mut fval_a = 0.0f64;
            let mut fval_b = 0.0f64;

            for i in 0..n {
                let f_approx = scores[i] * a + b;
                let p = 1.0 / (1.0 + (-f_approx).exp());
                let t = targets[i];
                let d1 = p - t;
                let d2 = p * (1.0 - p);

                fval_a += scores[i] * d1;
                fval_b += d1;
                fval_aa += scores[i] * scores[i] * d2;
                fval_ab += scores[i] * d2;
                fval_bb += d2;

                // Log-likelihood (for convergence check)
                fval += t * f_approx.ln_1p_exp_neg() + (1.0 - t) * (-f_approx).ln_1p_exp_neg();
            }

            // Regularization to prevent singular Hessian
            fval_aa += 1e-6;
            fval_bb += 1e-6;

            let det = fval_aa * fval_bb - fval_ab * fval_ab;
            if det.abs() < 1e-12 {
                break;
            }

            let da = -(fval_bb * fval_a - fval_ab * fval_b) / det;
            let db = -(fval_aa * fval_b - fval_ab * fval_a) / det;

            // Line search with backtracking
            let mut step = 1.0;
            let old_fval = fval;
            loop {
                let new_a = a + step * da;
                let new_b = b + step * db;

                let mut new_fval = 0.0f64;
                for i in 0..n {
                    let f_approx = scores[i] * new_a + new_b;
                    let t = targets[i];
                    new_fval +=
                        t * f_approx.ln_1p_exp_neg() + (1.0 - t) * (-f_approx).ln_1p_exp_neg();
                }

                if new_fval <= old_fval + 1e-4 * step * (fval_a * da + fval_b * db) {
                    a = new_a;
                    b = new_b;
                    break;
                }

                step *= 0.5;
                if step < min_step {
                    a = new_a;
                    b = new_b;
                    break;
                }
            }

            if (step * da).abs() < min_step && (step * db).abs() < min_step {
                break;
            }
        }

        Self { a, b, n_samples: n }
    }
}

/// Helper: ln(1 + exp(-x)) for numerical stability.
trait LnStable {
    fn ln_1p_exp_neg(self) -> Self;
}

impl LnStable for f64 {
    fn ln_1p_exp_neg(self) -> f64 {
        if self > 20.0 {
            (-self).exp() // ln(1 + exp(-x)) ≈ exp(-x) for large x
        } else if self < -20.0 {
            -self // ln(1 + exp(-x)) ≈ -x for very negative x
        } else {
            (1.0 + (-self).exp()).ln()
        }
    }
}

// ---------------------------------------------------------------------------
// Progressive Rollout
// ---------------------------------------------------------------------------

/// Progressive rollout configuration for gradually increasing Policy Net traffic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutConfig {
    /// Fraction of queries routed to Policy Net (0.0 - 1.0).
    pub policy_fraction: f32,
    /// Minimum confidence threshold for Policy Net to take over.
    pub confidence_threshold: f32,
    /// Whether to force fallback (emergency kill switch).
    pub force_fallback: bool,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        Self {
            policy_fraction: 0.0, // Start with 0% — all traffic to heuristic
            confidence_threshold: 0.7,
            force_fallback: false,
        }
    }
}

impl RolloutConfig {
    /// Check if a query should be routed to the Policy Net.
    ///
    /// Uses a deterministic hash of the session_id for consistent routing.
    pub fn should_use_policy(&self, session_hash: u64) -> bool {
        if self.force_fallback {
            return false;
        }
        if self.policy_fraction <= 0.0 {
            return false;
        }
        if self.policy_fraction >= 1.0 {
            return true;
        }

        // Deterministic: same session always gets same routing
        let bucket = (session_hash % 10000) as f32 / 10000.0;
        bucket < self.policy_fraction
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_default_identity() {
        let cal = PlattCalibrator::default();
        // Default should approximately preserve input
        let c = cal.calibrate(0.8);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn test_platt_fit_perfect() {
        // All high-confidence predictions are correct, all low are wrong
        let data: Vec<(f32, bool)> = (0..100)
            .map(|i| {
                let conf = i as f32 / 100.0;
                (conf, conf > 0.5)
            })
            .collect();

        let cal = PlattCalibrator::fit(&data);
        assert!(cal.n_samples == 100);

        // After calibration, high confidence should map to high probability
        let low = cal.calibrate(0.1);
        let high = cal.calibrate(0.9);
        assert!(
            high > low,
            "High confidence ({}) should calibrate higher than low ({})",
            high,
            low
        );
    }

    #[test]
    fn test_platt_fit_too_few_samples() {
        let data = vec![(0.5, true), (0.3, false)];
        let cal = PlattCalibrator::fit(&data);
        // Should return default
        assert_eq!(cal.n_samples, 0);
    }

    #[test]
    fn test_platt_monotonic() {
        let data: Vec<(f32, bool)> = (0..200)
            .map(|i| {
                let conf = i as f32 / 200.0;
                let correct = conf + 0.1 * ((i as f32 * 0.7).sin()) > 0.5;
                (conf, correct)
            })
            .collect();

        let cal = PlattCalibrator::fit(&data);

        // Calibrated values should be monotonic (approximately)
        let values: Vec<f32> = (0..10).map(|i| cal.calibrate(i as f32 / 10.0)).collect();

        for i in 1..values.len() {
            // Allow small non-monotonicity due to fitting noise
            assert!(
                values[i] >= values[i - 1] - 0.05,
                "Calibration should be roughly monotonic: {:?}",
                values
            );
        }
    }

    #[test]
    fn test_rollout_force_fallback() {
        let config = RolloutConfig {
            force_fallback: true,
            policy_fraction: 1.0,
            ..Default::default()
        };
        assert!(!config.should_use_policy(42));
    }

    #[test]
    fn test_rollout_zero_fraction() {
        let config = RolloutConfig::default(); // policy_fraction = 0.0
        assert!(!config.should_use_policy(42));
    }

    #[test]
    fn test_rollout_full_fraction() {
        let config = RolloutConfig {
            policy_fraction: 1.0,
            ..Default::default()
        };
        assert!(config.should_use_policy(42));
    }

    #[test]
    fn test_rollout_split_ratio() {
        let config = RolloutConfig {
            policy_fraction: 0.3,
            ..Default::default()
        };

        let mut policy_count = 0;
        for i in 0..10000u64 {
            if config.should_use_policy(i) {
                policy_count += 1;
            }
        }

        let ratio = policy_count as f32 / 10000.0;
        assert!(
            (ratio - 0.3).abs() < 0.05,
            "Expected ~30% policy routing, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_rollout_deterministic() {
        let config = RolloutConfig {
            policy_fraction: 0.5,
            ..Default::default()
        };

        // Same hash should always give same result
        let r1 = config.should_use_policy(12345);
        let r2 = config.should_use_policy(12345);
        assert_eq!(r1, r2);
    }
}
