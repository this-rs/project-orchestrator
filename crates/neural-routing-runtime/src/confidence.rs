//! Confidence calibration — Platt scaling and progressive rollout.
//!
//! Ensures that the Policy Net's raw confidence scores are well-calibrated:
//! P(correct | confidence = c) ≈ c
//!
//! Uses Platt scaling (logistic regression on raw logits) fitted on a validation set.

use serde::{Deserialize, Serialize};
use std::sync::Once;

// ---------------------------------------------------------------------------
// Platt scaling
// ---------------------------------------------------------------------------

/// Largest slope magnitude `|a|` a Platt calibrator is allowed to publish with.
///
/// `a` is Platt's inverse-temperature: it is the total logit swing the sigmoid
/// applies across the whole raw-confidence range `[0, 1]`. The larger `|a|`, the
/// closer the calibrator is to a step function, and the more a difference in raw
/// confidence that the validation set could not possibly have resolved gets turned
/// into a difference in published probability.
///
/// Concrete toxic fit, produced by `fit()` itself: 12 samples whose raw confidences
/// all sit in `[0.490, 0.512]`, perfectly separable at the midpoint. Newton converges
/// to `a = 272.99`, `b = -136.77` — an optimum of the likelihood, and mathematically
/// nothing is wrong. Applied as-is it publishes:
///
/// ```text
///   raw 0.49 -> 0.047        raw 0.51 -> 0.921
/// ```
///
/// A 0.02 swing in raw confidence — the entire width of the data the fit ever saw,
/// i.e. noise — flips a caller gating at 0.7 from "reject" to "accept". At `|a| = 273`
/// the band where the published probability travels from 0.12 to 0.88 is 0.015 raw
/// confidence wide; the calibrator is a step function wearing a sigmoid's clothes.
///
/// 50.0 is the ceiling because:
/// * every healthy fit this crate's own fixtures produce lands well below it —
///   `|a| = 4.5` on noisy data, `18.5` on cleanly separated data at n=100, `25.9` at
///   n=200. The cap leaves roughly 2x headroom over the steepest legitimate fit, so
///   like `MIN_CLI_VERSION` it is a safety ceiling, not a validation: it does not
///   bite configurations that are merely aggressive;
/// * at `|a| = 50` the 0.12..0.88 band is still 0.08 raw confidence wide — steep, but
///   a gradation rather than a threshold.
///
/// When the ceiling bites, `a` and `b` are rescaled *jointly* by `MAX / |a|`. That is
/// exactly a temperature floor: it caps the sharpening while leaving the decision
/// midpoint `-b/a` — the only part of a degenerate fit that carries real information —
/// untouched. The fit above becomes `a = 50.0`, `b = -25.05`, same midpoint 0.501,
/// and publishes `raw 0.49 -> 0.366`, `raw 0.51 -> 0.611`: still ordered, no longer
/// pretending to a certainty it never measured.
///
/// This is deliberately NOT a hard error. A calibrator that trips the ceiling still
/// produces monotone, usable output; what changes is that a warning tells the operator
/// the confidences involved must be treated as uncalibrated.
pub const PLATT_MAX_ABS_SLOPE: f64 = 50.0;

/// Emitted once per process, not once per `calibrate()` call.
static PLATT_BOUND_WARNED: Once = Once::new();

fn warn_platt_bounded(a: f64, b: f64, reason: &str) {
    PLATT_BOUND_WARNED.call_once(|| {
        tracing::warn!(
            fitted_a = a,
            fitted_b = b,
            max_abs_slope = PLATT_MAX_ABS_SLOPE,
            "Platt calibration {reason}; parameters bounded before publication. \
             Treat the confidence of the affected entries as UNCALIBRATED — the fit \
             resolves differences in raw confidence that the validation set cannot \
             support. This warning is emitted once per process."
        );
    });
}

/// Bound `(a, b)` before they are ever used to publish a probability.
///
/// Returns the parameters untouched — bit-for-bit, no arithmetic applied — whenever
/// they are finite and within `PLATT_MAX_ABS_SLOPE`, so a healthy fit is strictly
/// unaffected by this guard.
fn bounded_params(a: f64, b: f64) -> (f64, f64) {
    if !a.is_finite() || !b.is_finite() {
        // A NaN/inf calibrator (corrupt persisted model, division by a singular
        // Hessian upstream) would publish NaN. Fall back to the default parameters.
        warn_platt_bounded(a, b, "parameters are not finite");
        return (-1.0, 0.0);
    }

    let abs_a = a.abs();
    if abs_a > PLATT_MAX_ABS_SLOPE {
        let scale = PLATT_MAX_ABS_SLOPE / abs_a;
        warn_platt_bounded(a, b, "slope exceeds the safety ceiling");
        return (a * scale, b * scale);
    }

    (a, b)
}

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
    /// The parameters actually used to publish a probability.
    ///
    /// Equal to `(self.a, self.b)` for any healthy calibrator; see
    /// [`PLATT_MAX_ABS_SLOPE`] for when and why they differ.
    pub fn effective_params(&self) -> (f64, f64) {
        bounded_params(self.a, self.b)
    }

    /// Calibrate a raw confidence score.
    pub fn calibrate(&self, raw_confidence: f32) -> f32 {
        let x = raw_confidence as f64;
        // The safety ceiling is applied to (a, b) *here*, before the sigmoid is
        // evaluated — never to the probability afterwards. A guard that fires after
        // publication does not protect anything. Applying it on this path (and not
        // only in `fit()`) also covers calibrators that never went through `fit()`:
        // hand-built literals and deserialized persisted models.
        let (a, b) = self.effective_params();
        // Platt's sigmoid: P(y=1|x) = 1/(1 + exp(Ax + B))
        // With A < 0 for well-calibrated output (higher x → higher P).
        // The fit() uses sigmoid(f) = 1/(1+exp(-f)) where f = ax+b,
        // so calibrate must also use sigmoid(ax+b) = 1/(1+exp(-(ax+b))).
        let p = 1.0 / (1.0 + (-(a * x + b)).exp());
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

        // Bound the fit before it is stored, so a persisted calibrator is already
        // sane and the operator sees the warning when the model is built rather than
        // on the first request it answers. `calibrate()` re-checks anyway.
        let (a, b) = bounded_params(a, b);

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

    // ── Review-fix regression tests ──────────────────────────────────────

    #[test]
    fn test_platt_calibrate_sign_correctness() {
        // Critical: calibrate(x) must use sigmoid(ax+b) = 1/(1+exp(-(ax+b))),
        // NOT 1/(1+exp(ax+b)). With a=-2, b=1:
        //   sigmoid(-2*0.9 + 1) = sigmoid(-0.8) ≈ 0.31
        //   sigmoid(-2*0.1 + 1) = sigmoid(0.8) ≈ 0.69
        // If sign were wrong, high input would give LOW output.
        let cal = PlattCalibrator {
            a: -2.0,
            b: 1.0,
            n_samples: 100,
        };

        let low_input = cal.calibrate(0.1);
        let high_input = cal.calibrate(0.9);

        // With correct sign: higher raw → lower sigmoid (because a < 0)
        // sigmoid(-2*0.1+1) = sigmoid(0.8) ≈ 0.69
        // sigmoid(-2*0.9+1) = sigmoid(-0.8) ≈ 0.31
        assert!(
            low_input > high_input,
            "With a=-2, b=1: calibrate(0.1)={:.4} should > calibrate(0.9)={:.4}",
            low_input,
            high_input
        );

        // Output must be in (0, 1)
        assert!(low_input > 0.0 && low_input < 1.0);
        assert!(high_input > 0.0 && high_input < 1.0);
    }

    #[test]
    fn test_platt_calibrate_boundary_values() {
        let cal = PlattCalibrator::default(); // a=-1, b=0
        let at_zero = cal.calibrate(0.0);
        let at_one = cal.calibrate(1.0);

        // sigmoid(0) = 0.5
        assert!(
            (at_zero - 0.5).abs() < 0.01,
            "calibrate(0.0) should ≈ 0.5, got {}",
            at_zero
        );
        // sigmoid(1) = 1/(1+exp(-1)) ≈ 0.73  (with default a=-1, b=0: sigmoid(-1*1+0) = sigmoid(-1) ≈ 0.27)
        assert!(at_one > 0.0 && at_one < 1.0);
    }

    #[test]
    fn test_platt_fit_then_calibrate_preserves_ordering() {
        // Fit on realistic data, then verify calibrated ordering matches raw ordering
        let data: Vec<(f32, bool)> = (0..200)
            .map(|i| {
                let conf = i as f32 / 200.0;
                (conf, conf > 0.45) // slightly noisy threshold
            })
            .collect();

        let cal = PlattCalibrator::fit(&data);

        // Calibrate a range and check monotonicity
        let calibrated: Vec<f32> = (0..20).map(|i| cal.calibrate(i as f32 / 20.0)).collect();
        for i in 1..calibrated.len() {
            assert!(
                calibrated[i] >= calibrated[i - 1] - 0.01,
                "Calibration not monotonic at {}: {:.4} < {:.4}",
                i,
                calibrated[i],
                calibrated[i - 1]
            );
        }
    }

    // ── Safety-ceiling on the fitted slope (Laya-style guard) ────────────

    /// The unguarded formula, exactly as `calibrate()` computed it before the
    /// safety ceiling existed. Non-regression tests compare against this.
    fn unguarded_calibrate(a: f64, b: f64, raw: f32) -> f32 {
        let x = raw as f64;
        (1.0 / (1.0 + (-(a * x + b)).exp())) as f32
    }

    /// The three fixtures already exercised by the tests above, plus a noisy one.
    fn healthy_fixtures() -> Vec<(&'static str, Vec<(f32, bool)>)> {
        vec![
            (
                "separable_at_0.5_n100",
                (0..100)
                    .map(|i| {
                        let c = i as f32 / 100.0;
                        (c, c > 0.5)
                    })
                    .collect(),
            ),
            (
                "noisy_n200",
                (0..200)
                    .map(|i| {
                        let c = i as f32 / 200.0;
                        (c, c + 0.1 * ((i as f32 * 0.7).sin()) > 0.5)
                    })
                    .collect(),
            ),
            (
                "separable_at_0.45_n200",
                (0..200)
                    .map(|i| {
                        let c = i as f32 / 200.0;
                        (c, c > 0.45)
                    })
                    .collect(),
            ),
            (
                "realistic_noisy_n200",
                (0..200)
                    .map(|i| {
                        let c = 0.2 + 0.7 * (i as f32 / 200.0);
                        (c, (((i * 7919) % 100) as f32 / 100.0) < c)
                    })
                    .collect(),
            ),
        ]
    }

    #[test]
    fn test_platt_default_is_strictly_unchanged_by_guard() {
        let cal = PlattCalibrator::default(); // a = -1, b = 0
        let (a, b) = cal.effective_params();

        // Bit-for-bit: the guard must not touch the default parameters at all.
        assert_eq!(
            a, -1.0,
            "default slope must pass through the guard untouched"
        );
        assert_eq!(
            b, 0.0,
            "default intercept must pass through the guard untouched"
        );

        for i in 0..=20 {
            let raw = i as f32 / 20.0;
            assert_eq!(
                cal.calibrate(raw),
                unguarded_calibrate(-1.0, 0.0, raw),
                "default calibration changed at raw={raw}"
            );
        }
    }

    #[test]
    fn test_platt_healthy_fit_is_bitwise_unchanged_by_guard() {
        // Non-regression: for every healthy fixture, the guard is a strict no-op —
        // same (a, b) bits, and calibrate() returns exactly what the pre-guard
        // formula returns. Demonstrated, not asserted in prose.
        for (name, data) in healthy_fixtures() {
            let cal = PlattCalibrator::fit(&data);

            assert!(
                cal.a.abs() < PLATT_MAX_ABS_SLOPE,
                "fixture {name} produced |a| = {:.4}, at or above the ceiling {} — \
                 it is no longer a healthy-fit fixture",
                cal.a.abs(),
                PLATT_MAX_ABS_SLOPE
            );

            let (a, b) = cal.effective_params();
            assert_eq!(
                a, cal.a,
                "guard altered the slope of healthy fixture {name}"
            );
            assert_eq!(
                b, cal.b,
                "guard altered the intercept of healthy fixture {name}"
            );

            for i in 0..=100 {
                let raw = i as f32 / 100.0;
                assert_eq!(
                    cal.calibrate(raw),
                    unguarded_calibrate(cal.a, cal.b, raw),
                    "fixture {name}: calibrate({raw}) changed"
                );
            }
        }
    }

    #[test]
    fn test_platt_healthy_fit_matches_pre_guard_golden_values() {
        // Golden values captured from the implementation BEFORE the safety ceiling
        // was introduced. Any drift in fit() or calibrate() breaks this.
        struct Golden {
            name: &'static str,
            data: Vec<(f32, bool)>,
            a: f64,
            b: f64,
            p: [(f32, f64); 3],
        }

        let goldens = vec![
            Golden {
                name: "separable_at_0.5_n100",
                data: (0..100)
                    .map(|i| {
                        let c = i as f32 / 100.0;
                        (c, c > 0.5)
                    })
                    .collect(),
                a: 1.849_395_597_403_027_5e1,
                b: -9.339_209_806_439_31,
                p: [
                    (0.1, 0.000_558_435),
                    (0.5, 0.476_958_377),
                    (0.9, 0.999_328_517),
                ],
            },
            Golden {
                name: "separable_at_0.45_n200",
                data: (0..200)
                    .map(|i| {
                        let c = i as f32 / 200.0;
                        (c, c > 0.45)
                    })
                    .collect(),
                a: 2.591_085_318_809_990_7e1,
                b: -1.172_510_601_853_088_7e1,
                p: [
                    (0.1, 0.000_107_919),
                    (0.5, 0.773_874_678),
                    (0.9, 0.999_990_785),
                ],
            },
        ];

        for g in goldens {
            let cal = PlattCalibrator::fit(&g.data);
            assert!(
                (cal.a - g.a).abs() / g.a.abs() < 1e-12,
                "{}: slope drifted, expected {} got {}",
                g.name,
                g.a,
                cal.a
            );
            assert!(
                (cal.b - g.b).abs() / g.b.abs() < 1e-12,
                "{}: intercept drifted, expected {} got {}",
                g.name,
                g.b,
                cal.b
            );
            for (raw, expected) in g.p {
                let got = cal.calibrate(raw) as f64;
                assert!(
                    (got - expected).abs() < 1e-6,
                    "{}: calibrate({raw}) expected {expected}, got {got}",
                    g.name
                );
            }
        }
    }

    #[test]
    fn test_platt_degenerate_handbuilt_fit_is_bounded_not_applied() {
        // The toxic fit documented on PLATT_MAX_ABS_SLOPE, built by hand the way a
        // deserialized persisted model would arrive: it never goes through fit().
        let cal = PlattCalibrator {
            a: 272.994_801_540_981_54,
            b: -136.770_398_053_008_08,
            n_samples: 12,
        };

        let (a, b) = cal.effective_params();
        assert!(
            a.abs() <= PLATT_MAX_ABS_SLOPE + 1e-9,
            "slope was not bounded: {a}"
        );

        // The decision midpoint -b/a is preserved by the joint rescale.
        assert!(
            ((-b / a) - (-cal.b / cal.a)).abs() < 1e-9,
            "rescale moved the decision midpoint: {} -> {}",
            -cal.b / cal.a,
            -b / a
        );

        // Applied as-is, 0.49/0.51 would publish 0.047/0.921. The guard must NOT
        // publish those.
        let p_low = cal.calibrate(0.49);
        let p_high = cal.calibrate(0.51);

        assert!(
            p_low > 0.2,
            "raw 0.49 still published as a near-certainty of failure: {p_low}"
        );
        assert!(
            p_high < 0.8,
            "raw 0.51 still published as a near-certainty: {p_high}"
        );
        // Ordering — the one thing the degenerate fit did know — is kept.
        assert!(p_high > p_low, "bounding must not invert the ordering");

        // And it is a ceiling, not a rejection: the output is still a live sigmoid.
        assert!(cal.calibrate(0.0) < cal.calibrate(1.0));
    }

    #[test]
    fn test_platt_fit_on_tight_cluster_is_bounded() {
        // 12 samples spanning 0.490..0.512, perfectly separable. Newton converges to
        // |a| ≈ 273 — an honest likelihood optimum over data that resolves nothing.
        let data: Vec<(f32, bool)> = (0..12).map(|i| (0.49 + 0.002 * i as f32, i >= 6)).collect();

        let cal = PlattCalibrator::fit(&data);

        assert_eq!(cal.n_samples, 12, "the fit itself must still happen");
        assert!(
            cal.a.abs() <= PLATT_MAX_ABS_SLOPE + 1e-9,
            "fit() stored an unbounded slope: {}",
            cal.a
        );

        // The published band over the data's own width is a gradation, not a step.
        let p_low = cal.calibrate(0.49);
        let p_high = cal.calibrate(0.51);
        assert!(
            p_high - p_low < 0.7,
            "0.02 of raw confidence still moves the published probability by {:.3} \
             ({p_low:.3} -> {p_high:.3})",
            p_high - p_low
        );
    }

    #[test]
    fn test_platt_non_finite_params_fall_back_to_default() {
        for (a, b) in [
            (f64::NAN, 0.0),
            (f64::INFINITY, 0.0),
            (1.0, f64::NEG_INFINITY),
        ] {
            let cal = PlattCalibrator {
                a,
                b,
                n_samples: 50,
            };
            assert_eq!(
                cal.effective_params(),
                (-1.0, 0.0),
                "non-finite ({a}, {b}) must fall back to the default parameters"
            );
            let p = cal.calibrate(0.8);
            assert!(
                p.is_finite() && p > 0.0 && p < 1.0,
                "non-finite parameters published {p}"
            );
        }
    }
}
