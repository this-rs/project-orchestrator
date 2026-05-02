//! Student's t-tests (one-sample + two-sample with Welch / pooled variants).
//!
//! Replaces `rs_stats::hypothesis_tests::t_test` (GPL-3.0). Uses
//! [`statrs::distribution::StudentsT`] for the CDF (more accurate than
//! rs-stats's hand-rolled `incomplete_beta`).
//!
//! ## Variance convention
//!
//! All t-tests internally use **sample** variance (denom `n-1`, Bessel-
//! corrected) via [`super::mean_std::std_dev_sample`]. This matches the
//! private `calculate_variance` helper inside rs-stats's t_test.rs, NOT
//! the public `prob::std_dev` (which is population, denom `n`).
//!
//! See `docs/migration/rs-stats/audit-ttest-impl.md` for the full audit
//! and `audit-variance-semantic.md` for why both conventions co-exist.
//!
//! Plan: `00f0ca9a-816f-4fcc-bc53-da88d595de34`, task R4.

use statrs::distribution::{ContinuousCDF, StudentsT};

use super::mean_std::{mean, std_dev_sample};

/// Result of a t-test (one-sample, two-sample pooled, or Welch).
///
/// Mirrors the shape of `rs_stats::hypothesis_tests::t_test::TTestResult`
/// minus `std_error` (not consumed by `analytics::hypothesis`).
///
/// - One-sample: `mean_values = [μ]`, `std_devs = [σ]`
/// - Two-sample: `mean_values = [μ_a, μ_b]`, `std_devs = [σ_a, σ_b]`
#[derive(Debug, Clone, PartialEq)]
pub struct TTestResult {
    /// Computed t-statistic
    pub t_statistic: f64,
    /// Degrees of freedom (integer for pooled, fractional for Welch)
    pub degrees_of_freedom: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Sample mean(s) — 1 element (one-sample) or 2 (two-sample)
    pub mean_values: Vec<f64>,
    /// Sample standard deviation(s) — Bessel-corrected (denom n-1)
    pub std_devs: Vec<f64>,
}

/// Two-tailed p-value of a t-statistic under Student's t with `df` degrees
/// of freedom.
///
/// Returns `2 * (1 - cdf(|t|))`, clamped to `[0.0, 1.0]` for numerical safety.
/// Returns `1.0` if `t == 0.0` or `df` is non-positive (degenerate cases).
///
/// Replaces rs-stats's hand-rolled `incomplete_beta` (which has known
/// precision issues at extreme |t|). `statrs::distribution::StudentsT` uses
/// the regularized incomplete beta function with industrial-strength
/// numerics.
///
/// # Examples
///
/// ```
/// # use project_orchestrator::analytics::stats::t_test::two_tailed_p_value;
/// // df=10, t≈2.228 — the 0.05 critical value (two-tailed) → p ≈ 0.05
/// let p = two_tailed_p_value(2.228, 10.0);
/// assert!((p - 0.05).abs() < 1e-3);
///
/// // t = 0 → p = 1.0 (no evidence against null)
/// assert_eq!(two_tailed_p_value(0.0, 5.0), 1.0);
/// ```
pub fn two_tailed_p_value(t: f64, df: f64) -> f64 {
    if !df.is_finite() || df <= 0.0 {
        return 1.0;
    }
    if t == 0.0 {
        return 1.0;
    }
    // statrs requires location=0, scale=1, freedom=df.
    let dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(d) => d,
        Err(_) => return 1.0, // unreachable for finite positive df
    };
    let cdf = dist.cdf(t.abs());
    (2.0 * (1.0 - cdf)).clamp(0.0, 1.0)
}

/// One-sample t-test: tests whether the sample mean differs from a
/// reference (population) mean.
///
/// `t = (μ - μ₀) / (s / √n)` with sample std `s` (denom n-1), df = n-1.
///
/// Returns `None` if `n < 2` (rs-stats: `Err InvalidInput`).
///
/// # Examples
///
/// ```
/// # use project_orchestrator::analytics::stats::t_test::one_sample_t_test;
/// let r = one_sample_t_test(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0).unwrap();
/// assert!((r.degrees_of_freedom - 4.0).abs() < 1e-12);
/// // mean=3, sample_std=sqrt(2.5), std_error=sqrt(2.5)/sqrt(5), t = 3/std_error
/// let expected_t = 3.0 / ((2.5_f64.sqrt()) / 5.0_f64.sqrt());
/// assert!((r.t_statistic - expected_t).abs() < 1e-12);
/// ```
pub fn one_sample_t_test(values: &[f64], reference_mean: f64) -> Option<TTestResult> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mu = mean(values)?;
    let s = std_dev_sample(values)?;
    let std_error = s / n.sqrt();
    let t_statistic = (mu - reference_mean) / std_error;
    let df = n - 1.0;
    let p_value = two_tailed_p_value(t_statistic, df);
    Some(TTestResult {
        t_statistic,
        degrees_of_freedom: df,
        p_value,
        mean_values: vec![mu],
        std_devs: vec![s],
    })
}

/// Two-sample t-test (independent samples).
///
/// - `assume_equal_variance = true`: classic Student's pooled t-test.
///   `df = n_a + n_b - 2`.
/// - `assume_equal_variance = false`: **Welch's** t-test. Fractional df via
///   Welch-Satterthwaite. Robust to unequal variances/sizes (and is the
///   default for rs-stats's documented examples).
///
/// All variances are **sample** (denom n-1) to match rs-stats's internal
/// `calculate_variance`.
///
/// Returns `None` if either side has `n < 2`.
pub fn two_sample_t_test(a: &[f64], b: &[f64], assume_equal_variance: bool) -> Option<TTestResult> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let mean_a = mean(a)?;
    let mean_b = mean(b)?;
    let std_a = std_dev_sample(a)?;
    let std_b = std_dev_sample(b)?;
    let var_a = std_a * std_a;
    let var_b = std_b * std_b;

    let (t_statistic, df) = if assume_equal_variance {
        // Pooled (Student's): pooled_var = ((n_a-1)*var_a + (n_b-1)*var_b) / (n_a+n_b-2)
        let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
        let std_error = (pooled_var * (1.0 / n_a + 1.0 / n_b)).sqrt();
        let t = (mean_a - mean_b) / std_error;
        let df = n_a + n_b - 2.0;
        (t, df)
    } else {
        // Welch's: std_error = sqrt(var_a/n_a + var_b/n_b)
        let var_a_n = var_a / n_a;
        let var_b_n = var_b / n_b;
        let std_error = (var_a_n + var_b_n).sqrt();
        let t = (mean_a - mean_b) / std_error;
        // Welch-Satterthwaite df
        let num = (var_a_n + var_b_n).powi(2);
        let den = var_a_n.powi(2) / (n_a - 1.0) + var_b_n.powi(2) / (n_b - 1.0);
        let df = num / den;
        (t, df)
    };

    let p_value = two_tailed_p_value(t_statistic, df);

    Some(TTestResult {
        t_statistic,
        degrees_of_freedom: df,
        p_value,
        mean_values: vec![mean_a, mean_b],
        std_devs: vec![std_a, std_b],
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::stats::golden_fixtures::*;

    // ─── two_tailed_p_value ──────────────────────────────────────────────────

    #[test]
    fn p_value_t_zero_is_one() {
        assert_eq!(two_tailed_p_value(0.0, 5.0), 1.0);
    }

    #[test]
    fn p_value_classical_table_value() {
        // df=10, t=2.228 is the 0.05 critical value (two-tailed) → p ≈ 0.05
        let p = two_tailed_p_value(2.228, 10.0);
        assert!((p - 0.05).abs() < 1e-3, "got p={p}");
    }

    #[test]
    fn p_value_large_t_is_tiny() {
        // df=10, |t|=10 — very strong evidence against H0
        let p = two_tailed_p_value(10.0, 10.0);
        assert!(p < 1e-5, "got p={p}");
    }

    #[test]
    fn p_value_invalid_df_returns_one() {
        assert_eq!(two_tailed_p_value(2.0, 0.0), 1.0);
        assert_eq!(two_tailed_p_value(2.0, -1.0), 1.0);
        assert_eq!(two_tailed_p_value(2.0, f64::NAN), 1.0);
    }

    // ─── one_sample_t_test ───────────────────────────────────────────────────

    #[test]
    fn one_sample_n_too_small() {
        assert!(one_sample_t_test(&[], 0.0).is_none());
        assert!(one_sample_t_test(&[42.0], 0.0).is_none());
    }

    #[test]
    fn one_sample_basic() {
        let r = one_sample_t_test(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0).unwrap();
        // sample_std = sqrt(2.5), std_error = sqrt(2.5)/sqrt(5)
        // t = 3 / std_error = 3 * sqrt(5) / sqrt(2.5)
        let expected_t = 3.0 * 5.0_f64.sqrt() / 2.5_f64.sqrt();
        assert!((r.t_statistic - expected_t).abs() < 1e-12);
        assert!((r.degrees_of_freedom - 4.0).abs() < 1e-12);
        assert_eq!(r.mean_values, vec![3.0]);
        assert!((r.std_devs[0] - 2.5_f64.sqrt()).abs() < 1e-12);
    }

    // ─── two_sample_t_test ───────────────────────────────────────────────────

    #[test]
    fn two_sample_n_too_small() {
        assert!(two_sample_t_test(&[1.0], &[2.0, 3.0], false).is_none());
        assert!(two_sample_t_test(&[1.0, 2.0], &[3.0], false).is_none());
        assert!(two_sample_t_test(&[], &[2.0, 3.0], true).is_none());
    }

    #[test]
    fn two_sample_pooled_df_is_n1_plus_n2_minus_2() {
        let r = two_sample_t_test(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 3.0, 4.0, 5.0, 6.0], true)
            .unwrap();
        assert!((r.degrees_of_freedom - 8.0).abs() < 1e-12);
    }

    #[test]
    fn two_sample_welch_df_le_pooled_df() {
        let a = &[5.2, 6.4, 6.9, 7.3, 7.5, 7.8, 8.1, 8.4, 9.2, 9.5];
        let b = &[4.1, 5.0, 5.5, 6.2, 6.3, 6.5, 6.8, 7.1, 7.4, 7.5];
        let pooled = two_sample_t_test(a, b, true).unwrap();
        let welch = two_sample_t_test(a, b, false).unwrap();
        // Welch df ≤ pooled df (Welch-Satterthwaite penalty)
        assert!(welch.degrees_of_freedom <= pooled.degrees_of_freedom + 1e-10);
    }

    #[test]
    fn two_sample_identical_samples_t_is_zero() {
        let v = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let r = two_sample_t_test(v, v, false).unwrap();
        assert!(r.t_statistic.abs() < 1e-12);
    }

    // ─── ISO baseline against rs-stats EXPECTED_* ────────────────────────────

    /// Helper: assert TTestResult matches the captured TTestExpected within
    /// the plan tolerance matrix (t_stat at 1e-10 relative, df at 1e-10
    /// relative, p_value at 1e-3 absolute when rs-stats produced a
    /// meaningful value — saturated 0/1 outputs are accepted as "very
    /// small" / "very large").
    fn assert_iso_match(actual: &TTestResult, expected: TTestExpected, label: &str) {
        assert_relative_eq(
            actual.t_statistic,
            expected.t_stat,
            1e-10,
            &format!("{label} t_stat"),
        );
        assert_relative_eq(
            actual.degrees_of_freedom,
            expected.df,
            1e-10,
            &format!("{label} df"),
        );
        // p_value: rs-stats often saturates to 0.0 or 1.0 (incomplete_beta
        // precision issues — see audit-ttest-impl.md). statrs is more
        // accurate. Per plan matrix: 1e-3 absolute IF rs-stats was meaningful.
        if expected.p_value > 0.0 && expected.p_value < 1.0 {
            assert_absolute_eq(
                actual.p_value,
                expected.p_value,
                1e-3,
                &format!("{label} p_value"),
            );
        } else {
            assert!(
                (0.0..=1.0).contains(&actual.p_value),
                "{label} p_value out of range: {}",
                actual.p_value
            );
        }
    }

    #[test]
    fn iso_one_sample_small_integers() {
        let r = one_sample_t_test(SMALL_INTEGERS, 0.0).unwrap();
        assert_iso_match(&r, EXPECTED_TTEST1_SMALL_INTEGERS, "ttest1_SMALL_INTEGERS");
    }

    #[test]
    fn iso_one_sample_medium_normal() {
        let r = one_sample_t_test(MEDIUM_NORMAL, 0.0).unwrap();
        assert_iso_match(&r, EXPECTED_TTEST1_MEDIUM_NORMAL, "ttest1_MEDIUM_NORMAL");
    }

    #[test]
    fn iso_one_sample_near_zero_flat() {
        let r = one_sample_t_test(NEAR_ZERO_FLAT, 0.0).unwrap();
        assert_iso_match(&r, EXPECTED_TTEST1_NEAR_ZERO_FLAT, "ttest1_NEAR_ZERO_FLAT");
    }

    #[test]
    fn iso_two_sample_welch_shifted() {
        // From R1 capture: SMALL_INTEGERS vs [2,3,4,5,6] (shifted by 1)
        let r = two_sample_t_test(SMALL_INTEGERS, &[2.0, 3.0, 4.0, 5.0, 6.0], false).unwrap();
        assert_iso_match(&r, EXPECTED_TTEST2_WELCH_SHIFTED, "ttest2_WELCH_SHIFTED");
    }

    #[test]
    fn iso_two_sample_welch_identical() {
        // From R1 capture: SMALL_INTEGERS vs SMALL_INTEGERS — identical → t=0
        let r = two_sample_t_test(SMALL_INTEGERS, SMALL_INTEGERS, false).unwrap();
        assert_iso_match(
            &r,
            EXPECTED_TTEST2_WELCH_IDENTICAL,
            "ttest2_WELCH_IDENTICAL",
        );
    }
}
