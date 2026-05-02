//! Arithmetic mean and standard deviation primitives.
//!
//! Replaces `rs_stats::prob::average` and `rs_stats::prob::std_dev` (GPL-3.0).
//! See `docs/migration/rs-stats/audit-variance-semantic.md` for the variance
//! convention rationale (population vs sample) and `baseline-edge-cases.md`
//! for the rs-stats behavior contract preserved here.
//!
//! Plan: `00f0ca9a-816f-4fcc-bc53-da88d595de34`, task R3.

/// Arithmetic mean of `values`.
///
/// Returns `None` on empty input. Matches `rs_stats::prob::average` semantics:
/// rs-stats returned `Err(EmptyData)` on `&[]`; the `.unwrap_or_else(...)`
/// fallback at call sites turns this into the same effective behavior.
///
/// NaN and infinity propagate through the sum (no panicking, no special
/// handling) — this matches rs-stats.
///
/// # Examples
///
/// ```
/// # use project_orchestrator::analytics::stats::mean_std::mean;
/// assert_eq!(mean(&[]), None);
/// assert_eq!(mean(&[42.0]), Some(42.0));
/// assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(3.0));
/// ```
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

/// Population standard deviation: `sqrt(Σ(x_i - μ)² / n)`.
///
/// Uses denominator `n` (NOT Bessel-corrected `n-1`). This matches
/// `rs_stats::prob::std_dev`, which is the convention used throughout
/// `analytics::distribution`. For t-tests, use [`std_dev_sample`] (Bessel)
/// instead.
///
/// Returns `None` on empty input. For `n=1`, returns `Some(0.0)` (matches
/// rs-stats: `m2/1 = 0`, `sqrt(0) = 0`).
///
/// NaN and infinity propagate.
///
/// # References
///
/// - `docs/migration/rs-stats/audit-variance-semantic.md` — proves rs-stats
///   uses `denom = n` for `prob::std_dev` (cited line in
///   `~/.cargo/registry/.../rs-stats-2.0.3/src/prob/std_dev.rs`).
/// - Numerical proof: `std_dev_population(&[1.0, 2.0, 3.0, 4.0, 5.0])`
///   = `sqrt(10/5)` = `1.4142135623730951` (matches rs-stats exactly).
///
/// # Examples
///
/// ```
/// # use project_orchestrator::analytics::stats::mean_std::std_dev_population;
/// assert_eq!(std_dev_population(&[]), None);
/// assert_eq!(std_dev_population(&[42.0]), Some(0.0));
/// // sqrt(10/5) = sqrt(2)
/// let s = std_dev_population(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// assert!((s - 2.0_f64.sqrt()).abs() < 1e-12);
/// ```
pub fn std_dev_population(values: &[f64]) -> Option<f64> {
    let mu = mean(values)?;
    let n = values.len() as f64;
    let sum_sq = values.iter().map(|v| (v - mu).powi(2)).sum::<f64>();
    Some((sum_sq / n).sqrt())
}

/// Sample (Bessel-corrected) standard deviation: `sqrt(Σ(x_i - μ)² / (n-1))`.
///
/// Uses denominator `n-1`. This matches the **internal** convention used by
/// `rs_stats::hypothesis_tests::t_test::calculate_variance` (a private
/// helper inside rs-stats). t-tests in [`super::t_test`] use this variant
/// to preserve ISO behavior with rs-stats.
///
/// Returns `None` on empty input or `n=1` (denom would be 0). NaN/Inf
/// propagate.
///
/// # References
///
/// - `docs/migration/rs-stats/audit-variance-semantic.md` — cites the
///   exact rs-stats source line `Ok(sum_squared_diff / (n - 1.0))`.
/// - Numerical proof: `std_dev_sample(&[1.0, 2.0, 3.0, 4.0, 5.0])`
///   = `sqrt(10/4)` = `1.5811388300841898` (matches rs-stats t-test
///   internals exactly).
///
/// # Examples
///
/// ```
/// # use project_orchestrator::analytics::stats::mean_std::std_dev_sample;
/// assert_eq!(std_dev_sample(&[]), None);
/// assert_eq!(std_dev_sample(&[42.0]), None); // n-1 = 0
/// // sqrt(10/4) = sqrt(2.5)
/// let s = std_dev_sample(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
/// assert!((s - 2.5_f64.sqrt()).abs() < 1e-12);
/// ```
pub fn std_dev_sample(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mu = mean(values)?;
    let n = values.len() as f64;
    let sum_sq = values.iter().map(|v| (v - mu).powi(2)).sum::<f64>();
    Some((sum_sq / (n - 1.0)).sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::stats::golden_fixtures::*;

    // ─── mean ────────────────────────────────────────────────────────────────

    #[test]
    fn mean_empty_returns_none() {
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn mean_single_element() {
        assert_eq!(mean(&[42.0]), Some(42.0));
    }

    #[test]
    fn mean_all_equal() {
        assert_eq!(mean(&[1.0, 1.0, 1.0]), Some(1.0));
    }

    #[test]
    fn mean_negatives() {
        assert_eq!(mean(&[-1.0, -2.0, -3.0]), Some(-2.0));
    }

    #[test]
    fn mean_nan_propagates() {
        let r = mean(&[1.0, f64::NAN]).unwrap();
        assert!(r.is_nan());
    }

    #[test]
    fn mean_infinity_propagates() {
        let r = mean(&[1.0, f64::INFINITY]).unwrap();
        assert!(r.is_infinite() && r.is_sign_positive());
    }

    // ─── std_dev_population ──────────────────────────────────────────────────

    #[test]
    fn std_dev_population_empty_returns_none() {
        assert_eq!(std_dev_population(&[]), None);
    }

    #[test]
    fn std_dev_population_single_element_returns_zero() {
        assert_eq!(std_dev_population(&[42.0]), Some(0.0));
    }

    #[test]
    fn std_dev_population_all_equal_returns_zero() {
        let r = std_dev_population(&[1.0, 1.0, 1.0]).unwrap();
        assert!(r.abs() < 1e-12);
    }

    #[test]
    fn std_dev_population_nan_propagates() {
        let r = std_dev_population(&[1.0, f64::NAN]).unwrap();
        assert!(r.is_nan());
    }

    // ─── std_dev_sample ──────────────────────────────────────────────────────

    #[test]
    fn std_dev_sample_empty_returns_none() {
        assert_eq!(std_dev_sample(&[]), None);
    }

    #[test]
    fn std_dev_sample_single_returns_none() {
        // Bessel correction → denom = 0 → undefined → None.
        assert_eq!(std_dev_sample(&[42.0]), None);
    }

    #[test]
    fn std_dev_sample_two_elements() {
        // [1, 3]: μ=2, sum_sq=2, sample_var=2/1=2, sample_sd=sqrt(2)
        let r = std_dev_sample(&[1.0, 3.0]).unwrap();
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    // ─── ISO baseline against rs-stats EXPECTED_* values ─────────────────────
    //
    // These tests assert that our first-party impl produces the SAME values
    // as rs-stats did on the golden fixtures, within the tolerance matrix
    // from the plan (1e-12 relative for pure arithmetic).

    #[test]
    fn iso_mean_small_integers() {
        let actual = mean(SMALL_INTEGERS).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_MEAN_SMALL_INTEGERS,
            1e-12,
            "mean(SMALL_INTEGERS)",
        );
    }

    #[test]
    fn iso_mean_medium_normal() {
        let actual = mean(MEDIUM_NORMAL).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_MEAN_MEDIUM_NORMAL,
            1e-12,
            "mean(MEDIUM_NORMAL)",
        );
    }

    #[test]
    fn iso_mean_with_outliers() {
        let actual = mean(WITH_OUTLIERS).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_MEAN_WITH_OUTLIERS,
            1e-12,
            "mean(WITH_OUTLIERS)",
        );
    }

    #[test]
    fn iso_mean_near_zero_flat() {
        let actual = mean(NEAR_ZERO_FLAT).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_MEAN_NEAR_ZERO_FLAT,
            1e-12,
            "mean(NEAR_ZERO_FLAT)",
        );
    }

    #[test]
    fn iso_mean_lognormal_shape() {
        let actual = mean(LOGNORMAL_SHAPE).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_MEAN_LOGNORMAL_SHAPE,
            1e-12,
            "mean(LOGNORMAL_SHAPE)",
        );
    }

    #[test]
    fn iso_std_dev_population_small_integers() {
        let actual = std_dev_population(SMALL_INTEGERS).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_STD_DEV_SMALL_INTEGERS,
            1e-12,
            "std_dev_population(SMALL_INTEGERS)",
        );
    }

    #[test]
    fn iso_std_dev_population_medium_normal() {
        let actual = std_dev_population(MEDIUM_NORMAL).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_STD_DEV_MEDIUM_NORMAL,
            1e-12,
            "std_dev_population(MEDIUM_NORMAL)",
        );
    }

    #[test]
    fn iso_std_dev_population_with_outliers() {
        let actual = std_dev_population(WITH_OUTLIERS).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_STD_DEV_WITH_OUTLIERS,
            1e-12,
            "std_dev_population(WITH_OUTLIERS)",
        );
    }

    #[test]
    fn iso_std_dev_population_near_zero_flat() {
        let actual = std_dev_population(NEAR_ZERO_FLAT).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_STD_DEV_NEAR_ZERO_FLAT,
            1e-12,
            "std_dev_population(NEAR_ZERO_FLAT)",
        );
    }

    #[test]
    fn iso_std_dev_population_lognormal_shape() {
        let actual = std_dev_population(LOGNORMAL_SHAPE).unwrap();
        assert_relative_eq(
            actual,
            EXPECTED_STD_DEV_LOGNORMAL_SHAPE,
            1e-12,
            "std_dev_population(LOGNORMAL_SHAPE)",
        );
    }
}
