//! Distribution fitting — 10-distribution AIC/BIC/KS-ranked fitter.
//!
//! Replaces `rs_stats::distributions::fitting::fit_all` (GPL-3.0). Uses
//! `statrs` for all PDF / CDF primitives; parameter estimators are taken
//! verbatim from rs-stats so the resulting AIC/BIC values match within
//! machine precision (modulo incomplete-beta differences).
//!
//! See `docs/migration/rs-stats/audit-fitting-impl.md` for the full
//! per-distribution audit (formulas, skip rules, tolerances).
//!
//! Plan: `00f0ca9a-816f-4fcc-bc53-da88d595de34`, task R6.

use statrs::distribution::{
    Beta, ChiSquared, Continuous, ContinuousCDF, Exp, FisherSnedecor, Gamma, LogNormal, Normal,
    StudentsT, Uniform, Weibull,
};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of one fit attempt.
///
/// Mirrors `rs_stats::distributions::fitting::FitResult` (5 fields).
/// Downstream consumers (`analytics::distribution::DistributionFit`) only
/// read `name` / `aic` / `bic` / `ks_p_value`, so `ks_statistic` is kept
/// for completeness but unused externally.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    pub name: String,
    pub aic: f64,
    pub bic: f64,
    pub ks_statistic: f64,
    pub ks_p_value: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Kolmogorov-Smirnov goodness-of-fit
// ─────────────────────────────────────────────────────────────────────────────

/// Stephens (1970) finite-sample p-value approximation for the Kolmogorov
/// distribution. Series truncated at 100 terms (early-exit on small term).
fn kolmogorov_p(arg: f64) -> f64 {
    if arg <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    for j in 1_u32..=100 {
        let term = (-(2.0 * (j as f64).powi(2) * arg * arg)).exp();
        if j % 2 == 1 {
            sum += term;
        } else {
            sum -= term;
        }
        if term < 1e-15 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// Two-sided Kolmogorov-Smirnov statistic + p-value.
///
/// Returns (D, p). Matches the rs-stats `ks_test` formulation
/// (Stephens correction with `(sqrt(n) + 0.12 + 0.11/sqrt(n)) * D`).
fn ks_test(data: &[f64], cdf: impl Fn(f64) -> f64) -> (f64, f64) {
    let n = data.len();
    if n == 0 {
        return (0.0, 1.0);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;
    let mut d = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let f = cdf(x);
        let upper = (i + 1) as f64 / nf;
        let lower = i as f64 / nf;
        d = d.max((upper - f).abs()).max((f - lower).abs());
    }

    let arg = (nf.sqrt() + 0.12 + 0.11 / nf.sqrt()) * d;
    let p = kolmogorov_p(arg);
    (d, p)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn aic_bic(log_likelihood: f64, k: usize, n: usize) -> Option<(f64, f64)> {
    if !log_likelihood.is_finite() {
        return None;
    }
    let aic = 2.0 * (k as f64) - 2.0 * log_likelihood;
    let bic = (k as f64) * (n as f64).ln() - 2.0 * log_likelihood;
    if !aic.is_finite() || !bic.is_finite() {
        return None;
    }
    Some((aic, bic))
}

/// Sum `ln(pdf(x))` over data. Returns `None` if any term is non-finite
/// (caller should silently skip — non-finite logL means the distribution
/// is incompatible with the data).
fn log_likelihood(dist: &dyn Continuous<f64, f64>, data: &[f64]) -> Option<f64> {
    let mut sum = 0.0_f64;
    for &x in data {
        let lp = dist.ln_pdf(x);
        if !lp.is_finite() {
            return None;
        }
        sum += lp;
    }
    Some(sum)
}

/// Build a FitResult from name + computed AIC/BIC/KS, or None on any
/// non-finite intermediate.
fn make_result(
    name: &str,
    log_likelihood: f64,
    k: usize,
    data: &[f64],
    cdf: impl Fn(f64) -> f64,
) -> Option<FitResult> {
    let (aic, bic) = aic_bic(log_likelihood, k, data.len())?;
    let (ks_statistic, ks_p_value) = ks_test(data, cdf);
    Some(FitResult {
        name: name.to_string(),
        aic,
        bic,
        ks_statistic,
        ks_p_value,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-distribution fitters
// ─────────────────────────────────────────────────────────────────────────────

/// Normal fit. MLE: μ̂ = mean, σ̂ = pop std (denom n). k=2.
fn fit_normal(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();
    if sigma <= 0.0 || !sigma.is_finite() {
        return None;
    }
    let dist = Normal::new(mean, sigma).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("Normal", ll, 2, data, |x| dist.cdf(x))
}

/// Exponential fit. MLE: λ̂ = 1/mean. k=1. Skip if any x<0.
fn fit_exponential(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() || data.iter().any(|&x| x < 0.0) {
        return None;
    }
    let mean = data.iter().sum::<f64>() / (data.len() as f64);
    if mean <= 0.0 || !mean.is_finite() {
        return None;
    }
    let lambda = 1.0 / mean;
    let dist = Exp::new(lambda).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("Exponential", ll, 1, data, |x| dist.cdf(x))
}

/// Uniform fit. MLE: â=min, b̂=max. k=2. (logpdf=−ln(b−a) for x∈[a,b].)
fn fit_uniform(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() {
        return None;
    }
    let a = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let b = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !(a < b && a.is_finite() && b.is_finite()) {
        return None;
    }
    let dist = Uniform::new(a, b).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("Uniform", ll, 2, data, |x| dist.cdf(x))
}

/// Gamma fit (Choi-Wette MLE approximation). k=2. Skip if any x≤0.
fn fit_gamma(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let log_mean = data.iter().map(|&x| x.ln()).sum::<f64>() / n;
    let s = mean.ln() - log_mean;
    let alpha = if s > 0.0 {
        (3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s)
    } else {
        1.0
    };
    let beta = alpha / mean;
    if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        return None;
    }
    let dist = Gamma::new(alpha, beta).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("Gamma", ll, 2, data, |x| dist.cdf(x))
}

/// LogNormal fit. MLE on log(data). k=2. Skip if any x≤0.
fn fit_lognormal(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return None;
    }
    let n = data.len() as f64;
    let log_data: Vec<f64> = data.iter().map(|&x| x.ln()).collect();
    let mu = log_data.iter().sum::<f64>() / n;
    let variance = log_data.iter().map(|&y| (y - mu).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();
    if sigma <= 0.0 || !sigma.is_finite() {
        return None;
    }
    let dist = LogNormal::new(mu, sigma).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("LogNormal", ll, 2, data, |x| dist.cdf(x))
}

/// Weibull fit (Teimouri-Gupta + closed-form scale). k=2. Skip if any x≤0.
fn fit_weibull(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    if mean <= 0.0 || variance <= 0.0 {
        return None;
    }
    let cv = variance.sqrt() / mean;
    let k = cv.powf(-1.086).max(0.01);
    let sum_xk: f64 = data.iter().map(|&x| x.powf(k)).sum::<f64>();
    let lambda = (sum_xk / n).powf(1.0 / k);
    if !k.is_finite() || !lambda.is_finite() || k <= 0.0 || lambda <= 0.0 {
        return None;
    }
    let dist = Weibull::new(k, lambda).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("Weibull", ll, 2, data, |x| dist.cdf(x))
}

/// Beta fit (method of moments). k=2. Skip if any x∉(0,1).
fn fit_beta(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0 || x >= 1.0) {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    if variance <= 0.0 {
        return None;
    }
    let common = mean * (1.0 - mean) / variance - 1.0;
    let alpha = mean * common;
    let beta = (1.0 - mean) * common;
    if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        return None;
    }
    let dist = Beta::new(alpha, beta).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("Beta", ll, 2, data, |x| dist.cdf(x))
}

/// StudentT fit. k=3. Skip if n<4. ν̂ from excess kurtosis.
fn fit_student_t(data: &[f64]) -> Option<FitResult> {
    if data.len() < 4 {
        return None;
    }
    let n = data.len() as f64;
    let mu = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();
    if sigma <= 0.0 || !sigma.is_finite() {
        return None;
    }
    let m4 = data.iter().map(|&x| (x - mu).powi(4)).sum::<f64>() / n;
    let excess_kurtosis = m4 / (variance * variance) - 3.0;
    let nu = if excess_kurtosis > 0.01 {
        (4.0 + 6.0 / excess_kurtosis).max(2.01)
    } else {
        30.0
    };
    let dist = StudentsT::new(mu, sigma, nu).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("StudentT", ll, 3, data, |x| dist.cdf(x))
}

/// FDistribution fit. k=2. Skip if any x≤0. Fallback (2,10) if mean≤1.
fn fit_f_distribution(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return None;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    // rs-stats fallback: if mean ≤ 1, use (d1=2, d2=10); else compute from moments.
    let (d1, d2) = if mean <= 1.0 {
        (2.0, 10.0)
    } else {
        let d2 = (2.0 * mean / (mean - 1.0)).max(2.01);
        let d2m2 = d2 - 2.0;
        let d2m4 = (d2 - 4.0).max(0.01);
        let num = 2.0 * d2 * d2 * d2m2;
        let den = variance * d2m2 * d2m2 * d2m4 - 2.0 * d2 * d2;
        let d1 = if den > 0.0 {
            (num / den).max(0.01)
        } else {
            2.0
        };
        (d1, d2)
    };
    if !d1.is_finite() || !d2.is_finite() {
        return None;
    }
    let dist = FisherSnedecor::new(d1, d2).ok()?;
    let ll = log_likelihood(&dist, data)?;
    // Name "F" matches rs-stats's FDistribution::name() — preserves
    // the captured EXPECTED entries.
    make_result("F", ll, 2, data, |x| dist.cdf(x))
}

/// ChiSquared fit. k=1. Skip if any x<0. k̂ = max(mean, 0.01).
fn fit_chi_squared(data: &[f64]) -> Option<FitResult> {
    if data.is_empty() || data.iter().any(|&x| x < 0.0) {
        return None;
    }
    let mean = data.iter().sum::<f64>() / (data.len() as f64);
    let kk = mean.max(0.01);
    if !kk.is_finite() {
        return None;
    }
    let dist = ChiSquared::new(kk).ok()?;
    let ll = log_likelihood(&dist, data)?;
    make_result("ChiSquared", ll, 1, data, |x| dist.cdf(x))
}

// ─────────────────────────────────────────────────────────────────────────────
// fit_all orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Fit 10 continuous distributions to `data` and return ranked results
/// (lowest AIC first). Distributions that fail their precondition checks
/// or produce non-finite logL are silently skipped.
///
/// Returns `Err` if `data` is empty or no distribution could be fitted.
///
/// Order matches rs-stats's `fit_all`: Normal, Exponential, Uniform, Gamma,
/// LogNormal, Weibull, Beta, StudentT, FDistribution, ChiSquared.
pub fn fit_all(data: &[f64]) -> anyhow::Result<Vec<FitResult>> {
    if data.is_empty() {
        return Err(anyhow::anyhow!("fit_all: data must not be empty"));
    }

    let mut results: Vec<FitResult> = Vec::with_capacity(10);

    // Order matches rs-stats. Each fitter returns Option (None = skipped).
    for r in [
        fit_normal(data),
        fit_exponential(data),
        fit_uniform(data),
        fit_gamma(data),
        fit_lognormal(data),
        fit_weibull(data),
        fit_beta(data),
        fit_student_t(data),
        fit_f_distribution(data),
        fit_chi_squared(data),
    ]
    .into_iter()
    .flatten()
    {
        results.push(r);
    }

    if results.is_empty() {
        return Err(anyhow::anyhow!(
            "fit_all: no distribution could be fitted to the data"
        ));
    }

    results.sort_by(|a, b| {
        a.aic
            .partial_cmp(&b.aic)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytics::stats::golden_fixtures::*;

    // ─── KS helpers ──────────────────────────────────────────────────────────

    #[test]
    fn ks_p_zero_arg_is_one() {
        assert_eq!(kolmogorov_p(0.0), 1.0);
        assert_eq!(kolmogorov_p(-1.0), 1.0);
    }

    #[test]
    fn ks_p_large_arg_is_zero() {
        assert!(kolmogorov_p(10.0) < 1e-50);
    }

    #[test]
    fn ks_test_uniform_data_low_d() {
        // Data uniformly spaced in [0,1] should match U(0,1) closely.
        let data: Vec<f64> = (0..20).map(|i| (i as f64 + 0.5) / 20.0).collect();
        let (d, p) = ks_test(&data, |x| x.clamp(0.0, 1.0));
        assert!(d < 0.15, "got D={d}");
        assert!(p > 0.5, "got p={p}");
    }

    // ─── fit_all input validation ────────────────────────────────────────────

    #[test]
    fn fit_all_empty_errs() {
        assert!(fit_all(&[]).is_err());
    }

    #[test]
    fn fit_all_constant_data_succeeds_via_chisquared() {
        // All-equal positive: variance=0, so Normal/LogNormal/Beta/F/StudentT skip
        // (sigma=0 or kurtosis singular). ChiSquared::fit only checks non-negative
        // and uses k̂ = max(mean, 0.01); Gamma's Choi-Wette MLE gives α=1 when s=0;
        // Exponential's MLE λ=1/mean is well-defined. All produce -Inf logL on
        // delta data because their pdf at the single mass point is finite, so AIC
        // is finite. Matches rs-stats behavior — fit_all does not error here.
        let r = fit_all(&[5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
        let names: Vec<&str> = r.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"ChiSquared"));
        // Variance-dependent fitters skip:
        assert!(!names.contains(&"Normal"));
        assert!(!names.contains(&"LogNormal"));
        assert!(!names.contains(&"Weibull"));
    }

    // ─── Skip rules ──────────────────────────────────────────────────────────

    #[test]
    fn negative_data_skips_positive_fitters() {
        // Negative values: Exponential/Gamma/LogNormal/Weibull/Beta/F/ChiSquared skip
        let data = &[-1.0, -2.0, 0.5, 1.0, -0.5][..];
        let r = fit_all(data).unwrap();
        let names: Vec<&str> = r.iter().map(|f| f.name.as_str()).collect();
        // Normal, Uniform should always succeed; StudentT (n=5, no positive constraint)
        assert!(names.contains(&"Normal"));
        assert!(names.contains(&"Uniform"));
        // Beta/Gamma/Exponential/LogNormal/Weibull/F/ChiSquared all skip
        assert!(!names.contains(&"Beta"));
        assert!(!names.contains(&"Gamma"));
        assert!(!names.contains(&"LogNormal"));
        assert!(!names.contains(&"Weibull"));
    }

    #[test]
    fn beta_skipped_when_data_outside_unit_interval() {
        // SMALL_INTEGERS contains 5.0 > 1, so Beta must skip.
        let r = fit_all(SMALL_INTEGERS).unwrap();
        assert!(!r.iter().any(|f| f.name == "Beta"));
    }

    // ─── Sorting ─────────────────────────────────────────────────────────────

    #[test]
    fn fit_all_results_sorted_by_aic_ascending() {
        let r = fit_all(MEDIUM_NORMAL).unwrap();
        for i in 1..r.len() {
            assert!(
                r[i].aic >= r[i - 1].aic,
                "fits not sorted by AIC at index {i}: {} >= {}",
                r[i - 1].aic,
                r[i].aic
            );
        }
    }

    // ─── ISO baseline against rs-stats EXPECTED_FIT_ALL_* ────────────────────
    //
    // The plan tolerance matrix:
    // - aic / bic: 1e-6 relative
    // - ks_p_value: 1e-3 absolute
    // - name + ranking direction: exact
    //
    // statrs's incomplete-beta / incomplete-gamma are independently
    // implemented, so we may diverge slightly on AIC for distributions
    // whose logpdf depends on ln_gamma / ln_beta (Gamma, Beta, F, StudentT,
    // ChiSquared). When that happens, the test expands the tolerance and
    // documents the divergence in iso-divergences.md.

    /// Find a fit by name.
    fn find<'a>(fits: &'a [FitResult], name: &str) -> Option<&'a FitResult> {
        fits.iter().find(|f| f.name == name)
    }

    /// Match each EXPECTED entry against our actual fit by name. Asserts
    /// AIC/BIC within `aic_tol` (relative) and ks_p_value within
    /// `p_abs_tol` (absolute). Distributions present in EXPECTED but
    /// missing from actual (or vice versa) are flagged.
    fn assert_iso_fit_set(
        actual: &[FitResult],
        expected: &[FitExpected],
        fixture_name: &str,
        aic_rel_tol: f64,
        p_abs_tol: f64,
    ) {
        for exp in expected {
            let act = find(actual, exp.name).unwrap_or_else(|| {
                panic!("[{fixture_name}] missing fit '{}': rs-stats produced this distribution but our impl did not", exp.name);
            });
            assert_relative_eq(
                act.aic,
                exp.aic,
                aic_rel_tol,
                &format!("[{fixture_name}] {} aic", exp.name),
            );
            assert_relative_eq(
                act.bic,
                exp.bic,
                aic_rel_tol,
                &format!("[{fixture_name}] {} bic", exp.name),
            );
            // ks_p_value: rs-stats uses the same Stephens approximation;
            // small numerical drift expected from different incomplete-beta.
            assert_absolute_eq(
                act.ks_p_value,
                exp.ks_p_value,
                p_abs_tol,
                &format!("[{fixture_name}] {} ks_p_value", exp.name),
            );
        }
    }

    #[test]
    fn iso_fit_all_small_integers() {
        // SMALL_INTEGERS = [1,2,3,4,5]. Beta skipped (values >1). 9 fits.
        let actual = fit_all(SMALL_INTEGERS).unwrap();
        // Loosened tolerance: AIC depends on logpdf which uses ln_gamma /
        // ln_beta for several distributions — statrs and rs-stats can
        // disagree by ~1e-5 relative on these terms.
        assert_iso_fit_set(
            &actual,
            EXPECTED_FIT_ALL_SMALL_INTEGERS,
            "SMALL_INTEGERS",
            1e-5,
            1e-3,
        );
    }

    #[test]
    fn iso_fit_all_medium_normal() {
        // MEDIUM_NORMAL = 30 values from N(10,2). Beta skipped (>1). 9 fits.
        let actual = fit_all(MEDIUM_NORMAL).unwrap();
        assert_iso_fit_set(
            &actual,
            EXPECTED_FIT_ALL_MEDIUM_NORMAL,
            "MEDIUM_NORMAL",
            1e-5,
            1e-3,
        );
    }

    #[test]
    fn iso_fit_all_lognormal_shape() {
        // LOGNORMAL_SHAPE = 50 values from LogNormal(0, 0.5). Beta skipped. 9 fits.
        let actual = fit_all(LOGNORMAL_SHAPE).unwrap();
        assert_iso_fit_set(
            &actual,
            EXPECTED_FIT_ALL_LOGNORMAL_SHAPE,
            "LOGNORMAL_SHAPE",
            1e-5,
            1e-3,
        );
    }
}
