//! Distribution analysis helpers for Orchestrator graph metrics.
//!
//! Wraps `rs_stats` fitting API to produce rich distribution reports from raw
//! metric slices (PageRank vectors, risk scores, coupling coefficients, etc.).
//!
//! ## Usage
//! ```rust,ignore
//! use crate::analytics::distribution::{analyze_distribution, adaptive_threshold};
//!
//! let pageranks: Vec<f64> = nodes.iter().map(|n| n.pagerank).collect();
//! let analysis = analyze_distribution(&pageranks);
//! let p95_threshold = adaptive_threshold(&pageranks, 0.95);
//! ```

use rs_stats::distributions::fitting::FitResult;
use serde::{Deserialize, Serialize};
use tracing::warn;

// ─────────────────────────────────────────────────────────────────────────────
// Public output types
// ─────────────────────────────────────────────────────────────────────────────

/// Compact representation of a single distribution fit result for API serialization.
///
/// Mirrors `rs_stats::distributions::fitting::FitResult` but is serde-safe and
/// does not carry the boxed distribution trait object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionFit {
    /// Distribution name (e.g. "LogNormal", "Normal", "Weibull")
    pub name: String,
    /// Akaike Information Criterion — lower = better fit, penalised for complexity
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Kolmogorov-Smirnov p-value — higher = better goodness of fit
    pub ks_p_value: f64,
    /// Whether this distribution is the best fit (lowest AIC)
    pub is_best: bool,
}

impl DistributionFit {
    fn from_fit_result(r: &FitResult, is_best: bool) -> Self {
        Self {
            name: r.name.clone(),
            aic: r.aic,
            bic: r.bic,
            ks_p_value: r.ks_p_value,
            is_best,
        }
    }
}

/// Full statistical analysis of a metric distribution.
///
/// Contains the best-fit distribution model plus key percentiles derived
/// either from the fitted model (when reliable) or directly from sorted data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Number of data points
    pub count: usize,
    /// Arithmetic mean
    pub mean: f64,
    /// Population standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 50th percentile (empirical)
    pub p50: f64,
    /// 75th percentile (empirical)
    pub p75: f64,
    /// 90th percentile (empirical)
    pub p90: f64,
    /// 95th percentile (empirical) — use as adaptive threshold for "critical" tier
    pub p95: f64,
    /// 99th percentile (empirical)
    pub p99: f64,
    /// Fisher skewness: > 0 = right-tail (power-law), < 0 = left-tail
    pub skewness: f64,
    /// Best-fit distribution (lowest AIC among all candidates)
    pub best_fit: Option<DistributionFit>,
    /// All fitted distributions ranked by AIC
    pub all_fits: Vec<DistributionFit>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core functions
// ─────────────────────────────────────────────────────────────────────────────

/// Analyse the full distribution of `values`, fitting all applicable distributions.
///
/// Returns `None` if `values` is empty.
/// All rs-stats errors are caught and logged; the percentile fields still
/// reflect the empirical distribution even if model fitting fails.
pub fn analyze_distribution(values: &[f64]) -> Option<DistributionAnalysis> {
    if values.is_empty() {
        return None;
    }

    let n = values.len();
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // ── Basic statistics — delegate to rs-stats rather than reimplementing ────
    let mean =
        rs_stats::prob::average(values).unwrap_or_else(|_| values.iter().sum::<f64>() / n as f64);
    let std_dev = rs_stats::prob::std_dev(values).unwrap_or(0.0);

    let skewness = if std_dev > 0.0 && n >= 3 {
        sorted
            .iter()
            .map(|v| ((v - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n as f64
    } else {
        0.0
    };

    // ── Empirical percentiles (linear interpolation) ──────────────────────────
    let empirical = |p: f64| -> f64 {
        if n == 1 {
            return sorted[0];
        }
        let idx = p * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = (idx.ceil() as usize).min(n - 1);
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    };

    // ── Distribution fitting via rs-stats ─────────────────────────────────────
    let fit_results: Vec<FitResult> = rs_stats::fit_all(values).unwrap_or_else(|e| {
        warn!(
            "rs-stats fit_all failed (n={}): {} — using empty fit list",
            n, e
        );
        vec![]
    });

    let all_fits: Vec<DistributionFit> = fit_results
        .iter()
        .enumerate()
        .map(|(i, r)| DistributionFit::from_fit_result(r, i == 0))
        .collect();

    let best_fit = all_fits.first().cloned();

    Some(DistributionAnalysis {
        count: n,
        mean,
        std_dev,
        min: sorted[0],
        max: sorted[n - 1],
        p50: empirical(0.50),
        p75: empirical(0.75),
        p90: empirical(0.90),
        p95: empirical(0.95),
        p99: empirical(0.99),
        skewness,
        best_fit,
        all_fits,
    })
}

/// Compute an adaptive threshold at the given percentile from actual data.
///
/// This replaces hardcoded thresholds (e.g. `risk_score >= 0.75`) with a
/// percentile derived from the real distribution, adapting to each project's
/// unique characteristics.
///
/// Returns `fallback` if `values` is empty or the percentile computation fails.
///
/// # Arguments
/// * `values`     — metric values (e.g. all `risk_score` in a project)
/// * `percentile` — desired percentile as a fraction in `[0.0, 1.0]`
/// * `fallback`   — value to return when data is unavailable
///
/// # Example
/// ```rust,ignore
/// // Replace the hardcoded 0.75 "critical" threshold for risk_score
/// let critical_threshold = adaptive_threshold(&risk_scores, 0.95, 0.75);
/// ```
pub fn adaptive_threshold(values: &[f64], percentile: f64, fallback: f64) -> f64 {
    if values.is_empty() {
        return fallback;
    }
    let p = percentile.clamp(0.0, 1.0);
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (idx.ceil() as usize).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Identify statistical outliers using the IQR fence method.
///
/// An outlier is any value outside `[Q1 - k*IQR, Q3 + k*IQR]` where
/// `k = 1.5` (Tukey mild outlier) or `k = 3.0` (Tukey extreme outlier).
///
/// Returns the indices (into `values`) of detected outliers sorted by
/// distance from the fence (most extreme first).
pub fn detect_outliers(values: &[f64], k: f64) -> Vec<usize> {
    if values.len() < 4 {
        return vec![];
    }
    let mut sorted_vals = values.to_vec();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted_vals.len();

    let q1 = {
        let idx = 0.25 * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        sorted_vals[lo] * (1.0 - (idx - lo as f64)) + sorted_vals[hi] * (idx - lo as f64)
    };
    let q3 = {
        let idx = 0.75 * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        sorted_vals[lo] * (1.0 - (idx - lo as f64)) + sorted_vals[hi] * (idx - lo as f64)
    };

    let iqr = q3 - q1;
    let lower = q1 - k * iqr;
    let upper = q3 + k * iqr;

    let mut outliers: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter(|(_, &v)| v < lower || v > upper)
        .map(|(i, &v)| {
            let dist = if v < lower { lower - v } else { v - upper };
            (i, dist)
        })
        .collect();

    outliers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    outliers.into_iter().map(|(i, _)| i).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_distribution_empty() {
        assert!(analyze_distribution(&[]).is_none());
    }

    #[test]
    fn test_analyze_distribution_single() {
        let result = analyze_distribution(&[42.0]).unwrap();
        assert_eq!(result.count, 1);
        assert_eq!(result.mean, 42.0);
        assert_eq!(result.p95, 42.0);
    }

    #[test]
    fn test_adaptive_threshold_fallback() {
        assert_eq!(adaptive_threshold(&[], 0.95, 0.75), 0.75);
    }

    #[test]
    fn test_adaptive_threshold_basic() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p95 = adaptive_threshold(&values, 0.95, 0.0);
        assert!((p95 - 95.05).abs() < 1.0, "p95 should be ~95, got {}", p95);
    }

    #[test]
    fn test_detect_outliers() {
        let mut values: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        values.push(1000.0); // clear outlier
        let outliers = detect_outliers(&values, 1.5);
        assert!(
            outliers.contains(&20),
            "1000.0 should be detected as outlier"
        );
    }
}
