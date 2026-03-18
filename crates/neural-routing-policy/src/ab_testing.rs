//! A/B Testing Framework — split routing with sequential statistical testing.
//!
//! Components:
//! 1. **ABRouter**: routes queries to policy A (control) or B (treatment)
//!    based on a configurable split ratio with deterministic hashing.
//! 2. **SequentialTester**: O'Brien-Fleming sequential testing for early
//!    stopping — enables monitoring experiments at interim analyses without
//!    inflating Type I error.
//! 3. **ABReport**: JSON-serializable experiment report with effect sizes.
//!
//! Thread-safe: uses `RwLock<HashMap>` for concurrent metric updates.

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// A/B experiment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABConfig {
    /// Fraction of traffic assigned to treatment (B). Default: 0.5.
    pub split_ratio: f64,
    /// Significance level (α). Default: 0.05.
    pub significance_level: f64,
    /// Number of planned interim analyses (O'Brien-Fleming). Default: 5.
    pub num_analyses: usize,
    /// Minimum samples per bucket before running a test. Default: 30.
    pub min_samples_per_bucket: usize,
}

impl Default for ABConfig {
    fn default() -> Self {
        Self {
            split_ratio: 0.5,
            significance_level: 0.05,
            num_analyses: 5,
            min_samples_per_bucket: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Bucket & Metrics
// ---------------------------------------------------------------------------

/// Which bucket a query is routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Bucket {
    /// Control: existing policy (A).
    Control,
    /// Treatment: new policy (B).
    Treatment,
}

/// Online metrics for a single bucket using Welford's algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketMetrics {
    /// Number of observations.
    pub count: usize,
    /// Running sum of rewards (for mean).
    pub total_reward: f64,
    /// Running sum of squared deviations (Welford's M2).
    pub m2: f64,
    /// Current running mean.
    pub mean: f64,
}

impl BucketMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self {
            count: 0,
            total_reward: 0.0,
            m2: 0.0,
            mean: 0.0,
        }
    }

    /// Record an observation using Welford's online algorithm.
    pub fn observe(&mut self, reward: f64) {
        self.count += 1;
        self.total_reward += reward;
        let delta = reward - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = reward - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get the sample mean.
    pub fn sample_mean(&self) -> f64 {
        self.mean
    }

    /// Get the sample variance (unbiased, Bessel's correction).
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Get the standard error of the mean.
    pub fn standard_error(&self) -> f64 {
        if self.count < 2 {
            return f64::INFINITY;
        }
        (self.sample_variance() / self.count as f64).sqrt()
    }
}

impl Default for BucketMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ABRouter
// ---------------------------------------------------------------------------

/// A/B Router — assigns queries to buckets and tracks metrics.
///
/// Thread-safe via `RwLock`.
pub struct ABRouter {
    config: ABConfig,
    /// Per-experiment metrics: experiment_name → (control, treatment).
    metrics: RwLock<HashMap<String, (BucketMetrics, BucketMetrics)>>,
    /// Seed for deterministic hashing.
    seed: u64,
}

impl ABRouter {
    /// Create a new A/B router.
    pub fn new(config: ABConfig, seed: u64) -> Self {
        Self {
            config,
            metrics: RwLock::new(HashMap::new()),
            seed,
        }
    }

    /// Route a query to a bucket.
    ///
    /// Assignment is deterministic: the same `key` always maps to the same bucket.
    /// This ensures a user/session consistently sees the same variant.
    pub fn route(&self, key: &str) -> Bucket {
        let hash = self.hash_key(key);
        let fraction = (hash as f64) / (u64::MAX as f64);
        if fraction < self.config.split_ratio {
            Bucket::Treatment
        } else {
            Bucket::Control
        }
    }

    /// Record an observation for a given experiment.
    pub fn observe(&self, experiment: &str, bucket: Bucket, reward: f64) {
        let mut map = self.metrics.write().unwrap();
        let entry = map
            .entry(experiment.to_string())
            .or_insert_with(|| (BucketMetrics::new(), BucketMetrics::new()));
        match bucket {
            Bucket::Control => entry.0.observe(reward),
            Bucket::Treatment => entry.1.observe(reward),
        }
    }

    /// Get metrics for a specific experiment.
    pub fn get_metrics(&self, experiment: &str) -> Option<(BucketMetrics, BucketMetrics)> {
        let map = self.metrics.read().unwrap();
        map.get(experiment).cloned()
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ABConfig {
        &self.config
    }

    /// Generate a full experiment report.
    pub fn report(&self, experiment: &str) -> ABReport {
        let map = self.metrics.read().unwrap();
        let tester = SequentialTester::new(self.config.clone());

        let mut buckets = HashMap::new();

        if let Some((control, treatment)) = map.get(experiment) {
            let min_n = control.count.min(treatment.count);
            let analysis_index = if self.config.num_analyses > 0 {
                // Estimate which analysis we're at based on accumulated samples
                let target_per_analysis =
                    self.config.min_samples_per_bucket * self.config.num_analyses;
                if target_per_analysis > 0 {
                    ((min_n as f64 / target_per_analysis as f64) * self.config.num_analyses as f64)
                        .ceil() as usize
                } else {
                    1
                }
                .clamp(1, self.config.num_analyses)
            } else {
                1
            };

            let test_result = if control.count >= self.config.min_samples_per_bucket
                && treatment.count >= self.config.min_samples_per_bucket
            {
                Some(tester.test(control, treatment, analysis_index))
            } else {
                None
            };

            let effect_size = cohens_d(control, treatment);

            buckets.insert(
                experiment.to_string(),
                BucketPairReport {
                    control: control.clone(),
                    treatment: treatment.clone(),
                    test_result,
                    effect_size,
                },
            );
        }

        let overall_decision = buckets
            .values()
            .next()
            .and_then(|b| b.test_result.as_ref())
            .map(|r| r.decision.clone())
            .unwrap_or(TestDecision::Continue);

        ABReport {
            config: self.config.clone(),
            buckets,
            overall_decision,
        }
    }

    /// Deterministic hash of a key using SplitMix64.
    fn hash_key(&self, key: &str) -> u64 {
        let mut h = self.seed;
        for byte in key.bytes() {
            h = h.wrapping_add(byte as u64);
            h = splitmix(h);
        }
        h
    }
}

/// SplitMix64 hash step.
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------------------
// Sequential Tester (O'Brien-Fleming)
// ---------------------------------------------------------------------------

/// Sequential tester with O'Brien-Fleming spending function.
///
/// Allows early stopping at interim analyses while controlling the overall
/// Type I error rate. Boundaries are most conservative early (requiring very
/// strong evidence) and relax as more data accumulates.
pub struct SequentialTester {
    config: ABConfig,
}

impl SequentialTester {
    /// Create a new sequential tester.
    pub fn new(config: ABConfig) -> Self {
        Self { config }
    }

    /// Compute the O'Brien-Fleming critical boundary at a given analysis.
    ///
    /// At analysis `k` of `K` total:
    /// `z_k = z_{α/2} × √(K / k)`
    ///
    /// Early analyses have very high boundaries (strong evidence needed),
    /// while the final analysis uses the standard z-critical value.
    pub fn boundary(&self, analysis_index: usize) -> f64 {
        let k = analysis_index.max(1) as f64;
        let big_k = self.config.num_analyses.max(1) as f64;
        let z_alpha = inv_normal_cdf(1.0 - self.config.significance_level / 2.0);
        z_alpha * (big_k / k).sqrt()
    }

    /// Perform a two-sample z-test at the given interim analysis.
    pub fn test(
        &self,
        control: &BucketMetrics,
        treatment: &BucketMetrics,
        analysis_index: usize,
    ) -> TestResult {
        let n_c = control.count as f64;
        let n_t = treatment.count as f64;

        // Pooled standard error
        let se = (control.sample_variance() / n_c + treatment.sample_variance() / n_t).sqrt();

        let z_statistic = if se > 1e-12 {
            (treatment.sample_mean() - control.sample_mean()) / se
        } else {
            0.0
        };

        let boundary = self.boundary(analysis_index);
        let p_value = 2.0 * (1.0 - normal_cdf(z_statistic.abs()));

        let decision = if z_statistic.abs() > boundary {
            TestDecision::RejectNull
        } else if analysis_index >= self.config.num_analyses {
            // Final analysis reached without rejection → accept null
            TestDecision::AcceptNull
        } else {
            TestDecision::Continue
        };

        TestResult {
            decision,
            z_statistic,
            boundary,
            p_value,
            analysis_index,
            total_analyses: self.config.num_analyses,
        }
    }
}

// ---------------------------------------------------------------------------
// Test Decision & Results
// ---------------------------------------------------------------------------

/// Decision from a sequential test.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestDecision {
    /// Not enough evidence yet — continue collecting data.
    Continue,
    /// Treatment is significantly different from control.
    RejectNull,
    /// Final analysis reached — no significant difference found.
    AcceptNull,
}

/// Result of a single sequential test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// The test decision.
    pub decision: TestDecision,
    /// Two-sample z-statistic.
    pub z_statistic: f64,
    /// O'Brien-Fleming boundary at this analysis.
    pub boundary: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Current analysis number (1-indexed).
    pub analysis_index: usize,
    /// Total planned analyses.
    pub total_analyses: usize,
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Complete A/B experiment report.
#[derive(Debug, Clone, Serialize)]
pub struct ABReport {
    /// Experiment configuration.
    pub config: ABConfig,
    /// Per-experiment bucket metrics and test results.
    pub buckets: HashMap<String, BucketPairReport>,
    /// Overall decision across all experiments.
    pub overall_decision: TestDecision,
}

/// Report for a single experiment's bucket pair.
#[derive(Debug, Clone, Serialize)]
pub struct BucketPairReport {
    /// Control bucket (A) metrics.
    pub control: BucketMetrics,
    /// Treatment bucket (B) metrics.
    pub treatment: BucketMetrics,
    /// Sequential test result (None if not enough samples).
    pub test_result: Option<TestResult>,
    /// Cohen's d effect size.
    pub effect_size: f64,
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Cohen's d effect size between two groups.
fn cohens_d(control: &BucketMetrics, treatment: &BucketMetrics) -> f64 {
    let n_c = control.count as f64;
    let n_t = treatment.count as f64;
    if n_c < 2.0 || n_t < 2.0 {
        return 0.0;
    }

    // Pooled standard deviation
    let s_c = control.sample_variance();
    let s_t = treatment.sample_variance();
    let pooled_var = ((n_c - 1.0) * s_c + (n_t - 1.0) * s_t) / (n_c + n_t - 2.0);
    let pooled_sd = pooled_var.sqrt();

    if pooled_sd < 1e-12 {
        return 0.0;
    }

    (treatment.sample_mean() - control.sample_mean()) / pooled_sd
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
///
/// Absolute error < 7.5e-8.
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x_abs = x.abs();

    let t = 1.0 / (1.0 + 0.2316419 * x_abs);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let phi = (-x_abs * x_abs / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly =
        0.319381530 * t - 0.356563782 * t2 + 1.781477937 * t3 - 1.821255978 * t4 + 1.330274429 * t5;

    let cdf_abs = 1.0 - phi * poly;

    if sign >= 0.0 {
        cdf_abs
    } else {
        1.0 - cdf_abs
    }
}

/// Inverse standard normal CDF (Peter Acklam's algorithm).
///
/// Accurate to ~1.15e-9 over the full range.
#[allow(clippy::excessive_precision)]
fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BucketMetrics ---

    #[test]
    fn test_bucket_metrics_welford() {
        let mut m = BucketMetrics::new();
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for v in &values {
            m.observe(*v);
        }

        assert_eq!(m.count, 8);
        assert!((m.sample_mean() - 5.0).abs() < 1e-10);
        // Population variance = 4.0, sample variance (Bessel) = 32/7 ≈ 4.571
        assert!(
            (m.sample_variance() - 32.0 / 7.0).abs() < 1e-10,
            "Variance should be 32/7, got {}",
            m.sample_variance()
        );
    }

    #[test]
    fn test_bucket_metrics_single_observation() {
        let mut m = BucketMetrics::new();
        m.observe(42.0);
        assert_eq!(m.count, 1);
        assert!((m.sample_mean() - 42.0).abs() < 1e-10);
        assert!((m.sample_variance()).abs() < 1e-10); // <2 samples → 0
    }

    // --- ABRouter ---

    #[test]
    fn test_router_deterministic() {
        let config = ABConfig::default();
        let router = ABRouter::new(config, 42);

        let b1 = router.route("user-123");
        let b2 = router.route("user-123");
        let b3 = router.route("user-123");

        // Same key → same bucket every time
        assert_eq!(b1, b2);
        assert_eq!(b2, b3);
    }

    #[test]
    fn test_router_split_ratio() {
        let config = ABConfig {
            split_ratio: 0.5,
            ..Default::default()
        };
        let router = ABRouter::new(config, 42);

        let mut treatment_count = 0;
        let total = 1000;
        for i in 0..total {
            if router.route(&format!("user-{i}")) == Bucket::Treatment {
                treatment_count += 1;
            }
        }

        // With 50/50 split, expect ~500 ± 50
        let ratio = treatment_count as f64 / total as f64;
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "50/50 split should give ~50% treatment, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_router_observe_and_report() {
        let config = ABConfig {
            split_ratio: 0.5,
            min_samples_per_bucket: 5,
            num_analyses: 3,
            ..Default::default()
        };
        let router = ABRouter::new(config, 42);

        // Simulate observations
        for i in 0..100 {
            let key = format!("user-{i}");
            let bucket = router.route(&key);
            let reward = match bucket {
                Bucket::Control => 1.0 + (i as f64 * 0.01),
                Bucket::Treatment => 1.5 + (i as f64 * 0.01), // Treatment is better
            };
            router.observe("dt_vs_cql", bucket, reward);
        }

        let report = router.report("dt_vs_cql");
        assert!(!report.buckets.is_empty());

        let pair = report.buckets.get("dt_vs_cql").unwrap();
        assert!(pair.control.count > 0);
        assert!(pair.treatment.count > 0);
        assert!(pair.test_result.is_some());
    }

    // --- SequentialTester ---

    #[test]
    fn test_obrien_fleming_boundaries() {
        let config = ABConfig {
            significance_level: 0.05,
            num_analyses: 5,
            ..Default::default()
        };
        let tester = SequentialTester::new(config);

        let b1 = tester.boundary(1);
        let b2 = tester.boundary(2);
        let b3 = tester.boundary(3);
        let b5 = tester.boundary(5);

        // Boundaries should decrease (more lenient over time)
        assert!(b1 > b2, "b1={b1} should be > b2={b2}");
        assert!(b2 > b3, "b2={b2} should be > b3={b3}");
        assert!(b3 > b5, "b3={b3} should be > b5={b5}");

        // Final boundary should be close to z_{α/2} = 1.96
        assert!(
            (b5 - 1.96).abs() < 0.05,
            "Final boundary should be ~1.96, got {b5}"
        );

        // First boundary should be much larger (~4.38)
        assert!(b1 > 4.0, "First boundary should be >4.0, got {b1}");
    }

    #[test]
    fn test_sequential_test_reject() {
        let config = ABConfig {
            significance_level: 0.05,
            num_analyses: 5,
            min_samples_per_bucket: 10,
            ..Default::default()
        };
        let tester = SequentialTester::new(config);

        // Control: mean=1.0, low variance
        let mut control = BucketMetrics::new();
        for _ in 0..100 {
            control.observe(1.0 + 0.1 * (rand_like_f64() - 0.5));
        }

        // Treatment: mean=2.0, same variance → very strong signal
        let mut treatment = BucketMetrics::new();
        for _ in 0..100 {
            treatment.observe(2.0 + 0.1 * (rand_like_f64() - 0.5));
        }

        let result = tester.test(&control, &treatment, 5);
        assert_eq!(
            result.decision,
            TestDecision::RejectNull,
            "Strong signal should reject null, z={}, boundary={}",
            result.z_statistic,
            result.boundary
        );
    }

    #[test]
    fn test_sequential_test_continue() {
        let config = ABConfig {
            significance_level: 0.05,
            num_analyses: 5,
            min_samples_per_bucket: 10,
            ..Default::default()
        };
        let tester = SequentialTester::new(config);

        // Two identical populations
        let mut control = BucketMetrics::new();
        let mut treatment = BucketMetrics::new();
        for i in 0..50 {
            let v = (i as f64) * 0.1;
            control.observe(v);
            treatment.observe(v + 0.001); // tiny difference
        }

        // At early analysis with tiny difference → should continue
        let result = tester.test(&control, &treatment, 1);
        assert_eq!(
            result.decision,
            TestDecision::Continue,
            "Tiny difference at early analysis should continue"
        );
    }

    #[test]
    fn test_sequential_test_accept_null_final() {
        let config = ABConfig {
            significance_level: 0.05,
            num_analyses: 5,
            min_samples_per_bucket: 10,
            ..Default::default()
        };
        let tester = SequentialTester::new(config);

        // Two identical populations
        let mut control = BucketMetrics::new();
        let mut treatment = BucketMetrics::new();
        for i in 0..100 {
            let v = (i as f64) * 0.1;
            control.observe(v);
            treatment.observe(v);
        }

        // At final analysis with no difference → accept null
        let result = tester.test(&control, &treatment, 5);
        assert_eq!(
            result.decision,
            TestDecision::AcceptNull,
            "No difference at final analysis should accept null"
        );
    }

    // --- Statistical helpers ---

    #[test]
    fn test_normal_cdf() {
        // Φ(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);

        // Φ(1.96) ≈ 0.975
        assert!(
            (normal_cdf(1.96) - 0.975).abs() < 0.001,
            "Φ(1.96) should be ~0.975, got {}",
            normal_cdf(1.96)
        );

        // Φ(-1.96) ≈ 0.025
        assert!(
            (normal_cdf(-1.96) - 0.025).abs() < 0.001,
            "Φ(-1.96) should be ~0.025, got {}",
            normal_cdf(-1.96)
        );

        // Symmetry: Φ(x) + Φ(-x) = 1
        for x in [0.5, 1.0, 2.0, 3.0] {
            let sum = normal_cdf(x) + normal_cdf(-x);
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Φ({x}) + Φ(-{x}) should be 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_inv_normal_cdf() {
        // Φ⁻¹(0.5) = 0
        assert!(
            inv_normal_cdf(0.5).abs() < 1e-6,
            "Φ⁻¹(0.5) should be 0, got {}",
            inv_normal_cdf(0.5)
        );

        // Φ⁻¹(0.975) ≈ 1.96
        assert!(
            (inv_normal_cdf(0.975) - 1.96).abs() < 0.01,
            "Φ⁻¹(0.975) should be ~1.96, got {}",
            inv_normal_cdf(0.975)
        );

        // Φ⁻¹(0.025) ≈ -1.96
        assert!(
            (inv_normal_cdf(0.025) + 1.96).abs() < 0.01,
            "Φ⁻¹(0.025) should be ~-1.96, got {}",
            inv_normal_cdf(0.025)
        );

        // Round-trip: Φ(Φ⁻¹(p)) ≈ p
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let roundtrip = normal_cdf(inv_normal_cdf(p));
            assert!(
                (roundtrip - p).abs() < 1e-6,
                "Round-trip failed for p={p}: got {roundtrip}"
            );
        }
    }

    #[test]
    fn test_cohens_d() {
        let mut control = BucketMetrics::new();
        let mut treatment = BucketMetrics::new();

        // Same distribution → d ≈ 0
        for i in 0..100 {
            let v = i as f64 * 0.1;
            control.observe(v);
            treatment.observe(v);
        }
        let d = cohens_d(&control, &treatment);
        assert!(d.abs() < 0.01, "Same dist → d≈0, got {d}");

        // Shifted by a known amount → d = shift / pooled_sd
        // For uniform(0,1): sd = 1/sqrt(12) ≈ 0.2887
        // Shift by 0.2887 → d ≈ 1.0
        let mut c2 = BucketMetrics::new();
        let mut t2 = BucketMetrics::new();
        let shift = 1.0 / (12.0f64).sqrt(); // exactly 1 SD of uniform(0,1)
        for i in 0..1000 {
            let v = (i as f64) / 1000.0;
            c2.observe(v);
            t2.observe(v + shift);
        }
        let d2 = cohens_d(&c2, &t2);
        assert!((d2 - 1.0).abs() < 0.05, "Shifted by 1 SD → d≈1.0, got {d2}");
    }

    /// Deterministic pseudo-random for testing (not crypto-grade).
    fn rand_like_f64() -> f64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let h = splitmix(n.wrapping_add(0xdeadbeef));
        (h as f64) / (u64::MAX as f64)
    }
}
