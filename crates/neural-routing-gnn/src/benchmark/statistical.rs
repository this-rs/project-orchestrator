//! Statistical testing: Bootstrap CI & paired permutation test — pure Rust.

use serde::{Deserialize, Serialize};

/// Bootstrap confidence interval result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCI {
    pub lower: f64,
    pub upper: f64,
    pub mean: f64,
    pub std_err: f64,
}

/// Result of a comparison between two embedding sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub metric_name: String,
    pub source_a: String,
    pub source_b: String,
    pub value_a: f64,
    pub value_b: f64,
    pub diff: f64,
    pub diff_ci_95: BootstrapCI,
    pub p_value: f64,
    pub significant: bool,
}

// ---------------------------------------------------------------------------
// Simple deterministic RNG (LCG, no rand dependency)
// ---------------------------------------------------------------------------

pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() % max as u64) as usize
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    #[allow(dead_code)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Bootstrap CI
// ---------------------------------------------------------------------------

/// Compute bootstrap 95% confidence interval for a metric difference.
///
/// * `values_a` — metric values from source A (one per query)
/// * `values_b` — metric values from source B (one per query)
/// * `resamples` — number of bootstrap resamples (typically 1000)
/// * `seed` — RNG seed for reproducibility
///
/// Returns CI for (mean_a - mean_b).
pub fn bootstrap_ci(
    values_a: &[f64],
    values_b: &[f64],
    resamples: usize,
    seed: u64,
) -> BootstrapCI {
    let n = values_a.len().min(values_b.len());
    if n == 0 {
        return BootstrapCI {
            lower: 0.0,
            upper: 0.0,
            mean: 0.0,
            std_err: 0.0,
        };
    }

    let mut rng = SimpleRng::new(seed);
    let mut diffs = Vec::with_capacity(resamples);

    for _ in 0..resamples {
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;

        for _ in 0..n {
            let idx = rng.next_usize(n);
            sum_a += values_a[idx];
            sum_b += values_b[idx];
        }

        diffs.push(sum_a / n as f64 - sum_b / n as f64);
    }

    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = diffs.iter().sum::<f64>() / resamples as f64;
    let variance = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / resamples as f64;
    let std_err = variance.sqrt();

    let lo = (resamples as f64 * 0.025) as usize;
    let hi = (resamples as f64 * 0.975).min((resamples - 1) as f64) as usize;

    BootstrapCI {
        lower: diffs[lo],
        upper: diffs[hi],
        mean,
        std_err,
    }
}

// ---------------------------------------------------------------------------
// Paired permutation test
// ---------------------------------------------------------------------------

/// Paired permutation test for significance.
///
/// Tests H0: the mean difference between paired samples is zero.
///
/// * `values_a` — metric values from source A
/// * `values_b` — metric values from source B
/// * `num_permutations` — number of permutations (typically 10_000)
/// * `seed` — RNG seed
///
/// Returns p-value (two-tailed).
pub fn paired_permutation_test(
    values_a: &[f64],
    values_b: &[f64],
    num_permutations: usize,
    seed: u64,
) -> f64 {
    let n = values_a.len().min(values_b.len());
    if n == 0 {
        return 1.0;
    }

    // Compute observed difference
    let diffs: Vec<f64> = values_a.iter().zip(values_b).map(|(a, b)| a - b).collect();
    let observed = diffs.iter().sum::<f64>() / n as f64;

    let mut rng = SimpleRng::new(seed);
    let mut count_extreme = 0usize;

    for _ in 0..num_permutations {
        let mut perm_sum = 0.0;
        for &d in &diffs {
            // Randomly flip the sign
            let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            perm_sum += d * sign;
        }
        let perm_mean = perm_sum / n as f64;

        if perm_mean.abs() >= observed.abs() {
            count_extreme += 1;
        }
    }

    count_extreme as f64 / num_permutations as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_ci_positive_diff() {
        let a: Vec<f64> = vec![0.9, 0.8, 0.85, 0.95, 0.88, 0.92, 0.87, 0.91, 0.86, 0.93];
        let b: Vec<f64> = vec![0.5, 0.4, 0.45, 0.55, 0.48, 0.52, 0.47, 0.51, 0.46, 0.53];

        let ci = bootstrap_ci(&a, &b, 1000, 42);
        assert!(ci.lower > 0.2, "Lower CI should be > 0.2, got {}", ci.lower);
        assert!(ci.upper < 0.6, "Upper CI should be < 0.6, got {}", ci.upper);
        assert!(ci.mean > 0.3, "Mean diff should be > 0.3, got {}", ci.mean);
    }

    #[test]
    fn test_bootstrap_ci_no_diff() {
        let a: Vec<f64> = vec![0.5, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50];
        let b: Vec<f64> = vec![0.5, 0.49, 0.51, 0.50, 0.48, 0.52, 0.50, 0.49, 0.51, 0.50];

        let ci = bootstrap_ci(&a, &b, 1000, 42);
        // With no real difference, CI should straddle zero
        assert!(ci.lower < 0.02);
        assert!(ci.upper > -0.02);
    }

    #[test]
    fn test_permutation_test_significant() {
        let a: Vec<f64> = vec![0.9, 0.8, 0.85, 0.95, 0.88, 0.92, 0.87, 0.91, 0.86, 0.93];
        let b: Vec<f64> = vec![0.5, 0.4, 0.45, 0.55, 0.48, 0.52, 0.47, 0.51, 0.46, 0.53];

        let p = paired_permutation_test(&a, &b, 5000, 42);
        assert!(p < 0.05, "Should be significant (p < 0.05), got {}", p);
    }

    #[test]
    fn test_permutation_test_not_significant() {
        let a: Vec<f64> = vec![0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50];
        let b: Vec<f64> = vec![0.50, 0.49, 0.51, 0.50, 0.48, 0.52, 0.50, 0.49, 0.51, 0.50];

        let p = paired_permutation_test(&a, &b, 5000, 42);
        assert!(p > 0.05, "Should not be significant (p > 0.05), got {}", p);
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
