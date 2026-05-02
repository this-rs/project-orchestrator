# API shapes baseline (pre-migration)

Captures the exact public/serde-exposed shapes that downstream code depends on.
Any field added/removed/renamed/typed differently after migration breaks ISO.

## `analytics::distribution`

### `pub struct DistributionFit`
*(Used: REST API serialization, returned in `DistributionAnalysis.all_fits`)*

| Field | Type | Note |
|-------|------|------|
| `name` | `String` | `"Normal"`, `"LogNormal"`, `"Exponential"`, `"Uniform"`, `"Gamma"`, `"Weibull"`, `"Beta"`, `"StudentT"` |
| `aic` | `f64` | Lower = better |
| `bic` | `f64` | Lower = better |
| `ks_p_value` | `f64` | Higher = better goodness of fit |
| `is_best` | `bool` | True for the entry with lowest AIC |

Derives: `Debug, Clone, Serialize, Deserialize`.

### `pub struct DistributionAnalysis`
*(Used: REST handlers `pagerank_distribution`, `risk_score_distribution`)*

| Field | Type | Source |
|-------|------|--------|
| `count` | `usize` | `values.len()` |
| `mean` | `f64` | rs_stats `prob::average` (or fallback) |
| `std_dev` | `f64` | rs_stats `prob::std_dev` (or 0.0) |
| `min` | `f64` | sorted[0] |
| `max` | `f64` | sorted[n-1] |
| `p50` | `f64` | empirical, linear interp |
| `p75` | `f64` | empirical |
| `p90` | `f64` | empirical |
| `p95` | `f64` | empirical |
| `p99` | `f64` | empirical |
| `skewness` | `f64` | Σ((v-μ)/σ)³ / n |
| `best_fit` | `Option<DistributionFit>` | first of `all_fits` |
| `all_fits` | `Vec<DistributionFit>` | sorted by AIC asc |

Derives: `Debug, Clone, Serialize, Deserialize`.

### Free fns
- `pub fn analyze_distribution(values: &[f64]) -> Option<DistributionAnalysis>`
- `pub fn adaptive_threshold(values: &[f64], percentile: f64, fallback: f64) -> f64`
- `pub fn detect_outliers(values: &[f64], k: f64) -> Vec<usize>`

## `analytics::hypothesis`

### `pub struct AnovaResult`

| Field | Type |
|-------|------|
| `f_statistic` | `f64` |
| `df_between` | `usize` |
| `df_within` | `usize` |
| `p_value` | `f64` |
| `ss_between` | `f64` |
| `ss_within` | `f64` |
| `ms_between` | `f64` |
| `ms_within` | `f64` |
| `significance` | `SignificanceLevel` |

Derives: `Debug, Clone, Serialize, Deserialize`.

### `pub struct TTestResult`

| Field | Type |
|-------|------|
| `t_statistic` | `f64` |
| `degrees_of_freedom` | `f64` |
| `p_value` | `f64` |
| `mean_values` | `Vec<f64>` |
| `std_devs` | `Vec<f64>` |
| `significance` | `SignificanceLevel` |

Derives: `Debug, Clone, Serialize, Deserialize`.

### `pub enum SignificanceLevel`

```rust
#[serde(rename_all = "snake_case")]
pub enum SignificanceLevel {
    HighlySignificant,    // p < 0.001
    Significant,          // p < 0.01
    MarginallySignificant,// p < 0.05
    NotSignificant,       // p >= 0.05
}
```

Derives: `Debug, Clone, Serialize, Deserialize, PartialEq, Eq`.

Method: `pub fn from_p_value(p: f64) -> Self` + `pub fn is_significant(&self) -> bool`.

### Free fns
- `pub fn test_community_homogeneity(groups: &[Vec<f64>]) -> Option<AnovaResult>`
- `pub fn compare_distributions(a: &[f64], b: &[f64], assume_equal_variance: bool) -> Option<TTestResult>`
- `pub fn test_stagnation(values: &[f64], reference_mean: f64) -> Option<TTestResult>`

## ISO contract

Post-migration:
- **No struct field added, removed, renamed, or retyped.**
- All free functions keep the same signatures (params, return types).
- All `#[derive(...)]` lists identical.
- `serde(rename_all = "snake_case")` preserved.
