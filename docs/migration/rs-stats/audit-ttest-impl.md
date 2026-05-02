# Audit — rs-stats t-test implementation

Captured from `~/.cargo/registry/src/.../rs-stats-2.0.3/src/hypothesis_tests/t_test.rs`.

This document is the contract our first-party `src/analytics/stats/t_test.rs` must
honor for ISO migration in plan `00f0ca9a`.

## Variance convention (CRITICAL)

`rs-stats t_test` uses a **private** `calculate_variance` helper (line 370-397):

```rust
fn calculate_variance<T>(data: &[T], mean: f64) -> StatsResult<f64> { ... }
//   sum_squared_diff += (v - mean).powi(2);
//   ...
//   Ok(sum_squared_diff / (n - 1.0))   // ← line 396 — SAMPLE variance (Bessel)
```

So all rs-stats t-tests internally use **sample variance (denom n-1)**, NOT
the public `prob::std_dev` (which uses population denom n). See
`audit-variance-semantic.md` for the full explanation of this dual convention.

Our migration uses `super::mean_std::std_dev_sample` (denom n-1) for t-tests
to preserve ISO behavior.

## one_sample_t_test (lines 71-115)

```
n = data.len()
n < 2                          → Err InvalidInput "Need at least 2 data points"
mean        = Σ x_i / n
variance    = Σ(x_i - mean)² / (n-1)         ← SAMPLE (Bessel)
std_dev     = sqrt(variance)
std_error   = std_dev / sqrt(n)
t_statistic = (mean - pop_mean) / std_error
df          = n - 1
p_value     = calculate_p_value(|t|, df)     ← incomplete_beta — replaced
```

`TTestResult` populated as:
- `mean_values = vec![mean]`
- `std_devs = vec![std_dev]`
- `std_error = std_dev / sqrt(n)`

## two_sample_t_test (lines 147-210)

```
Each side: n_i < 2 → Err InvalidInput
mean_i      = Σ x / n_i
var_i       = Σ(x - mean_i)² / (n_i - 1)     ← SAMPLE (Bessel)
std_dev_i   = sqrt(var_i)
```

### Pooled (Student's, equal_variances = true) — lines 181-186

```
pooled_var  = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
std_error   = sqrt(pooled_var * (1/n1 + 1/n2))
t_statistic = (mean1 - mean2) / std_error
df          = n1 + n2 - 2                    ← INTEGER df
```

### Welch (unequal_variances = false) — lines 187-198

```
var1_n1     = var1 / n1
var2_n2     = var2 / n2
std_error   = sqrt(var1_n1 + var2_n2)
t_statistic = (mean1 - mean2) / std_error

# Welch-Satterthwaite df:
numerator   = (var1_n1 + var2_n2)²
denominator = var1_n1² / (n1 - 1) + var2_n2² / (n2 - 1)
df          = numerator / denominator         ← FRACTIONAL df
```

`TTestResult`:
- `mean_values = vec![mean1, mean2]`
- `std_devs = vec![std_dev1, std_dev2]`

## p-value computation (lines 408-517)

rs-stats implements its own `calculate_p_value`:

1. If `df > 1000.0`: normal approximation `2 * (1 - Φ(z))` via `erf` (line 410-414)
2. Otherwise: incomplete beta `I_a(0.5*df, 0.5)` where `a = df/(df + t²)`, then
   `2 * (1 - I_a)` clamped to `[0, 1]` (line 424-428)

The incomplete_beta is computed via continued fraction (Lentz's method, 200
iterations max) using a Lanczos approximation for ln(Γ).

**Documented precision issues**: comments in `analytics/hypothesis.rs` and the
test outputs from R1 confirm rs-stats often returns 0.0 or 1.0 for p-values
when |t| is very small or very large (incomplete_beta saturation).

### Our replacement

We use `statrs::distribution::StudentsT::new(0.0, 1.0, df)` and:

```rust
let p = 2.0 * (1.0 - dist.cdf(t.abs()));
```

`statrs` is more accurate than rs-stats's hand-rolled incomplete_beta. The plan
allows divergence within 1e-3 absolute tolerance + the `SignificanceLevel`
classification must match rs-stats.

## Edge cases preserved (from baseline-edge-cases.md)

| Input                              | Behavior                             |
|------------------------------------|--------------------------------------|
| `&[]` empty                        | `None`                               |
| `&[42.0]` n=1                      | `None` (rs-stats: `Err InvalidInput`)|
| `&[1.0, 1.0]` zero variance        | `t = NaN/Inf` (divide by zero)       |
| n≥2 normal                         | `Some(TTestResult)`                  |
| Two-sample either side n<2         | `None`                               |

The wrappers in `analytics::hypothesis` (`test_stagnation`,
`compare_distributions`) already guard `if values.len() < 2 { return None; }`,
so the n<2 case never reaches the underlying primitive. This guard is
preserved.

## TTestResult shape

Five fields exposed at the `analytics::hypothesis::TTestResult` level (serde-safe):
`t_statistic`, `degrees_of_freedom`, `p_value`, `mean_values: Vec<f64>`,
`std_devs: Vec<f64>`, plus `significance: SignificanceLevel` derived from p.

rs-stats also exposes `std_error: f64` but we don't expose it (consumers in
`analytics::hypothesis::TTestResult` don't read it). Our internal struct may
or may not carry it — irrelevant for ISO since it's not in the public contract.
