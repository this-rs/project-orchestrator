# Audit — rs-stats variance/std_dev convention

## TL;DR

| Function | Denominator | Convention |
|----------|-------------|------------|
| `rs_stats::prob::variance(data)` | `n` | **POPULATION** |
| `rs_stats::prob::std_dev(data)` | `n` (via variance) | **POPULATION** |
| `rs_stats::hypothesis_tests::t_test::calculate_variance(data, mean)` (private helper) | `n - 1` | **SAMPLE** (Bessel-corrected) |

So rs-stats uses **two different conventions** depending on context.

## Source citations

### `prob::variance` (population)

`~/.cargo/registry/.../rs-stats-2.0.3/src/prob/variance.rs`:

```rust
pub fn variance<T>(data: &[T]) -> StatsResult<f64>
where T: ToPrimitive + Debug {
    if data.is_empty() {
        return Err(StatsError::empty_data(...));
    }

    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut n = 0.0;

    for (i, x) in data.iter().enumerate() {
        let value = x.to_f64()...;
        n += 1.0;
        let delta = value - mean;
        mean += delta / n;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    Ok(m2 / n)        // ← POPULATION variance
}
```

Welford's online algorithm. Final division by `n`, not `n - 1`. The unit test inside the file is even named `test_population_std_dev_integers`, confirming intent.

### `prob::std_dev` (population)

`~/.cargo/registry/.../rs-stats-2.0.3/src/prob/std_dev.rs`:

```rust
#[inline]
pub fn std_dev<T>(data: &[T]) -> StatsResult<f64>
where T: ToPrimitive + Debug {
    variance(data).map(|x| x.sqrt())
}
```

Just `sqrt(prob::variance)`, so also POPULATION.

### `t_test::calculate_variance` (sample)

`~/.cargo/registry/.../rs-stats-2.0.3/src/hypothesis_tests/t_test.rs`:

```rust
fn calculate_variance<T>(data: &[T], mean: f64) -> StatsResult<f64>
where T: ToPrimitive + Debug {
    if data.is_empty() {
        return Err(...);
    }
    if data.len() < 2 {
        return Err(...);
    }

    let mut sum_squared_diff = 0.0;
    let n = data.len() as f64;

    for (i, value) in data.iter().enumerate() {
        let v = value.to_f64()...;
        sum_squared_diff += (v - mean).powi(2);
    }

    Ok(sum_squared_diff / (n - 1.0))    // ← SAMPLE variance (Bessel)
}
```

Note this is a **private helper inside t_test.rs**, NOT exposed as `prob::variance`. So it overrides the public population convention specifically for t-tests.

## Numerical proof on `[1.0, 2.0, 3.0, 4.0, 5.0]`

- mean = 3.0
- sum_squared_diff = (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)² = 4 + 1 + 0 + 1 + 4 = **10.0**
- Population variance = 10.0 / 5 = **2.0** → std = **1.4142135623730951**
- Sample variance = 10.0 / 4 = **2.5** → std = **1.5811388300841898**

So:
- `rs_stats::prob::std_dev([1,2,3,4,5])` returns **1.4142135623730951** (population)
- Inside `rs_stats::one_sample_t_test([1,2,3,4,5], …)`, the std_error uses **1.5811388300841898** (sample)

## Implications for our migration

### `crate::analytics::stats::mean_std`

Implement two functions:

```rust
/// Population standard deviation: sqrt(Σ(x-μ)² / n).
/// Matches `rs_stats::prob::std_dev`.
pub fn std_dev_population(values: &[f64]) -> Option<f64>;

/// Sample standard deviation: sqrt(Σ(x-μ)² / (n-1)).  
/// Bessel-corrected. Matches the convention used internally
/// by `rs_stats::hypothesis_tests::t_test`.
pub fn std_dev_sample(values: &[f64]) -> Option<f64>;
```

`distribution.rs::analyze_distribution` uses `std_dev_population` (replaces `rs_stats::prob::std_dev`).

`stats::t_test::*_t_test` use `std_dev_sample` internally (replaces the private rs-stats `calculate_variance`).

This is the single most important detail to get right for ISO behavior — using the wrong one would produce systematically wrong values that no naive testing would catch.
