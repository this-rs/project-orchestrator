# Edge case behavior of rs-stats (pre-migration baseline)

Captured by inspecting rs-stats source + running mini probes.
Our first-party impl must match these on the same inputs.

## `prob::average(data)`

| Input | Behavior |
|-------|----------|
| `&[]` (empty) | `Err(StatsError::EmptyData)` |
| `&[42.0]` (n=1) | `Ok(42.0)` |
| `&[1.0, 1.0, 1.0]` (all-equal) | `Ok(1.0)` |
| `&[1.0, f64::NAN]` (NaN included) | `Ok(NaN)` (NaN propagates through Σ/n) |
| `&[1.0, f64::INFINITY]` (Inf included) | `Ok(Inf)` |
| `&[-1.0, -2.0]` (negatives) | `Ok(-1.5)` |

Our `mean(values)` returns `Option<f64>`: empty → `None`, otherwise computes Σ/n unchecked (NaN/Inf propagate).

## `prob::std_dev(data)` (population)

| Input | Behavior |
|-------|----------|
| `&[]` (empty) | `Err(StatsError::EmptyData)` |
| `&[42.0]` (n=1) | `Ok(0.0)` (m2 / 1 = 0; sqrt(0) = 0) |
| `&[1.0, 1.0, 1.0]` (all-equal) | `Ok(0.0)` |
| `&[1.0, f64::NAN]` | `Ok(NaN)` |
| Negatives | normal computation |

Our `std_dev_population(values)` returns `Option<f64>`: empty → `None`, n=1 → `Some(0.0)`, otherwise sqrt(m2/n).

## `one_sample_t_test(data, ref_mean)`

| Input | Behavior |
|-------|----------|
| `&[]` | `Err(StatsError::EmptyData)` |
| `&[42.0]` (n=1) | `Err(InvalidInput "Need at least 2 data points")` |
| `&[1.0, 1.0]` (zero variance) | `t_stat = NaN/Inf` (divide by zero) |
| n≥2, normal data | `Ok(TTestResult)` |

Our wrapper `analytics::hypothesis::test_stagnation` already guards `if values.len() < 2 { return None; }` — so the n<2 case never reaches the underlying function. We preserve this behavior.

## `two_sample_t_test(a, b, equal_var)`

| Input | Behavior |
|-------|----------|
| Either empty | `Err(StatsError::EmptyData)` |
| Either n=1 | `Err(InvalidInput)` |
| Identical samples | `t_stat = 0.0` |
| equal_var=false (Welch) | df via Welch-Satterthwaite |
| equal_var=true (Student) | df = n_a + n_b - 2 |

Our wrapper `compare_distributions` guards n<2 → None.

## `one_way_anova(groups: &[&[f64]])`

| Input | Behavior |
|-------|----------|
| `&[]` (no groups) | likely `Err` (we filter to None upstream) |
| 1 group | `Err(InvalidInput "Need at least 2 groups")` |
| Group with n=1 | rs-stats may panic or err — our wrapper filters these out first |
| Identical groups | F = 0, p = 1 |
| Perfectly separated (ss_within=0) | F = ∞, p ≈ 0 (rs-stats has precision issues with extreme F) |

Our wrapper `test_community_homogeneity`:
1. Returns `None` if `groups.len() < 2`.
2. Filters out groups with `len() < 2`.
3. Returns `None` if filtered count < 2.
4. Otherwise calls `one_way_anova` and maps the result.

## `fit_all(data)`

| Input | Behavior |
|-------|----------|
| `&[]` (empty) | `Err(StatsError::EmptyData "fit_all: data must not be empty")` |
| `&[42.0]` (n=1) | individual fitters may return `Err`, fit_all returns those that succeed |
| `&[1.0, 1.0, ...]` (all-equal) | most fitters fail (zero variance), `Err` if all fail |
| Data with x ≤ 0 | LogNormal, Exponential, Gamma, Weibull skip silently (try_fit! macro discards Err) |
| Data outside (0,1) | Beta skip |
| Data with NaN/Inf | undefined; not in our use cases |

Our wrapper `analyze_distribution`:
1. Returns `None` if `values.is_empty()`.
2. Calls `fit_all` and on `Err` logs a warning + uses empty vec.
3. Empirical percentiles always computed (not dependent on fit_all).

## ISO contract for edge cases

After migration, our first-party impl must:

- ✅ Return `None` (or `Some(0.0)` for std_dev n=1) on empty/single-element inputs as today.
- ✅ NaN/Inf propagate through arithmetic without panicking.
- ✅ Wrappers guard low-n cases before calling primitives.
- ✅ `fit_all` skips inapplicable distributions silently (matching rs-stats).
- ✅ Zero-variance inputs produce `0.0` std_dev / `Inf` or `NaN` t_stat (whichever rs-stats does — we'll match exactly).
