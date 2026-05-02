# Audit — rs-stats one-way ANOVA implementation

Captured from `~/.cargo/registry/src/.../rs-stats-2.0.3/src/hypothesis_tests/anova.rs`.

This document is the contract our first-party `src/analytics/stats/anova.rs` must
honor for ISO migration in plan `00f0ca9a`.

## API shape

```rust
pub struct AnovaResult {
    pub f_statistic: f64,
    pub df_between: usize,
    pub df_within: usize,
    pub p_value: f64,
    pub ss_between: f64,
    pub ss_within: f64,
    pub ms_between: f64,
    pub ms_within: f64,
}
```

8 fields — preserved exactly in our impl. The serde-safe
`analytics::hypothesis::AnovaResult` adds a 9th field (`significance`)
derived from `p_value`.

## Validation (lines 92-120)

```
groups.len() < 2          → Err InvalidInput "ANOVA requires at least 2 groups"
any group with len < 2    → Err InvalidInput "Each group must have at least 2 observations"
```

The wrapper `analytics::hypothesis::test_community_homogeneity` already
**pre-filters** groups with `len < 2` and returns `None` if fewer than 2
remain. So our primitive can use the same `None` semantic; downstream
behavior is preserved.

## Core formulas (lines 122-186)

```
n_total      = Σ_i |group_i|
grand_mean   = (Σ over all values across all groups) / n_total
group_mean_i = (Σ over group_i) / |group_i|

ss_between   = Σ_i |group_i| * (group_mean_i - grand_mean)²
ss_within    = Σ_i Σ_j (x_ij - group_mean_i)²

df_between   = k - 1                       (where k = #groups)
df_within    = n_total - k

ms_between   = ss_between / df_between
ms_within    = ss_within  / df_within

f_statistic  = ms_between / ms_within

p_value      = 1.0 - F_cdf(f_statistic; df_between, df_within)
```

These are textbook one-way ANOVA formulas. No surprises.

**Note**: `df_between` and `df_within` are exposed as `usize` (integers).
Some test code (`hypothesis.rs::test_community_homogeneity_distinct_groups`)
asserts on `df_between=2, df_within=12` — these must remain integers.

## p-value computation (lines 191-...)

rs-stats uses its own `f_distribution_cdf` built on a hand-rolled
`regularized_incomplete_beta`. Symmetry trick for `F < 1`:

```
F_cdf(f; df1, df2) = 1 - F_cdf(1/f; df2, df1)   when f < 1
```

For `f ≥ 1`:
```
x = df2 / (df2 + df1 * f)
a = df2 / 2
b = df1 / 2
cdf = regularized_incomplete_beta(x, a, b)
```

The same precision issues as t-test apply (incomplete_beta saturates
near 0/1 for extreme F values).

### Our replacement

`statrs::distribution::FisherSnedecor::new(df_between as f64, df_within as f64)`,
then:

```rust
let p_value = 1.0 - dist.cdf(f_statistic);
```

`statrs::FisherSnedecor` uses the same regularized incomplete beta
formulation but with industrial-strength numerics.

## Degenerate case: F = +∞

When `ss_within = 0` (all observations within each group are identical),
`ms_within = 0` and `f_statistic = +∞`.

rs-stats: `1.0 - cdf(∞)` returns `0.0` (or near-zero) but with documented
imprecision. The existing test `test_community_homogeneity_distinct_groups`
notes this issue.

`statrs::FisherSnedecor::cdf(f64::INFINITY) = 1.0` exactly, so we get
`p_value = 0.0` cleanly. We document this and assert it explicitly in
the AnovaResult — no need for sentinels.

## Edge cases preserved (from baseline-edge-cases.md)

| Input                    | rs-stats behavior            | Our behavior              |
|--------------------------|------------------------------|---------------------------|
| `&[]` (no groups)        | `Err InvalidInput`           | `None`                    |
| `&[group]` (1 group)     | `Err InvalidInput`           | `None`                    |
| Group with len < 2       | `Err InvalidInput`           | wrapper filters; `None` if <2 valid |
| All groups identical     | `f = 0`, `p ≈ 1`             | `f = 0`, `p = 1` exactly  |
| ss_within = 0 (constant) | `f = ∞`, `p ≈ 0` (noisy)     | `f = ∞`, `p = 0` exactly  |
| Normal case              | `Ok(AnovaResult)`            | `Some(AnovaResult)`       |
