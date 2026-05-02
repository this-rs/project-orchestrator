# Audit — rs-stats distribution fitting (R6)

Captured from `~/.cargo/registry/.../rs-stats-2.0.3/src/distributions/`.

This is the contract our first-party `src/analytics/stats/fitting.rs` must
honor for ISO migration in plan `00f0ca9a`.

## fit_all surface

`pub fn fit_all(data: &[f64]) -> StatsResult<Vec<FitResult>>`

- Empty input → `Err InvalidInput`
- Tries 10 continuous distributions (in declaration order):
  Normal, Exponential, Uniform, Gamma, LogNormal, Weibull, Beta, StudentT,
  FDistribution, ChiSquared
- Each that succeeds (fit + finite AIC/BIC) is included
- Returned `Vec<FitResult>` is sorted by AIC ascending (best fit first)
- All fail → `Err InvalidInput "no distribution could be fitted"`

`FitResult` shape (rs-stats):
```rust
pub struct FitResult {
    pub name: String,
    pub aic: f64,
    pub bic: f64,
    pub ks_statistic: f64,
    pub ks_p_value: f64,
}
```

The downstream consumer `analytics::distribution::DistributionFit` only
reads `name`, `aic`, `bic`, `ks_p_value` — `ks_statistic` is not exposed.
Our impl preserves all 5 fields for completeness.

## AIC / BIC

```
log_likelihood = Σ logpdf(x_i; θ̂)
AIC = 2k - 2*log_likelihood
BIC = k*ln(n) - 2*log_likelihood
```

`k` is `num_params()` per distribution (see table below).

## KS test

```
sort data ascending
D = max over i of {|F_emp(x_i) - F_theo(x_i)|}  with two-sided variant
  upper = (i+1)/n,  lower = i/n
  D_i = max(|upper - F(x_i)|, |F(x_i) - lower|)
D = max_i D_i
```

p-value uses Stephens (1970) finite-sample correction:
```
arg = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D
p = 2 * Σ_{j=1..100} (-1)^{j+1} * exp(-2*j²*arg²)
clamp to [0, 1]
```

## Per-distribution parameters

All formulas from rs-stats (`docs/migration/rs-stats/audit-fitting-impl.md`).

| Dist | k | Skip rule | Parameters |
|------|---|-----------|------------|
| Normal | 2 | none | μ̂ = mean(data), σ̂ = sqrt(Σ(x-μ̂)²/n) (population) |
| Exponential | 1 | any x<0 | λ̂ = 1/mean(data) |
| Uniform | 2 | none | â = min(data), b̂ = max(data) |
| Gamma | 2 | any x≤0 | Choi-Wette α; β̂ = α̂/mean |
| LogNormal | 2 | any x≤0 | μ̂ = mean(log(data)), σ̂ = pop std of log(data) |
| Weibull | 2 | any x≤0 | k̂ = cv^{-1.086} (Teimouri-Gupta); λ̂ = (Σxᵏ/n)^{1/k} |
| Beta | 2 | any x∉(0,1) | MoM: c = mean(1-mean)/var-1; α̂ = mean*c; β̂ = (1-mean)*c |
| StudentT | 3 | n<4 | μ̂, σ̂ population; ν̂ from kurtosis |
| FDistribution | 2 | any x≤0 | d₂ from mean (or fallback); d₁ from variance |
| ChiSquared | 1 | any x<0 | k̂ = max(mean, 0.01) |

### Gamma details (Choi-Wette MLE approximation)

```
mean    = Σx/n
log_mean= Σ ln(x)/n
s       = ln(mean) - log_mean
α̂      = (3 - s + sqrt((s-3)² + 24s)) / (12s)   if s > 0
       = 1.0                                     otherwise
β̂      = α̂ / mean
```

### Weibull details (Teimouri & Gupta 2013 + closed-form scale)

```
mean    = Σx/n
var     = Σ(x-mean)²/n
cv      = sqrt(var) / mean
k̂      = max(cv^{-1.086}, 0.01)
λ̂      = (Σxᵏ̂/n)^{1/k̂}
```

### StudentT details (Pearson method-of-moments via excess kurtosis)

```
μ̂      = mean(data)
σ̂      = pop std (denom n)
m4      = Σ(x-μ̂)⁴/n
κ_excess= m4/var² - 3
ν̂      = max(4 + 6/κ_excess, 2.01)   if κ_excess > 0.01
       = 30.0                          otherwise (≈ Normal, avoid noise)
```

### F details (MoM via mean and variance)

```
mean    = Σx/n
var     = Σ(x-mean)²/n
if mean ≤ 1.0: fallback (d1=2.0, d2=10.0)
else:
  d2̂  = max(2*mean / (mean-1), 2.01)
  num   = 2 * d2̂² * (d2̂ - 2)
  den   = var * (d2̂-2)² * max(d2̂-4, 0.01) - 2*d2̂²
  d1̂  = max(num/den, 0.01)  if den > 0 else 2.0
```

## Edge cases (from baseline-edge-cases.md)

| Input              | Behavior                                  |
|--------------------|-------------------------------------------|
| `&[]`              | `Err`                                     |
| `&[42.0]` n=1      | individual fitters err; if all err → Err  |
| All-equal data     | most fitters err (zero variance) → Err    |
| Any negative       | LogNormal/Exp/Gamma/Weibull skip silently |
| Out of (0,1)       | Beta skip silently                        |
| n<4                | StudentT skip silently                    |

## CDF / PDF — substituted with statrs

statrs distributions use the same parametrizations as rs-stats:
- `statrs::Normal(mean, std_dev)`
- `statrs::Exp(rate)`
- `statrs::Uniform(min, max)`
- `statrs::Gamma(shape, rate)` ← same convention as rs-stats (β̂=rate)
- `statrs::LogNormal(location, scale)`
- `statrs::Weibull(shape, scale)`
- `statrs::Beta(α, β)`
- `statrs::StudentsT(location, scale, freedom)`
- `statrs::FisherSnedecor(d1, d2)`
- `statrs::ChiSquared(k)`

We use `statrs` for `pdf`/`cdf`/`ln_pdf`. statrs's mathematical
implementations are equivalent to rs-stats's hand-rolled
`regularized_incomplete_beta` / `regularized_incomplete_gamma` /
`ln_gamma` / `erf`, so AIC/BIC should agree to within machine
precision (modulo subtle rounding differences in different
incomplete-beta algorithms).

## ISO acceptance

Per the plan tolerance matrix:
- `name`, ranking direction, `is_best`: exact
- `aic`, `bic`: 1e-6 relative
- `ks_p_value`: 1e-3 absolute

If any specific fixture × distribution combination diverges beyond
tolerance, document in `iso-divergences.md` with the numerical
explanation (typically: incomplete beta precision differences).
