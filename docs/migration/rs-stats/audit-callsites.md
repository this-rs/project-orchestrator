# rs-stats — Inventaire exhaustif des call sites

Generated as part of plan `00f0ca9a` (Replace rs-stats with first-party impl).
Captured before any migration to lock the surface area.

## Executable call sites (6 sites — these are the migration target)

| # | File | Line | Symbol | Args | Returns | Used in |
|---|------|------|--------|------|---------|---------|
| 1 | `src/skills/maintenance.rs` | 200 | `rs_stats::prob::average` | `&[f64]` (synapse_weights) | `Result<f64>` (then `.unwrap_or(0.0)`) | Skill maintenance loop — mean weight check |
| 2 | `src/analytics/hypothesis.rs` | 17 | `rs_stats::hypothesis_tests::anova::one_way_anova` | `&[&[f64]]` | `Result<AnovaResult>` | `test_community_homogeneity` (REST) |
| 3 | `src/analytics/hypothesis.rs` | 18 | `rs_stats::hypothesis_tests::t_test::one_sample_t_test` | `&[f64], f64` | `Result<TTestResult>` | `test_stagnation` (REST) |
| 4 | `src/analytics/hypothesis.rs` | 18 | `rs_stats::hypothesis_tests::t_test::two_sample_t_test` | `&[f64], &[f64], bool` | `Result<TTestResult>` | `compare_distributions` (REST) |
| 5 | `src/analytics/distribution.rs` | 107 | `rs_stats::prob::average` | `&[f64]` | `Result<f64>` (`.unwrap_or_else(|_| sum/n)`) | `analyze_distribution` |
| 5 | `src/analytics/distribution.rs` | 108 | `rs_stats::prob::std_dev` | `&[f64]` | `Result<f64>` (`.unwrap_or(0.0)`) | `analyze_distribution` |
| 6 | `src/analytics/distribution.rs` | 133 | `rs_stats::fit_all` | `&[f64]` | `Result<Vec<FitResult>>` | `analyze_distribution` |
| 6 | `src/analytics/distribution.rs` | 15 | `rs_stats::distributions::fitting::FitResult` | type import | struct | `analyze_distribution` mapping |

**Total: 6 executable sites + 1 type import = 7 lines of code to migrate.**

## Doc-comment references (no code change required, just text)

These mention rs-stats by name in comments / module docs. They should be updated post-R7 to reflect the migration. Listed here to track the audit was complete.

| File | Lines | Type |
|------|-------|------|
| `src/analytics/mod.rs` | 1, 3 | Module doc — "rs-stats Integration" header |
| `src/analytics/hypothesis.rs` | 3, 23, 28, 123, 243, 267 | Module/struct/inline comments |
| `src/analytics/distribution.rs` | 3, 25, 94, 105, 132, 135 | Module/struct/inline comments |
| `src/skills/maintenance.rs` | (none — only the executable line) | — |
| `src/neo4j/persona.rs` | 1111, 2058, 2097 | Doc comments mentioning "rs-stats" naming |
| `src/neo4j/analytics.rs` | 2898 | Section comment "rs-stats data providers" |
| `src/neo4j/traits.rs` | 3111 | Section comment "rs-stats data providers" |
| `src/graph/algorithms.rs` | 767 | Comment "rs-stats enriched fields" |
| `src/graph/models.rs` | 434 | Field-group comment "rs-stats enriched fields" |
| `src/api/code_handlers.rs` | 2073, 2134 | Section comments "rs-stats Statistical Analytics" |

**Total: ~20 doc/comment lines to update post-migration (cosmetic, no behavior change).**

## Cargo.toml declaration

```
Cargo.toml:179:rs-stats = { version = "2.0.3", features = ["parallel"] }
```

Single line, removed in R7.

## Summary

- **6 executable call sites** in 3 files (`skills/maintenance.rs`, `analytics/hypothesis.rs`, `analytics/distribution.rs`)
- **1 type import** (`FitResult`)
- **1 Cargo.toml line**
- **~20 doc/comment references** (cosmetic)

This matches the scope declared in the plan description. No surprises.
