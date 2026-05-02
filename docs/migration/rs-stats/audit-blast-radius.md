# rs-stats — Blast radius (downstream consumers)

Lists every code path that reads outputs derived from `rs-stats` (directly or via our wrappers).
Used to scope ISO testing post-migration.

## Direct consumers of `analyze_distribution` / `analytics::distribution::*`

### REST API surface (public contract — must remain ISO)

| File | Lines | Function | rs-stats-derived field consumed |
|------|-------|----------|----------------------------------|
| `src/api/code_handlers.rs` | 13 | (import) | — |
| `src/api/code_handlers.rs` | 2086 | `pagerank_distribution` REST handler | `analyze_distribution(pr_vals)` → `DistributionAnalysis` (all fields) |
| `src/api/code_handlers.rs` | 2098 | `risk_score_distribution` REST handler | `analyze_distribution(risk_vals)` → `DistributionAnalysis` |
| `src/api/code_handlers.rs` | 2100 | `risk_score_p95_threshold` REST handler | `adaptive_threshold(risk_vals, 0.95, 0.75)` → `f64` |
| `src/api/code_handlers.rs` | 2115 | `community_homogeneity` REST handler | `test_community_homogeneity(groups)` → `Option<AnovaResult>` |

### Internal Neo4j layer (graph algorithms)

| File | Lines | Function | Use |
|------|-------|----------|-----|
| `src/neo4j/persona.rs` | 7 | (import `adaptive_threshold`, `detect_outliers`) | — |
| `src/neo4j/persona.rs` | 2143 | `compute_adaptive_thresholds` | `adaptive_threshold(weights, 0.05, 0.1)` — prune_cutoff |
| `src/neo4j/persona.rs` | 2148-2149 | `compute_adaptive_thresholds` | `adaptive_threshold(sorted, 0.25/0.75, …)` — q1/q3 |
| `src/neo4j/persona.rs` | 2154 | `compute_adaptive_thresholds` | `detect_outliers(weights, 1.5)` |
| `src/neo4j/persona.rs` | 2192 | `compute_adaptive_thresholds` | `adaptive_threshold(file_counts, 0.90, 20.0)` |
| `src/neo4j/analytics.rs` | 1455 | (analytics impl) | `adaptive_threshold(weights, 0.25, 0.3)` |
| `src/neo4j/analytics.rs` | 1823, 1825, 1827 | `risk_score` thresholding | `adaptive_threshold(risk_vals, 0.40/0.70/0.90, …)` (3-tier) |
| `src/neo4j/skill.rs` | 8, 756 | (import + use) | `adaptive_threshold(energies, 0.05, 0.05)` |
| `src/neo4j/mock.rs` | 9075-9097 | mock implementation of `compute_adaptive_thresholds` | mirrors persona.rs usage |
| `src/neo4j/impl_graph_store.rs` | 2906, 2910 | trait dispatch | calls persona.rs impl |
| `src/neo4j/traits.rs` | 2514 | trait declaration | no body |

### Notes manager

| File | Lines | Function | Use |
|------|-------|----------|-----|
| `src/notes/manager.rs` | 1541 | note ranking | `adaptive_threshold(weights, 0.70, 0.75)` |

### Skills lifecycle

| File | Lines | Function | Use |
|------|-------|----------|-----|
| `src/skills/maintenance.rs` | 153 | maintenance loop | `adaptive_threshold(...)` (via crate path) |
| `src/skills/maintenance.rs` | 197 | maintenance loop | `test_stagnation(synapse_weights, 0.5)` |
| `src/skills/maintenance.rs` | 200 | maintenance loop | `rs_stats::prob::average(&synapse_weights)` (DIRECT rs-stats — see callsites doc) |
| `src/skills/lifecycle.rs` | 81-89 | promotion/demotion thresholds | `adaptive_threshold` × 4 (promotion energy/cohesion + demotion energy/cohesion) |

## Existing test sites that exercise these paths

| File | Function | What it asserts |
|------|----------|------------------|
| `src/analytics/hypothesis.rs::tests` | `test_community_homogeneity_too_few_groups` | None on <2 groups |
| `src/analytics/hypothesis.rs::tests` | `test_community_homogeneity_distinct_groups` | F > 500.0, df_between=2, df_within=12 |
| `src/analytics/hypothesis.rs::tests` | `test_compare_distributions_same` | NotSignificant on identical samples |
| `src/analytics/hypothesis.rs::tests` | `test_stagnation_flat_series` | t_statistic.abs() < 1.0, df = 5 |
| `src/analytics/hypothesis.rs::tests` | `test_significance_levels` | p-value → SignificanceLevel mapping |
| `src/analytics/distribution.rs::tests` | `test_analyze_distribution_empty` | None on empty input |
| `src/analytics/distribution.rs::tests` | `test_analyze_distribution_single` | n=1: count=1, mean=42.0, p95=42.0 |
| `src/analytics/distribution.rs::tests` | `test_adaptive_threshold_fallback` | empty → fallback |
| `src/analytics/distribution.rs::tests` | `test_adaptive_threshold_basic` | p95 of 1..=100 ≈ 95 |
| `src/analytics/distribution.rs::tests` | `test_detect_outliers` | 1000.0 detected as outlier |

**Total: 10 existing tests in analytics module.**

## Summary

- **5 REST handlers** consume rs-stats-derived results (public API contract — ISO critical)
- **6 internal modules** (`neo4j/persona`, `neo4j/analytics`, `neo4j/skill`, `neo4j/mock`, `notes/manager`, `skills/maintenance`, `skills/lifecycle`) consume `adaptive_threshold` / `detect_outliers` / `test_stagnation`
- **10 existing unit tests** to keep green

Every consumer uses our wrappers (`adaptive_threshold`, `analyze_distribution`, `test_community_homogeneity`, etc.) — they do NOT touch rs-stats directly. So the migration boundary is **inside `analytics/{distribution.rs, hypothesis.rs}` + 1 line in `skills/maintenance.rs`**. Everything else remains unchanged.
