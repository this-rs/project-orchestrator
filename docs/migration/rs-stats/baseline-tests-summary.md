# Baseline tests summary (pre-migration)

Captured before any rs-stats migration code change.

## `cargo test --lib analytics` — 17 tests, all green

```
test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 5789 filtered out
```

Tests touching rs-stats-derived paths (10 of the 17):
- `analytics::distribution::tests::test_analyze_distribution_empty`
- `analytics::distribution::tests::test_analyze_distribution_single`
- `analytics::distribution::tests::test_adaptive_threshold_fallback`
- `analytics::distribution::tests::test_adaptive_threshold_basic`
- `analytics::distribution::tests::test_detect_outliers`
- `analytics::hypothesis::tests::test_significance_levels`
- `analytics::hypothesis::tests::test_community_homogeneity_too_few_groups`
- `analytics::hypothesis::tests::test_community_homogeneity_distinct_groups`
- `analytics::hypothesis::tests::test_compare_distributions_same`
- `analytics::hypothesis::tests::test_stagnation_flat_series`

Other tests (graph::models, graph::writer, orchestrator::runner) — touch graph analytics indirectly but do not call rs-stats. Documented for completeness.

## `cargo test --lib skills` — 647 tests, all green

```
test result: ok. 647 passed; 0 failed; 0 ignored; 0 measured; 5159 filtered out
```

Tests touching `skills::maintenance` (where the `rs_stats::prob::average` call lives) — covered by the broader skills suite, no isolated test for the synapse_weights mean line.

## ISO contract

After migration, **both commands above must report exactly the same counts** (17 + 647 = 664 passed, 0 failed). Any regression on these baselines fails ISO.
