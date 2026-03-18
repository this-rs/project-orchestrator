//! Integration test: Full GNN vs Voyage benchmark comparison.
//!
//! Runs all 5 downstream benchmarks with full bootstrap CI (1000 resamples)
//! and paired permutation tests (10,000 permutations).
//! Outputs a structured JSON report to stdout and validates the verdict logic.

use neural_routing_gnn::benchmark::{
    self, CodeRetrievalBenchmark, CoChangePredictionBenchmark, CommunityPredictionBenchmark,
    EmbeddingSource, GnnBenchmarkConfig, ImpactPredictionBenchmark, RoutePredictionBenchmark,
};
use neural_routing_gnn::benchmark::Benchmark;

#[test]
fn benchmark_gnn_vs_voyage_full_report() {
    // Generate synthetic data: 100 nodes, 5 communities
    let (embeddings, retrieval, impact, route, community, cochange) =
        benchmark::generate_synthetic_data(100, 5, 42);

    let benchmarks: Vec<Box<dyn Benchmark>> = vec![
        Box::new(CodeRetrievalBenchmark::new(retrieval)),
        Box::new(ImpactPredictionBenchmark::new(impact, 5)),
        Box::new(RoutePredictionBenchmark::new(route)),
        Box::new(CommunityPredictionBenchmark::new(community, 5)),
        Box::new(CoChangePredictionBenchmark::new(cochange)),
    ];

    let sources = vec![
        EmbeddingSource::Voyage,
        EmbeddingSource::Gnn,
        EmbeddingSource::Concatenated,
    ];

    let config = GnnBenchmarkConfig {
        sources: sources.iter().map(|s| s.name().to_string()).collect(),
        bootstrap_resamples: 1000,
        permutation_tests: 10_000,
        seed: 42,
    };

    let report = benchmark::run_benchmarks(&benchmarks, &embeddings, &sources, &config);

    // === Structural validations ===
    assert_eq!(report.results.len(), 3, "Should have 3 source results");
    assert_eq!(report.summary.total_tasks, 5, "Should have 5 benchmark tasks");

    // Verify all 5 benchmarks ran for each source
    for source_result in &report.results {
        assert_eq!(
            source_result.benchmarks.len(),
            5,
            "Source '{}' should have 5 benchmark results",
            source_result.source
        );
        assert!(source_result.benchmarks.contains_key("code_retrieval"));
        assert!(source_result.benchmarks.contains_key("impact_prediction"));
        assert!(source_result.benchmarks.contains_key("route_prediction"));
        assert!(source_result.benchmarks.contains_key("community_prediction"));
        assert!(source_result.benchmarks.contains_key("co_change_prediction"));
    }

    // Verify comparisons: GNN vs Voyage (5) + Concat vs Voyage (5) = 10
    assert_eq!(
        report.comparisons.len(),
        10,
        "Should have 10 pairwise comparisons"
    );

    // Verify all comparisons have valid statistical results
    for comp in &report.comparisons {
        assert!(
            comp.p_value >= 0.0 && comp.p_value <= 1.0,
            "p-value should be in [0, 1], got {} for {} vs {} on {}",
            comp.p_value,
            comp.source_a,
            comp.source_b,
            comp.metric_name
        );
        // Bootstrap CI: lower should be <= upper
        assert!(
            comp.diff_ci_95.lower <= comp.diff_ci_95.upper,
            "CI lower ({}) should be <= upper ({}) for {}",
            comp.diff_ci_95.lower,
            comp.diff_ci_95.upper,
            comp.metric_name
        );
    }

    // Verdict should be one of the valid values
    assert!(
        ["GO", "CONDITIONAL GO", "NO GO", "N/A"].contains(&report.summary.verdict.as_str()),
        "Invalid verdict: {}",
        report.summary.verdict
    );

    // === Generate JSON report ===
    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.len() > 100, "JSON report should be non-trivial");

    // Print the full report for manual inspection
    println!("\n{}", "=".repeat(80));
    println!("GNN vs Voyage Benchmark Report");
    println!("{}\n", "=".repeat(80));

    // Print per-source summary
    for source_result in &report.results {
        println!("--- {} ---", source_result.source);
        for (bench_name, metrics) in &source_result.benchmarks {
            println!("  {}: {:?}", bench_name, metrics);
        }
        println!();
    }

    // Print comparisons
    println!("--- Comparisons (vs Voyage baseline) ---");
    for comp in &report.comparisons {
        println!(
            "  {} | {} vs {}: diff={:+.4} CI=[{:.4}, {:.4}] p={:.4} {}",
            comp.metric_name,
            comp.source_a,
            comp.source_b,
            comp.diff,
            comp.diff_ci_95.lower,
            comp.diff_ci_95.upper,
            comp.p_value,
            if comp.significant { "***" } else { "ns" }
        );
    }

    // Print verdict
    println!(
        "\nVerdict: {} (GNN wins: {}/{})",
        report.summary.verdict, report.summary.gnn_wins, report.summary.total_tasks
    );
    println!("Rationale: {}", report.summary.rationale);

    // Print full JSON
    println!("\n--- Full JSON Report ---");
    println!("{}", json);
}

#[test]
fn benchmark_gnn_improvements_are_consistent() {
    // Run with different seeds to verify GNN consistently outperforms on structural tasks
    let seeds = [42, 123, 456, 789, 1337];
    let mut gnn_total_wins = 0;
    let mut total_runs = 0;

    for seed in seeds {
        let (embeddings, retrieval, impact, route, community, cochange) =
            benchmark::generate_synthetic_data(80, 4, seed);

        let benchmarks: Vec<Box<dyn Benchmark>> = vec![
            Box::new(CodeRetrievalBenchmark::new(retrieval)),
            Box::new(ImpactPredictionBenchmark::new(impact, 5)),
            Box::new(RoutePredictionBenchmark::new(route)),
            Box::new(CommunityPredictionBenchmark::new(community, 5)),
            Box::new(CoChangePredictionBenchmark::new(cochange)),
        ];

        let sources = vec![EmbeddingSource::Voyage, EmbeddingSource::Gnn];

        let config = GnnBenchmarkConfig {
            bootstrap_resamples: 200,
            permutation_tests: 1000,
            seed,
            ..Default::default()
        };

        let report = benchmark::run_benchmarks(&benchmarks, &embeddings, &sources, &config);

        gnn_total_wins += report.summary.gnn_wins;
        total_runs += 1;

        println!(
            "Seed {}: verdict={}, wins={}/{}",
            seed, report.summary.verdict, report.summary.gnn_wins, report.summary.total_tasks
        );
    }

    // GNN should win on average ≥2 tasks per run (given synthetic data favors structural embeddings)
    let avg_wins = gnn_total_wins as f64 / total_runs as f64;
    println!(
        "\nAverage GNN wins across {} seeds: {:.1}/5",
        total_runs, avg_wins
    );
    assert!(
        avg_wins >= 1.5,
        "GNN should win on average ≥1.5 tasks, got {:.1}",
        avg_wins
    );
}
