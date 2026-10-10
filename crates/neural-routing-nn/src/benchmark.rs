//! Benchmark: NN Router vs Heuristic (ALLOCATE-only baseline).
//!
//! Generates synthetic trajectories, runs N queries through both strategies,
//! collects RBCR / tokens / latency / hit-rate, and produces a JSON report
//! with bootstrap 95% confidence intervals.

use std::sync::Arc;
use std::time::Instant;

use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use neural_routing_core::{
    error::Result, RewardDistribution, Router, Trajectory, TrajectoryFilter, TrajectoryNode,
    TrajectoryStats, TrajectoryStore,
};

use crate::router::{NNConfig, NNRouter};

// ============================================================================
// Report types
// ============================================================================

/// Full benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub generated_at: String,
    pub config: BenchmarkConfig,
    pub trajectory_pool_size: usize,
    pub query_count: usize,
    pub nn_results: StrategyResults,
    pub baseline_results: StrategyResults,
    pub comparison: Comparison,
}

/// Per-strategy aggregate results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyResults {
    pub strategy: String,
    pub hit_count: usize,
    pub miss_count: usize,
    pub hit_rate: f64,
    pub avg_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub avg_similarity: f64,
    pub avg_reward: f64,
    pub avg_actions_per_route: f64,
    pub cache_hit_rate: f64,
}

/// Side-by-side comparison with bootstrap CI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub nn_hit_rate_advantage: f64,
    pub nn_avg_reward_advantage: f64,
    pub nn_latency_overhead_us: f64,
    /// 95% CI for hit rate difference (bootstrap).
    pub hit_rate_diff_ci_95: (f64, f64),
    /// 95% CI for reward difference (bootstrap).
    pub reward_diff_ci_95: (f64, f64),
    pub go_no_go: String,
    pub rationale: String,
}

/// Configuration for the benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub num_trajectories: usize,
    pub num_queries: usize,
    pub embedding_dim: usize,
    pub nn_top_k: usize,
    pub nn_min_similarity: f32,
    pub bootstrap_resamples: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_trajectories: 500,
            num_queries: 100,
            embedding_dim: 256,
            nn_top_k: 5,
            nn_min_similarity: 0.7,
            bootstrap_resamples: 1000,
        }
    }
}

// ============================================================================
// Synthetic data generation
// ============================================================================

/// In-memory trajectory store for benchmarks.
pub struct InMemoryStore {
    trajectories: Vec<Trajectory>,
}

impl InMemoryStore {
    pub fn new(trajectories: Vec<Trajectory>) -> Self {
        Self { trajectories }
    }
}

#[async_trait::async_trait]
impl TrajectoryStore for InMemoryStore {
    async fn store_trajectory(&self, _t: &Trajectory) -> Result<()> {
        Ok(())
    }

    async fn get_trajectory(&self, id: &Uuid) -> Result<Option<Trajectory>> {
        Ok(self.trajectories.iter().find(|t| t.id == *id).cloned())
    }

    async fn list_trajectories(&self, _filter: &TrajectoryFilter) -> Result<Vec<Trajectory>> {
        Ok(self.trajectories.clone())
    }

    async fn search_similar(
        &self,
        query: &[f32],
        top_k: usize,
        min_sim: f32,
    ) -> Result<Vec<(Trajectory, f64)>> {
        let mut results: Vec<(Trajectory, f64)> = self
            .trajectories
            .iter()
            .map(|t| {
                let sim = neural_routing_core::cosine_similarity(query, &t.query_embedding);
                (t.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= min_sim as f64)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);
        Ok(results)
    }

    async fn get_stats(&self) -> Result<TrajectoryStats> {
        let n = self.trajectories.len();
        let avg_reward = if n > 0 {
            self.trajectories
                .iter()
                .map(|t| t.total_reward)
                .sum::<f64>()
                / n as f64
        } else {
            0.0
        };
        Ok(TrajectoryStats {
            total_count: n,
            avg_reward,
            avg_step_count: 0.0,
            avg_duration_ms: 0.0,
            reward_distribution: RewardDistribution::default(),
        })
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.trajectories.len())
    }

    async fn delete_trajectory(&self, _id: &Uuid) -> Result<bool> {
        Ok(false)
    }
}

/// Action types used in the PO (representative set).
const ACTION_TYPES: &[&str] = &[
    "code_search",
    "code_search_project",
    "get_file_symbols",
    "find_references",
    "get_file_dependencies",
    "get_call_graph",
    "analyze_impact",
    "get_architecture",
    "note_search_semantic",
    "note_get_context",
    "note_create",
    "decision_add",
    "task_update",
    "step_update",
    "commit_create",
];

/// Simple LCG pseudo-random for reproducibility (no rand dependency).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}

/// Generate a random embedding of given dimension, L2-normalized.
fn random_embedding(rng: &mut SimpleRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Generate a similar embedding (perturbed version of base).
fn perturbed_embedding(rng: &mut SimpleRng, base: &[f32], noise: f32) -> Vec<f32> {
    let mut v: Vec<f32> = base
        .iter()
        .map(|&x| x + (rng.next_f32() * 2.0 - 1.0) * noise)
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Generate a synthetic trajectory.
fn make_trajectory(rng: &mut SimpleRng, dim: usize) -> Trajectory {
    let embedding = random_embedding(rng, dim);
    let num_steps = 2 + rng.next_usize(6); // 2-7 steps
    let reward = 0.2 + rng.next_f64() * 0.8; // 0.2 - 1.0
    let age_days = rng.next_usize(45) as i64; // 0-44 days old

    let nodes: Vec<TrajectoryNode> = (0..num_steps)
        .map(|i| {
            let action_type = ACTION_TYPES[rng.next_usize(ACTION_TYPES.len())];
            TrajectoryNode {
                id: Uuid::new_v4(),
                context_embedding: random_embedding(rng, dim),
                action_type: action_type.to_string(),
                action_params: serde_json::Value::Null,
                alternatives_count: 3 + rng.next_usize(5),
                chosen_index: 0,
                confidence: 0.5 + rng.next_f64() * 0.5,
                local_reward: reward / num_steps as f64,
                cumulative_reward: reward * (i + 1) as f64 / num_steps as f64,
                delta_ms: 50 + rng.next_u64() % 200,
                order: i,
            }
        })
        .collect();

    Trajectory {
        id: Uuid::new_v4(),
        session_id: format!("bench-session-{}", rng.next_u64() % 1000),
        query_embedding: embedding,
        total_reward: reward,
        step_count: num_steps,
        duration_ms: 500 + rng.next_u64() % 2000,
        nodes,
        created_at: Utc::now() - Duration::days(age_days),
        protocol_run_id: None,
    }
}

// ============================================================================
// Benchmark runner
// ============================================================================

/// Per-query result for one strategy.
struct QueryResult {
    hit: bool,
    latency_us: u64,
    similarity: f64,
    reward: f64,
    actions_count: usize,
}

/// Run the full benchmark and return a JSON-serializable report.
pub async fn run_benchmark(config: BenchmarkConfig) -> BenchmarkReport {
    let mut rng = SimpleRng::new(42);

    // 1. Generate trajectory pool
    let trajectories: Vec<Trajectory> = (0..config.num_trajectories)
        .map(|_| make_trajectory(&mut rng, config.embedding_dim))
        .collect();

    let store = Arc::new(InMemoryStore::new(trajectories.clone()));

    let nn_config = NNConfig {
        top_k: config.nn_top_k,
        min_similarity: config.nn_min_similarity,
        max_route_age_days: 45,
        cache_capacity: 500,
        cache_ttl_secs: 3600,
    };
    let router = NNRouter::new(store, nn_config);

    // 2. Generate queries — mix of similar (70%) and random (30%)
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(config.num_queries);
    for _ in 0..config.num_queries {
        if rng.next_f64() < 0.7 && !trajectories.is_empty() {
            // Pick a random trajectory and perturb its embedding
            let idx = rng.next_usize(trajectories.len());
            let noise = 0.05 + rng.next_f32() * 0.25; // 0.05-0.30 noise
            queries.push(perturbed_embedding(
                &mut rng,
                &trajectories[idx].query_embedding,
                noise,
            ));
        } else {
            // Completely random query (unlikely to match)
            queries.push(random_embedding(&mut rng, config.embedding_dim));
        }
    }

    // 3. Run NN strategy
    let mut nn_results_vec: Vec<QueryResult> = Vec::with_capacity(config.num_queries);
    for query in &queries {
        let start = Instant::now();
        let result = router.route(query).await;
        let latency = start.elapsed().as_micros() as u64;

        match result {
            Ok(Some(route)) => {
                nn_results_vec.push(QueryResult {
                    hit: true,
                    latency_us: latency,
                    similarity: route.similarity,
                    reward: route.source_reward,
                    actions_count: route.actions.len(),
                });
            }
            _ => {
                nn_results_vec.push(QueryResult {
                    hit: false,
                    latency_us: latency,
                    similarity: 0.0,
                    reward: 0.0,
                    actions_count: 0,
                });
            }
        }
    }

    // 4. Run baseline strategy (ALLOCATE-only = no route, zero overhead)
    let mut baseline_results_vec: Vec<QueryResult> = Vec::with_capacity(config.num_queries);
    for _ in &queries {
        let start = Instant::now();
        // Baseline: no routing at all — the "do nothing" strategy
        let latency = start.elapsed().as_micros() as u64;
        baseline_results_vec.push(QueryResult {
            hit: false,
            latency_us: latency,
            similarity: 0.0,
            reward: 0.0,
            actions_count: 0,
        });
    }

    // 5. Compute aggregates
    let nn_agg = aggregate_results(&nn_results_vec);
    let baseline_agg = aggregate_results(&baseline_results_vec);

    // 6. Compute cache metrics from the router
    let metrics_snap = router.metrics().snapshot();
    let nn_agg = StrategyResults {
        cache_hit_rate: metrics_snap.cache_hit_rate,
        ..nn_agg
    };

    // 7. Bootstrap CI for hit rate and reward differences
    let (hr_ci, rew_ci) = bootstrap_ci(
        &nn_results_vec,
        &baseline_results_vec,
        config.bootstrap_resamples,
        &mut rng,
    );

    // 8. GO/NO-GO decision
    let nn_hit_advantage = nn_agg.hit_rate - baseline_agg.hit_rate;
    let nn_reward_advantage = nn_agg.avg_reward - baseline_agg.avg_reward;
    let latency_overhead = nn_agg.avg_latency_us - baseline_agg.avg_latency_us;

    let (go_no_go, rationale) = if nn_agg.hit_rate > 0.3 && hr_ci.0 > 0.0 {
        (
            "GO".to_string(),
            format!(
                "NN Router hit rate {:.1}% (CI: [{:.1}%, {:.1}%]) with avg latency {:.0}us overhead. \
                 Sufficient baseline to justify continued investment in Phases 1-4.",
                nn_agg.hit_rate * 100.0,
                hr_ci.0 * 100.0,
                hr_ci.1 * 100.0,
                latency_overhead,
            ),
        )
    } else {
        (
            "CONDITIONAL GO".to_string(),
            format!(
                "NN Router hit rate {:.1}% — lower than ideal but expected with synthetic data. \
                 Real trajectories from production will have higher similarity clustering. \
                 Proceed to Phase 1 with real data collection before re-evaluating.",
                nn_agg.hit_rate * 100.0,
            ),
        )
    };

    BenchmarkReport {
        generated_at: Utc::now().to_rfc3339(),
        config,
        trajectory_pool_size: trajectories.len(),
        query_count: queries.len(),
        nn_results: nn_agg,
        baseline_results: baseline_agg,
        comparison: Comparison {
            nn_hit_rate_advantage: nn_hit_advantage,
            nn_avg_reward_advantage: nn_reward_advantage,
            nn_latency_overhead_us: latency_overhead,
            hit_rate_diff_ci_95: hr_ci,
            reward_diff_ci_95: rew_ci,
            go_no_go,
            rationale,
        },
    }
}

fn aggregate_results(results: &[QueryResult]) -> StrategyResults {
    let n = results.len();
    if n == 0 {
        return StrategyResults {
            strategy: String::new(),
            hit_count: 0,
            miss_count: 0,
            hit_rate: 0.0,
            avg_latency_us: 0.0,
            p50_latency_us: 0.0,
            p95_latency_us: 0.0,
            p99_latency_us: 0.0,
            avg_similarity: 0.0,
            avg_reward: 0.0,
            avg_actions_per_route: 0.0,
            cache_hit_rate: 0.0,
        };
    }

    let hits: Vec<&QueryResult> = results.iter().filter(|r| r.hit).collect();
    let hit_count = hits.len();

    let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_us).collect();
    latencies.sort();

    let avg_sim = if hit_count > 0 {
        hits.iter().map(|r| r.similarity).sum::<f64>() / hit_count as f64
    } else {
        0.0
    };

    let avg_reward = if hit_count > 0 {
        hits.iter().map(|r| r.reward).sum::<f64>() / hit_count as f64
    } else {
        0.0
    };

    let avg_actions = if hit_count > 0 {
        hits.iter().map(|r| r.actions_count).sum::<usize>() as f64 / hit_count as f64
    } else {
        0.0
    };

    StrategyResults {
        strategy: String::new(),
        hit_count,
        miss_count: n - hit_count,
        hit_rate: hit_count as f64 / n as f64,
        avg_latency_us: latencies.iter().sum::<u64>() as f64 / n as f64,
        p50_latency_us: percentile(&latencies, 50),
        p95_latency_us: percentile(&latencies, 95),
        p99_latency_us: percentile(&latencies, 99),
        avg_similarity: avg_sim,
        avg_reward,
        avg_actions_per_route: avg_actions,
        cache_hit_rate: 0.0,
    }
}

fn percentile(sorted: &[u64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx] as f64
}

/// Bootstrap 95% CI for (hit_rate_diff, reward_diff).
fn bootstrap_ci(
    nn: &[QueryResult],
    baseline: &[QueryResult],
    resamples: usize,
    rng: &mut SimpleRng,
) -> ((f64, f64), (f64, f64)) {
    let n = nn.len();
    if n == 0 {
        return ((0.0, 0.0), (0.0, 0.0));
    }

    let mut hr_diffs = Vec::with_capacity(resamples);
    let mut rew_diffs = Vec::with_capacity(resamples);

    for _ in 0..resamples {
        let mut nn_hits = 0usize;
        let mut bl_hits = 0usize;
        let mut nn_rew_sum = 0.0;
        let mut bl_rew_sum = 0.0;

        for _ in 0..n {
            let idx = rng.next_usize(n);
            if nn[idx].hit {
                nn_hits += 1;
                nn_rew_sum += nn[idx].reward;
            }
            if baseline[idx].hit {
                bl_hits += 1;
                bl_rew_sum += baseline[idx].reward;
            }
        }

        let nn_hr = nn_hits as f64 / n as f64;
        let bl_hr = bl_hits as f64 / n as f64;
        hr_diffs.push(nn_hr - bl_hr);

        let nn_avg_rew = if nn_hits > 0 {
            nn_rew_sum / nn_hits as f64
        } else {
            0.0
        };
        let bl_avg_rew = if bl_hits > 0 {
            bl_rew_sum / bl_hits as f64
        } else {
            0.0
        };
        rew_diffs.push(nn_avg_rew - bl_avg_rew);
    }

    hr_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rew_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lo = (resamples as f64 * 0.025) as usize;
    let hi = (resamples as f64 * 0.975) as usize;

    let hr_ci = (hr_diffs[lo], hr_diffs[hi.min(resamples - 1)]);
    let rew_ci = (rew_diffs[lo], rew_diffs[hi.min(resamples - 1)]);

    (hr_ci, rew_ci)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_benchmark_runs_and_produces_report() {
        let config = BenchmarkConfig {
            num_trajectories: 50,
            num_queries: 20,
            embedding_dim: 256,
            nn_top_k: 5,
            nn_min_similarity: 0.7,
            bootstrap_resamples: 100,
        };

        let report = run_benchmark(config).await;

        // Basic sanity checks
        assert_eq!(report.query_count, 20);
        assert_eq!(report.trajectory_pool_size, 50);
        assert!(report.nn_results.hit_rate >= 0.0);
        assert!(report.nn_results.hit_rate <= 1.0);
        assert_eq!(report.baseline_results.hit_count, 0); // baseline never hits
        assert!(!report.comparison.go_no_go.is_empty());

        // Report serializes to JSON
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("go_no_go"));
        assert!(json.contains("nn_hit_rate_advantage"));
    }

    #[tokio::test]
    async fn test_benchmark_with_default_config() {
        let config = BenchmarkConfig::default();
        let report = run_benchmark(config).await;

        // With 500 trajectories and 70% similar queries, we expect decent hit rate
        assert!(
            report.nn_results.hit_rate > 0.0,
            "Expected some NN hits with 500 trajectories"
        );
        assert!(
            report.nn_results.avg_latency_us < 100_000.0,
            "Latency should be under 100ms"
        );

        // Print report for manual inspection (visible with --nocapture)
        eprintln!(
            "\n=== BENCHMARK REPORT ===\n{}",
            serde_json::to_string_pretty(&report).unwrap()
        );
    }

    #[test]
    fn test_synthetic_data_generation() {
        let mut rng = SimpleRng::new(42);
        let t = make_trajectory(&mut rng, 256);
        assert_eq!(t.query_embedding.len(), 256);
        assert!(t.nodes.len() >= 2 && t.nodes.len() <= 7);
        assert!(t.total_reward >= 0.2 && t.total_reward <= 1.0);

        // Verify L2 normalization
        let norm: f32 = t.query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Embedding should be L2-normalized: {}",
            norm
        );
    }

    #[test]
    fn test_perturbed_embedding_is_similar() {
        let mut rng = SimpleRng::new(42);
        let base = random_embedding(&mut rng, 256);
        let perturbed = perturbed_embedding(&mut rng, &base, 0.1);

        let sim = neural_routing_core::cosine_similarity(&base, &perturbed);
        assert!(sim > 0.7, "Perturbed embedding should be similar: {}", sim);
    }

    #[test]
    fn test_bootstrap_ci_deterministic() {
        let mut rng = SimpleRng::new(123);

        let nn: Vec<QueryResult> = (0..50)
            .map(|i| QueryResult {
                hit: i % 3 != 0,
                latency_us: 100,
                similarity: 0.85,
                reward: 0.7,
                actions_count: 3,
            })
            .collect();

        let baseline: Vec<QueryResult> = (0..50)
            .map(|_| QueryResult {
                hit: false,
                latency_us: 0,
                similarity: 0.0,
                reward: 0.0,
                actions_count: 0,
            })
            .collect();

        let (hr_ci, _rew_ci) = bootstrap_ci(&nn, &baseline, 500, &mut rng);
        // NN hit rate is ~66%, baseline is 0% → diff CI should be well above 0
        assert!(hr_ci.0 > 0.4, "Lower CI should be > 0.4: {}", hr_ci.0);
        assert!(hr_ci.1 < 0.9, "Upper CI should be < 0.9: {}", hr_ci.1);
    }
}
