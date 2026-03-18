//! Benchmark: GNN embeddings vs Voyage embeddings on downstream tasks.
//!
//! Evaluates the structural GNN encoding against semantic Voyage embeddings
//! across 5 downstream tasks, entirely in Rust (no Python dependency).
//!
//! ## Tasks
//! 1. **Code retrieval** — query → relevant files (Recall@K, MRR)
//! 2. **Impact prediction** — modified file → impacted files (F1)
//! 3. **Route prediction** — query → first decision (Accuracy)
//! 4. **Community prediction** — node → Louvain community (F1)
//! 5. **Co-change prediction** — file pair → co-change? (AUC-ROC)

pub mod metrics;
pub mod statistical;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::info;

use metrics::{cosine_similarity, MetricSet};
use statistical::{bootstrap_ci, paired_permutation_test, ComparisonResult, SimpleRng};

// ---------------------------------------------------------------------------
// Embedding source abstraction
// ---------------------------------------------------------------------------

/// Source of embeddings for benchmark comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingSource {
    /// Semantic embeddings from Voyage (768d)
    Voyage,
    /// Structural embeddings from GNN (256d)
    Gnn,
    /// Concatenation of Voyage + GNN
    Concatenated,
    /// Weighted fusion
    Fused { gnn_weight: f32 },
}

impl EmbeddingSource {
    pub fn name(&self) -> &str {
        match self {
            Self::Voyage => "Voyage",
            Self::Gnn => "GNN",
            Self::Concatenated => "Concat",
            Self::Fused { .. } => "Fused",
        }
    }
}

/// Embeddings for a single node, indexed by source.
#[derive(Debug, Clone)]
pub struct NodeEmbeddings {
    pub node_id: String,
    pub node_type: String,
    /// Voyage embedding (768d, may be empty if unavailable)
    pub voyage: Vec<f32>,
    /// GNN embedding (256d, may be empty if unavailable)
    pub gnn: Vec<f32>,
}

impl NodeEmbeddings {
    /// Get embedding for a given source.
    pub fn get(&self, source: &EmbeddingSource) -> Vec<f32> {
        match source {
            EmbeddingSource::Voyage => self.voyage.clone(),
            EmbeddingSource::Gnn => self.gnn.clone(),
            EmbeddingSource::Concatenated => {
                let mut concat = self.voyage.clone();
                concat.extend_from_slice(&self.gnn);
                concat
            }
            EmbeddingSource::Fused { gnn_weight } => {
                if self.voyage.is_empty() || self.gnn.is_empty() {
                    return if self.gnn.is_empty() {
                        self.voyage.clone()
                    } else {
                        self.gnn.clone()
                    };
                }
                // Weighted concatenation: scale each source by its weight, then concat.
                // Output dim = voyage_dim + gnn_dim (preserves all information).
                let w = *gnn_weight;
                let vw = 1.0 - w;
                let mut fused = Vec::with_capacity(self.voyage.len() + self.gnn.len());
                fused.extend(self.voyage.iter().map(|v| v * vw));
                fused.extend(self.gnn.iter().map(|g| g * w));
                fused
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark trait
// ---------------------------------------------------------------------------

/// Trait for a downstream benchmark task.
pub trait Benchmark: Send + Sync {
    /// Name of this benchmark.
    fn name(&self) -> &str;

    /// Run the benchmark with the given embedding source.
    fn run(&self, embeddings: &[NodeEmbeddings], source: &EmbeddingSource) -> MetricSet;

    /// Get per-query scores (for bootstrap CI). Each f64 is the query-level metric.
    fn per_query_scores(
        &self,
        embeddings: &[NodeEmbeddings],
        source: &EmbeddingSource,
    ) -> Vec<f64>;
}

// ---------------------------------------------------------------------------
// Benchmark data: synthetic test cases
// ---------------------------------------------------------------------------

/// A test case for code retrieval benchmark.
#[derive(Debug, Clone)]
pub struct RetrievalTestCase {
    /// Query node ID
    pub query_id: String,
    /// Ground truth relevant node IDs
    pub relevant_ids: Vec<String>,
}

/// A test case for impact prediction.
#[derive(Debug, Clone)]
pub struct ImpactTestCase {
    /// Modified file node ID
    pub modified_id: String,
    /// Files actually impacted
    pub impacted_ids: Vec<String>,
}

/// A test case for co-change prediction.
#[derive(Debug, Clone)]
pub struct CoChangeTestCase {
    pub file_a: String,
    pub file_b: String,
    pub co_changes: bool,
}

// ---------------------------------------------------------------------------
// 1. Code Retrieval Benchmark
// ---------------------------------------------------------------------------

pub struct CodeRetrievalBenchmark {
    pub test_cases: Vec<RetrievalTestCase>,
}

impl CodeRetrievalBenchmark {
    pub fn new(test_cases: Vec<RetrievalTestCase>) -> Self {
        Self { test_cases }
    }
}

impl Benchmark for CodeRetrievalBenchmark {
    fn name(&self) -> &str {
        "code_retrieval"
    }

    fn run(&self, embeddings: &[NodeEmbeddings], source: &EmbeddingSource) -> MetricSet {
        let emb_map = build_embedding_map(embeddings, source);
        let ranked = self.rank_results(&emb_map);

        MetricSet {
            recall_at_1: Some(metrics::recall_at_k(&ranked, 1)),
            recall_at_5: Some(metrics::recall_at_k(&ranked, 5)),
            recall_at_10: Some(metrics::recall_at_k(&ranked, 10)),
            mrr: Some(metrics::mrr(&ranked)),
            ..Default::default()
        }
    }

    fn per_query_scores(
        &self,
        embeddings: &[NodeEmbeddings],
        source: &EmbeddingSource,
    ) -> Vec<f64> {
        let emb_map = build_embedding_map(embeddings, source);
        self.rank_results(&emb_map)
            .iter()
            .map(|results| {
                // Per-query: reciprocal rank of first hit
                results
                    .iter()
                    .position(|(_, rel)| *rel)
                    .map(|pos| 1.0 / (pos + 1) as f64)
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

impl CodeRetrievalBenchmark {
    fn rank_results(
        &self,
        emb_map: &HashMap<String, Vec<f32>>,
    ) -> Vec<Vec<(usize, bool)>> {
        self.test_cases
            .iter()
            .filter_map(|tc| {
                let query_emb = emb_map.get(&tc.query_id)?;

                let mut scored: Vec<(String, f64)> = emb_map
                    .iter()
                    .filter(|(id, _)| **id != tc.query_id)
                    .map(|(id, emb)| (id.clone(), cosine_similarity(query_emb, emb)))
                    .collect();

                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let ranked: Vec<(usize, bool)> = scored
                    .iter()
                    .enumerate()
                    .map(|(rank, (id, _))| (rank, tc.relevant_ids.contains(id)))
                    .collect();

                Some(ranked)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 2. Impact Prediction Benchmark
// ---------------------------------------------------------------------------

pub struct ImpactPredictionBenchmark {
    pub test_cases: Vec<ImpactTestCase>,
    /// Top-K to consider as predicted impacted
    pub top_k: usize,
}

impl ImpactPredictionBenchmark {
    pub fn new(test_cases: Vec<ImpactTestCase>, top_k: usize) -> Self {
        Self { test_cases, top_k }
    }
}

impl Benchmark for ImpactPredictionBenchmark {
    fn name(&self) -> &str {
        "impact_prediction"
    }

    fn run(&self, embeddings: &[NodeEmbeddings], source: &EmbeddingSource) -> MetricSet {
        let emb_map = build_embedding_map(embeddings, source);
        let predictions = self.predict(&emb_map);
        let (precision, recall, f1) = metrics::f1_score(&predictions);

        MetricSet {
            f1: Some(f1),
            precision: Some(precision),
            recall: Some(recall),
            ..Default::default()
        }
    }

    fn per_query_scores(
        &self,
        embeddings: &[NodeEmbeddings],
        source: &EmbeddingSource,
    ) -> Vec<f64> {
        let emb_map = build_embedding_map(embeddings, source);

        self.test_cases
            .iter()
            .map(|tc| {
                let top_k = self.rank_top_k(&emb_map, &tc.modified_id);
                let hits = tc
                    .impacted_ids
                    .iter()
                    .filter(|id| top_k.contains(*id))
                    .count();

                // Compute per-query F1 (consistent with run() which reports F1)
                let precision = if self.top_k > 0 {
                    hits as f64 / self.top_k as f64
                } else {
                    0.0
                };
                let recall = if tc.impacted_ids.is_empty() {
                    0.0
                } else {
                    hits as f64 / tc.impacted_ids.len() as f64
                };
                if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl ImpactPredictionBenchmark {
    /// Rank all nodes by cosine similarity to `query_id` and return the top-K as a set.
    /// Returns None if `query_id` is not found in the embedding map.
    fn rank_top_k(
        &self,
        emb_map: &HashMap<String, Vec<f32>>,
        query_id: &str,
    ) -> std::collections::HashSet<String> {
        let query_emb = match emb_map.get(query_id) {
            Some(e) => e,
            None => return std::collections::HashSet::new(),
        };

        let mut scored: Vec<(String, f64)> = emb_map
            .iter()
            .filter(|(id, _)| id.as_str() != query_id)
            .map(|(id, emb)| (id.clone(), cosine_similarity(query_emb, emb)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .iter()
            .take(self.top_k)
            .map(|(id, _)| id.clone())
            .collect()
    }

    fn predict(&self, emb_map: &HashMap<String, Vec<f32>>) -> Vec<(bool, bool)> {
        let mut predictions = Vec::new();

        for tc in &self.test_cases {
            let top_k = self.rank_top_k(emb_map, &tc.modified_id);
            if top_k.is_empty() && emb_map.get(&tc.modified_id).is_none() {
                continue;
            }

            // For each candidate, generate a (predicted, actual) pair
            for id in emb_map.keys() {
                if id == &tc.modified_id {
                    continue;
                }
                let predicted = top_k.contains(id);
                let actual = tc.impacted_ids.contains(id);
                predictions.push((predicted, actual));
            }
        }

        predictions
    }
}

// ---------------------------------------------------------------------------
// 3. Route Prediction Benchmark
// ---------------------------------------------------------------------------

/// A test case for route prediction.
#[derive(Debug, Clone)]
pub struct RouteTestCase {
    pub query_embedding_id: String,
    pub correct_first_action: String,
    /// Candidate actions with their node IDs
    pub candidates: Vec<(String, String)>, // (action_name, node_id)
}

pub struct RoutePredictionBenchmark {
    pub test_cases: Vec<RouteTestCase>,
}

impl RoutePredictionBenchmark {
    pub fn new(test_cases: Vec<RouteTestCase>) -> Self {
        Self { test_cases }
    }
}

impl Benchmark for RoutePredictionBenchmark {
    fn name(&self) -> &str {
        "route_prediction"
    }

    fn run(&self, embeddings: &[NodeEmbeddings], source: &EmbeddingSource) -> MetricSet {
        let scores = self.per_query_scores(embeddings, source);
        let accuracy = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        MetricSet {
            accuracy: Some(accuracy),
            ..Default::default()
        }
    }

    fn per_query_scores(
        &self,
        embeddings: &[NodeEmbeddings],
        source: &EmbeddingSource,
    ) -> Vec<f64> {
        let emb_map = build_embedding_map(embeddings, source);

        self.test_cases
            .iter()
            .filter_map(|tc| {
                let query_emb = emb_map.get(&tc.query_embedding_id)?;

                // Find most similar candidate
                let best = tc
                    .candidates
                    .iter()
                    .filter_map(|(action, node_id)| {
                        let emb = emb_map.get(node_id)?;
                        Some((action.clone(), cosine_similarity(query_emb, emb)))
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                best.map(|(predicted_action, _)| {
                    if predicted_action == tc.correct_first_action {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 4. Community Prediction Benchmark
// ---------------------------------------------------------------------------

/// Node with its community label for community prediction.
#[derive(Debug, Clone)]
pub struct CommunityTestCase {
    pub node_id: String,
    pub community_id: usize,
}

pub struct CommunityPredictionBenchmark {
    pub test_cases: Vec<CommunityTestCase>,
    /// K nearest neighbors for classification
    pub k_neighbors: usize,
}

impl CommunityPredictionBenchmark {
    pub fn new(test_cases: Vec<CommunityTestCase>, k_neighbors: usize) -> Self {
        Self {
            test_cases,
            k_neighbors,
        }
    }
}

impl Benchmark for CommunityPredictionBenchmark {
    fn name(&self) -> &str {
        "community_prediction"
    }

    fn run(&self, embeddings: &[NodeEmbeddings], source: &EmbeddingSource) -> MetricSet {
        let predictions = self.predict_all(embeddings, source);

        // Accuracy
        let correct = predictions.iter().filter(|(p, a)| p == a).count();
        let accuracy = if predictions.is_empty() {
            0.0
        } else {
            correct as f64 / predictions.len() as f64
        };

        // Macro-F1: average F1 across all classes
        let macro_f1 = compute_macro_f1(&predictions);

        MetricSet {
            f1: Some(macro_f1),
            accuracy: Some(accuracy),
            ..Default::default()
        }
    }

    fn per_query_scores(
        &self,
        embeddings: &[NodeEmbeddings],
        source: &EmbeddingSource,
    ) -> Vec<f64> {
        let predictions = self.predict_all(embeddings, source);
        // Per-query: 1.0 if correct, 0.0 if wrong (used for bootstrap CI on accuracy)
        predictions
            .iter()
            .map(|(predicted, actual)| if predicted == actual { 1.0 } else { 0.0 })
            .collect()
    }
}

impl CommunityPredictionBenchmark {
    /// Predict community for each test case via K-NN majority vote.
    /// Returns Vec<(predicted_community, actual_community)>.
    fn predict_all(
        &self,
        embeddings: &[NodeEmbeddings],
        source: &EmbeddingSource,
    ) -> Vec<(usize, usize)> {
        let emb_map = build_embedding_map(embeddings, source);
        let community_map: HashMap<String, usize> = self
            .test_cases
            .iter()
            .map(|tc| (tc.node_id.clone(), tc.community_id))
            .collect();

        self.test_cases
            .iter()
            .filter_map(|tc| {
                let query_emb = emb_map.get(&tc.node_id)?;

                let mut scored: Vec<(String, f64)> = emb_map
                    .iter()
                    .filter(|(id, _)| **id != tc.node_id)
                    .map(|(id, emb)| (id.clone(), cosine_similarity(query_emb, emb)))
                    .collect();

                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut votes: HashMap<usize, usize> = HashMap::new();
                for (id, _) in scored.iter().take(self.k_neighbors) {
                    if let Some(&comm) = community_map.get(id) {
                        *votes.entry(comm).or_default() += 1;
                    }
                }

                let predicted = votes
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(comm, _)| *comm)
                    .unwrap_or(usize::MAX);

                Some((predicted, tc.community_id))
            })
            .collect()
    }
}

/// Compute macro-F1: average of per-class F1 scores.
fn compute_macro_f1(predictions: &[(usize, usize)]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }

    // Collect all unique classes
    let mut classes: Vec<usize> = predictions
        .iter()
        .flat_map(|(p, a)| [*p, *a])
        .collect();
    classes.sort();
    classes.dedup();

    if classes.is_empty() {
        return 0.0;
    }

    // Per-class F1
    let mut f1_sum = 0.0;
    for &cls in &classes {
        let tp = predictions
            .iter()
            .filter(|(p, a)| *p == cls && *a == cls)
            .count() as f64;
        let fp = predictions
            .iter()
            .filter(|(p, a)| *p == cls && *a != cls)
            .count() as f64;
        let fn_ = predictions
            .iter()
            .filter(|(p, a)| *p != cls && *a == cls)
            .count() as f64;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        f1_sum += f1;
    }

    f1_sum / classes.len() as f64
}

// ---------------------------------------------------------------------------
// 5. Co-change Prediction Benchmark
// ---------------------------------------------------------------------------

pub struct CoChangePredictionBenchmark {
    pub test_cases: Vec<CoChangeTestCase>,
}

impl CoChangePredictionBenchmark {
    pub fn new(test_cases: Vec<CoChangeTestCase>) -> Self {
        Self { test_cases }
    }
}

impl Benchmark for CoChangePredictionBenchmark {
    fn name(&self) -> &str {
        "co_change_prediction"
    }

    fn run(&self, embeddings: &[NodeEmbeddings], source: &EmbeddingSource) -> MetricSet {
        let emb_map = build_embedding_map(embeddings, source);

        let scores: Vec<(f64, bool)> = self
            .test_cases
            .iter()
            .filter_map(|tc| {
                let emb_a = emb_map.get(&tc.file_a)?;
                let emb_b = emb_map.get(&tc.file_b)?;
                let sim = cosine_similarity(emb_a, emb_b);
                Some((sim, tc.co_changes))
            })
            .collect();

        let auc = metrics::auc_roc(&scores);

        MetricSet {
            auc_roc: Some(auc),
            ..Default::default()
        }
    }

    fn per_query_scores(
        &self,
        embeddings: &[NodeEmbeddings],
        source: &EmbeddingSource,
    ) -> Vec<f64> {
        let emb_map = build_embedding_map(embeddings, source);

        self.test_cases
            .iter()
            .filter_map(|tc| {
                let emb_a = emb_map.get(&tc.file_a)?;
                let emb_b = emb_map.get(&tc.file_b)?;
                let sim = cosine_similarity(emb_a, emb_b);
                // Score: how well this pair's similarity matches its label
                if tc.co_changes {
                    Some(sim) // higher sim = better for positive pairs
                } else {
                    Some(1.0 - sim) // lower sim = better for negative pairs
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Full benchmark report comparing embedding sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnBenchmarkReport {
    pub generated_at: String,
    pub config: GnnBenchmarkConfig,
    pub node_count: usize,
    pub results: Vec<SourceResults>,
    pub comparisons: Vec<ComparisonResult>,
    pub summary: BenchmarkSummary,
}

/// Per-source results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceResults {
    pub source: String,
    pub benchmarks: HashMap<String, MetricSet>,
}

/// Config for the benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnBenchmarkConfig {
    pub sources: Vec<String>,
    pub bootstrap_resamples: usize,
    pub permutation_tests: usize,
    pub seed: u64,
}

impl Default for GnnBenchmarkConfig {
    fn default() -> Self {
        Self {
            sources: vec![
                "Voyage".to_string(),
                "GNN".to_string(),
                "Concat".to_string(),
            ],
            bootstrap_resamples: 1000,
            permutation_tests: 10_000,
            seed: 42,
        }
    }
}

/// Overall summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Number of tasks where GNN improves ≥15% over Voyage
    pub gnn_wins: usize,
    /// Total benchmark tasks
    pub total_tasks: usize,
    /// GO / NO-GO / CONDITIONAL
    pub verdict: String,
    pub rationale: String,
}

/// Run all benchmarks and produce a full comparison report.
pub fn run_benchmarks(
    benchmarks: &[Box<dyn Benchmark>],
    embeddings: &[NodeEmbeddings],
    sources: &[EmbeddingSource],
    config: &GnnBenchmarkConfig,
) -> GnnBenchmarkReport {
    info!(
        "Running {} benchmarks × {} sources on {} nodes",
        benchmarks.len(),
        sources.len(),
        embeddings.len()
    );

    // 1. Run all benchmarks for each source
    let mut results: Vec<SourceResults> = Vec::new();
    let mut per_query: HashMap<(String, String), Vec<f64>> = HashMap::new();

    for source in sources {
        let mut bench_results = HashMap::new();
        for bench in benchmarks {
            let metric_set = bench.run(embeddings, source);
            let scores = bench.per_query_scores(embeddings, source);

            per_query.insert(
                (source.name().to_string(), bench.name().to_string()),
                scores,
            );
            bench_results.insert(bench.name().to_string(), metric_set);
        }

        results.push(SourceResults {
            source: source.name().to_string(),
            benchmarks: bench_results,
        });
    }

    // 2. Compute pairwise comparisons (GNN vs Voyage, Concat vs Voyage)
    let mut comparisons = Vec::new();
    let voyage = EmbeddingSource::Voyage;

    for source in sources {
        if *source == voyage {
            continue;
        }

        for bench in benchmarks {
            let key_a = (source.name().to_string(), bench.name().to_string());
            let key_b = (voyage.name().to_string(), bench.name().to_string());

            if let (Some(scores_a), Some(scores_b)) =
                (per_query.get(&key_a), per_query.get(&key_b))
            {
                let mean_a = if scores_a.is_empty() {
                    0.0
                } else {
                    scores_a.iter().sum::<f64>() / scores_a.len() as f64
                };
                let mean_b = if scores_b.is_empty() {
                    0.0
                } else {
                    scores_b.iter().sum::<f64>() / scores_b.len() as f64
                };

                let ci = bootstrap_ci(scores_a, scores_b, config.bootstrap_resamples, config.seed);
                let p_value = paired_permutation_test(
                    scores_a,
                    scores_b,
                    config.permutation_tests,
                    config.seed + 1,
                );

                comparisons.push(ComparisonResult {
                    metric_name: bench.name().to_string(),
                    source_a: source.name().to_string(),
                    source_b: voyage.name().to_string(),
                    value_a: mean_a,
                    value_b: mean_b,
                    diff: mean_a - mean_b,
                    diff_ci_95: ci,
                    p_value,
                    significant: p_value < 0.05,
                });
            }
        }
    }

    // 3. Compute summary (count GNN wins ≥15%)
    let gnn_comparisons: Vec<&ComparisonResult> = comparisons
        .iter()
        .filter(|c| c.source_a == "GNN")
        .collect();

    let gnn_wins = gnn_comparisons
        .iter()
        .filter(|c| {
            let relative_improvement = if c.value_b.abs() > 1e-10 {
                c.diff / c.value_b
            } else {
                c.diff
            };
            relative_improvement >= 0.15 && c.significant
        })
        .count();

    let total_tasks = gnn_comparisons.len();
    let (verdict, rationale) = compute_verdict(gnn_wins, total_tasks, &gnn_comparisons);

    let summary = BenchmarkSummary {
        gnn_wins,
        total_tasks,
        verdict,
        rationale,
    };

    GnnBenchmarkReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        config: config.clone(),
        node_count: embeddings.len(),
        results,
        comparisons,
        summary,
    }
}

fn compute_verdict(
    gnn_wins: usize,
    total: usize,
    comparisons: &[&ComparisonResult],
) -> (String, String) {
    if total == 0 {
        return ("N/A".to_string(), "No benchmarks ran.".to_string());
    }

    let win_ratio = gnn_wins as f64 / total as f64;
    let any_significant_improvement = comparisons.iter().any(|c| c.diff > 0.0 && c.significant);

    if gnn_wins >= 3 {
        (
            "GO".to_string(),
            format!(
                "GNN improves ≥15% on {}/{} tasks (threshold: 3). \
                 Structural embeddings provide significant value over Voyage alone.",
                gnn_wins, total
            ),
        )
    } else if any_significant_improvement {
        (
            "CONDITIONAL GO".to_string(),
            format!(
                "GNN improves ≥15% on only {}/{} tasks, but shows significant improvement on some. \
                 Recommend Concat/Fused mode rather than GNN standalone.",
                gnn_wins, total
            ),
        )
    } else {
        (
            "NO GO".to_string(),
            format!(
                "GNN wins: {}/{}. Win ratio: {:.0}%. No significant improvement detected. \
                 Structural encoding may need more training data or architectural changes.",
                gnn_wins, total, win_ratio * 100.0
            ),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_embedding_map(
    embeddings: &[NodeEmbeddings],
    source: &EmbeddingSource,
) -> HashMap<String, Vec<f32>> {
    embeddings
        .iter()
        .map(|ne| (ne.node_id.clone(), ne.get(source)))
        .collect()
}

/// Generate synthetic benchmark data for testing.
///
/// Creates N nodes with random Voyage (768d) and GNN (256d) embeddings,
/// plus ground-truth labels based on known structure.
pub fn generate_synthetic_data(
    num_nodes: usize,
    num_communities: usize,
    seed: u64,
) -> (
    Vec<NodeEmbeddings>,
    Vec<RetrievalTestCase>,
    Vec<ImpactTestCase>,
    Vec<RouteTestCase>,
    Vec<CommunityTestCase>,
    Vec<CoChangeTestCase>,
) {
    let mut rng = SimpleRng::new(seed);

    // Generate community centers (both in Voyage and GNN space)
    let voyage_centers: Vec<Vec<f32>> = (0..num_communities)
        .map(|_| random_embedding(&mut rng, 768))
        .collect();
    let gnn_centers: Vec<Vec<f32>> = (0..num_communities)
        .map(|_| random_embedding(&mut rng, 256))
        .collect();

    // Assign nodes to communities and generate embeddings
    let mut embeddings = Vec::with_capacity(num_nodes);
    let mut community_labels = Vec::with_capacity(num_nodes);

    for i in 0..num_nodes {
        let comm = i % num_communities;
        community_labels.push(comm);

        // Voyage: perturb community center (less structural signal)
        let voyage = perturbed_embedding(&mut rng, &voyage_centers[comm], 0.3);
        // GNN: perturb community center (more structural signal for structural tasks)
        let gnn = perturbed_embedding(&mut rng, &gnn_centers[comm], 0.15);

        embeddings.push(NodeEmbeddings {
            node_id: format!("node_{}", i),
            node_type: "File".to_string(),
            voyage,
            gnn,
        });
    }

    // Generate retrieval test cases (query → same-community nodes)
    let retrieval: Vec<RetrievalTestCase> = (0..num_nodes.min(20))
        .map(|i| {
            let comm = community_labels[i];
            let relevant: Vec<String> = community_labels
                .iter()
                .enumerate()
                .filter(|&(j, &c)| j != i && c == comm)
                .map(|(j, _)| format!("node_{}", j))
                .take(5)
                .collect();

            RetrievalTestCase {
                query_id: format!("node_{}", i),
                relevant_ids: relevant,
            }
        })
        .collect();

    // Generate impact test cases (neighbors in same community)
    let impact: Vec<ImpactTestCase> = (0..num_nodes.min(20))
        .map(|i| {
            let comm = community_labels[i];
            let impacted: Vec<String> = community_labels
                .iter()
                .enumerate()
                .filter(|&(j, &c)| j != i && c == comm)
                .map(|(j, _)| format!("node_{}", j))
                .take(3)
                .collect();

            ImpactTestCase {
                modified_id: format!("node_{}", i),
                impacted_ids: impacted,
            }
        })
        .collect();

    // Generate route test cases
    let route: Vec<RouteTestCase> = (0..num_nodes.min(15))
        .map(|i| {
            let comm = community_labels[i];
            // Correct action: a node from the same community
            let correct_idx = community_labels
                .iter()
                .enumerate()
                .find(|&(j, &c)| j != i && c == comm)
                .map(|(j, _)| j)
                .unwrap_or(0);

            let mut candidates: Vec<(String, String)> = vec![(
                format!("action_{}", correct_idx),
                format!("node_{}", correct_idx),
            )];
            // Add some wrong candidates from other communities
            for j in 0..3 {
                let wrong_idx = (i + j * num_communities / 2 + 1) % num_nodes;
                if wrong_idx != correct_idx && wrong_idx != i {
                    candidates.push((
                        format!("action_{}", wrong_idx),
                        format!("node_{}", wrong_idx),
                    ));
                }
            }

            RouteTestCase {
                query_embedding_id: format!("node_{}", i),
                correct_first_action: format!("action_{}", correct_idx),
                candidates,
            }
        })
        .collect();

    // Community test cases
    let community: Vec<CommunityTestCase> = community_labels
        .iter()
        .enumerate()
        .map(|(i, &comm)| CommunityTestCase {
            node_id: format!("node_{}", i),
            community_id: comm,
        })
        .collect();

    // Co-change test cases (same community → co-change, different → no)
    let mut cochange = Vec::new();
    for i in 0..num_nodes.min(30) {
        for j in (i + 1)..num_nodes.min(30) {
            cochange.push(CoChangeTestCase {
                file_a: format!("node_{}", i),
                file_b: format!("node_{}", j),
                co_changes: community_labels[i] == community_labels[j],
            });
        }
    }

    (embeddings, retrieval, impact, route, community, cochange)
}

/// Generate L2-normalized random embedding.
fn random_embedding(rng: &mut SimpleRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Generate a perturbed (similar) embedding.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() {
        let (embeddings, retrieval, impact, route, community, cochange) =
            generate_synthetic_data(50, 5, 42);

        assert_eq!(embeddings.len(), 50);
        assert_eq!(embeddings[0].voyage.len(), 768);
        assert_eq!(embeddings[0].gnn.len(), 256);
        assert!(!retrieval.is_empty());
        assert!(!impact.is_empty());
        assert!(!route.is_empty());
        assert_eq!(community.len(), 50);
        assert!(!cochange.is_empty());
    }

    #[test]
    fn test_code_retrieval_benchmark() {
        let (embeddings, retrieval, _, _, _, _) = generate_synthetic_data(50, 5, 42);

        let bench = CodeRetrievalBenchmark::new(retrieval);

        // GNN should perform better on community-based retrieval
        let gnn_metrics = bench.run(&embeddings, &EmbeddingSource::Gnn);
        let voyage_metrics = bench.run(&embeddings, &EmbeddingSource::Voyage);

        assert!(gnn_metrics.recall_at_5.unwrap() > 0.0);
        assert!(voyage_metrics.recall_at_5.unwrap() > 0.0);

        // GNN should have higher recall (less noise in community-structured data)
        info!(
            "Retrieval R@5: GNN={:.3}, Voyage={:.3}",
            gnn_metrics.recall_at_5.unwrap(),
            voyage_metrics.recall_at_5.unwrap()
        );
    }

    #[test]
    fn test_community_prediction_benchmark() {
        let (embeddings, _, _, _, community, _) = generate_synthetic_data(50, 5, 42);

        let bench = CommunityPredictionBenchmark::new(community, 5);

        let gnn_metrics = bench.run(&embeddings, &EmbeddingSource::Gnn);
        let voyage_metrics = bench.run(&embeddings, &EmbeddingSource::Voyage);

        assert!(gnn_metrics.accuracy.unwrap() > 0.0);
        assert!(voyage_metrics.accuracy.unwrap() > 0.0);

        info!(
            "Community F1: GNN={:.3}, Voyage={:.3}",
            gnn_metrics.f1.unwrap(),
            voyage_metrics.f1.unwrap()
        );
    }

    #[test]
    fn test_co_change_prediction_benchmark() {
        let (embeddings, _, _, _, _, cochange) = generate_synthetic_data(30, 3, 42);

        let bench = CoChangePredictionBenchmark::new(cochange);

        let gnn_metrics = bench.run(&embeddings, &EmbeddingSource::Gnn);
        let voyage_metrics = bench.run(&embeddings, &EmbeddingSource::Voyage);

        // Both should produce valid AUC-ROC
        let gnn_auc = gnn_metrics.auc_roc.unwrap();
        let voyage_auc = voyage_metrics.auc_roc.unwrap();
        assert!(gnn_auc >= 0.0 && gnn_auc <= 1.0);
        assert!(voyage_auc >= 0.0 && voyage_auc <= 1.0);

        info!("Co-change AUC: GNN={:.3}, Voyage={:.3}", gnn_auc, voyage_auc);
    }

    #[test]
    fn test_full_benchmark_run() {
        let (embeddings, retrieval, impact, route, community, cochange) =
            generate_synthetic_data(50, 5, 42);

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
            bootstrap_resamples: 100, // fewer for speed in tests
            permutation_tests: 500,
            ..Default::default()
        };

        let report = run_benchmarks(&benchmarks, &embeddings, &sources, &config);

        // Validate report structure
        assert_eq!(report.results.len(), 3); // 3 sources
        assert_eq!(report.summary.total_tasks, 5);
        assert!(!report.summary.verdict.is_empty());

        // Report serializes to JSON
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("verdict"));
        assert!(json.contains("code_retrieval"));
        assert!(json.contains("co_change_prediction"));

        info!(
            "Benchmark verdict: {} — GNN wins: {}/{}",
            report.summary.verdict, report.summary.gnn_wins, report.summary.total_tasks
        );
        info!("Rationale: {}", report.summary.rationale);
    }

    #[test]
    fn test_macro_f1_perfect() {
        // All predictions correct: 3 classes, each predicted perfectly
        let predictions = vec![
            (0, 0), (0, 0),
            (1, 1), (1, 1),
            (2, 2), (2, 2),
        ];
        let f1 = compute_macro_f1(&predictions);
        assert!((f1 - 1.0).abs() < 1e-10, "Perfect predictions → macro-F1=1.0, got {}", f1);
    }

    #[test]
    fn test_macro_f1_partial() {
        // Class 0: TP=2, FP=1, FN=0 → P=2/3, R=1.0, F1=0.8
        // Class 1: TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1=2/3
        let predictions = vec![
            (0, 0), (0, 0), (0, 1), // 2 correct class-0, 1 class-1 misclassified as 0
            (1, 1),                  // 1 correct class-1
        ];
        let f1 = compute_macro_f1(&predictions);
        let expected = (0.8 + 2.0 / 3.0) / 2.0; // avg of per-class F1
        assert!(
            (f1 - expected).abs() < 1e-10,
            "Partial predictions → macro-F1={:.4}, got {:.4}",
            expected,
            f1
        );
    }

    #[test]
    fn test_macro_f1_all_wrong() {
        // Every prediction is wrong class
        let predictions = vec![(0, 1), (1, 0)];
        let f1 = compute_macro_f1(&predictions);
        assert!((f1 - 0.0).abs() < 1e-10, "All wrong → macro-F1=0.0, got {}", f1);
    }

    #[test]
    fn test_embedding_source_get() {
        let ne = NodeEmbeddings {
            node_id: "test".to_string(),
            node_type: "File".to_string(),
            voyage: vec![1.0; 768],
            gnn: vec![2.0; 256],
        };

        assert_eq!(ne.get(&EmbeddingSource::Voyage).len(), 768);
        assert_eq!(ne.get(&EmbeddingSource::Gnn).len(), 256);
        assert_eq!(ne.get(&EmbeddingSource::Concatenated).len(), 768 + 256);
        // Fused = weighted concat: (768 * vw) ++ (256 * w) → 1024d
        assert_eq!(
            ne.get(&EmbeddingSource::Fused { gnn_weight: 0.5 }).len(),
            768 + 256
        );
        // Verify weighting: voyage scaled by 0.5, gnn scaled by 0.5
        let fused = ne.get(&EmbeddingSource::Fused { gnn_weight: 0.5 });
        assert!((fused[0] - 0.5).abs() < 1e-6, "Voyage component should be 1.0 * 0.5 = 0.5");
        assert!(
            (fused[768] - 1.0).abs() < 1e-6,
            "GNN component should be 2.0 * 0.5 = 1.0"
        );

        // Edge case: gnn_weight = 0.0 → 100% Voyage, 0% GNN
        let fused_voyage_only = ne.get(&EmbeddingSource::Fused { gnn_weight: 0.0 });
        assert_eq!(fused_voyage_only.len(), 768 + 256);
        assert!((fused_voyage_only[0] - 1.0).abs() < 1e-6, "Voyage should be unscaled");
        assert!((fused_voyage_only[768]).abs() < 1e-6, "GNN should be zeroed");

        // Edge case: gnn_weight = 1.0 → 0% Voyage, 100% GNN
        let fused_gnn_only = ne.get(&EmbeddingSource::Fused { gnn_weight: 1.0 });
        assert_eq!(fused_gnn_only.len(), 768 + 256);
        assert!((fused_gnn_only[0]).abs() < 1e-6, "Voyage should be zeroed");
        assert!((fused_gnn_only[768] - 2.0).abs() < 1e-6, "GNN should be unscaled");
    }
}
