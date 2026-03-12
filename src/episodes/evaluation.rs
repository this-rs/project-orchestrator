//! Episodic Evaluation Metrics — M2 Plan 2
//!
//! Implements 3 episodic evaluation metrics that extend the-source's
//! 5-factor structural evaluation:
//!
//! 1. **Lesson Comprehensibility** — Is the lesson understandable without context?
//! 2. **Process Replay Fidelity** — Can the FSM trace be reproduced locally?
//! 3. **Gap-Closing Precision** — Ratio of episodes that wire to local entities.
//!
//! Plus a composite score with configurable weights.

use crate::episodes::{PortableEpisode, PortableLesson};
use serde::{Deserialize, Serialize};

// ============================================================================
// Metric Scores
// ============================================================================

/// Score for a single episode on a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeScore {
    /// Episode index in the artifact
    pub episode_index: usize,
    /// Score 0.0 - 1.0
    pub score: f64,
    /// Human-readable explanation
    pub explanation: String,
}

/// Result of evaluating an entire artifact on all 3 metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEvaluationReport {
    /// Metric 1: Lesson comprehensibility scores
    pub comprehensibility: MetricResult,
    /// Metric 2: Process replay fidelity scores
    pub replay_fidelity: MetricResult,
    /// Metric 3: Gap-closing precision
    pub gap_closing: MetricResult,
    /// Composite episodic score (weighted average)
    pub composite: CompositeScore,
    /// Number of episodes evaluated
    pub episodes_evaluated: usize,
    /// Number of episodes with lessons
    pub episodes_with_lessons: usize,
}

/// Aggregated result for a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Name of the metric
    pub name: String,
    /// Per-episode scores
    pub scores: Vec<EpisodeScore>,
    /// Average score across all evaluated episodes
    pub average: f64,
    /// Min score
    pub min: f64,
    /// Max score
    pub max: f64,
    /// Standard deviation
    pub std_dev: f64,
}

/// Composite score with weight breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeScore {
    /// Final composite score 0.0 - 1.0
    pub score: f64,
    /// Weight applied to comprehensibility
    pub comprehensibility_weight: f64,
    /// Weight applied to replay fidelity
    pub replay_fidelity_weight: f64,
    /// Weight applied to gap-closing
    pub gap_closing_weight: f64,
    /// Comparison with the-source structural benchmark
    pub vs_structural_benchmark: StructuralComparison,
}

/// Comparison with the-source's structural benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralComparison {
    /// the-source composite structural score (from their benchmark)
    pub structural_score: f64,
    /// Our episodic composite score
    pub episodic_score: f64,
    /// Delta: episodic - structural
    pub delta: f64,
    /// Whether episodic beats structural
    pub episodic_wins: bool,
    /// Human-readable conclusion
    pub conclusion: String,
}

// ============================================================================
// Comprehensibility Scorer (trait for LLM abstraction)
// ============================================================================

/// Trait for scoring lesson comprehensibility.
///
/// In production, this calls an LLM. In tests, we use a heuristic scorer.
pub trait ComprehensibilityScorer: Send + Sync {
    /// Score a lesson's comprehensibility (0.0 - 1.0).
    ///
    /// The scorer should evaluate whether the lesson's abstract_pattern
    /// is understandable without additional context.
    fn score_lesson(&self, lesson: &PortableLesson) -> f64;
}

/// Heuristic-based comprehensibility scorer (no LLM needed).
///
/// Scores based on:
/// - Length (too short = vague, too long = complex)
/// - Has domain tags (improves routing)
/// - Portability layer (higher = more universal = more comprehensible)
/// - Contains actionable keywords ("always", "never", "when", "use")
pub struct HeuristicComprehensibilityScorer;

impl ComprehensibilityScorer for HeuristicComprehensibilityScorer {
    fn score_lesson(&self, lesson: &PortableLesson) -> f64 {
        let mut score = 0.0;
        let pattern = &lesson.abstract_pattern;

        // Length score: 20-200 chars is ideal
        let len = pattern.len();
        let length_score = if len < 10 {
            0.2
        } else if len < 20 {
            0.5
        } else if len <= 200 {
            0.9
        } else if len <= 500 {
            0.7
        } else {
            0.4
        };
        score += length_score * 0.3;

        // Domain tags score
        let tags_score = if lesson.domain_tags.is_empty() {
            0.3
        } else if lesson.domain_tags.len() <= 3 {
            0.9
        } else {
            0.7
        };
        score += tags_score * 0.2;

        // Portability score
        let portability_score = match lesson.portability_layer {
            3 => 1.0, // universal
            2 => 0.7, // language-specific
            1 => 0.4, // project-specific
            _ => 0.3,
        };
        score += portability_score * 0.2;

        // Actionable keywords
        let lower = pattern.to_lowercase();
        let actionable_keywords = [
            "always", "never", "when", "use", "avoid", "prefer", "must", "should", "ensure",
            "check", "before", "after", "instead",
        ];
        let keyword_count = actionable_keywords
            .iter()
            .filter(|kw| lower.contains(**kw))
            .count();
        let actionable_score = (keyword_count as f64 * 0.25).min(1.0);
        score += actionable_score * 0.3;

        score.min(1.0)
    }
}

// ============================================================================
// Metric 1: Lesson Comprehensibility
// ============================================================================

/// Evaluate lesson comprehensibility for all episodes in an artifact.
pub fn evaluate_comprehensibility(
    episodes: &[PortableEpisode],
    scorer: &dyn ComprehensibilityScorer,
) -> MetricResult {
    let mut scores = Vec::new();

    for (i, ep) in episodes.iter().enumerate() {
        if let Some(lesson) = &ep.lesson {
            let score = scorer.score_lesson(lesson);
            scores.push(EpisodeScore {
                episode_index: i,
                score,
                explanation: format!(
                    "Lesson '{}...' scored {:.2} on comprehensibility (pattern_len={}, tags={}, portability={})",
                    &lesson.abstract_pattern[..lesson.abstract_pattern.len().min(60)],
                    score,
                    lesson.abstract_pattern.len(),
                    lesson.domain_tags.len(),
                    lesson.portability_layer,
                ),
            });
        }
    }

    compute_metric_result("Lesson Comprehensibility".to_string(), scores)
}

// ============================================================================
// Metric 2: Process Replay Fidelity
// ============================================================================

/// Evaluate process replay fidelity.
///
/// For each episode, check how many of its FSM states match known protocol
/// state names. In a real scenario, `known_states` comes from the local
/// protocol registry. Higher match ratio = higher fidelity.
pub fn evaluate_replay_fidelity(
    episodes: &[PortableEpisode],
    known_states: &[String],
) -> MetricResult {
    let known_set: std::collections::HashSet<&str> =
        known_states.iter().map(|s| s.as_str()).collect();

    let mut scores = Vec::new();

    for (i, ep) in episodes.iter().enumerate() {
        let states = &ep.process.states_visited;
        if states.is_empty() {
            scores.push(EpisodeScore {
                episode_index: i,
                score: 0.0,
                explanation: "No states visited — cannot evaluate replay fidelity".to_string(),
            });
            continue;
        }

        let matched = states
            .iter()
            .filter(|s| known_set.contains(s.as_str()))
            .count();
        let fidelity = matched as f64 / states.len() as f64;

        scores.push(EpisodeScore {
            episode_index: i,
            score: fidelity,
            explanation: format!(
                "{}/{} states match local protocols (matched: {:?})",
                matched,
                states.len(),
                states
                    .iter()
                    .filter(|s| known_set.contains(s.as_str()))
                    .collect::<Vec<_>>(),
            ),
        });
    }

    compute_metric_result("Process Replay Fidelity".to_string(), scores)
}

// ============================================================================
// Metric 3: Gap-Closing Precision
// ============================================================================

/// Evaluate gap-closing precision.
///
/// `linked_episode_indices` = indices of episodes that successfully wired
/// to local entities (via LINKED_TO after import). This is determined by
/// checking which imported episode-notes have LINKED_TO relations.
///
/// Score = |linked| / |total episodes received|
pub fn evaluate_gap_closing(
    total_episodes: usize,
    linked_episode_indices: &[usize],
) -> MetricResult {
    let linked_count = linked_episode_indices.len();
    let precision = if total_episodes > 0 {
        linked_count as f64 / total_episodes as f64
    } else {
        0.0
    };

    let mut scores = Vec::new();
    for i in 0..total_episodes {
        let is_linked = linked_episode_indices.contains(&i);
        scores.push(EpisodeScore {
            episode_index: i,
            score: if is_linked { 1.0 } else { 0.0 },
            explanation: if is_linked {
                "Episode wired to local entity via LINKED_TO".to_string()
            } else {
                "Episode did not wire to any local entity".to_string()
            },
        });
    }

    let mut result = compute_metric_result("Gap-Closing Precision".to_string(), scores);
    // Override average with the actual precision ratio
    result.average = precision;
    result
}

// ============================================================================
// Composite Score
// ============================================================================

/// Default weights: comprehensibility 40%, replay_fidelity 30%, gap_closing 30%
pub const DEFAULT_COMPREHENSIBILITY_WEIGHT: f64 = 0.4;
pub const DEFAULT_REPLAY_FIDELITY_WEIGHT: f64 = 0.3;
pub const DEFAULT_GAP_CLOSING_WEIGHT: f64 = 0.3;

/// the-source structural benchmark (from their evaluation: 76.3% composite)
pub const THE_SOURCE_STRUCTURAL_BENCHMARK: f64 = 0.763;

/// Compute the full episodic evaluation report.
pub fn evaluate_artifact(
    episodes: &[PortableEpisode],
    scorer: &dyn ComprehensibilityScorer,
    known_states: &[String],
    linked_episode_indices: &[usize],
) -> EpisodicEvaluationReport {
    let comprehensibility = evaluate_comprehensibility(episodes, scorer);
    let replay_fidelity = evaluate_replay_fidelity(episodes, known_states);
    let gap_closing = evaluate_gap_closing(episodes.len(), linked_episode_indices);

    let episodes_with_lessons = episodes.iter().filter(|ep| ep.lesson.is_some()).count();

    let composite_score = comprehensibility.average * DEFAULT_COMPREHENSIBILITY_WEIGHT
        + replay_fidelity.average * DEFAULT_REPLAY_FIDELITY_WEIGHT
        + gap_closing.average * DEFAULT_GAP_CLOSING_WEIGHT;

    let delta = composite_score - THE_SOURCE_STRUCTURAL_BENCHMARK;

    let conclusion = if delta > 0.05 {
        format!(
            "Episodic evaluation ({:.1}%) significantly outperforms structural benchmark ({:.1}%). \
             The episodic layer adds measurable value for cross-instance knowledge transfer.",
            composite_score * 100.0,
            THE_SOURCE_STRUCTURAL_BENCHMARK * 100.0,
        )
    } else if delta > -0.05 {
        format!(
            "Episodic evaluation ({:.1}%) is comparable to structural benchmark ({:.1}%). \
             The episodic layer matches structural quality while adding semantic depth.",
            composite_score * 100.0,
            THE_SOURCE_STRUCTURAL_BENCHMARK * 100.0,
        )
    } else {
        format!(
            "Episodic evaluation ({:.1}%) underperforms structural benchmark ({:.1}%). \
             Gap-closing and replay fidelity need improvement for cross-instance value.",
            composite_score * 100.0,
            THE_SOURCE_STRUCTURAL_BENCHMARK * 100.0,
        )
    };

    EpisodicEvaluationReport {
        comprehensibility,
        replay_fidelity,
        gap_closing,
        composite: CompositeScore {
            score: composite_score,
            comprehensibility_weight: DEFAULT_COMPREHENSIBILITY_WEIGHT,
            replay_fidelity_weight: DEFAULT_REPLAY_FIDELITY_WEIGHT,
            gap_closing_weight: DEFAULT_GAP_CLOSING_WEIGHT,
            vs_structural_benchmark: StructuralComparison {
                structural_score: THE_SOURCE_STRUCTURAL_BENCHMARK,
                episodic_score: composite_score,
                delta,
                episodic_wins: delta > 0.0,
                conclusion,
            },
        },
        episodes_evaluated: episodes.len(),
        episodes_with_lessons,
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn compute_metric_result(name: String, scores: Vec<EpisodeScore>) -> MetricResult {
    if scores.is_empty() {
        return MetricResult {
            name,
            scores: Vec::new(),
            average: 0.0,
            min: 0.0,
            max: 0.0,
            std_dev: 0.0,
        };
    }

    let values: Vec<f64> = scores.iter().map(|s| s.score).collect();
    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let avg = sum / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    MetricResult {
        name,
        scores,
        average: avg,
        min,
        max,
        std_dev,
    }
}

// ============================================================================
// M2 Final Comparative Report
// ============================================================================

/// The final M2 comparative report combining artifact comparison + episodic evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct M2ComparativeReport {
    /// Part 1: Artifact format comparison (structure vs enriched)
    pub artifact_comparison: crate::episodes::artifact_comparison::ArtifactComparisonReport,
    /// Part 2: Episodic evaluation metrics
    pub episodic_evaluation: EpisodicEvaluationReport,
    /// Part 3: Scenario analysis
    pub scenarios: Vec<ScenarioAnalysis>,
    /// Final conclusion — actionable recommendation
    pub final_conclusion: FinalConclusion,
}

/// Analysis of a specific transfer scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioAnalysis {
    /// Scenario name
    pub name: String,
    /// Description
    pub description: String,
    /// Does Layer 0 (structural) suffice?
    pub structural_sufficient: bool,
    /// Does Layer 1 (episodic) add value?
    pub episodic_adds_value: bool,
    /// Explanation
    pub explanation: String,
}

/// Final actionable conclusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalConclusion {
    /// When Layer 0 (structural only) is enough
    pub when_structural_suffices: String,
    /// When Layer 1 (episodic) is necessary
    pub when_episodic_needed: String,
    /// Cost of the episodic layer
    pub episodic_cost: EpisodicCost,
    /// Recommendation for M3
    pub m3_recommendation: String,
}

/// Cost breakdown of the episodic layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicCost {
    /// Additional JSON size (bytes, approximate)
    pub size_overhead_bytes: usize,
    /// Additional collection latency (ms, approximate)
    pub collection_latency_ms: u64,
    /// LLM calls needed for lesson extraction
    pub llm_calls_per_episode: u32,
    /// Summary
    pub summary: String,
}

/// Produce the final M2 comparative report.
///
/// This is the milestone deliverable — answers "what does episodic add?"
pub fn produce_m2_report(
    enriched: &crate::api::episode_handlers::EnrichedArtifact,
    the_source: &crate::episodes::artifact_comparison::DistilledArtifactV1,
    evaluation: &EpisodicEvaluationReport,
) -> M2ComparativeReport {
    let artifact_comparison =
        crate::episodes::artifact_comparison::compare_artifacts(enriched, the_source);

    let enriched_json_size = serde_json::to_string(enriched)
        .map(|s| s.len())
        .unwrap_or(0);
    let the_source_json_size = serde_json::to_string(the_source)
        .map(|s| s.len())
        .unwrap_or(0);
    let size_overhead = enriched_json_size.saturating_sub(the_source_json_size);

    let scenarios = vec![
        ScenarioAnalysis {
            name: "Intra-project transfer (same codebase)".to_string(),
            description:
                "Transferring knowledge within the same project, e.g., after a team member change."
                    .to_string(),
            structural_sufficient: true,
            episodic_adds_value: true,
            explanation: "Structural edges already capture file relationships. \
                         Episodic adds WHY decisions were made — critical for onboarding. \
                         Lessons like 'always batch UNWIND' prevent repeating past mistakes."
                .to_string(),
        },
        ScenarioAnalysis {
            name: "Inter-project transfer (same language)".to_string(),
            description:
                "Transferring knowledge between two Rust projects on the same PO instance."
                    .to_string(),
            structural_sufficient: false,
            episodic_adds_value: true,
            explanation: "Structural edges are project-specific (file paths don't transfer). \
                         Episodic lessons with portability_layer ≥ 2 transfer well — \
                         'When modifying a trait, check find_trait_implementations first' \
                         applies to any Rust project. Gap-closing precision measures this."
                .to_string(),
        },
        ScenarioAnalysis {
            name: "Cross-domain transfer (different language)".to_string(),
            description: "Transferring knowledge from a Rust project to a TypeScript project."
                .to_string(),
            structural_sufficient: false,
            episodic_adds_value: true,
            explanation: "Structural edges are completely useless (different file system). \
                         Only episodic lessons with portability_layer = 3 (universal) transfer: \
                         'Always prefer batch operations over N+1 loops'. \
                         This is where episodic UNIQUELY adds value."
                .to_string(),
        },
    ];

    let episodic_score = evaluation.composite.score;
    let structural_score = THE_SOURCE_STRUCTURAL_BENCHMARK;

    let m3_recommendation = if episodic_score > structural_score {
        "PROCEED with M3 (P2P Knowledge Exchange). Episodic layer demonstrates measurable value. \
         Priority: implement delta sync for incremental episode exchange, \
         trust scoring weighted by lesson portability_layer, \
         and guided replay for process-heavy episodes."
            .to_string()
    } else if episodic_score > structural_score * 0.9 {
        "PROCEED with M3, but FOCUS on gap-closing and lesson extraction quality. \
         The episodic layer is promising but needs better LLM-driven lesson extraction \
         and more aggressive gap-closing (semantic matching between lessons and local entities)."
            .to_string()
    } else {
        "RECONSIDER M3 scope. Episodic layer underperforms expectations. \
         Possible improvements: better lesson extraction (LLM-driven), \
         richer process traces (include code diffs, not just FSM states), \
         and domain-aware gap-closing."
            .to_string()
    };

    M2ComparativeReport {
        artifact_comparison,
        episodic_evaluation: evaluation.clone(),
        scenarios,
        final_conclusion: FinalConclusion {
            when_structural_suffices: "Intra-project transfers where file paths are stable \
                                       and the receiving agent has access to the same codebase. \
                                       Layer 0 (structural edges) captures WHAT connects to WHAT."
                .to_string(),
            when_episodic_needed: "Inter-project and cross-domain transfers where structural \
                                    edges don't transfer. Also for onboarding scenarios where \
                                    understanding WHY decisions were made is critical. \
                                    Layer 1 (episodic) captures WHY + HOW + LESSON."
                .to_string(),
            episodic_cost: EpisodicCost {
                size_overhead_bytes: size_overhead,
                collection_latency_ms: 50, // estimate: graph queries + assembly
                llm_calls_per_episode: 1,  // for lesson extraction
                summary: format!(
                    "~{}KB overhead per artifact, ~50ms collection per episode, \
                     1 LLM call per episode for lesson extraction. \
                     Cost scales linearly with episode count.",
                    size_overhead / 1024,
                ),
            },
            m3_recommendation,
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::artifact_comparison::generate_fake_enriched;

    #[test]
    fn test_heuristic_comprehensibility_scorer() {
        let scorer = HeuristicComprehensibilityScorer;

        // Good lesson: medium length, actionable, tagged, portable
        let good_lesson = PortableLesson {
            abstract_pattern: "When adding a new graph relation, always create an index and a backfill script for existing data.".to_string(),
            domain_tags: vec!["neo4j".to_string(), "schema-migration".to_string()],
            portability_layer: 2,
            confidence: Some(0.9),
        };
        let good_score = scorer.score_lesson(&good_lesson);
        assert!(
            good_score > 0.6,
            "Good lesson should score >0.6, got {:.2}",
            good_score
        );

        // Poor lesson: too short, no tags, low portability
        let poor_lesson = PortableLesson {
            abstract_pattern: "Fix it".to_string(),
            domain_tags: vec![],
            portability_layer: 1,
            confidence: Some(0.3),
        };
        let poor_score = scorer.score_lesson(&poor_lesson);
        assert!(
            poor_score < 0.5,
            "Poor lesson should score <0.5, got {:.2}",
            poor_score
        );

        // Universal lesson
        let universal_lesson = PortableLesson {
            abstract_pattern: "Always prefer batch operations over N+1 loops when writing to a database — use UNWIND or bulk insert instead of individual queries.".to_string(),
            domain_tags: vec!["performance".to_string(), "database".to_string()],
            portability_layer: 3,
            confidence: Some(0.85),
        };
        let universal_score = scorer.score_lesson(&universal_lesson);
        assert!(
            universal_score > good_score,
            "Universal lesson should score higher than good: {:.2} vs {:.2}",
            universal_score,
            good_score
        );
    }

    #[test]
    fn test_evaluate_comprehensibility() {
        let enriched = generate_fake_enriched(10, 12);
        let scorer = HeuristicComprehensibilityScorer;
        let result = evaluate_comprehensibility(&enriched.episodes, &scorer);

        assert_eq!(result.name, "Lesson Comprehensibility");
        // Only episodes with lessons are scored (8 out of 12)
        assert_eq!(result.scores.len(), 8);
        assert!(
            result.average > 0.5,
            "Average comprehensibility should be >0.5, got {:.2}",
            result.average
        );
        assert!(result.min >= 0.0);
        assert!(result.max <= 1.0);
    }

    #[test]
    fn test_evaluate_replay_fidelity() {
        let enriched = generate_fake_enriched(10, 12);
        let known_states = vec![
            "analyze".to_string(),
            "implement".to_string(),
            "validate".to_string(),
            "done".to_string(),
            "plan".to_string(),
            "execute".to_string(),
            "review".to_string(),
            // Note: "detect", "diagnose", "fix", "start", "process" are NOT known
        ];
        let result = evaluate_replay_fidelity(&enriched.episodes, &known_states);

        assert_eq!(result.name, "Process Replay Fidelity");
        assert_eq!(result.scores.len(), 12);
        // Some episodes should have partial matches
        assert!(result.average > 0.0, "Some states should match");
        assert!(result.average < 1.0, "Not all states should match");

        // Print for documentation
        println!(
            "Replay fidelity: avg={:.2}, min={:.2}, max={:.2}",
            result.average, result.min, result.max
        );
    }

    #[test]
    fn test_evaluate_gap_closing() {
        // 12 episodes, 7 linked
        let result = evaluate_gap_closing(12, &[0, 1, 3, 5, 7, 9, 11]);

        assert_eq!(result.name, "Gap-Closing Precision");
        assert_eq!(result.scores.len(), 12);
        assert!((result.average - 7.0 / 12.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_gap_closing_empty() {
        let result = evaluate_gap_closing(0, &[]);
        assert_eq!(result.average, 0.0);
    }

    #[test]
    fn test_full_evaluation_report() {
        let enriched = generate_fake_enriched(50, 12);
        let scorer = HeuristicComprehensibilityScorer;
        let known_states = vec![
            "analyze".to_string(),
            "implement".to_string(),
            "validate".to_string(),
            "done".to_string(),
            "plan".to_string(),
            "execute".to_string(),
            "review".to_string(),
        ];
        // Simulate: 8 out of 12 episodes linked to local entities
        let linked = vec![0, 1, 2, 4, 5, 7, 9, 11];

        let report = evaluate_artifact(&enriched.episodes, &scorer, &known_states, &linked);

        // Print full report
        let json = serde_json::to_string_pretty(&report).unwrap();
        println!("=== EPISODIC EVALUATION REPORT (M2) ===\n{}", json);

        assert_eq!(report.episodes_evaluated, 12);
        assert_eq!(report.episodes_with_lessons, 8);
        assert!(report.composite.score > 0.0);
        assert!(report.composite.score <= 1.0);

        // Verify weights sum to 1.0
        let weight_sum = report.composite.comprehensibility_weight
            + report.composite.replay_fidelity_weight
            + report.composite.gap_closing_weight;
        assert!((weight_sum - 1.0).abs() < 0.001);

        // Verify structural comparison exists
        assert_eq!(
            report.composite.vs_structural_benchmark.structural_score,
            THE_SOURCE_STRUCTURAL_BENCHMARK
        );
        assert!(!report
            .composite
            .vs_structural_benchmark
            .conclusion
            .is_empty());
    }

    #[test]
    fn test_evaluation_report_serialization() {
        let enriched = generate_fake_enriched(10, 5);
        let scorer = HeuristicComprehensibilityScorer;
        let report = evaluate_artifact(&enriched.episodes, &scorer, &["done".to_string()], &[0, 2]);

        let json = serde_json::to_string_pretty(&report).unwrap();
        let restored: EpisodicEvaluationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.episodes_evaluated, 5);
        assert_eq!(restored.composite.comprehensibility_weight, 0.4);
    }

    #[test]
    fn test_m2_final_comparative_report() {
        use crate::episodes::artifact_comparison::{
            generate_fake_enriched, generate_fake_the_source,
        };

        // Generate realistic artifacts
        let the_source = generate_fake_the_source(2244);
        let enriched = generate_fake_enriched(80, 12);

        // Run evaluation with realistic conditions
        let scorer = HeuristicComprehensibilityScorer;
        let known_states = vec![
            "analyze".to_string(),
            "implement".to_string(),
            "validate".to_string(),
            "done".to_string(),
            "plan".to_string(),
            "execute".to_string(),
            "review".to_string(),
        ];
        let linked = vec![0, 1, 2, 4, 5, 7, 9, 11]; // 8/12 linked

        let evaluation = evaluate_artifact(&enriched.episodes, &scorer, &known_states, &linked);
        let report = produce_m2_report(&enriched, &the_source, &evaluation);

        // Print the full M2 deliverable
        let json = serde_json::to_string_pretty(&report).unwrap();
        println!(
            "=== M2 COMPARATIVE REPORT (FINAL DELIVERABLE) ===\n{}",
            json
        );

        // Verify report structure
        assert_eq!(report.scenarios.len(), 3);
        assert!(report.scenarios[0].structural_sufficient); // intra-project
        assert!(!report.scenarios[1].structural_sufficient); // inter-project
        assert!(!report.scenarios[2].structural_sufficient); // cross-domain
        assert!(report.scenarios.iter().all(|s| s.episodic_adds_value));

        // Verify final conclusion is non-empty
        assert!(!report.final_conclusion.when_structural_suffices.is_empty());
        assert!(!report.final_conclusion.when_episodic_needed.is_empty());
        assert!(!report.final_conclusion.m3_recommendation.is_empty());

        // Verify cost breakdown
        assert!(report.final_conclusion.episodic_cost.collection_latency_ms > 0);
    }

    #[test]
    fn test_m2_report_serialization() {
        use crate::episodes::artifact_comparison::{
            generate_fake_enriched, generate_fake_the_source,
        };

        let the_source = generate_fake_the_source(100);
        let enriched = generate_fake_enriched(20, 5);
        let scorer = HeuristicComprehensibilityScorer;
        let evaluation =
            evaluate_artifact(&enriched.episodes, &scorer, &["done".to_string()], &[0, 2]);
        let report = produce_m2_report(&enriched, &the_source, &evaluation);

        let json = serde_json::to_string_pretty(&report).unwrap();
        let restored: M2ComparativeReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.scenarios.len(), 3);
        assert_eq!(restored.episodic_evaluation.episodes_evaluated, 5);
    }

    #[test]
    fn test_composite_beats_structural_with_good_data() {
        // Create episodes with very good lessons
        let enriched = generate_fake_enriched(50, 12);
        let scorer = HeuristicComprehensibilityScorer;
        // All known states → perfect replay fidelity
        let known_states = vec![
            "analyze".to_string(),
            "implement".to_string(),
            "validate".to_string(),
            "done".to_string(),
            "plan".to_string(),
            "execute".to_string(),
            "review".to_string(),
            "detect".to_string(),
            "diagnose".to_string(),
            "fix".to_string(),
            "start".to_string(),
            "process".to_string(),
        ];
        // All episodes linked → perfect gap-closing
        let all_linked: Vec<usize> = (0..12).collect();

        let report = evaluate_artifact(&enriched.episodes, &scorer, &known_states, &all_linked);

        println!(
            "Composite: {:.1}% vs structural benchmark {:.1}% (delta: {:.1}%)",
            report.composite.score * 100.0,
            THE_SOURCE_STRUCTURAL_BENCHMARK * 100.0,
            report.composite.vs_structural_benchmark.delta * 100.0,
        );

        // With perfect fidelity and gap-closing, episodic should beat structural
        assert!(
            report.composite.vs_structural_benchmark.episodic_wins,
            "With perfect replay fidelity and gap-closing, episodic should beat structural. \
             Got {:.1}% vs {:.1}%",
            report.composite.score * 100.0,
            THE_SOURCE_STRUCTURAL_BENCHMARK * 100.0,
        );
    }
}
