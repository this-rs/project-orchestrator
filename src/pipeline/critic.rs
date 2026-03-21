//! # Mutation Critic — Pre-evaluation of Protocol Mutations
//!
//! Inspired by EvoFSM (2026), this module adds a **Critic** that scores each
//! mutation candidate BEFORE it is applied to a protocol FSM. This reduces
//! wasteful mutation→revert cycles and provides traceability for rejected mutations.
//!
//! The critic evaluates candidates on three axes:
//! 1. **Pattern history** — how often similar patterns were seen (frequency weight)
//! 2. **Structural impact** — number of existing transitions (betweenness proxy)
//! 3. **Coherence** — no duplicate states or transitions
//!
//! # References
//! - EvoFSM (2026): "Controllable Self-Evolution for Deep Research with FSMs"
//!   — separates Flow optimization (protocol structure) from Skill optimization
//!   — uses a critic to gate mutations before application

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::debug;

use crate::neo4j::traits::GraphStore;
use crate::pipeline::evolve::MutationRule;
use crate::pipeline::feedback::DetectedPattern;
use crate::protocol::models::{ProtocolState, ProtocolTransition};

// ─── Types ──────────────────────────────────────────────────────────────────

/// A candidate mutation bundling the rule, source pattern, and protocol context.
///
/// # References
/// - EvoFSM (2026): "Controllable Self-Evolution for Deep Research with FSMs"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationCandidate {
    /// The detected pattern that triggered this candidate.
    pub pattern: DetectedPattern,
    /// Name of the proposed new state.
    pub proposed_state: String,
    /// Description of the proposed state.
    pub proposed_state_description: String,
    /// Proposed transitions (trigger names).
    pub proposed_transitions: Vec<String>,
    /// Current protocol state names (context for coherence check).
    pub protocol_context: Vec<String>,
    /// Current protocol transition count (context for structural impact).
    pub protocol_transition_count: usize,
}

impl MutationCandidate {
    /// Build a candidate from a mutation rule, pattern, and protocol context.
    pub fn from_rule(
        rule: &MutationRule,
        pattern: &DetectedPattern,
        existing_states: &[ProtocolState],
        existing_transitions: &[ProtocolTransition],
    ) -> Self {
        Self {
            pattern: pattern.clone(),
            proposed_state: rule.state_name.clone(),
            proposed_state_description: rule.state_description.clone(),
            proposed_transitions: vec![rule.trigger_in.clone(), rule.trigger_out.clone()],
            protocol_context: existing_states.iter().map(|s| s.name.clone()).collect(),
            protocol_transition_count: existing_transitions.len(),
        }
    }
}

/// Result of a critic evaluation.
///
/// # References
/// - EvoFSM (2026): "Controllable Self-Evolution for Deep Research with FSMs"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticScore {
    /// Overall score in [0.0, 1.0]. Higher = more favorable.
    pub score: f64,
    /// Breakdown of individual scoring components.
    pub components: CriticComponents,
    /// Human-readable rationale for the score.
    pub rationale: String,
}

/// Breakdown of the three scoring axes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticComponents {
    /// Pattern history score [0, 1]: based on frequency and confidence.
    pub history_score: f64,
    /// Structural impact score [0, 1]: lower is better (fewer disruptions).
    pub structural_score: f64,
    /// Coherence score [0, 1]: 1.0 if no duplicates, 0.0 if duplicate found.
    pub coherence_score: f64,
}

/// Configuration mode for the critic.
///
/// # References
/// - EvoFSM (2026): "Controllable Self-Evolution for Deep Research with FSMs"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriticMode {
    /// Score mutations and apply those above threshold.
    Apply,
    /// Score mutations but never apply — suggest only (for review).
    SuggestOnly,
}

impl Default for CriticMode {
    fn default() -> Self {
        Self::Apply
    }
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Trait for mutation critics that evaluate candidates before application.
///
/// Implementations score each [`MutationCandidate`] and return a [`CriticScore`].
/// The caller decides whether to apply based on the score and configured threshold.
///
/// # References
/// - EvoFSM (2026): "Controllable Self-Evolution for Deep Research with FSMs"
///   — Section 4.2: "Critic-gated mutation prevents destabilizing FSM changes"
#[async_trait::async_trait]
pub trait MutationCritic: Send + Sync {
    /// Score a mutation candidate.
    ///
    /// Returns a [`CriticScore`] with an overall score in [0.0, 1.0].
    /// Scores ≥ threshold should be applied; scores below should be rejected.
    async fn score_mutation(&self, candidate: &MutationCandidate) -> anyhow::Result<CriticScore>;
}

// ─── GraphBasedCritic ────────────────────────────────────────────────────────

/// Critic that uses graph-based heuristics to score mutations.
///
/// Scoring formula (equal weights):
/// - `history_score` = clamp(pattern.confidence × log2(frequency + 1) / 4, 0, 1)
/// - `structural_score` = 1.0 - clamp(transition_count / 20, 0, 1)
///   (fewer transitions = less disruption risk)
/// - `coherence_score` = 0.0 if proposed state already exists, else 1.0
///
/// `overall = (history × 0.4) + (structural × 0.3) + (coherence × 0.3)`
///
/// # References
/// - EvoFSM (2026): "Controllable Self-Evolution for Deep Research with FSMs"
///   — Section 4.3: "Graph-theoretic scoring for mutation candidates"
pub struct GraphBasedCritic {
    #[allow(dead_code)]
    graph: Arc<dyn GraphStore>,
}

impl GraphBasedCritic {
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self { graph }
    }

    /// Compute history score from pattern confidence and frequency.
    fn compute_history_score(pattern: &DetectedPattern) -> f64 {
        let freq_factor = ((pattern.frequency as f64) + 1.0).log2() / 4.0;
        (pattern.confidence * freq_factor).clamp(0.0, 1.0)
    }

    /// Compute structural impact score from existing transition count.
    /// Fewer transitions = higher score (less risk of disruption).
    fn compute_structural_score(transition_count: usize) -> f64 {
        (1.0 - (transition_count as f64 / 20.0)).clamp(0.0, 1.0)
    }

    /// Check coherence: proposed state must not already exist.
    fn compute_coherence_score(proposed_state: &str, existing_states: &[String]) -> f64 {
        if existing_states.iter().any(|s| s == proposed_state) {
            0.0
        } else {
            1.0
        }
    }
}

#[async_trait::async_trait]
impl MutationCritic for GraphBasedCritic {
    async fn score_mutation(&self, candidate: &MutationCandidate) -> anyhow::Result<CriticScore> {
        let history_score = Self::compute_history_score(&candidate.pattern);
        let structural_score = Self::compute_structural_score(candidate.protocol_transition_count);
        let coherence_score = Self::compute_coherence_score(
            &candidate.proposed_state,
            &candidate.protocol_context,
        );

        let overall = (history_score * 0.4) + (structural_score * 0.3) + (coherence_score * 0.3);

        let rationale = format!(
            "history={:.2} (conf={:.2}, freq={}), structural={:.2} ({} transitions), \
             coherence={:.2} (duplicate={})",
            history_score,
            candidate.pattern.confidence,
            candidate.pattern.frequency,
            structural_score,
            candidate.protocol_transition_count,
            coherence_score,
            coherence_score == 0.0,
        );

        debug!(
            proposed_state = %candidate.proposed_state,
            score = format!("{:.3}", overall),
            %rationale,
            "Critic scored mutation candidate"
        );

        Ok(CriticScore {
            score: overall,
            components: CriticComponents {
                history_score,
                structural_score,
                coherence_score,
            },
            rationale,
        })
    }
}

// ─── Learning Config extensions ─────────────────────────────────────────────

/// Configuration for the mutation critic within the learning loop.
///
/// # References
/// - EvoFSM (2026): "Controllable Self-Evolution for Deep Research with FSMs"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Minimum confidence for a pattern to be considered for mutation.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
    /// Maximum mutations per evolution cycle.
    #[serde(default = "default_max_mutations")]
    pub max_mutations_per_cycle: usize,
    /// Critic threshold: mutations with score < this are rejected.
    #[serde(default = "default_critic_threshold")]
    pub critic_threshold: f64,
    /// Critic operating mode.
    #[serde(default)]
    pub critic_mode: CriticMode,
}

fn default_min_confidence() -> f64 {
    0.8
}
fn default_max_mutations() -> usize {
    3
}
fn default_critic_threshold() -> f64 {
    0.7
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            min_confidence: default_min_confidence(),
            max_mutations_per_cycle: default_max_mutations(),
            critic_threshold: default_critic_threshold(),
            critic_mode: CriticMode::default(),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::pipeline::feedback::PatternType;
    use uuid::Uuid;

    fn make_candidate(
        confidence: f64,
        frequency: usize,
        transition_count: usize,
        proposed_state: &str,
        existing_states: Vec<&str>,
    ) -> MutationCandidate {
        MutationCandidate {
            pattern: DetectedPattern {
                id: format!("test-{}", Uuid::new_v4()),
                pattern_type: PatternType::FrequentGateFailure,
                description: "test pattern".to_string(),
                frequency,
                confidence,
                tech_stacks: vec!["rust".to_string()],
                related_gates: vec!["test".to_string()],
                recommendation: "test recommendation".to_string(),
                affected_files: vec![],
            },
            proposed_state: proposed_state.to_string(),
            proposed_state_description: "Auto-added test state".to_string(),
            proposed_transitions: vec!["trigger_in".to_string(), "trigger_out".to_string()],
            protocol_context: existing_states.into_iter().map(String::from).collect(),
            protocol_transition_count: transition_count,
        }
    }

    #[tokio::test]
    async fn high_confidence_favorable_history_scores_above_threshold() {
        let store = Arc::new(MockGraphStore::new());
        let critic = GraphBasedCritic::new(store);

        // High confidence (0.95), high frequency (10), few transitions (2), no duplicate
        let candidate = make_candidate(0.95, 10, 2, "run_tests", vec!["start", "implement", "done"]);
        let result = critic.score_mutation(&candidate).await.unwrap();

        assert!(
            result.score > 0.7,
            "Expected score > 0.7 for high-confidence pattern, got {:.3}",
            result.score
        );
        assert!(result.components.history_score > 0.5);
        assert!(result.components.coherence_score == 1.0);
    }

    #[tokio::test]
    async fn isolated_pattern_scores_below_threshold() {
        let store = Arc::new(MockGraphStore::new());
        let critic = GraphBasedCritic::new(store);

        // Low confidence (0.3), low frequency (1), many transitions (15), no duplicate
        let candidate = make_candidate(0.3, 1, 15, "run_tests", vec!["start", "implement", "done"]);
        let result = critic.score_mutation(&candidate).await.unwrap();

        assert!(
            result.score < 0.5,
            "Expected score < 0.5 for isolated pattern, got {:.3}",
            result.score
        );
    }

    #[tokio::test]
    async fn duplicate_state_scores_zero_coherence() {
        let store = Arc::new(MockGraphStore::new());
        let critic = GraphBasedCritic::new(store);

        // State "run_tests" already exists in context
        let candidate = make_candidate(
            0.95,
            10,
            2,
            "run_tests",
            vec!["start", "implement", "run_tests", "done"],
        );
        let result = critic.score_mutation(&candidate).await.unwrap();

        assert_eq!(result.components.coherence_score, 0.0);
        // Overall should be lower due to zero coherence
        assert!(
            result.score < 0.7,
            "Expected score < 0.7 when state already exists, got {:.3}",
            result.score
        );
    }

    #[test]
    fn history_score_scales_with_frequency() {
        let low = GraphBasedCritic::compute_history_score(&DetectedPattern {
            id: "a".to_string(),
            pattern_type: PatternType::FrequentGateFailure,
            description: String::new(),
            frequency: 1,
            confidence: 0.9,
            tech_stacks: vec![],
            related_gates: vec![],
            recommendation: String::new(),
            affected_files: vec![],
        });
        let high = GraphBasedCritic::compute_history_score(&DetectedPattern {
            id: "b".to_string(),
            pattern_type: PatternType::FrequentGateFailure,
            description: String::new(),
            frequency: 15,
            confidence: 0.9,
            tech_stacks: vec![],
            related_gates: vec![],
            recommendation: String::new(),
            affected_files: vec![],
        });
        assert!(high > low, "Higher frequency should yield higher score");
    }

    #[test]
    fn structural_score_penalizes_many_transitions() {
        let few = GraphBasedCritic::compute_structural_score(2);
        let many = GraphBasedCritic::compute_structural_score(18);
        assert!(few > many, "Fewer transitions should score higher");
        assert!(few > 0.8);
        assert!(many < 0.2);
    }

    #[test]
    fn learning_config_serialization_roundtrip() {
        let config = LearningConfig {
            min_confidence: 0.85,
            max_mutations_per_cycle: 5,
            critic_threshold: 0.75,
            critic_mode: CriticMode::SuggestOnly,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: LearningConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.min_confidence, 0.85);
        assert_eq!(deserialized.max_mutations_per_cycle, 5);
        assert_eq!(deserialized.critic_threshold, 0.75);
        assert_eq!(deserialized.critic_mode, CriticMode::SuggestOnly);
    }

    #[test]
    fn learning_config_default_values() {
        let config = LearningConfig::default();
        assert_eq!(config.min_confidence, 0.8);
        assert_eq!(config.max_mutations_per_cycle, 3);
        assert_eq!(config.critic_threshold, 0.7);
        assert_eq!(config.critic_mode, CriticMode::Apply);
    }

    #[test]
    fn learning_config_deserialize_with_defaults() {
        // Deserialize empty JSON — all fields should get defaults
        let config: LearningConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.critic_threshold, 0.7);
        assert_eq!(config.critic_mode, CriticMode::Apply);
    }

    #[test]
    fn critic_mode_serialization() {
        let apply_json = serde_json::to_string(&CriticMode::Apply).unwrap();
        assert_eq!(apply_json, "\"apply\"");

        let suggest_json = serde_json::to_string(&CriticMode::SuggestOnly).unwrap();
        assert_eq!(suggest_json, "\"suggest_only\"");

        let roundtrip: CriticMode = serde_json::from_str(&apply_json).unwrap();
        assert_eq!(roundtrip, CriticMode::Apply);
    }
}
