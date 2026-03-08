//! Trust Scoring for Skills
//!
//! Calculates a composite trust score for skills based on multiple signals.
//! Trust scores range from 0.0 to 1.0 and are used to:
//! - Filter search results in the skill registry
//! - Rank skills for import recommendations
//! - Decide whether an imported skill should be auto-activated
//!
//! # Score Composition
//!
//! | Signal                  | Weight | Source                        |
//! |-------------------------|--------|-------------------------------|
//! | Energy                  | 0.20   | SkillNode.energy              |
//! | Cohesion                | 0.20   | SkillNode.cohesion            |
//! | Activation (normalized) | 0.20   | SkillNode.activation_count    |
//! | Success rate            | 0.30   | Protocol runs or hit_rate     |
//! | Source projects count   | 0.10   | ExecutionHistory or 1         |

use serde::{Deserialize, Serialize};

use crate::skills::models::SkillNode;
use crate::skills::package::ExecutionHistory;

// ============================================================================
// Constants
// ============================================================================

/// Weight for the energy component.
const W_ENERGY: f64 = 0.20;
/// Weight for the cohesion component.
const W_COHESION: f64 = 0.20;
/// Weight for the normalized activation count.
const W_ACTIVATION: f64 = 0.20;
/// Weight for the success rate (protocol runs).
const W_SUCCESS_RATE: f64 = 0.30;
/// Weight for the source projects count factor.
const W_SOURCE_PROJECTS: f64 = 0.10;

/// Activation count at which the normalized value reaches ~0.9.
/// Uses sigmoid-like normalization: count / (count + K).
/// With K=20, 20 activations → 0.5, 100 → 0.83, 200 → 0.91.
const ACTIVATION_HALF_LIFE: f64 = 20.0;

/// Number of source projects at which the factor reaches ~0.9.
/// Uses the same normalization: count / (count + K).
/// With K=3, 3 projects → 0.5, 10 → 0.77, 30 → 0.91.
const SOURCE_PROJECTS_HALF_LIFE: f64 = 3.0;

// ============================================================================
// Types
// ============================================================================

/// Detailed breakdown of a trust score computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScore {
    /// Final composite score (0.0–1.0).
    pub score: f64,
    /// Individual component scores (before weighting).
    pub components: TrustComponents,
    /// Human-readable trust level.
    pub level: TrustLevel,
}

/// Individual signal values used in trust computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustComponents {
    /// Energy signal (0.0–1.0), from SkillNode.energy.
    pub energy: f64,
    /// Cohesion signal (0.0–1.0), from SkillNode.cohesion.
    pub cohesion: f64,
    /// Normalized activation count (0.0–1.0).
    pub activation: f64,
    /// Success rate (0.0–1.0), from protocol runs or hit_rate.
    pub success_rate: f64,
    /// Source projects factor (0.0–1.0).
    pub source_projects: f64,
}

/// Human-readable trust level categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    /// Score ≥ 0.8 — highly trusted, auto-activate candidate.
    High,
    /// Score ≥ 0.5 — moderately trusted, manual review recommended.
    Medium,
    /// Score ≥ 0.3 — low trust, use with caution.
    Low,
    /// Score < 0.3 — untrusted, newly imported or problematic.
    Untrusted,
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::High => write!(f, "high"),
            Self::Medium => write!(f, "medium"),
            Self::Low => write!(f, "low"),
            Self::Untrusted => write!(f, "untrusted"),
        }
    }
}

// ============================================================================
// Computation
// ============================================================================

/// Compute the trust score for a skill.
///
/// # Arguments
///
/// * `skill` — The skill node with current metrics
/// * `execution_history` — Optional execution history (from package import or
///   computed from protocol runs). If `None`, success_rate and source_projects
///   are derived from the skill's own metrics.
///
/// # Returns
///
/// A [`TrustScore`] with the composite score, component breakdown, and level.
pub fn compute_trust_score(
    skill: &SkillNode,
    execution_history: Option<&ExecutionHistory>,
) -> TrustScore {
    // Energy: directly from skill (already 0.0–1.0)
    let energy = skill.energy.clamp(0.0, 1.0);

    // Cohesion: directly from skill (already 0.0–1.0)
    let cohesion = skill.cohesion.clamp(0.0, 1.0);

    // Activation: sigmoid normalization
    let activation = normalize_count(skill.activation_count as f64, ACTIVATION_HALF_LIFE);

    // Success rate: from execution history or skill's hit_rate
    let success_rate = execution_history
        .map(|h| h.success_rate.clamp(0.0, 1.0))
        .unwrap_or_else(|| skill.hit_rate.clamp(0.0, 1.0));

    // Source projects: from execution history or default to 1
    let source_count = execution_history
        .map(|h| h.source_projects_count as f64)
        .unwrap_or(1.0);
    let source_projects = normalize_count(source_count, SOURCE_PROJECTS_HALF_LIFE);

    // Weighted sum
    let score = (W_ENERGY * energy)
        + (W_COHESION * cohesion)
        + (W_ACTIVATION * activation)
        + (W_SUCCESS_RATE * success_rate)
        + (W_SOURCE_PROJECTS * source_projects);

    // Clamp to [0, 1] (should already be, but defensive)
    let score = score.clamp(0.0, 1.0);

    let level = match score {
        s if s >= 0.8 => TrustLevel::High,
        s if s >= 0.5 => TrustLevel::Medium,
        s if s >= 0.3 => TrustLevel::Low,
        _ => TrustLevel::Untrusted,
    };

    TrustScore {
        score,
        components: TrustComponents {
            energy,
            cohesion,
            activation,
            success_rate,
            source_projects,
        },
        level,
    }
}

/// Sigmoid-like normalization: value / (value + half_life).
///
/// Maps [0, ∞) → [0, 1) with the half-life value mapping to 0.5.
/// Negative values are clamped to 0.
fn normalize_count(value: f64, half_life: f64) -> f64 {
    if value <= 0.0 {
        return 0.0;
    }
    value / (value + half_life)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_skill(energy: f64, cohesion: f64, activation_count: i64, hit_rate: f64) -> SkillNode {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Test Skill");
        skill.energy = energy;
        skill.cohesion = cohesion;
        skill.activation_count = activation_count;
        skill.hit_rate = hit_rate;
        skill
    }

    #[test]
    fn test_high_trust_active_skill() {
        let skill = make_skill(0.9, 0.85, 100, 0.9);
        let trust = compute_trust_score(&skill, None);

        assert!(
            trust.score > 0.7,
            "Expected high trust, got {}",
            trust.score
        );
        assert_eq!(trust.level, TrustLevel::High);
    }

    #[test]
    fn test_low_trust_dormant_skill() {
        let skill = make_skill(0.0, 0.1, 0, 0.0);
        let trust = compute_trust_score(&skill, None);

        assert!(trust.score < 0.3, "Expected low trust, got {}", trust.score);
        assert_eq!(trust.level, TrustLevel::Untrusted);
    }

    #[test]
    fn test_medium_trust_imported_skill() {
        let skill = make_skill(0.0, 0.75, 5, 0.0);
        let history = ExecutionHistory {
            activation_count: 50,
            success_rate: 0.8,
            avg_score: 0.7,
            source_projects_count: 5,
        };
        let trust = compute_trust_score(&skill, Some(&history));

        // energy=0, cohesion=0.75, activation=5/25≈0.2, success=0.8, source=5/8≈0.625
        // score = 0*0.2 + 0.75*0.2 + 0.2*0.2 + 0.8*0.3 + 0.625*0.1 ≈ 0 + 0.15 + 0.04 + 0.24 + 0.0625 ≈ 0.4925
        assert!(
            trust.score >= 0.3,
            "Expected at least low, got {}",
            trust.score
        );
        assert!(
            trust.score < 0.8,
            "Should not be high trust, got {}",
            trust.score
        );
    }

    #[test]
    fn test_trust_with_execution_history() {
        let skill = make_skill(0.8, 0.9, 200, 0.5);
        let history = ExecutionHistory {
            activation_count: 200,
            success_rate: 0.95,
            avg_score: 0.9,
            source_projects_count: 10,
        };
        let trust = compute_trust_score(&skill, Some(&history));

        // Should be high: all signals strong
        assert!(
            trust.score >= 0.8,
            "Expected high trust, got {}",
            trust.score
        );
        assert_eq!(trust.level, TrustLevel::High);

        // Verify execution_history overrides hit_rate
        assert!((trust.components.success_rate - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_count() {
        assert!((normalize_count(0.0, 20.0) - 0.0).abs() < f64::EPSILON);
        assert!((normalize_count(20.0, 20.0) - 0.5).abs() < f64::EPSILON);
        assert!((normalize_count(100.0, 20.0) - 100.0 / 120.0).abs() < f64::EPSILON);
        assert!((normalize_count(-5.0, 20.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trust_score_bounds() {
        // Maximum possible score
        let max_skill = make_skill(1.0, 1.0, 10000, 1.0);
        let max_history = ExecutionHistory {
            activation_count: 10000,
            success_rate: 1.0,
            avg_score: 1.0,
            source_projects_count: 100,
        };
        let max_trust = compute_trust_score(&max_skill, Some(&max_history));
        assert!(max_trust.score <= 1.0);
        assert!(max_trust.score > 0.95); // should be very close to 1.0

        // Minimum possible score
        let min_skill = make_skill(0.0, 0.0, 0, 0.0);
        let min_trust = compute_trust_score(&min_skill, None);
        assert!(min_trust.score >= 0.0);
        assert!(min_trust.score < 0.05); // should be very close to 0.0
    }

    #[test]
    fn test_weights_sum_to_one() {
        let total = W_ENERGY + W_COHESION + W_ACTIVATION + W_SUCCESS_RATE + W_SOURCE_PROJECTS;
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Weights must sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_trust_level_boundaries() {
        // Test exact boundary values
        let skill_08 = make_skill(1.0, 1.0, 10000, 1.0);
        let trust_08 = compute_trust_score(&skill_08, None);
        // With all 1.0 signals (activation ≈ 1.0), score should be very high
        assert_eq!(trust_08.level, TrustLevel::High);

        let skill_0 = make_skill(0.0, 0.0, 0, 0.0);
        let trust_0 = compute_trust_score(&skill_0, None);
        assert_eq!(trust_0.level, TrustLevel::Untrusted);
    }

    #[test]
    fn test_trust_level_display() {
        assert_eq!(TrustLevel::High.to_string(), "high");
        assert_eq!(TrustLevel::Medium.to_string(), "medium");
        assert_eq!(TrustLevel::Low.to_string(), "low");
        assert_eq!(TrustLevel::Untrusted.to_string(), "untrusted");
    }

    #[test]
    fn test_success_rate_has_highest_weight() {
        // Two skills identical except success_rate
        let skill_high_sr = make_skill(0.5, 0.5, 50, 0.9);
        let skill_low_sr = make_skill(0.5, 0.5, 50, 0.1);

        let trust_high = compute_trust_score(&skill_high_sr, None);
        let trust_low = compute_trust_score(&skill_low_sr, None);

        let diff = trust_high.score - trust_low.score;
        // Difference should be approximately 0.8 * 0.3 = 0.24
        assert!(
            diff > 0.2,
            "Success rate should have strong impact, diff={}",
            diff
        );
    }
}
