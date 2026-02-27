//! Skill Validation & Health Tracking
//!
//! Implements a Darwinian survival model for imported skills:
//! - Imported skills start with low energy and must prove their value
//! - Metrics track adoption: activation count, hit rate, energy trend
//! - Accelerated degradation for unused imported skills (7-day probation)
//! - Validated skills are promoted and marked as proven

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::skills::models::{SkillNode, SkillStatus};

// ============================================================================
// Configuration
// ============================================================================

/// Number of days before an unused imported skill is degraded.
pub const IMPORT_PROBATION_DAYS: i64 = 7;

/// Minimum activation count to consider an imported skill validated.
pub const IMPORT_VALIDATION_ACTIVATIONS: i64 = 2;

/// Energy threshold below which a skill is considered at risk.
const AT_RISK_ENERGY: f64 = 0.1;

/// Energy threshold below which a skill needs attention.
const NEEDS_ATTENTION_ENERGY: f64 = 0.3;

// ============================================================================
// Health metrics
// ============================================================================

/// Health metrics and survival assessment for a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillHealthMetrics {
    /// Unique skill ID.
    pub skill_id: uuid::Uuid,
    /// Current skill name.
    pub skill_name: String,
    /// Current lifecycle status.
    pub status: String,
    /// Total activation count.
    pub activation_count: i64,
    /// Hit rate: successful activations / total (0.0-1.0).
    pub hit_rate: f64,
    /// Current energy level (0.0-1.0).
    pub energy: f64,
    /// Louvain cohesion score (0.0-1.0).
    pub cohesion: f64,
    /// Number of member notes.
    pub note_count: i64,
    /// Number of member decisions.
    pub decision_count: i64,
    /// Days since the skill was imported (None if natively detected).
    pub days_since_import: Option<i64>,
    /// Whether this imported skill has been validated through usage.
    pub is_validated: bool,
    /// Whether the skill is in its probation period.
    pub in_probation: bool,
    /// Days until probation expires (None if not in probation).
    pub probation_days_remaining: Option<i64>,
    /// Health recommendation.
    pub recommendation: HealthRecommendation,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Health recommendation for a skill.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HealthRecommendation {
    /// Skill is healthy and actively used.
    Healthy,
    /// Skill needs more usage or attention.
    NeedsAttention,
    /// Skill is at risk of degradation.
    AtRisk,
    /// Skill should be archived (unused or degraded).
    ShouldArchive,
}

impl std::fmt::Display for HealthRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::NeedsAttention => write!(f, "needs_attention"),
            Self::AtRisk => write!(f, "at_risk"),
            Self::ShouldArchive => write!(f, "should_archive"),
        }
    }
}

// ============================================================================
// Computation
// ============================================================================

/// Compute health metrics for a skill.
///
/// This is a pure function that takes the skill node and current time,
/// returning a complete health assessment. No database calls needed.
pub fn compute_health(skill: &SkillNode, now: DateTime<Utc>) -> SkillHealthMetrics {
    let days_since_import = skill.imported_at.map(|dt| (now - dt).num_days());
    let is_imported = skill.imported_at.is_some();

    // Probation: imported skills within IMPORT_PROBATION_DAYS
    let in_probation = is_imported
        && days_since_import.is_some_and(|d| d <= IMPORT_PROBATION_DAYS)
        && !skill.is_validated;

    let probation_days_remaining = if in_probation {
        days_since_import.map(|d| (IMPORT_PROBATION_DAYS - d).max(0))
    } else {
        None
    };

    // Compute recommendation
    let (recommendation, explanation) =
        assess_health(skill, is_imported, days_since_import, in_probation);

    SkillHealthMetrics {
        skill_id: skill.id,
        skill_name: skill.name.clone(),
        status: skill.status.to_string(),
        activation_count: skill.activation_count,
        hit_rate: skill.hit_rate,
        energy: skill.energy,
        cohesion: skill.cohesion,
        note_count: skill.note_count,
        decision_count: skill.decision_count,
        days_since_import,
        is_validated: skill.is_validated,
        in_probation,
        probation_days_remaining,
        recommendation,
        explanation,
    }
}

/// Assess the health recommendation for a skill.
fn assess_health(
    skill: &SkillNode,
    is_imported: bool,
    days_since_import: Option<i64>,
    in_probation: bool,
) -> (HealthRecommendation, String) {
    // Already archived or dormant
    if skill.status == SkillStatus::Archived {
        return (
            HealthRecommendation::ShouldArchive,
            "Skill is archived.".to_string(),
        );
    }

    if skill.status == SkillStatus::Dormant {
        return (
            HealthRecommendation::ShouldArchive,
            "Skill is dormant with low activity. Consider archiving.".to_string(),
        );
    }

    // Imported skill: accelerated degradation check
    if is_imported && !skill.is_validated {
        if let Some(days) = days_since_import {
            if days > IMPORT_PROBATION_DAYS && skill.activation_count == 0 {
                return (
                    HealthRecommendation::ShouldArchive,
                    format!(
                        "Imported skill unused after {} days (probation: {} days). Should be archived.",
                        days, IMPORT_PROBATION_DAYS
                    ),
                );
            }
            if days > IMPORT_PROBATION_DAYS
                && skill.activation_count < IMPORT_VALIDATION_ACTIVATIONS
            {
                return (
                    HealthRecommendation::AtRisk,
                    format!(
                        "Imported skill has only {} activation(s) after {} days. Needs {} for validation.",
                        skill.activation_count, days, IMPORT_VALIDATION_ACTIVATIONS
                    ),
                );
            }
        }
    }

    // In probation: check progress
    if in_probation {
        if skill.activation_count > 0 {
            return (
                HealthRecommendation::NeedsAttention,
                format!(
                    "Imported skill in probation ({} days remaining). {} activation(s) so far — needs {} for validation.",
                    days_since_import.map(|d| (IMPORT_PROBATION_DAYS - d).max(0)).unwrap_or(0),
                    skill.activation_count,
                    IMPORT_VALIDATION_ACTIVATIONS,
                ),
            );
        } else {
            return (
                HealthRecommendation::NeedsAttention,
                format!(
                    "Imported skill in probation ({} days remaining). No activations yet.",
                    days_since_import
                        .map(|d| (IMPORT_PROBATION_DAYS - d).max(0))
                        .unwrap_or(0),
                ),
            );
        }
    }

    // General energy assessment
    if skill.energy < AT_RISK_ENERGY {
        return (
            HealthRecommendation::AtRisk,
            format!(
                "Very low energy ({:.2}). Risk of demotion without activation.",
                skill.energy
            ),
        );
    }

    if skill.energy < NEEDS_ATTENTION_ENERGY {
        return (
            HealthRecommendation::NeedsAttention,
            format!(
                "Low energy ({:.2}). Consider activating to prevent decay.",
                skill.energy
            ),
        );
    }

    // Healthy
    (
        HealthRecommendation::Healthy,
        format!(
            "Skill is healthy. Energy: {:.2}, {} activations, hit rate: {:.0}%.",
            skill.energy,
            skill.activation_count,
            skill.hit_rate * 100.0
        ),
    )
}

/// Evaluate whether an imported skill should undergo accelerated degradation.
///
/// Returns `true` if the skill should be immediately archived (skip Dormant).
/// This is called during lifecycle updates for Imported-status skills.
pub fn should_accelerate_archive(skill: &SkillNode, now: DateTime<Utc>) -> bool {
    if skill.status != SkillStatus::Imported {
        return false;
    }

    let days_since_import = skill
        .imported_at
        .map(|dt| (now - dt).num_days())
        .unwrap_or(0);

    // Past probation with zero activations → archive immediately
    days_since_import > IMPORT_PROBATION_DAYS && skill.activation_count == 0
}

/// Check if an imported skill qualifies for validation (promotion to Active).
///
/// Imported skills are validated when they reach the minimum activation
/// threshold, proving they're useful in the target project.
pub fn qualifies_for_validation(skill: &SkillNode) -> bool {
    skill.imported_at.is_some()
        && !skill.is_validated
        && skill.activation_count >= IMPORT_VALIDATION_ACTIVATIONS
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use uuid::Uuid;

    fn make_imported_skill(days_ago: i64, activations: i64) -> SkillNode {
        let now = Utc::now();
        let mut skill = SkillNode::new(Uuid::new_v4(), "Test Import Skill");
        skill.status = SkillStatus::Imported;
        skill.imported_at = Some(now - Duration::days(days_ago));
        skill.activation_count = activations;
        skill.energy = 0.5;
        skill.note_count = 3;
        skill
    }

    fn make_active_skill(energy: f64, activations: i64) -> SkillNode {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Active Skill");
        skill.status = SkillStatus::Active;
        skill.energy = energy;
        skill.activation_count = activations;
        skill.hit_rate = 0.8;
        skill.note_count = 5;
        skill
    }

    #[test]
    fn test_healthy_active_skill() {
        let skill = make_active_skill(0.7, 10);
        let health = compute_health(&skill, Utc::now());

        assert_eq!(health.recommendation, HealthRecommendation::Healthy);
        assert!(health.explanation.contains("healthy"));
        assert!(!health.is_validated);
        assert!(!health.in_probation);
        assert!(health.days_since_import.is_none());
    }

    #[test]
    fn test_imported_skill_in_probation() {
        let skill = make_imported_skill(3, 0);
        let health = compute_health(&skill, Utc::now());

        assert!(health.in_probation);
        assert_eq!(health.days_since_import, Some(3));
        assert_eq!(health.probation_days_remaining, Some(4)); // 7 - 3
        assert_eq!(health.recommendation, HealthRecommendation::NeedsAttention);
        assert!(health.explanation.contains("probation"));
    }

    #[test]
    fn test_imported_skill_in_probation_with_activations() {
        let skill = make_imported_skill(3, 1);
        let health = compute_health(&skill, Utc::now());

        assert!(health.in_probation);
        assert_eq!(health.recommendation, HealthRecommendation::NeedsAttention);
        assert!(health.explanation.contains("1 activation"));
    }

    #[test]
    fn test_imported_skill_past_probation_no_activations() {
        let skill = make_imported_skill(10, 0);
        let health = compute_health(&skill, Utc::now());

        assert!(!health.in_probation);
        assert_eq!(health.recommendation, HealthRecommendation::ShouldArchive);
        assert!(health.explanation.contains("unused after"));
    }

    #[test]
    fn test_imported_skill_past_probation_insufficient_activations() {
        let skill = make_imported_skill(10, 1);
        let health = compute_health(&skill, Utc::now());

        assert_eq!(health.recommendation, HealthRecommendation::AtRisk);
        assert!(health.explanation.contains("1 activation"));
    }

    #[test]
    fn test_imported_skill_validated() {
        let mut skill = make_imported_skill(10, 5);
        skill.is_validated = true;
        skill.energy = 0.6;
        let health = compute_health(&skill, Utc::now());

        // Validated imported skills are treated as normal
        assert!(health.is_validated);
        assert!(!health.in_probation);
        assert_eq!(health.recommendation, HealthRecommendation::Healthy);
    }

    #[test]
    fn test_low_energy_active_skill() {
        let skill = make_active_skill(0.05, 2);
        let health = compute_health(&skill, Utc::now());

        assert_eq!(health.recommendation, HealthRecommendation::AtRisk);
        assert!(health.explanation.contains("Very low energy"));
    }

    #[test]
    fn test_medium_energy_needs_attention() {
        let skill = make_active_skill(0.2, 3);
        let health = compute_health(&skill, Utc::now());

        assert_eq!(health.recommendation, HealthRecommendation::NeedsAttention);
        assert!(health.explanation.contains("Low energy"));
    }

    #[test]
    fn test_dormant_should_archive() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Dormant Skill");
        skill.status = SkillStatus::Dormant;
        let health = compute_health(&skill, Utc::now());

        assert_eq!(health.recommendation, HealthRecommendation::ShouldArchive);
    }

    #[test]
    fn test_archived_should_archive() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Archived Skill");
        skill.status = SkillStatus::Archived;
        let health = compute_health(&skill, Utc::now());

        assert_eq!(health.recommendation, HealthRecommendation::ShouldArchive);
    }

    #[test]
    fn test_should_accelerate_archive() {
        // Not imported → false
        let native = SkillNode::new(Uuid::new_v4(), "Native");
        assert!(!should_accelerate_archive(&native, Utc::now()));

        // Imported, within probation → false
        let recent = make_imported_skill(3, 0);
        assert!(!should_accelerate_archive(&recent, Utc::now()));

        // Imported, past probation, no activations → true
        let old_unused = make_imported_skill(10, 0);
        assert!(should_accelerate_archive(&old_unused, Utc::now()));

        // Imported, past probation, with activations → false
        let old_used = make_imported_skill(10, 1);
        assert!(!should_accelerate_archive(&old_used, Utc::now()));
    }

    #[test]
    fn test_qualifies_for_validation() {
        // Not imported → false
        let native = SkillNode::new(Uuid::new_v4(), "Native");
        assert!(!qualifies_for_validation(&native));

        // Imported, not enough activations → false
        let low = make_imported_skill(5, 1);
        assert!(!qualifies_for_validation(&low));

        // Imported, enough activations → true
        let ready = make_imported_skill(5, 2);
        assert!(qualifies_for_validation(&ready));

        // Already validated → false
        let mut validated = make_imported_skill(5, 5);
        validated.is_validated = true;
        assert!(!qualifies_for_validation(&validated));
    }

    #[test]
    fn test_health_recommendation_display() {
        assert_eq!(HealthRecommendation::Healthy.to_string(), "healthy");
        assert_eq!(
            HealthRecommendation::NeedsAttention.to_string(),
            "needs_attention"
        );
        assert_eq!(HealthRecommendation::AtRisk.to_string(), "at_risk");
        assert_eq!(
            HealthRecommendation::ShouldArchive.to_string(),
            "should_archive"
        );
    }
}
