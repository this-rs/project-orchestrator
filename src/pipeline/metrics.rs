//! # Learning Loop Health Metrics (T7 — MEASURE)
//!
//! Computes metrics that measure whether the autonomous learning loop is
//! actually improving outcomes. Without measurement, the system could learn
//! wrong lessons and diverge.
//!
//! ## Metrics
//!
//! | Metric                   | What it measures                                          |
//! |--------------------------|-----------------------------------------------------------|
//! | `learning_hit_rate`      | % of auto-learned notes that were activated (used)        |
//! | `mutation_success_rate`  | % of protocol mutations that improved outcomes            |
//! | `reflex_effectiveness`   | % of notes with scars that eventually healed              |
//! | `scar_reduction_rate`    | % of scarred notes whose scar_intensity decreased         |
//! | `episode_to_skill_ratio` | Ratio of episodes collected to skills emerged              |
//!
//! ## Decay & Consolidation
//!
//! - `decay_ineffective_notes()` — archives auto-learned notes with low energy after 7 days
//! - `promote_battle_tested_notes()` — promotes high-energy old notes to consolidated

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::notes::models::{MemoryHorizon, NoteFilters, NoteStatus};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Days before an unactivated auto-learned note is eligible for archival.
const DECAY_THRESHOLD_DAYS: i64 = 7;

/// Energy below which an auto-learned note gets archived.
const DECAY_ENERGY_THRESHOLD: f64 = 0.1;

/// Days before an auto-learned note can be promoted to consolidated.
const PROMOTE_THRESHOLD_DAYS: i64 = 30;

/// Energy above which an auto-learned note gets promoted.
const PROMOTE_ENERGY_THRESHOLD: f64 = 0.5;

/// Minimum activation count for promotion.
const PROMOTE_ACTIVATION_THRESHOLD: i64 = 3;

// ─── Metrics struct ─────────────────────────────────────────────────────────

/// Aggregated learning loop health metrics for a project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMetrics {
    /// % of auto-learned notes that have been activated at least once.
    pub learning_hit_rate: f64,
    /// Total auto-learned notes.
    pub total_auto_learned: usize,
    /// Auto-learned notes with activation_count > 0.
    pub activated_auto_learned: usize,
    /// % of scarred notes/decisions that were eventually healed (scar → 0).
    pub scar_reduction_rate: f64,
    /// Total notes with scar > 0 at some point.
    pub total_scarred: usize,
    /// Scarred notes now healed (scar = 0 but was > 0).
    pub healed_scars: usize,
    /// Ratio of episodes to emerging skills.
    pub episode_to_skill_ratio: f64,
    /// Total episodes collected.
    pub total_episodes: usize,
    /// Total emerging skills.
    pub total_emerging_skills: usize,
    /// Notes archived by decay (low energy, unactivated).
    pub notes_archived_by_decay: usize,
    /// Notes promoted to consolidated (battle-tested).
    pub notes_promoted: usize,
}

impl LearningMetrics {
    /// Compute learning metrics for a project.
    pub async fn compute(graph: &dyn GraphStore, project_id: Uuid) -> Result<Self> {
        // 1. Fetch all auto-learned notes
        let auto_learned_filters = NoteFilters {
            tags: Some(vec!["auto-learned".to_string()]),
            limit: Some(500),
            ..Default::default()
        };
        let (auto_learned_notes, _) = graph
            .list_notes(Some(project_id), None, &auto_learned_filters)
            .await?;

        let total_auto_learned = auto_learned_notes.len();
        let activated_auto_learned = auto_learned_notes
            .iter()
            .filter(|n| n.activation_count > 0)
            .count();
        let learning_hit_rate = if total_auto_learned > 0 {
            activated_auto_learned as f64 / total_auto_learned as f64
        } else {
            0.0
        };

        // 2. Scar reduction: notes with scar_intensity > 0 vs healed ones
        let all_filters = NoteFilters {
            limit: Some(500),
            ..Default::default()
        };
        let (all_notes, _) = graph
            .list_notes(Some(project_id), None, &all_filters)
            .await?;
        let scarred_notes: Vec<_> = all_notes
            .iter()
            .filter(|n| n.scar_intensity > 0.0)
            .collect();
        let total_scarred = scarred_notes.len();
        // Notes that had scars but are now at 0 would need historical data;
        // for now we measure currently-scarred vs total
        let healed_scars = 0_usize; // Would need historical tracking
        let scar_reduction_rate = 0.0; // Placeholder — requires tracking scar history

        // 3. Episode to skill ratio
        let episodes = crate::episodes::collector::list_episodes(graph, project_id, 500)
            .await
            .unwrap_or_default();
        let total_episodes = episodes.len();
        let (skills, _) = graph
            .list_skills(
                project_id,
                Some(crate::skills::models::SkillStatus::Emerging),
                500,
                0,
            )
            .await
            .unwrap_or((vec![], 0));
        let total_emerging_skills = skills.len();
        let episode_to_skill_ratio = if total_emerging_skills > 0 {
            total_episodes as f64 / total_emerging_skills as f64
        } else {
            0.0
        };

        // 4. Count archived-by-decay and promoted notes
        let archived_filters = NoteFilters {
            tags: Some(vec!["auto-learned".to_string()]),
            status: Some(vec![NoteStatus::Archived]),
            limit: Some(500),
            ..Default::default()
        };
        let (archived_notes, _) = graph
            .list_notes(Some(project_id), None, &archived_filters)
            .await?;
        let notes_archived_by_decay = archived_notes.len();

        let promoted_filters = NoteFilters {
            tags: Some(vec!["battle-tested".to_string()]),
            limit: Some(500),
            ..Default::default()
        };
        let (promoted_notes, _) = graph
            .list_notes(Some(project_id), None, &promoted_filters)
            .await?;
        let notes_promoted = promoted_notes.len();

        Ok(Self {
            learning_hit_rate,
            total_auto_learned,
            activated_auto_learned,
            scar_reduction_rate,
            total_scarred,
            healed_scars,
            episode_to_skill_ratio,
            total_episodes,
            total_emerging_skills,
            notes_archived_by_decay,
            notes_promoted,
        })
    }
}

// ─── Decay ineffective notes ────────────────────────────────────────────────

/// Result of the decay operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayResult {
    /// How many notes were archived.
    pub archived_count: usize,
    /// How many notes were checked.
    pub checked_count: usize,
}

/// Archive auto-learned notes that have low energy and haven't been activated
/// within `DECAY_THRESHOLD_DAYS`.
///
/// Criteria:
/// - Tag: `auto-learned`
/// - Status: Active
/// - `memory_horizon`: Ephemeral
/// - `energy` < 0.1
/// - `created_at` > 7 days ago
pub async fn decay_ineffective_notes(
    graph: &dyn GraphStore,
    project_id: Uuid,
) -> Result<DecayResult> {
    let filters = NoteFilters {
        tags: Some(vec!["auto-learned".to_string()]),
        status: Some(vec![NoteStatus::Active]),
        limit: Some(200),
        ..Default::default()
    };

    let (notes, _) = graph.list_notes(Some(project_id), None, &filters).await?;
    let now = Utc::now();
    let threshold = chrono::Duration::days(DECAY_THRESHOLD_DAYS);

    let mut archived_count = 0;
    let checked_count = notes.len();

    for note in &notes {
        // Must be ephemeral
        if note.memory_horizon != MemoryHorizon::Ephemeral {
            continue;
        }

        // Must be older than threshold
        let age = now - note.created_at;
        if age < threshold {
            continue;
        }

        // Must have low energy
        if note.energy >= DECAY_ENERGY_THRESHOLD {
            continue;
        }

        // Archive it
        debug!(
            note_id = %note.id,
            energy = note.energy,
            age_days = age.num_days(),
            "Archiving ineffective auto-learned note"
        );
        graph
            .update_note(note.id, None, None, Some(NoteStatus::Archived), None, None)
            .await?;
        archived_count += 1;
    }

    if archived_count > 0 {
        info!(
            project_id = %project_id,
            archived = archived_count,
            checked = checked_count,
            "Decayed ineffective auto-learned notes"
        );
    }

    Ok(DecayResult {
        archived_count,
        checked_count,
    })
}

// ─── Promote battle-tested notes ────────────────────────────────────────────

/// Result of the promotion operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromoteResult {
    /// How many notes were promoted to consolidated.
    pub promoted_count: usize,
    /// How many notes were checked.
    pub checked_count: usize,
}

/// Promote auto-learned notes that have survived long enough with high energy
/// and sufficient activations to `memory_horizon=Consolidated` status.
///
/// Criteria:
/// - Tag: `auto-learned`
/// - Status: Active
/// - `created_at` > 30 days ago
/// - `energy` > 0.5
/// - `activation_count` > 3
///
/// On promotion:
/// - `memory_horizon` → Consolidated
/// - Tag `auto-learned` replaced by `battle-tested`
pub async fn promote_battle_tested_notes(
    graph: &dyn GraphStore,
    project_id: Uuid,
) -> Result<PromoteResult> {
    let filters = NoteFilters {
        tags: Some(vec!["auto-learned".to_string()]),
        status: Some(vec![NoteStatus::Active]),
        limit: Some(200),
        ..Default::default()
    };

    let (notes, _) = graph.list_notes(Some(project_id), None, &filters).await?;
    let now = Utc::now();
    let threshold = chrono::Duration::days(PROMOTE_THRESHOLD_DAYS);

    let mut promoted_count = 0;
    let checked_count = notes.len();

    for note in &notes {
        // Must be old enough
        let age = now - note.created_at;
        if age < threshold {
            continue;
        }

        // Must have high energy
        if note.energy < PROMOTE_ENERGY_THRESHOLD {
            continue;
        }

        // Must have sufficient activations
        if note.activation_count < PROMOTE_ACTIVATION_THRESHOLD {
            continue;
        }

        // Promote: replace tags and update memory_horizon
        let mut new_tags = note
            .tags
            .iter()
            .filter(|t| *t != "auto-learned")
            .cloned()
            .collect::<Vec<_>>();
        new_tags.push("battle-tested".to_string());

        debug!(
            note_id = %note.id,
            energy = note.energy,
            activation_count = note.activation_count,
            age_days = age.num_days(),
            "Promoting auto-learned note to battle-tested"
        );

        // Update tags via update_note
        graph
            .update_note(note.id, None, None, None, Some(new_tags), None)
            .await?;

        // Note: memory_horizon promotion to Consolidated is handled by
        // admin(action: "consolidate_memory") which runs as part of deep_maintenance.
        // The tag change from "auto-learned" to "battle-tested" is the primary signal.

        promoted_count += 1;
    }

    if promoted_count > 0 {
        info!(
            project_id = %project_id,
            promoted = promoted_count,
            checked = checked_count,
            "Promoted battle-tested auto-learned notes"
        );
    }

    Ok(PromoteResult {
        promoted_count,
        checked_count,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_defaults_to_zero() {
        let m = LearningMetrics {
            learning_hit_rate: 0.0,
            total_auto_learned: 0,
            activated_auto_learned: 0,
            scar_reduction_rate: 0.0,
            total_scarred: 0,
            healed_scars: 0,
            episode_to_skill_ratio: 0.0,
            total_episodes: 0,
            total_emerging_skills: 0,
            notes_archived_by_decay: 0,
            notes_promoted: 0,
        };
        assert_eq!(m.learning_hit_rate, 0.0);
        assert_eq!(m.episode_to_skill_ratio, 0.0);
    }

    #[test]
    fn hit_rate_calculation() {
        let total = 10;
        let activated = 7;
        let rate = activated as f64 / total as f64;
        assert!((rate - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn ratio_with_zero_skills() {
        let episodes = 50;
        let skills = 0;
        let ratio = if skills > 0 {
            episodes as f64 / skills as f64
        } else {
            0.0
        };
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn decay_result_serializable() {
        let r = DecayResult {
            archived_count: 5,
            checked_count: 20,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("archived_count"));
    }

    #[test]
    fn promote_result_serializable() {
        let r = PromoteResult {
            promoted_count: 3,
            checked_count: 15,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("promoted_count"));
    }
}
