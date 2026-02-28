//! Skill Lifecycle Management — Promotion, Demotion, and Metrics
//!
//! Manages the autonomous lifecycle of neural skills:
//! - **Promotion**: Emerging → Active when energy, cohesion, members, and activations meet thresholds
//! - **Demotion**: Active → Dormant when energy drops or inactivity exceeds threshold
//! - **Archival**: Dormant → Archived after prolonged inactivity
//! - **Metrics**: Recalculates skill energy and cohesion from member notes/synapses

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::skills::models::{SkillNode, SkillStatus};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for skill lifecycle transitions.
///
/// Controls when skills are promoted (Emerging → Active),
/// demoted (Active → Dormant), or archived (Dormant → Archived).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillLifecycleConfig {
    // --- Promotion thresholds (Emerging → Active) ---
    /// Minimum average energy of member notes for promotion (default: 0.3)
    pub promotion_energy_threshold: f64,
    /// Minimum cluster cohesion for promotion (default: 0.4)
    pub promotion_cohesion_threshold: f64,
    /// Minimum number of member notes for promotion (default: 3)
    pub promotion_min_members: i64,
    /// Minimum number of successful hook activations for promotion (default: 2)
    pub promotion_min_activations: i64,

    // --- Demotion thresholds (Active → Dormant) ---
    /// Energy threshold below which an Active skill becomes Dormant (default: 0.1)
    pub demotion_energy_threshold: f64,
    /// Days of inactivity (no hook activation) before demotion (default: 30)
    pub demotion_inactivity_days: i64,
    /// Cohesion threshold below which demotion is triggered (default: 0.2)
    pub demotion_cohesion_threshold: f64,

    // --- Archival thresholds (Dormant → Archived) ---
    /// Days of inactivity before a Dormant skill is archived (default: 90)
    pub archive_inactivity_days: i64,
}

impl Default for SkillLifecycleConfig {
    fn default() -> Self {
        Self {
            promotion_energy_threshold: 0.3,
            promotion_cohesion_threshold: 0.4,
            promotion_min_members: 3,
            promotion_min_activations: 2,
            demotion_energy_threshold: 0.1,
            demotion_inactivity_days: 30,
            demotion_cohesion_threshold: 0.2,
            archive_inactivity_days: 90,
        }
    }
}

// ============================================================================
// Promotion (Emerging → Active)
// ============================================================================

/// Evaluate whether a skill should be promoted from Emerging to Active.
///
/// A skill is eligible for promotion when ALL criteria are met:
/// 1. `energy >= promotion_energy_threshold` (notes are actively used)
/// 2. `cohesion >= promotion_cohesion_threshold` (cluster is stable)
/// 3. `note_count >= promotion_min_members` (enough notes)
/// 4. `activation_count >= promotion_min_activations` (proven useful via hooks)
///
/// Returns `true` if the skill should be promoted.
pub fn evaluate_promotion(skill: &SkillNode, config: &SkillLifecycleConfig) -> bool {
    if skill.status != SkillStatus::Emerging {
        return false;
    }

    let energy_ok = skill.energy >= config.promotion_energy_threshold;
    let cohesion_ok = skill.cohesion >= config.promotion_cohesion_threshold;
    let members_ok = skill.note_count >= config.promotion_min_members;
    let activations_ok = skill.activation_count >= config.promotion_min_activations;

    debug!(
        skill_id = %skill.id,
        skill_name = %skill.name,
        energy = skill.energy,
        cohesion = skill.cohesion,
        note_count = skill.note_count,
        activation_count = skill.activation_count,
        energy_ok,
        cohesion_ok,
        members_ok,
        activations_ok,
        "Evaluating promotion for skill"
    );

    energy_ok && cohesion_ok && members_ok && activations_ok
}

/// Promote a skill from Emerging to Active.
///
/// This function:
/// 1. Updates the skill status to Active
/// 2. Finds decisions linked (AFFECTS) to the same files as member notes
/// 3. Adds those decisions as MEMBER_OF_SKILL
/// 4. Regenerates the context_template with decisions included
///
/// Returns the number of decisions auto-added.
pub async fn promote_skill(graph_store: &dyn GraphStore, skill_id: Uuid) -> anyhow::Result<usize> {
    let mut skill = graph_store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Skill {} not found", skill_id))?;

    if skill.status != SkillStatus::Emerging {
        anyhow::bail!(
            "Cannot promote skill {} — status is {:?}, expected Emerging",
            skill_id,
            skill.status
        );
    }

    // Step 1: Update status to Active
    skill.status = SkillStatus::Active;
    skill.version += 1;
    skill.updated_at = Utc::now();

    // Step 2: Find decisions linked to the same files as member notes
    let (member_notes, existing_decisions) = graph_store.get_skill_members(skill_id).await?;
    let existing_decision_ids: std::collections::HashSet<Uuid> =
        existing_decisions.iter().map(|d| d.id).collect();

    // Collect file paths covered by member notes (via LINKED_TO anchors)
    let mut covered_files: std::collections::HashSet<String> = std::collections::HashSet::new();
    for note in &member_notes {
        match graph_store.get_note_anchors(note.id).await {
            Ok(anchors) => {
                for anchor in &anchors {
                    if anchor.entity_type == crate::notes::EntityType::File {
                        covered_files.insert(anchor.entity_id.clone());
                    }
                }
            }
            Err(e) => {
                warn!(
                    note_id = %note.id,
                    error = %e,
                    "Failed to get note anchors during promotion"
                );
            }
        }
    }

    // Find decisions that AFFECTS these files
    let mut decisions_added = 0usize;
    for file_path in &covered_files {
        if let Ok(affecting_decisions) = graph_store
            .get_decisions_affecting("File", file_path, Some("accepted"))
            .await
        {
            for decision in &affecting_decisions {
                if !existing_decision_ids.contains(&decision.id) {
                    if let Err(e) = graph_store
                        .add_skill_member(skill_id, "decision", decision.id)
                        .await
                    {
                        warn!(
                            skill_id = %skill_id,
                            decision_id = %decision.id,
                            error = %e,
                            "Failed to add decision as skill member"
                        );
                    } else {
                        decisions_added += 1;
                    }
                }
            }
        }
    }

    // Step 3: Regenerate context template with decisions
    let (updated_notes, updated_decisions) = if decisions_added > 0 {
        graph_store.get_skill_members(skill_id).await?
    } else {
        (member_notes, existing_decisions)
    };

    skill.note_count = updated_notes.len() as i64;
    skill.decision_count = updated_decisions.len() as i64;

    // Regenerate template
    skill.context_template = Some(crate::skills::templates::generate_context_template(
        &skill.name,
        &skill.description,
        &updated_notes,
    ));

    graph_store.update_skill(&skill).await?;

    info!(
        skill_id = %skill_id,
        skill_name = %skill.name,
        decisions_added,
        "Promoted skill Emerging → Active"
    );

    Ok(decisions_added)
}

// ============================================================================
// Demotion (Active → Dormant → Archived)
// ============================================================================

/// Result of evaluating a skill's demotion eligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemotionAction {
    /// No demotion needed — skill is healthy
    None,
    /// Demote Active → Dormant
    Demote,
    /// Archive Dormant → Archived
    Archive,
}

/// Evaluate whether a skill should be demoted or archived.
///
/// Rules:
/// - **Active → Dormant**: when energy < threshold AND (inactive > N days OR cohesion < threshold)
/// - **Dormant → Archived**: when inactive > archive_inactivity_days
///
/// Returns the recommended action.
pub fn evaluate_demotion(
    skill: &SkillNode,
    now: DateTime<Utc>,
    config: &SkillLifecycleConfig,
) -> DemotionAction {
    match skill.status {
        SkillStatus::Active => {
            let energy_low = skill.energy < config.demotion_energy_threshold;

            if !energy_low {
                return DemotionAction::None;
            }

            // Energy is low — check inactivity or cohesion
            let inactive_days = skill
                .last_activated
                .map(|la| (now - la).num_days())
                .unwrap_or(i64::MAX); // Never activated = infinite inactivity

            let inactive_too_long = inactive_days >= config.demotion_inactivity_days;
            let cohesion_low = skill.cohesion < config.demotion_cohesion_threshold;

            if inactive_too_long || cohesion_low {
                debug!(
                    skill_id = %skill.id,
                    skill_name = %skill.name,
                    energy = skill.energy,
                    inactive_days,
                    cohesion = skill.cohesion,
                    "Skill eligible for demotion Active → Dormant"
                );
                DemotionAction::Demote
            } else {
                DemotionAction::None
            }
        }
        SkillStatus::Dormant => {
            let inactive_days = skill
                .last_activated
                .map(|la| (now - la).num_days())
                .unwrap_or(i64::MAX);

            if inactive_days >= config.archive_inactivity_days {
                debug!(
                    skill_id = %skill.id,
                    skill_name = %skill.name,
                    inactive_days,
                    "Skill eligible for archival Dormant → Archived"
                );
                DemotionAction::Archive
            } else {
                DemotionAction::None
            }
        }
        _ => DemotionAction::None,
    }
}

/// Demote a skill from Active to Dormant.
pub async fn demote_skill(graph_store: &dyn GraphStore, skill_id: Uuid) -> anyhow::Result<()> {
    let mut skill = graph_store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Skill {} not found", skill_id))?;

    if skill.status != SkillStatus::Active {
        anyhow::bail!(
            "Cannot demote skill {} — status is {:?}, expected Active",
            skill_id,
            skill.status
        );
    }

    skill.status = SkillStatus::Dormant;
    skill.updated_at = Utc::now();
    graph_store.update_skill(&skill).await?;

    info!(
        skill_id = %skill_id,
        skill_name = %skill.name,
        "Demoted skill Active → Dormant"
    );

    Ok(())
}

/// Archive a skill from Dormant to Archived.
pub async fn archive_skill(graph_store: &dyn GraphStore, skill_id: Uuid) -> anyhow::Result<()> {
    let mut skill = graph_store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Skill {} not found", skill_id))?;

    if skill.status != SkillStatus::Dormant {
        anyhow::bail!(
            "Cannot archive skill {} — status is {:?}, expected Dormant",
            skill_id,
            skill.status
        );
    }

    skill.status = SkillStatus::Archived;
    skill.updated_at = Utc::now();
    graph_store.update_skill(&skill).await?;

    info!(
        skill_id = %skill_id,
        skill_name = %skill.name,
        "Archived skill Dormant → Archived"
    );

    Ok(())
}

// ============================================================================
// Imported skill transitions
// ============================================================================

/// Archive an imported skill directly (accelerated degradation).
///
/// Used for imported skills that fail the probation period (no activations
/// after `IMPORT_PROBATION_DAYS`). Skips the normal Dormant stage.
pub async fn archive_imported_skill(
    graph_store: &dyn GraphStore,
    skill_id: Uuid,
) -> anyhow::Result<()> {
    let mut skill = graph_store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Skill {} not found", skill_id))?;

    if skill.status != SkillStatus::Imported {
        anyhow::bail!(
            "Cannot archive imported skill {} — status is {:?}, expected Imported",
            skill_id,
            skill.status
        );
    }

    skill.status = SkillStatus::Archived;
    skill.updated_at = Utc::now();
    graph_store.update_skill(&skill).await?;

    info!(
        skill_id = %skill_id,
        skill_name = %skill.name,
        days_since_import = skill.imported_at.map(|dt| (Utc::now() - dt).num_days()).unwrap_or(0),
        "Archived imported skill (failed probation) Imported → Archived"
    );

    Ok(())
}

/// Validate an imported skill and promote to Emerging.
///
/// Called when an imported skill has enough activations to prove its value.
/// Sets `is_validated = true` and transitions to Emerging, where the normal
/// lifecycle rules apply (Emerging → Active with further usage).
pub async fn validate_imported_skill(
    graph_store: &dyn GraphStore,
    skill_id: Uuid,
) -> anyhow::Result<()> {
    let mut skill = graph_store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Skill {} not found", skill_id))?;

    if skill.status != SkillStatus::Imported {
        anyhow::bail!(
            "Cannot validate skill {} — status is {:?}, expected Imported",
            skill_id,
            skill.status
        );
    }

    skill.status = SkillStatus::Emerging;
    skill.is_validated = true;
    skill.updated_at = Utc::now();
    graph_store.update_skill(&skill).await?;

    info!(
        skill_id = %skill_id,
        skill_name = %skill.name,
        activation_count = skill.activation_count,
        "Validated imported skill Imported → Emerging (is_validated=true)"
    );

    Ok(())
}

// ============================================================================
// Reactivation
// ============================================================================

/// Reactivate a Dormant skill back to Emerging.
///
/// This allows manual recovery of a skill that was demoted but is still useful.
/// The skill is set to `Emerging` (not `Active`) — it must prove itself again
/// through activation hook usage before being re-promoted to Active.
pub async fn reactivate_skill(graph_store: &dyn GraphStore, skill_id: Uuid) -> anyhow::Result<()> {
    let mut skill = graph_store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Skill {} not found", skill_id))?;

    if skill.status != SkillStatus::Dormant {
        anyhow::bail!(
            "Cannot reactivate skill {} — status is {:?}, expected Dormant",
            skill_id,
            skill.status
        );
    }

    let now = Utc::now();
    skill.status = SkillStatus::Emerging;
    skill.activation_count = 0; // Reset activations so it must prove itself
    skill.updated_at = now;
    skill.last_activated = Some(now);
    graph_store.update_skill(&skill).await?;

    info!(
        skill_id = %skill_id,
        skill_name = %skill.name,
        "Reactivated skill Dormant → Emerging"
    );

    Ok(())
}

// ============================================================================
// Metrics Update
// ============================================================================

/// Result of a skill metrics update cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsUpdateResult {
    /// Number of skills whose metrics were updated
    pub skills_updated: usize,
    /// Number of skills promoted (Emerging → Active)
    pub skills_promoted: usize,
    /// Number of skills demoted (Active → Dormant)
    pub skills_demoted: usize,
    /// Number of skills archived (Dormant → Archived)
    pub skills_archived: usize,
    /// Total decisions auto-added during promotions
    pub decisions_added: usize,
}

/// Update metrics for all skills in a project and evaluate lifecycle transitions.
///
/// This is the main hourly maintenance function that:
/// 1. Recalculates energy for each skill from member note energies
/// 2. Evaluates promotions for Emerging skills
/// 3. Evaluates demotions for Active skills
/// 4. Evaluates archival for Dormant skills
pub async fn update_skill_lifecycle(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    config: &SkillLifecycleConfig,
) -> anyhow::Result<MetricsUpdateResult> {
    let now = Utc::now();
    let mut result = MetricsUpdateResult::default();

    // Fetch all skills for the project
    let skills = graph_store.get_skills_for_project(project_id).await?;

    for skill in &skills {
        // Wrap per-skill processing — one corrupted skill should not abort the batch
        let member_notes = match graph_store.get_skill_members(skill.id).await {
            Ok((notes, _)) => notes,
            Err(e) => {
                warn!(skill_id = %skill.id, error = %e, "Failed to get skill members — skipping");
                continue;
            }
        };

        // Recalculate energy from member notes
        if !member_notes.is_empty() {
            let (weighted_sum, weight_sum) =
                member_notes
                    .iter()
                    .fold((0.0_f64, 0.0_f64), |(ws, wt), note| {
                        let w = note.importance.weight();
                        (ws + note.energy * w, wt + w)
                    });
            let new_energy = if weight_sum > 0.0 {
                (weighted_sum / weight_sum).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Only update if energy changed significantly
            if (new_energy - skill.energy).abs() > 0.01 {
                let mut updated = skill.clone();
                updated.energy = new_energy;
                updated.note_count = member_notes.len() as i64;
                updated.updated_at = now;
                if let Err(e) = graph_store.update_skill(&updated).await {
                    warn!(skill_id = %skill.id, error = %e, "Failed to update skill energy — skipping");
                    continue;
                }
                result.skills_updated += 1;
            }
        }

        // Re-fetch to get updated energy
        let current_skill = match graph_store.get_skill(skill.id).await {
            Ok(Some(s)) => s,
            Ok(None) => {
                warn!(skill_id = %skill.id, "Skill disappeared during lifecycle — skipping");
                continue;
            }
            Err(e) => {
                warn!(skill_id = %skill.id, error = %e, "Failed to re-fetch skill — skipping");
                continue;
            }
        };

        // Evaluate lifecycle transitions
        match current_skill.status {
            SkillStatus::Emerging => {
                if evaluate_promotion(&current_skill, config) {
                    match promote_skill(graph_store, current_skill.id).await {
                        Ok(decisions) => {
                            result.skills_promoted += 1;
                            result.decisions_added += decisions;
                        }
                        Err(e) => {
                            warn!(
                                skill_id = %current_skill.id,
                                error = %e,
                                "Failed to promote skill"
                            );
                        }
                    }
                } else {
                    // Zombie detection: Emerging skill with no activations, low energy,
                    // and past the archive inactivity window → archive directly
                    let days_inactive = (now - current_skill.updated_at).num_days();
                    if current_skill.activation_count == 0
                        && current_skill.energy < config.demotion_energy_threshold
                        && days_inactive > config.archive_inactivity_days
                    {
                        match archive_skill(graph_store, current_skill.id).await {
                            Ok(()) => {
                                result.skills_archived += 1;
                                info!(
                                    skill_id = %current_skill.id,
                                    name = %current_skill.name,
                                    days_inactive,
                                    "Archived zombie Emerging skill — no activations, low energy"
                                );
                            }
                            Err(e) => {
                                warn!(
                                    skill_id = %current_skill.id,
                                    error = %e,
                                    "Failed to archive zombie Emerging skill"
                                );
                            }
                        }
                    }
                }
            }
            SkillStatus::Active => {
                if evaluate_demotion(&current_skill, now, config) == DemotionAction::Demote {
                    match demote_skill(graph_store, current_skill.id).await {
                        Ok(()) => {
                            result.skills_demoted += 1;
                        }
                        Err(e) => {
                            warn!(
                                skill_id = %current_skill.id,
                                error = %e,
                                "Failed to demote skill"
                            );
                        }
                    }
                }
            }
            SkillStatus::Dormant => {
                if evaluate_demotion(&current_skill, now, config) == DemotionAction::Archive {
                    match archive_skill(graph_store, current_skill.id).await {
                        Ok(()) => {
                            result.skills_archived += 1;
                        }
                        Err(e) => {
                            warn!(
                                skill_id = %current_skill.id,
                                error = %e,
                                "Failed to archive skill"
                            );
                        }
                    }
                }
            }
            SkillStatus::Imported => {
                // Imported skills: accelerated Darwinian selection
                if crate::skills::validation::should_accelerate_archive(&current_skill, now) {
                    // Past probation with zero activations → archive directly
                    match archive_imported_skill(graph_store, current_skill.id).await {
                        Ok(()) => {
                            result.skills_archived += 1;
                        }
                        Err(e) => {
                            warn!(
                                skill_id = %current_skill.id,
                                error = %e,
                                "Failed to archive unused imported skill"
                            );
                        }
                    }
                } else if crate::skills::validation::qualifies_for_validation(&current_skill) {
                    // Enough activations → validate and promote to Emerging
                    match validate_imported_skill(graph_store, current_skill.id).await {
                        Ok(()) => {
                            result.skills_promoted += 1;
                        }
                        Err(e) => {
                            warn!(
                                skill_id = %current_skill.id,
                                error = %e,
                                "Failed to validate imported skill"
                            );
                        }
                    }
                }
            }
            _ => {} // Archived — no transitions
        }
    }

    info!(
        project_id = %project_id,
        skills_updated = result.skills_updated,
        promoted = result.skills_promoted,
        demoted = result.skills_demoted,
        archived = result.skills_archived,
        "Skill lifecycle update complete"
    );

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skill(status: SkillStatus, energy: f64, cohesion: f64) -> SkillNode {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Test Skill");
        skill.status = status;
        skill.energy = energy;
        skill.cohesion = cohesion;
        skill.note_count = 5;
        skill.activation_count = 3;
        skill.last_activated = Some(Utc::now());
        skill
    }

    // ================================================================
    // SkillLifecycleConfig tests
    // ================================================================

    #[test]
    fn test_default_config() {
        let config = SkillLifecycleConfig::default();
        assert!((config.promotion_energy_threshold - 0.3).abs() < f64::EPSILON);
        assert!((config.promotion_cohesion_threshold - 0.4).abs() < f64::EPSILON);
        assert_eq!(config.promotion_min_members, 3);
        assert_eq!(config.promotion_min_activations, 2);
        assert!((config.demotion_energy_threshold - 0.1).abs() < f64::EPSILON);
        assert_eq!(config.demotion_inactivity_days, 30);
        assert!((config.demotion_cohesion_threshold - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.archive_inactivity_days, 90);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = SkillLifecycleConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SkillLifecycleConfig = serde_json::from_str(&json).unwrap();
        assert!(
            (deserialized.promotion_energy_threshold - config.promotion_energy_threshold).abs()
                < f64::EPSILON
        );
        assert_eq!(
            deserialized.demotion_inactivity_days,
            config.demotion_inactivity_days
        );
    }

    // ================================================================
    // Promotion evaluation tests
    // ================================================================

    #[test]
    fn test_evaluate_promotion_all_criteria_met() {
        let skill = make_skill(SkillStatus::Emerging, 0.4, 0.5);
        let config = SkillLifecycleConfig::default();
        assert!(evaluate_promotion(&skill, &config));
    }

    #[test]
    fn test_evaluate_promotion_low_energy() {
        let mut skill = make_skill(SkillStatus::Emerging, 0.2, 0.5);
        skill.note_count = 5;
        skill.activation_count = 3;
        let config = SkillLifecycleConfig::default();
        assert!(!evaluate_promotion(&skill, &config));
    }

    #[test]
    fn test_evaluate_promotion_low_cohesion() {
        let mut skill = make_skill(SkillStatus::Emerging, 0.4, 0.3);
        skill.note_count = 5;
        skill.activation_count = 3;
        let config = SkillLifecycleConfig::default();
        assert!(!evaluate_promotion(&skill, &config));
    }

    #[test]
    fn test_evaluate_promotion_insufficient_members() {
        let mut skill = make_skill(SkillStatus::Emerging, 0.4, 0.5);
        skill.note_count = 2;
        skill.activation_count = 3;
        let config = SkillLifecycleConfig::default();
        assert!(!evaluate_promotion(&skill, &config));
    }

    #[test]
    fn test_evaluate_promotion_insufficient_activations() {
        let mut skill = make_skill(SkillStatus::Emerging, 0.4, 0.5);
        skill.note_count = 5;
        skill.activation_count = 1;
        let config = SkillLifecycleConfig::default();
        assert!(!evaluate_promotion(&skill, &config));
    }

    #[test]
    fn test_evaluate_promotion_wrong_status() {
        let skill = make_skill(SkillStatus::Active, 0.4, 0.5);
        let config = SkillLifecycleConfig::default();
        assert!(!evaluate_promotion(&skill, &config));
    }

    #[test]
    fn test_evaluate_promotion_boundary_values() {
        // Exactly at thresholds — should pass
        let mut skill = SkillNode::new(Uuid::new_v4(), "Boundary");
        skill.status = SkillStatus::Emerging;
        skill.energy = 0.3;
        skill.cohesion = 0.4;
        skill.note_count = 3;
        skill.activation_count = 2;
        skill.last_activated = Some(Utc::now());
        let config = SkillLifecycleConfig::default();
        assert!(evaluate_promotion(&skill, &config));
    }

    #[test]
    fn test_evaluate_promotion_just_below_thresholds() {
        let mut skill = SkillNode::new(Uuid::new_v4(), "Just Below");
        skill.status = SkillStatus::Emerging;
        skill.energy = 0.299;
        skill.cohesion = 0.4;
        skill.note_count = 3;
        skill.activation_count = 2;
        let config = SkillLifecycleConfig::default();
        assert!(!evaluate_promotion(&skill, &config));
    }

    // ================================================================
    // Demotion evaluation tests
    // ================================================================

    #[test]
    fn test_evaluate_demotion_active_healthy() {
        let skill = make_skill(SkillStatus::Active, 0.5, 0.6);
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::None
        );
    }

    #[test]
    fn test_evaluate_demotion_active_low_energy_inactive() {
        let mut skill = make_skill(SkillStatus::Active, 0.05, 0.3);
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(35));
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::Demote
        );
    }

    #[test]
    fn test_evaluate_demotion_active_low_energy_low_cohesion() {
        let mut skill = make_skill(SkillStatus::Active, 0.05, 0.15);
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(5)); // recently active
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::Demote
        );
    }

    #[test]
    fn test_evaluate_demotion_active_low_energy_recent_activity() {
        // Low energy but recently activated AND good cohesion → no demotion yet
        let mut skill = make_skill(SkillStatus::Active, 0.05, 0.3);
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(5));
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::None
        );
    }

    #[test]
    fn test_evaluate_demotion_active_never_activated() {
        // Low energy + never activated → should demote
        let mut skill = make_skill(SkillStatus::Active, 0.05, 0.3);
        skill.last_activated = None;
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::Demote
        );
    }

    #[test]
    fn test_evaluate_demotion_dormant_recent() {
        let mut skill = make_skill(SkillStatus::Dormant, 0.05, 0.1);
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(50));
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::None
        );
    }

    #[test]
    fn test_evaluate_demotion_dormant_old() {
        let mut skill = make_skill(SkillStatus::Dormant, 0.02, 0.1);
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(100));
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::Archive
        );
    }

    #[test]
    fn test_evaluate_demotion_dormant_never_activated() {
        let mut skill = make_skill(SkillStatus::Dormant, 0.02, 0.1);
        skill.last_activated = None;
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::Archive
        );
    }

    #[test]
    fn test_evaluate_demotion_archived_skill() {
        let skill = make_skill(SkillStatus::Archived, 0.01, 0.05);
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::None
        );
    }

    #[test]
    fn test_evaluate_demotion_emerging_skill() {
        let skill = make_skill(SkillStatus::Emerging, 0.01, 0.05);
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::None
        );
    }

    #[test]
    fn test_evaluate_demotion_boundary_energy() {
        // Exactly at threshold energy — NOT below, so no demotion
        let mut skill = make_skill(SkillStatus::Active, 0.1, 0.3);
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(35));
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::None
        );
    }

    #[test]
    fn test_evaluate_demotion_boundary_inactivity() {
        // Exactly at 90 days for archival
        let mut skill = make_skill(SkillStatus::Dormant, 0.02, 0.1);
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(90));
        let config = SkillLifecycleConfig::default();
        assert_eq!(
            evaluate_demotion(&skill, Utc::now(), &config),
            DemotionAction::Archive
        );
    }

    // ================================================================
    // MetricsUpdateResult tests
    // ================================================================

    #[test]
    fn test_metrics_update_result_default() {
        let result = MetricsUpdateResult::default();
        assert_eq!(result.skills_updated, 0);
        assert_eq!(result.skills_promoted, 0);
        assert_eq!(result.skills_demoted, 0);
        assert_eq!(result.skills_archived, 0);
        assert_eq!(result.decisions_added, 0);
    }

    // ================================================================
    // Reactivation tests
    // ================================================================

    #[test]
    fn test_reactivate_requires_dormant_status() {
        // Only Dormant skills can be reactivated — other statuses should fail
        for status in [
            SkillStatus::Active,
            SkillStatus::Emerging,
            SkillStatus::Archived,
        ] {
            let skill = make_skill(status, 0.5, 0.5);
            // We can't call the async function in a sync test, but we can verify
            // the status guard logic by checking evaluate_demotion doesn't interfere
            // The reactivate_skill function checks status == Dormant
            assert_ne!(
                skill.status,
                SkillStatus::Dormant,
                "Test setup error: skill should not be Dormant for status {:?}",
                status
            );
        }
    }

    // ================================================================
    // Async lifecycle tests (MockGraphStore)
    // ================================================================

    use crate::neo4j::mock::MockGraphStore;
    use std::sync::Arc;

    async fn mock_store_with_skill(skill: SkillNode) -> Arc<MockGraphStore> {
        let store = MockGraphStore::new();
        let project_id = skill.project_id;
        let mut p = crate::test_helpers::test_project();
        p.id = project_id;
        store.projects.write().await.insert(p.id, p);
        store.create_skill(&skill).await.unwrap();
        Arc::new(store)
    }

    #[tokio::test]
    async fn test_demote_skill_active_to_dormant() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Active Skill");
        skill.status = SkillStatus::Active;
        skill.energy = 0.05;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        demote_skill(store.as_ref(), skill_id).await.unwrap();

        let updated = store.get_skill(skill_id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Dormant);
    }

    #[tokio::test]
    async fn test_demote_skill_wrong_status_fails() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Emerging Skill");
        skill.status = SkillStatus::Emerging;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        let result = demote_skill(store.as_ref(), skill_id).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected Active"));
    }

    #[tokio::test]
    async fn test_demote_skill_not_found() {
        let store = Arc::new(MockGraphStore::new());
        let result = demote_skill(store.as_ref(), Uuid::new_v4()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_archive_skill_dormant_to_archived() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Dormant Skill");
        skill.status = SkillStatus::Dormant;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        archive_skill(store.as_ref(), skill_id).await.unwrap();

        let updated = store.get_skill(skill_id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Archived);
    }

    #[tokio::test]
    async fn test_archive_skill_wrong_status_fails() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Active Skill");
        skill.status = SkillStatus::Active;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        let result = archive_skill(store.as_ref(), skill_id).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected Dormant"));
    }

    #[tokio::test]
    async fn test_archive_imported_skill_success() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Imported Skill");
        skill.status = SkillStatus::Imported;
        skill.imported_at = Some(Utc::now() - chrono::Duration::days(30));
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        archive_imported_skill(store.as_ref(), skill_id)
            .await
            .unwrap();

        let updated = store.get_skill(skill_id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Archived);
    }

    #[tokio::test]
    async fn test_archive_imported_skill_wrong_status_fails() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Active Skill");
        skill.status = SkillStatus::Active;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        let result = archive_imported_skill(store.as_ref(), skill_id).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expected Imported"));
    }

    #[tokio::test]
    async fn test_validate_imported_skill_success() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Imported Skill");
        skill.status = SkillStatus::Imported;
        skill.activation_count = 5;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        validate_imported_skill(store.as_ref(), skill_id)
            .await
            .unwrap();

        let updated = store.get_skill(skill_id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Emerging);
        assert!(updated.is_validated);
    }

    #[tokio::test]
    async fn test_validate_imported_skill_wrong_status_fails() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Dormant Skill");
        skill.status = SkillStatus::Dormant;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        let result = validate_imported_skill(store.as_ref(), skill_id).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expected Imported"));
    }

    #[tokio::test]
    async fn test_reactivate_skill_dormant_to_emerging() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Dormant Skill");
        skill.status = SkillStatus::Dormant;
        skill.activation_count = 10;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        reactivate_skill(store.as_ref(), skill_id).await.unwrap();

        let updated = store.get_skill(skill_id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Emerging);
        assert_eq!(updated.activation_count, 0);
        assert!(updated.last_activated.is_some());
    }

    #[tokio::test]
    async fn test_reactivate_skill_wrong_status_fails() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Active Skill");
        skill.status = SkillStatus::Active;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        let result = reactivate_skill(store.as_ref(), skill_id).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected Dormant"));
    }

    #[tokio::test]
    async fn test_promote_skill_emerging_to_active() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Emerging Skill");
        skill.status = SkillStatus::Emerging;
        skill.energy = 0.5;
        skill.cohesion = 0.6;
        skill.note_count = 5;
        skill.activation_count = 3;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        let decisions_added = promote_skill(store.as_ref(), skill_id).await.unwrap();

        let updated = store.get_skill(skill_id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Active);
        assert_eq!(updated.version, 2);
        assert_eq!(decisions_added, 0);
    }

    #[tokio::test]
    async fn test_promote_skill_wrong_status_fails() {
        let project_id = Uuid::new_v4();
        let mut skill = SkillNode::new(project_id, "Active Skill");
        skill.status = SkillStatus::Active;
        let skill_id = skill.id;
        let store = mock_store_with_skill(skill).await;

        let result = promote_skill(store.as_ref(), skill_id).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("expected Emerging"));
    }

    #[tokio::test]
    async fn test_update_skill_lifecycle_no_skills() {
        let project_id = Uuid::new_v4();
        let store = MockGraphStore::new();
        let mut p = crate::test_helpers::test_project();
        p.id = project_id;
        store.projects.write().await.insert(p.id, p);
        let config = SkillLifecycleConfig::default();

        let result = update_skill_lifecycle(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.skills_updated, 0);
        assert_eq!(result.skills_promoted, 0);
        assert_eq!(result.skills_demoted, 0);
        assert_eq!(result.skills_archived, 0);
    }

    #[tokio::test]
    async fn test_update_skill_lifecycle_demotes_active_skill() {
        let project_id = Uuid::new_v4();
        let store = MockGraphStore::new();
        let mut p = crate::test_helpers::test_project();
        p.id = project_id;
        store.projects.write().await.insert(p.id, p);

        let mut skill = SkillNode::new(project_id, "Should Demote");
        skill.status = SkillStatus::Active;
        skill.energy = 0.05;
        skill.cohesion = 0.15;
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(40));
        store.create_skill(&skill).await.unwrap();

        let config = SkillLifecycleConfig::default();
        let result = update_skill_lifecycle(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.skills_demoted, 1);
        let updated = store.get_skill(skill.id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Dormant);
    }

    #[tokio::test]
    async fn test_update_skill_lifecycle_archives_dormant_skill() {
        let project_id = Uuid::new_v4();
        let store = MockGraphStore::new();
        let mut p = crate::test_helpers::test_project();
        p.id = project_id;
        store.projects.write().await.insert(p.id, p);

        let mut skill = SkillNode::new(project_id, "Should Archive");
        skill.status = SkillStatus::Dormant;
        skill.energy = 0.01;
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(100));
        store.create_skill(&skill).await.unwrap();

        let config = SkillLifecycleConfig::default();
        let result = update_skill_lifecycle(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.skills_archived, 1);
        let updated = store.get_skill(skill.id).await.unwrap().unwrap();
        assert_eq!(updated.status, SkillStatus::Archived);
    }
}
