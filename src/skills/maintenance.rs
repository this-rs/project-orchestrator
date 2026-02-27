//! Skill Maintenance Scheduler — Hourly, Daily, Weekly
//!
//! Provides three levels of periodic maintenance for the skill system:
//!
//! - **Hourly**: Evaluate promotions/demotions, update metrics
//! - **Daily**: Decay synapses, update energy scores, recalculate cohesion
//! - **Weekly**: Re-detect skills via Louvain, evolve (merge/split/grow/shrink)
//!
//! Each level can be triggered manually via the admin MCP tool or API,
//! or automatically via an optional background timer.

use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::neo4j::traits::GraphStore;
use crate::skills::detection::{detect_skills_pipeline, SkillDetectionConfig};
use crate::skills::evolution::{analyze_evolution, execute_evolution, EvolutionResult};
use crate::skills::lifecycle::{update_skill_lifecycle, MetricsUpdateResult, SkillLifecycleConfig};

// ============================================================================
// Maintenance Result
// ============================================================================

/// Combined result of a maintenance run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaintenanceResult {
    /// Level that was run
    pub level: String,
    /// Lifecycle metrics (hourly)
    pub lifecycle: Option<MetricsUpdateResult>,
    /// Synapse decay results (daily)
    pub synapses_decayed: Option<usize>,
    /// Synapses pruned below threshold (daily)
    pub synapses_pruned: Option<usize>,
    /// Evolution results (weekly)
    pub evolution: Option<EvolutionResult>,
    /// Skills detected (weekly)
    pub skills_detected: Option<usize>,
    /// Any errors encountered (non-fatal)
    pub warnings: Vec<String>,
}

// ============================================================================
// Maintenance Configuration
// ============================================================================

/// Configuration for the skill maintenance system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMaintenanceConfig {
    /// Lifecycle config for promotion/demotion thresholds
    pub lifecycle: SkillLifecycleConfig,
    /// Detection config for Louvain parameters
    pub detection: SkillDetectionConfig,

    // --- Decay parameters (daily) ---
    /// Amount to subtract from each synapse weight per decay cycle (default: 0.01)
    pub synapse_decay_amount: f64,
    /// Prune synapses below this weight (default: 0.1)
    pub synapse_prune_threshold: f64,

    // --- Evolution parameters (weekly) ---
    /// Jaccard overlap threshold for skill deduplication (default: 0.7)
    pub evolution_overlap_threshold: f64,
    /// Minimum new synapses from backfill to trigger re-detection (default: 50)
    pub backfill_recluster_threshold: usize,

    // --- Scheduler (optional) ---
    /// Whether automatic maintenance is enabled (default: false)
    pub enabled: bool,
}

impl Default for SkillMaintenanceConfig {
    fn default() -> Self {
        Self {
            lifecycle: SkillLifecycleConfig::default(),
            detection: SkillDetectionConfig::default(),
            synapse_decay_amount: 0.01,
            synapse_prune_threshold: 0.1,
            evolution_overlap_threshold: 0.7,
            backfill_recluster_threshold: 50,
            enabled: false,
        }
    }
}

// ============================================================================
// Hourly Maintenance
// ============================================================================

/// Run hourly maintenance: evaluate promotions/demotions for all skills.
///
/// This is lightweight and safe to run frequently.
pub async fn run_hourly_maintenance(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    config: &SkillMaintenanceConfig,
) -> anyhow::Result<MaintenanceResult> {
    info!(project_id = %project_id, "Starting hourly skill maintenance");

    let lifecycle = update_skill_lifecycle(graph_store, project_id, &config.lifecycle).await?;

    let result = MaintenanceResult {
        level: "hourly".to_string(),
        lifecycle: Some(lifecycle),
        ..Default::default()
    };

    info!(
        project_id = %project_id,
        promoted = result.lifecycle.as_ref().map_or(0, |l| l.skills_promoted),
        demoted = result.lifecycle.as_ref().map_or(0, |l| l.skills_demoted),
        "Hourly maintenance complete"
    );

    Ok(result)
}

// ============================================================================
// Daily Maintenance
// ============================================================================

/// Run daily maintenance: decay synapses + update energy scores + lifecycle.
///
/// This includes hourly maintenance plus synapse decay.
pub async fn run_daily_maintenance(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    config: &SkillMaintenanceConfig,
) -> anyhow::Result<MaintenanceResult> {
    info!(project_id = %project_id, "Starting daily skill maintenance");

    let mut result = MaintenanceResult {
        level: "daily".to_string(),
        ..Default::default()
    };

    // Step 1: Decay synapses (global — not project-scoped)
    match graph_store
        .decay_synapses(config.synapse_decay_amount, config.synapse_prune_threshold)
        .await
    {
        Ok((decayed, pruned)) => {
            result.synapses_decayed = Some(decayed);
            result.synapses_pruned = Some(pruned);
        }
        Err(e) => {
            warn!(error = %e, "Failed to decay synapses");
            result.warnings.push(format!("Synapse decay failed: {}", e));
        }
    }

    // Step 2: Update energy scores (half-life decay for note energies, global)
    if let Err(e) = graph_store.update_energy_scores(90.0).await {
        warn!(error = %e, "Failed to update energy scores");
        result.warnings.push(format!("Energy update failed: {}", e));
    }

    // Step 3: Lifecycle evaluation (promotions/demotions)
    match update_skill_lifecycle(graph_store, project_id, &config.lifecycle).await {
        Ok(lifecycle) => {
            result.lifecycle = Some(lifecycle);
        }
        Err(e) => {
            warn!(error = %e, "Failed to update skill lifecycle");
            result
                .warnings
                .push(format!("Lifecycle update failed: {}", e));
        }
    }

    info!(
        project_id = %project_id,
        synapses_decayed = result.synapses_decayed.unwrap_or(0),
        synapses_pruned = result.synapses_pruned.unwrap_or(0),
        "Daily maintenance complete"
    );

    Ok(result)
}

// ============================================================================
// Weekly Maintenance
// ============================================================================

/// Run weekly maintenance: re-detect skills + evolve + daily maintenance.
///
/// This is the most expensive operation and includes re-running Louvain
/// community detection to catch new clusters and evolve existing skills.
pub async fn run_weekly_maintenance(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    config: &SkillMaintenanceConfig,
) -> anyhow::Result<MaintenanceResult> {
    info!(project_id = %project_id, "Starting weekly skill maintenance");

    // Step 1: Run daily maintenance first
    let mut result = run_daily_maintenance(graph_store, project_id, config).await?;
    result.level = "weekly".to_string();

    // Step 2: Re-detect skills via Louvain
    match detect_skills_pipeline(graph_store, project_id, &config.detection).await {
        Ok(detection_result) => {
            result.skills_detected = Some(detection_result.skills_detected);

            if detection_result.status == crate::skills::detection::ClusterDetectionStatus::Success
                && !detection_result.skill_ids.is_empty()
            {
                // Step 3: Evolve existing skills based on new clusters
                let existing_skills = graph_store.get_skills_for_project(project_id).await?;
                let mut existing_members: Vec<(Uuid, Vec<String>)> = Vec::new();
                for skill in &existing_skills {
                    let (notes, _) = graph_store.get_skill_members(skill.id).await?;
                    let member_ids: Vec<String> = notes.iter().map(|n| n.id.to_string()).collect();
                    existing_members.push((skill.id, member_ids));
                }

                // Get new candidates from a fresh detection
                let edges = graph_store
                    .get_synapse_graph(project_id, config.detection.min_synapse_weight)
                    .await?;
                let detection = crate::skills::detection::detect_skill_candidates(
                    &edges,
                    &project_id.to_string(),
                    &config.detection,
                );

                if !detection.candidates.is_empty() {
                    let evolutions = analyze_evolution(
                        &existing_members,
                        &detection.candidates,
                        config.evolution_overlap_threshold,
                    );

                    // Fetch notes for evolution
                    let mut notes_map = std::collections::HashMap::new();
                    for evolution in &evolutions {
                        let note_ids: Vec<&str> = match evolution {
                            crate::skills::evolution::SkillEvolution::Stable {
                                notes_to_add,
                                ..
                            } => notes_to_add.iter().map(|s| s.as_str()).collect(),
                            crate::skills::evolution::SkillEvolution::Split {
                                candidates, ..
                            } => candidates
                                .iter()
                                .flat_map(|c| c.member_note_ids.iter().map(|s| s.as_str()))
                                .collect(),
                            crate::skills::evolution::SkillEvolution::New { candidate } => {
                                candidate
                                    .member_note_ids
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect()
                            }
                            _ => vec![],
                        };
                        for note_id_str in note_ids {
                            if !notes_map.contains_key(note_id_str) {
                                if let Ok(uuid) = Uuid::parse_str(note_id_str) {
                                    if let Ok(Some(note)) = graph_store.get_note(uuid).await {
                                        notes_map.insert(note_id_str.to_string(), note);
                                    }
                                }
                            }
                        }
                    }

                    match execute_evolution(graph_store, &evolutions, &notes_map, project_id).await
                    {
                        Ok(evolution) => {
                            result.evolution = Some(evolution);
                        }
                        Err(e) => {
                            error!(error = %e, "Failed to execute skill evolution");
                            result
                                .warnings
                                .push(format!("Evolution execution failed: {}", e));
                        }
                    }
                }
            }
        }
        Err(e) => {
            warn!(error = %e, "Failed to detect skills");
            result
                .warnings
                .push(format!("Skill detection failed: {}", e));
        }
    }

    info!(
        project_id = %project_id,
        skills_detected = result.skills_detected.unwrap_or(0),
        merged = result.evolution.as_ref().map_or(0, |e| e.merged.len()),
        split = result.evolution.as_ref().map_or(0, |e| e.split.len()),
        grown = result.evolution.as_ref().map_or(0, |e| e.grown.len()),
        "Weekly maintenance complete"
    );

    Ok(result)
}

// ============================================================================
// Full Maintenance
// ============================================================================

/// Run all maintenance levels sequentially: hourly + daily + weekly.
pub async fn run_full_maintenance(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    config: &SkillMaintenanceConfig,
) -> anyhow::Result<MaintenanceResult> {
    info!(project_id = %project_id, "Starting full skill maintenance");
    let mut result = run_weekly_maintenance(graph_store, project_id, config).await?;
    result.level = "full".to_string();
    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SkillMaintenanceConfig::default();
        assert!(!config.enabled);
        assert!((config.synapse_decay_amount - 0.01).abs() < f64::EPSILON);
        assert!((config.synapse_prune_threshold - 0.1).abs() < f64::EPSILON);
        assert!((config.evolution_overlap_threshold - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.backfill_recluster_threshold, 50);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = SkillMaintenanceConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SkillMaintenanceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.enabled, config.enabled);
        assert!(
            (deserialized.synapse_decay_amount - config.synapse_decay_amount).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_maintenance_result_default() {
        let result = MaintenanceResult::default();
        assert!(result.level.is_empty());
        assert!(result.lifecycle.is_none());
        assert!(result.evolution.is_none());
        assert!(result.warnings.is_empty());
    }
}
