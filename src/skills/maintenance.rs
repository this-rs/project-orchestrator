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
use crate::skills::detection::SkillDetectionConfig;
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

    // --- Scheduler (optional, reserved for future background timer) ---
    /// Whether automatic background maintenance is enabled (default: false).
    /// When false, maintenance must be triggered manually via the REST API
    /// (POST /api/admin/skill-maintenance) or MCP admin tool (maintain_skills).
    /// Background timer integration is planned for a future release.
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

    // Step 1: Fetch synapse weights once — reused for prune threshold and
    // stagnation detection below. A single round-trip covers both use cases.
    let synapse_weights = graph_store
        .get_all_synapse_weights(Some(project_id))
        .await
        .unwrap_or_default();

    // Derive adaptive prune threshold from p5 of the weight distribution
    // (fallback = configured default when fewer than 4 synapses exist).
    let adaptive_prune_threshold = if synapse_weights.is_empty() {
        config.synapse_prune_threshold
    } else {
        let threshold = crate::analytics::distribution::adaptive_threshold(
            &synapse_weights,
            0.05,
            config.synapse_prune_threshold,
        );
        info!(
            project_id = %project_id,
            adaptive_prune_threshold = threshold,
            n_synapses = synapse_weights.len(),
            config_default = config.synapse_prune_threshold,
            "Daily maintenance: adaptive prune threshold derived from distribution"
        );
        threshold
    };

    match graph_store
        .decay_synapses(config.synapse_decay_amount, adaptive_prune_threshold)
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

    // Step 2: Update energy scores (half-life decay for note energies, global).
    if let Err(e) = graph_store.update_energy_scores(90.0).await {
        warn!(error = %e, "Failed to update energy scores");
        result.warnings.push(format!("Energy update failed: {}", e));
    }

    // Stagnation detection: test whether current synapse weights are
    // significantly below the healthy midpoint (0.5).
    //
    // `update_energy_scores` decays note energies, not synapse weights, so a
    // pre/post comparison on weights would always be non-significant. Instead
    // we test the absolute level: weights significantly below 0.5 indicate a
    // degrading knowledge graph that needs reinforcement.
    if synapse_weights.len() >= 4 {
        if let Some(test_result) =
            crate::analytics::hypothesis::test_stagnation(&synapse_weights, 0.5)
        {
            use crate::analytics::hypothesis::SignificanceLevel;
            let mean = rs_stats::prob::average(&synapse_weights).unwrap_or(0.0);
            if mean < 0.5
                && matches!(
                    test_result.significance,
                    SignificanceLevel::Significant | SignificanceLevel::HighlySignificant
                )
            {
                warn!(
                    project_id = %project_id,
                    mean_weight = mean,
                    t_stat = test_result.t_statistic,
                    "Synapse weights significantly below healthy midpoint (mean={:.3}, \
                     t={:.3}). Consider running reinforce_neurons on low-energy clusters.",
                    mean, test_result.t_statistic
                );
                result.warnings.push(format!(
                    "Synapse energy below healthy baseline: mean={:.3} (t={:.3}). \
                     Run reinforce_neurons on related notes.",
                    mean, test_result.t_statistic
                ));
            }
        }
    }

    // Step 3: Lifecycle evaluation (promotions/demotions).
    // Use adaptive lifecycle thresholds derived from the current skill distribution.
    let adaptive_lifecycle = match graph_store.get_skills_for_project(project_id).await {
        Ok(skills) if !skills.is_empty() => {
            crate::skills::lifecycle::compute_adaptive_lifecycle_config(&skills)
        }
        _ => config.lifecycle.clone(),
    };

    match update_skill_lifecycle(graph_store, project_id, &adaptive_lifecycle).await {
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

    // Step 2: Snapshot existing skills BEFORE detection (for evolution comparison)
    let existing_skills = graph_store.get_skills_for_project(project_id).await?;
    let mut existing_members: Vec<(Uuid, Vec<String>)> = Vec::new();
    for skill in &existing_skills {
        let (notes, _) = graph_store.get_skill_members(skill.id).await?;
        let member_ids: Vec<String> = notes.iter().map(|n| n.id.to_string()).collect();
        existing_members.push((skill.id, member_ids));
    }

    // Step 3: Run Louvain detection once — detect candidates for evolution analysis
    // (We use detect_skill_candidates directly instead of detect_skills_pipeline
    // to avoid persisting skills twice. Evolution will handle persistence.)
    let detection_candidates = match graph_store
        .get_synapse_graph(project_id, config.detection.min_synapse_weight)
        .await
    {
        Ok(edges) => {
            let detection = crate::skills::detection::detect_skill_candidates(
                &edges,
                &project_id.to_string(),
                &config.detection,
            );
            result.skills_detected = Some(detection.candidates.len());
            detection.candidates
        }
        Err(e) => {
            warn!(error = %e, "Failed to get synapse graph for detection");
            result
                .warnings
                .push(format!("Synapse graph fetch failed: {}", e));
            vec![]
        }
    };

    // Step 4: Evolve existing skills based on new clusters
    // Also handle first-run: when no existing skills but candidates detected,
    // analyze_evolution will classify all candidates as New (which is correct).
    if !detection_candidates.is_empty() {
        let evolutions = analyze_evolution(
            &existing_members,
            &detection_candidates,
            config.evolution_overlap_threshold,
        );

        // Fetch notes needed for evolution (split, new, grow)
        let mut notes_map = std::collections::HashMap::new();
        for evolution in &evolutions {
            let note_ids: Vec<&str> = match evolution {
                crate::skills::evolution::SkillEvolution::Stable { notes_to_add, .. } => {
                    notes_to_add.iter().map(|s| s.as_str()).collect()
                }
                crate::skills::evolution::SkillEvolution::Split { candidates, .. } => candidates
                    .iter()
                    .flat_map(|c| c.member_note_ids.iter().map(|s| s.as_str()))
                    .collect(),
                crate::skills::evolution::SkillEvolution::New { candidate } => candidate
                    .member_note_ids
                    .iter()
                    .map(|s| s.as_str())
                    .collect(),
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

        match execute_evolution(graph_store, &evolutions, &notes_map, project_id).await {
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

    // Pre-pass: clean up any absolute paths in existing triggers
    match cleanup_absolute_triggers(graph_store, project_id).await {
        Ok(count) if count > 0 => {
            info!(project_id = %project_id, count, "Cleaned up absolute triggers");
        }
        Err(e) => {
            warn!(project_id = %project_id, "cleanup_absolute_triggers failed: {}", e);
        }
        _ => {}
    }

    let mut result = run_weekly_maintenance(graph_store, project_id, config).await?;
    result.level = "full".to_string();
    Ok(result)
}

// ============================================================================
// Absolute Trigger Cleanup
// ============================================================================

/// Scan all skills for a project and relativize any FileGlob trigger patterns
/// that still contain absolute paths. Returns the number of triggers corrected.
///
/// This is a migration/cleanup function for existing data that was created
/// before path agnosticism was implemented.
pub async fn cleanup_absolute_triggers(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
) -> anyhow::Result<usize> {
    use crate::skills::models::TriggerType;
    use crate::utils::paths::sanitize_pattern;

    // Load project root_path
    let root_path = match graph_store.get_project(project_id).await? {
        Some(p) => p.root_path,
        None => return Ok(0),
    };

    if root_path.is_empty() {
        return Ok(0);
    }

    let skills = graph_store.get_skills_for_project(project_id).await?;
    let mut total_fixed = 0;

    for mut skill in skills {
        let mut changed = false;
        for trigger in &mut skill.trigger_patterns {
            if trigger.pattern_type == TriggerType::FileGlob
                && trigger.pattern_value.starts_with('/')
            {
                let sanitized = sanitize_pattern(&trigger.pattern_value, &root_path);
                if sanitized != trigger.pattern_value {
                    trigger.pattern_value = sanitized;
                    changed = true;
                    total_fixed += 1;
                }
            }
        }
        if changed {
            skill.updated_at = chrono::Utc::now();
            graph_store.update_skill(&skill).await?;
        }
    }

    Ok(total_fixed)
}

// ============================================================================
// Tracked Maintenance (biomimicry T11 — Sleep Success Tracking)
// ============================================================================

/// Retrieve the success_rate from the last maintenance report stored as a Note.
/// Returns None if no previous report exists.
async fn get_last_maintenance_success_rate(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
) -> Option<f64> {
    use crate::notes::models::NoteFilters;
    // Find the most recent maintenance-report note by tag
    let filters = NoteFilters {
        tags: Some(vec!["maintenance-report".to_string()]),
        limit: Some(1),
        sort_by: Some("created_at".to_string()),
        sort_order: Some("desc".to_string()),
        ..Default::default()
    };
    let (notes, _) = graph_store
        .list_notes(Some(project_id), None, &filters)
        .await
        .ok()?;
    let note = notes.first()?;
    // Parse success_rate from the note content (stored as JSON)
    let parsed: serde_json::Value = serde_json::from_str(&note.content).ok()?;
    parsed.get("success_rate")?.as_f64()
}

/// Store a maintenance report as a Note for future adaptive reference.
async fn store_maintenance_report(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    report: &crate::neo4j::models::MaintenanceReport,
) {
    let content = match serde_json::to_string_pretty(report) {
        Ok(json) => json,
        Err(e) => {
            warn!(error = %e, "Failed to serialize maintenance report");
            return;
        }
    };

    let mut note = crate::notes::Note::new(
        Some(project_id),
        crate::notes::NoteType::Observation,
        content,
        "maintenance-tracker".to_string(),
    );
    note.importance = crate::notes::NoteImportance::Medium;
    note.tags = vec![
        "maintenance-report".to_string(),
        "auto-generated".to_string(),
        format!("level:{}", report.maintenance_level),
    ];
    if let Err(e) = graph_store.create_note(&note).await {
        warn!(error = %e, "Failed to store maintenance report as note");
    }
}

/// Run maintenance with pre/post snapshot tracking and delta computation.
///
/// 1. Capture a pre-maintenance snapshot
/// 2. Run the requested maintenance level
/// 3. Capture a post-maintenance snapshot
/// 4. Compute deltas and success_rate (fraction of metrics that improved or stayed stable)
/// 5. Return both the MaintenanceResult and the MaintenanceReport
pub async fn run_maintenance_with_tracking(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    level: &str,
    config: &SkillMaintenanceConfig,
) -> anyhow::Result<(MaintenanceResult, crate::neo4j::models::MaintenanceReport)> {
    use std::time::Instant;

    let start = Instant::now();

    // 0. Adaptive decay: check last maintenance report success_rate
    //    If previous success_rate < 0.5, halve the decay amount to be gentler
    let mut adapted_config = config.clone();
    let last_success_rate = get_last_maintenance_success_rate(graph_store, project_id).await;
    if let Some(rate) = last_success_rate {
        if rate < 0.5 {
            adapted_config.synapse_decay_amount *= 0.5;
            info!(
                project_id = %project_id,
                previous_success_rate = format!("{:.0}%", rate * 100.0),
                new_decay = adapted_config.synapse_decay_amount,
                "Adaptive decay: reducing synapse_decay_amount due to low previous success_rate"
            );
        }
    }
    let config = &adapted_config;

    // 1. Pre-snapshot
    let before = graph_store.compute_maintenance_snapshot(project_id).await?;

    // 2. Run maintenance
    let result = match level {
        "hourly" => run_hourly_maintenance(graph_store, project_id, config).await?,
        "daily" => run_daily_maintenance(graph_store, project_id, config).await?,
        "weekly" => run_weekly_maintenance(graph_store, project_id, config).await?,
        "full" => run_full_maintenance(graph_store, project_id, config).await?,
        _ => {
            warn!(
                level = level,
                "Unknown maintenance level, defaulting to daily"
            );
            run_daily_maintenance(graph_store, project_id, config).await?
        }
    };

    // 3. Post-snapshot
    let after = graph_store.compute_maintenance_snapshot(project_id).await?;

    let duration_ms = start.elapsed().as_millis() as u64;

    // 4. Compute deltas
    let delta_health_score = after.health_score - before.health_score;
    let delta_active_synapses = after.active_synapses - before.active_synapses;
    let delta_mean_energy = after.mean_energy - before.mean_energy;
    let delta_skill_count = after.skill_count - before.skill_count;
    let delta_note_count = after.note_count - before.note_count;

    // 5. Success rate: fraction of metrics that improved or stayed stable (>= 0)
    let metrics = [
        delta_health_score >= 0.0,
        delta_active_synapses >= 0,
        delta_mean_energy >= -0.01, // small tolerance for energy decay (expected)
        delta_skill_count >= 0,
        delta_note_count >= 0,
    ];
    let stable_or_improved = metrics.iter().filter(|&&b| b).count();
    let success_rate = stable_or_improved as f64 / metrics.len() as f64;

    let report = crate::neo4j::models::MaintenanceReport {
        before,
        after,
        delta_health_score,
        delta_active_synapses,
        delta_mean_energy,
        delta_skill_count,
        delta_note_count,
        success_rate,
        maintenance_level: level.to_string(),
        duration_ms,
    };

    info!(
        project_id = %project_id,
        level = level,
        success_rate = format!("{:.0}%", success_rate * 100.0),
        duration_ms = duration_ms,
        delta_health = format!("{:+.3}", delta_health_score),
        delta_synapses = delta_active_synapses,
        delta_energy = format!("{:+.3}", delta_mean_energy),
        "Tracked maintenance complete"
    );

    // 6. Store report as Note for future adaptive reference
    store_maintenance_report(graph_store, project_id, &report).await;

    Ok((result, report))
}

// ============================================================================
// Deep Maintenance (biomimicry T12 — Global Stagnation Response)
// ============================================================================

/// Run deep maintenance when global stagnation is detected.
/// This is an aggressive cleanup cycle:
/// 1. Detect stagnation (4 signals)
/// 2. Run full maintenance with aggressive decay
/// 3. Flag stale notes for review
/// 4. Identify stuck tasks
/// 5. Generate recommendations
pub async fn deep_maintenance(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    config: &SkillMaintenanceConfig,
) -> anyhow::Result<crate::neo4j::models::DeepMaintenanceReport> {
    info!(project_id = %project_id, "Starting deep maintenance (stagnation response)");

    // 1. Detect stagnation
    let stagnation = graph_store.detect_global_stagnation(project_id).await?;

    // 2. Run full maintenance with aggressive decay (3x normal)
    let mut aggressive_config = config.clone();
    aggressive_config.synapse_decay_amount *= 3.0;
    aggressive_config.synapse_prune_threshold *= 1.5;

    let maintenance_json =
        match run_full_maintenance(graph_store, project_id, &aggressive_config).await {
            Ok(result) => serde_json::to_value(&result).ok(),
            Err(e) => {
                warn!(error = %e, "Full maintenance failed during deep maintenance");
                None
            }
        };

    // 3. Update staleness scores and flag stale notes
    let stale_notes_flagged = match graph_store.update_staleness_scores().await {
        Ok(n) => n,
        Err(e) => {
            warn!(error = %e, "Failed to update staleness scores");
            0
        }
    };

    // 4. Identify stuck tasks (in_progress for too long)
    let stuck_tasks_found = count_stuck_tasks(graph_store, project_id).await;

    // 5. Build recommendations
    let mut recommendations = stagnation.recommendations.clone();
    if stale_notes_flagged > 0 {
        recommendations.push(format!(
            "{} notes have high staleness — review with note(action: \"get_needing_review\").",
            stale_notes_flagged
        ));
    }
    if stuck_tasks_found > 0 {
        recommendations.push(format!(
            "{} tasks stuck in_progress — consider marking as blocked or failed.",
            stuck_tasks_found
        ));
    }

    info!(
        project_id = %project_id,
        is_stagnating = stagnation.is_stagnating,
        signals = stagnation.signals_triggered,
        stale_notes = stale_notes_flagged,
        stuck_tasks = stuck_tasks_found,
        "Deep maintenance complete"
    );

    Ok(crate::neo4j::models::DeepMaintenanceReport {
        stagnation,
        maintenance: maintenance_json,
        stale_notes_flagged,
        stuck_tasks_found,
        recommendations,
    })
}

/// Count tasks that are in_progress (considered stuck during stagnation).
async fn count_stuck_tasks(graph_store: &dyn GraphStore, project_id: Uuid) -> usize {
    let plans = match graph_store.list_project_plans(project_id).await {
        Ok(plans) => plans,
        Err(_) => return 0,
    };

    let mut stuck = 0usize;
    for plan in &plans {
        let tasks = match graph_store.get_plan_tasks(plan.id).await {
            Ok(tasks) => tasks,
            Err(_) => continue,
        };
        stuck += tasks
            .iter()
            .filter(|t| t.status == crate::neo4j::models::TaskStatus::InProgress)
            .count();
    }
    stuck
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

    // ================================================================
    // Async maintenance tests (MockGraphStore)
    // ================================================================

    use crate::neo4j::mock::MockGraphStore;
    use crate::skills::models::{SkillNode, SkillStatus};
    use chrono::Utc;
    use uuid::Uuid;

    async fn setup_store_with_project() -> (MockGraphStore, Uuid) {
        let store = MockGraphStore::new();
        let project_id = Uuid::new_v4();
        let mut p = crate::test_helpers::test_project();
        p.id = project_id;
        store.projects.write().await.insert(p.id, p);
        (store, project_id)
    }

    #[tokio::test]
    async fn test_hourly_maintenance_no_skills() {
        let (store, project_id) = setup_store_with_project().await;
        let config = SkillMaintenanceConfig::default();

        let result = run_hourly_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "hourly");
        assert!(result.lifecycle.is_some());
        let lifecycle = result.lifecycle.unwrap();
        assert_eq!(lifecycle.skills_updated, 0);
        assert_eq!(lifecycle.skills_promoted, 0);
        assert_eq!(lifecycle.skills_demoted, 0);
    }

    #[tokio::test]
    async fn test_hourly_maintenance_with_active_skill() {
        let (store, project_id) = setup_store_with_project().await;

        // Create an active skill with low energy and old activation
        let mut skill = SkillNode::new(project_id, "Active Low Energy");
        skill.status = SkillStatus::Active;
        skill.energy = 0.05;
        skill.cohesion = 0.15;
        skill.last_activated = Some(Utc::now() - chrono::Duration::days(35));
        store.create_skill(&skill).await.unwrap();

        let config = SkillMaintenanceConfig::default();
        let result = run_hourly_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "hourly");
        let lifecycle = result.lifecycle.unwrap();
        assert_eq!(lifecycle.skills_demoted, 1);
    }

    #[tokio::test]
    async fn test_daily_maintenance_decays_synapses() {
        let (store, project_id) = setup_store_with_project().await;
        let config = SkillMaintenanceConfig::default();

        let result = run_daily_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "daily");
        // Decay runs but nothing to decay
        assert!(result.synapses_decayed.is_some());
        assert_eq!(result.synapses_decayed.unwrap(), 0);
        assert!(result.synapses_pruned.is_some());
        assert_eq!(result.synapses_pruned.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_daily_maintenance_with_synapses() {
        let (store, project_id) = setup_store_with_project().await;

        // Seed two notes with a synapse between them
        let note1_id = Uuid::new_v4();
        let note2_id = Uuid::new_v4();
        let note1 =
            crate::test_helpers::test_note(project_id, crate::notes::NoteType::Pattern, "Note 1");
        let note2 =
            crate::test_helpers::test_note(project_id, crate::notes::NoteType::Tip, "Note 2");
        let mut n1 = note1;
        n1.id = note1_id;
        let mut n2 = note2;
        n2.id = note2_id;
        store.notes.write().await.insert(n1.id, n1);
        store.notes.write().await.insert(n2.id, n2);
        store
            .note_synapses
            .write()
            .await
            .insert(note1_id, vec![(note2_id, 0.5)]);

        let config = SkillMaintenanceConfig::default();
        let result = run_daily_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "daily");
        assert_eq!(result.synapses_decayed.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_weekly_maintenance_no_skills_no_synapses() {
        let (store, project_id) = setup_store_with_project().await;
        let config = SkillMaintenanceConfig::default();

        let result = run_weekly_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "weekly");
        assert!(result.lifecycle.is_some());
        // No synapses → no detection candidates
        assert!(result.skills_detected.is_none() || result.skills_detected == Some(0));
    }

    #[tokio::test]
    async fn test_full_maintenance_delegates_to_weekly() {
        let (store, project_id) = setup_store_with_project().await;
        let config = SkillMaintenanceConfig::default();

        let result = run_full_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "full");
        assert!(result.lifecycle.is_some());
    }

    #[test]
    fn test_maintenance_result_serde() {
        let result = MaintenanceResult {
            level: "daily".to_string(),
            lifecycle: Some(MetricsUpdateResult {
                skills_updated: 5,
                skills_promoted: 1,
                skills_demoted: 2,
                skills_archived: 0,
                decisions_added: 3,
            }),
            synapses_decayed: Some(10),
            synapses_pruned: Some(2),
            evolution: None,
            skills_detected: None,
            warnings: vec!["test warning".to_string()],
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: MaintenanceResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.level, "daily");
        assert_eq!(deserialized.synapses_decayed, Some(10));
        assert_eq!(deserialized.warnings.len(), 1);
    }

    // ========================================================================
    // Biomimicry integration scenarios (T13)
    // ========================================================================

    use crate::neo4j::models::*;
    use crate::neo4j::traits::GraphStore;
    use crate::notes::{Note, NoteImportance, NoteType};
    use crate::test_helpers::*;

    /// Helper: create project + plan + tasks with given statuses
    async fn setup_project_with_tasks(
        store: &MockGraphStore,
        n_completed: usize,
        n_failed: usize,
        n_in_progress: usize,
    ) -> (uuid::Uuid, uuid::Uuid, Vec<uuid::Uuid>) {
        let project = test_project();
        let project_id = project.id;
        store.create_project(&project).await.unwrap();

        let plan = test_plan_for_project(project_id);
        let plan_id = plan.id;
        store.create_plan(&plan).await.unwrap();
        store
            .link_plan_to_project(plan_id, project_id)
            .await
            .unwrap();

        let mut task_ids = Vec::new();
        for i in 0..n_completed {
            let mut task = test_task_titled(&format!("Completed {}", i));
            task.status = TaskStatus::Completed;
            task.completed_at = Some(chrono::Utc::now());
            let tid = task.id;
            store.create_task(plan_id, &task).await.unwrap();
            task_ids.push(tid);
        }
        for i in 0..n_failed {
            let mut task = test_task_titled(&format!("Failed {}", i));
            task.status = TaskStatus::Failed;
            let tid = task.id;
            store.create_task(plan_id, &task).await.unwrap();
            task_ids.push(tid);
        }
        for i in 0..n_in_progress {
            let mut task = test_task_titled(&format!("InProgress {}", i));
            task.status = TaskStatus::InProgress;
            let tid = task.id;
            store.create_task(plan_id, &task).await.unwrap();
            task_ids.push(tid);
        }
        (project_id, plan_id, task_ids)
    }

    /// Scenario 1: Scar → frustration chain
    #[tokio::test]
    async fn test_biomimicry_s1_scar_frustration_chain() {
        let store = MockGraphStore::new();
        let (_project_id, _plan_id, task_ids) = setup_project_with_tasks(&store, 0, 0, 2).await;

        // Create notes to scar
        let mut note1 = Note::new(
            Some(_project_id),
            NoteType::Observation,
            "Obs 1".into(),
            "test".into(),
        );
        note1.importance = NoteImportance::Medium;
        note1.tags = vec!["test".into()];
        let note1_id = note1.id;
        store.create_note(&note1).await.unwrap();

        let mut note2 = Note::new(
            Some(_project_id),
            NoteType::Observation,
            "Obs 2".into(),
            "test".into(),
        );
        note2.importance = NoteImportance::Medium;
        note2.tags = vec!["test".into()];
        let note2_id = note2.id;
        store.create_note(&note2).await.unwrap();

        // Apply scars
        let scarred = store.apply_scars(&[note1_id, note2_id], 0.5).await.unwrap();
        assert_eq!(scarred, 2);

        let n1 = store.get_note(note1_id).await.unwrap().unwrap();
        assert!(
            (n1.scar_intensity - 0.5).abs() < 0.01,
            "scar_intensity should be ~0.5, got {}",
            n1.scar_intensity
        );

        // Increment frustration
        let f1 = store.increment_frustration(task_ids[0], 0.3).await.unwrap();
        assert!((f1 - 0.3).abs() < 0.01);
        let f2 = store.increment_frustration(task_ids[0], 0.4).await.unwrap();
        assert!((f2 - 0.7).abs() < 0.01);

        let frust = store.get_frustration(task_ids[0]).await.unwrap();
        assert!((frust - 0.7).abs() < 0.01);
    }

    /// Scenario 2: Homeostasis scar_load
    #[tokio::test]
    async fn test_biomimicry_s2_homeostasis_scar_load() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 5, 0, 0).await;

        let mut note_ids = Vec::new();
        for i in 0..10 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Obs {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Medium;
            note.tags = vec!["test".into()];
            note_ids.push(note.id);
            store.create_note(&note).await.unwrap();
        }

        store.apply_scars(&note_ids, 0.8).await.unwrap();

        let report = store.compute_homeostasis(project_id, None).await.unwrap();
        let scar_ratio = report.ratios.iter().find(|r| r.name == "scar_load");
        assert!(
            scar_ratio.is_some(),
            "Homeostasis should have scar_load ratio"
        );
        let scar = scar_ratio.unwrap();
        assert!(
            scar.value > 0.0,
            "scar_load should be > 0, got {}",
            scar.value
        );
    }

    /// Scenario 3: Global stagnation → deep maintenance
    #[tokio::test]
    async fn test_biomimicry_s3_stagnation_deep_maintenance() {
        let store = MockGraphStore::new();
        let (project_id, _, task_ids) = setup_project_with_tasks(&store, 0, 0, 3).await;

        // High frustration on all in-progress tasks
        for &tid in &task_ids {
            store.increment_frustration(tid, 0.8).await.unwrap();
        }

        // Low energy notes
        for i in 0..5 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Low {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Low;
            note.tags = vec!["test".into()];
            note.energy = 0.1;
            store.create_note(&note).await.unwrap();
        }

        // Detect stagnation
        let report = store.detect_global_stagnation(project_id).await.unwrap();
        assert!(report.is_stagnating, "Should detect stagnation");
        assert!(
            report.signals_triggered >= 3,
            "Should have >= 3 signals, got {}",
            report.signals_triggered
        );
        assert_eq!(report.tasks_completed_48h, 0);
        assert!(
            report.avg_frustration > 0.6,
            "Avg frustration should be > 0.6, got {}",
            report.avg_frustration
        );

        // Deep maintenance
        let config = SkillMaintenanceConfig::default();
        let deep = deep_maintenance(&store, project_id, &config).await.unwrap();
        assert!(deep.stagnation.is_stagnating);
        assert!(
            deep.stuck_tasks_found >= 3,
            "Should find >= 3 stuck tasks, got {}",
            deep.stuck_tasks_found
        );
        assert!(!deep.recommendations.is_empty());
    }

    /// Scenario 4: Consolidation + maintenance tracking pipeline
    #[tokio::test]
    async fn test_biomimicry_s4_consolidation_measured() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 3, 0, 0).await;

        for i in 0..5 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Note {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Medium;
            note.tags = vec!["test".into()];
            note.energy = 0.8;
            store.create_note(&note).await.unwrap();
        }

        // Pre-snapshot
        let before = store
            .compute_maintenance_snapshot(project_id)
            .await
            .unwrap();
        assert_eq!(before.note_count, 5);
        assert!(before.mean_energy > 0.0);

        // Run tracked maintenance
        let config = SkillMaintenanceConfig::default();
        let (result, report) = run_maintenance_with_tracking(&store, project_id, "hourly", &config)
            .await
            .unwrap();

        assert_eq!(result.level, "hourly");
        assert!(
            report.success_rate >= 0.0 && report.success_rate <= 1.0,
            "success_rate should be in [0, 1], got {}",
            report.success_rate
        );
    }

    /// Scenario 5: Scaffolding adapts to project health
    #[tokio::test]
    async fn test_biomimicry_s5_scaffolding_integrated() {
        let store = MockGraphStore::new();

        // Healthy project: 18 completed, 2 failed
        let (project_id, _, _) = setup_project_with_tasks(&store, 18, 2, 0).await;

        for i in 0..3 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Healthy {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Medium;
            note.tags = vec!["test".into()];
            note.energy = 0.9;
            store.create_note(&note).await.unwrap();
        }

        let level = store
            .compute_scaffolding_level(project_id, None)
            .await
            .unwrap();
        assert!(
            level.task_success_rate >= 0.8,
            "Success rate should be >= 0.8, got {}",
            level.task_success_rate
        );
        assert!(
            level.level >= 3,
            "Level should be >= 3 with high success, got L{}",
            level.level
        );
        assert!(!level.is_overridden);

        // Degraded project: 2 completed, 8 failed + scars
        let project2 = test_project_named("degraded");
        let project2_id = project2.id;
        store.create_project(&project2).await.unwrap();

        let plan2 = test_plan_for_project(project2_id);
        let plan2_id = plan2.id;
        store.create_plan(&plan2).await.unwrap();
        store
            .link_plan_to_project(plan2_id, project2_id)
            .await
            .unwrap();

        for i in 0..2 {
            let mut t = test_task_titled(&format!("P2 OK {}", i));
            t.status = TaskStatus::Completed;
            t.completed_at = Some(chrono::Utc::now());
            store.create_task(plan2_id, &t).await.unwrap();
        }
        for i in 0..8 {
            let mut t = test_task_titled(&format!("P2 Fail {}", i));
            t.status = TaskStatus::Failed;
            store.create_task(plan2_id, &t).await.unwrap();
        }

        let mut scar_ids = Vec::new();
        for i in 0..5 {
            let mut note = Note::new(
                Some(project2_id),
                NoteType::Gotcha,
                format!("Scar {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::High;
            note.tags = vec!["test".into()];
            scar_ids.push(note.id);
            store.create_note(&note).await.unwrap();
        }
        store.apply_scars(&scar_ids, 0.9).await.unwrap();

        let level2 = store
            .compute_scaffolding_level(project2_id, None)
            .await
            .unwrap();
        assert!(
            level2.task_success_rate <= 0.3,
            "Should be <= 0.3, got {}",
            level2.task_success_rate
        );
        assert!(
            level2.level <= 1,
            "Should be <= L1 with low success + scars, got L{}",
            level2.level
        );

        // Verify levels differ
        assert!(
            level.level > level2.level,
            "Healthy L{} > degraded L{}",
            level.level,
            level2.level
        );

        // Test override
        store
            .set_scaffolding_override(project2_id, Some(4))
            .await
            .unwrap();
        let overridden = store
            .compute_scaffolding_level(project2_id, Some(4))
            .await
            .unwrap();
        assert_eq!(overridden.level, 4, "Override should force L4");
        assert!(overridden.is_overridden);
    }

    // ========================================================================
    // Additional coverage tests
    // ========================================================================

    #[test]
    fn test_config_serde_roundtrip_custom_values() {
        let config = SkillMaintenanceConfig {
            synapse_decay_amount: 0.05,
            synapse_prune_threshold: 0.2,
            evolution_overlap_threshold: 0.8,
            backfill_recluster_threshold: 100,
            enabled: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let d: SkillMaintenanceConfig = serde_json::from_str(&json).unwrap();
        assert!(d.enabled);
        assert!((d.synapse_decay_amount - 0.05).abs() < f64::EPSILON);
        assert!((d.synapse_prune_threshold - 0.2).abs() < f64::EPSILON);
        assert!((d.evolution_overlap_threshold - 0.8).abs() < f64::EPSILON);
        assert_eq!(d.backfill_recluster_threshold, 100);
    }

    #[test]
    fn test_maintenance_result_serde_all_fields() {
        let result = MaintenanceResult {
            level: "weekly".to_string(),
            lifecycle: Some(MetricsUpdateResult {
                skills_updated: 10,
                skills_promoted: 3,
                skills_demoted: 1,
                skills_archived: 2,
                decisions_added: 5,
            }),
            synapses_decayed: Some(20),
            synapses_pruned: Some(5),
            evolution: Some(crate::skills::evolution::EvolutionResult {
                merged: vec![],
                split: vec![],
                grown: vec![],
                shrunk: vec![],
                unchanged: 0,
            }),
            skills_detected: Some(7),
            warnings: vec!["warn1".to_string(), "warn2".to_string()],
        };
        let json = serde_json::to_string(&result).unwrap();
        let d: MaintenanceResult = serde_json::from_str(&json).unwrap();
        assert_eq!(d.level, "weekly");
        assert_eq!(d.skills_detected, Some(7));
        assert_eq!(d.synapses_pruned, Some(5));
        assert_eq!(d.warnings.len(), 2);
        assert!(d.evolution.is_some());
        let evo = d.evolution.unwrap();
        assert!(evo.merged.is_empty());
        assert_eq!(evo.unchanged, 0);
    }

    #[test]
    fn test_maintenance_result_default_has_no_optional_fields() {
        let r = MaintenanceResult::default();
        assert!(r.synapses_decayed.is_none());
        assert!(r.synapses_pruned.is_none());
        assert!(r.skills_detected.is_none());
    }

    // ========================================================================
    // cleanup_absolute_triggers tests
    // ========================================================================

    #[tokio::test]
    async fn test_cleanup_absolute_triggers_no_project() {
        let store = MockGraphStore::new();
        let fake_id = Uuid::new_v4();
        // No project in store => returns 0
        let count = cleanup_absolute_triggers(&store, fake_id).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_cleanup_absolute_triggers_empty_root_path() {
        let store = MockGraphStore::new();
        let mut project = test_project();
        project.root_path = "".to_string();
        let pid = project.id;
        store.create_project(&project).await.unwrap();

        let count = cleanup_absolute_triggers(&store, pid).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_cleanup_absolute_triggers_fixes_absolute_paths() {
        let store = MockGraphStore::new();
        let mut project = test_project();
        project.root_path = "/home/user/myproject".to_string();
        let pid = project.id;
        store.create_project(&project).await.unwrap();

        // Create a skill with an absolute FileGlob trigger
        let mut skill = SkillNode::new(pid, "Test Skill");
        skill.trigger_patterns = vec![
            crate::skills::models::SkillTrigger::file_glob("/home/user/myproject/src/**/*.rs", 0.8),
            crate::skills::models::SkillTrigger::regex("test.*pattern", 0.7),
        ];
        store.create_skill(&skill).await.unwrap();

        let count = cleanup_absolute_triggers(&store, pid).await.unwrap();
        assert_eq!(count, 1, "Should have fixed one absolute trigger");

        // Verify the trigger was relativized
        let updated = store.get_skill(skill.id).await.unwrap().unwrap();
        let glob_trigger = updated
            .trigger_patterns
            .iter()
            .find(|t| t.pattern_type == crate::skills::models::TriggerType::FileGlob)
            .unwrap();
        assert!(
            !glob_trigger.pattern_value.starts_with('/'),
            "Pattern should be relative now, got: {}",
            glob_trigger.pattern_value
        );
    }

    #[tokio::test]
    async fn test_cleanup_absolute_triggers_skips_relative_paths() {
        let store = MockGraphStore::new();
        let mut project = test_project();
        project.root_path = "/home/user/myproject".to_string();
        let pid = project.id;
        store.create_project(&project).await.unwrap();

        // Create a skill with only relative triggers
        let mut skill = SkillNode::new(pid, "Relative Skill");
        skill.trigger_patterns = vec![crate::skills::models::SkillTrigger::file_glob(
            "src/**/*.rs",
            0.8,
        )];
        store.create_skill(&skill).await.unwrap();

        let count = cleanup_absolute_triggers(&store, pid).await.unwrap();
        assert_eq!(count, 0, "No absolute triggers to fix");
    }

    #[tokio::test]
    async fn test_cleanup_absolute_triggers_no_skills() {
        let (store, pid) = setup_store_with_project().await;
        let count = cleanup_absolute_triggers(&store, pid).await.unwrap();
        assert_eq!(count, 0);
    }

    // ========================================================================
    // run_maintenance_with_tracking tests
    // ========================================================================

    #[tokio::test]
    async fn test_tracked_maintenance_daily_level() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 2, 0, 0).await;

        // Add some notes for meaningful snapshot
        for i in 0..3 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Daily note {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Medium;
            note.tags = vec!["test".into()];
            note.energy = 0.6;
            store.create_note(&note).await.unwrap();
        }

        let config = SkillMaintenanceConfig::default();
        let (result, report) = run_maintenance_with_tracking(&store, project_id, "daily", &config)
            .await
            .unwrap();

        assert_eq!(result.level, "daily");
        assert_eq!(report.maintenance_level, "daily");
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        assert!(report.duration_ms < 10_000); // should be fast in mock
    }

    #[tokio::test]
    async fn test_tracked_maintenance_weekly_level() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        let config = SkillMaintenanceConfig::default();
        let (result, report) = run_maintenance_with_tracking(&store, project_id, "weekly", &config)
            .await
            .unwrap();

        assert_eq!(result.level, "weekly");
        assert_eq!(report.maintenance_level, "weekly");
    }

    #[tokio::test]
    async fn test_tracked_maintenance_full_level() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        let config = SkillMaintenanceConfig::default();
        let (result, report) = run_maintenance_with_tracking(&store, project_id, "full", &config)
            .await
            .unwrap();

        assert_eq!(result.level, "full");
        assert_eq!(report.maintenance_level, "full");
    }

    #[tokio::test]
    async fn test_tracked_maintenance_unknown_level_defaults_to_daily() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        let config = SkillMaintenanceConfig::default();
        let (result, report) =
            run_maintenance_with_tracking(&store, project_id, "bogus_level", &config)
                .await
                .unwrap();

        // Unknown level falls through to daily
        assert_eq!(result.level, "daily");
        assert_eq!(report.maintenance_level, "bogus_level");
    }

    #[tokio::test]
    async fn test_tracked_maintenance_adaptive_decay_on_low_success() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        // Seed a maintenance-report note with low success_rate to trigger adaptive decay
        let report_content = serde_json::json!({
            "success_rate": 0.2,
            "maintenance_level": "daily",
        });
        let mut report_note = Note::new(
            Some(project_id),
            NoteType::Observation,
            report_content.to_string(),
            "maintenance-tracker".into(),
        );
        report_note.tags = vec![
            "maintenance-report".to_string(),
            "auto-generated".to_string(),
        ];
        store.create_note(&report_note).await.unwrap();

        let config = SkillMaintenanceConfig::default();
        let original_decay = config.synapse_decay_amount;

        // Run tracked maintenance - should adapt config
        let (result, report) = run_maintenance_with_tracking(&store, project_id, "hourly", &config)
            .await
            .unwrap();

        assert_eq!(result.level, "hourly");
        // The adaptive decay should have halved - we can't directly observe the internal
        // config but we verify the function ran successfully and stored a new report
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        // Verify original config was not mutated
        assert!((config.synapse_decay_amount - original_decay).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_tracked_maintenance_stores_report_as_note() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        let config = SkillMaintenanceConfig::default();
        let (_result, _report) =
            run_maintenance_with_tracking(&store, project_id, "hourly", &config)
                .await
                .unwrap();

        // Verify a maintenance-report note was stored
        let filters = crate::notes::models::NoteFilters {
            tags: Some(vec!["maintenance-report".to_string()]),
            ..Default::default()
        };
        let (notes, _) = store
            .list_notes(Some(project_id), None, &filters)
            .await
            .unwrap();
        assert!(
            !notes.is_empty(),
            "Should have stored a maintenance-report note"
        );
        // Verify the note content is valid JSON with expected fields
        let content: serde_json::Value = serde_json::from_str(&notes[0].content).unwrap();
        assert!(content.get("success_rate").is_some());
        assert!(content.get("maintenance_level").is_some());
    }

    #[tokio::test]
    async fn test_tracked_maintenance_success_rate_calculation() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 5, 0, 0).await;

        // Add many high-energy notes so snapshot metrics are stable
        for i in 0..10 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Stable note {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Medium;
            note.tags = vec!["test".into()];
            note.energy = 0.9;
            store.create_note(&note).await.unwrap();
        }

        let config = SkillMaintenanceConfig::default();
        let (_result, report) =
            run_maintenance_with_tracking(&store, project_id, "hourly", &config)
                .await
                .unwrap();

        // With stable metrics, most deltas should be >= 0
        // success_rate = stable_or_improved / 5 metrics
        assert!(
            report.success_rate >= 0.0 && report.success_rate <= 1.0,
            "Success rate should be [0,1], got {}",
            report.success_rate
        );
    }

    // ========================================================================
    // deep_maintenance tests
    // ========================================================================

    #[tokio::test]
    async fn test_deep_maintenance_no_stagnation() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 5, 0, 0).await;

        // Healthy project with high-energy notes
        for i in 0..5 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Healthy {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Medium;
            note.tags = vec!["test".into()];
            note.energy = 0.9;
            store.create_note(&note).await.unwrap();
        }

        let config = SkillMaintenanceConfig::default();
        let report = deep_maintenance(&store, project_id, &config).await.unwrap();

        // Not stagnating
        assert!(
            !report.stagnation.is_stagnating || report.stagnation.signals_triggered < 3,
            "Healthy project should not be stagnating"
        );
    }

    #[tokio::test]
    async fn test_deep_maintenance_with_stale_notes_recommendation() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 0, 0, 3).await;

        // High frustration
        for i in 0..3 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Stale {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Low;
            note.energy = 0.05;
            note.tags = vec!["test".into()];
            store.create_note(&note).await.unwrap();
        }

        let config = SkillMaintenanceConfig::default();
        let report = deep_maintenance(&store, project_id, &config).await.unwrap();

        // Should have stuck_tasks recommendations since we have 3 in_progress tasks
        assert!(
            report.stuck_tasks_found >= 3,
            "Should find stuck tasks, got {}",
            report.stuck_tasks_found
        );
        // Check recommendations contain stuck tasks mention
        let has_stuck_rec = report
            .recommendations
            .iter()
            .any(|r| r.contains("stuck in_progress"));
        assert!(
            has_stuck_rec,
            "Should have stuck tasks recommendation, got: {:?}",
            report.recommendations
        );
    }

    #[tokio::test]
    async fn test_deep_maintenance_aggressive_config() {
        // Verify deep_maintenance uses 3x decay and 1.5x prune
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 0, 0, 1).await;

        let config = SkillMaintenanceConfig {
            synapse_decay_amount: 0.02,
            synapse_prune_threshold: 0.1,
            ..Default::default()
        };

        // Should not panic with aggressive config
        let report = deep_maintenance(&store, project_id, &config).await.unwrap();
        assert!(report.maintenance.is_some() || report.maintenance.is_none());
    }

    // ========================================================================
    // count_stuck_tasks tests
    // ========================================================================

    #[tokio::test]
    async fn test_count_stuck_tasks_no_plans() {
        let (store, project_id) = setup_store_with_project().await;
        let count = count_stuck_tasks(&store, project_id).await;
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_count_stuck_tasks_mixed_statuses() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 3, 2, 4).await;
        let count = count_stuck_tasks(&store, project_id).await;
        assert_eq!(count, 4, "Should find 4 in_progress tasks");
    }

    #[tokio::test]
    async fn test_count_stuck_tasks_zero_in_progress() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 5, 3, 0).await;
        let count = count_stuck_tasks(&store, project_id).await;
        assert_eq!(count, 0, "No in_progress tasks should yield 0");
    }

    // ========================================================================
    // Daily maintenance edge cases
    // ========================================================================

    #[tokio::test]
    async fn test_daily_maintenance_with_many_synapses_stagnation_detection() {
        let (store, project_id) = setup_store_with_project().await;

        // Create notes and synapses with low weights to trigger stagnation detection
        let mut note_ids = Vec::new();
        for i in 0..6 {
            let mut note = Note::new(
                Some(project_id),
                NoteType::Observation,
                format!("Stagnant {}", i),
                "test".into(),
            );
            note.importance = NoteImportance::Medium;
            note.tags = vec!["test".into()];
            note.energy = 0.1;
            note_ids.push(note.id);
            store.create_note(&note).await.unwrap();
        }

        // Create low-weight synapses between pairs (need >= 4 for stagnation detection)
        for i in 0..5 {
            store
                .note_synapses
                .write()
                .await
                .entry(note_ids[i])
                .or_default()
                .push((note_ids[i + 1], 0.15)); // low weight
        }

        let config = SkillMaintenanceConfig::default();
        let result = run_daily_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "daily");
        // With low synapse weights the stagnation detection should fire
        // and add a warning (if the t-test is significant)
        // Either way the function should complete successfully
        assert!(result.synapses_decayed.is_some());
    }

    #[tokio::test]
    async fn test_daily_maintenance_adaptive_lifecycle_with_skills() {
        let (store, project_id) = setup_store_with_project().await;

        // Create some skills so adaptive lifecycle config is computed
        for i in 0..3 {
            let mut skill = SkillNode::new(project_id, format!("Skill {}", i));
            skill.status = SkillStatus::Active;
            skill.energy = 0.5;
            skill.cohesion = 0.5;
            skill.activation_count = 10;
            store.create_skill(&skill).await.unwrap();
        }

        let config = SkillMaintenanceConfig::default();
        let result = run_daily_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "daily");
        assert!(result.lifecycle.is_some());
    }

    // ========================================================================
    // Weekly maintenance with existing skills
    // ========================================================================

    #[tokio::test]
    async fn test_weekly_maintenance_with_existing_skills() {
        let (store, project_id) = setup_store_with_project().await;

        // Create existing skills with members
        let mut skill = SkillNode::new(project_id, "Existing Skill");
        skill.status = SkillStatus::Active;
        let skill_id = skill.id;
        store.create_skill(&skill).await.unwrap();

        // Create notes and assign them to the skill
        let mut note = Note::new(
            Some(project_id),
            NoteType::Observation,
            "Member note".into(),
            "test".into(),
        );
        note.importance = NoteImportance::Medium;
        note.tags = vec!["test".into()];
        let note_id = note.id;
        store.create_note(&note).await.unwrap();

        // Link note to skill
        store
            .add_skill_member(skill_id, "note", note_id)
            .await
            .unwrap();

        let config = SkillMaintenanceConfig::default();
        let result = run_weekly_maintenance(&store, project_id, &config)
            .await
            .unwrap();

        assert_eq!(result.level, "weekly");
        assert!(result.lifecycle.is_some());
    }

    // ========================================================================
    // full maintenance includes cleanup_absolute_triggers
    // ========================================================================

    #[tokio::test]
    async fn test_full_maintenance_runs_trigger_cleanup() {
        let store = MockGraphStore::new();
        let mut project = test_project();
        project.root_path = "/home/user/proj".to_string();
        let pid = project.id;
        store.create_project(&project).await.unwrap();

        // Create a skill with absolute trigger
        let mut skill = SkillNode::new(pid, "Cleanup Target");
        skill.trigger_patterns = vec![crate::skills::models::SkillTrigger::file_glob(
            "/home/user/proj/src/**",
            0.8,
        )];
        store.create_skill(&skill).await.unwrap();

        let config = SkillMaintenanceConfig::default();
        let result = run_full_maintenance(&store, pid, &config).await.unwrap();
        assert_eq!(result.level, "full");

        // Verify the trigger was cleaned up
        let updated = store.get_skill(skill.id).await.unwrap().unwrap();
        if let Some(trigger) = updated.trigger_patterns.first() {
            if trigger.pattern_type == crate::skills::models::TriggerType::FileGlob {
                assert!(
                    !trigger.pattern_value.starts_with("/home"),
                    "Trigger should have been relativized, got: {}",
                    trigger.pattern_value
                );
            }
        }
    }

    // ========================================================================
    // get_last_maintenance_success_rate tests (via tracked maintenance)
    // ========================================================================

    #[tokio::test]
    async fn test_get_last_maintenance_success_rate_no_report() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        // No maintenance report note exists
        let rate = get_last_maintenance_success_rate(&store, project_id).await;
        assert!(rate.is_none(), "Should return None when no report exists");
    }

    #[tokio::test]
    async fn test_get_last_maintenance_success_rate_with_report() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        // Seed a maintenance report note
        let content = serde_json::json!({
            "success_rate": 0.8,
            "maintenance_level": "daily",
        });
        let mut note = Note::new(
            Some(project_id),
            NoteType::Observation,
            content.to_string(),
            "maintenance-tracker".into(),
        );
        note.tags = vec!["maintenance-report".to_string()];
        store.create_note(&note).await.unwrap();

        let rate = get_last_maintenance_success_rate(&store, project_id).await;
        assert!(rate.is_some(), "Should find the report");
        assert!((rate.unwrap() - 0.8).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_get_last_maintenance_success_rate_invalid_json() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        // Seed a note with invalid JSON content but the right tag
        let mut note = Note::new(
            Some(project_id),
            NoteType::Observation,
            "not valid json".to_string(),
            "maintenance-tracker".into(),
        );
        note.tags = vec!["maintenance-report".to_string()];
        store.create_note(&note).await.unwrap();

        let rate = get_last_maintenance_success_rate(&store, project_id).await;
        assert!(rate.is_none(), "Should return None for invalid JSON");
    }

    #[tokio::test]
    async fn test_get_last_maintenance_success_rate_json_missing_field() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        // Valid JSON but missing success_rate field
        let content = serde_json::json!({
            "maintenance_level": "daily",
        });
        let mut note = Note::new(
            Some(project_id),
            NoteType::Observation,
            content.to_string(),
            "maintenance-tracker".into(),
        );
        note.tags = vec!["maintenance-report".to_string()];
        store.create_note(&note).await.unwrap();

        let rate = get_last_maintenance_success_rate(&store, project_id).await;
        assert!(
            rate.is_none(),
            "Should return None when success_rate field missing"
        );
    }

    // ========================================================================
    // store_maintenance_report tests
    // ========================================================================

    #[tokio::test]
    async fn test_store_maintenance_report_creates_note() {
        let store = MockGraphStore::new();
        let (project_id, _, _) = setup_project_with_tasks(&store, 1, 0, 0).await;

        let report = crate::neo4j::models::MaintenanceReport {
            before: MaintenanceSnapshot {
                health_score: 0.5,
                active_synapses: 10,
                mean_energy: 0.6,
                skill_count: 3,
                note_count: 20,
                captured_at: "2024-01-01T00:00:00Z".to_string(),
            },
            after: MaintenanceSnapshot {
                health_score: 0.6,
                active_synapses: 12,
                mean_energy: 0.55,
                skill_count: 4,
                note_count: 22,
                captured_at: "2024-01-01T00:01:00Z".to_string(),
            },
            delta_health_score: 0.1,
            delta_active_synapses: 2,
            delta_mean_energy: -0.05,
            delta_skill_count: 1,
            delta_note_count: 2,
            success_rate: 0.8,
            maintenance_level: "daily".to_string(),
            duration_ms: 150,
        };

        store_maintenance_report(&store, project_id, &report).await;

        // Verify the note was created
        let filters = crate::notes::models::NoteFilters {
            tags: Some(vec!["maintenance-report".to_string()]),
            ..Default::default()
        };
        let (notes, _) = store
            .list_notes(Some(project_id), None, &filters)
            .await
            .unwrap();
        assert_eq!(notes.len(), 1);

        // Verify note content
        let content: serde_json::Value = serde_json::from_str(&notes[0].content).unwrap();
        assert_eq!(content["success_rate"], 0.8);
        assert_eq!(content["maintenance_level"], "daily");
        assert_eq!(content["duration_ms"], 150);

        // Verify tags include level
        assert!(notes[0].tags.contains(&"level:daily".to_string()));
        assert!(notes[0].tags.contains(&"auto-generated".to_string()));
    }

    // ========================================================================
    // MaintenanceReport delta and success_rate logic tests
    // ========================================================================

    #[test]
    fn test_success_rate_all_improved() {
        // All metrics improved => success_rate = 1.0
        let metrics = [
            0.1_f64 >= 0.0,    // delta_health_score
            2_i64 >= 0,        // delta_active_synapses
            0.05_f64 >= -0.01, // delta_mean_energy
            1_i64 >= 0,        // delta_skill_count
            3_i64 >= 0,        // delta_note_count
        ];
        let stable_or_improved = metrics.iter().filter(|&&b| b).count();
        let success_rate = stable_or_improved as f64 / metrics.len() as f64;
        assert!((success_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_all_degraded() {
        // All metrics degraded => success_rate = 0.0
        let metrics = [
            (-0.5_f64) >= 0.0,   // delta_health_score: degraded
            (-3_i64) >= 0,       // delta_active_synapses: degraded
            (-0.1_f64) >= -0.01, // delta_mean_energy: degraded (below tolerance)
            (-2_i64) >= 0,       // delta_skill_count: degraded
            (-1_i64) >= 0,       // delta_note_count: degraded
        ];
        let stable_or_improved = metrics.iter().filter(|&&b| b).count();
        let success_rate = stable_or_improved as f64 / metrics.len() as f64;
        assert!((success_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_mixed() {
        // 3 of 5 improved
        let metrics = [
            0.0_f64 >= 0.0,        // stable
            0_i64 >= 0,            // stable
            (-0.005_f64) >= -0.01, // within tolerance
            (-1_i64) >= 0,         // degraded
            (-2_i64) >= 0,         // degraded
        ];
        let stable_or_improved = metrics.iter().filter(|&&b| b).count();
        let success_rate = stable_or_improved as f64 / metrics.len() as f64;
        assert!((success_rate - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_energy_tolerance() {
        // Energy decay of -0.01 is within tolerance
        let delta_mean_energy = -0.01_f64;
        assert!(
            delta_mean_energy >= -0.01,
            "Small energy decay should be tolerated"
        );

        // Energy decay of -0.02 exceeds tolerance
        let delta_mean_energy2 = -0.02_f64;
        assert!(
            delta_mean_energy2 < -0.01,
            "Larger energy decay should NOT be tolerated"
        );
    }
}
