//! HomeostasisCheck — auto-corrective thermostat for the knowledge graph.
//!
//! Runs every 2 hours. For each project:
//! 1. Computes homeostasis report via `compute_homeostasis`
//! 2. **Dormancy guard**: if `pain_score < 0.1`, the system is stable — skip
//!    corrections to avoid eroding a dormant graph with no new input.
//! 3. Feeds metrics to `HomeostasisController::evaluate`
//! 4. Executes corrective actions via `execute_actions` (paginated backfill)
//!
//! Note: dead notes archival is handled by `ConsolidationCheck` (separate check).
//! This check focuses on synapse health and note density only.
//!
//! Backfill pagination: instead of trying to process all notes in one cycle
//! (which gets killed by the engine timeout), each cycle processes one batch.
//! A `BackfillCursor` is persisted in-memory between cycles to resume where
//! we left off. When all notes are processed, the cursor resets.
//!
//! Creates alerts when corrective actions are taken.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::homeostasis::{
    compute_backfill_params, execute_actions, BackfillCursor, ExecuteContext,
    HomeostasisController, HomeostasisMetrics,
};

/// Pain score below which we consider the system dormant/stable.
/// No corrective actions are taken — only observation.
const DORMANCY_THRESHOLD: f64 = 0.1;

/// Timeout override for homeostasis check.
/// Backfill + decay can take longer than the default 5s engine timeout.
const HOMEOSTASIS_TIMEOUT: Duration = Duration::from_secs(30);

/// Run homeostasis evaluation and correction on all projects (every 2 hours).
///
/// Includes a dormancy guard: if `pain_score < 0.1`, the project is considered
/// stable and no corrective actions are executed. This prevents erosion of a
/// quiet graph (e.g., synapse decay with no new input to compensate).
///
/// Backfill is paginated: each cycle processes one batch of notes, persisting
/// a cursor in memory. The cursor resets when all notes have been processed
/// or when the system becomes dormant.
pub struct HomeostasisCheck {
    /// Backfill cursor persisted across cycles (in-memory only).
    /// Uses a Mutex because HeartbeatCheck::run takes &self.
    cursor: Mutex<BackfillCursor>,
}

impl HomeostasisCheck {
    pub fn new() -> Self {
        Self {
            cursor: Mutex::new(BackfillCursor::new()),
        }
    }
}

impl Default for HomeostasisCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HeartbeatCheck for HomeostasisCheck {
    fn name(&self) -> &str {
        "homeostasis"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(2 * 60 * 60) // 2 hours
    }

    fn timeout_override(&self) -> Option<Duration> {
        Some(HOMEOSTASIS_TIMEOUT)
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let projects = ctx.graph.list_projects().await?;

        for project in &projects {
            debug!("HomeostasisCheck: evaluating project '{}'", project.name);

            // 1. Compute homeostasis report from the graph
            let report = match ctx.graph.compute_homeostasis(project.id, None).await {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        "HomeostasisCheck: compute_homeostasis failed for '{}': {}",
                        project.name, e
                    );
                    continue;
                }
            };

            // 2. Dormancy guard — stable systems are left alone.
            //    A dormant graph with pain < 0.1 should not receive corrective
            //    actions (decay, archive) that would erode it without new input.
            //    Also reset the backfill cursor — no point resuming stale state.
            if report.pain_score < DORMANCY_THRESHOLD {
                debug!(
                    pain_score = report.pain_score,
                    threshold = DORMANCY_THRESHOLD,
                    "HomeostasisCheck: project '{}' is stable (pain < threshold), skipping corrections",
                    project.name
                );
                // Reset cursor when dormant — stale offset is meaningless
                let mut cursor = self.cursor.lock().await;
                cursor.reset();
                continue;
            }

            // 3. Map report ratios to HomeostasisMetrics
            //    Ratio names from compute_homeostasis: "synapse_health", "note_density",
            //    "decision_coverage", "churn_balance", "scar_load"
            let metrics = HomeostasisMetrics {
                synapse_health: report
                    .ratios
                    .iter()
                    .find(|r| r.name == "synapse_health")
                    .map(|r| r.value)
                    .unwrap_or(1.0),
                note_density: report
                    .ratios
                    .iter()
                    .find(|r| r.name == "note_density")
                    .map(|r| r.value)
                    .unwrap_or(1.0),
                pain_score: report.pain_score,
            };

            // 4. Evaluate corrective actions
            let actions = HomeostasisController::evaluate(&metrics);

            if actions.is_empty() {
                debug!(
                    "HomeostasisCheck: project '{}' has pain but no actionable corrections",
                    project.name
                );
                continue;
            }

            info!(
                "HomeostasisCheck: project '{}' (pain={:.3}) needs {} corrective action(s): {}",
                project.name,
                report.pain_score,
                actions.len(),
                actions
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            // 5. Execute corrective actions (paginated backfill)
            let current_cursor = {
                let cursor = self.cursor.lock().await;
                cursor.clone()
            };

            // Reset default_note_energy when density drops back to normal
            if metrics.note_density <= 2.5 {
                if let Err(e) = ctx.graph.set_default_note_energy(project.id, None).await {
                    warn!(
                        "HomeostasisCheck: failed to reset default_note_energy for '{}': {}",
                        project.name, e
                    );
                }
            }

            // 5b. Compute adaptive backfill parameters from neural metrics
            let backfill_params = match ctx.graph.get_neural_metrics(project.id).await {
                Ok(neural) => {
                    let total_notes = neural.total_notes_count as usize;
                    let params = compute_backfill_params(total_notes, metrics.synapse_health);
                    debug!(
                        total_notes,
                        batch_size = params.batch_size,
                        max_neighbors = params.max_neighbors,
                        "HomeostasisCheck: computed adaptive backfill params for '{}'",
                        project.name
                    );
                    Some(params)
                }
                Err(e) => {
                    warn!(
                        "HomeostasisCheck: get_neural_metrics failed for '{}': {}, using defaults",
                        project.name, e
                    );
                    None
                }
            };

            let graph_arc: Arc<dyn crate::neo4j::traits::GraphStore> = Arc::clone(&ctx.graph);
            match execute_actions(
                &graph_arc,
                ctx.search.as_ref(),
                &actions,
                ExecuteContext {
                    cursor: Some(&current_cursor),
                    project_id: Some(project.id),
                    backfill_params: backfill_params.as_ref(),
                },
            )
            .await
            {
                Ok(result) => {
                    info!(
                        "HomeostasisCheck: executed {}/{} actions for '{}'",
                        result.executed,
                        actions.len(),
                        project.name
                    );

                    // Update the persisted cursor
                    if let Some(new_cursor) = result.backfill_cursor {
                        let mut cursor = self.cursor.lock().await;
                        if new_cursor.completed {
                            debug!(
                                "HomeostasisCheck: backfill completed for '{}', resetting cursor",
                                project.name
                            );
                            cursor.reset();
                        } else {
                            debug!(
                                offset = new_cursor.offset,
                                "HomeostasisCheck: backfill partial for '{}', cursor advanced",
                                project.name
                            );
                            *cursor = new_cursor;
                        }
                    }

                    // Create alert if actions were taken
                    if result.executed > 0 {
                        if let Some(ref emitter) = ctx.emitter {
                            emitter.emit_created(
                                crate::events::EntityType::Alert,
                                &project.id.to_string(),
                                serde_json::json!({
                                    "alert_type": "homeostasis_correction",
                                    "project": project.name,
                                    "pain_score": report.pain_score,
                                    "actions_executed": result.executed,
                                    "actions": actions.iter().map(|a| a.to_string()).collect::<Vec<_>>(),
                                }),
                                Some(project.id.to_string()),
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "HomeostasisCheck: execute_actions failed for '{}': {}",
                        project.name, e
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homeostasis_check_name() {
        let check = HomeostasisCheck::new();
        assert_eq!(check.name(), "homeostasis");
    }

    #[test]
    fn test_homeostasis_check_interval_2h() {
        let check = HomeostasisCheck::new();
        assert_eq!(check.interval(), Duration::from_secs(2 * 3600));
    }

    #[test]
    fn test_homeostasis_check_timeout_override() {
        let check = HomeostasisCheck::new();
        assert_eq!(check.timeout_override(), Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_dormancy_threshold_is_reasonable() {
        // Threshold should be low enough that only truly stable systems are skipped.
        // Use const block to satisfy clippy::assertions_on_constants.
        const {
            assert!(DORMANCY_THRESHOLD > 0.0);
            assert!(DORMANCY_THRESHOLD <= 0.15);
        }
    }

    #[test]
    fn test_default_impl() {
        let check = HomeostasisCheck::default();
        assert_eq!(check.name(), "homeostasis");
    }
}
