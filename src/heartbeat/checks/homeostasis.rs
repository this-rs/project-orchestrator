//! HomeostasisCheck — auto-corrective thermostat for the knowledge graph.
//!
//! Runs every 24 hours. For each project:
//! 1. Computes homeostasis report via `compute_homeostasis`
//! 2. Fetches neural metrics via `get_neural_metrics`
//! 3. Feeds them to `HomeostasisController::evaluate`
//! 4. Executes corrective actions via `execute_actions`
//!
//! Creates alerts when corrective actions are taken.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::homeostasis::{execute_actions, HomeostasisController, HomeostasisMetrics};

/// Run homeostasis evaluation and correction on all projects (every 24 hours).
pub struct HomeostasisCheck;

#[async_trait]
impl HeartbeatCheck for HomeostasisCheck {
    fn name(&self) -> &str {
        "homeostasis"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(24 * 60 * 60) // 24 hours
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let projects = ctx.graph.list_projects().await?;

        for project in &projects {
            debug!(
                "HomeostasisCheck: evaluating project '{}'",
                project.name
            );

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

            // 2. Fetch neural metrics for dead_notes info
            let neural = match ctx.graph.get_neural_metrics(project.id).await {
                Ok(n) => n,
                Err(e) => {
                    warn!(
                        "HomeostasisCheck: get_neural_metrics failed for '{}': {}",
                        project.name, e
                    );
                    continue;
                }
            };

            // 3. Map report ratios + neural metrics to HomeostasisMetrics
            //    Ratio names from compute_homeostasis: "synapse_health", "note_density",
            //    "decision_coverage", "churn_balance", "scar_load"
            let metrics = HomeostasisMetrics {
                synapse_health: report
                    .ratios
                    .iter()
                    .find(|r| r.name == "synapse_health")
                    .map(|r| r.value)
                    .unwrap_or(1.0),
                dead_notes_ratio: if neural.dead_notes_count > 0 {
                    // Approximate: dead_notes / (dead_notes + active_synapses as proxy for active notes)
                    // This is a heuristic; consolidate_memory will handle the actual cleanup.
                    let total = (neural.dead_notes_count as f64)
                        + (neural.active_synapses.max(1) as f64);
                    neural.dead_notes_count as f64 / total
                } else {
                    0.0
                },
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
                    "HomeostasisCheck: project '{}' is healthy, no actions needed",
                    project.name
                );
                continue;
            }

            info!(
                "HomeostasisCheck: project '{}' needs {} corrective action(s): {}",
                project.name,
                actions.len(),
                actions
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            // 5. Execute corrective actions
            let graph_arc: Arc<dyn crate::neo4j::traits::GraphStore> =
                Arc::clone(&ctx.graph);
            match execute_actions(&graph_arc, &actions).await {
                Ok(executed) => {
                    info!(
                        "HomeostasisCheck: executed {}/{} actions for '{}'",
                        executed,
                        actions.len(),
                        project.name
                    );

                    // Create alert if actions were taken
                    if executed > 0 {
                        if let Some(ref emitter) = ctx.emitter {
                            emitter.emit_created(
                                crate::events::EntityType::Alert,
                                &project.id.to_string(),
                                serde_json::json!({
                                    "alert_type": "homeostasis_correction",
                                    "project": project.name,
                                    "actions_executed": executed,
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
        let check = HomeostasisCheck;
        assert_eq!(check.name(), "homeostasis");
    }

    #[test]
    fn test_homeostasis_check_interval() {
        let check = HomeostasisCheck;
        assert_eq!(check.interval(), Duration::from_secs(24 * 3600));
    }
}
