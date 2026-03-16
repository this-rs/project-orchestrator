//! MaintenanceCheck — runs deep_maintenance for all projects periodically.
//!
//! Wraps `skills::maintenance::deep_maintenance` as a heartbeat check.
//! Runs every 24 hours to perform aggressive cleanup: decay, energy update,
//! staleness scoring, stuck task detection, and recommendations.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::skills::maintenance::SkillMaintenanceConfig;

/// Run deep maintenance on all projects (every 24 hours).
pub struct MaintenanceCheck;

#[async_trait]
impl HeartbeatCheck for MaintenanceCheck {
    fn name(&self) -> &str {
        "deep_maintenance"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(24 * 60 * 60) // 24 hours
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let projects = ctx.graph.list_projects().await?;
        let config = SkillMaintenanceConfig::default();

        for project in &projects {
            info!(
                "MaintenanceCheck: running deep maintenance for '{}'",
                project.name
            );

            match crate::skills::maintenance::deep_maintenance(
                ctx.graph.as_ref(),
                project.id,
                &config,
            )
            .await
            {
                Ok(report) => {
                    debug!(
                        "MaintenanceCheck: deep maintenance completed for '{}' — \
                         stale_notes_flagged: {}, stuck_tasks: {}, recommendations: {}",
                        project.name,
                        report.stale_notes_flagged,
                        report.stuck_tasks_found,
                        report.recommendations.len(),
                    );

                    // Create alert if stagnation was detected
                    if report.stagnation.is_stagnating {
                        let alert = crate::neo4j::models::AlertNode::new(
                            "stagnation".to_string(),
                            crate::neo4j::models::AlertSeverity::Warning,
                            format!(
                                "Stagnation detected in project '{}': {} stale notes, {} stuck tasks. {}",
                                project.name,
                                report.stale_notes_flagged,
                                report.stuck_tasks_found,
                                report.recommendations.first().cloned().unwrap_or_default(),
                            ),
                            Some(project.id),
                        );

                        if let Err(e) = ctx.graph.create_alert(&alert).await {
                            warn!(
                                "MaintenanceCheck: failed to create stagnation alert for '{}': {}",
                                project.name, e
                            );
                        }

                        if let Some(ref emitter) = ctx.emitter {
                            emitter.emit_created(
                                crate::events::EntityType::Alert,
                                &alert.id.to_string(),
                                serde_json::json!({
                                    "alert_type": "stagnation",
                                    "project": project.name,
                                }),
                                Some(project.id.to_string()),
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "MaintenanceCheck: deep maintenance failed for '{}': {}",
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
    fn test_maintenance_check_name() {
        let check = MaintenanceCheck;
        assert_eq!(check.name(), "deep_maintenance");
    }

    #[test]
    fn test_maintenance_check_interval() {
        let check = MaintenanceCheck;
        assert_eq!(check.interval(), Duration::from_secs(24 * 3600));
    }
}
