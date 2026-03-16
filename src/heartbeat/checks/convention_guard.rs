//! ConventionGuardCheck — loads guideline notes for structurally drifted files.
//!
//! Uses `compute_structural_drift` to find files that have drifted from their
//! community centroid, then checks if relevant guideline notes exist.
//! Creates alerts for drifted files that lack convention coverage.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::notes::models::{NoteFilters, NoteStatus, NoteType};

/// Check for convention drift (every 12 hours).
pub struct ConventionGuardCheck;

#[async_trait]
impl HeartbeatCheck for ConventionGuardCheck {
    fn name(&self) -> &str {
        "convention_guard"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(12 * 60 * 60) // 12 hours
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let projects = ctx.graph.list_projects().await?;

        for project in &projects {
            // Get structural drift report
            let drift_report = match ctx
                .graph
                .compute_structural_drift(project.id, None, None)
                .await
            {
                Ok(report) => report,
                Err(e) => {
                    debug!(
                        "ConventionGuardCheck: drift computation skipped for '{}': {}",
                        project.name, e
                    );
                    continue;
                }
            };

            // Count drifted files (warning + critical)
            let drifted_count = drift_report.warning_count + drift_report.critical_count;
            if drifted_count == 0 {
                debug!(
                    "ConventionGuardCheck: no drift detected for '{}'",
                    project.name
                );
                continue;
            }

            // Check if we have guideline notes for this project
            let guidelines = ctx
                .graph
                .list_notes(
                    Some(project.id),
                    None,
                    &NoteFilters {
                        note_type: Some(vec![NoteType::Guideline]),
                        status: Some(vec![NoteStatus::Active]),
                        ..Default::default()
                    },
                )
                .await;

            let guideline_count = match guidelines {
                Ok((notes, _)) => notes.len(),
                Err(_) => 0,
            };

            if guideline_count == 0 && drifted_count > 0 {
                // Create alert: drifted files with no guidelines
                let alert = crate::neo4j::models::AlertNode::new(
                    "convention_gap".to_string(),
                    crate::neo4j::models::AlertSeverity::Info,
                    format!(
                        "Project '{}' has {} structurally drifted file(s) but no guideline notes. \
                         Consider creating conventions with note(action: \"create\", type: \"guideline\").",
                        project.name, drifted_count
                    ),
                    Some(project.id),
                );

                if let Err(e) = ctx.graph.create_alert(&alert).await {
                    warn!(
                        "ConventionGuardCheck: failed to create alert for '{}': {}",
                        project.name, e
                    );
                }

                if let Some(ref emitter) = ctx.emitter {
                    emitter.emit_created(
                        crate::events::EntityType::Alert,
                        &alert.id.to_string(),
                        serde_json::json!({
                            "alert_type": "convention_gap",
                            "project": project.name,
                            "drifted_files": drifted_count,
                        }),
                        Some(project.id.to_string()),
                    );
                }
            }

            debug!(
                "ConventionGuardCheck: project '{}' — {} drifted files, {} guidelines",
                project.name, drifted_count, guideline_count
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convention_guard_check_name() {
        let check = ConventionGuardCheck;
        assert_eq!(check.name(), "convention_guard");
    }

    #[test]
    fn test_convention_guard_check_interval() {
        let check = ConventionGuardCheck;
        assert_eq!(check.interval(), Duration::from_secs(12 * 3600));
    }
}
