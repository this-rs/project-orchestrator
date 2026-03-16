//! GitDriftCheck — detects when local branches are behind their remote.
//!
//! Runs `git fetch` + `git log HEAD..origin/main` per watched project.
//! Creates an alert if there are new upstream commits.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};

/// Check for git drift on watched projects (every 10 minutes).
pub struct GitDriftCheck;

#[async_trait]
impl HeartbeatCheck for GitDriftCheck {
    fn name(&self) -> &str {
        "git_drift"
    }

    fn interval(&self) -> Duration {
        Duration::from_secs(10 * 60) // 10 minutes
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let projects = ctx.graph.list_projects().await?;

        for project in &projects {
            let root_path = &project.root_path;

            // git fetch (quiet, timeout handled by engine)
            let fetch_output = tokio::process::Command::new("git")
                .args(["fetch", "--quiet"])
                .current_dir(root_path)
                .output()
                .await;

            if let Err(e) = &fetch_output {
                warn!(
                    "GitDriftCheck: git fetch failed for '{}': {}",
                    project.name, e
                );
                continue;
            }

            // Check for new commits on origin/main
            let log_output = tokio::process::Command::new("git")
                .args(["log", "--oneline", "HEAD..origin/main"])
                .current_dir(root_path)
                .output()
                .await;

            match log_output {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let commit_count = stdout.lines().count();

                    if commit_count > 0 {
                        debug!(
                            "GitDriftCheck: project '{}' is {} commit(s) behind origin/main",
                            project.name, commit_count
                        );

                        // Create alert
                        let alert = crate::neo4j::models::AlertNode::new(
                            "git_drift".to_string(),
                            crate::neo4j::models::AlertSeverity::Warning,
                            format!(
                                "Project '{}' is {} commit(s) behind origin/main",
                                project.name, commit_count
                            ),
                            Some(project.id),
                        );

                        if let Err(e) = ctx.graph.create_alert(&alert).await {
                            warn!(
                                "GitDriftCheck: failed to create alert for '{}': {}",
                                project.name, e
                            );
                        }

                        // Emit event
                        if let Some(ref emitter) = ctx.emitter {
                            emitter.emit_created(
                                crate::events::EntityType::Alert,
                                &alert.id.to_string(),
                                serde_json::json!({
                                    "alert_type": "git_drift",
                                    "project": project.name,
                                    "commits_behind": commit_count,
                                }),
                                Some(project.id.to_string()),
                            );
                        }
                    } else {
                        debug!(
                            "GitDriftCheck: project '{}' is up to date with origin/main",
                            project.name
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "GitDriftCheck: git log failed for '{}': {}",
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
    fn test_git_drift_check_name() {
        let check = GitDriftCheck;
        assert_eq!(check.name(), "git_drift");
    }

    #[test]
    fn test_git_drift_check_interval() {
        let check = GitDriftCheck;
        assert_eq!(check.interval(), Duration::from_secs(600));
    }
}
