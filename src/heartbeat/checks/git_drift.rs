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

/// Per-repository limit for `git fetch` and the comparison.
const GIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Repositories fetched concurrently.
const FETCH_CONCURRENCY: usize = 8;

/// Commits `HEAD` is behind `origin/main` in the repository at `root`, or
/// `None` when it is not a git repository, has no `origin/main`, or git
/// fails. `git fetch` is bounded by [`GIT_TIMEOUT`], never prompts for
/// credentials, and is killed when abandoned; if it fails, the comparison
/// uses the last fetched state.
async fn commits_behind(root: &str) -> Option<usize> {
    if !std::path::Path::new(root).join(".git").exists() {
        return None;
    }
    let git = |args: &[&str]| {
        let mut cmd = tokio::process::Command::new("git");
        cmd.args(args)
            .current_dir(root)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
            .kill_on_drop(true);
        cmd
    };
    match tokio::time::timeout(GIT_TIMEOUT, git(&["fetch", "--quiet"]).output()).await {
        Ok(Ok(out)) if out.status.success() => {}
        _ => {
            debug!("GitDriftCheck: fetch failed or timed out for {root}, using last fetched state")
        }
    }
    let out = tokio::time::timeout(
        GIT_TIMEOUT,
        git(&["rev-list", "--count", "HEAD..origin/main"]).output(),
    )
    .await
    .ok()?
    .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

#[async_trait]
impl HeartbeatCheck for GitDriftCheck {
    fn name(&self) -> &str {
        "git_drift"
    }

    fn interval(&self) -> Duration {
        // Fetching every repository takes ~40-70s on m4 (large ones like
        // pytorch, llama.cpp); the engine runs checks sequentially, so do it
        // every 30 minutes, not every 10.
        Duration::from_secs(30 * 60)
    }

    /// Network fetches of every repository: the engine's 5s default could
    /// never cover them — it used to fetch 40+ repositories sequentially,
    /// time out on every tick and never report anything.
    fn timeout_override(&self) -> Option<Duration> {
        Some(Duration::from_secs(90))
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        use futures::StreamExt;

        let projects = ctx.graph.list_projects().await?;
        // Several projects can share one repository: fetch each root once.
        let mut by_root: std::collections::HashMap<String, Vec<_>> =
            std::collections::HashMap::new();
        for project in &projects {
            if !project.root_path.is_empty() {
                by_root
                    .entry(crate::expand_tilde(&project.root_path))
                    .or_default()
                    .push(project);
            }
        }

        let results: Vec<(String, Option<usize>)> = futures::stream::iter(by_root.keys().cloned())
            .map(|root| async move {
                let behind = commits_behind(&root).await;
                (root, behind)
            })
            .buffer_unordered(FETCH_CONCURRENCY)
            .collect()
            .await;

        for (root, behind) in results {
            let Some(commit_count) = behind.filter(|n| *n > 0) else {
                continue;
            };
            for project in by_root.get(&root).into_iter().flatten() {
                debug!(
                    "GitDriftCheck: project '{}' is {} commit(s) behind origin/main",
                    project.name, commit_count
                );
                let alert = crate::neo4j::models::AlertNode::new_for_subject(
                    "git_drift".to_string(),
                    crate::neo4j::models::AlertSeverity::Warning,
                    format!(
                        "Project '{}' is {} commit(s) behind origin/main",
                        project.name, commit_count
                    ),
                    Some(project.id),
                    "behind-origin",
                );
                if let Err(e) = ctx.graph.create_alert(&alert).await {
                    warn!(
                        "GitDriftCheck: failed to create alert for '{}': {}",
                        project.name, e
                    );
                }
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
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clone that is `behind` commits behind its origin's main.
    fn repo_behind(behind: usize) -> (tempfile::TempDir, String) {
        let tmp = tempfile::tempdir().unwrap();
        let run = |dir: &std::path::Path, args: &[&str]| {
            let ok = std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .env("GIT_AUTHOR_NAME", "t")
                .env("GIT_AUTHOR_EMAIL", "t@t")
                .env("GIT_COMMITTER_NAME", "t")
                .env("GIT_COMMITTER_EMAIL", "t@t")
                .status()
                .unwrap()
                .success();
            assert!(ok, "git {args:?}");
        };
        let origin = tmp.path().join("origin");
        std::fs::create_dir(&origin).unwrap();
        run(&origin, &["init", "-q", "-b", "main"]);
        run(&origin, &["commit", "-q", "--allow-empty", "-m", "init"]);
        run(
            tmp.path(),
            &["clone", "-q", origin.to_str().unwrap(), "clone"],
        );
        for i in 0..behind {
            run(
                &origin,
                &["commit", "-q", "--allow-empty", "-m", &format!("c{i}")],
            );
        }
        let clone = tmp.path().join("clone").display().to_string();
        (tmp, clone)
    }

    #[tokio::test]
    async fn test_commits_behind_fetches_and_counts() {
        let (_tmp, clone) = repo_behind(3);
        assert_eq!(commits_behind(&clone).await, Some(3));
        let (_tmp2, up_to_date) = repo_behind(0);
        assert_eq!(commits_behind(&up_to_date).await, Some(0));
    }

    #[tokio::test]
    async fn test_commits_behind_ignores_non_repositories() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            commits_behind(&tmp.path().display().to_string()).await,
            None
        );
        assert_eq!(commits_behind("/definitely/not/here").await, None);
    }

    #[test]
    fn test_git_drift_check_has_a_realistic_timeout() {
        // Regression: 40+ network fetches under the engine's 5s default
        // timed out on every tick.
        assert!(GitDriftCheck.timeout_override().unwrap() >= Duration::from_secs(60));
    }

    #[test]
    fn test_git_drift_check_name() {
        let check = GitDriftCheck;
        assert_eq!(check.name(), "git_drift");
    }

    #[test]
    fn test_git_drift_check_interval() {
        let check = GitDriftCheck;
        assert_eq!(check.interval(), Duration::from_secs(1800));
    }
}
