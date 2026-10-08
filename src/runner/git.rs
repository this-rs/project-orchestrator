//! Git Worktree Collector — recovers commits from agent worktrees.
//!
//! When agents (Claude Code) create git worktrees for isolation, their commits
//! stay in the worktree branch and are never merged back to the run branch.
//! This module provides:
//!
//! - `WorktreeCollector` — scans, collects, cherry-picks, and cleans up worktrees
//! - `WorktreeInfo` — parsed worktree metadata from `git worktree list --porcelain`
//! - `MergeResult` / `WorktreeResolution` — structured results for the runner
//!
//! A `Mutex` serializes git operations to prevent race conditions in parallel waves.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

// ============================================================================
// Types
// ============================================================================

/// Parsed information about a single git worktree.
#[derive(Debug, Clone, Default)]
pub struct WorktreeInfo {
    /// Absolute path to the worktree directory.
    pub path: PathBuf,
    /// Branch checked out in this worktree (None if detached HEAD).
    pub branch: Option<String>,
    /// HEAD commit SHA.
    pub head_sha: String,
    /// Whether the worktree is in detached HEAD state.
    pub is_detached: bool,
    /// `locked` (optionally with a reason) — `git worktree lock` was used.
    pub is_locked: bool,
    /// `prunable` — the worktree directory is gone and `git worktree prune`
    /// would drop its administrative files.
    pub is_prunable: bool,
    /// `bare` — the main entry of a bare repository (no working tree).
    pub is_bare: bool,
}

/// Result of cherry-picking commits from a single worktree.
#[derive(Debug, Clone, Default)]
pub struct MergeResult {
    /// SHAs that were successfully cherry-picked.
    pub merged: Vec<String>,
    /// (SHA, error message) for commits that caused conflicts.
    pub conflicts: Vec<(String, String)>,
}

/// Per-worktree resolution detail.
#[derive(Debug, Clone)]
pub struct WorktreeResult {
    pub worktree: WorktreeInfo,
    pub merge: MergeResult,
    pub cleaned_up: bool,
}

/// Aggregated resolution result for all worktrees.
#[derive(Debug, Clone, Default)]
pub struct WorktreeResolution {
    pub total_merged: usize,
    pub total_conflicts: usize,
    pub details: Vec<WorktreeResult>,
}

// ============================================================================
// WorktreeCollector
// ============================================================================

/// Collects and recovers commits from agent-created git worktrees.
///
/// All git-mutating operations are serialized via an `Arc<Mutex<()>>` to
/// prevent race conditions when multiple tasks run in parallel (JoinSet).
pub struct WorktreeCollector;

impl WorktreeCollector {
    // --------------------------------------------------------------------
    // Step 1 — list_worktrees
    // --------------------------------------------------------------------

    /// Parse `git worktree list --porcelain` and return structured worktree info.
    ///
    /// Each porcelain entry looks like:
    /// ```text
    /// worktree /absolute/path
    /// HEAD <sha>
    /// branch refs/heads/<name>
    ///
    /// ```
    /// Detached worktrees have `detached` instead of `branch ...`.
    pub async fn list_worktrees(cwd: &str) -> Result<Vec<WorktreeInfo>> {
        let output = Command::new("git")
            .args(["worktree", "list", "--porcelain"])
            .current_dir(cwd)
            .output()
            .await
            .context("Failed to run git worktree list")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("git worktree list failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(Self::parse_worktree_porcelain(&stdout))
    }

    /// Pure parsing logic, separated for testability.
    ///
    /// Understands the attributes `git worktree list --porcelain` can emit:
    /// `worktree`, `HEAD`, `branch`, `detached`, `bare`, and `locked` /
    /// `prunable` (each with an optional free-text reason after a space).
    pub fn parse_worktree_porcelain(output: &str) -> Vec<WorktreeInfo> {
        let mut worktrees = Vec::new();
        let mut current: Option<WorktreeInfo> = None;

        for line in output.lines() {
            if let Some(path) = line.strip_prefix("worktree ") {
                // A new record starts: flush a previous one that had no blank
                // line separator (defensive — git always emits one).
                if let Some(done) = current.take() {
                    worktrees.push(done);
                }
                current = Some(WorktreeInfo {
                    path: PathBuf::from(path),
                    ..Default::default()
                });
            } else if line.is_empty() {
                if let Some(done) = current.take() {
                    worktrees.push(done);
                }
            } else if let Some(wt) = current.as_mut() {
                if let Some(head) = line.strip_prefix("HEAD ") {
                    wt.head_sha = head.to_string();
                } else if let Some(branch) = line.strip_prefix("branch ") {
                    wt.branch = Some(
                        branch
                            .strip_prefix("refs/heads/")
                            .unwrap_or(branch)
                            .to_string(),
                    );
                } else if line == "detached" {
                    wt.is_detached = true;
                } else if line == "bare" {
                    wt.is_bare = true;
                } else if line == "locked" || line.starts_with("locked ") {
                    wt.is_locked = true;
                } else if line == "prunable" || line.starts_with("prunable ") {
                    wt.is_prunable = true;
                }
            }
        }

        // Handle last entry if no trailing blank line
        if let Some(done) = current.take() {
            worktrees.push(done);
        }

        worktrees
    }

    // --------------------------------------------------------------------
    // Step 2 — collect_commits
    // --------------------------------------------------------------------

    /// List commit SHAs present in `worktree_branch` but NOT in `run_branch`.
    ///
    /// Uses `git log <worktree_branch> --not <run_branch> --format=%H` to get
    /// only the commits that need to be cherry-picked. Results are returned in
    /// reverse chronological order (oldest first for cherry-pick).
    pub async fn collect_commits(
        worktree_branch: &str,
        run_branch: &str,
        cwd: &str,
    ) -> Result<Vec<String>> {
        let output = Command::new("git")
            .args([
                "log",
                worktree_branch,
                "--not",
                run_branch,
                "--format=%H",
                "--reverse",
            ])
            .current_dir(cwd)
            .output()
            .await
            .context("Failed to run git log")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Non-fatal: branch may not exist or have diverged
            debug!(
                "git log {}..{} failed: {}",
                run_branch, worktree_branch, stderr
            );
            return Ok(Vec::new());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let commits: Vec<String> = stdout
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.trim().to_string())
            .collect();

        Ok(commits)
    }

    // --------------------------------------------------------------------
    // Step 3 — cherry_pick_commits
    // --------------------------------------------------------------------

    /// Cherry-pick a list of commits onto `run_branch`.
    ///
    /// On conflict: aborts the cherry-pick (repo stays clean), logs the conflict,
    /// and stops processing further commits from this worktree.
    ///
    /// # Safety
    /// The caller MUST hold the git lock before calling this method.
    pub async fn cherry_pick_commits(
        commits: &[String],
        run_branch: &str,
        cwd: &str,
    ) -> Result<MergeResult> {
        let mut result = MergeResult::default();

        if commits.is_empty() {
            return Ok(result);
        }

        // Ensure we're on the run branch
        let current = current_branch(cwd).await.unwrap_or_default();
        if current != run_branch {
            let checkout = Command::new("git")
                .args(["checkout", run_branch])
                .current_dir(cwd)
                .output()
                .await
                .context("Failed to checkout run branch")?;

            if !checkout.status.success() {
                let stderr = String::from_utf8_lossy(&checkout.stderr);
                return Err(anyhow::anyhow!(
                    "Failed to checkout {}: {}",
                    run_branch,
                    stderr
                ));
            }
        }

        for sha in commits {
            let output = Command::new("git")
                .args(["cherry-pick", sha])
                .current_dir(cwd)
                .output()
                .await
                .context("Failed to run git cherry-pick")?;

            if output.status.success() {
                info!("Cherry-picked commit {}", &sha[..7.min(sha.len())]);
                result.merged.push(sha.clone());
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                error!(
                    "Cherry-pick conflict on {}: {}",
                    &sha[..7.min(sha.len())],
                    stderr
                );

                // ALWAYS abort to leave the repo clean
                let _ = Command::new("git")
                    .args(["cherry-pick", "--abort"])
                    .current_dir(cwd)
                    .output()
                    .await;

                result.conflicts.push((sha.clone(), stderr));
                // Stop processing this worktree on first conflict
                break;
            }
        }

        Ok(result)
    }

    // --------------------------------------------------------------------
    // Step 4 — resolve_worktrees (orchestration)
    // --------------------------------------------------------------------

    /// Orchestrate worktree resolution: list → collect → cherry-pick for each worktree.
    ///
    /// The `git_lock` serializes all git operations to prevent race conditions
    /// when called from parallel tasks in a JoinSet.
    pub async fn resolve_worktrees(
        run_branch: &str,
        cwd: &str,
        git_lock: Arc<Mutex<()>>,
    ) -> Result<WorktreeResolution> {
        let _lock = git_lock.lock().await;

        let mut resolution = WorktreeResolution::default();

        // 1. List all worktrees
        let worktrees = Self::list_worktrees(cwd).await?;

        // Filter to agent worktrees only (.claude/worktrees/agent-*)
        let agent_worktrees: Vec<WorktreeInfo> = worktrees
            .into_iter()
            .filter(|wt| {
                wt.path
                    .to_string_lossy()
                    .contains(".claude/worktrees/agent-")
            })
            .collect();

        if agent_worktrees.is_empty() {
            debug!("No agent worktrees found");
            return Ok(resolution);
        }

        info!("Found {} agent worktrees to resolve", agent_worktrees.len());

        // Stash uncommitted changes to avoid conflicts
        let was_stashed = stash_if_needed(cwd).await.unwrap_or(false);

        // 2–3. For each worktree: collect commits → cherry-pick
        for wt in &agent_worktrees {
            let branch_name = match &wt.branch {
                Some(b) => b.clone(),
                None => {
                    debug!("Skipping detached worktree at {}", wt.path.display());
                    resolution.details.push(WorktreeResult {
                        worktree: wt.clone(),
                        merge: MergeResult::default(),
                        cleaned_up: false,
                    });
                    continue;
                }
            };

            let commits = match Self::collect_commits(&branch_name, run_branch, cwd).await {
                Ok(c) => c,
                Err(e) => {
                    warn!(
                        "Failed to collect commits from {}: {}",
                        wt.path.display(),
                        e
                    );
                    resolution.details.push(WorktreeResult {
                        worktree: wt.clone(),
                        merge: MergeResult::default(),
                        cleaned_up: false,
                    });
                    continue;
                }
            };

            if commits.is_empty() {
                debug!(
                    "No new commits in worktree {} (branch {})",
                    wt.path.display(),
                    branch_name
                );
                // Clean up even if no commits — the worktree is stale
                let cleaned = Self::cleanup_single_worktree(cwd, wt).await;
                resolution.details.push(WorktreeResult {
                    worktree: wt.clone(),
                    merge: MergeResult::default(),
                    cleaned_up: cleaned,
                });
                continue;
            }

            info!(
                "Recovering {} commits from worktree {} (branch {})",
                commits.len(),
                wt.path.display(),
                branch_name
            );

            let merge = Self::cherry_pick_commits(&commits, run_branch, cwd).await?;

            let all_merged = merge.conflicts.is_empty();
            resolution.total_merged += merge.merged.len();
            resolution.total_conflicts += merge.conflicts.len();

            // Only cleanup if ALL commits were recovered successfully
            let cleaned = if all_merged {
                Self::cleanup_single_worktree(cwd, wt).await
            } else {
                false
            };

            resolution.details.push(WorktreeResult {
                worktree: wt.clone(),
                merge,
                cleaned_up: cleaned,
            });
        }

        // Restore stash
        unstash_if_needed(cwd, was_stashed).await?;

        if resolution.total_merged > 0 || resolution.total_conflicts > 0 {
            info!(
                "Worktree resolution complete: {} merged, {} conflicts, {} worktrees processed",
                resolution.total_merged,
                resolution.total_conflicts,
                resolution.details.len()
            );
        }

        Ok(resolution)
    }

    // --------------------------------------------------------------------
    // Step 5 — cleanup_worktrees
    // --------------------------------------------------------------------

    /// Remove the disposable agent worktrees (`.claude/worktrees/agent-*`)
    /// after resolution.
    ///
    /// Called in `finalize_run()` to clean up at the end of a plan. Only agent
    /// worktrees are candidates: a user's own `.claude/worktrees/<name>` (e.g. an
    /// interactive session in the same repo) is never touched. An agent
    /// worktree whose branch still holds commits that were not recovered onto
    /// the current branch is kept as well, since `worktree remove --force` +
    /// `branch -D` would destroy that work. Returns the number of removed
    /// worktrees.
    pub async fn cleanup_worktrees(cwd: &str) -> Result<usize> {
        let worktrees = Self::list_worktrees(cwd).await?;

        let agent_worktrees: Vec<&WorktreeInfo> = worktrees
            .iter()
            .filter(|wt| {
                wt.path
                    .to_string_lossy()
                    .contains(".claude/worktrees/agent-")
            })
            .collect();

        if agent_worktrees.is_empty() {
            return Ok(0);
        }

        info!(
            "Cleaning up {} worktrees in finalize_run",
            agent_worktrees.len()
        );

        let mut cleaned = 0;
        for wt in agent_worktrees {
            if let Some(branch) = &wt.branch {
                match Self::unrecovered_commit_count(branch, cwd).await {
                    Ok(0) => {}
                    Ok(n) => {
                        warn!(
                            "Keeping worktree {}: {} commit(s) on '{}' were not recovered",
                            wt.path.display(),
                            n,
                            branch
                        );
                        continue;
                    }
                    Err(e) => {
                        warn!(
                            "Keeping worktree {}: cannot verify its commits are recovered: {}",
                            wt.path.display(),
                            e
                        );
                        continue;
                    }
                }
            }
            if Self::cleanup_single_worktree(cwd, wt).await {
                cleaned += 1;
            }
        }

        Ok(cleaned)
    }

    /// Number of commits on `branch` with no patch-equivalent commit on `HEAD`.
    ///
    /// Recovered commits are cherry-picked (new SHAs), so ancestry can't be
    /// used: `git cherry HEAD <branch>` matches by patch-id instead and marks
    /// commits missing from HEAD with `+`.
    async fn unrecovered_commit_count(branch: &str, cwd: &str) -> Result<usize> {
        let output = Command::new("git")
            .args(["cherry", "HEAD", branch])
            .current_dir(cwd)
            .output()
            .await
            .context("Failed to run git cherry")?;

        if !output.status.success() {
            anyhow::bail!(
                "git cherry HEAD {} failed: {}",
                branch,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|l| l.starts_with('+'))
            .count())
    }

    /// Remove a single worktree and optionally delete its branch.
    async fn cleanup_single_worktree(cwd: &str, wt: &WorktreeInfo) -> bool {
        let wt_path = wt.path.to_string_lossy();

        // Remove the worktree
        let output = Command::new("git")
            .args(["worktree", "remove", &*wt_path, "--force"])
            .current_dir(cwd)
            .output()
            .await;

        match &output {
            Ok(o) if o.status.success() => {
                debug!("Removed worktree {}", wt_path);
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                warn!("Failed to remove worktree {}: {}", wt_path, stderr);
                return false;
            }
            Err(e) => {
                warn!("Failed to run git worktree remove for {}: {}", wt_path, e);
                return false;
            }
        }

        // Delete the branch if it looks like an agent branch
        if let Some(ref branch) = wt.branch {
            let _ = Command::new("git")
                .args(["branch", "-D", branch])
                .current_dir(cwd)
                .output()
                .await;
        }

        true
    }
}

// ============================================================================
// Standalone helpers (used by runner.rs directly)
// ============================================================================

/// Get the HEAD commit SHA.
pub async fn head_sha(cwd: &str) -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(cwd)
        .output()
        .await
        .context("Failed to get HEAD SHA")?;

    if !output.status.success() {
        anyhow::bail!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Get the current branch name.
pub async fn current_branch(cwd: &str) -> Result<String> {
    let output = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(cwd)
        .output()
        .await
        .context("Failed to get current branch")?;

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Check if there are uncommitted changes in the working directory.
async fn has_uncommitted_changes(cwd: &str) -> Result<bool> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(cwd)
        .output()
        .await
        .context("Failed to run git status")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(!stdout.trim().is_empty())
}

/// Stash uncommitted changes if any exist.
async fn stash_if_needed(cwd: &str) -> Result<bool> {
    if has_uncommitted_changes(cwd).await? {
        let output = Command::new("git")
            .args(["stash", "push", "-m", "runner-worktree-collector"])
            .current_dir(cwd)
            .output()
            .await
            .context("Failed to git stash")?;

        if output.status.success() {
            info!("Stashed uncommitted changes before worktree collection");
            return Ok(true);
        }
    }
    Ok(false)
}

/// Pop stash if we stashed earlier.
async fn unstash_if_needed(cwd: &str, was_stashed: bool) -> Result<()> {
    if was_stashed {
        let _ = Command::new("git")
            .args(["stash", "pop"])
            .current_dir(cwd)
            .output()
            .await;
    }
    Ok(())
}

// ============================================================================
// Backward-compat shim — called by runner.rs post-wave integration
// ============================================================================

/// Backward-compatible result type for the existing runner.rs integration.
#[derive(Debug, Clone)]
pub struct WorktreeCollectionResult {
    pub recovered_commits: Vec<String>,
    pub conflicts: Vec<WorktreeConflict>,
    pub cleaned_up: Vec<String>,
}

/// Backward-compatible conflict type.
#[derive(Debug, Clone)]
pub struct WorktreeConflict {
    pub worktree_path: String,
    pub branch: String,
    pub commit_sha: String,
    pub error: String,
}

/// Backward-compatible entry point used by the existing runner.rs post-wave code.
///
/// Wraps `WorktreeCollector::resolve_worktrees` with its own internal mutex.
pub async fn collect_worktree_commits(
    cwd: &str,
    run_branch: &str,
) -> Result<WorktreeCollectionResult> {
    static GIT_MUTEX: std::sync::LazyLock<Arc<Mutex<()>>> =
        std::sync::LazyLock::new(|| Arc::new(Mutex::new(())));

    let resolution =
        WorktreeCollector::resolve_worktrees(run_branch, cwd, GIT_MUTEX.clone()).await?;

    // Convert to backward-compat types
    let mut result = WorktreeCollectionResult {
        recovered_commits: Vec::new(),
        conflicts: Vec::new(),
        cleaned_up: Vec::new(),
    };

    for detail in &resolution.details {
        for sha in &detail.merge.merged {
            result.recovered_commits.push(sha.clone());
        }
        for (sha, err) in &detail.merge.conflicts {
            result.conflicts.push(WorktreeConflict {
                worktree_path: detail.worktree.path.to_string_lossy().to_string(),
                branch: detail.worktree.branch.clone().unwrap_or_default(),
                commit_sha: sha.clone(),
                error: err.clone(),
            });
        }
        if detail.cleaned_up {
            result
                .cleaned_up
                .push(detail.worktree.path.to_string_lossy().to_string());
        }
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // parse_worktree_porcelain — pure parsing, no git needed
    // ----------------------------------------------------------------

    #[test]
    fn test_parse_porcelain_locked_prunable_bare_and_slashed_branch() {
        let output = "worktree /repo.git\nbare\n\n\
worktree /wt/a b\nHEAD aaa\nbranch refs/heads/feat/deep/name\nlocked being moved\n\n\
worktree /wt/c\nHEAD bbb\ndetached\nlocked\nprunable gitdir file points to non-existent location\n";
        let wts = WorktreeCollector::parse_worktree_porcelain(output);
        assert_eq!(wts.len(), 3);
        assert!(wts[0].is_bare && wts[0].branch.is_none());
        assert_eq!(wts[1].path, PathBuf::from("/wt/a b"));
        assert_eq!(wts[1].branch.as_deref(), Some("feat/deep/name"));
        assert!(wts[1].is_locked && !wts[1].is_prunable);
        assert!(wts[2].is_detached && wts[2].is_locked && wts[2].is_prunable);
    }

    #[test]
    fn test_parse_single_worktree() {
        let output = "\
worktree /home/user/project
HEAD abc123def456
branch refs/heads/main

";
        let worktrees = WorktreeCollector::parse_worktree_porcelain(output);
        assert_eq!(worktrees.len(), 1);
        assert_eq!(worktrees[0].path, PathBuf::from("/home/user/project"));
        assert_eq!(worktrees[0].branch.as_deref(), Some("main"));
        assert_eq!(worktrees[0].head_sha, "abc123def456");
        assert!(!worktrees[0].is_detached);
    }

    #[test]
    fn test_parse_multiple_worktrees_with_agent() {
        let output = "\
worktree /home/user/project
HEAD aaa111
branch refs/heads/main

worktree /home/user/project/.claude/worktrees/agent-abc123
HEAD bbb222
branch refs/heads/agent-abc123-branch

worktree /home/user/project/.claude/worktrees/agent-def456
HEAD ccc333
detached

";
        let worktrees = WorktreeCollector::parse_worktree_porcelain(output);
        assert_eq!(worktrees.len(), 3);

        // Main worktree
        assert_eq!(worktrees[0].branch.as_deref(), Some("main"));
        assert!(!worktrees[0].is_detached);

        // Agent worktree with branch
        assert!(worktrees[1]
            .path
            .to_string_lossy()
            .contains(".claude/worktrees/agent-"));
        assert_eq!(worktrees[1].branch.as_deref(), Some("agent-abc123-branch"));
        assert_eq!(worktrees[1].head_sha, "bbb222");
        assert!(!worktrees[1].is_detached);

        // Agent worktree detached
        assert!(worktrees[2]
            .path
            .to_string_lossy()
            .contains(".claude/worktrees/agent-"));
        assert!(worktrees[2].branch.is_none());
        assert_eq!(worktrees[2].head_sha, "ccc333");
        assert!(worktrees[2].is_detached);
    }

    #[test]
    fn test_parse_empty_output() {
        let output = "";
        let worktrees = WorktreeCollector::parse_worktree_porcelain(output);
        assert!(worktrees.is_empty());
    }

    #[test]
    fn test_parse_no_trailing_newline() {
        let output = "\
worktree /home/user/project
HEAD abc123
branch refs/heads/main";
        let worktrees = WorktreeCollector::parse_worktree_porcelain(output);
        assert_eq!(worktrees.len(), 1);
        assert_eq!(worktrees[0].branch.as_deref(), Some("main"));
    }

    #[test]
    fn test_parse_branch_strips_refs_heads() {
        let output = "\
worktree /tmp/wt
HEAD 000aaa
branch refs/heads/feat/my-feature

";
        let worktrees = WorktreeCollector::parse_worktree_porcelain(output);
        assert_eq!(worktrees[0].branch.as_deref(), Some("feat/my-feature"));
    }

    #[test]
    fn test_parse_detached_head() {
        let output = "\
worktree /tmp/wt
HEAD deadbeef
detached

";
        let worktrees = WorktreeCollector::parse_worktree_porcelain(output);
        assert_eq!(worktrees.len(), 1);
        assert!(worktrees[0].is_detached);
        assert!(worktrees[0].branch.is_none());
        assert_eq!(worktrees[0].head_sha, "deadbeef");
    }

    // ----------------------------------------------------------------
    // MergeResult / WorktreeResolution — struct behavior
    // ----------------------------------------------------------------

    #[test]
    fn test_merge_result_default() {
        let mr = MergeResult::default();
        assert!(mr.merged.is_empty());
        assert!(mr.conflicts.is_empty());
    }

    #[test]
    fn test_worktree_resolution_default() {
        let wr = WorktreeResolution::default();
        assert_eq!(wr.total_merged, 0);
        assert_eq!(wr.total_conflicts, 0);
        assert!(wr.details.is_empty());
    }

    // ----------------------------------------------------------------
    // Agent worktree filtering
    // ----------------------------------------------------------------

    #[test]
    fn test_filter_agent_worktrees() {
        let output = "\
worktree /project
HEAD aaa
branch refs/heads/main

worktree /project/.claude/worktrees/agent-123
HEAD bbb
branch refs/heads/wt-branch

worktree /project/other-worktree
HEAD ccc
branch refs/heads/feature

";
        let worktrees = WorktreeCollector::parse_worktree_porcelain(output);
        let agent_wts: Vec<_> = worktrees
            .iter()
            .filter(|wt| {
                wt.path
                    .to_string_lossy()
                    .contains(".claude/worktrees/agent-")
            })
            .collect();
        assert_eq!(agent_wts.len(), 1);
        assert_eq!(agent_wts[0].head_sha, "bbb");
    }

    // ----------------------------------------------------------------
    // Backward-compat conversion
    // ----------------------------------------------------------------

    #[test]
    fn test_worktree_resolution_to_collection_result() {
        let resolution = WorktreeResolution {
            total_merged: 2,
            total_conflicts: 1,
            details: vec![
                WorktreeResult {
                    worktree: WorktreeInfo {
                        path: PathBuf::from("/project/.claude/worktrees/agent-1"),
                        branch: Some("wt-1".to_string()),
                        head_sha: "aaa".to_string(),
                        is_detached: false,
                        ..Default::default()
                    },
                    merge: MergeResult {
                        merged: vec!["sha1".to_string(), "sha2".to_string()],
                        conflicts: vec![],
                    },
                    cleaned_up: true,
                },
                WorktreeResult {
                    worktree: WorktreeInfo {
                        path: PathBuf::from("/project/.claude/worktrees/agent-2"),
                        branch: Some("wt-2".to_string()),
                        head_sha: "bbb".to_string(),
                        is_detached: false,
                        ..Default::default()
                    },
                    merge: MergeResult {
                        merged: vec![],
                        conflicts: vec![("sha3".to_string(), "conflict msg".to_string())],
                    },
                    cleaned_up: false,
                },
            ],
        };

        // Simulate the conversion done in collect_worktree_commits
        let mut recovered = Vec::new();
        let mut conflicts = Vec::new();
        let mut cleaned_up = Vec::new();

        for detail in &resolution.details {
            for sha in &detail.merge.merged {
                recovered.push(sha.clone());
            }
            for (sha, err) in &detail.merge.conflicts {
                conflicts.push((sha.clone(), err.clone()));
            }
            if detail.cleaned_up {
                cleaned_up.push(detail.worktree.path.to_string_lossy().to_string());
            }
        }

        assert_eq!(recovered.len(), 2);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(cleaned_up.len(), 1);
        assert_eq!(conflicts[0].0, "sha3");
    }

    // ----------------------------------------------------------------
    // Integration tests (require git — run in actual repo)
    // ----------------------------------------------------------------

    #[tokio::test]
    #[ignore] // Requires real git repo with branches — skipped in CI
    async fn test_list_worktrees_in_real_repo() {
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let result = WorktreeCollector::list_worktrees(&cwd).await;
        assert!(result.is_ok());
        // Should at least return the main worktree
        let worktrees = result.unwrap();
        assert!(!worktrees.is_empty());
        // The first worktree should have a branch (main/master)
        assert!(worktrees[0].branch.is_some());
    }

    #[tokio::test]
    #[ignore] // Requires real git repo with branches — skipped in CI
    async fn test_current_branch_in_real_repo() {
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let branch = current_branch(&cwd).await;
        assert!(branch.is_ok());
        assert!(!branch.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_collect_commits_nonexistent_branch() {
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let commits =
            WorktreeCollector::collect_commits("nonexistent-branch-xyz", "main", &cwd).await;
        assert!(commits.is_ok());
        // Should return empty vec, not an error
        assert!(commits.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_resolve_worktrees_no_agents() {
        let cwd = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let lock = Arc::new(Mutex::new(()));
        let branch = current_branch(&cwd).await.unwrap_or("main".to_string());
        let resolution = WorktreeCollector::resolve_worktrees(&branch, &cwd, lock).await;
        assert!(resolution.is_ok());
        let r = resolution.unwrap();
        // In a clean repo without agent worktrees, everything should be zero
        assert_eq!(r.total_merged, 0);
        assert_eq!(r.total_conflicts, 0);
    }

    /// `cleanup_worktrees` runs at the end of every plan run. It must only
    /// touch disposable agent worktrees: a user's own `.claude/worktrees/<name>`
    /// (e.g. an interactive session working in the same repo) and agent
    /// worktrees holding commits that were NOT recovered onto the run branch
    /// must survive — `worktree remove --force` + `branch -D` would destroy
    /// that work.
    #[tokio::test]
    async fn test_cleanup_worktrees_preserves_user_and_unrecovered_worktrees() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |cwd: &std::path::Path, args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(["-c", "user.email=t@t", "-c", "user.name=t"])
                .args(args)
                .current_dir(cwd)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {:?}: {}",
                args,
                String::from_utf8_lossy(&out.stderr)
            );
        };
        git(root, &["init", "-q", "-b", "main"]);
        git(root, &["commit", "-q", "--allow-empty", "-m", "init"]);
        // A user worktree (not an agent one) and an agent worktree with an
        // unrecovered commit, plus a stale agent worktree with no new commit.
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "feat",
                ".claude/worktrees/my-feature",
            ],
        );
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-a",
                ".claude/worktrees/agent-a",
            ],
        );
        git(
            &root.join(".claude/worktrees/agent-a"),
            &["commit", "-q", "--allow-empty", "-m", "agent work"],
        );
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-b",
                ".claude/worktrees/agent-b",
            ],
        );

        let cwd = root.to_string_lossy().to_string();
        let cleaned = WorktreeCollector::cleanup_worktrees(&cwd).await.unwrap();

        assert!(
            root.join(".claude/worktrees/my-feature").exists(),
            "user worktree removed"
        );
        assert!(
            root.join(".claude/worktrees/agent-a").exists(),
            "unrecovered agent work removed"
        );
        assert!(
            !root.join(".claude/worktrees/agent-b").exists(),
            "stale agent worktree kept"
        );
        assert_eq!(cleaned, 1);
    }

    /// Recovered commits are cherry-picked onto the run branch (new SHAs), so a
    /// recovered agent worktree is NOT an ancestor-merge: it must still be
    /// recognised as fully recovered (patch-equivalent) and cleaned up.
    #[tokio::test]
    async fn test_cleanup_worktrees_removes_cherry_picked_agent_worktree() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |cwd: &std::path::Path, args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(["-c", "user.email=t@t", "-c", "user.name=t"])
                .args(args)
                .current_dir(cwd)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {:?}: {}",
                args,
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        git(root, &["init", "-q", "-b", "main"]);
        git(root, &["commit", "-q", "--allow-empty", "-m", "init"]);
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-c",
                ".claude/worktrees/agent-c",
            ],
        );
        let wt = root.join(".claude/worktrees/agent-c");
        std::fs::write(wt.join("f.txt"), "hello").unwrap();
        git(&wt, &["add", "f.txt"]);
        git(&wt, &["commit", "-q", "-m", "agent work"]);
        let sha = git(&wt, &["rev-parse", "HEAD"]);
        // Recover it onto main via cherry-pick (different SHA, same patch).
        git(root, &["cherry-pick", &sha]);

        let cwd = root.to_string_lossy().to_string();
        let cleaned = WorktreeCollector::cleanup_worktrees(&cwd).await.unwrap();

        assert_eq!(cleaned, 1);
        assert!(!wt.exists(), "recovered agent worktree kept");
    }
}
