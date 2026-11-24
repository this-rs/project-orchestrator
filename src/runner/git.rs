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
#[allow(unused_imports)]
use crate::neo4j::traits::*;

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
    /// The collector stashed the uncommitted changes of `cwd` before recovering
    /// commits and could NOT restore them: what went wrong, and the name of the
    /// stash entry that still holds them (#1299). `None`: nothing was stashed,
    /// or it was restored.
    pub unpopped_stash: Option<String>,
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

    /// List commit SHAs present in `worktree_branch` that still need to be
    /// cherry-picked onto `run_branch`, oldest first.
    ///
    /// `git log <worktree_branch> --not <run_branch>` gives the commits that are
    /// not ANCESTORS of the run branch. A recovered commit is a cherry-pick (a
    /// new SHA), so a worktree that is kept after its recovery (#1299: another
    /// agent still works in it) would list the same commits again at every
    /// sweep; cherry-picking them a second time gives an empty pick, which was
    /// read as a conflict and stopped the recovery of every commit behind it.
    /// The list is therefore narrowed to the commits with no patch-equivalent
    /// on the run branch (`git cherry`, as `cleanup_worktrees` already judges).
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

        if commits.is_empty() {
            return Ok(commits);
        }

        // `+ <sha>`: no patch-equivalent commit on the run branch; `- <sha>`:
        // already there under another SHA. When `git cherry` itself fails, the
        // ancestry list stands, as before.
        let cherry = Command::new("git")
            .args(["cherry", run_branch, worktree_branch])
            .current_dir(cwd)
            .output()
            .await
            .context("Failed to run git cherry")?;
        if !cherry.status.success() {
            debug!(
                "git cherry {} {} failed, recovering every non-ancestor commit: {}",
                run_branch,
                worktree_branch,
                String::from_utf8_lossy(&cherry.stderr).trim()
            );
            return Ok(commits);
        }
        let unrecovered: std::collections::HashSet<String> =
            String::from_utf8_lossy(&cherry.stdout)
                .lines()
                .filter_map(|l| l.strip_prefix("+ "))
                .map(|sha| sha.trim().to_string())
                .collect();
        let already = commits.len();
        let commits: Vec<String> = commits
            .into_iter()
            .filter(|sha| unrecovered.contains(sha))
            .collect();
        if commits.len() < already {
            debug!(
                "{} commit(s) of {} already recovered onto {} (patch-equivalent), skipped",
                already - commits.len(),
                worktree_branch,
                run_branch
            );
        }

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
        let stashed = match stash_if_needed(cwd).await {
            Ok(stashed) => stashed,
            Err(e) => {
                warn!("Could not stash the uncommitted changes of {}: {}", cwd, e);
                None
            }
        };

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
                // No commit to recover: clean the worktree up IF it is really
                // stale — one with uncommitted work in it is kept (#1299).
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

        // Restore the stash entry this sweep made (by its nonce, never the top
        // of the shared stack). A failure is carried, not swallowed.
        resolution.unpopped_stash = unstash_if_needed(cwd, stashed).await;

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
    /// the current branch is kept as well, and so is one with uncommitted work
    /// in it (`cleanup_single_worktree`, #1299): removing it and deleting its
    /// branch would destroy that work. Returns the number of removed worktrees.
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
    ///
    /// A worktree with anything uncommitted in it (tracked changes or untracked
    /// files) is KEPT. "No new commit" does not mean "stale": it is also the
    /// state of a worktree in which another agent is still working and has not
    /// committed yet (#1299). The check fails closed: if git cannot say the
    /// worktree is clean, it stays. Removal is not forced, so git's own refusal
    /// of a dirty or locked worktree stands behind the check — with one
    /// exception: a worktree that holds submodules is refused by git even when
    /// clean, and is removed with `--force` once the check has passed.
    ///
    /// A worktree whose directory is gone (`prunable`) has nothing left to
    /// protect: it is removed without the check, which could not run in it.
    async fn cleanup_single_worktree(cwd: &str, wt: &WorktreeInfo) -> bool {
        let wt_path = wt.path.to_string_lossy();

        let gone = wt.is_prunable || !wt.path.exists();
        let mut force = false;
        if gone {
            debug!(
                "Worktree {} has no directory any more: removing its entry",
                wt_path
            );
        } else {
            match worktree_is_clean(&wt_path).await {
                Ok(true) => {}
                Ok(false) => {
                    warn!(
                        "Keeping worktree {}: it has uncommitted changes (another agent may still be working in it)",
                        wt_path
                    );
                    return false;
                }
                Err(e) => {
                    warn!(
                        "Keeping worktree {}: cannot verify it is clean: {}",
                        wt_path, e
                    );
                    return false;
                }
            }
            // `git worktree remove` refuses a worktree with submodules unless
            // forced, clean or not (man git-worktree). The check above has
            // passed, so forcing here only bypasses that refusal.
            force = wt.path.join(".gitmodules").exists();
        }

        let mut args = vec!["worktree", "remove", &*wt_path];
        if force {
            args.push("--force");
        }
        let output = Command::new("git")
            .args(&args)
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

/// Whether the worktree at `path` has nothing uncommitted: no tracked change
/// and no untracked file.
///
/// Unlike [`has_uncommitted_changes`], this fails CLOSED: when git itself
/// fails, the answer is an error, never "clean". It guards a deletion, so a
/// doubt must keep the worktree.
async fn worktree_is_clean(path: &str) -> Result<bool> {
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(path)
        .output()
        .await
        .context("Failed to run git status")?;

    if !output.status.success() {
        anyhow::bail!(
            "git status failed in {}: {}",
            path,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().is_empty())
}

/// The marker every stash entry this collector makes carries in its message,
/// followed by a nonce that names one sweep.
const STASH_MARKER: &str = "runner-worktree-collector";

/// The `stash@{n}` reference of the entry whose message carries `nonce`, if it
/// is still on the stack. The stack is shared by every worktree of the
/// repository and other processes push and pop on it: an entry is found by
/// what only it carries, never by its position.
async fn stash_entry_with(cwd: &str, nonce: &str) -> Result<Option<String>> {
    let output = Command::new("git")
        .args(["stash", "list", "--format=%gd %s"])
        .current_dir(cwd)
        .output()
        .await
        .context("Failed to run git stash list")?;
    if !output.status.success() {
        anyhow::bail!(
            "git stash list failed in {}: {}",
            cwd,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| {
            let (reference, subject) = line.split_once(' ')?;
            subject.contains(nonce).then(|| reference.to_string())
        }))
}

/// Stash the uncommitted changes of `cwd` if there are any.
///
/// Returns the nonce of the entry that was really created, or `None`. `git
/// stash push` leaves untracked files alone and, when nothing else changed,
/// answers "No local changes to save" with exit code 0: believing that a stash
/// was made would make the later pop restore an entry that belongs to someone
/// else (#1299). The entry is therefore looked up by its own nonce after the
/// push, and popped by it later.
async fn stash_if_needed(cwd: &str) -> Result<Option<String>> {
    if !has_uncommitted_changes(cwd).await? {
        return Ok(None);
    }
    let nonce = format!("{STASH_MARKER} {}", uuid::Uuid::new_v4());
    let output = Command::new("git")
        .args(["stash", "push", "-m", &nonce])
        .current_dir(cwd)
        .output()
        .await
        .context("Failed to git stash")?;
    if !output.status.success() {
        warn!(
            "`git stash push` failed in {}: {}",
            cwd,
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return Ok(None);
    }
    match stash_entry_with(cwd, &nonce).await? {
        Some(_) => {
            info!("Stashed uncommitted changes before worktree collection ({nonce})");
            Ok(Some(nonce))
        }
        None => {
            debug!("Nothing to stash (only untracked files): no stash entry created");
            Ok(None)
        }
    }
}

/// Pop the stash entry `stash_if_needed` made, found by its nonce. Returns what
/// went wrong when the changes could not be restored — logged AND handed back,
/// so the caller carries it (`WorktreeResolution::unpopped_stash`) instead of
/// the person discovering an emptied working directory. On a conflict git
/// keeps the entry: the message names it.
async fn unstash_if_needed(cwd: &str, stashed: Option<String>) -> Option<String> {
    let nonce = stashed?;
    let entry = match stash_entry_with(cwd, &nonce).await {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            let what = format!(
                "the stash entry '{nonce}' made in {cwd} is no longer on the stack: another process popped or dropped it"
            );
            warn!("{what}");
            return Some(what);
        }
        Err(e) => {
            let what = format!(
                "could not look the stash entry '{nonce}' up in {cwd}: {e} — the changes stay in it"
            );
            warn!("{what}");
            return Some(what);
        }
    };
    match Command::new("git")
        .args(["stash", "pop", &entry])
        .current_dir(cwd)
        .output()
        .await
    {
        Ok(o) if o.status.success() => None,
        Ok(o) => {
            let what = format!(
                "`git stash pop {entry}` failed in {cwd}: {} — the changes stay in the stash entry '{nonce}'",
                String::from_utf8_lossy(&o.stderr).trim()
            );
            warn!("{what}");
            Some(what)
        }
        Err(e) => {
            let what = format!(
                "could not run `git stash pop {entry}` in {cwd}: {e} — the changes stay in the stash entry '{nonce}'"
            );
            warn!("{what}");
            Some(what)
        }
    }
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
    /// See [`WorktreeResolution::unpopped_stash`].
    pub unpopped_stash: Option<String>,
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
        unpopped_stash: resolution.unpopped_stash.clone(),
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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

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
            unpopped_stash: None,
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

    /// `resolve_worktrees` cherry-picks every registered agent worktree's
    /// commits onto `branch`. It must therefore NEVER run against the real
    /// repository the tests are launched from: on a machine where an agent
    /// worktree (`.claude/worktrees/agent-*`) is registered, the old version of
    /// this test rewrote the developer's current branch with that agent's
    /// commits (11/10/2026: ten foreign commits reached a pushed branch). It
    /// now runs in a throwaway repository without agent worktrees.
    #[tokio::test]
    async fn test_resolve_worktrees_no_agents() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for args in [
            &["init", "-q", "-b", "main"][..],
            &["commit", "-q", "--allow-empty", "-m", "init"][..],
        ] {
            let out = std::process::Command::new("git")
                .args(["-c", "user.email=t@t", "-c", "user.name=t"])
                .args(args)
                .current_dir(root)
                .output()
                .unwrap();
            assert!(out.status.success(), "git {:?}", args);
        }
        let cwd = root.to_string_lossy().to_string();
        let head_before = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(root)
            .output()
            .unwrap()
            .stdout;
        let lock = Arc::new(Mutex::new(()));
        let resolution = WorktreeCollector::resolve_worktrees("main", &cwd, lock).await;
        assert!(resolution.is_ok());
        let r = resolution.unwrap();
        assert_eq!(r.total_merged, 0);
        assert_eq!(r.total_conflicts, 0);
        let head_after = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(root)
            .output()
            .unwrap()
            .stdout;
        assert_eq!(head_before, head_after, "the branch must not move");
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

    /// Throwaway repository with one tracked file, for the #1299 tests.
    fn sweep_repo() -> (
        tempfile::TempDir,
        impl Fn(&std::path::Path, &[&str]) -> String,
    ) {
        let dir = tempfile::tempdir().unwrap();
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
        git(dir.path(), &["init", "-q", "-b", "main"]);
        // The code under test runs `git cherry-pick` and `git stash` in this
        // repository with no `-c user.*`: give the repository an identity, so
        // the tests do not depend on the machine's global config.
        git(dir.path(), &["config", "user.email", "t@t"]);
        git(dir.path(), &["config", "user.name", "t"]);
        std::fs::write(dir.path().join("f.txt"), "base").unwrap();
        git(dir.path(), &["add", "f.txt"]);
        git(dir.path(), &["commit", "-q", "-m", "init"]);
        (dir, git)
    }

    /// Review of #1309 (R1) — a worktree whose directory is gone (`prunable`)
    /// has nothing left to protect. The clean check cannot run in a directory
    /// that does not exist, and failing closed there would keep the entry and
    /// its branch forever, re-warning at every sweep.
    #[tokio::test]
    async fn test_cleanup_worktrees_removes_a_prunable_worktree() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-gone",
                ".claude/worktrees/agent-gone",
            ],
        );
        std::fs::remove_dir_all(root.join(".claude/worktrees/agent-gone")).unwrap();

        let cwd = root.to_string_lossy().to_string();
        let cleaned = WorktreeCollector::cleanup_worktrees(&cwd).await.unwrap();

        assert_eq!(cleaned, 1, "the prunable worktree was kept");
        assert!(
            !git(root, &["worktree", "list", "--porcelain"]).contains("agent-gone"),
            "the prunable entry is still registered"
        );
        assert!(
            git(root, &["branch", "--list", "agent-gone"]).is_empty(),
            "the prunable worktree's branch was kept"
        );
    }

    /// Review of #1309 (R2) — `git worktree remove` refuses a worktree that
    /// holds submodules unless forced, clean or not. Without `--force` for that
    /// case, no agent worktree of a repository with submodules could ever be
    /// cleaned up again.
    #[tokio::test]
    async fn test_cleanup_worktrees_removes_a_clean_worktree_with_submodules() {
        let (sub, _) = sweep_repo();
        let (dir, git) = sweep_repo();
        let root = dir.path();
        git(
            root,
            &[
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "-q",
                &sub.path().to_string_lossy(),
                "sub",
            ],
        );
        git(root, &["commit", "-q", "-m", "submodule"]);
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-sub",
                ".claude/worktrees/agent-sub",
            ],
        );
        let wt = root.join(".claude/worktrees/agent-sub");
        assert!(wt.join(".gitmodules").exists());

        let cwd = root.to_string_lossy().to_string();
        let cleaned = WorktreeCollector::cleanup_worktrees(&cwd).await.unwrap();

        assert_eq!(cleaned, 1, "the clean worktree with submodules was kept");
        assert!(!wt.exists());
    }

    /// Review of #1309 (R3) — a kept worktree (dirty neighbour) whose commit was
    /// already recovered must not be read as a conflict at the next sweep: a
    /// recovered commit is a cherry-pick (new SHA), so ancestry lists it again,
    /// and an empty pick stopped the recovery of every commit behind it.
    #[tokio::test]
    async fn test_resolve_worktrees_recovers_later_commits_of_a_kept_worktree() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-n",
                ".claude/worktrees/agent-n",
            ],
        );
        let wt = root.join(".claude/worktrees/agent-n");
        std::fs::write(wt.join("a.txt"), "A").unwrap();
        git(&wt, &["add", "a.txt"]);
        git(&wt, &["commit", "-q", "-m", "A"]);
        // The neighbour keeps working: untracked, uncommitted.
        std::fs::write(wt.join("wip.txt"), "not yet").unwrap();

        let cwd = root.to_string_lossy().to_string();
        let lock = Arc::new(Mutex::new(()));
        let first = WorktreeCollector::resolve_worktrees("main", &cwd, lock.clone())
            .await
            .unwrap();
        assert_eq!((first.total_merged, first.total_conflicts), (1, 0));
        assert!(root.join("a.txt").exists(), "A was not recovered onto main");
        assert!(wt.exists(), "the dirty neighbour was removed");

        // The neighbour commits again; the earlier commit is already on main.
        std::fs::write(wt.join("c.txt"), "C").unwrap();
        git(&wt, &["add", "c.txt"]);
        git(&wt, &["commit", "-q", "-m", "C"]);
        let second = WorktreeCollector::resolve_worktrees("main", &cwd, lock)
            .await
            .unwrap();
        assert_eq!(
            (second.total_merged, second.total_conflicts),
            (1, 0),
            "the already-recovered commit was read as a conflict: {:?}",
            second.details
        );
        assert!(
            root.join("c.txt").exists(),
            "the later commit was not recovered onto main"
        );
    }

    /// Review of #1309 (R4) — the stash stack is shared and other processes
    /// push on it while a sweep runs: the entry popped must be the one THIS
    /// sweep made, found by its nonce, never the top of the stack.
    #[tokio::test]
    async fn test_unstash_pops_its_own_entry_not_the_top_of_the_stack() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        std::fs::write(root.join("f.txt"), "ours").unwrap();

        let cwd = root.to_string_lossy().to_string();
        let ours = stash_if_needed(&cwd)
            .await
            .unwrap()
            .expect("a tracked change must be stashed");
        assert_eq!(std::fs::read_to_string(root.join("f.txt")).unwrap(), "base");
        // Someone else pushes on top while the sweep is recovering commits.
        std::fs::write(root.join("f.txt"), "theirs").unwrap();
        git(root, &["stash", "push", "-q", "-m", "not-ours"]);

        let failure = unstash_if_needed(&cwd, Some(ours)).await;

        assert_eq!(failure, None, "{failure:?}");
        assert_eq!(
            std::fs::read_to_string(root.join("f.txt")).unwrap(),
            "ours",
            "the top of the stack was popped instead of our entry"
        );
        let list = git(root, &["stash", "list"]);
        assert!(
            list.contains("not-ours"),
            "someone else's entry is gone: {list}"
        );
        assert!(
            !list.contains(STASH_MARKER),
            "our entry is still on the stack: {list}"
        );
    }

    /// #1299 — a delegation that ends NORMALLY sweeps every agent worktree of
    /// the repository, not only its own. A worktree with no new commit is not
    /// "stale": it is also the state of one where another agent is still working
    /// and has not committed yet. Its modified and untracked files, and its
    /// branch, must survive; a clean worktree with no commit is still cleaned.
    #[tokio::test]
    async fn test_resolve_worktrees_spares_a_neighbour_with_uncommitted_work() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        for (branch, path) in [
            ("agent-busy", ".claude/worktrees/agent-busy"),
            ("agent-idle", ".claude/worktrees/agent-idle"),
        ] {
            git(root, &["worktree", "add", "-q", "-b", branch, path]);
        }
        let busy = root.join(".claude/worktrees/agent-busy");
        std::fs::write(busy.join("f.txt"), "edited, not committed").unwrap();
        std::fs::write(busy.join("new.txt"), "untracked, not committed").unwrap();

        let cwd = root.to_string_lossy().to_string();
        let lock = Arc::new(Mutex::new(()));
        WorktreeCollector::resolve_worktrees("main", &cwd, lock)
            .await
            .unwrap();

        assert!(busy.exists(), "a neighbour's worktree was removed");
        assert_eq!(
            std::fs::read_to_string(busy.join("f.txt")).unwrap(),
            "edited, not committed",
            "a neighbour's tracked edit was lost"
        );
        assert_eq!(
            std::fs::read_to_string(busy.join("new.txt")).unwrap(),
            "untracked, not committed",
            "a neighbour's untracked file was lost"
        );
        assert!(
            git(root, &["branch", "--list", "agent-busy"]).contains("agent-busy"),
            "a neighbour's branch was deleted"
        );
        assert!(
            !root.join(".claude/worktrees/agent-idle").exists(),
            "a clean worktree with no commit must still be cleaned up"
        );
    }

    /// The same guard covers the end-of-run sweep.
    #[tokio::test]
    async fn test_cleanup_worktrees_spares_a_worktree_with_uncommitted_work() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-busy",
                ".claude/worktrees/agent-busy",
            ],
        );
        let busy = root.join(".claude/worktrees/agent-busy");
        std::fs::write(busy.join("new.txt"), "untracked, not committed").unwrap();

        let cwd = root.to_string_lossy().to_string();
        let cleaned = WorktreeCollector::cleanup_worktrees(&cwd).await.unwrap();

        assert_eq!(cleaned, 0);
        assert!(busy.join("new.txt").exists(), "uncommitted work was lost");
    }

    /// #1299 (point 3) — `git stash push` leaves untracked files alone and, when
    /// nothing else changed, says "No local changes to save" with exit code 0.
    /// Believing a stash was made, the sweep popped the top entry of the stash
    /// stack, which is shared by every worktree: someone else's stash was
    /// restored into the working directory. It must be left where it is.
    #[tokio::test]
    async fn test_resolve_worktrees_does_not_pop_a_stash_it_did_not_create() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        // Somebody else's stash entry.
        std::fs::write(root.join("f.txt"), "someone else's change").unwrap();
        git(root, &["stash", "push", "-q", "-m", "not-ours"]);
        // The checkout now only has an untracked file, which stash ignores.
        std::fs::write(root.join("scratch.txt"), "untracked").unwrap();
        git(
            root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "agent-x",
                ".claude/worktrees/agent-x",
            ],
        );

        let cwd = root.to_string_lossy().to_string();
        let lock = Arc::new(Mutex::new(()));
        WorktreeCollector::resolve_worktrees("main", &cwd, lock)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(root.join("f.txt")).unwrap(),
            "base",
            "someone else's stash entry was popped into the checkout"
        );
        assert!(
            git(root, &["stash", "list"]).contains("not-ours"),
            "someone else's stash entry is gone"
        );
    }


    /// A clean checkout has nothing to stash: `stash_if_needed` returns `None`
    /// without creating an entry on the shared stash stack.
    #[tokio::test]
    async fn test_stash_if_needed_returns_none_when_checkout_is_clean() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        let cwd = root.to_string_lossy().to_string();

        let stashed = stash_if_needed(&cwd).await.unwrap();

        assert!(stashed.is_none());
        assert!(git(root, &["stash", "list"]).is_empty());
    }

    /// When `git stash push` cannot write its ref, the sweep must not pretend a
    /// stash was made: it returns `None` and leaves the working tree untouched.
    #[tokio::test]
    async fn test_stash_if_needed_returns_none_when_push_fails() {
        let (dir, _git) = sweep_repo();
        let root = dir.path();
        std::fs::write(root.join("f.txt"), "ours").unwrap();
        // A non-empty directory in place of `refs/stash` makes the ref write
        // fail, while `git status --porcelain` still reports the tracked edit.
        let stash_ref = root.join(".git/refs/stash");
        std::fs::create_dir_all(&stash_ref).unwrap();
        std::fs::write(stash_ref.join("blocker"), "blocked").unwrap();

        let cwd = root.to_string_lossy().to_string();
        let stashed = stash_if_needed(&cwd).await.unwrap();

        assert!(stashed.is_none());
        assert_eq!(std::fs::read_to_string(root.join("f.txt")).unwrap(), "ours");
    }

    /// A nonce that is no longer on the stack must be reported, not silently
    /// ignored: the caller learns the entry is gone so it can carry the loss.
    #[tokio::test]
    async fn test_unstash_reports_an_entry_no_longer_on_the_stack() {
        let (dir, _git) = sweep_repo();
        let cwd = dir.path().to_string_lossy().to_string();

        let what = unstash_if_needed(&cwd, Some("absent-nonce".to_string())).await;
        let what = what.expect("an absent entry must be reported");

        assert!(what.contains("no longer on the stack"), "{what}");
    }

    /// In a directory that is not a git repository, the stash entry cannot even
    /// be looked up: the sweep says so and reports it, rather than failing open.
    #[tokio::test]
    async fn test_unstash_reports_a_lookup_failure_outside_a_repository() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_string_lossy().to_string();

        let what = unstash_if_needed(&cwd, Some("some-nonce".to_string())).await;
        let what = what.expect("a failed lookup must be reported");

        assert!(what.contains("could not look"), "{what}");
    }

    /// On a conflict, `git stash pop` keeps the entry: the sweep reports the
    /// failure and the changes stay in the named stash entry.
    #[tokio::test]
    async fn test_unstash_reports_a_pop_conflict_and_keeps_the_entry() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        std::fs::write(root.join("f.txt"), "ours").unwrap();

        let cwd = root.to_string_lossy().to_string();
        let nonce = stash_if_needed(&cwd)
            .await
            .unwrap()
            .expect("a tracked change must be stashed");
        assert_eq!(std::fs::read_to_string(root.join("f.txt")).unwrap(), "base");
        // Commit a different content so the pop conflicts.
        std::fs::write(root.join("f.txt"), "theirs").unwrap();
        git(root, &["commit", "-q", "-am", "theirs"]);

        let what = unstash_if_needed(&cwd, Some(nonce.clone())).await;
        let what = what.expect("a conflicting pop must be reported");

        assert!(what.contains("failed"), "{what}");
        assert!(git(root, &["stash", "list"]).contains(&nonce));
    }

    /// `worktree_is_clean` fails CLOSED: git mute means an error, never "clean".
    #[tokio::test]
    async fn test_worktree_is_clean_fails_closed_outside_a_repository() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_string_lossy().to_string();

        let result = worktree_is_clean(&cwd).await;

        assert!(result.is_err());
    }

    /// A worktree whose git status cannot be run is KEPT: the sweep warns and
    /// returns false instead of deleting something it could not verify.
    #[tokio::test]
    async fn test_cleanup_keeps_a_worktree_it_cannot_verify() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().to_string_lossy().to_string();
        let wt = WorktreeInfo {
            path: dir.path().to_path_buf(),
            ..Default::default()
        };

        let kept = WorktreeCollector::cleanup_single_worktree(&cwd, &wt).await;

        assert!(!kept);
        assert!(dir.path().exists());
    }

    /// A commit whose patch already landed on the run branch (under another
    /// SHA) is skipped, not reported as a new commit to recover.
    #[tokio::test]
    async fn test_collect_commits_skips_patch_equivalent_commits() {
        let (dir, git) = sweep_repo();
        let root = dir.path();
        let cwd = root.to_string_lossy().to_string();
        git(root, &["checkout", "-q", "-b", "agent-x"]);
        std::fs::write(root.join("f.txt"), "agent work").unwrap();
        git(root, &["commit", "-q", "-am", "agent work"]);
        let agent_sha = git(root, &["rev-parse", "HEAD"]);
        // The same patch lands on `main` under a different SHA.
        git(root, &["checkout", "-q", "main"]);
        git(root, &["cherry-pick", &agent_sha]);

        let commits = WorktreeCollector::collect_commits("agent-x", "main", &cwd)
            .await
            .unwrap();

        assert!(commits.is_empty(), "{commits:?}");
    }

    /// The backward-compat shim surfaces the unpopped-stash message: a sweep
    /// that could not restore its stash reports it through the returned struct.
    #[tokio::test]
    async fn test_collect_worktree_commits_returns_empty_without_agent_worktrees() {
        let (dir, _git) = sweep_repo();
        let cwd = dir.path().to_string_lossy().to_string();

        let result = collect_worktree_commits(&cwd, "main").await.unwrap();

        assert!(result.recovered_commits.is_empty());
        assert!(result.conflicts.is_empty());
        assert!(result.cleaned_up.is_empty());
        assert!(result.unpopped_stash.is_none());
    }
}
