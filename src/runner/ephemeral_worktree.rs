//! The ephemeral git worktree a delegated task writes in (T5).
//!
//! A delegation asked with `execution_place: "ephemeral_worktree"` does not write in the
//! repository the caller named: PO makes a NEW linked worktree of that repository for it, and
//! only after a person confirmed « exécute dans X » (`chat::place_confirmation`). This module is
//! the git half:
//!
//! - [`plan`] resolves, WITHOUT touching anything, the place X: the repository (the MAIN worktree
//!   of the repository the caller's directory belongs to, canonical), the base (a commit SHA read
//!   now and frozen: `origin/HEAD` when the repository has one, its `HEAD` otherwise), a branch
//!   of its own (`po/task-<task>-<nonce>`) and the path, under PO's application directory
//!   ([`root`]), never inside the repository nor inside any of its worktrees. X is PO's: no text
//!   of a model ever becomes the path.
//! - [`create`] makes exactly that worktree, from exactly that SHA.
//! - [`EphemeralWorktree::release`] removes it when the task is over (success or failure), with
//!   the rule of `gate.sh janitor`, stricter: NOTHING is removed while the worktree has
//!   uncommitted changes (untracked files included), or commits that are neither in the base nor
//!   on any remote-tracking branch (unpushed work), or when either cannot be verified. Such a
//!   worktree and its branch are KEPT and the reason is returned; nothing committed is ever lost.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use uuid::Uuid;

use super::git::WorktreeCollector;

/// Directory, under PO's application directory, that holds the ephemeral worktrees.
pub const DIR_NAME: &str = "task-worktrees";

/// Prefix of the branch of an ephemeral worktree.
pub const BRANCH_PREFIX: &str = "po/task-";

/// Root of every ephemeral worktree: next to the neutral chat directories, outside any
/// repository.
pub fn root() -> PathBuf {
    let neutral = crate::chat::neutral_place::root();
    neutral
        .parent()
        .map(|app| app.join(DIR_NAME))
        .unwrap_or_else(|| {
            std::env::temp_dir()
                .join("project-orchestrator")
                .join(DIR_NAME)
        })
}

/// The place a delegated task would run in, resolved by PO and shown to the person who
/// confirms it. Nothing exists yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedWorktree {
    /// Canonical path of the repository's main worktree.
    pub repository: PathBuf,
    /// What the base was read from: `origin/HEAD` or `HEAD`.
    pub base_ref: String,
    /// The commit the worktree starts at, frozen when the place was planned.
    pub base_sha: String,
    /// The branch made for the worktree.
    pub branch: String,
    /// X: canonical path of the worktree to create.
    pub path: PathBuf,
}

/// Why a place cannot be planned or made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlaceError {
    #[error("execution_place_not_a_repository: {0} is not inside a git repository")]
    NotARepository(String),
    #[error("execution_place_bare_repository: {0} has no working tree to start from")]
    BareRepository(String),
    #[error("execution_place_no_base: no commit to start from in {0}")]
    NoBase(String),
    #[error("execution_place_inside_a_worktree: {0} would be inside an existing worktree")]
    InsideAWorktree(String),
    #[error("execution_place_exists: {0} already exists")]
    AlreadyExists(String),
    #[error("execution_place_git_failed: {0}")]
    Git(String),
}

impl PlaceError {
    /// The stable code at the head of the message.
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotARepository(_) => "execution_place_not_a_repository",
            Self::BareRepository(_) => "execution_place_bare_repository",
            Self::NoBase(_) => "execution_place_no_base",
            Self::InsideAWorktree(_) => "execution_place_inside_a_worktree",
            Self::AlreadyExists(_) => "execution_place_exists",
            Self::Git(_) => "execution_place_git_failed",
        }
    }
}

/// Why an ephemeral worktree was kept instead of removed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum KeepReason {
    /// `git status --porcelain` is not empty (untracked files included).
    UncommittedChanges,
    /// Commits neither in the base nor on any remote-tracking branch.
    UnpushedCommits { count: usize },
    /// The state could not be read: nothing is removed on a guess.
    Unverifiable { error: String },
}

/// What [`EphemeralWorktree::release`] did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ReleaseOutcome {
    /// The worktree and its branch are gone.
    Removed,
    /// Kept, with its branch: removing it could lose work.
    Kept {
        #[serde(flatten)]
        reason: KeepReason,
    },
}

/// A worktree [`create`] made: the only value [`EphemeralWorktree::release`] works on, so only a
/// worktree PO made itself can ever be removed by this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EphemeralWorktree {
    planned: PlannedWorktree,
}

async fn git(cwd: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .await
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(format!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn create_private(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Resolve the place a task delegated from `cwd` would run in, under `root` ([`root`] in
/// production). Reads git, creates `root` if needed, creates nothing else.
pub async fn plan(cwd: &str, task_id: Uuid, root: &Path) -> Result<PlannedWorktree, PlaceError> {
    let dir = Path::new(cwd);
    if !dir.is_dir() {
        return Err(PlaceError::NotARepository(cwd.to_string()));
    }
    git(dir, &["rev-parse", "--git-dir"])
        .await
        .map_err(|_| PlaceError::NotARepository(cwd.to_string()))?;
    // The repository is its MAIN worktree (the first entry), whichever worktree `cwd` is in:
    // the base is the repository's, not a sibling agent's checkout.
    let worktrees = WorktreeCollector::list_worktrees(cwd)
        .await
        .map_err(|e| PlaceError::Git(e.to_string()))?;
    let main = worktrees
        .first()
        .ok_or_else(|| PlaceError::NotARepository(cwd.to_string()))?;
    if main.is_bare {
        return Err(PlaceError::BareRepository(main.path.display().to_string()));
    }
    let repository = canonical(&main.path);

    let (base_ref, base_sha) = match git(
        &repository,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            "refs/remotes/origin/HEAD^{commit}",
        ],
    )
    .await
    {
        Ok(sha) if !sha.is_empty() => ("origin/HEAD".to_string(), sha),
        _ => {
            let sha = git(&repository, &["rev-parse", "--verify", "HEAD^{commit}"])
                .await
                .map_err(|_| PlaceError::NoBase(repository.display().to_string()))?;
            ("HEAD".to_string(), sha)
        }
    };

    create_private(root).map_err(|e| PlaceError::Git(format!("{}: {e}", root.display())))?;
    let root = canonical(root);
    let nonce = Uuid::new_v4().simple().to_string();
    let task = task_id.simple().to_string();
    let name = format!("{}-{}", &task[..8], &nonce[..8]);
    let path = root.join(&name);
    if path.exists() {
        return Err(PlaceError::AlreadyExists(path.display().to_string()));
    }
    // Never inside the repository nor inside any of its worktrees (a config directory that
    // happens to live in a checkout would put X there).
    let nested = worktrees
        .iter()
        .map(|wt| canonical(&wt.path))
        .any(|wt| path.starts_with(&wt));
    if nested {
        return Err(PlaceError::InsideAWorktree(path.display().to_string()));
    }
    Ok(PlannedWorktree {
        repository,
        base_ref,
        base_sha,
        branch: format!("{BRANCH_PREFIX}{name}"),
        path,
    })
}

/// Make the planned worktree: a new branch at the frozen SHA, checked out at X.
pub async fn create(planned: &PlannedWorktree) -> Result<EphemeralWorktree, PlaceError> {
    if planned.path.exists() {
        return Err(PlaceError::AlreadyExists(
            planned.path.display().to_string(),
        ));
    }
    let path = planned.path.to_string_lossy().to_string();
    git(
        &planned.repository,
        &[
            "worktree",
            "add",
            "--quiet",
            "-b",
            &planned.branch,
            &path,
            &planned.base_sha,
        ],
    )
    .await
    .map_err(PlaceError::Git)?;
    Ok(EphemeralWorktree {
        planned: planned.clone(),
    })
}

impl EphemeralWorktree {
    /// The place this worktree was made at.
    pub fn planned(&self) -> &PlannedWorktree {
        &self.planned
    }

    /// X, as the child session's working directory.
    pub fn cwd(&self) -> String {
        self.planned.path.to_string_lossy().to_string()
    }

    /// Commits reachable from the worktree's `HEAD` or its branch that are neither in the base
    /// nor on any remote-tracking branch.
    async fn unpushed(&self) -> Result<usize, String> {
        let branch = format!("refs/heads/{}", self.planned.branch);
        let base = format!("^{}", self.planned.base_sha);
        let mut args = vec!["rev-list", "--count"];
        // The worktree's HEAD covers a detached checkout the agent may have committed on.
        let head = if self.planned.path.is_dir() {
            git(&self.planned.path, &["rev-parse", "--verify", "HEAD"])
                .await
                .ok()
        } else {
            None
        };
        if let Some(head) = head.as_deref() {
            args.push(head);
        }
        let branch_exists = git(
            &self.planned.repository,
            &["rev-parse", "--verify", "--quiet", &branch],
        )
        .await
        .is_ok();
        if branch_exists {
            args.push(&branch);
        }
        if head.is_none() && !branch_exists {
            return Ok(0);
        }
        args.push(&base);
        args.push("--not");
        args.push("--remotes");
        let count = git(&self.planned.repository, &args).await?;
        count
            .parse::<usize>()
            .map_err(|e| format!("rev-list --count: {e}"))
    }

    /// Remove the worktree and its branch, unless that could lose work (see the module).
    pub async fn release(&self) -> ReleaseOutcome {
        let kept = |reason| ReleaseOutcome::Kept { reason };
        if self.planned.path.is_dir() {
            match git(&self.planned.path, &["status", "--porcelain"]).await {
                Ok(status) if !status.is_empty() => {
                    return kept(KeepReason::UncommittedChanges);
                }
                Ok(_) => {}
                Err(error) => return kept(KeepReason::Unverifiable { error }),
            }
        }
        match self.unpushed().await {
            Ok(0) => {}
            Ok(count) => return kept(KeepReason::UnpushedCommits { count }),
            Err(error) => return kept(KeepReason::Unverifiable { error }),
        }
        let repo = &self.planned.repository;
        if self.planned.path.is_dir() {
            // No `--force`: git itself refuses a dirty or locked worktree, a second guard.
            let path = self.planned.path.to_string_lossy().to_string();
            if let Err(error) = git(repo, &["worktree", "remove", &path]).await {
                return kept(KeepReason::Unverifiable { error });
            }
        } else {
            let _ = git(repo, &["worktree", "prune"]).await;
        }
        // Only the branch PO made, and only now that nothing on it is unpushed.
        let _ = git(repo, &["branch", "-D", &self.planned.branch]).await;
        ReleaseOutcome::Removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn run(cwd: &Path, args: &[&str]) -> String {
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
    }

    /// A temporary repository with one commit, and a separate temporary root for the
    /// ephemeral worktrees. Never the real repository.
    fn repo() -> (tempfile::TempDir, tempfile::TempDir) {
        let repo = tempfile::tempdir().unwrap();
        run(repo.path(), &["init", "-q", "-b", "main"]);
        run(
            repo.path(),
            &["commit", "-q", "--allow-empty", "-m", "init"],
        );
        (repo, tempfile::tempdir().unwrap())
    }

    async fn made(repo: &Path, root: &Path) -> EphemeralWorktree {
        let planned = plan(&repo.to_string_lossy(), Uuid::new_v4(), root)
            .await
            .unwrap();
        create(&planned).await.unwrap()
    }

    #[tokio::test]
    async fn the_place_is_planned_from_the_main_worktree_at_a_frozen_sha_outside_any_worktree() {
        let (repo, root) = repo();
        // Asked from a sibling worktree: the base is still the repository's.
        run(
            repo.path(),
            &["worktree", "add", "-q", "-b", "side", "side"],
        );
        run(
            &repo.path().join("side"),
            &["commit", "-q", "--allow-empty", "-m", "side work"],
        );
        let head = run(repo.path(), &["rev-parse", "HEAD"]);
        let task = Uuid::new_v4();
        let planned = plan(
            &repo.path().join("side").to_string_lossy(),
            task,
            root.path(),
        )
        .await
        .unwrap();
        assert_eq!(planned.repository, canonical(repo.path()));
        assert_eq!(planned.base_ref, "HEAD");
        assert_eq!(
            planned.base_sha, head,
            "the main worktree's HEAD, not the sibling's"
        );
        assert!(planned.path.starts_with(canonical(root.path())));
        assert!(!planned.path.starts_with(canonical(repo.path())));
        assert!(planned.branch.starts_with(BRANCH_PREFIX));
        assert!(!planned.path.exists(), "planning creates nothing");

        let wt = create(&planned).await.unwrap();
        assert_eq!(run(&planned.path, &["rev-parse", "HEAD"]), head);
        assert_eq!(
            run(&planned.path, &["branch", "--show-current"]),
            planned.branch
        );
        assert_eq!(wt.release().await, ReleaseOutcome::Removed);
    }

    #[tokio::test]
    async fn origin_head_is_the_base_when_the_repository_has_one() {
        let (origin, root) = repo();
        let clone_parent = tempfile::tempdir().unwrap();
        run(
            clone_parent.path(),
            &["clone", "-q", &origin.path().to_string_lossy(), "c"],
        );
        let clone = clone_parent.path().join("c");
        // A local, unpushed commit on the clone's HEAD must not be the base.
        run(&clone, &["commit", "-q", "--allow-empty", "-m", "local"]);
        let origin_head = run(origin.path(), &["rev-parse", "HEAD"]);
        let planned = plan(&clone.to_string_lossy(), Uuid::new_v4(), root.path())
            .await
            .unwrap();
        assert_eq!(planned.base_ref, "origin/HEAD");
        assert_eq!(planned.base_sha, origin_head);
    }

    #[tokio::test]
    async fn a_directory_outside_git_has_no_place() {
        let dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let err = plan(&dir.path().to_string_lossy(), Uuid::new_v4(), root.path())
            .await
            .unwrap_err();
        assert_eq!(err.code(), "execution_place_not_a_repository");
    }

    #[tokio::test]
    async fn a_root_inside_the_repository_is_refused() {
        let (repo, _root) = repo();
        let inside = repo.path().join("nested-root");
        let err = plan(&repo.path().to_string_lossy(), Uuid::new_v4(), &inside)
            .await
            .unwrap_err();
        assert_eq!(err.code(), "execution_place_inside_a_worktree");
    }

    #[tokio::test]
    async fn a_clean_worktree_is_removed_with_its_branch() {
        let (repo, root) = repo();
        let wt = made(repo.path(), root.path()).await;
        let path = wt.planned().path.clone();
        assert!(path.is_dir());
        assert_eq!(wt.release().await, ReleaseOutcome::Removed);
        assert!(!path.exists(), "the clean worktree must be removed");
        let branches = run(repo.path(), &["branch", "--list", &wt.planned().branch]);
        assert!(
            branches.is_empty(),
            "its branch must be deleted: {branches}"
        );
    }

    #[tokio::test]
    async fn a_worktree_with_unpushed_commits_is_kept_with_its_branch() {
        let (repo, root) = repo();
        let wt = made(repo.path(), root.path()).await;
        let path = wt.planned().path.clone();
        std::fs::write(path.join("work.txt"), "done").unwrap();
        run(&path, &["add", "work.txt"]);
        run(&path, &["commit", "-q", "-m", "agent work"]);
        assert_eq!(
            wt.release().await,
            ReleaseOutcome::Kept {
                reason: KeepReason::UnpushedCommits { count: 1 }
            }
        );
        assert!(
            path.join("work.txt").exists(),
            "committed work must survive"
        );
        let branches = run(repo.path(), &["branch", "--list", &wt.planned().branch]);
        assert!(!branches.is_empty(), "the branch must survive");
    }

    #[tokio::test]
    async fn a_detached_commit_in_the_worktree_is_unpushed_work_too() {
        let (repo, root) = repo();
        let wt = made(repo.path(), root.path()).await;
        let path = wt.planned().path.clone();
        run(&path, &["checkout", "-q", "--detach"]);
        run(
            &path,
            &["commit", "-q", "--allow-empty", "-m", "detached work"],
        );
        assert!(matches!(
            wt.release().await,
            ReleaseOutcome::Kept {
                reason: KeepReason::UnpushedCommits { count: 1 }
            }
        ));
        assert!(path.exists());
    }

    #[tokio::test]
    async fn a_worktree_with_uncommitted_or_untracked_files_is_kept() {
        let (repo, root) = repo();
        let wt = made(repo.path(), root.path()).await;
        let path = wt.planned().path.clone();
        std::fs::write(path.join("draft.txt"), "wip").unwrap();
        assert_eq!(
            wt.release().await,
            ReleaseOutcome::Kept {
                reason: KeepReason::UncommittedChanges
            }
        );
        assert!(path.join("draft.txt").exists());
    }

    #[tokio::test]
    async fn pushed_commits_do_not_keep_the_worktree() {
        let (origin, root) = repo();
        let clone_parent = tempfile::tempdir().unwrap();
        run(
            clone_parent.path(),
            &["clone", "-q", &origin.path().to_string_lossy(), "c"],
        );
        let clone = clone_parent.path().join("c");
        let wt = made(&clone, root.path()).await;
        let path = wt.planned().path.clone();
        run(
            &path,
            &["commit", "-q", "--allow-empty", "-m", "pushed work"],
        );
        run(
            &path,
            &[
                "push",
                "-q",
                "origin",
                &format!("HEAD:refs/heads/{}", wt.planned().branch),
            ],
        );
        assert_eq!(wt.release().await, ReleaseOutcome::Removed);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn a_worktree_already_gone_is_pruned_without_losing_its_commits() {
        let (repo, root) = repo();
        let wt = made(repo.path(), root.path()).await;
        let path = wt.planned().path.clone();
        run(&path, &["commit", "-q", "--allow-empty", "-m", "work"]);
        std::fs::remove_dir_all(&path).unwrap();
        // The branch still holds a commit nobody pushed: kept.
        assert!(matches!(
            wt.release().await,
            ReleaseOutcome::Kept {
                reason: KeepReason::UnpushedCommits { count: 1 }
            }
        ));
        let branches = run(repo.path(), &["branch", "--list", &wt.planned().branch]);
        assert!(!branches.is_empty());
    }
}
