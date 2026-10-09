//! The neutral working directory of a chat session that was given none.
//!
//! A conversation that is not about one project (a question about the whole
//! workspace, a chat opened from a client that has no folder) used to need a
//! made-up `cwd`. Now the caller may leave it out; the HOST then makes an empty
//! directory of its own for that session, and the session is marked
//! `execution_place: "neutral"`.
//!
//! The directory:
//! - lives under the application directory (`<config dir>/project-orchestrator/
//!   chat-neutral/<session id>`), never inside a repository, and holds no
//!   `.claude/` and no configuration file;
//! - is DETERMINISTIC from the session id, so a resume finds the same path again
//!   (Claude Code keys its transcripts on the cwd) and simply recreates the
//!   directory when it is gone;
//! - is removed when the session is closed, and the ones a crash left behind are
//!   swept at start-up once they are older than [`ORPHAN_MAX_AGE`].
//!
//! It says WHERE the session runs. No tool permission is derived from it.

use std::path::{Path, PathBuf};
use std::time::Duration;
use uuid::Uuid;

pub use crate::neo4j::models::ExecutionPlace;

/// A directory not touched for this long, with no live session to keep it, is an orphan.
pub const ORPHAN_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);

const DIR_NAME: &str = "chat-neutral";

/// Root of every neutral directory.
pub fn root() -> PathBuf {
    let app = match dirs::config_dir() {
        #[cfg(target_os = "windows")]
        Some(c) => c.join("ProjectOrchestrator"),
        #[cfg(not(target_os = "windows"))]
        Some(c) => c.join("project-orchestrator"),
        None => dirs::home_dir()
            .map(|h| h.join(".project-orchestrator"))
            .unwrap_or_else(|| std::env::temp_dir().join("project-orchestrator")),
    };
    app.join(DIR_NAME)
}

/// The neutral directory of a session (not created).
pub fn dir_for(session_id: Uuid) -> PathBuf {
    dir_under(&root(), session_id)
}

fn dir_under(root: &Path, session_id: Uuid) -> PathBuf {
    root.join(session_id.to_string())
}

/// Whether `cwd` is a neutral directory made by the host.
pub fn is_neutral_path(cwd: &str) -> bool {
    is_under(&root(), Path::new(cwd))
}

fn is_under(root: &Path, path: &Path) -> bool {
    path.starts_with(root)
        && path
            .components()
            .all(|c| !matches!(c, std::path::Component::ParentDir))
}

/// Makes `path` when it is a neutral directory (no-op for any other path).
pub fn ensure(path: &str) -> std::io::Result<()> {
    if is_neutral_path(path) {
        create_private(Path::new(path))?;
    }
    Ok(())
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

/// Removes the neutral directory of `session_id` (best effort; absent is fine).
pub fn remove_for(session_id: &str) {
    if let Ok(id) = Uuid::parse_str(session_id) {
        remove_dir(&dir_for(id));
    }
}

fn remove_dir(dir: &Path) {
    if let Err(e) = std::fs::remove_dir_all(dir) {
        if e.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(dir = %dir.display(), error = %e, "could not remove a neutral chat directory");
        }
    }
}

/// Removes the directories of [`root`] untouched for `max_age`. Returns how many.
pub fn sweep_orphans(max_age: Duration) -> usize {
    sweep_under(&root(), max_age)
}

fn sweep_under(root: &Path, max_age: Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        // Only what this module makes: a directory named by a session id.
        let ours = path.is_dir()
            && entry
                .file_name()
                .to_str()
                .is_some_and(|n| Uuid::parse_str(n).is_ok());
        if !ours {
            continue;
        }
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age >= max_age);
        if old {
            remove_dir(&path);
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("po-neutral-test-{name}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn the_directory_is_deterministic_from_the_session_id() {
        let id = Uuid::new_v4();
        assert_eq!(dir_for(id), dir_for(id));
        assert_ne!(dir_for(id), dir_for(Uuid::new_v4()));
        assert!(is_neutral_path(&dir_for(id).to_string_lossy()));
    }

    #[test]
    fn only_paths_under_the_root_are_neutral() {
        let root = Path::new("/data/app/chat-neutral");
        assert!(is_under(root, &root.join("abc")));
        assert!(!is_under(root, Path::new("/home/dev/project")));
        assert!(!is_under(
            root,
            Path::new("/data/app/chat-neutral/../secrets")
        ));
        assert!(!is_neutral_path(""));
    }

    #[test]
    fn ensure_ignores_a_project_path() {
        let project = scratch("project").join("not-created");
        ensure(&project.to_string_lossy()).unwrap();
        assert!(!project.exists());
    }

    #[test]
    fn a_resume_recreates_the_same_directory() {
        let root = scratch("resume");
        let dir = dir_under(&root, Uuid::new_v4());
        create_private(&dir).unwrap();
        assert!(dir.is_dir());
        // No configuration of any kind.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        remove_dir(&dir);
        assert!(!dir.exists());
        create_private(&dir).unwrap();
        assert!(dir.is_dir());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn the_sweep_removes_old_session_directories_only() {
        let root = scratch("sweep");
        let mine = dir_under(&root, Uuid::new_v4());
        let foreign = root.join("keep-me");
        std::fs::create_dir_all(&mine).unwrap();
        std::fs::create_dir_all(&foreign).unwrap();
        // Young: kept.
        assert_eq!(sweep_under(&root, Duration::from_secs(3600)), 0);
        assert!(mine.exists());
        // Everything is older than zero: ours goes, the foreign one stays.
        assert_eq!(sweep_under(&root, Duration::ZERO), 1);
        assert!(!mine.exists());
        assert!(foreign.exists());
        std::fs::remove_dir_all(root).ok();
    }
}
