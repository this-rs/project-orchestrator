//! Stable identity of a *file*, independent of where it is mounted.
//!
//! Today a file is identified by its absolute path (`MERGE (f:File {path})`).
//! That makes the same file a different node as soon as it is reached through
//! a symlink, a git worktree or a moved repository root — measured: the PO
//! backend was indexed under `~/.openclaw/workspace/skills/project-orchestrator`
//! while its declared `root_path` is `~/projects/project-orchestrator/backend`,
//! a symlink to the very same directory (same inode).
//!
//! `file_uid` derives the identity from the *project* and the path **relative
//! to the project root**, so the mounting point is no longer part of it.

use uuid::Uuid;

/// Length of the hex digest kept for a uid (128 bits).
const UID_HEX_LEN: usize = 32;

/// Compute the stable uid of a file: `sha256(project:relative_path)`.
///
/// `relative_path` MUST be relative to the project root and use `/` as
/// separator, so that the same file reached through another mount produces
/// the same uid. When there is no project (an ad-hoc sync), the absolute
/// relative path is still used — identity stays stable for that call site.
pub fn file_uid(project_id: Option<Uuid>, relative_path: &str) -> String {
    use sha2::{Digest, Sha256};
    let owner = project_id
        .map(|id| id.hyphenated().to_string())
        .unwrap_or_else(|| "-".to_string());
    let mut hasher = Sha256::new();
    hasher.update(owner.as_bytes());
    hasher.update(b":");
    hasher.update(relative_path.trim_start_matches('/').as_bytes());
    let digest = hasher.finalize();
    hex::encode(digest)[..UID_HEX_LEN].to_string()
}

/// Strip `root` from `path`, both expected to be canonical.
///
/// Returns `None` when `path` is not under `root` — callers then keep the
/// absolute path rather than inventing a misleading relative one.
pub fn relative_to_root(root: &str, path: &str) -> Option<String> {
    let root = root.trim_end_matches('/');
    if root.is_empty() {
        return None;
    }
    let rest = path.strip_prefix(root)?;
    if rest.is_empty() {
        return None;
    }
    Some(rest.trim_start_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> Uuid {
        Uuid::parse_str("00333b5f-2d0a-4467-9c98-155e55d2b7e5").unwrap()
    }

    #[test]
    fn same_project_and_relative_path_give_the_same_uid() {
        assert_eq!(
            file_uid(Some(project()), "src/neo4j/mod.rs"),
            file_uid(Some(project()), "src/neo4j/mod.rs")
        );
    }

    #[test]
    fn the_mount_point_is_not_part_of_the_identity() {
        // The same file reached through two different absolute prefixes.
        let real = "/Users/triviere/.openclaw/workspace/skills/project-orchestrator";
        let link = "/Users/triviere/projects/project-orchestrator/backend";
        let rel_a = relative_to_root(real, &format!("{real}/src/neo4j/mod.rs")).unwrap();
        let rel_b = relative_to_root(link, &format!("{link}/src/neo4j/mod.rs")).unwrap();
        assert_eq!(rel_a, rel_b);
        assert_eq!(
            file_uid(Some(project()), &rel_a),
            file_uid(Some(project()), &rel_b)
        );
    }

    #[test]
    fn two_projects_never_share_a_uid() {
        let other = Uuid::parse_str("b40221d7-bb2d-40c8-ab09-cbaa8fe61106").unwrap();
        assert_ne!(
            file_uid(Some(project()), "src/main.rs"),
            file_uid(Some(other), "src/main.rs")
        );
    }

    #[test]
    fn a_path_outside_the_root_has_no_relative_form() {
        assert_eq!(
            relative_to_root("/Users/triviere/repo", "/Users/triviere/other/a.rs"),
            None
        );
    }
}
