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

/// Compute the stable uid of a file: `<project>:<relative_path>`.
///
/// `relative_path` MUST be relative to the project root and use `/` as
/// separator, so that the same file reached through another mount produces
/// the same uid. When there is no project (an ad-hoc sync), the absolute
/// path is still used — identity stays stable for that call site.
///
/// # Why a readable form and not a hash
///
/// The backfill of the 33 499 files that already exist has to run **in Cypher**,
/// at startup, with no Rust pass over the graph. A digest is not expressible
/// there (`apoc.util.sha256` is not guaranteed on every install), so a hashed
/// uid could only be backfilled by a second implementation — two grammars for
/// one identity, which is exactly the defect this series removes. A
/// concatenation is expressible on both sides *and* verifiable by eye: when a
/// duplicate shows up, the reader can see which part differs.
///
/// The owner is the hyphenated project id, or `-` when there is none.
pub fn file_uid(project_id: Option<Uuid>, relative_path: &str) -> String {
    let owner = project_id
        .map(|id| id.hyphenated().to_string())
        .unwrap_or_else(|| "-".to_string());
    format!("{}:{}", owner, relative_path.trim_start_matches('/'))
}

/// Canonical form of a project root: the path the graph actually stores.
///
/// `Project.root_path` keeps the path as the user declared it (often a symlink);
/// the sync resolves links before writing, so the stored file paths are
/// canonical. Without this second value, `strip_prefix` (or its Cypher
/// equivalent) fails on every project configured through a link — and the
/// relative path, hence the identity, cannot be computed.
///
/// Falls back to the declared path when canonicalisation fails (a missing
/// directory, a path that is not yet reachable): a project we cannot resolve is
/// still a project, and an empty value would silently drop it from the backfill.
pub fn canonical_root(declared: Option<&str>) -> String {
    let Some(declared) = declared.map(str::trim).filter(|r| !r.is_empty()) else {
        return String::new();
    };
    std::fs::canonicalize(declared)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| declared.to_string())
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

    /// La forme est celle que le Cypher de `upsert_file` recalcule : une
    /// identité qu'on doit pouvoir relire, pas une empreinte opaque.
    #[test]
    fn the_uid_is_readable_and_names_its_project() {
        assert_eq!(
            file_uid(Some(project()), "src/neo4j/mod.rs"),
            format!("{}:src/neo4j/mod.rs", project())
        );
        assert_eq!(file_uid(None, "/tmp/a.rs"), "-:tmp/a.rs");
    }

    #[test]
    fn a_canonical_root_is_resolved_and_never_empty_when_declared() {
        // A directory that exists: canonicalisation resolves it.
        let here = canonical_root(Some("src"));
        assert!(here.ends_with("src"), "got {here}");
        // Nothing declared: nothing canonical.
        assert_eq!(canonical_root(None), "");
        assert_eq!(canonical_root(Some("   ")), "");
        // Unresolvable: the declared path is kept, not dropped.
        assert_eq!(
            canonical_root(Some("/definitely/not/a/real/path-42")),
            "/definitely/not/a/real/path-42"
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
