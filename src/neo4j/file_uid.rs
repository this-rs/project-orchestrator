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

/// Rewrite a stored (canonical) path under the project's **declared** root.
///
/// The sync resolves links before writing, so the graph holds `/alias/…` while
/// the user configured `/declared/…`. Both name one directory; only the second
/// one is what a human reads in the graph, in a note anchor, or in a picker.
/// This is the mapping that makes a rename possible and keeps it from being
/// undone: applied at the write boundary, every later sync produces the
/// declared form instead of the alias.
///
/// Returns `None` when `path` is not under `canonical_root` — the caller then
/// keeps the stored path rather than inventing one.
pub fn declared_path(declared_root: &str, canonical_root: &str, path: &str) -> Option<String> {
    let declared = declared_root.trim_end_matches('/');
    let canonical = canonical_root.trim_end_matches('/');
    if declared.is_empty() || canonical.is_empty() || declared == canonical {
        return None;
    }
    let rest = path.strip_prefix(canonical)?;
    if rest.is_empty() {
        return None;
    }
    Some(format!("{}{}", declared, rest))
}

/// Bring a path under the project's **declared** root, resolving that root on the spot.
///
/// This is [`declared_path`] with `canonical_root(declared_root)` computed here,
/// for the write boundary of the sync: a path that is not under the root (or a
/// root that is not a link) comes back unchanged — the sync never invents a
/// path. Applied where the sync produces `File.path` (and, through the parser,
/// every symbol's `file_path`), it is what keeps a rescan from re-creating the
/// alias form the migration just removed.
///
/// A path may reach the directory through a *third* alias — neither the
/// declared root nor its canonical form (`/var/…` vs `/private/var/…` on macOS,
/// a second link elsewhere). When the prefix does not match, the path itself is
/// resolved and mapped from its real form; a path that cannot be resolved is
/// kept as given.
pub fn under_declared_root(declared_root: &str, path: &str) -> String {
    let declared = declared_root.trim_end_matches('/');
    if declared.is_empty() || path.starts_with(&format!("{declared}/")) {
        return path.to_string();
    }
    let canonical = canonical_root(Some(declared));
    if let Some(mapped) = declared_path(declared, &canonical, path) {
        return mapped;
    }
    match std::fs::canonicalize(path) {
        Ok(real) => declared_path(declared, &canonical, &real.to_string_lossy())
            .unwrap_or_else(|| path.to_string()),
        Err(_) => path.to_string(),
    }
}

/// Is `path` really inside `root`, once links are resolved?
///
/// The name-based ignore list ([`crate::orchestrator::IGNORED_PATH_SEGMENTS`])
/// cannot answer this: it excludes directories *by name*, and a project reached
/// through a link is walked with `follow_links(true)`, so a symlink pointing
/// outside the repository would be indexed as if it were part of it. The durable
/// rule is not a list of names — it is this predicate.
///
/// Never answers `false` on a resolution failure: an unreadable path is kept,
/// because dropping files silently is worse than indexing one too many.
pub fn realpath_within_root(root: &std::path::Path, path: &std::path::Path) -> bool {
    let Ok(root_real) = root.canonicalize() else {
        return path.starts_with(root);
    };
    match path.canonicalize() {
        Ok(real) => real.starts_with(&root_real),
        Err(_) => path.starts_with(root),
    }
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
    fn a_canonical_path_is_rewritten_under_the_declared_root() {
        let declared = "/Users/triviere/projects/project-orchestrator/backend";
        let canonical = "/Users/triviere/.openclaw/workspace/skills/project-orchestrator";
        let stored = format!("{canonical}/src/main.rs");

        assert_eq!(
            declared_path(declared, canonical, &stored).as_deref(),
            Some("/Users/triviere/projects/project-orchestrator/backend/src/main.rs"),
            "the alias becomes the path the user configured"
        );
        // Same roots: nothing to rewrite, and silence is not a rewrite.
        assert_eq!(declared_path(declared, declared, &stored), None);
        // Outside the canonical root: no mapping, no invention.
        assert_eq!(declared_path(declared, canonical, "/elsewhere/a.rs"), None);
        // Empty roots are not a licence to rewrite everything.
        assert_eq!(declared_path("", canonical, &stored), None);
    }

    #[test]
    fn the_perimeter_is_the_real_path_not_the_name() {
        let base = std::env::temp_dir().join("po-perimetre-test");
        let _ = std::fs::create_dir_all(base.join("inside"));
        let inside = base.join("inside/a.rs");
        let _ = std::fs::write(&inside, "fn a() {}");
        let outside = base.join("../outside.rs");
        let _ = std::fs::write(&outside, "fn b() {}");

        assert!(realpath_within_root(&base, &inside));
        assert!(
            !realpath_within_root(&base, &outside),
            "a path that only looks like it belongs must not pass"
        );
        // A path that cannot be resolved is kept, never dropped by accident.
        assert!(realpath_within_root(&base, &base.join("nope.rs")));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_path_outside_the_root_has_no_relative_form() {
        assert_eq!(
            relative_to_root("/Users/triviere/repo", "/Users/triviere/other/a.rs"),
            None
        );
    }
}
