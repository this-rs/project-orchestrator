//! Central path utility module for project-agnostic path handling.
//!
//! This module is the **single source of truth** for relativizing and resolving
//! paths across the codebase. All code that needs to strip a project root from
//! a path or resolve a relative path back to absolute MUST use these functions.
//!
//! # Design decisions
//! - Neo4j `File` nodes keep their absolute `path` property (internal matching key).
//! - All **exit points** (API responses, exports, triggers, anchors) are normalized
//!   to relative paths via these helpers.

use std::path::{Path, PathBuf};

/// Relativize an absolute path against a project root.
///
/// Returns a relative path string (no leading `/`).
///
/// # Behaviour
/// - If `path` starts with `root_path`, strips the prefix.
/// - If `path` is already relative (no leading `/`), returns it as-is.
/// - If `path` doesn't start with `root_path`, returns it unchanged.
/// - Handles trailing slashes on `root_path`.
/// - Handles double slashes gracefully.
///
/// ```
/// use project_orchestrator::utils::paths::relativize;
/// assert_eq!(relativize("/Users/foo/project/src/main.rs", "/Users/foo/project"), "src/main.rs");
/// assert_eq!(relativize("src/main.rs", "/Users/foo/project"), "src/main.rs");
/// ```
pub fn relativize(path: &str, root_path: &str) -> String {
    // Already relative — nothing to do
    if !path.starts_with('/') {
        return path.to_string();
    }

    // Empty root — can't relativize
    if root_path.is_empty() {
        return path.to_string();
    }

    let root = root_path.trim_end_matches('/');

    // Exact match: path == root → return empty (project root itself)
    if path == root {
        return String::new();
    }

    // Standard prefix strip: /Users/foo/project/src/main.rs → src/main.rs
    if let Some(rest) = path.strip_prefix(root) {
        let rest = rest.trim_start_matches('/');
        return rest.to_string();
    }

    // Path is outside the root — return unchanged
    path.to_string()
}

/// Resolve a relative path against a project root, producing an absolute [`PathBuf`].
///
/// If the path is already absolute, returns it as-is.
///
/// ```
/// use project_orchestrator::utils::paths::resolve;
/// use std::path::PathBuf;
/// assert_eq!(resolve("src/main.rs", "/Users/foo/project"), PathBuf::from("/Users/foo/project/src/main.rs"));
/// ```
pub fn resolve(relative: &str, root_path: &str) -> PathBuf {
    let p = Path::new(relative);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    Path::new(root_path).join(relative)
}

/// Sanitize a glob pattern by stripping an absolute root prefix.
///
/// This is specifically designed for `FileGlob` trigger patterns that may
/// contain absolute prefixes (e.g., `/Users/foo/project/src/**`).
///
/// # Behaviour
/// - Strips `root_path` prefix from the pattern.
/// - Preserves glob wildcards (`*`, `**`, `?`, `[...]`).
/// - If pattern is already relative, returns it unchanged.
///
/// ```
/// use project_orchestrator::utils::paths::sanitize_pattern;
/// assert_eq!(sanitize_pattern("/Users/foo/project/src/**", "/Users/foo/project"), "src/**");
/// assert_eq!(sanitize_pattern("src/**/*.rs", "/Users/foo/project"), "src/**/*.rs");
/// ```
pub fn sanitize_pattern(pattern: &str, root_path: &str) -> String {
    // Already relative — nothing to do
    if !pattern.starts_with('/') {
        return pattern.to_string();
    }

    // Empty root — can't sanitize
    if root_path.is_empty() {
        return pattern.to_string();
    }

    let root = root_path.trim_end_matches('/');

    if let Some(rest) = pattern.strip_prefix(root) {
        let rest = rest.trim_start_matches('/');
        return rest.to_string();
    }

    // Pattern is outside root — return unchanged
    pattern.to_string()
}

/// Check whether a path looks relative (does not start with `/` or a Windows drive letter).
pub fn is_relative(path: &str) -> bool {
    !path.starts_with('/') && !path.as_bytes().get(1).is_some_and(|&b| b == b':')
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── relativize ──────────────────────────────────────────────────────

    #[test]
    fn relativize_strips_root_prefix() {
        assert_eq!(
            relativize("/Users/foo/project/src/main.rs", "/Users/foo/project"),
            "src/main.rs"
        );
    }

    #[test]
    fn relativize_handles_trailing_slash_on_root() {
        assert_eq!(
            relativize("/Users/foo/project/src/main.rs", "/Users/foo/project/"),
            "src/main.rs"
        );
    }

    #[test]
    fn relativize_already_relative() {
        assert_eq!(
            relativize("src/main.rs", "/Users/foo/project"),
            "src/main.rs"
        );
    }

    #[test]
    fn relativize_path_equals_root() {
        assert_eq!(relativize("/Users/foo/project", "/Users/foo/project"), "");
    }

    #[test]
    fn relativize_path_outside_root() {
        assert_eq!(
            relativize("/other/place/file.rs", "/Users/foo/project"),
            "/other/place/file.rs"
        );
    }

    #[test]
    fn relativize_empty_root() {
        assert_eq!(
            relativize("/Users/foo/project/src/main.rs", ""),
            "/Users/foo/project/src/main.rs"
        );
    }

    #[test]
    fn relativize_double_slash() {
        // Root without trailing slash + path with double slash after root
        assert_eq!(
            relativize("/Users/foo/project//src/main.rs", "/Users/foo/project"),
            "src/main.rs"
        );
    }

    #[test]
    fn relativize_nested_deep() {
        assert_eq!(
            relativize(
                "/Users/foo/project/src/api/handlers/sharing.rs",
                "/Users/foo/project"
            ),
            "src/api/handlers/sharing.rs"
        );
    }

    // ── resolve ─────────────────────────────────────────────────────────

    #[test]
    fn resolve_joins_relative_to_root() {
        assert_eq!(
            resolve("src/main.rs", "/Users/foo/project"),
            PathBuf::from("/Users/foo/project/src/main.rs")
        );
    }

    #[test]
    fn resolve_absolute_unchanged() {
        assert_eq!(
            resolve("/absolute/path.rs", "/Users/foo/project"),
            PathBuf::from("/absolute/path.rs")
        );
    }

    #[test]
    fn resolve_empty_relative() {
        assert_eq!(
            resolve("", "/Users/foo/project"),
            PathBuf::from("/Users/foo/project")
        );
    }

    // ── sanitize_pattern ────────────────────────────────────────────────

    #[test]
    fn sanitize_strips_root_from_glob() {
        assert_eq!(
            sanitize_pattern("/Users/foo/project/src/**", "/Users/foo/project"),
            "src/**"
        );
    }

    #[test]
    fn sanitize_already_relative_glob() {
        assert_eq!(
            sanitize_pattern("src/**/*.rs", "/Users/foo/project"),
            "src/**/*.rs"
        );
    }

    #[test]
    fn sanitize_glob_outside_root() {
        assert_eq!(
            sanitize_pattern("/other/src/**", "/Users/foo/project"),
            "/other/src/**"
        );
    }

    #[test]
    fn sanitize_root_with_trailing_slash() {
        assert_eq!(
            sanitize_pattern("/Users/foo/project/src/**", "/Users/foo/project/"),
            "src/**"
        );
    }

    #[test]
    fn sanitize_empty_root() {
        assert_eq!(
            sanitize_pattern("/Users/foo/project/src/**", ""),
            "/Users/foo/project/src/**"
        );
    }

    // ── is_relative ─────────────────────────────────────────────────────

    #[test]
    fn is_relative_true_for_relative() {
        assert!(is_relative("src/main.rs"));
        assert!(is_relative("main.rs"));
    }

    #[test]
    fn is_relative_false_for_absolute() {
        assert!(!is_relative("/Users/foo/project/src/main.rs"));
    }

    #[test]
    fn is_relative_false_for_windows_drive() {
        assert!(!is_relative("C:\\Users\\foo\\project"));
    }

    // ── roundtrip ───────────────────────────────────────────────────────

    #[test]
    fn roundtrip_relativize_then_resolve() {
        let root = "/Users/foo/project";
        let abs = "/Users/foo/project/src/api/mod.rs";
        let rel = relativize(abs, root);
        assert_eq!(rel, "src/api/mod.rs");
        assert_eq!(resolve(&rel, root), PathBuf::from(abs));
    }
}
