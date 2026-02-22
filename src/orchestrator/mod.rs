//! Orchestrator module for coordinating agents

pub mod context;
pub mod planner;
pub mod runner;
pub mod watcher;

pub use context::ContextBuilder;
pub use planner::ImplementationPlanner;
pub use runner::Orchestrator;
pub use watcher::spawn_project_watcher_bridge;
pub use watcher::FileWatcher;

// ============================================================================
// Shared sync filters
// ============================================================================

/// Directory segments that should be excluded from code sync (both
/// `sync_directory` in runner.rs and `should_sync_file` in watcher.rs).
///
/// Each entry is matched with `path.contains(segment)`.
/// Use leading/trailing slashes to avoid false positives (e.g. "/dist/" won't
/// match a file named "distribution.rs").
pub const IGNORED_PATH_SEGMENTS: &[&str] = &[
    "node_modules",
    "/target/",
    "/.git/",
    "__pycache__",
    "/dist/",
    "/build/",
    "/vendor/",
    "/.next/",
    "/.nuxt/",
    "/coverage/",
    "/.cache/",
];

/// Check whether a file path should be ignored during sync.
///
/// Returns `true` if the path contains any of the [`IGNORED_PATH_SEGMENTS`].
pub fn should_ignore_path(path_str: &str) -> bool {
    IGNORED_PATH_SEGMENTS
        .iter()
        .any(|seg| path_str.contains(seg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_ignore_all_patterns() {
        // Every segment in IGNORED_PATH_SEGMENTS must trigger ignoring
        let test_paths: Vec<String> = IGNORED_PATH_SEGMENTS
            .iter()
            .map(|seg| format!("/project{seg}some/file.rs"))
            .collect();

        for path in &test_paths {
            assert!(
                should_ignore_path(path),
                "Expected path to be ignored: {path}"
            );
        }
    }

    #[test]
    fn test_should_not_ignore_normal_paths() {
        assert!(!should_ignore_path("/project/src/main.rs"));
        assert!(!should_ignore_path("/project/lib/utils.ts"));
        assert!(!should_ignore_path("/project/tests/integration.py"));
    }

    #[test]
    fn test_no_false_positive_on_similar_names() {
        // "distribution" should NOT be caught by "/dist/"
        assert!(!should_ignore_path("/project/src/distribution.rs"));
        // "rebuild" should NOT be caught by "/build/"
        assert!(!should_ignore_path("/project/src/rebuild.rs"));
    }
}
