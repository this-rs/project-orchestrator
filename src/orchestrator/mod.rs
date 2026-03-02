//! Orchestrator module for coordinating agents

pub mod context;
pub mod planner;
pub mod runner;
pub mod topology_hook;
pub mod watcher;

pub use context::ContextBuilder;
pub use planner::ImplementationPlanner;
pub use runner::BackfillResult;
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
    "/.claude/",
    // Mobile (iOS / Android / React Native)
    "/Pods/",
    "/.expo/",
    "/DerivedData/",
    "/.gradle/",
    "/.swiftpm/",
    "/xcuserdata/",
    // IDE settings
    "/.idea/",
    "/.vscode/",
];

/// Check whether a file path should be ignored during sync.
///
/// Returns `true` if the path contains any of the [`IGNORED_PATH_SEGMENTS`].
pub fn should_ignore_path(path_str: &str) -> bool {
    // Normalize Windows backslashes to forward slashes for consistent matching.
    // This is a no-op on Unix where paths already use '/'.
    let normalized = path_str.replace('\\', "/");
    IGNORED_PATH_SEGMENTS
        .iter()
        .any(|seg| normalized.contains(seg))
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

    #[test]
    fn test_should_ignore_claude_directory() {
        // .claude/ can contain worktrees (copies of the repo) created by Claude Code
        assert!(should_ignore_path(
            "/project/.claude/worktrees/abc123/src/main.rs"
        ));
        assert!(should_ignore_path("/project/.claude/settings.json"));
        // But "claude" without dots/slashes should NOT be caught
        assert!(!should_ignore_path("/project/src/claude/handler.rs"));
    }

    #[test]
    fn test_should_ignore_mobile_patterns() {
        // CocoaPods (iOS)
        assert!(should_ignore_path(
            "/project/ios/Pods/GoogleMaps/Maps.framework/Headers/GMSMapView.h"
        ));
        // Expo cache
        assert!(should_ignore_path(
            "/project/.expo/web/cache/production/images/favicon.png"
        ));
        // Xcode build output
        assert!(should_ignore_path(
            "/project/DerivedData/App-abc123/Build/Products/Debug/App.app"
        ));
        // Gradle cache (Android)
        assert!(should_ignore_path(
            "/project/.gradle/8.0/executionHistory/executionHistory.bin"
        ));
        // Swift Package Manager
        assert!(should_ignore_path(
            "/project/.swiftpm/xcode/xcuserdata/user.xcuserdatad"
        ));
        // Xcode user data
        assert!(should_ignore_path(
            "/project/App.xcodeproj/xcuserdata/user.xcuserdatad/xcschemes.plist"
        ));
    }

    #[test]
    fn test_should_ignore_ide_settings() {
        assert!(should_ignore_path("/project/.idea/workspace.xml"));
        assert!(should_ignore_path("/project/.vscode/settings.json"));
        // But similar names without dots should NOT be caught
        assert!(!should_ignore_path("/project/src/idea/module.rs"));
        assert!(!should_ignore_path("/project/src/vscode/extension.ts"));
    }

    #[test]
    fn test_should_ignore_windows_paths() {
        // Windows-style paths with backslashes should be correctly matched
        assert!(
            should_ignore_path("C:\\project\\target\\debug\\file.rs"),
            "Windows target path should be ignored"
        );
        assert!(
            should_ignore_path("C:\\repo\\.git\\config"),
            "Windows .git path should be ignored"
        );
        assert!(
            should_ignore_path("D:\\app\\node_modules\\pkg\\index.js"),
            "Windows node_modules path should be ignored"
        );
        assert!(
            should_ignore_path("C:\\project\\dist\\bundle.js"),
            "Windows dist path should be ignored"
        );
        // Normal Windows paths should NOT be ignored
        assert!(
            !should_ignore_path("C:\\project\\src\\main.rs"),
            "Windows src path should not be ignored"
        );
        assert!(
            !should_ignore_path("D:\\workspace\\lib\\utils.ts"),
            "Windows lib path should not be ignored"
        );
    }
}
