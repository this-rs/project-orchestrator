//! Orchestrator module for coordinating agents
#[allow(unused_imports)]
use crate::neo4j::traits::*;

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
    // Git WORKTREES. A worktree is a checkout of the same repository on another
    // branch, not code of the project: indexing it multiplies every file and
    // every symbol by the number of branches, and buries the real graph under
    // copies. Measured on 29/10/2026 in the local store (4 665 569 nodes):
    // 511 678 `Function`, 356 834 `Impl`, 317 250 `Import` and 98 195 `Struct`
    // lived under these segments — more than a million nodes of duplicates,
    // and FIVE times more `Impl` in the copies than in the real source. Each
    // repository names its worktrees differently, hence one entry per name.
    "/_wt/",
    "/.wt/",
    "/.rlwt/",
    "/worktrees/",
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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

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

    /// La racine réelle du backend vit sous `~/.openclaw/…`. Ajouter `.openclaw`
    /// à `IGNORED_PATH_SEGMENTS` — le réflexe évident en voyant les chemins du
    /// graphe — **effacerait le backend entier** au prochain sync. Ce test est là
    /// pour que ce réflexe échoue bruyamment.
    #[test]
    fn the_backend_real_root_is_never_ignored() {
        assert!(
            !should_ignore_path(
                "/Users/triviere/.openclaw/workspace/skills/project-orchestrator/src/main.rs"
            ),
            "`.openclaw` must never become an ignored segment: it is the backend's real root"
        );
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

    /// A worktree is a copy of the repository on another branch, never the
    /// project's code: no path under `_wt/`, `.wt/`, `.rlwt/` or `worktrees/`
    /// may enter the graph.
    #[test]
    fn a_worktree_is_never_synced() {
        for path in [
            "/repo/_wt/feature-a/src/lib.rs",
            "/repo/.wt/feature-b/src/lib.rs",
            "/repo/.rlwt/feature-c/src/lib.rs",
            "/repo/worktrees/feature-d/src/lib.rs",
        ] {
            assert!(should_ignore_path(path), "{path} is a worktree, not code");
        }
    }

    /// The rule is anchored on the DIRECTORY: a source file whose name merely
    /// ends like a worktree directory is still the project's code.
    #[test]
    fn a_file_named_after_a_worktree_is_still_code() {
        for path in [
            "/repo/src/worktrees.rs",
            "/repo/src/wt.rs",
            "/repo/src/features/rlwt.rs",
        ] {
            assert!(!should_ignore_path(path), "{path} is project code");
        }
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
