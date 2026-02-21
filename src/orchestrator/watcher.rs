//! File watcher for auto-syncing code changes
//!
//! This module watches directories for file changes and automatically
//! updates Neo4j and Meilisearch when files are modified.
//!
//! ## Multi-project support
//!
//! The watcher maintains a map of `root_path -> (project_id, project_slug)`
//! to resolve which project a changed file belongs to. When a file change is
//! detected, the watcher finds the longest matching root_path prefix to
//! determine the correct project context for syncing.

use anyhow::{Context, Result};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use super::Orchestrator;

/// Project context resolved from a file path
#[derive(Debug, Clone)]
struct ProjectContext {
    project_id: Uuid,
    project_slug: String,
}

/// File watcher that auto-syncs changes to the knowledge base.
///
/// Supports watching multiple projects simultaneously. Each registered project
/// has its `root_path` mapped to a `(project_id, project_slug)` pair. When a
/// file change is detected, the watcher resolves the project by finding the
/// longest matching root_path prefix.
pub struct FileWatcher {
    orchestrator: Arc<Orchestrator>,
    watched_paths: Arc<RwLock<HashSet<PathBuf>>>,
    /// Maps canonicalized root_path -> project context
    project_map: Arc<RwLock<HashMap<PathBuf, ProjectContext>>>,
    stop_tx: Option<mpsc::Sender<()>>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(orchestrator: Arc<Orchestrator>) -> Self {
        Self {
            orchestrator,
            watched_paths: Arc::new(RwLock::new(HashSet::new())),
            project_map: Arc::new(RwLock::new(HashMap::new())),
            stop_tx: None,
        }
    }

    /// Set project context for a single project (legacy compatibility).
    ///
    /// This is equivalent to calling `register_project` with the first
    /// watched path. Kept for backward compatibility with existing API.
    pub fn set_project_context(&mut self, project_id: Uuid, slug: String) {
        // Legacy: store context to be applied when watch() is called next.
        // For multi-project, use register_project() instead.
        let map = self.project_map.clone();
        let watched = self.watched_paths.clone();
        tokio::spawn(async move {
            let paths = watched.read().await;
            let mut pm = map.write().await;
            for path in paths.iter() {
                pm.insert(
                    path.clone(),
                    ProjectContext {
                        project_id,
                        project_slug: slug.clone(),
                    },
                );
            }
        });
    }

    /// Register a project for watching.
    ///
    /// Associates a root_path with a project context. The path is canonicalized
    /// and added to the watch list. If the watcher is already running, the new
    /// path will be picked up but requires a restart to be actively watched by
    /// the underlying notify watcher.
    pub async fn register_project(
        &mut self,
        root_path: &Path,
        project_id: Uuid,
        slug: String,
    ) -> Result<()> {
        let canonical = root_path
            .canonicalize()
            .context("Failed to canonicalize project root_path")?;

        // Add to project map
        {
            let mut pm = self.project_map.write().await;
            pm.insert(
                canonical.clone(),
                ProjectContext {
                    project_id,
                    project_slug: slug.clone(),
                },
            );
        }

        // Add to watched paths
        {
            let mut watched = self.watched_paths.write().await;
            watched.insert(canonical.clone());
        }

        tracing::info!(
            "Registered project '{}' ({}) for watching: {}",
            slug,
            project_id,
            canonical.display()
        );
        Ok(())
    }

    /// Unregister a project from watching.
    ///
    /// Removes the project from the project map. The underlying notify watcher
    /// may still receive events for this path, but they will be ignored since
    /// `resolve_project` won't find a match.
    pub async fn unregister_project(&self, project_id: Uuid) {
        let mut pm = self.project_map.write().await;
        let before = pm.len();
        pm.retain(|_, ctx| ctx.project_id != project_id);
        let removed = before - pm.len();
        if removed > 0 {
            tracing::info!(
                "Unregistered project {} from watcher ({} paths removed)",
                project_id,
                removed
            );
        }
    }

    /// Check if a project is currently registered for watching.
    pub async fn is_project_registered(&self, project_id: Uuid) -> bool {
        let pm = self.project_map.read().await;
        pm.values().any(|ctx| ctx.project_id == project_id)
    }

    /// Start watching a directory (legacy compatibility).
    ///
    /// For multi-project use, prefer `register_project()` which combines
    /// path registration with project context.
    pub async fn watch(&mut self, path: &Path) -> Result<()> {
        let path = path.canonicalize().context("Failed to canonicalize path")?;

        // Already watching?
        {
            let watched = self.watched_paths.read().await;
            if watched.contains(&path) {
                return Ok(());
            }
        }

        // Add to watched paths
        {
            let mut watched = self.watched_paths.write().await;
            watched.insert(path.clone());
        }

        tracing::info!("Now watching: {}", path.display());
        Ok(())
    }

    /// Start the watcher background task.
    ///
    /// Spawns a background tokio task that:
    /// 1. Creates a `notify::RecommendedWatcher` for filesystem events
    /// 2. Watches all registered paths recursively
    /// 3. On file change, resolves the project via `resolve_project`
    /// 4. Calls `sync_file_for_project` with the resolved project context
    /// 5. Triggers the analytics debouncer for the affected project
    pub async fn start(&mut self) -> Result<()> {
        if self.stop_tx.is_some() {
            return Ok(()); // Already running
        }

        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);
        let (event_tx, mut event_rx) = mpsc::channel::<PathBuf>(100);

        self.stop_tx = Some(stop_tx);

        let watched_paths = self.watched_paths.clone();
        let project_map = self.project_map.clone();
        let orchestrator = self.orchestrator.clone();

        // Spawn the file system watcher
        tokio::spawn(async move {
            let rt = tokio::runtime::Handle::current();
            let event_tx_clone = event_tx.clone();

            let mut watcher = match RecommendedWatcher::new(
                move |res: Result<Event, notify::Error>| {
                    if let Ok(event) = res {
                        for path in event.paths {
                            let _ = rt.block_on(async { event_tx_clone.send(path).await });
                        }
                    }
                },
                Config::default().with_poll_interval(Duration::from_secs(2)),
            ) {
                Ok(w) => w,
                Err(e) => {
                    tracing::error!("Failed to create watcher: {}", e);
                    return;
                }
            };

            // Watch all registered paths
            let paths = watched_paths.read().await;
            for path in paths.iter() {
                if let Err(e) = watcher.watch(path, RecursiveMode::Recursive) {
                    tracing::error!("Failed to watch {}: {}", path.display(), e);
                }
            }
            drop(paths);

            // Keep watcher alive until stop signal
            loop {
                tokio::select! {
                    _ = stop_rx.recv() => {
                        tracing::info!("File watcher stopping");
                        break;
                    }
                    Some(path) = event_rx.recv() => {
                        // Check if file should be synced
                        if should_sync_file(&path) {
                            tracing::debug!("File changed: {}", path.display());

                            // Debounce: wait a bit before syncing
                            tokio::time::sleep(Duration::from_millis(500)).await;

                            if path.exists() {
                                // Resolve which project this file belongs to
                                let resolved = resolve_project(&path, &project_map).await;
                                let (pid, pslug) = match &resolved {
                                    Some(ctx) => (Some(ctx.project_id), Some(ctx.project_slug.as_str())),
                                    None => (None, None),
                                };

                                match orchestrator
                                    .sync_file_for_project(&path, pid, pslug)
                                    .await
                                {
                                    Ok(true) => {
                                        if let Some(ctx) = &resolved {
                                            tracing::info!(
                                                "Auto-synced: {} (project: {})",
                                                path.display(),
                                                ctx.project_slug
                                            );
                                        } else {
                                            tracing::info!("Auto-synced: {} (no project)", path.display());
                                        }
                                        // Trigger debounced analytics for the resolved project
                                        if let Some(ctx) = &resolved {
                                            orchestrator
                                                .analytics_debouncer()
                                                .trigger(ctx.project_id);
                                        }
                                    }
                                    Ok(false) => {
                                        // File unchanged (hash match), skip
                                        tracing::debug!("File unchanged, skipped: {}", path.display());
                                    }
                                    Err(e) => {
                                        tracing::warn!("Failed to sync {}: {}", path.display(), e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        let paths = self.watched_paths.read().await;
        let projects = self.project_map.read().await;
        tracing::info!(
            "File watcher started ({} paths, {} projects)",
            paths.len(),
            projects.len()
        );
        Ok(())
    }

    /// Stop the watcher
    pub async fn stop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(()).await;
        }
    }

    /// Get currently watched paths
    pub async fn watched_paths(&self) -> Vec<PathBuf> {
        self.watched_paths.read().await.iter().cloned().collect()
    }

    /// Get the number of registered projects
    pub async fn registered_project_count(&self) -> usize {
        self.project_map.read().await.len()
    }

    /// Get registered project IDs and their root paths
    pub async fn registered_projects(&self) -> Vec<(Uuid, String, PathBuf)> {
        self.project_map
            .read()
            .await
            .iter()
            .map(|(path, ctx)| (ctx.project_id, ctx.project_slug.clone(), path.clone()))
            .collect()
    }
}

/// Resolve which project a file belongs to by finding the longest matching
/// root_path prefix in the project map.
///
/// Uses longest prefix match to handle nested projects correctly. For example,
/// if project A has root `/projects/mono` and project B has root
/// `/projects/mono/packages/web`, a file at `/projects/mono/packages/web/src/app.ts`
/// will resolve to project B (more specific match).
async fn resolve_project(
    file_path: &Path,
    project_map: &Arc<RwLock<HashMap<PathBuf, ProjectContext>>>,
) -> Option<ProjectContext> {
    let pm = project_map.read().await;

    pm.iter()
        .filter(|(root, _)| file_path.starts_with(root))
        .max_by_key(|(root, _)| root.components().count())
        .map(|(_, ctx)| ctx.clone())
}

/// Check if a file should be synced based on extension and path
///
/// Supports all 21 extensions matching the main sync engine in runner.rs.
fn should_sync_file(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();

    // All supported languages — must stay aligned with runner.rs sync_directory_for_project()
    let supported_extensions = [
        "rs", // Rust
        "ts", "tsx", "js", "jsx", // TypeScript/JavaScript
        "py",   // Python
        "go",   // Go
        "java", // Java
        "c", "h", // C
        "cpp", "cc", "cxx", "hpp", "hxx", // C++
        "rb",  // Ruby
        "php", // PHP
        "kt", "kts", // Kotlin
        "swift", // Swift
        "sh", "bash", // Bash
    ];

    if !supported_extensions.contains(&ext) {
        return false;
    }

    let path_str = path.to_string_lossy();

    // Skip common non-source directories
    if path_str.contains("node_modules")
        || path_str.contains("/target/")
        || path_str.contains("/.git/")
        || path_str.contains("__pycache__")
        || path_str.contains("/dist/")
        || path_str.contains("/build/")
    {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── should_sync_file tests (unchanged) ────────────────────────────

    #[test]
    fn test_should_sync_rust_files() {
        assert!(should_sync_file(Path::new("/project/src/main.rs")));
        assert!(should_sync_file(Path::new("/project/src/lib.rs")));
        assert!(should_sync_file(Path::new("/project/tests/test.rs")));
    }

    #[test]
    fn test_should_sync_typescript_files() {
        assert!(should_sync_file(Path::new("/project/src/index.ts")));
        assert!(should_sync_file(Path::new("/project/components/App.tsx")));
    }

    #[test]
    fn test_should_sync_javascript_files() {
        assert!(should_sync_file(Path::new("/project/lib/utils.js")));
        assert!(should_sync_file(Path::new(
            "/project/components/Button.jsx"
        )));
    }

    #[test]
    fn test_should_sync_python_files() {
        assert!(should_sync_file(Path::new("/project/app.py")));
        assert!(should_sync_file(Path::new("/project/src/main.py")));
    }

    #[test]
    fn test_should_sync_go_files() {
        assert!(should_sync_file(Path::new("/project/main.go")));
        assert!(should_sync_file(Path::new("/project/pkg/handler.go")));
    }

    #[test]
    fn test_should_sync_java_files() {
        assert!(should_sync_file(Path::new("/project/src/Main.java")));
    }

    #[test]
    fn test_should_sync_c_cpp_files() {
        assert!(should_sync_file(Path::new("/project/src/main.c")));
        assert!(should_sync_file(Path::new("/project/include/header.h")));
        assert!(should_sync_file(Path::new("/project/src/app.cpp")));
        assert!(should_sync_file(Path::new("/project/src/lib.cc")));
        assert!(should_sync_file(Path::new("/project/src/util.cxx")));
        assert!(should_sync_file(Path::new("/project/include/types.hpp")));
        assert!(should_sync_file(Path::new("/project/include/utils.hxx")));
    }

    #[test]
    fn test_should_sync_ruby_files() {
        assert!(should_sync_file(Path::new("/project/app.rb")));
    }

    #[test]
    fn test_should_sync_php_files() {
        assert!(should_sync_file(Path::new("/project/index.php")));
    }

    #[test]
    fn test_should_sync_kotlin_files() {
        assert!(should_sync_file(Path::new("/project/src/Main.kt")));
        assert!(should_sync_file(Path::new("/project/build.gradle.kts")));
    }

    #[test]
    fn test_should_sync_swift_files() {
        assert!(should_sync_file(Path::new("/project/Sources/App.swift")));
    }

    #[test]
    fn test_should_sync_bash_files() {
        assert!(should_sync_file(Path::new("/project/scripts/deploy.sh")));
        assert!(should_sync_file(Path::new("/project/scripts/setup.bash")));
    }

    #[test]
    fn test_should_not_sync_unsupported_extensions() {
        assert!(!should_sync_file(Path::new("/project/README.md")));
        assert!(!should_sync_file(Path::new("/project/config.json")));
        assert!(!should_sync_file(Path::new("/project/style.css")));
        assert!(!should_sync_file(Path::new("/project/data.yaml")));
        assert!(!should_sync_file(Path::new("/project/image.png")));
    }

    #[test]
    fn test_should_not_sync_node_modules() {
        assert!(!should_sync_file(Path::new(
            "/project/node_modules/lib/index.js"
        )));
        assert!(!should_sync_file(Path::new(
            "/project/node_modules/package/src/utils.ts"
        )));
    }

    #[test]
    fn test_should_not_sync_target_directory() {
        assert!(!should_sync_file(Path::new(
            "/project/target/debug/main.rs"
        )));
        assert!(!should_sync_file(Path::new(
            "/project/target/release/lib.rs"
        )));
    }

    #[test]
    fn test_should_not_sync_git_directory() {
        assert!(!should_sync_file(Path::new(
            "/project/.git/hooks/pre-commit"
        )));
        assert!(!should_sync_file(Path::new(
            "/project/.git/objects/pack/file.rs"
        )));
    }

    #[test]
    fn test_should_not_sync_pycache() {
        assert!(!should_sync_file(Path::new(
            "/project/__pycache__/module.cpython-311.pyc"
        )));
        assert!(!should_sync_file(Path::new(
            "/project/src/__pycache__/utils.py"
        )));
    }

    #[test]
    fn test_should_not_sync_dist_build_directories() {
        assert!(!should_sync_file(Path::new("/project/dist/bundle.js")));
        assert!(!should_sync_file(Path::new("/project/build/output.js")));
    }

    #[test]
    fn test_should_sync_empty_extension() {
        // Files without extension should not be synced
        assert!(!should_sync_file(Path::new("/project/Makefile")));
        assert!(!should_sync_file(Path::new("/project/Dockerfile")));
    }

    // ── resolve_project tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_resolve_project_single_match() {
        let map = Arc::new(RwLock::new(HashMap::new()));
        let pid = Uuid::new_v4();
        {
            let mut pm = map.write().await;
            pm.insert(
                PathBuf::from("/projects/my-app"),
                ProjectContext {
                    project_id: pid,
                    project_slug: "my-app".to_string(),
                },
            );
        }

        let result = resolve_project(Path::new("/projects/my-app/src/main.rs"), &map).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().project_id, pid);
    }

    #[tokio::test]
    async fn test_resolve_project_no_match() {
        let map = Arc::new(RwLock::new(HashMap::new()));
        {
            let mut pm = map.write().await;
            pm.insert(
                PathBuf::from("/projects/my-app"),
                ProjectContext {
                    project_id: Uuid::new_v4(),
                    project_slug: "my-app".to_string(),
                },
            );
        }

        let result = resolve_project(Path::new("/other/path/file.rs"), &map).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_resolve_project_longest_prefix_wins() {
        let map = Arc::new(RwLock::new(HashMap::new()));
        let parent_id = Uuid::new_v4();
        let child_id = Uuid::new_v4();
        {
            let mut pm = map.write().await;
            pm.insert(
                PathBuf::from("/projects/mono"),
                ProjectContext {
                    project_id: parent_id,
                    project_slug: "mono".to_string(),
                },
            );
            pm.insert(
                PathBuf::from("/projects/mono/packages/web"),
                ProjectContext {
                    project_id: child_id,
                    project_slug: "web".to_string(),
                },
            );
        }

        // File in child project → should resolve to child
        let result =
            resolve_project(Path::new("/projects/mono/packages/web/src/app.ts"), &map).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().project_id, child_id);

        // File in parent project (not in child) → should resolve to parent
        let result =
            resolve_project(Path::new("/projects/mono/src/lib.rs"), &map).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().project_id, parent_id);
    }

    #[tokio::test]
    async fn test_resolve_project_empty_map() {
        let map = Arc::new(RwLock::new(HashMap::new()));
        let result = resolve_project(Path::new("/any/path/file.rs"), &map).await;
        assert!(result.is_none());
    }
}
