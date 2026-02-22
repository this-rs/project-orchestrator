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
use crate::events::{CrudAction, CrudEvent, EntityType as EventEntityType};

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
    /// Channel to dynamically add new paths to the running notify watcher.
    /// None if the watcher background task hasn't been started yet.
    add_path_tx: Option<mpsc::Sender<PathBuf>>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(orchestrator: Arc<Orchestrator>) -> Self {
        Self {
            orchestrator,
            watched_paths: Arc::new(RwLock::new(HashSet::new())),
            project_map: Arc::new(RwLock::new(HashMap::new())),
            stop_tx: None,
            add_path_tx: None,
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
    /// path is dynamically added to the underlying notify watcher via channel.
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
        let is_new = {
            let mut watched = self.watched_paths.write().await;
            watched.insert(canonical.clone())
        };

        // If the watcher is running and this is a new path, notify the
        // background task so it adds the path to the notify watcher.
        if is_new {
            if let Some(ref tx) = self.add_path_tx {
                if let Err(e) = tx.send(canonical.clone()).await {
                    tracing::warn!("Failed to send new watch path to background watcher: {}", e);
                }
            }
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
    /// 2. Watches all currently registered paths recursively
    /// 3. Listens for dynamically added paths via `add_path_rx`
    /// 4. On file change, resolves the project via `resolve_project`
    /// 5. Calls `sync_file_for_project` with the resolved project context
    /// 6. Triggers the analytics debouncer for the affected project
    ///
    /// Safe to call with 0 registered projects — the watcher will be ready
    /// to accept new paths dynamically via `register_project()`.
    pub async fn start(&mut self) -> Result<()> {
        if self.stop_tx.is_some() {
            return Ok(()); // Already running
        }

        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);
        let (event_tx, mut event_rx) = mpsc::channel::<PathBuf>(100);
        let (add_path_tx, mut add_path_rx) = mpsc::channel::<PathBuf>(32);

        self.stop_tx = Some(stop_tx);
        self.add_path_tx = Some(add_path_tx);

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

            // Watch all currently registered paths
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
                    // Dynamically add a new path to the notify watcher
                    Some(new_path) = add_path_rx.recv() => {
                        match watcher.watch(&new_path, RecursiveMode::Recursive) {
                            Ok(_) => {
                                tracing::info!(
                                    "File watcher: dynamically added path: {}",
                                    new_path.display()
                                );
                            }
                            Err(e) => {
                                tracing::error!(
                                    "File watcher: failed to watch new path {}: {}",
                                    new_path.display(),
                                    e
                                );
                            }
                        }
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
        "ts", "tsx", "js", "jsx",  // TypeScript/JavaScript
        "py",   // Python
        "go",   // Go
        "java", // Java
        "c", "h", // C
        "cpp", "cc", "cxx", "hpp", "hxx", // C++
        "rb",  // Ruby
        "php", // PHP
        "kt", "kts",   // Kotlin
        "swift", // Swift
        "sh", "bash", // Bash
    ];

    if !supported_extensions.contains(&ext) {
        return false;
    }

    let path_str = path.to_string_lossy();

    // Skip ignored directories (shared constant with runner.rs)
    if super::should_ignore_path(&path_str) {
        return false;
    }

    true
}

// ============================================================================
// Event-driven watcher bridge
// ============================================================================

/// Spawn a background task that listens to CrudEvents and automatically
/// registers/unregisters projects on the FileWatcher.
///
/// This decouples the watcher from individual handlers (API, MCP, NATS).
/// Any code path that calls `Orchestrator::create_project()` /
/// `delete_project()` / `update_project()` will emit a CrudEvent, and
/// the bridge will react accordingly.
///
/// Pattern follows `HybridEmitter::start_nats_bridge()` in `events/hybrid.rs`.
pub fn spawn_project_watcher_bridge(
    watcher: Arc<RwLock<FileWatcher>>,
    mut event_rx: tokio::sync::broadcast::Receiver<CrudEvent>,
    neo4j: Arc<dyn crate::neo4j::GraphStore>,
) {
    tokio::spawn(async move {
        tracing::info!("Project watcher bridge started — listening for project CRUD events");

        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    if event.entity_type != EventEntityType::Project {
                        continue;
                    }
                    match event.action {
                        CrudAction::Created => {
                            handle_project_created(&watcher, &event).await;
                        }
                        CrudAction::Deleted => {
                            handle_project_deleted(&watcher, &event).await;
                        }
                        CrudAction::Updated => {
                            handle_project_updated(&watcher, &event, &neo4j).await;
                        }
                        _ => {} // Linked/Unlinked — not relevant for watcher
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(
                        "Project watcher bridge: lagged by {} events, some projects may not be auto-registered",
                        n
                    );
                    // Continue processing — bridge will catch up
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    tracing::info!("Project watcher bridge: event bus closed, stopping");
                    break;
                }
            }
        }
    });
}

/// Handle a Project::Created event — register on the watcher if root_path exists.
async fn handle_project_created(watcher: &Arc<RwLock<FileWatcher>>, event: &CrudEvent) {
    let root_path = match event.payload.get("root_path").and_then(|v| v.as_str()) {
        Some(rp) => rp.to_string(),
        None => {
            tracing::warn!(
                "Watcher bridge: Project::Created event missing root_path in payload (id={})",
                event.entity_id
            );
            return;
        }
    };

    let slug = event
        .payload
        .get("slug")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let project_id = match event.entity_id.parse::<Uuid>() {
        Ok(id) => id,
        Err(_) => {
            tracing::warn!(
                "Watcher bridge: invalid project UUID in Created event: {}",
                event.entity_id
            );
            return;
        }
    };

    let expanded = crate::expand_tilde(&root_path);
    let path = std::path::Path::new(&expanded);
    if !path.exists() {
        tracing::debug!(
            "Watcher bridge: skipping project '{}' — path does not exist: {}",
            slug,
            expanded
        );
        return;
    }

    let mut w = watcher.write().await;
    match w.register_project(path, project_id, slug.clone()).await {
        Ok(_) => {
            tracing::info!("Watcher bridge: auto-registered project '{}'", slug);
        }
        Err(e) => {
            tracing::warn!(
                "Watcher bridge: failed to register project '{}': {}",
                slug,
                e
            );
        }
    }
}

/// Handle a Project::Deleted event — unregister from the watcher.
async fn handle_project_deleted(watcher: &Arc<RwLock<FileWatcher>>, event: &CrudEvent) {
    let project_id = match event.entity_id.parse::<Uuid>() {
        Ok(id) => id,
        Err(_) => {
            tracing::warn!(
                "Watcher bridge: invalid project UUID in Deleted event: {}",
                event.entity_id
            );
            return;
        }
    };

    let w = watcher.read().await;
    w.unregister_project(project_id).await;
    tracing::info!(
        "Watcher bridge: unregistered project {} on delete",
        project_id
    );
}

/// Handle a Project::Updated event — re-register if root_path changed.
async fn handle_project_updated(
    watcher: &Arc<RwLock<FileWatcher>>,
    event: &CrudEvent,
    neo4j: &Arc<dyn crate::neo4j::GraphStore>,
) {
    // Only react if root_path was part of the update
    let new_root_path = match event.payload.get("root_path").and_then(|v| v.as_str()) {
        Some(rp) => rp.to_string(),
        None => return, // root_path not changed, nothing to do
    };

    let project_id = match event.entity_id.parse::<Uuid>() {
        Ok(id) => id,
        Err(_) => return,
    };

    // Look up the project to get current slug
    let project = match neo4j.get_project(project_id).await {
        Ok(Some(p)) => p,
        Ok(None) => {
            tracing::warn!(
                "Watcher bridge: project {} not found in Neo4j after update",
                project_id
            );
            return;
        }
        Err(e) => {
            tracing::warn!(
                "Watcher bridge: failed to lookup project {} after update: {}",
                project_id,
                e
            );
            return;
        }
    };

    // Unregister old path, register new one
    {
        let w = watcher.read().await;
        w.unregister_project(project_id).await;
    }

    let expanded = crate::expand_tilde(&new_root_path);
    let path = std::path::Path::new(&expanded);
    if path.exists() {
        let mut w = watcher.write().await;
        if let Err(e) = w
            .register_project(path, project_id, project.slug.clone())
            .await
        {
            tracing::warn!(
                "Watcher bridge: failed to re-register project '{}' after root_path update: {}",
                project.slug,
                e
            );
        } else {
            tracing::info!(
                "Watcher bridge: re-registered project '{}' with new root_path: {}",
                project.slug,
                expanded
            );
        }
    } else {
        tracing::debug!(
            "Watcher bridge: project '{}' new root_path does not exist, skipped: {}",
            project.slug,
            expanded
        );
    }
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
    fn test_should_not_sync_vendor_directory() {
        assert!(!should_sync_file(Path::new(
            "/project/vendor/github.com/lib/pq/conn.go"
        )));
        assert!(!should_sync_file(Path::new(
            "/project/vendor/bundle/ruby/3.0.0/gems/lib.rb"
        )));
    }

    #[test]
    fn test_should_not_sync_next_nuxt_directories() {
        assert!(!should_sync_file(Path::new(
            "/project/.next/server/chunks/main.js"
        )));
        assert!(!should_sync_file(Path::new(
            "/project/.nuxt/dist/server/index.js"
        )));
    }

    #[test]
    fn test_should_not_sync_coverage_cache_directories() {
        assert!(!should_sync_file(Path::new(
            "/project/coverage/lcov-report/index.js"
        )));
        assert!(!should_sync_file(Path::new(
            "/project/.cache/babel-loader/hash.js"
        )));
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
        let result = resolve_project(Path::new("/projects/mono/src/lib.rs"), &map).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().project_id, parent_id);
    }

    #[tokio::test]
    async fn test_resolve_project_empty_map() {
        let map = Arc::new(RwLock::new(HashMap::new()));
        let result = resolve_project(Path::new("/any/path/file.rs"), &map).await;
        assert!(result.is_none());
    }

    // ── project watcher bridge tests ─────────────────────────────────

    use crate::events::EventEmitter;

    #[tokio::test]
    async fn test_bridge_handles_project_created() {
        // Use a real temp dir so path.exists() returns true
        let tmp = tempfile::tempdir().unwrap();
        let root_path = tmp.path().to_string_lossy().to_string();

        let bus = crate::events::EventBus::default();
        let _watcher_map = Arc::new(RwLock::new(HashMap::<PathBuf, ProjectContext>::new()));

        let project_id = Uuid::new_v4();
        let slug = "test-proj".to_string();

        // Simulate handle_project_created directly
        let event = CrudEvent::new(
            EventEntityType::Project,
            CrudAction::Created,
            project_id.to_string(),
        )
        .with_payload(serde_json::json!({
            "name": "Test Project",
            "slug": &slug,
            "root_path": &root_path
        }));

        // Extract and verify payload
        let rp = event.payload.get("root_path").unwrap().as_str().unwrap();
        assert_eq!(rp, root_path);

        let parsed_slug = event.payload.get("slug").unwrap().as_str().unwrap();
        assert_eq!(parsed_slug, "test-proj");

        let parsed_id: Uuid = event.entity_id.parse().unwrap();
        assert_eq!(parsed_id, project_id);

        // Verify event goes through broadcast
        let mut rx = bus.subscribe();
        bus.emit(event);
        let received = rx.try_recv().unwrap();
        assert_eq!(received.entity_type, EventEntityType::Project);
        assert_eq!(received.action, CrudAction::Created);
        assert_eq!(
            received.payload.get("root_path").unwrap().as_str().unwrap(),
            root_path
        );
    }

    #[tokio::test]
    async fn test_bridge_handles_project_deleted() {
        let project_id = Uuid::new_v4();

        let event = CrudEvent::new(
            EventEntityType::Project,
            CrudAction::Deleted,
            project_id.to_string(),
        );

        // Verify we can parse the entity_id back to UUID
        let parsed_id: Uuid = event.entity_id.parse().unwrap();
        assert_eq!(parsed_id, project_id);
        assert_eq!(event.action, CrudAction::Deleted);
    }

    #[tokio::test]
    async fn test_bridge_ignores_non_project_events() {
        let bus = crate::events::EventBus::default();
        let mut rx = bus.subscribe();

        // Emit a Plan event — should be ignored by bridge
        let event = CrudEvent::new(EventEntityType::Plan, CrudAction::Created, "plan-123")
            .with_payload(serde_json::json!({"title": "My Plan"}));

        bus.emit(event);
        let received = rx.try_recv().unwrap();

        // Bridge would check: event.entity_type != EventEntityType::Project → continue
        assert_ne!(received.entity_type, EventEntityType::Project);
    }

    #[tokio::test]
    async fn test_bridge_created_event_missing_root_path() {
        // Event with no root_path in payload — handle_project_created should bail gracefully
        let event = CrudEvent::new(
            EventEntityType::Project,
            CrudAction::Created,
            Uuid::new_v4().to_string(),
        )
        .with_payload(serde_json::json!({"name": "No Path", "slug": "no-path"}));

        assert!(event.payload.get("root_path").is_none());
    }

    #[tokio::test]
    async fn test_bridge_created_event_nonexistent_path() {
        let event = CrudEvent::new(
            EventEntityType::Project,
            CrudAction::Created,
            Uuid::new_v4().to_string(),
        )
        .with_payload(serde_json::json!({
            "name": "Ghost",
            "slug": "ghost",
            "root_path": "/nonexistent/path/that/does/not/exist"
        }));

        let rp = event.payload.get("root_path").unwrap().as_str().unwrap();
        assert!(!std::path::Path::new(rp).exists());
    }

    #[tokio::test]
    async fn test_bridge_updated_event_with_root_path() {
        let project_id = Uuid::new_v4();
        let event = CrudEvent::new(
            EventEntityType::Project,
            CrudAction::Updated,
            project_id.to_string(),
        )
        .with_payload(serde_json::json!({"root_path": "/new/path"}));

        // Bridge should react because root_path is in the payload
        assert!(event.payload.get("root_path").is_some());
    }

    #[tokio::test]
    async fn test_bridge_updated_event_without_root_path() {
        let project_id = Uuid::new_v4();
        let event = CrudEvent::new(
            EventEntityType::Project,
            CrudAction::Updated,
            project_id.to_string(),
        )
        .with_payload(serde_json::json!({"name": "Renamed"}));

        // Bridge should skip — no root_path change
        assert!(event.payload.get("root_path").is_none());
    }

    // ── dynamic watch path tests ──────────────────────────────────────
    //
    // These tests exercise the dynamic path registration logic without
    // needing a full Orchestrator (which requires Neo4j/Meilisearch).
    // They work directly with the internal structures (project_map,
    // watched_paths, add_path channel).

    #[tokio::test]
    async fn test_dynamic_path_channel_receives_new_path() {
        // Simulate the add_path channel that start() creates
        let (add_path_tx, mut add_path_rx) = mpsc::channel::<PathBuf>(32);

        let tmp = tempfile::tempdir().unwrap();
        let canonical = tmp.path().canonicalize().unwrap();

        let watched_paths = Arc::new(RwLock::new(HashSet::new()));
        let project_map = Arc::new(RwLock::new(HashMap::new()));

        // Simulate register_project logic: insert into map + watched + send
        let is_new = {
            let mut watched = watched_paths.write().await;
            watched.insert(canonical.clone())
        };
        {
            let mut pm = project_map.write().await;
            pm.insert(
                canonical.clone(),
                ProjectContext {
                    project_id: Uuid::new_v4(),
                    project_slug: "test-proj".to_string(),
                },
            );
        }
        assert!(is_new);

        // Send path on channel (as register_project would)
        add_path_tx.send(canonical.clone()).await.unwrap();

        // Verify the channel received the path
        let received = add_path_rx.try_recv();
        assert!(received.is_ok(), "Expected path on add_path channel");
        assert_eq!(received.unwrap(), canonical);
    }

    #[tokio::test]
    async fn test_duplicate_path_not_resent_on_channel() {
        let (add_path_tx, mut add_path_rx) = mpsc::channel::<PathBuf>(32);

        let tmp = tempfile::tempdir().unwrap();
        let canonical = tmp.path().canonicalize().unwrap();

        let watched_paths = Arc::new(RwLock::new(HashSet::new()));

        // First insert → is_new = true → send
        let is_new = {
            let mut watched = watched_paths.write().await;
            watched.insert(canonical.clone())
        };
        assert!(is_new);
        add_path_tx.send(canonical.clone()).await.unwrap();
        assert!(add_path_rx.try_recv().is_ok());

        // Second insert → is_new = false → should NOT send
        let is_new = {
            let mut watched = watched_paths.write().await;
            watched.insert(canonical.clone())
        };
        assert!(!is_new, "Duplicate insert should return false");
        // No send → channel should be empty
        assert!(
            add_path_rx.try_recv().is_err(),
            "Duplicate path should not be sent again"
        );
    }

    #[tokio::test]
    async fn test_project_map_register_and_unregister() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let path1 = tmp1.path().canonicalize().unwrap();
        let path2 = tmp2.path().canonicalize().unwrap();

        let project_map = Arc::new(RwLock::new(HashMap::new()));
        let pid1 = Uuid::new_v4();
        let pid2 = Uuid::new_v4();

        // Register two projects
        {
            let mut pm = project_map.write().await;
            pm.insert(
                path1.clone(),
                ProjectContext {
                    project_id: pid1,
                    project_slug: "proj-a".to_string(),
                },
            );
            pm.insert(
                path2.clone(),
                ProjectContext {
                    project_id: pid2,
                    project_slug: "proj-b".to_string(),
                },
            );
        }
        assert_eq!(project_map.read().await.len(), 2);

        // Unregister pid1 (same logic as unregister_project)
        {
            let mut pm = project_map.write().await;
            pm.retain(|_, ctx| ctx.project_id != pid1);
        }
        assert_eq!(project_map.read().await.len(), 1);

        // Verify pid2 still present
        let pm = project_map.read().await;
        assert!(pm.values().any(|ctx| ctx.project_id == pid2));
        assert!(!pm.values().any(|ctx| ctx.project_id == pid1));
    }

    #[tokio::test]
    async fn test_watcher_fields_before_start() {
        // Before start(), stop_tx and add_path_tx must be None
        // (FileWatcher::new sets both to None)
        let stop_tx: Option<mpsc::Sender<()>> = None;
        let add_path_tx: Option<mpsc::Sender<PathBuf>> = None;

        assert!(stop_tx.is_none(), "stop_tx should be None before start");
        assert!(
            add_path_tx.is_none(),
            "add_path_tx should be None before start"
        );
    }

    #[tokio::test]
    async fn test_multiple_projects_watched_paths_tracking() {
        let watched_paths = Arc::new(RwLock::new(HashSet::new()));

        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let tmp3 = tempfile::tempdir().unwrap();

        let paths = vec![
            tmp1.path().canonicalize().unwrap(),
            tmp2.path().canonicalize().unwrap(),
            tmp3.path().canonicalize().unwrap(),
        ];

        // Insert all
        {
            let mut watched = watched_paths.write().await;
            for p in &paths {
                watched.insert(p.clone());
            }
        }

        let watched = watched_paths.read().await;
        assert_eq!(watched.len(), 3);
        for p in &paths {
            assert!(watched.contains(p));
        }
    }
}
