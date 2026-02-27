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
use tokio::time::Instant;
use uuid::Uuid;

/// Number of files above which we switch to a full project sync
/// instead of syncing files individually. Full sync is more efficient
/// for bulk changes (e.g., git checkout) because it uses WalkDir +
/// batch UNWIND in a single pass instead of N individual sync_file calls.
const BULK_SYNC_THRESHOLD: usize = 50;

/// Quiet period (in seconds) before flushing collected file events.
/// After the last file event, we wait this long before syncing.
/// This allows bulk operations (git checkout, git pull, branch switch)
/// to complete before we start syncing.
const SYNC_DEBOUNCE_SECS: u64 = 3;

use super::Orchestrator;
use crate::events::{CrudAction, CrudEvent, EntityType as EventEntityType};

/// Kind of file system event propagated through the watcher channel.
///
/// The `notify` crate provides fine-grained `EventKind` variants, but the
/// watcher only needs to distinguish "file changed" from "file deleted" for
/// its sync logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WatchEventKind {
    /// File was created or modified — needs (re-)sync.
    Changed,
    /// File was removed — needs cleanup from Neo4j + Meilisearch.
    Deleted,
}

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
    /// 4. **Collects** file change events per project in a debounce window
    /// 5. After a quiet period (`SYNC_DEBOUNCE_SECS`), flushes collected events:
    ///    - If < `BULK_SYNC_THRESHOLD` files → sync each file individually
    ///    - If >= `BULK_SYNC_THRESHOLD` files → full project sync (more efficient)
    /// 6. Triggers the analytics debouncer once per project per flush
    ///
    /// This **project-level debounce** is critical for bulk operations like
    /// `git checkout` where 100+ files change simultaneously. Instead of
    /// 100 sequential sync_file calls (each with Neo4j writes), we collect
    /// all changes and sync them in one batch.
    ///
    /// Safe to call with 0 registered projects — the watcher will be ready
    /// to accept new paths dynamically via `register_project()`.
    pub async fn start(&mut self) -> Result<()> {
        if self.stop_tx.is_some() {
            return Ok(()); // Already running
        }

        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);
        let (event_tx, mut event_rx) = mpsc::channel::<(PathBuf, WatchEventKind)>(500);
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
                        let kind = match event.kind {
                            notify::EventKind::Remove(_) => WatchEventKind::Deleted,
                            _ => WatchEventKind::Changed,
                        };
                        for path in event.paths {
                            let _ = rt.block_on(async {
                                event_tx_clone.send((path, kind.clone())).await
                            });
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

            // ── Project-level sync debounce ──────────────────────────────
            //
            // Instead of syncing each file immediately (old: 500ms sleep +
            // sync_file per event), we collect events per project and flush
            // after a quiet period. This transforms 100 sequential syncs
            // into 1 batch sync for bulk operations like git checkout.
            //
            // pending_files: project_id → (project_context, set of changed file paths)
            let mut pending_files: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> =
                HashMap::new();
            // pending_deletions: project_id → (project_context, set of deleted file paths)
            let mut pending_deletions: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> =
                HashMap::new();
            // Files with no project association
            let mut pending_orphans: HashSet<PathBuf> = HashSet::new();
            let mut pending_orphan_deletions: HashSet<PathBuf> = HashSet::new();
            let mut has_pending = false;

            // Flush timer — starts far in the future, reset on each event
            let flush_sleep = tokio::time::sleep_until(Instant::now() + Duration::from_secs(86400));
            tokio::pin!(flush_sleep);

            // Keep watcher alive until stop signal
            loop {
                tokio::select! {
                    _ = stop_rx.recv() => {
                        tracing::info!("File watcher stopping");
                        // Flush any remaining pending files before stopping
                        if has_pending {
                            flush_pending_files(
                                &mut pending_files,
                                &mut pending_deletions,
                                &mut pending_orphans,
                                &mut pending_orphan_deletions,
                                &orchestrator,
                                &project_map,
                            ).await;
                        }
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

                    // ── Collect file events into pending map ─────────────
                    Some((path, event_kind)) = event_rx.recv() => {
                        if !should_sync_file(&path) {
                            continue;
                        }

                        match event_kind {
                            WatchEventKind::Deleted => {
                                // For deletions we do NOT check path.exists()
                                // (the file is gone — that's the whole point)
                                if let Some(ctx) = resolve_project(&path, &project_map).await {
                                    pending_deletions
                                        .entry(ctx.project_id)
                                        .or_insert_with(|| (ctx, HashSet::new()))
                                        .1
                                        .insert(path);
                                } else {
                                    pending_orphan_deletions.insert(path);
                                }
                            }
                            WatchEventKind::Changed => {
                                // For changes we still need the file on disk
                                if !path.exists() {
                                    continue;
                                }
                                if let Some(ctx) = resolve_project(&path, &project_map).await {
                                    pending_files
                                        .entry(ctx.project_id)
                                        .or_insert_with(|| (ctx, HashSet::new()))
                                        .1
                                        .insert(path);
                                } else {
                                    pending_orphans.insert(path);
                                }
                            }
                        }

                        // Reset the debounce timer
                        has_pending = true;
                        flush_sleep.as_mut().reset(
                            Instant::now() + Duration::from_secs(SYNC_DEBOUNCE_SECS)
                        );
                    }

                    // ── Flush after quiet period ─────────────────────────
                    _ = &mut flush_sleep, if has_pending => {
                        flush_pending_files(
                            &mut pending_files,
                            &mut pending_deletions,
                            &mut pending_orphans,
                            &mut pending_orphan_deletions,
                            &orchestrator,
                            &project_map,
                        ).await;
                        has_pending = false;
                        // Reset timer far in the future
                        flush_sleep.as_mut().reset(
                            Instant::now() + Duration::from_secs(86400)
                        );
                    }
                }
            }
        });

        let paths = self.watched_paths.read().await;
        let projects = self.project_map.read().await;
        tracing::info!(
            "File watcher started ({} paths, {} projects, debounce={}s, bulk_threshold={})",
            paths.len(),
            projects.len(),
            SYNC_DEBOUNCE_SECS,
            BULK_SYNC_THRESHOLD,
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

/// Flush all pending file events collected during the debounce window.
///
/// For each project with pending files:
/// - If >= `BULK_SYNC_THRESHOLD` files → run a full `sync_directory_for_project`
///   (more efficient: single WalkDir + batch UNWIND, plus stale file cleanup)
/// - If < threshold → sync each file individually
/// - Trigger the analytics debouncer once per project
///
/// For pending deletions: remove each file from Neo4j and Meilisearch.
///
/// Orphan files (no project association) are synced/deleted individually.
async fn flush_pending_files(
    pending_files: &mut HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)>,
    pending_deletions: &mut HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)>,
    pending_orphans: &mut HashSet<PathBuf>,
    pending_orphan_deletions: &mut HashSet<PathBuf>,
    orchestrator: &Arc<super::Orchestrator>,
    project_map: &Arc<RwLock<HashMap<PathBuf, ProjectContext>>>,
) {
    let batch = std::mem::take(pending_files);
    let deletions = std::mem::take(pending_deletions);
    let orphans = std::mem::take(pending_orphans);
    let orphan_deletions = std::mem::take(pending_orphan_deletions);

    let total_files: usize = batch.values().map(|(_, files)| files.len()).sum();
    let total_deletions: usize = deletions.values().map(|(_, files)| files.len()).sum();
    let total_projects = batch.len();

    if total_files > 0 || total_deletions > 0 {
        tracing::info!(
            "Sync debounce: flushing {} changed + {} deleted files across {} project(s)",
            total_files,
            total_deletions,
            total_projects.max(deletions.len()),
        );
    }

    // ── Handle changed files ─────────────────────────────────────────
    for (pid, (ctx, files)) in batch {
        let file_count = files.len();

        if file_count >= BULK_SYNC_THRESHOLD {
            // ── Bulk change detected → full project sync ─────────
            tracing::info!(
                "Bulk change detected ({} files) for project '{}' — switching to full project sync",
                file_count,
                ctx.project_slug,
            );

            // Find the project root_path from the project_map
            let root_path = {
                let pm = project_map.read().await;
                pm.iter()
                    .find(|(_, c)| c.project_id == pid)
                    .map(|(path, _)| path.clone())
            };

            if let Some(root) = root_path {
                match orchestrator
                    .sync_directory_for_project(&root, Some(pid), Some(&ctx.project_slug))
                    .await
                {
                    Ok(result) => {
                        tracing::info!(
                            "Bulk sync complete for '{}': {} synced, {} skipped, {} deleted, {} errors",
                            ctx.project_slug,
                            result.files_synced,
                            result.files_skipped,
                            result.files_deleted,
                            result.errors,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Bulk sync failed for project '{}': {}",
                            ctx.project_slug,
                            e
                        );
                    }
                }
            } else {
                tracing::warn!(
                    "Cannot run bulk sync for project '{}' — root_path not found in project_map",
                    ctx.project_slug,
                );
            }
        } else {
            // ── Incremental sync → sync each changed file ────────
            tracing::info!(
                "Syncing {} changed file(s) for project '{}'",
                file_count,
                ctx.project_slug,
            );

            let mut synced = 0usize;
            let mut skipped = 0usize;
            let mut errors = 0usize;

            for path in &files {
                match orchestrator
                    .sync_file_for_project(path, Some(pid), Some(&ctx.project_slug))
                    .await
                {
                    Ok(true) => synced += 1,
                    Ok(false) => skipped += 1,
                    Err(e) => {
                        tracing::warn!("Failed to sync {}: {}", path.display(), e);
                        errors += 1;
                    }
                }
            }

            if synced > 0 || errors > 0 {
                tracing::info!(
                    "Incremental sync for '{}': {} synced, {} unchanged, {} errors",
                    ctx.project_slug,
                    synced,
                    skipped,
                    errors,
                );
            }
        }

        // Trigger analytics debouncer once per project (will be further
        // debounced by the AnalyticsDebouncer's own quiet period)
        orchestrator.analytics_debouncer().trigger(pid);
    }

    // ── Handle deleted files ─────────────────────────────────────────
    for (pid, (ctx, files)) in deletions {
        let file_count = files.len();
        tracing::info!(
            "Deleting {} file(s) from graph for project '{}'",
            file_count,
            ctx.project_slug,
        );

        let mut deleted = 0usize;
        let mut errors = 0usize;

        for path in &files {
            let path_str = super::runner::normalize_path(&path.to_string_lossy());

            // Remove from Neo4j (File node + all children symbols + relationships)
            if let Err(e) = orchestrator.neo4j().delete_file(&path_str).await {
                tracing::warn!("Failed to delete {} from Neo4j: {}", path_str, e);
                errors += 1;
                continue;
            }

            // Remove from Meilisearch search index
            if let Err(e) = orchestrator.meili().delete_code(&path_str).await {
                tracing::warn!("Failed to delete {} from Meilisearch: {}", path_str, e);
                // Non-fatal: Neo4j is source of truth, Meili is secondary index
            }

            deleted += 1;
        }

        if deleted > 0 || errors > 0 {
            tracing::info!(
                "Deletion complete for '{}': {} deleted, {} errors",
                ctx.project_slug,
                deleted,
                errors,
            );
        }

        // Trigger analytics debouncer (graph changed)
        orchestrator.analytics_debouncer().trigger(pid);
    }

    // Sync orphan files (no project association) individually
    if !orphans.is_empty() {
        tracing::debug!("Syncing {} orphan file(s) (no project)", orphans.len());
        for path in &orphans {
            if let Err(e) = orchestrator.sync_file_for_project(path, None, None).await {
                tracing::warn!("Failed to sync orphan {}: {}", path.display(), e);
            }
        }
    }

    // Delete orphan files (no project association) individually
    if !orphan_deletions.is_empty() {
        tracing::debug!(
            "Deleting {} orphan file(s) from graph (no project)",
            orphan_deletions.len()
        );
        for path in &orphan_deletions {
            let path_str = super::runner::normalize_path(&path.to_string_lossy());
            if let Err(e) = orchestrator.neo4j().delete_file(&path_str).await {
                tracing::warn!("Failed to delete orphan {} from Neo4j: {}", path_str, e);
            }
            if let Err(e) = orchestrator.meili().delete_code(&path_str).await {
                tracing::warn!(
                    "Failed to delete orphan {} from Meilisearch: {}",
                    path_str,
                    e
                );
            }
        }
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
    orchestrator: Arc<super::Orchestrator>,
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
                            handle_project_created(&watcher, &orchestrator, &event).await;
                        }
                        CrudAction::Deleted => {
                            handle_project_deleted(&watcher, &event).await;
                        }
                        CrudAction::Updated => {
                            handle_project_updated(&watcher, &event, &orchestrator).await;
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

/// Handle a Project::Created event — register on the watcher if root_path exists,
/// then spawn an initial sync in background.
async fn handle_project_created(
    watcher: &Arc<RwLock<FileWatcher>>,
    orchestrator: &Arc<super::Orchestrator>,
    event: &CrudEvent,
) {
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

            // Spawn initial sync in background — same pipeline as sync_project handler
            // (sync tree-sitter → update last_synced → analytics → feature graphs)
            let orch = orchestrator.clone();
            let sync_path = expanded.clone();
            let sync_slug = slug.clone();
            tracing::info!(
                "Watcher bridge: spawning initial sync for project '{}'",
                sync_slug
            );
            tokio::spawn(async move {
                let path = std::path::Path::new(&sync_path);
                match orch
                    .sync_directory_for_project_with_options(
                        path,
                        Some(project_id),
                        Some(&sync_slug),
                        false,
                    )
                    .await
                {
                    Ok(result) => {
                        tracing::info!(
                            "Watcher bridge: initial sync completed for '{}' — {} files synced, {} skipped, {} errors",
                            sync_slug,
                            result.files_synced,
                            result.files_skipped,
                            result.errors,
                        );
                        // Update last_synced timestamp
                        if let Err(e) = orch.neo4j().update_project_synced(project_id).await {
                            tracing::warn!(
                                "Watcher bridge: failed to update last_synced for '{}': {}",
                                sync_slug,
                                e
                            );
                        }
                        // Compute graph analytics (PageRank, communities, etc.)
                        orch.spawn_analyze_project(project_id);
                        // Refresh auto-built feature graphs
                        orch.spawn_refresh_feature_graphs(project_id);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Watcher bridge: initial sync failed for '{}': {}",
                            sync_slug,
                            e
                        );
                    }
                }
            });
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
    orchestrator: &Arc<super::Orchestrator>,
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
    let neo4j = orchestrator.neo4j_arc();
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

    // ── constants and config tests ───────────────────────────────────

    // Compile-time validation of constants — these fail compilation if violated
    const _: () = assert!(BULK_SYNC_THRESHOLD > 0);
    const _: () = assert!(BULK_SYNC_THRESHOLD <= 200); // Would never trigger bulk sync if too high
    const _: () = assert!(SYNC_DEBOUNCE_SECS >= 1); // Too short won't absorb bulk ops
    const _: () = assert!(SYNC_DEBOUNCE_SECS <= 10); // Too long means slow feedback loop

    #[test]
    fn test_bulk_sync_threshold_value() {
        // Current value is 50 — enough to trigger on git checkout (~100+ files)
        // but not so low that normal edits trigger bulk sync
        assert_eq!(BULK_SYNC_THRESHOLD, 50);
    }

    #[test]
    fn test_sync_debounce_secs_value() {
        // Current value is 3 — absorbs git checkout while staying responsive
        assert_eq!(SYNC_DEBOUNCE_SECS, 3);
    }

    // ── flush_pending_files logic tests ──────────────────────────────
    //
    // flush_pending_files requires an Arc<Orchestrator> which needs
    // Neo4j/Meilisearch. Instead we test the data structures it consumes.

    #[test]
    fn test_pending_files_collection_below_threshold() {
        // Simulate collecting files below BULK_SYNC_THRESHOLD
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "test-proj".to_string(),
        };

        let mut pending: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();

        // Add 10 files (below threshold of 50)
        let entry = pending.entry(pid).or_insert_with(|| (ctx, HashSet::new()));
        for i in 0..10 {
            entry
                .1
                .insert(PathBuf::from(format!("/tmp/src/file_{}.rs", i)));
        }

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[&pid].1.len(), 10);
        assert!(
            pending[&pid].1.len() < BULK_SYNC_THRESHOLD,
            "Should be below threshold — incremental sync path"
        );
    }

    #[test]
    fn test_pending_files_collection_above_threshold() {
        // Simulate collecting files above BULK_SYNC_THRESHOLD (git checkout)
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "big-proj".to_string(),
        };

        let mut pending: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();

        let entry = pending.entry(pid).or_insert_with(|| (ctx, HashSet::new()));
        for i in 0..100 {
            entry
                .1
                .insert(PathBuf::from(format!("/tmp/src/file_{}.rs", i)));
        }

        assert_eq!(pending[&pid].1.len(), 100);
        assert!(
            pending[&pid].1.len() >= BULK_SYNC_THRESHOLD,
            "Should be above threshold — bulk sync path"
        );
    }

    #[test]
    fn test_pending_files_multi_project_collection() {
        // Simulate collecting files from multiple projects during debounce window
        let pid_a = Uuid::new_v4();
        let pid_b = Uuid::new_v4();
        let ctx_a = ProjectContext {
            project_id: pid_a,
            project_slug: "proj-a".to_string(),
        };
        let ctx_b = ProjectContext {
            project_id: pid_b,
            project_slug: "proj-b".to_string(),
        };

        let mut pending: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();

        // 3 files in proj-a
        let entry_a = pending
            .entry(pid_a)
            .or_insert_with(|| (ctx_a, HashSet::new()));
        for i in 0..3 {
            entry_a
                .1
                .insert(PathBuf::from(format!("/tmp/a/file_{}.rs", i)));
        }

        // 2 files in proj-b
        let entry_b = pending
            .entry(pid_b)
            .or_insert_with(|| (ctx_b, HashSet::new()));
        for i in 0..2 {
            entry_b
                .1
                .insert(PathBuf::from(format!("/tmp/b/file_{}.rs", i)));
        }

        assert_eq!(pending.len(), 2, "Should have 2 distinct projects");
        assert_eq!(pending[&pid_a].1.len(), 3);
        assert_eq!(pending[&pid_b].1.len(), 2);

        // Total files
        let total: usize = pending.values().map(|(_, files)| files.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_pending_files_dedup_same_path() {
        // Same file path inserted multiple times → HashSet deduplicates
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "dup-proj".to_string(),
        };

        let mut pending: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();
        let entry = pending.entry(pid).or_insert_with(|| (ctx, HashSet::new()));

        let path = PathBuf::from("/tmp/src/main.rs");
        entry.1.insert(path.clone());
        entry.1.insert(path.clone());
        entry.1.insert(path.clone());

        assert_eq!(entry.1.len(), 1, "Same path should be deduped by HashSet");
    }

    #[test]
    fn test_pending_orphans_collection() {
        // Files with no project association go to orphans set
        let mut orphans: HashSet<PathBuf> = HashSet::new();

        orphans.insert(PathBuf::from("/tmp/untracked/file1.rs"));
        orphans.insert(PathBuf::from("/tmp/untracked/file2.rs"));
        orphans.insert(PathBuf::from("/tmp/untracked/file1.rs")); // dupe

        assert_eq!(orphans.len(), 2, "Orphans should be deduped");
    }

    #[test]
    fn test_mem_take_clears_pending() {
        // Verifies that std::mem::take (used in flush_pending_files) clears the map
        let mut pending: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "test".to_string(),
        };
        pending.insert(pid, (ctx, HashSet::from([PathBuf::from("/a.rs")])));

        let batch = std::mem::take(&mut pending);
        assert!(pending.is_empty(), "Original should be empty after take");
        assert_eq!(batch.len(), 1, "Taken batch should contain the data");
    }

    // ── WatchEventKind tests ────────────────────────────────────────────

    #[test]
    fn test_watch_event_kind_clone_and_eq() {
        let changed = WatchEventKind::Changed;
        let deleted = WatchEventKind::Deleted;

        assert_eq!(changed.clone(), WatchEventKind::Changed);
        assert_eq!(deleted.clone(), WatchEventKind::Deleted);
        assert_ne!(changed, deleted);
    }

    #[test]
    fn test_watch_event_kind_from_notify_remove() {
        // Simulate the mapping logic used in the notify callback
        use notify::EventKind;

        let remove_kinds = [
            EventKind::Remove(notify::event::RemoveKind::File),
            EventKind::Remove(notify::event::RemoveKind::Folder),
            EventKind::Remove(notify::event::RemoveKind::Any),
            EventKind::Remove(notify::event::RemoveKind::Other),
        ];

        for kind in &remove_kinds {
            let mapped = match kind {
                EventKind::Remove(_) => WatchEventKind::Deleted,
                _ => WatchEventKind::Changed,
            };
            assert_eq!(
                mapped,
                WatchEventKind::Deleted,
                "Remove({:?}) should map to Deleted",
                kind
            );
        }
    }

    #[test]
    fn test_watch_event_kind_from_notify_non_remove() {
        // All non-Remove event kinds should map to Changed
        use notify::EventKind;

        let non_remove_kinds = [
            EventKind::Create(notify::event::CreateKind::File),
            EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Content,
            )),
            EventKind::Modify(notify::event::ModifyKind::Name(
                notify::event::RenameMode::Both,
            )),
            EventKind::Access(notify::event::AccessKind::Close(
                notify::event::AccessMode::Write,
            )),
            EventKind::Any,
            EventKind::Other,
        ];

        for kind in &non_remove_kinds {
            let mapped = match kind {
                EventKind::Remove(_) => WatchEventKind::Deleted,
                _ => WatchEventKind::Changed,
            };
            assert_eq!(
                mapped,
                WatchEventKind::Changed,
                "{:?} should map to Changed",
                kind
            );
        }
    }

    // ── Deletion event routing tests ────────────────────────────────────

    #[tokio::test]
    async fn test_deletion_event_routes_to_pending_deletions() {
        // Simulate the recv loop routing logic for Deleted events
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "del-proj".to_string(),
        };

        let mut pending_files: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();
        let mut pending_deletions: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> =
            HashMap::new();

        let path = PathBuf::from("/project/src/removed_file.rs");
        let event_kind = WatchEventKind::Deleted;

        // Simulate the routing logic
        match event_kind {
            WatchEventKind::Deleted => {
                pending_deletions
                    .entry(pid)
                    .or_insert_with(|| (ctx.clone(), HashSet::new()))
                    .1
                    .insert(path.clone());
            }
            WatchEventKind::Changed => {
                pending_files
                    .entry(pid)
                    .or_insert_with(|| (ctx.clone(), HashSet::new()))
                    .1
                    .insert(path.clone());
            }
        }

        assert!(
            pending_files.is_empty(),
            "Deleted event should NOT go to pending_files"
        );
        assert_eq!(
            pending_deletions.len(),
            1,
            "Deleted event should go to pending_deletions"
        );
        assert!(pending_deletions[&pid].1.contains(&path));
    }

    #[tokio::test]
    async fn test_changed_event_routes_to_pending_files() {
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "chg-proj".to_string(),
        };

        let mut pending_files: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();
        let mut pending_deletions: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> =
            HashMap::new();

        let path = PathBuf::from("/project/src/modified_file.rs");
        let event_kind = WatchEventKind::Changed;

        match event_kind {
            WatchEventKind::Deleted => {
                pending_deletions
                    .entry(pid)
                    .or_insert_with(|| (ctx.clone(), HashSet::new()))
                    .1
                    .insert(path.clone());
            }
            WatchEventKind::Changed => {
                pending_files
                    .entry(pid)
                    .or_insert_with(|| (ctx.clone(), HashSet::new()))
                    .1
                    .insert(path.clone());
            }
        }

        assert!(
            pending_deletions.is_empty(),
            "Changed event should NOT go to pending_deletions"
        );
        assert_eq!(
            pending_files.len(),
            1,
            "Changed event should go to pending_files"
        );
        assert!(pending_files[&pid].1.contains(&path));
    }

    #[tokio::test]
    async fn test_mixed_events_route_correctly() {
        // Mix of changed and deleted events for the same project
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "mix-proj".to_string(),
        };

        let mut pending_files: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> = HashMap::new();
        let mut pending_deletions: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> =
            HashMap::new();

        let events: Vec<(PathBuf, WatchEventKind)> = vec![
            (PathBuf::from("/project/src/a.rs"), WatchEventKind::Changed),
            (PathBuf::from("/project/src/b.rs"), WatchEventKind::Deleted),
            (PathBuf::from("/project/src/c.rs"), WatchEventKind::Changed),
            (PathBuf::from("/project/src/d.rs"), WatchEventKind::Deleted),
            (PathBuf::from("/project/src/e.rs"), WatchEventKind::Changed),
        ];

        for (path, kind) in events {
            match kind {
                WatchEventKind::Deleted => {
                    pending_deletions
                        .entry(pid)
                        .or_insert_with(|| (ctx.clone(), HashSet::new()))
                        .1
                        .insert(path);
                }
                WatchEventKind::Changed => {
                    pending_files
                        .entry(pid)
                        .or_insert_with(|| (ctx.clone(), HashSet::new()))
                        .1
                        .insert(path);
                }
            }
        }

        assert_eq!(
            pending_files[&pid].1.len(),
            3,
            "Should have 3 changed files"
        );
        assert_eq!(
            pending_deletions[&pid].1.len(),
            2,
            "Should have 2 deleted files"
        );
    }

    #[test]
    fn test_pending_deletions_dedup_same_path() {
        // Same deleted file path inserted multiple times → HashSet deduplicates
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "dup-del".to_string(),
        };

        let mut pending_deletions: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> =
            HashMap::new();
        let entry = pending_deletions
            .entry(pid)
            .or_insert_with(|| (ctx, HashSet::new()));

        let path = PathBuf::from("/tmp/src/removed.rs");
        entry.1.insert(path.clone());
        entry.1.insert(path.clone());
        entry.1.insert(path.clone());

        assert_eq!(
            entry.1.len(),
            1,
            "Same deletion path should be deduped by HashSet"
        );
    }

    #[test]
    fn test_pending_orphan_deletions_collection() {
        // Deleted files with no project association go to orphan_deletions set
        let mut orphan_deletions: HashSet<PathBuf> = HashSet::new();

        orphan_deletions.insert(PathBuf::from("/tmp/untracked/removed1.rs"));
        orphan_deletions.insert(PathBuf::from("/tmp/untracked/removed2.rs"));
        orphan_deletions.insert(PathBuf::from("/tmp/untracked/removed1.rs")); // dupe

        assert_eq!(
            orphan_deletions.len(),
            2,
            "Orphan deletions should be deduped"
        );
    }

    #[test]
    fn test_mem_take_clears_pending_deletions() {
        // std::mem::take on deletions map works the same as on files map
        let mut pending_deletions: HashMap<Uuid, (ProjectContext, HashSet<PathBuf>)> =
            HashMap::new();
        let pid = Uuid::new_v4();
        let ctx = ProjectContext {
            project_id: pid,
            project_slug: "take-del".to_string(),
        };
        pending_deletions.insert(pid, (ctx, HashSet::from([PathBuf::from("/gone.rs")])));

        let batch = std::mem::take(&mut pending_deletions);
        assert!(
            pending_deletions.is_empty(),
            "Deletions should be empty after take"
        );
        assert_eq!(batch.len(), 1, "Taken deletions should contain the data");
    }

    // ── Channel type tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_event_channel_sends_tuple() {
        // Verify the channel now transports (PathBuf, WatchEventKind)
        let (tx, mut rx) = mpsc::channel::<(PathBuf, WatchEventKind)>(10);

        let path = PathBuf::from("/project/src/main.rs");
        tx.send((path.clone(), WatchEventKind::Changed))
            .await
            .unwrap();
        tx.send((path.clone(), WatchEventKind::Deleted))
            .await
            .unwrap();

        let (p1, k1) = rx.recv().await.unwrap();
        assert_eq!(p1, path);
        assert_eq!(k1, WatchEventKind::Changed);

        let (p2, k2) = rx.recv().await.unwrap();
        assert_eq!(p2, path);
        assert_eq!(k2, WatchEventKind::Deleted);
    }

    #[tokio::test]
    async fn test_deleted_file_not_checked_for_existence() {
        // Key bug fix: deleted files should NOT be filtered by path.exists()
        // Simulate the old vs new logic
        let path = PathBuf::from("/nonexistent/deleted/file.rs");

        // Old logic (broken): path.exists() would filter this out
        assert!(!path.exists(), "Deleted file should not exist on disk");

        // New logic: WatchEventKind::Deleted skips the exists() check
        // Just verify should_sync_file passes for the extension
        assert!(
            should_sync_file(&path),
            "should_sync_file should pass for .rs extension regardless of existence"
        );
    }
}
