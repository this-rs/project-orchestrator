//! File watcher for auto-syncing code changes
//!
//! This module watches directories for file changes and automatically
//! updates Neo4j and Meilisearch when files are modified.

use anyhow::{Context, Result};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};

use super::Orchestrator;

/// File watcher that auto-syncs changes to the knowledge base
pub struct FileWatcher {
    orchestrator: Arc<Orchestrator>,
    watched_paths: Arc<RwLock<HashSet<PathBuf>>>,
    stop_tx: Option<mpsc::Sender<()>>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(orchestrator: Arc<Orchestrator>) -> Self {
        Self {
            orchestrator,
            watched_paths: Arc::new(RwLock::new(HashSet::new())),
            stop_tx: None,
        }
    }

    /// Start watching a directory
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

    /// Start the watcher background task
    pub async fn start(&mut self) -> Result<()> {
        if self.stop_tx.is_some() {
            return Ok(()); // Already running
        }

        let (stop_tx, mut stop_rx) = mpsc::channel::<()>(1);
        let (event_tx, mut event_rx) = mpsc::channel::<PathBuf>(100);

        self.stop_tx = Some(stop_tx);

        let watched_paths = self.watched_paths.clone();
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
                                if let Err(e) = orchestrator.sync_file(&path).await {
                                    tracing::warn!("Failed to sync {}: {}", path.display(), e);
                                } else {
                                    tracing::info!("Auto-synced: {}", path.display());
                                }
                            }
                        }
                    }
                }
            }
        });

        tracing::info!("File watcher started");
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
}

/// Check if a file should be synced based on extension and path
fn should_sync_file(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();

    let supported_extensions = ["rs", "ts", "tsx", "js", "jsx", "py", "go"];

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
}
