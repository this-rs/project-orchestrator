//! Storage backend factory.
//!
//! `AppState` no longer builds the concrete storage implementations itself: it
//! asks this module for a [`StorageBackend`], selected by configuration
//! (`storage.backend`). This module is the only place in the repository allowed
//! to call `Neo4jClient::new` and `MeiliClient::new`.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::sync::Arc;

use anyhow::{bail, Result};

use crate::meilisearch::{MeiliClient, SearchStore};
use crate::neo4j::{GraphStore, Neo4jClient, StoreCapabilities};
use crate::Config;

/// Identifier of the storage backend built by default — the implementation
/// shipped today. An unknown `storage.backend` is a startup error, so an
/// operator typo fails loudly instead of silently building another engine.
pub const DEFAULT_BACKEND: &str = "default";

/// Identifier of the in-memory backend — the graph and search mock stores
/// shipped for tests (`src/neo4j/mock.rs`, `src/meilisearch/mock.rs`), selected
/// via `storage.backend = "memory"` (or `STORAGE_BACKEND=memory`). Ephemeral:
/// everything is lost at shutdown, so it is deliberately never advertised as a
/// production mode.
pub const MEMORY_BACKEND: &str = "memory";

/// Identifier of the obrain backend — the graph store on an embedded obrain
/// base (`src/obrain/`), selected via `storage.backend = "obrain"` (or
/// `STORAGE_BACKEND=obrain`) and built only with the `obrain-backend` feature.
/// Ported one domain at a time: what is not ported yet answers a named error.
pub const OBRAIN_BACKEND: &str = "obrain";

/// The assembled storage pieces of the application.
pub struct StorageBackend {
    /// Knowledge-graph store.
    pub graph: Arc<dyn GraphStore>,
    /// Full-text search store.
    pub search: Arc<dyn SearchStore>,
    /// Trajectory store (neural-routing persistence).
    pub trajectories: Arc<dyn neural_routing_runtime::TrajectoryStore>,
    /// What the built graph store actually supports.
    pub capabilities: StoreCapabilities,
}

/// Which storage backend to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    /// The default backend — the adapters in use today.
    Default,
    /// The in-memory backend — the mock graph/search stores. Ephemeral: no
    /// external service is touched, and nothing survives a restart.
    Memory,
    /// The obrain backend — an embedded obrain base, PO as its single writer.
    Obrain,
}

/// Resolve the backend to build from its configuration value. Pure selection:
/// the construction itself lives in [`StorageBackend::from_config`].
fn select_backend(storage_backend: &str) -> Result<BackendKind> {
    match storage_backend {
        DEFAULT_BACKEND => Ok(BackendKind::Default),
        MEMORY_BACKEND => Ok(BackendKind::Memory),
        OBRAIN_BACKEND => Ok(BackendKind::Obrain),
        other => bail!(
            "unknown storage backend '{other}'; accepted values: \
             ['{DEFAULT_BACKEND}', '{MEMORY_BACKEND}', '{OBRAIN_BACKEND}']"
        ),
    }
}

impl StorageBackend {
    /// Build the storage backend selected by `config.storage_backend`.
    ///
    /// This is the single selection point: a new backend is added here, never
    /// in `AppState`.
    pub async fn from_config(config: &Config) -> Result<Self> {
        match select_backend(&config.storage_backend)? {
            BackendKind::Default => Self::default_backend(config).await,
            BackendKind::Memory => Self::memory_backend(config).await,
            BackendKind::Obrain => Self::obrain_backend(config).await,
        }
    }

    /// The default backend: the graph and search adapters in use today.
    async fn default_backend(config: &Config) -> Result<Self> {
        let graph: Arc<dyn GraphStore> = Arc::new(
            Neo4jClient::new(
                &config.neo4j_uri,
                &config.neo4j_user,
                &config.neo4j_password,
            )
            .await?,
        );
        let search: Arc<dyn SearchStore> =
            Arc::new(MeiliClient::new(&config.meilisearch_url, &config.meilisearch_key).await?);

        // Trajectory store — the adapter owns its pool and its schema.
        let nr_store = crate::neo4j::trajectory::open(
            &config.neo4j_uri,
            &config.neo4j_user,
            &config.neo4j_password,
        )
        .await?;

        let capabilities = graph.capabilities();

        Ok(Self {
            graph,
            search,
            trajectories: nr_store.clone(),
            capabilities,
        })
    }

    /// The in-memory backend: the mock graph/search stores, no Neo4j client, no
    /// Meilisearch client, no Neo4j trajectory store. Nothing is persisted —
    /// the startup line says so, loud enough that nobody mistakes it for a
    /// production mode.
    ///
    /// Only compiled when the mock stores are: `cfg(test)` (unit tests) or the
    /// `mock-store` feature (integration test targets, and any dev build that
    /// wants to run PO without external services).
    #[cfg(any(test, feature = "mock-store"))]
    async fn memory_backend(_config: &Config) -> Result<Self> {
        tracing::warn!(
            "storage backend: memory — ephemeral, all data is lost at shutdown (not a production mode)"
        );

        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let search: Arc<dyn SearchStore> =
            Arc::new(crate::meilisearch::mock::MockSearchStore::new());

        let capabilities = graph.capabilities();

        Ok(Self {
            graph,
            search,
            // No trajectory persistence: neural routing still initializes, it
            // just sees an empty store (consistent with "lost at shutdown").
            trajectories: Arc::new(NoopTrajectoryStore),
            capabilities,
        })
    }

    /// Without the mock stores compiled in, `memory` cannot be built. The value
    /// is *known* (so it is not the "unknown backend" error), but it needs a
    /// build that carries the in-memory stores.
    #[cfg(not(any(test, feature = "mock-store")))]
    async fn memory_backend(_config: &Config) -> Result<Self> {
        bail!(
            "storage backend 'memory' needs the in-memory stores, which are only compiled \
             for test builds (the `mock-store` feature); build with `--features mock-store`"
        )
    }

    /// The obrain backend: the graph store on an embedded obrain base, no
    /// Neo4j client, no Meilisearch client.
    ///
    /// The base directory comes from `storage.obrain_path` (or
    /// `STORAGE_OBRAIN_PATH`) and has NO default: without it the backend
    /// refuses to start instead of guessing a location. Opening takes obrain's
    /// exclusive lock on the directory — PO is the single writer of the base.
    ///
    /// Only the ported domains answer (see `src/obrain/`); the rest of the
    /// graph contract, the whole search store and the trajectory store refuse
    /// by name. The startup line says how much that is.
    #[cfg(feature = "obrain-backend")]
    async fn obrain_backend(config: &Config) -> Result<Self> {
        use crate::obrain::{ObrainGraphStore, ObrainSearchStore, ObrainTrajectoryStore};

        let Some(path) = config
            .storage_obrain_path
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
        else {
            bail!(
                "storage backend '{OBRAIN_BACKEND}' needs the directory of the obrain base: set \
                 `storage.obrain_path` in config.yaml (nested under `storage:`) or \
                 STORAGE_OBRAIN_PATH; there is no default location"
            );
        };
        let path = std::path::PathBuf::from(crate::expand_tilde(path));

        // obrain opens synchronously (mmap, WAL replay, index rebuild).
        let store = tokio::task::spawn_blocking(move || ObrainGraphStore::open(path)).await??;

        tracing::warn!(
            path = %store.chemin().display(),
            ported = crate::obrain::methodes_portees().len(),
            refused = crate::obrain::methodes_refusees().len(),
            "storage backend: obrain — partially ported: every method outside the ported \
             domains answers the error « non porté sur le backend obrain » (not a production mode)"
        );

        let graph: Arc<dyn GraphStore> = Arc::new(store);
        let capabilities = graph.capabilities();

        Ok(Self {
            graph,
            search: Arc::new(ObrainSearchStore),
            trajectories: Arc::new(ObrainTrajectoryStore),
            capabilities,
        })
    }

    /// Without the feature, `obrain` cannot be built. The value is *known* (so
    /// it is not the "unknown backend" error), but this binary does not carry
    /// the obrain engine.
    #[cfg(not(feature = "obrain-backend"))]
    async fn obrain_backend(_config: &Config) -> Result<Self> {
        bail!(
            "storage backend '{OBRAIN_BACKEND}' is not compiled into this binary: the obrain \
             engine is behind the `obrain-backend` feature; build with `--features obrain-backend`"
        )
    }
}

/// A no-op trajectory store for the memory backend: trajectories are not
/// persisted, matching "everything is lost at shutdown". Neural routing still
/// initializes against it; reads answer empty, writes are dropped.
#[cfg(any(test, feature = "mock-store"))]
struct NoopTrajectoryStore;

#[cfg(any(test, feature = "mock-store"))]
#[async_trait::async_trait]
impl neural_routing_runtime::TrajectoryStore for NoopTrajectoryStore {
    async fn store_trajectory(
        &self,
        _t: &neural_routing_runtime::Trajectory,
    ) -> neural_routing_runtime::routing_error::Result<()> {
        Ok(())
    }

    async fn get_trajectory(
        &self,
        _id: &uuid::Uuid,
    ) -> neural_routing_runtime::routing_error::Result<Option<neural_routing_runtime::Trajectory>>
    {
        Ok(None)
    }

    async fn list_trajectories(
        &self,
        _filter: &neural_routing_runtime::TrajectoryFilter,
    ) -> neural_routing_runtime::routing_error::Result<Vec<neural_routing_runtime::Trajectory>>
    {
        Ok(vec![])
    }

    async fn search_similar(
        &self,
        _embedding: &[f32],
        _top_k: usize,
        _min_sim: f32,
    ) -> neural_routing_runtime::routing_error::Result<Vec<(neural_routing_runtime::Trajectory, f64)>>
    {
        Ok(vec![])
    }

    async fn get_stats(
        &self,
    ) -> neural_routing_runtime::routing_error::Result<neural_routing_runtime::TrajectoryStats>
    {
        Ok(neural_routing_runtime::TrajectoryStats {
            total_count: 0,
            avg_reward: 0.0,
            avg_step_count: 0.0,
            avg_duration_ms: 0.0,
            reward_distribution: Default::default(),
        })
    }

    async fn count(&self) -> neural_routing_runtime::routing_error::Result<usize> {
        Ok(0)
    }

    async fn delete_trajectory(
        &self,
        _id: &uuid::Uuid,
    ) -> neural_routing_runtime::routing_error::Result<bool> {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn default_backend_is_selected_for_the_default_value() {
        assert_eq!(
            select_backend(DEFAULT_BACKEND).unwrap(),
            BackendKind::Default
        );
    }

    #[test]
    fn memory_backend_is_selected_for_the_memory_value() {
        assert_eq!(select_backend(MEMORY_BACKEND).unwrap(), BackendKind::Memory);
    }

    #[test]
    fn obrain_backend_is_selected_for_the_obrain_value() {
        assert_eq!(select_backend(OBRAIN_BACKEND).unwrap(), BackendKind::Obrain);
    }

    #[test]
    fn unknown_backend_value_names_the_value_and_the_accepted_values() {
        let err = select_backend("mystery").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("mystery"),
            "error must name the rejected value: {msg}"
        );
        assert!(
            msg.contains(DEFAULT_BACKEND)
                && msg.contains(MEMORY_BACKEND)
                && msg.contains(OBRAIN_BACKEND),
            "error must name the accepted values: {msg}"
        );
    }
}
