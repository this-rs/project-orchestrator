//! Storage backend factory.
//!
//! `AppState` no longer builds the concrete storage implementations itself: it
//! asks this module for a [`StorageBackend`], selected by configuration
//! (`storage.backend`). This module is the only place in the repository allowed
//! to call `Neo4jClient::new` and `MeiliClient::new`.
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
    /// The concrete trajectory store, kept for the lazy initialization of the
    /// trajectory collector.
    ///
    /// The collector itself now takes the [`neural_routing_runtime::TrajectoryStore`]
    /// trait object; this typed handle only survives because the main crate is the
    /// adapter's home and may legitimately hold the concrete implementation it built.
    pub trajectory_store_concrete:
        Option<Arc<crate::neo4j::trajectory_store::Neo4jTrajectoryStore>>,
}

/// Which storage backend to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    /// The default backend — the adapters in use today.
    Default,
}

/// Resolve the backend to build from its configuration value. Pure selection:
/// the construction itself lives in [`StorageBackend::from_config`].
fn select_backend(storage_backend: &str) -> Result<BackendKind> {
    match storage_backend {
        DEFAULT_BACKEND => Ok(BackendKind::Default),
        other => bail!("unknown storage backend '{other}'; accepted values: ['{DEFAULT_BACKEND}']"),
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
            trajectory_store_concrete: Some(nr_store),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    #[test]
    fn default_backend_is_selected_for_the_default_value() {
        assert_eq!(
            select_backend(DEFAULT_BACKEND).unwrap(),
            BackendKind::Default
        );
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
            msg.contains(DEFAULT_BACKEND),
            "error must name the accepted values: {msg}"
        );
    }
}
