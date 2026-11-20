//! Storage backend factory.
//!
//! `AppState` no longer builds the concrete storage implementations itself: it
//! asks this module for a [`StorageBackend`], selected by configuration
//! (`storage.backend`). This module is the only place in the repository allowed
//! to call `Neo4jClient::new` and `MeiliClient::new`.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod migrate;

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

/// What [`StorageBackend::resolve_kind`] chose: the backend, the configuration
/// to build it with (its obrain path is the verified target), the scratch
/// directory of a disposable run (owned by the store that serves it) and the
/// read-only scope, held until the backend is built.
struct Chosen {
    kind: BackendKind,
    config: Config,
    scratch: Option<tempfile::TempDir>,
    read_only: Option<migrate::ReadOnlyScope>,
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
        Self::from_config_with(config, &migrate::NoMigrator).await
    }

    /// [`Self::from_config`] with the data-copy seam injected (the migrator
    /// lives outside this repository). Raises the PROCESS gate during a
    /// migration.
    pub async fn from_config_with(
        config: &Config,
        migrator: &dyn migrate::Migrator,
    ) -> Result<Self> {
        Self::from_config_in(config, migrator, migrate::process_gate()).await
    }

    /// [`Self::from_config_with`] on an explicit read-only gate (tests use a
    /// private one, so they never raise the gate other tests' routers read).
    pub async fn from_config_in(
        config: &Config,
        migrator: &dyn migrate::Migrator,
        gate: &migrate::ReadOnlyGate,
    ) -> Result<Self> {
        // The gate is held until the chosen backend is BUILT: `start_server`
        // creates every route and background writer after `AppState::new`
        // returns, so nothing can write while a migration runs.
        let chosen = Self::resolve_kind(config, migrator, gate).await?;
        let built = match chosen.kind {
            BackendKind::Default => Self::default_backend(&chosen.config).await,
            BackendKind::Memory => Self::memory_backend(&chosen.config).await,
            BackendKind::Obrain => Self::obrain_backend(&chosen.config, chosen.scratch).await,
        };
        drop(chosen.read_only);
        built
    }

    /// Decide which backend to build, running the startup migration first when
    /// one is configured (`storage.migration.from`). See
    /// [`migrate::run_startup_migration`] for the rules; in short:
    ///
    /// - the only migration is `default` → `obrain` (any other target is a
    ///   named refusal: copying into an in-memory store makes no sense);
    /// - a target already migrated (its marker) is served without a copy;
    /// - a rollback stays on the source and says so at `error` level (the
    ///   writes made on the target since the switch are not on the source);
    /// - a disposable run that fails refuses to start;
    /// - a non-disposable verification failure stays on the source, by name.
    async fn resolve_kind(
        config: &Config,
        migrator: &dyn migrate::Migrator,
        gate: &migrate::ReadOnlyGate,
    ) -> Result<Chosen> {
        let requested = select_backend(&config.storage_backend)?;
        let mig = &config.storage_migration;
        let plain = |kind| Chosen {
            kind,
            config: config.clone(),
            scratch: None,
            read_only: None,
        };

        let Some(from) = mig.from.as_deref() else {
            // No migration configured. The rollback still re-selects the
            // source (it never touches its data).
            if mig.rollback && requested != BackendKind::Default {
                tracing::error!(
                    requested = ?requested,
                    "storage: rollback demanded by configuration — staying on the source \
                     backend ('default'); writes made on the requested backend are not visible \
                     there"
                );
                return Ok(plain(BackendKind::Default));
            }
            return Ok(plain(requested));
        };
        if select_backend(from)? != BackendKind::Default {
            bail!("storage.migration.from = '{from}': only the '{DEFAULT_BACKEND}' source can be migrated from");
        }
        if requested == BackendKind::Default {
            return Ok(plain(requested));
        }
        if requested != BackendKind::Obrain {
            bail!(
                "storage.migration.from = '{from}' with storage.backend = '{}': a migration only \
                 targets '{OBRAIN_BACKEND}' (a copy into '{}' would be verified, then lost)",
                config.storage_backend,
                config.storage_backend
            );
        }

        // Read-only BEFORE the source is read, held until the backend is built.
        let scope = gate.enter();
        let request = migrate::MigrationRequest {
            source: from.to_string(),
            target: config.storage_backend.clone(),
            rollback: mig.rollback,
            disposable: mig.disposable,
            configured_dest: config
                .storage_obrain_path
                .as_deref()
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .map(|p| std::path::PathBuf::from(crate::expand_tilde(p))),
        };
        match migrate::run_startup_migration(&request, migrator, &scope).await? {
            migrate::StartupOutcome::ServeTarget {
                dest,
                already_migrated,
                scratch,
            } => {
                let mut built = config.clone();
                built.storage_obrain_path = Some(dest.display().to_string());
                if already_migrated {
                    tracing::info!(target = %dest.display(),
                        "storage: target already migrated and verified (marker present) — serving it, no copy");
                } else {
                    tracing::info!(target = %dest.display(), disposable = mig.disposable,
                        "storage: per-type verification passed — switching to the target");
                }
                Ok(Chosen {
                    kind: BackendKind::Obrain,
                    config: built,
                    scratch,
                    read_only: Some(scope),
                })
            }
            migrate::StartupOutcome::StayOnSource { reason } => {
                if mig.rollback {
                    tracing::error!(%reason, "storage: staying on the source backend");
                } else {
                    tracing::warn!(%reason, "storage: staying on the source backend");
                }
                Ok(Chosen {
                    kind: BackendKind::Default,
                    config: config.clone(),
                    scratch: None,
                    read_only: Some(scope),
                })
            }
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
    /// refuses to start instead of guessing a location. It must be absolute and
    /// hold a base; creating one needs `storage.obrain_create: true`. Opening takes obrain's
    /// exclusive lock on the directory — PO is the single writer of the base.
    ///
    /// Only the ported domains answer (see `src/obrain/`); the rest of the
    /// graph contract, the whole search store and the trajectory store refuse
    /// by name. The startup line says how much that is.
    #[cfg(feature = "obrain-backend")]
    async fn obrain_backend(config: &Config, scratch: Option<tempfile::TempDir>) -> Result<Self> {
        use crate::obrain::{
            Creation, ObrainGraphStore, ObrainSearchStore, ObrainTrajectoryStore, RapportOuverture,
        };

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
        // Creating a base is a decision, not a side effect of a typo: see
        // `storage.obrain_create`.
        let creation = if config.storage_obrain_create {
            Creation::Permise
        } else {
            Creation::Interdite
        };
        // A disposable run's scratch copy is owned by its store: removed when
        // the store is dropped, never left behind.
        let store = tokio::task::spawn_blocking(move || match scratch {
            Some(dir) => ObrainGraphStore::open_disposable(dir),
            None => ObrainGraphStore::open(path, creation),
        })
        .await??;

        let rapport = store.rapport_d_ouverture();
        if *rapport != RapportOuverture::default() {
            tracing::warn!(
                ?rapport,
                "obrain base repaired at open (interrupted writes) — quarantined nodes, if any, \
                 are left untouched and ignored"
            );
        }
        tracing::warn!(
            path = %store.chemin().display(),
            ported = crate::obrain::methodes_qui_servent().len(),
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
    async fn obrain_backend(_config: &Config, _scratch: Option<tempfile::TempDir>) -> Result<Self> {
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

    // ── Startup migration through the real factory ───────────────────────

    use super::migrate::{Migrator, PerTypeCounts, ReadOnlyGate, DONE_MARKER};

    /// Source counts fixed; the "copy" writes them (minus `lose` Project nodes)
    /// into `dest` and `target_counts` reads them back from there. Asserts it
    /// runs under ITS gate (a private one: the process gate is never raised).
    struct Scripted {
        gate: ReadOnlyGate,
        lose: usize,
        copied: std::sync::atomic::AtomicBool,
    }
    fn data() -> PerTypeCounts {
        PerTypeCounts {
            nodes: [("Project".to_string(), 3usize), ("Note".to_string(), 7)].into(),
            relationships: [("LINKED_TO".to_string(), 4usize)].into(),
        }
    }
    #[async_trait::async_trait]
    impl Migrator for Scripted {
        async fn source_counts(&self) -> Result<PerTypeCounts> {
            assert!(self.gate.is_read_only());
            Ok(data())
        }
        async fn copy(&self, dest: &std::path::Path) -> Result<()> {
            assert!(self.gate.is_read_only());
            self.copied.store(true, std::sync::atomic::Ordering::SeqCst);
            let mut c = data();
            *c.nodes.get_mut("Project").unwrap() -= self.lose;
            std::fs::write(
                dest.join("n.json"),
                serde_json::to_string(&(c.nodes, c.relationships))?,
            )?;
            Ok(())
        }
        async fn target_counts(&self, dest: &std::path::Path) -> Result<PerTypeCounts> {
            let (nodes, relationships) =
                serde_json::from_str(&std::fs::read_to_string(dest.join("n.json"))?)?;
            Ok(PerTypeCounts {
                nodes,
                relationships,
            })
        }
    }
    fn scripted(gate: &ReadOnlyGate, lose: usize) -> Scripted {
        Scripted {
            gate: gate.clone(),
            lose,
            copied: Default::default(),
        }
    }

    fn migrating_config(target: &str, dir: &std::path::Path) -> Config {
        let file = dir.join("config.yaml");
        std::fs::write(&file, "setup_completed: true\n").unwrap();
        let mut c = Config::from_yaml_and_env(Some(&file)).unwrap();
        c.storage_backend = target.to_string();
        c.storage_obrain_path = Some(dir.join("target").display().to_string());
        c.storage_migration = crate::StorageMigrationConfig {
            rollback: false,
            disposable: false,
            from: Some("default".to_string()),
        };
        c
    }

    #[tokio::test]
    async fn the_factory_switches_to_the_verified_target_path_and_holds_the_gate() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let config = migrating_config(OBRAIN_BACKEND, dir.path());
        let m = scripted(&gate, 0);
        let chosen = StorageBackend::resolve_kind(&config, &m, &gate)
            .await
            .unwrap();
        assert_eq!(chosen.kind, BackendKind::Obrain);
        assert_eq!(
            chosen.config.storage_obrain_path,
            Some(dir.path().join("target").display().to_string()),
            "the backend is built on the path that was verified"
        );
        assert!(m.copied.load(std::sync::atomic::Ordering::SeqCst));
        assert!(dir.path().join("target").join(DONE_MARKER).is_file());
        assert!(
            chosen.read_only.is_some() && gate.is_read_only(),
            "still read-only until the backend is built"
        );
        drop(chosen);
        assert!(!gate.is_read_only());
    }

    /// Regression (MEDIUM): a migration into `memory` used to be accepted —
    /// verified, then served from an empty in-memory store.
    #[tokio::test]
    async fn a_migration_only_targets_obrain() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let config = migrating_config(MEMORY_BACKEND, dir.path());
        let m = scripted(&gate, 0);
        let err = StorageBackend::resolve_kind(&config, &m, &gate)
            .await
            .err()
            .expect("a memory target is refused")
            .to_string();
        assert!(
            err.contains("only") && err.contains(OBRAIN_BACKEND),
            "{err}"
        );
        assert!(!m.copied.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!gate.is_read_only());
    }

    #[tokio::test]
    async fn a_parity_gap_keeps_the_source_and_the_gate_is_released() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let config = migrating_config(OBRAIN_BACKEND, dir.path());
        let chosen = StorageBackend::resolve_kind(&config, &scripted(&gate, 1), &gate)
            .await
            .unwrap();
        assert_eq!(chosen.kind, BackendKind::Default, "a gap must never flip");
        drop(chosen);
        assert!(!gate.is_read_only());
    }

    #[tokio::test]
    async fn without_a_migrator_the_factory_stays_on_the_source() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let config = migrating_config(OBRAIN_BACKEND, dir.path());
        let chosen = StorageBackend::resolve_kind(&config, &migrate::NoMigrator, &gate)
            .await
            .unwrap();
        assert_eq!(chosen.kind, BackendKind::Default);
    }

    #[tokio::test]
    async fn rollback_by_configuration_selects_the_source_without_copying() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let mut config = migrating_config(OBRAIN_BACKEND, dir.path());
        config.storage_migration.rollback = true;
        let m = scripted(&gate, 0);
        let chosen = StorageBackend::resolve_kind(&config, &m, &gate)
            .await
            .unwrap();
        assert_eq!(chosen.kind, BackendKind::Default);
        assert!(
            !m.copied.load(std::sync::atomic::Ordering::SeqCst),
            "rollback copies nothing"
        );
    }

    #[tokio::test]
    async fn a_failed_disposable_run_refuses_to_start_instead_of_serving_the_source() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let mut config = migrating_config(OBRAIN_BACKEND, dir.path());
        config.storage_migration.disposable = true;
        for m in [&scripted(&gate, 1) as &dyn Migrator, &migrate::NoMigrator] {
            let err = StorageBackend::resolve_kind(&config, m, &gate)
                .await
                .err()
                .expect("a failed disposable run never starts")
                .to_string();
            assert!(err.contains("disposable environment refused"), "{err}");
        }
        config.storage_migration.rollback = true;
        assert!(
            StorageBackend::resolve_kind(&config, &scripted(&gate, 0), &gate)
                .await
                .is_err()
        );
        assert!(!gate.is_read_only());
    }

    #[tokio::test]
    async fn disposable_mounts_a_migrated_copy_never_the_configured_target() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let mut config = migrating_config(OBRAIN_BACKEND, dir.path());
        config.storage_migration.disposable = true;
        let m = scripted(&gate, 0);
        let chosen = StorageBackend::resolve_kind(&config, &m, &gate)
            .await
            .unwrap();
        assert_eq!(chosen.kind, BackendKind::Obrain);
        let scratch = chosen.config.storage_obrain_path.clone().unwrap();
        assert_eq!(
            Some(std::path::Path::new(&scratch)),
            chosen.scratch.as_ref().map(|d| d.path()),
            "the backend is built on the scratch that was verified"
        );
        assert!(std::path::Path::new(&scratch).join("n.json").exists());
        assert!(
            !dir.path().join("target").exists(),
            "configured target untouched"
        );
        drop(chosen);
        assert!(
            !std::path::Path::new(&scratch).exists(),
            "the scratch is removed with its owner"
        );
    }

    /// The backend the factory BUILDS is the obrain base that was verified,
    /// at the same path — not some other store.
    #[cfg(feature = "obrain-backend")]
    #[tokio::test]
    async fn the_built_backend_is_the_verified_obrain_base() {
        use crate::obrain::{Creation, ObrainGraphStore};

        /// Copies an (empty) source by creating a real PO obrain base in `dest`.
        struct CreatesBase(ReadOnlyGate);
        #[async_trait::async_trait]
        impl Migrator for CreatesBase {
            async fn source_counts(&self) -> Result<PerTypeCounts> {
                assert!(self.0.is_read_only());
                Ok(PerTypeCounts::default())
            }
            async fn copy(&self, dest: &std::path::Path) -> Result<()> {
                let dest = dest.to_path_buf();
                tokio::task::spawn_blocking(move || {
                    ObrainGraphStore::open(dest, Creation::Permise).map(drop)
                })
                .await?
            }
            async fn target_counts(&self, _dest: &std::path::Path) -> Result<PerTypeCounts> {
                Ok(PerTypeCounts::default())
            }
        }

        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let config = migrating_config(OBRAIN_BACKEND, dir.path());
        let target = dir.path().join("target");
        let backend = StorageBackend::from_config_in(&config, &CreatesBase(gate.clone()), &gate)
            .await
            .unwrap();
        assert!(!gate.is_read_only(), "the gate is lowered once built");
        assert!(target.join(DONE_MARKER).is_file());
        // The built store holds obrain's exclusive lock on THAT directory.
        let t = target.clone();
        let reopen =
            tokio::task::spawn_blocking(move || ObrainGraphStore::open(t, Creation::Interdite))
                .await
                .unwrap();
        assert!(
            reopen.is_err(),
            "the verified base is the one the factory opened"
        );
        drop(reopen);
        drop(backend);
        // And a second start serves it, without copying.
        struct NeverCopies;
        #[async_trait::async_trait]
        impl Migrator for NeverCopies {
            async fn source_counts(&self) -> Result<PerTypeCounts> {
                panic!("a migrated target is not read again")
            }
            async fn copy(&self, _dest: &std::path::Path) -> Result<()> {
                panic!("a migrated target is never copied over")
            }
            async fn target_counts(&self, _dest: &std::path::Path) -> Result<PerTypeCounts> {
                panic!("not read again")
            }
        }
        StorageBackend::from_config_in(&config, &NeverCopies, &gate)
            .await
            .unwrap();
    }
}
