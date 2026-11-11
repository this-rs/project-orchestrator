//! The obrain storage backend (`storage.backend = "obrain"`, feature
//! `obrain-backend`).
//!
//! The `GraphStore` traits stay the domain layer; this module implements them
//! on an embedded obrain base, **one domain at a time**. What is ported is
//! written by hand here (one file per domain); everything else is refused by
//! name — see [`NonPorte`] and `build/refus_obrain.rs`, which generates those
//! refusals from the contracts so that none is written, or forgotten, by hand.
//!
//! Ported so far: [`ProjectStore`](crate::neo4j::traits::ProjectStore)
//! (`project.rs`).
//!
//! # How obrain is used
//!
//! - Through its **Rust API** (`ObrainDB::create_node_with_props`,
//!   `set_node_property`, `delete_node`, …), not through a query language: the
//!   Cypher front-end is not covered by obrain's default CI, and one grouping
//!   form is under investigation.
//! - Those direct writes are outside any obrain transaction and are made
//!   durable by `commit_wal()` (obrain `docs/LIMITES-v0.0.1.md`). So every
//!   mutation here runs under ONE write lock, validates before it writes, and
//!   ends with `commit_wal()`, the durability point obrain documents. What the
//!   tests prove is narrower: a write survives a reopen without a clean close
//!   (the process-crash model). That test passes with `commit_wal` removed
//!   too (measured), so the fsync itself is NOT covered by a test here.
//! - obrain addresses a node by an internal `NodeId`; the domain addresses it
//!   by `Uuid`. The index between the two lives in memory and is **rebuilt
//!   from the persisted nodes at every open** (`Etat::reconstruire`).
//! - **PO is the single writer** of the base. obrain v0.0.1 does not guarantee
//!   properties written by two processes; it takes an exclusive lock on the
//!   directory at open, and refuses a second writable handle in the same
//!   process. This module adds nothing to that lock and never opens the base
//!   in shared-write mode.
//! - obrain calls are synchronous (mmap, fsync): they run on the blocking
//!   pool, never on a runtime worker.

mod project;

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use obrain_engine::ObrainDB;

use crate::neo4j::traits::{GraphStore, StoreCapabilities};

// The refusals: one generated `impl` per trait no file of this module
// implements, and the lists of what is refused / ported.
include!(concat!(env!("OUT_DIR"), "/obrain_refus.rs"));

/// A contract method the obrain backend does not implement yet.
///
/// Carried as a typed error (downcast it from the `anyhow::Error`) so a caller
/// can tell "not ported" from a real storage failure. The payload is
/// `<Trait>::<method>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("non porté sur le backend obrain : {0}")]
pub struct NonPorte(pub &'static str);

/// Every contract method this build refuses, as `<Trait>::<method>`.
pub fn methodes_refusees() -> &'static [&'static str] {
    REFUSEES
}

/// Every contract method of a trait implemented by hand, as `<Trait>::<method>`.
pub fn methodes_portees() -> &'static [&'static str] {
    PORTEES
}

/// Contract methods that cannot carry a refusal (they do not return a
/// `Result`): the default body of the contract is kept for them.
pub fn defauts_conserves() -> &'static [&'static str] {
    DEFAUTS_CONSERVES
}

/// The graph store on obrain.
pub struct ObrainGraphStore {
    interne: Arc<Interne>,
}

struct Interne {
    db: ObrainDB,
    chemin: PathBuf,
    /// The `Uuid → NodeId` indexes. The lock is also THE write lock: a
    /// mutation holds it from its first check to its `commit_wal`, so readers
    /// never see half an update and two commits never overlap.
    etat: RwLock<Etat>,
    /// The directory itself, when the base is a throwaway one
    /// ([`ObrainGraphStore::open_temporary`]). Declared LAST: fields drop in
    /// order, so the base is closed before its directory is removed.
    _jetable: Option<tempfile::TempDir>,
}

/// In-memory indexes, rebuilt from the base at open.
#[derive(Default)]
struct Etat {
    projets: project::IndexProjets,
}

impl Etat {
    fn reconstruire(db: &ObrainDB) -> Result<Self> {
        Ok(Self {
            projets: project::IndexProjets::reconstruire(db)?,
        })
    }
}

impl ObrainGraphStore {
    /// Open (or create) the obrain base at `chemin` and rebuild the indexes
    /// from what it holds. Synchronous: call it from the blocking pool.
    ///
    /// Fails if the base is already open for writing — by this process or by
    /// another one. That refusal is obrain's, and it is what makes PO the
    /// single writer.
    pub fn open(chemin: impl AsRef<Path>) -> Result<Self> {
        Self::ouvrir(chemin.as_ref().to_path_buf(), None)
    }

    /// A base in a fresh temporary directory, removed when the store is
    /// dropped. For tests and conformance runs: nothing outlives the store.
    pub fn open_temporary() -> Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix("po-obrain-")
            .tempdir()
            .context("création du répertoire temporaire de la base obrain")?;
        Self::ouvrir(dir.path().to_path_buf(), Some(dir))
    }

    fn ouvrir(chemin: PathBuf, jetable: Option<tempfile::TempDir>) -> Result<Self> {
        let db = ObrainDB::open(&chemin)
            .map_err(|e| anyhow!("ouverture de la base obrain {} : {e}", chemin.display()))?;
        let etat = Etat::reconstruire(&db).with_context(|| {
            format!(
                "reconstruction des index de la base obrain {}",
                chemin.display()
            )
        })?;
        Ok(Self {
            interne: Arc::new(Interne {
                db,
                chemin,
                etat: RwLock::new(etat),
                _jetable: jetable,
            }),
        })
    }

    /// The directory of the base.
    pub fn chemin(&self) -> &Path {
        &self.interne.chemin
    }

    /// Flush and close the base. Dropping the store does the same but can only
    /// log a failure; call this where the failure must be seen. Once closed,
    /// the directory can be opened again.
    pub async fn close(&self) -> Result<()> {
        let interne = self.interne.clone();
        tokio::task::spawn_blocking(move || {
            // Under the write lock: no mutation is half-way when we flush.
            let _etat = interne
                .etat
                .write()
                .map_err(|_| anyhow!("verrou de l'état obrain empoisonné"))?;
            interne
                .db
                .close()
                .map_err(|e| anyhow!("fermeture de la base obrain : {e}"))
        })
        .await
        .context("tâche bloquante obrain interrompue")?
    }

    /// Run a read on the blocking pool, under the read lock.
    async fn lire<R, F>(&self, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&ObrainDB, &Etat) -> Result<R> + Send + 'static,
    {
        let interne = self.interne.clone();
        tokio::task::spawn_blocking(move || {
            let etat = interne
                .etat
                .read()
                .map_err(|_| anyhow!("verrou de l'état obrain empoisonné"))?;
            f(&interne.db, &etat)
        })
        .await
        .context("tâche bloquante obrain interrompue")?
    }

    /// Run a mutation on the blocking pool, under the write lock, and make it
    /// durable before answering.
    ///
    /// `f` must validate BEFORE it writes: obrain's direct API has no
    /// rollback, so an `Err` has to mean "nothing was written".
    async fn ecrire<R, F>(&self, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&ObrainDB, &mut Etat) -> Result<R> + Send + 'static,
    {
        let interne = self.interne.clone();
        tokio::task::spawn_blocking(move || {
            let mut etat = interne
                .etat
                .write()
                .map_err(|_| anyhow!("verrou de l'état obrain empoisonné"))?;
            let rendu = f(&interne.db, &mut etat)?;
            interne.db.commit_wal().map_err(|e| {
                anyhow!("écriture obrain appliquée en mémoire mais non rendue durable : {e}")
            })?;
            Ok(rendu)
        })
        .await
        .context("tâche bloquante obrain interrompue")?
    }
}

#[async_trait]
impl GraphStore for ObrainGraphStore {
    /// Nothing is claimed. Every flag describes what the ENGINE does for the
    /// queries of the contract, and no query of the contract runs on obrain
    /// yet: the one ported domain goes through the Rust API. The uniqueness of
    /// `Project.id` / `Project.slug` is enforced by this adapter under its
    /// write lock (`project.rs`), not by an engine constraint, so
    /// `unique_constraints` stays false. A flag turns true in the lot that
    /// proves it.
    fn capabilities(&self) -> StoreCapabilities {
        StoreCapabilities::default()
    }
}

/// The search store of the obrain backend: nothing is ported, every method of
/// `SearchStore` is refused by name (generated).
#[derive(Debug, Default, Clone, Copy)]
pub struct ObrainSearchStore;

/// The trajectory store of the obrain backend: nothing is ported.
///
/// Written by hand (the trait lives in another crate and has 11 methods). The
/// four methods the contract defaults to a silent `Ok` are refused too.
#[derive(Debug, Default, Clone, Copy)]
pub struct ObrainTrajectoryStore;

fn trajectoire_non_portee<T>(
    methode: &'static str,
) -> neural_routing_runtime::routing_error::Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).into())
}

#[async_trait]
impl neural_routing_runtime::TrajectoryStore for ObrainTrajectoryStore {
    async fn store_trajectory(
        &self,
        _trajectory: &neural_routing_runtime::Trajectory,
    ) -> neural_routing_runtime::routing_error::Result<()> {
        trajectoire_non_portee("TrajectoryStore::store_trajectory")
    }

    async fn get_trajectory(
        &self,
        _id: &uuid::Uuid,
    ) -> neural_routing_runtime::routing_error::Result<Option<neural_routing_runtime::Trajectory>>
    {
        trajectoire_non_portee("TrajectoryStore::get_trajectory")
    }

    async fn list_trajectories(
        &self,
        _filter: &neural_routing_runtime::TrajectoryFilter,
    ) -> neural_routing_runtime::routing_error::Result<Vec<neural_routing_runtime::Trajectory>>
    {
        trajectoire_non_portee("TrajectoryStore::list_trajectories")
    }

    async fn search_similar(
        &self,
        _query_embedding: &[f32],
        _top_k: usize,
        _min_similarity: f32,
    ) -> neural_routing_runtime::routing_error::Result<Vec<(neural_routing_runtime::Trajectory, f64)>>
    {
        trajectoire_non_portee("TrajectoryStore::search_similar")
    }

    async fn get_stats(
        &self,
    ) -> neural_routing_runtime::routing_error::Result<neural_routing_runtime::TrajectoryStats>
    {
        trajectoire_non_portee("TrajectoryStore::get_stats")
    }

    async fn count(&self) -> neural_routing_runtime::routing_error::Result<usize> {
        trajectoire_non_portee("TrajectoryStore::count")
    }

    async fn delete_trajectory(
        &self,
        _id: &uuid::Uuid,
    ) -> neural_routing_runtime::routing_error::Result<bool> {
        trajectoire_non_portee("TrajectoryStore::delete_trajectory")
    }

    async fn store_trajectories_batch(
        &self,
        _trajectories: &[neural_routing_runtime::Trajectory],
    ) -> neural_routing_runtime::routing_error::Result<usize> {
        trajectoire_non_portee("TrajectoryStore::store_trajectories_batch")
    }

    async fn link_trajectory_to_run(
        &self,
        _trajectory_id: &uuid::Uuid,
        _run_id: &uuid::Uuid,
    ) -> neural_routing_runtime::routing_error::Result<()> {
        trajectoire_non_portee("TrajectoryStore::link_trajectory_to_run")
    }

    async fn link_tool_usages_batch(
        &self,
        _node_id: &uuid::Uuid,
        _usages: &[neural_routing_runtime::ToolUsage],
    ) -> neural_routing_runtime::routing_error::Result<()> {
        trajectoire_non_portee("TrajectoryStore::link_tool_usages_batch")
    }

    async fn link_touched_entities_batch(
        &self,
        _node_id: &uuid::Uuid,
        _entities: &[neural_routing_runtime::TouchedEntity],
    ) -> neural_routing_runtime::routing_error::Result<()> {
        trajectoire_non_portee("TrajectoryStore::link_touched_entities_batch")
    }
}
