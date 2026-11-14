//! The obrain storage backend (`storage.backend = "obrain"`, feature
//! `obrain-backend`).
//!
//! The `GraphStore` traits stay the domain layer; this module implements them
//! on an embedded obrain base, **one domain at a time**. What is ported is
//! written by hand here (one file per domain); everything else is refused by
//! name — see [`NonPorte`] and `build/refus_obrain.rs`, which generates those
//! refusals from the contracts so that none is written, or forgotten, by hand,
//! and which FAILS THE BUILD if a hand-written `impl` leaves out a method of
//! its trait (an omitted method would silently inherit the contract's default).
//!
//! Ported so far: [`ProjectStore`](crate::neo4j::traits::ProjectStore)
//! (`project.rs`), the core of [`PlanStore`](crate::neo4j::traits::PlanStore)
//! (`plan.rs`: no tasks, steps, constraints or day plan yet),
//! [`AuthStore`](crate::neo4j::traits::AuthStore) (`auth.rs`),
//! [`ReleaseStore`](crate::neo4j::traits::ReleaseStore)
//! for releases, environments, deployments and milestones (`release.rs`), whose
//! refusals (tasks, plans, commits) are listed in `release::REFUSEES`,
//! [`ProtocolStore`](crate::neo4j::traits::ProtocolStore) (`protocol.rs`, all 11
//! methods) and [`ProtocolRunStore`](crate::neo4j::traits::ProtocolRunStore)
//! (`protocolrun.rs`, 14 of its 18; the four that name notes, decisions or
//! reasoning trees are refused by name),
//! [`WorkspaceStore`](crate::neo4j::traits::WorkspaceStore)
//! (`workspace.rs`, 35 of its 45 methods; the rest are refused by name), the note
//! node of [`NoteStore`](crate::neo4j::traits::NoteStore) (`note.rs`: 13 of its 41
//! methods; the others are refused by name there), and the node part of
//! [`FederationStore`](crate::neo4j::traits::FederationStore)
//! (`federation.rs`: the five relation writes and the two backfills are refused by name),
//! and the decision node of [`DecisionStore`](crate::neo4j::traits::DecisionStore)
//! (`decision.rs`: the node, its direct project and its supersession are ported;
//! the task-bound creation, the AFFECTS edges and the embeddings are refused by name).
//!
//! # How obrain is used
//!
//! - Through its **Rust API** (`ObrainDB::create_node`, `set_node_property`,
//!   `delete_node`, …), not through a query language: the Cypher front-end is
//!   not covered by obrain's default CI, and one grouping form is under
//!   investigation.
//! - Those direct writes are outside any obrain transaction: each property is
//!   its own WAL record and there is no multi-property atomic write. Atomicity
//!   is therefore built here, by ordering (see `project.rs`: a draft marker
//!   written first and removed last, repaired at open).
//! - **Durability** is the WAL fsync (`ObrainDB::flush_wal`), once per write,
//!   which obrain documents as the source of truth that replay rebuilds the rest
//!   from — PLUS the full flush (`commit_wal`) for a write that carries a value
//!   the WAL does not recover: a string above [`SEUIL_COLONNE_BLOB`] goes to a
//!   blob column whose registry reaches the disk only at a flush. MEASURED: a
//!   project with a 585-byte description, acknowledged and then killed with
//!   SIGKILL, reopened with no description (kill -9 test, first run). The full
//!   flush costs in proportion to the base (msync of every zone, the dictionary,
//!   and on a base in the inherited format a full rewrite of the property
//!   sidecar — see `examples/obrain_mesures.rs` for the numbers), so it is paid
//!   only by those writes, and otherwise every [`FLUSH_COMPLET_TOUS_LES`] writes
//!   and at close, to bound the WAL a restart replays. The fsync is proved by
//!   obrain's own counter (`wal_fsyncs`); what a power cut does to an
//!   acknowledged write is obrain's guarantee, not re-proved here.
//! - obrain addresses a node by an internal `NodeId`; the domain addresses it
//!   by `Uuid`. The index between the two lives in memory and is **rebuilt
//!   from the persisted nodes at every open** (a scan of the node zone: O(nodes),
//!   measured in `examples/obrain_mesures.rs`).
//! - **PO is the single writer** of the base. obrain v0.0.1 does not guarantee
//!   properties written by two processes; it takes an exclusive lock on the
//!   directory at open, and refuses a second writable handle in the same
//!   process. This module adds nothing to that lock and never opens the base
//!   in shared-write mode.
//!
//! # Locks
//!
//! Two levels, so that nobody reads behind an fsync:
//! - `ecrivain` serializes WRITERS from validation to fsync;
//! - `etat` (the indexes, which hold the decoded projects) is write-locked only
//!   to publish a write — AFTER its fsync — never around disk or fsync. A reader
//!   waits, at worst, for a handful of in-memory operations, and is served the
//!   previous state until the write is durable (create, update, delete alike).
//!
//! Reads are served inline (obrain reads are mmap and hash lookups); writes run
//! on the blocking pool.

mod auth;
mod chat;
mod commit;
mod decision;
mod federation;
#[cfg(any(test, feature = "mock-store"))]
mod mesure;
mod note;
mod plan;
mod planrun;
mod project;
mod protocol;
mod protocolrun;
mod release;
mod trigger;
mod workspace;

#[cfg(any(test, feature = "mock-store"))]
pub use mesure::{Durabilite, Interruption, ValeurBrute};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use obrain_engine::ObrainDB;
use uuid::Uuid;

use crate::neo4j::models::{ProgressKind, TaskCounts};
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

/// Every method of a trait implemented by hand, as `<Trait>::<method>`. The
/// build has checked that each one is written (see `build/refus_obrain.rs`).
pub fn methodes_portees() -> &'static [&'static str] {
    PORTEES
}

/// The methods of `NoteStore` refused on every call, as `NoteStore::<method>`
/// (`note.rs`). Three more note methods refuse only in some cases.
pub fn refusees_note() -> &'static [&'static str] {
    note::REFUSEES
}

/// The methods of `CommitStore` refused on every call, as
/// `CommitStore::<method>` (`commit.rs`): the dependency analysis (the code
/// graph), the task links, and the freshness ping over `Note → File` links.
pub fn refusees_commit() -> &'static [&'static str] {
    commit::REFUSEES
}

/// The methods of `ReleaseStore` refused on every call, as
/// `ReleaseStore::<method>` (`release.rs`): they read or link a `Task`, a
/// `Plan`, a `Commit` or a `Step`, domains not yet ported.
pub fn refusees_release() -> &'static [&'static str] {
    release::REFUSEES
}

/// The methods of `ChatStore` refused on every call, as `ChatStore::<method>`
/// (`chat.rs`).
pub fn refusees_chat() -> &'static [&'static str] {
    chat::REFUSEES
}

/// The methods of `DecisionStore` refused on every call, as
/// `DecisionStore::<method>` (`decision.rs`): the task-bound creation, the
/// AFFECTS edges to code entities, and the three embedding methods.
pub fn refusees_decision() -> &'static [&'static str] {
    decision::REFUSEES
}

/// The methods of `PlanStore` refused on every call, as `PlanStore::<method>`
/// (`plan.rs`): tasks, steps, constraints, the day plan, dependencies, the
/// graphs and waves, the enrichment, and the HAS_PLAN links to several
/// projects.
pub fn refusees_plan() -> &'static [&'static str] {
    plan::REFUSEES
}

/// The methods of `ProtocolRunStore` refused by name (the runs' links to notes,
/// decisions and reasoning trees).
pub fn refusees_protocolrun() -> &'static [&'static str] {
    protocolrun::REFUSEES
}

/// The methods of `PlanRunStore` refused on every call, as
/// `PlanRunStore::<method>` (`planrun.rs`): the agent executions (they need a
/// `Task`) and the statistics of the code graph.
pub fn refusees_planrun() -> &'static [&'static str] {
    planrun::REFUSEES
}

/// The hand-written methods that REFUSE: `GraphStore::get_progress_batch` (its
/// contract default loops over getters of other domains), the five
/// `FederationStore` methods whose relation writes and backfills are not ported
/// (`federation.rs`), and the ten `WorkspaceStore` methods whose domains (plan,
/// cross-workspace listing) are not ported. They are in [`methodes_portees`]
/// because they are written, not because they work.
pub const REFUS_ECRITS_A_LA_MAIN: &[&str] = &[
    "GraphStore::get_progress_batch",
    "FederationStore::create_mcp_similar_to",
    "FederationStore::create_mcp_co_activated",
    "FederationStore::create_mcp_often_follows",
    "FederationStore::backfill_co_activated_with",
    "FederationStore::backfill_often_follows",
    // WorkspaceStore: the domains they read (plan, cross-workspace listing) are not ported.
    "WorkspaceStore::list_all_workspace_milestones_filtered",
    "WorkspaceStore::count_all_workspace_milestones",
    "WorkspaceStore::add_task_to_workspace_milestone",
    "WorkspaceStore::remove_task_from_workspace_milestone",
    "WorkspaceStore::link_plan_to_workspace_milestone",
    "WorkspaceStore::unlink_plan_from_workspace_milestone",
    "WorkspaceStore::get_workspace_milestone_progress",
    "WorkspaceStore::get_workspace_milestone_tasks",
    "WorkspaceStore::get_workspace_milestone_steps",
    "WorkspaceStore::compute_coupling_matrix",
];

/// The methods that really serve something: the ported methods of the domains,
/// without the hand-written refusals and without `GraphStore`'s own manifest.
pub fn methodes_qui_servent() -> Vec<&'static str> {
    PORTEES
        .iter()
        .copied()
        .filter(|m| {
            !REFUS_ECRITS_A_LA_MAIN.contains(m)
                && !release::REFUSEES.contains(m)
                && !note::REFUSEES.contains(m)
                && !commit::REFUSEES.contains(m)
                && !chat::REFUSEES.contains(m)
                && !decision::REFUSEES.contains(m)
                && !plan::REFUSEES.contains(m)
                && !protocolrun::REFUSEES.contains(m)
                && !planrun::REFUSEES.contains(m)
                && !m.starts_with("GraphStore::")
        })
        .collect()
}

/// Contract methods that cannot carry a refusal (they do not return a
/// `Result`): the default body of the contract is kept for them.
pub fn defauts_conserves() -> &'static [&'static str] {
    DEFAUTS_CONSERVES
}

/// obrain moves a string longer than this many bytes out of the property zone, into a
/// "blob column" (`BLOB_COLUMN_THRESHOLD_BYTES`, obrain-substrate). Those writes are NOT
/// recovered by the WAL: the column and its registry reach the disk at the next full flush.
/// MEASURED here: a project whose description is longer, acknowledged and then killed with
/// SIGKILL before any flush, reopens with NO description. So a write that carries such a value
/// pays the full flush before it is acknowledged (`tests/obrain_backend.rs`, kill -9 test).
pub const SEUIL_COLONNE_BLOB: usize = 256;

/// A full flush (`commit_wal`) every this many writes, on top of the per-write
/// WAL fsync: it bounds how much WAL a restart has to replay.
pub const FLUSH_COMPLET_TOUS_LES: u64 = 256;

/// What makes a write durable, once per write. In a production build only the WAL
/// fsync exists; the test builds add the measures' and the tests' variants
/// (`mesure.rs`).
#[cfg(not(any(test, feature = "mock-store")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durabilite {
    /// The WAL fsync (`flush_wal`), and a full flush every
    /// [`FLUSH_COMPLET_TOUS_LES`] writes.
    Wal,
}

/// The file that says a base is PO's: written when PO creates the base (or by the
/// migration that produces one), required at open. Any directory with a
/// `substrate.meta` is an obrain base; only one with this file is a PO base.
pub const MARQUEUR_PO: &str = "po-obrain-base";
/// Its content: the schema of what PO writes in the base. Bumped by a lot that
/// changes the shape of a node this backend writes.
pub const SCHEMA_PO: &str = "schema = 1\n";

/// Whether opening may create the base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Creation {
    /// The directory must already hold a base (`substrate.meta`).
    Interdite,
    /// An absent or empty directory becomes a new base.
    Permise,
}

/// What opening found and repaired. Nothing here is fatal; everything here is
/// logged by the factory and readable by the caller.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RapportOuverture {
    /// `Project`, `McpServer` and `McpTool` nodes of a creation that never reached
    /// its commit point (the draft marker still there, or nothing on the node):
    /// deleted, whatever they held (`project.rs`, "Atomicity").
    pub brouillons_supprimes: usize,
    /// Deletions that had started and not finished (a `Workspace` or an
    /// `Environment` carrying the `_suppression` marker, `workspace.rs` and
    /// `release.rs`): finished at open, the node and what it held.
    pub suppressions_achevees: usize,
    /// Projects whose path trio (`root_path`, `declared_path`,
    /// `canonical_path`) was torn by an interrupted update: recomputed from
    /// `root_path`, which is written last.
    pub chemins_repares: usize,
    /// `ProtocolRun` updates cut after their update marker: finished from the
    /// marker (`protocolrun.rs`, "Atomicity").
    pub mises_a_jour_achevees: usize,
    /// `Project` nodes this backend cannot address (no readable `id`/`name`/
    /// `slug`/`created_at`, or an id or slug already taken): left untouched,
    /// ignored, named here.
    pub en_quarantaine: Vec<String>,
}

/// The graph store on obrain.
pub struct ObrainGraphStore {
    interne: Arc<Interne>,
}

struct Interne {
    db: ObrainDB,
    chemin: PathBuf,
    /// Serializes the writers, from validation to the end of the fsync.
    ecrivain: Mutex<()>,
    /// The `Uuid → NodeId` indexes. Never held across disk or fsync.
    etat: RwLock<Etat>,
    ferme: AtomicBool,
    /// The last durability step failed: what is on disk is unknown.
    indetermine: AtomicBool,
    ecritures: AtomicU64,
    durabilite: Durabilite,
    rapport: RapportOuverture,
    /// A test hook: the point where the next write stops (`mesure.rs`).
    #[cfg(any(test, feature = "mock-store"))]
    interruption: Mutex<Option<Interruption>>,
    /// The directory itself, when the base is a throwaway one
    /// ([`ObrainGraphStore::open_temporary`]). Declared LAST: fields drop in
    /// order, so the base is closed before its directory is removed.
    _jetable: Option<tempfile::TempDir>,
}

/// In-memory indexes, rebuilt from the base at open.
#[derive(Default)]
struct Etat {
    projets: project::IndexProjets,
    plans: plan::IndexPlans,
    protocoles: protocol::IndexProtocoles,
    runs: protocolrun::IndexRuns,
    auth: auth::IndexAuth,
    releases: release::IndexReleases,
    espaces: workspace::IndexEspaces,
    notes: note::IndexNotes,
    chat: chat::IndexChat,
    decisions: decision::IndexDecisions,
    federation: federation::IndexFederation,
    commits: commit::IndexCommits,

    surveillance: trigger::IndexSurveillance,
    executions: planrun::IndexExecutions,
}

/// What a write closure is given: the engine, the indexes, and the durability
/// step.
struct Ecriture<'a> {
    db: &'a ObrainDB,
    interne: &'a Interne,
    rendu_durable: bool,
    /// The write put a value in a place the WAL does not recover (see
    /// [`SEUIL_COLONNE_BLOB`]): the WAL fsync is not enough, the full flush runs.
    flush_complet: bool,
    /// How many engine writes of this closure have been made (the test hook
    /// `Interruption::AvantEtape` counts them).
    #[cfg_attr(not(any(test, feature = "mock-store")), allow(dead_code))]
    etapes: usize,
}

impl Ecriture<'_> {
    fn index(&self) -> Result<RwLockReadGuard<'_, Etat>> {
        self.interne
            .etat
            .read()
            .map_err(|_| anyhow!("verrou de l'état obrain empoisonné"))
    }

    /// Write-lock the indexes. Hold the guard for in-memory work only.
    fn index_mut(&self) -> Result<RwLockWriteGuard<'_, Etat>> {
        self.interne
            .etat
            .write()
            .map_err(|_| anyhow!("verrou de l'état obrain empoisonné"))
    }

    /// A test hook, compiled for test builds only: stop here if the test armed
    /// this point. Nothing is durable, nothing is published — what a crash at
    /// this instant leaves is what the engine has written so far.
    #[cfg(any(test, feature = "mock-store"))]
    fn point(&mut self, point: Interruption) -> Result<()> {
        let mut arme = self
            .interne
            .interruption
            .lock()
            .map_err(|_| anyhow!("verrou du point d'interruption empoisonné"))?;
        if *arme == Some(point) {
            *arme = None;
            bail!("interruption injectée par le test : {point:?}");
        }
        Ok(())
    }

    /// A test hook at each `etape()` call: a crash simulation at the k-th call. Where that
    /// is relative to the write depends on the call site: `note.rs` calls it BEFORE each
    /// write (the k-th write not yet made), `workspace.rs` and `release.rs` AFTER each
    /// delete. The variant name keeps the note convention. Nothing in production.
    fn etape(&mut self) -> Result<()> {
        #[cfg(any(test, feature = "mock-store"))]
        {
            self.etapes += 1;
            let n = self.etapes;
            self.point(Interruption::AvantEtape(n))?;
        }
        Ok(())
    }

    /// Test hook: when `quoi` is armed (`Interruption::SuppressionEchouee`,
    /// `Interruption::SuppressionDAreteEchouee`), the next deletion of that kind answers
    /// "not deleted" without calling the engine. The engine does not produce that answer on
    /// a healthy base; this reaches the branches that handle it. Nothing in production.
    #[cfg(any(test, feature = "mock-store"))]
    fn refus_simule(&mut self, quoi: Interruption) -> bool {
        let Ok(mut arme) = self.interne.interruption.lock() else {
            return false;
        };
        if *arme == Some(quoi) {
            *arme = None;
            true
        } else {
            false
        }
    }

    /// `Interruption::SuppressionEchouee` armed: the next node deletion is refused.
    #[cfg(any(test, feature = "mock-store"))]
    fn suppression_de_noeud_refusee(&mut self) -> bool {
        self.refus_simule(Interruption::SuppressionEchouee)
    }

    #[cfg(not(any(test, feature = "mock-store")))]
    fn suppression_de_noeud_refusee(&mut self) -> bool {
        false
    }

    /// `Interruption::SuppressionDAreteEchouee` armed: the next edge deletion is refused.
    #[cfg(any(test, feature = "mock-store"))]
    fn suppression_d_arete_refusee(&mut self) -> bool {
        self.refus_simule(Interruption::SuppressionDAreteEchouee)
    }

    #[cfg(not(any(test, feature = "mock-store")))]
    fn suppression_d_arete_refusee(&mut self) -> bool {
        false
    }

    /// A failure AFTER the engine was written to (decoding what was just written)
    /// leaves the base and the index disagreeing: the store is indeterminate
    /// until a restart rebuilds the index from the base.
    fn apres_ecriture<T>(&self, resultat: Result<T>) -> Result<T> {
        resultat.map_err(|e| {
            self.interne.indetermine.store(true, Ordering::Release);
            e.context(format!(
                "base obrain {} : écriture appliquée mais non relisible ; l'index et la base \
                 divergent. Tout est refusé jusqu'au redémarrage, qui reconstruit l'index",
                self.interne.chemin.display()
            ))
        })
    }

    /// Refuse, writing nothing, when one of these nodes is held by the index and
    /// absent from the engine: the index and the base disagree.
    fn verifier_vivants(&self, noeuds: &[obrain_common::types::NodeId], quoi: &str) -> Result<()> {
        if let Some(absent) = noeuds.iter().find(|n| self.db.get_node(**n).is_none()) {
            bail!(
                "{quoi} : le nœud {absent:?} est indexé mais absent de la base (index et base \
                 divergent) ; rien n'a été supprimé"
            );
        }
        Ok(())
    }

    /// Delete these nodes, in this order, so that an `Err` keeps its meaning:
    /// - every node must be live in the engine BEFORE the first deletion; one that
    ///   the index holds and the engine does not have is refused with NOTHING
    ///   written (the index and the base disagree; deleting the others first would
    ///   leave the index serving them);
    /// - an engine refusal after a first deletion leaves the base written and the
    ///   index stale: the store becomes indeterminate ([`Self::apres_ecriture`]).
    ///
    /// Each deletion is one step of [`Self::etape`] (the test hook).
    fn supprimer_noeuds(
        &mut self,
        noeuds: &[obrain_common::types::NodeId],
        quoi: &str,
    ) -> Result<()> {
        self.verifier_vivants(noeuds, quoi)?;
        for (k, noeud) in noeuds.iter().enumerate() {
            if !self.db.delete_node(*noeud) {
                let e = anyhow!("{quoi} : obrain a refusé la suppression du nœud {noeud:?}");
                return if k == 0 {
                    Err(e)
                } else {
                    self.apres_ecriture(Err(e))
                };
            }
            self.etape()?;
        }
        Ok(())
    }

    /// For a creation that carries a blob-sized value: run the FULL flush now, before
    /// the draft marker is removed. A crash before the flush completes then leaves a
    /// draft (its description lost with the blob, so it reads incomplete and is
    /// deleted at open), never a project that looks whole and is missing a value.
    /// No-op when the write carries no such value.
    fn rendre_durable_avant_le_marqueur(&mut self) -> Result<()> {
        if !self.flush_complet {
            return Ok(());
        }
        self.rendre_durable()?;
        self.flush_complet = false;
        self.rendu_durable = false;
        Ok(())
    }

    /// Make what was written durable. Call it BEFORE publishing the write to
    /// readers when the write is a creation; [`ObrainGraphStore::ecrire`] calls
    /// it itself if the closure did not.
    fn rendre_durable(&mut self) -> Result<()> {
        self.rendu_durable = true;
        let n = self.interne.ecritures.fetch_add(1, Ordering::Relaxed) + 1;
        let resultat = match self.interne.durabilite {
            Durabilite::Wal => {
                let wal = self.db.flush_wal();
                match wal {
                    Ok(()) if self.flush_complet || n.is_multiple_of(FLUSH_COMPLET_TOUS_LES) => {
                        self.db.commit_wal()
                    }
                    autre => autre,
                }
            }
            #[cfg(any(test, feature = "mock-store"))]
            Durabilite::Complet => self.db.commit_wal(),
            #[cfg(any(test, feature = "mock-store"))]
            Durabilite::Aucune => Ok(()),
            #[cfg(any(test, feature = "mock-store"))]
            Durabilite::LenteAPourTest(ms) => {
                std::thread::sleep(std::time::Duration::from_millis(ms));
                self.db.flush_wal()
            }
            #[cfg(any(test, feature = "mock-store"))]
            Durabilite::EchoueAPourTest => Err(obrain_common::utils::error::Error::Internal(
                "échec injecté par le test".into(),
            )),
        };
        resultat.map_err(|e| {
            self.interne.indetermine.store(true, Ordering::Release);
            anyhow!(
                "base obrain {} : la durabilité de la dernière écriture n'est pas assurée ({e}). \
                 L'écriture est appliquée en mémoire ; l'état sur disque est INDÉTERMINÉ. \
                 Toute écriture et toute lecture sont refusées jusqu'au redémarrage, qui rejoue \
                 le journal",
                self.interne.chemin.display()
            )
        })
    }
}

impl ObrainGraphStore {
    /// Open the obrain base at `chemin` and rebuild the indexes from what it
    /// holds. Synchronous: call it from the blocking pool.
    ///
    /// Guards, because opening is "open or create" in obrain and a typo would
    /// otherwise become a new empty base that PO then serves:
    /// - the path must be ABSOLUTE;
    /// - a directory that holds files but no `substrate.meta` is refused (the
    ///   engine would write its own files into it, over what is there);
    /// - an obrain base that is not PO's (no [`MARQUEUR_PO`]) is refused;
    /// - an absent or empty directory is a new base only with
    ///   [`Creation::Permise`]: a base is created on purpose, never by accident.
    ///
    /// Fails if the base is already open for writing — by this process or by
    /// another one. That refusal is obrain's, and it is what makes PO the
    /// single writer.
    pub fn open(chemin: impl AsRef<Path>, creation: Creation) -> Result<Self> {
        let chemin = chemin.as_ref().to_path_buf();
        let creer = Self::garde_fou_d_ouverture(&chemin, creation)?;
        Self::ouvrir(chemin, Durabilite::Wal, creer, None)
    }

    /// A base in a fresh temporary directory, removed when the store is
    /// dropped. For tests and conformance runs: nothing outlives the store.
    pub fn open_temporary() -> Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix("po-obrain-")
            .tempdir()
            .context("création du répertoire temporaire de la base obrain")?;
        Self::ouvrir(dir.path().to_path_buf(), Durabilite::Wal, true, Some(dir))
    }

    /// `Ok(true)` when the directory holds no base and one may be created there.
    fn garde_fou_d_ouverture(chemin: &Path, creation: Creation) -> Result<bool> {
        if !chemin.is_absolute() {
            bail!(
                "chemin de base obrain relatif refusé : « {} » (il dépendrait du répertoire de \
                 lancement). Donner un chemin absolu dans `storage.obrain_path`",
                chemin.display()
            );
        }
        if chemin.join("substrate.meta").is_file() {
            if !chemin.join(MARQUEUR_PO).is_file() {
                bail!(
                    "{} est une base obrain mais pas une base PO : le fichier `{MARQUEUR_PO}` \
                     manque. Une base d'un autre outil ne s'ouvre pas ici ; une base migrée le \
                     porte. Rien n'a été touché",
                    chemin.display()
                );
            }
            return Ok(false);
        }
        // `metadata` follows a link: a link to a directory is that directory.
        let existe = match std::fs::metadata(chemin) {
            Ok(m) if m.is_dir() => true,
            Ok(_) => bail!(
                "« {} » existe et n'est pas un répertoire : pas une base obrain",
                chemin.display()
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => {
                return Err(e).with_context(|| format!("lecture de {}", chemin.display()));
            }
        };
        if existe {
            let mut contenu = std::fs::read_dir(chemin)
                .with_context(|| format!("lecture du répertoire {}", chemin.display()))?;
            if let Some(premier) = contenu.next() {
                let premier = premier?.file_name();
                bail!(
                    "le répertoire {} n'est pas vide (il contient {:?}…) et n'a pas de \
                     `substrate.meta` : ce n'est pas une base obrain, et l'ouvrir écrirait les \
                     fichiers du moteur par-dessus. Rien n'a été touché",
                    chemin.display(),
                    premier
                );
            }
        }
        match creation {
            Creation::Permise => Ok(true),
            Creation::Interdite => bail!(
                "aucune base obrain dans {} ({}) et la création n'est pas demandée : une faute \
                 de frappe dans `storage.obrain_path` donnerait sinon une base vide, servie comme \
                 si de rien n'était. Pour CRÉER une base, poser `storage.obrain_create: true` \
                 (ou STORAGE_OBRAIN_CREATE=true)",
                chemin.display(),
                if existe {
                    "répertoire vide"
                } else {
                    "répertoire absent"
                }
            ),
        }
    }

    fn ouvrir(
        chemin: PathBuf,
        durabilite: Durabilite,
        creer: bool,
        jetable: Option<tempfile::TempDir>,
    ) -> Result<Self> {
        let db = ObrainDB::open(&chemin)
            .map_err(|e| anyhow!("ouverture de la base obrain {} : {e}", chemin.display()))?;
        if creer {
            // The base is PO's from its first instant: the marker is written before
            // anything else, and a failure here is an open failure.
            std::fs::write(chemin.join(MARQUEUR_PO), SCHEMA_PO).with_context(|| {
                format!(
                    "écriture du marqueur {MARQUEUR_PO} dans {}",
                    chemin.display()
                )
            })?;
        }
        let contexte = || {
            format!(
                "reconstruction des index de la base obrain {}",
                chemin.display()
            )
        };
        let (projets, mut rapport) =
            project::IndexProjets::reconstruire(&db).with_context(contexte)?;
        let (plans, brouillons, quarantaine) =
            plan::IndexPlans::reconstruire(&db, &projets).with_context(contexte)?;
        rapport.brouillons_supprimes += brouillons;
        rapport.en_quarantaine.extend(quarantaine);
        let protocoles = protocol::IndexProtocoles::reconstruire(&db, &mut rapport);
        let runs = protocolrun::IndexRuns::reconstruire(&db, &protocoles, &mut rapport);
        let auth = auth::IndexAuth::reconstruire(&db, &mut rapport);
        let (releases, rapport_releases) =
            release::IndexReleases::reconstruire(&db).with_context(contexte)?;
        rapport.brouillons_supprimes += rapport_releases.brouillons_supprimes;
        rapport.suppressions_achevees += rapport_releases.suppressions_achevees;
        rapport
            .en_quarantaine
            .extend(rapport_releases.en_quarantaine);
        let espaces = workspace::IndexEspaces::reconstruire(&db, &projets, &mut rapport)
            .with_context(contexte)?;
        let notes =
            note::IndexNotes::reconstruire(&db, &projets, &mut rapport).with_context(contexte)?;
        let chat = chat::IndexChat::reconstruire(&db, &mut rapport).with_context(contexte)?;
        let decisions = decision::IndexDecisions::reconstruire(&db, &projets, &mut rapport)
            .with_context(contexte)?;
        let (federation, brouillons, quarantaine) = federation::IndexFederation::reconstruire(&db)
            .with_context(|| {
                format!(
                    "reconstruction des index fédération de la base obrain {}",
                    chemin.display()
                )
            })?;
        rapport.brouillons_supprimes += brouillons;
        rapport.en_quarantaine.extend(quarantaine);
        let commits =
            commit::IndexCommits::reconstruire(&db, &mut rapport).with_context(contexte)?;

        let surveillance = trigger::IndexSurveillance::reconstruire(&db, &mut rapport);
        let executions = planrun::IndexExecutions::reconstruire(&db, &plans, &mut rapport)
            .with_context(contexte)?;
        let etat = Etat {
            projets,
            plans,
            protocoles,
            runs,
            auth,
            releases,
            espaces,
            notes,
            chat,
            decisions,
            federation,
            commits,

            surveillance,
            executions,
        };
        let store = Self {
            interne: Arc::new(Interne {
                db,
                chemin,
                ecrivain: Mutex::new(()),
                etat: RwLock::new(etat),
                ferme: AtomicBool::new(false),
                indetermine: AtomicBool::new(false),
                ecritures: AtomicU64::new(0),
                durabilite,
                rapport,
                #[cfg(any(test, feature = "mock-store"))]
                interruption: Mutex::new(None),
                _jetable: jetable,
            }),
        };
        // What opening repaired is made durable before anything else runs.
        if store.interne.rapport != RapportOuverture::default() {
            store
                .interne
                .db
                .flush_wal()
                .map_err(|e| anyhow!("réparations d'ouverture non rendues durables : {e}"))?;
        }
        Ok(store)
    }

    /// The directory of the base.
    pub fn chemin(&self) -> &Path {
        &self.interne.chemin
    }

    /// What opening found and repaired.
    pub fn rapport_d_ouverture(&self) -> &RapportOuverture {
        &self.interne.rapport
    }

    /// How many fsyncs the WAL has really run (obrain's oracle `wal_fsyncs`): between
    /// two acknowledged writes of the single writer it must have grown. (The other
    /// oracle, `wal_octets_durables`, is only published by the commit path of a
    /// session, not by an explicit `fsync_wal`: it reads 0 here whatever happens.)
    #[doc(hidden)]
    pub fn fsyncs_du_wal(&self) -> Option<u64> {
        self.interne.db.wal_fsyncs()
    }

    /// Flush and close the base. Dropping the store does the same but can only
    /// log a failure; call this where the failure must be seen. Once closed,
    /// EVERY operation on this store refuses, and the directory can be opened
    /// again.
    pub async fn close(&self) -> Result<()> {
        let interne = self.interne.clone();
        tokio::task::spawn_blocking(move || {
            // The writers drain first: nothing is half-way when we flush.
            let _ecrivain = interne
                .ecrivain
                .lock()
                .map_err(|_| anyhow!("verrou d'écriture obrain empoisonné"))?;
            if interne.ferme.swap(true, Ordering::AcqRel) {
                return Ok(());
            }
            interne
                .db
                .close()
                .map_err(|e| anyhow!("fermeture de la base obrain : {e}"))
        })
        .await
        .context("tâche bloquante obrain interrompue")?
    }

    fn garde(&self) -> Result<()> {
        if self.interne.ferme.load(Ordering::Acquire) {
            bail!(
                "la base obrain {} est fermée : l'opération est refusée",
                self.interne.chemin.display()
            );
        }
        if self.interne.indetermine.load(Ordering::Acquire) {
            bail!(
                "la base obrain {} est dans un état indéterminé (échec de durabilité) : redémarrer",
                self.interne.chemin.display()
            );
        }
        Ok(())
    }

    /// Run a read, inline, under the read lock of the indexes.
    fn lire<R>(&self, f: impl FnOnce(&ObrainDB, &Etat) -> Result<R>) -> Result<R> {
        self.garde()?;
        let etat = self
            .interne
            .etat
            .read()
            .map_err(|_| anyhow!("verrou de l'état obrain empoisonné"))?;
        f(&self.interne.db, &etat)
    }

    /// Run a mutation on the blocking pool, as the one writer, and make it
    /// durable before answering.
    ///
    /// `preparer` runs first, on the blocking pool but BEFORE the writer lock: the
    /// place for what touches the filesystem (resolving a root) without holding
    /// anyone. The closure must validate BEFORE it writes (obrain's direct API has
    /// no rollback, so an `Err` has to mean "nothing was written"), must publish to
    /// the index only after its durability step, and must hold the write lock of
    /// the indexes for in-memory work only.
    async fn ecrire<P, R, Prep, F>(&self, preparer: Prep, f: F) -> Result<R>
    where
        P: Send + 'static,
        R: Send + 'static,
        Prep: FnOnce() -> P + Send + 'static,
        F: FnOnce(P, &mut Ecriture<'_>) -> Result<R> + Send + 'static,
    {
        let interne = self.interne.clone();
        tokio::task::spawn_blocking(move || {
            let prepare = preparer();
            let _ecrivain = interne
                .ecrivain
                .lock()
                .map_err(|_| anyhow!("verrou d'écriture obrain empoisonné"))?;
            // Checked under the writer lock: `close` takes it too.
            if interne.ferme.load(Ordering::Acquire) {
                bail!(
                    "la base obrain {} est fermée : l'opération est refusée",
                    interne.chemin.display()
                );
            }
            if interne.indetermine.load(Ordering::Acquire) {
                bail!(
                    "la base obrain {} est dans un état indéterminé (échec de durabilité) : \
                     redémarrer",
                    interne.chemin.display()
                );
            }
            let mut ecriture = Ecriture {
                db: &interne.db,
                interne: &interne,
                rendu_durable: false,
                flush_complet: false,
                etapes: 0,
            };
            let rendu = f(prepare, &mut ecriture)?;
            if !ecriture.rendu_durable {
                ecriture.rendre_durable()?;
            }
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
    /// writer lock (`project.rs`), not by an engine constraint, so
    /// `unique_constraints` stays false. A flag turns true in the lot that
    /// proves it.
    fn capabilities(&self) -> StoreCapabilities {
        StoreCapabilities::default()
    }

    /// Written by hand and refusing: the contract's default loops over the task
    /// getters of other domains, so it would refuse anyway — but by the name of
    /// whichever getter it reached first, and only for a non-empty `ids`.
    async fn get_progress_batch(
        &self,
        _kind: ProgressKind,
        _ids: &[Uuid],
    ) -> Result<HashMap<Uuid, TaskCounts>> {
        Err(NonPorte("GraphStore::get_progress_batch").into())
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
