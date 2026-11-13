//! What the measures and the durability tests need from the engine, kept
//! inside the adapter: the storage-boundary gate allows the obrain engine here
//! and nowhere else, so `examples/obrain_mesures.rs` and the tests go through
//! these hooks rather than naming `obrain_engine`.
//!
//! Nothing here is part of the backend's contract, and none of it exists in a
//! production build: the whole module is compiled for test builds only
//! (`cfg(test)` or the `mock-store` feature, which the dev-dependency of the
//! crate on itself turns on for every test and example target and no binary).

use anyhow::{anyhow, Result};
use obrain_common::types::Value;

use super::ObrainGraphStore;

/// Where a write can be interrupted, for the tests of the write ORDER: the
/// write stops right there with an error, as a crash would, and the test then
/// closes and reopens the base to see what opening makes of that exact state.
/// Armed once with [`ObrainGraphStore::interrompre_au_prochain`], consumed by the
/// first write that reaches the point.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interruption {
    /// `create_project`: the node exists, nothing on it.
    ApresLeNoeud,
    /// `create_project`: the draft marker is set, no property yet.
    ApresLeMarqueur,
    /// `create_project`: after the n-th of the eleven properties (1-based).
    ApresLaPropriete(usize),
    /// `create_project`: every property written, the marker still there.
    AvantLeRetraitDuMarqueur,
    /// `create_project`: the marker removed (the commit point), before the fsync
    /// that acknowledges.
    ApresLeRetraitDuMarqueur,
    /// `update_project` with a root: after the n-th of `declared_path`,
    /// `canonical_path`, `root_path` (1-based).
    ApresLeChemin(usize),
    /// `delete_project`: the node deleted, before the fsync.
    ApresLaSuppression,
    /// After the k-th engine write of a multi-write (notes: supersede, rescope, delete,
    /// update, confirm, link). Each such write is a separate record, with no marker.
    ApresEtape(usize),
}

/// What makes a write durable, once per write. Only `Wal` exists in production
/// (see `ObrainGraphStore::open`); the variants here are the measures' and the
/// tests'.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durabilite {
    /// The WAL fsync (`flush_wal`), and a full flush every
    /// [`FLUSH_COMPLET_TOUS_LES`](super::FLUSH_COMPLET_TOUS_LES) writes. The
    /// production step.
    Wal,
    /// A full flush (`commit_wal`) after EVERY write: what the first version of
    /// this adapter did, kept to measure what it cost.
    Complet,
    /// Nothing: to show that the durability test goes red without the fsync.
    Aucune,
    /// The step fails: to exercise the "indeterminate state" path.
    EchoueAPourTest,
    /// The WAL fsync, after sleeping this many milliseconds: a slow disk, to show
    /// that readers do not wait behind it.
    LenteAPourTest(u64),
}

impl ObrainGraphStore {
    /// [`ObrainGraphStore::open`] with an explicit durability step.
    #[doc(hidden)]
    pub fn open_avec(
        chemin: impl AsRef<std::path::Path>,
        creation: super::Creation,
        durabilite: Durabilite,
    ) -> Result<Self> {
        let chemin = chemin.as_ref().to_path_buf();
        let creer = Self::garde_fou_d_ouverture(&chemin, creation)?;
        Self::ouvrir(chemin, durabilite, creer, None)
    }

    /// A throwaway base with an explicit durability step.
    #[doc(hidden)]
    pub fn open_temporary_avec(durabilite: Durabilite) -> Result<Self> {
        let dir = tempfile::Builder::new()
            .prefix("po-obrain-")
            .tempdir()
            .map_err(|e| anyhow!("création du répertoire temporaire de la base obrain : {e}"))?;
        Self::ouvrir(dir.path().to_path_buf(), durabilite, true, Some(dir))
    }

    /// How many full flushes (`commit_wal`) the write path has run since open:
    /// periodic, by bytes, or right after a value obrain's WAL does not recover.
    #[doc(hidden)]
    pub fn flushs_complets(&self) -> u64 {
        self.interne
            .flushs_complets
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Arm an [`Interruption`]: the next write that reaches that point stops there.
    #[doc(hidden)]
    pub fn interrompre_au_prochain(&self, point: Interruption) {
        *self.interne.interruption.lock().unwrap() = Some(point);
    }

    /// The raw properties of a project's node as the ENGINE reads them now, for
    /// a test that must tell "not repaired" from "not written".
    #[doc(hidden)]
    pub fn proprietes_brutes(&self, id: uuid::Uuid) -> Option<Vec<(String, String)>> {
        let noeud = self.interne.etat.read().ok()?.projets.noeud_de(id)?;
        let n = self.interne.db.get_node(noeud)?;
        let mut props: Vec<(String, String)> = n
            .properties
            .iter()
            .map(|(k, v)| (k.to_string(), format!("{v:?}")))
            .collect();
        props.sort();
        Some(props)
    }

    /// Fill the base with `noeuds` `Filler` nodes (two properties each) and
    /// `aretes` `FILLS` edges between them, the way a migrated base looks to the
    /// engine, then flush. Returns the time it took.
    #[doc(hidden)]
    pub fn remplir_pour_mesure(&self, noeuds: u64, aretes: u64) -> Result<std::time::Duration> {
        let t0 = std::time::Instant::now();
        let db = &self.interne.db;
        let mut ids = Vec::with_capacity(noeuds as usize);
        for i in 0..noeuds {
            ids.push(db.create_node_with_props(
                &["Filler"],
                vec![
                    ("rank", Value::from(i as i64)),
                    ("tag", Value::from(format!("filler-{i}"))),
                ],
            ));
        }
        if !ids.is_empty() {
            let n = ids.len() as u64;
            for k in 0..aretes {
                // A deterministic spread: no hub, no chain.
                let a = ids[((k * 2_654_435_761) % n) as usize];
                let b = ids[((k * 40_503 + 7) % n) as usize];
                db.create_edge(a, b, "FILLS");
            }
        }
        db.commit_wal().map_err(|e| anyhow!("flush : {e}"))?;
        Ok(t0.elapsed())
    }

    /// A full flush now: what `Durabilite::Complet` does after each write.
    #[doc(hidden)]
    pub fn flush_complet_pour_mesure(&self) -> Result<()> {
        self.interne
            .db
            .commit_wal()
            .map_err(|e| anyhow!("flush : {e}"))
    }

    /// Live nodes of the base, all labels.
    #[doc(hidden)]
    pub fn noeuds_pour_mesure(&self) -> usize {
        self.interne.db.store().node_count()
    }
}
