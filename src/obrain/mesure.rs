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
    /// A creation under the draft marker (`create_project`, `create_plan`, the
    /// release domain's and the federation's creations): the node exists, nothing
    /// on it.
    ApresLeNoeud,
    /// A creation: the draft marker is set, no property yet.
    ApresLeMarqueur,
    /// A creation: after the n-th of its properties (1-based).
    ApresLaPropriete(usize),
    /// A creation: every property written (and a blob-carrying one flushed),
    /// the marker still there.
    AvantLeRetraitDuMarqueur,
    /// A creation: the marker removed (the commit point), before the fsync
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

/// A property value for the raw-write hooks below, so that a test never names
/// the engine's own `Value`.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValeurBrute<'a> {
    Texte(&'a str),
    Entier(i64),
    Booleen(bool),
}

impl From<ValeurBrute<'_>> for Value {
    fn from(v: ValeurBrute<'_>) -> Self {
        match v {
            ValeurBrute::Texte(s) => Value::from(s),
            ValeurBrute::Entier(i) => Value::from(i),
            ValeurBrute::Booleen(b) => Value::from(b),
        }
    }
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

    /// The engine node of label `etiquette` whose `id` property is `id`, found by a
    /// scan of the label (a test helper: no index is read).
    fn noeud_brut(&self, etiquette: &str, id: uuid::Uuid) -> Option<obrain_common::types::NodeId> {
        let db = &self.interne.db;
        let id = id.to_string();
        db.store().nodes_by_label(etiquette).into_iter().find(|n| {
            db.get_node(*n).is_some_and(|node| {
                node.get_property("id")
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| s == id)
            })
        })
    }

    /// Write a node of `etiquette` with these string properties straight into the
    /// ENGINE, behind the index (a node of a migrated base, or a corrupted one),
    /// and fsync the WAL. The index sees it at the next open.
    #[doc(hidden)]
    pub fn creer_noeud_brut_pour_test(
        &self,
        etiquette: &str,
        proprietes: &[(&str, ValeurBrute<'_>)],
    ) {
        let db = &self.interne.db;
        let n = db.create_node(&[etiquette]);
        for (cle, valeur) in proprietes {
            db.set_node_property(n, cle, Value::from(*valeur));
        }
        db.flush_wal().expect("fsync du WAL");
    }

    /// Set (`Some`) or remove (`None`) one property of the node of `etiquette`
    /// whose `id` is `id`, straight into the ENGINE, behind the index, and fsync
    /// the WAL. `false` when no such node exists.
    #[doc(hidden)]
    pub fn ecrire_propriete_brute_pour_test(
        &self,
        etiquette: &str,
        id: uuid::Uuid,
        cle: &str,
        valeur: Option<ValeurBrute<'_>>,
    ) -> bool {
        let Some(n) = self.noeud_brut(etiquette, id) else {
            return false;
        };
        let db = &self.interne.db;
        match valeur {
            Some(v) => db.set_node_property(n, cle, Value::from(v)),
            None => {
                db.remove_node_property(n, cle);
            }
        }
        db.flush_wal().expect("fsync du WAL");
        true
    }

    /// Delete the node of `etiquette` whose `id` is `id` straight from the ENGINE,
    /// behind the index (the index still holds it), and fsync the WAL.
    #[doc(hidden)]
    pub fn supprimer_noeud_brut_pour_test(&self, etiquette: &str, id: uuid::Uuid) -> bool {
        let Some(n) = self.noeud_brut(etiquette, id) else {
            return false;
        };
        let ok = self.interne.db.delete_node(n);
        self.interne.db.flush_wal().expect("fsync du WAL");
        ok
    }

    /// An edge `type_` from the node (`de`, `id_de`) to the node (`vers`, `id_vers`),
    /// written straight into the ENGINE (a migrated base), and fsync the WAL.
    #[doc(hidden)]
    pub fn creer_arete_brute_pour_test(
        &self,
        de: (&str, uuid::Uuid),
        vers: (&str, uuid::Uuid),
        type_: &str,
    ) -> bool {
        let (Some(a), Some(b)) = (self.noeud_brut(de.0, de.1), self.noeud_brut(vers.0, vers.1))
        else {
            return false;
        };
        let _ = self.interne.db.create_edge(a, b, type_);
        self.interne.db.flush_wal().expect("fsync du WAL");
        true
    }

    /// How many live nodes carry `etiquette`, as the engine sees them.
    #[doc(hidden)]
    pub fn noeuds_de_label_pour_test(&self, etiquette: &str) -> usize {
        self.interne.db.store().nodes_by_label(etiquette).len()
    }

    /// Live nodes of the base, all labels.
    #[doc(hidden)]
    pub fn noeuds_pour_mesure(&self) -> usize {
        self.interne.db.store().node_count()
    }
}
