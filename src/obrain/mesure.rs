//! What the measures and the durability tests need from the engine, kept
//! inside the adapter: the storage-boundary gate allows the obrain engine here
//! and nowhere else, so `examples/obrain_mesures.rs` and the tests go through
//! these hooks rather than naming `obrain_engine`.
//!
//! Nothing here is part of the backend's contract (`doc(hidden)` throughout).

use anyhow::{anyhow, Result};
use obrain_common::types::Value;

use super::ObrainGraphStore;

/// What makes a write durable, once per write.
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
