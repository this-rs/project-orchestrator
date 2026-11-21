//! `AnalyticsStore` on obrain, sub-lot A2 (`src/obrain/synapse.rs`): the `SYNAPSE` edges of
//! notes and decisions, the energy and the scars of a note, the deep-maintenance clock.
//!
//! The behaviour shared with the reference stores (mock, Neo4j) is in
//! `tests/store_conformance.rs` (`neural_checks!`). What is here is what only this backend
//! does, and what the mock cannot be the reference for (it models a synapse as a bare
//! weight): the expected values below are the ones the Cypher of `src/neo4j/note.rs`
//! computes, derived by hand and named in each test. Everything the engine holds that the
//! API does not show (the properties of an edge, a stored date) is read from the closed
//! base.
#![cfg(feature = "obrain-backend")]

use std::collections::HashMap;
use std::path::Path;

use chrono::{Duration, TimeZone, Utc};
use obrain_common::types::{Timestamp, Value, ZonedDatetime};
use obrain_core::graph::Direction;
use project_orchestrator::graph::rules::AccountableProvenance;
use project_orchestrator::neo4j::models::*;
use project_orchestrator::neo4j::traits::*;
use project_orchestrator::notes::{EntityType, Note, NoteType};
use project_orchestrator::obrain::{
    methodes_qui_servent, refusees_analytics, Creation, Interruption, NonPorte, ObrainGraphStore,
    MARQUEUR_PO, SCHEMA_PO,
};
use uuid::Uuid;

async fn open(path: &Path) -> ObrainGraphStore {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, Creation::Permise))
        .await
        .unwrap()
        .unwrap()
}

/// Read or write the closed base directly, the way the migrator does.
async fn base_brute<R: Send + 'static>(
    base: &Path,
    f: impl FnOnce(&obrain_engine::ObrainDB) -> R + Send + 'static,
) -> R {
    let base = base.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let db = obrain_engine::ObrainDB::open(&base).unwrap();
        let r = f(&db);
        db.commit_wal().unwrap();
        db.close().unwrap();
        std::fs::write(base.join(MARQUEUR_PO), SCHEMA_PO).unwrap();
        r
    })
    .await
    .unwrap()
}

type Brute = (String, String, HashMap<String, Value>);

/// Every edge of `type_` leaving a note or a decision: `(from id, to id, properties)`.
fn aretes_brutes(db: &obrain_engine::ObrainDB, type_: &str) -> Vec<Brute> {
    let id_de = |n| {
        db.get_node(n)
            .and_then(|x| {
                x.get_property("id")
                    .and_then(|v| v.as_str().map(str::to_string))
            })
            .unwrap_or_default()
    };
    let mut v = Vec::new();
    for etiquette in ["Note", "Decision"] {
        for n in db.store().nodes_by_label(etiquette) {
            for (dst, e) in db.store().edges_from(n, Direction::Outgoing) {
                let Some(e) = db.get_edge(e) else { continue };
                if &*e.edge_type == type_ {
                    let p = e
                        .properties
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.clone()))
                        .collect();
                    v.push((id_de(n), id_de(dst), p));
                }
            }
        }
    }
    v.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    v
}

fn poids(b: &Brute) -> f64 {
    b.2["weight"].as_float64().unwrap()
}

fn a_project() -> ProjectNode {
    let id = Uuid::new_v4();
    ProjectNode {
        id,
        name: format!("neural {id}"),
        slug: format!("neural-{id}"),
        root_path: String::new(),
        description: None,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        watch_enabled: false,
        profile: Default::default(),
    }
}

/// An ACTIVE project: its code was synced now, so its energy is not frozen.
async fn projet_actif(store: &ObrainGraphStore) -> ProjectNode {
    let p = a_project();
    store.create_project(&p).await.unwrap();
    store.update_project_synced(p.id).await.unwrap();
    p
}

async fn une_note(store: &ObrainGraphStore, projet: Option<Uuid>) -> Note {
    let n = Note::new(projet, NoteType::Observation, "c".into(), "t".into());
    store.create_note(&n).await.unwrap();
    n
}

async fn des_notes(store: &ObrainGraphStore, projet: Option<Uuid>, n: usize) -> Vec<Note> {
    let mut v = Vec::new();
    for _ in 0..n {
        v.push(une_note(store, projet).await);
    }
    v
}

fn zoned(d: chrono::DateTime<Utc>) -> Value {
    Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
        Timestamp::from_micros(d.timestamp_micros()),
        0,
    ))
}

fn proche(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

// ---------------------------------------------------------------------------
// Synapses
// ---------------------------------------------------------------------------

/// `create_synapses`: two edges per matched neighbour, a neighbour that does not exist is
/// skipped (`MATCH (target:Note {id: nid})`), a repeat updates the weight and never doubles
/// the edge, the reads see both directions, and `delete_synapses` answers the edges it took.
#[tokio::test]
async fn synapses_are_created_read_updated_and_deleted() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let p = projet_actif(&store).await;
    let n = des_notes(&store, Some(p.id), 3).await;
    let (a, b, c) = (n[0].id, n[1].id, n[2].id);
    let ghost = Uuid::new_v4();

    assert_eq!(store.create_synapses(a, &[]).await.unwrap(), 0);
    assert_eq!(
        store
            .create_synapses(a, &[(b, 0.8), (c, 0.6), (ghost, 0.9)])
            .await
            .unwrap(),
        4,
        "two edges per neighbour that exists"
    );
    assert_eq!(
        store.get_synapses(a).await.unwrap(),
        vec![(b, 0.8), (c, 0.6)]
    );
    assert_eq!(store.get_synapses(b).await.unwrap(), vec![(a, 0.8)]);
    // ORDER BY weight DESC.
    assert_eq!(store.get_synapses(c).await.unwrap(), vec![(a, 0.6)]);

    // The same pair again: the weight moves, no edge is added.
    assert_eq!(store.create_synapses(a, &[(b, 0.3)]).await.unwrap(), 2);
    assert_eq!(
        store.get_synapses(a).await.unwrap(),
        vec![(c, 0.6), (b, 0.3)]
    );
    assert_eq!(store.get_synapses(b).await.unwrap(), vec![(a, 0.3)]);

    // A source that does not exist matches nothing.
    assert_eq!(store.create_synapses(ghost, &[(a, 0.5)]).await.unwrap(), 0);
    assert!(store.get_synapses(ghost).await.unwrap().is_empty());
    // Non-finite: refused before any write, with the contract's message.
    let e = store
        .create_synapses(a, &[(b, f64::NAN)])
        .await
        .unwrap_err();
    assert!(
        format!("{e:#}").contains("non-finite synapse weight"),
        "{e:#}"
    );
    assert_eq!(store.get_synapses(a).await.unwrap().len(), 2);

    // Four edges touch `a`; the ones of `b` and `c` toward each other do not exist.
    assert_eq!(store.delete_synapses(a).await.unwrap(), 4);
    assert!(store.get_synapses(a).await.unwrap().is_empty());
    assert!(store.get_synapses(b).await.unwrap().is_empty());
    assert_eq!(store.delete_synapses(a).await.unwrap(), 0);
    assert_eq!(store.delete_synapses(ghost).await.unwrap(), 0);
}

/// `DISTINCT neighbor.id, s.weight`: a neighbour linked in the two directions with two
/// different weights (a migrated base can have that) answers once per weight.
#[tokio::test]
async fn a_neighbour_with_two_weights_answers_twice_and_a_weightless_edge_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, b);
    {
        let store = open(&base).await;
        let n = des_notes(&store, None, 2).await;
        (a, b) = (n[0].id, n[1].id);
        store.create_synapses(a, &[(b, 0.5)]).await.unwrap();
        store.close().await.unwrap();
    }
    // Rewrite one direction to another weight.
    base_brute(&base, move |db| {
        let na = db
            .store()
            .nodes_by_label("Note")
            .into_iter()
            .find(|n| {
                db.get_node(*n)
                    .unwrap()
                    .get_property("id")
                    .and_then(|v| v.as_str().map(str::to_string))
                    == Some(a.to_string())
            })
            .unwrap();
        let (_, e) = db.store().edges_from(na, Direction::Outgoing)[0];
        db.set_edge_property(e, "weight", Value::from(0.9_f64));
    })
    .await;
    let store = open(&base).await;
    assert_eq!(
        store.get_synapses(a).await.unwrap(),
        vec![(b, 0.9), (b, 0.5)]
    );
    store.close().await.unwrap();
    // An edge with no weight is not a row Neo4j can read (`row.get("weight")?` fails).
    base_brute(&base, move |db| {
        let na = db.store().nodes_by_label("Note")[0];
        let (_, e) = db.store().edges_from(na, Direction::Outgoing)[0];
        db.remove_edge_property(e, "weight");
    })
    .await;
    let store = open(&base).await;
    let e = store.get_synapses(a).await.unwrap_err();
    assert!(format!("{e:#}").contains("weight"), "{e:#}");
}

/// `reinforce_synapses`: only pairs of two notes that share a non-empty project and are
/// distinct; two edges per pair counted; a created edge starts at 0.5 as `coactivation`
/// with `reinforcement_count` 1, a matched one gains `boost` (capped at 1.0), its count and
/// its `last_reinforced_at`/`decayed_at`, and a `cosine` edge becomes `coactivation`.
#[tokio::test]
async fn reinforce_creates_then_strengthens_pairs_of_one_project_only() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let a = des_notes(&store, Some(p.id), 4).await;
    let (x, y) = (
        une_note(&store, Some(q.id)).await,
        une_note(&store, None).await,
    );
    let ids = [a[0].id, a[1].id, a[2].id, x.id, y.id, Uuid::new_v4()];

    let e = store.reinforce_synapses(&ids[..1], 0.1).await.unwrap_err();
    assert!(
        format!("{e:#}").contains("needs at least 2 note ids (got 1)"),
        "{e:#}"
    );
    // 3 pairs among the three notes of `p`; every pair with `x`, `y` or the ghost is skipped.
    assert_eq!(store.reinforce_synapses(&ids, 0.2).await.unwrap(), 6);
    assert!(store.get_synapses(x.id).await.unwrap().is_empty());
    assert!(store.get_synapses(y.id).await.unwrap().is_empty());
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let s = store.get_synapses(a[i].id).await.unwrap();
        assert!(s.contains(&(a[j].id, 0.5)), "{s:?}");
    }
    // A second pass: 0.5 + 0.2, and 0.7 + 0.5 is capped at 1.0.
    assert_eq!(store.reinforce_synapses(&ids[..2], 0.2).await.unwrap(), 2);
    assert!(proche(
        store
            .get_synapses(a[1].id)
            .await
            .unwrap()
            .iter()
            .find(|s| s.0 == a[0].id)
            .unwrap()
            .1,
        0.7
    ));
    assert_eq!(store.reinforce_synapses(&ids[..2], 0.5).await.unwrap(), 2);
    assert!(proche(
        store
            .get_synapses(a[0].id)
            .await
            .unwrap()
            .iter()
            .find(|s| s.0 == a[1].id)
            .unwrap()
            .1,
        1.0
    ));
    // A `cosine` edge (create_synapses) turns `coactivation` when reinforced.
    store
        .create_synapses(a[0].id, &[(a[3].id, 0.4)])
        .await
        .unwrap();
    store
        .reinforce_synapses(&[a[0].id, a[3].id], 0.1)
        .await
        .unwrap();
    store.close().await.unwrap();

    let brutes = base_brute(&base, |db| aretes_brutes(db, "SYNAPSE")).await;
    assert_eq!(brutes.len(), 8, "four pairs, two directions");
    let de_a0_a1 = brutes
        .iter()
        .find(|b| b.0 == a[0].id.to_string() && b.1 == a[1].id.to_string())
        .unwrap();
    assert_eq!(de_a0_a1.2["reinforcement_count"].as_int64(), Some(3));
    assert_eq!(de_a0_a1.2["source"].as_str(), Some("coactivation"));
    for k in ["created_at", "last_reinforced_at", "decayed_at"] {
        assert!(de_a0_a1.2.contains_key(k), "{k}");
    }
    let de_a0_a3 = brutes
        .iter()
        .find(|b| b.0 == a[0].id.to_string() && b.1 == a[3].id.to_string())
        .unwrap();
    assert!(proche(poids(de_a0_a3), 0.5), "0.4 cosine + 0.1 boost");
    assert_eq!(de_a0_a3.2["source"].as_str(), Some("coactivation"));
    assert_eq!(
        de_a0_a3.2["reinforcement_count"].as_int64(),
        Some(1),
        "coalesce(null, 0) + 1"
    );
    // The `x`/`y` pairs wrote nothing.
    assert!(brutes
        .iter()
        .all(|b| b.0 != x.id.to_string() && b.0 != y.id.to_string()));
}

/// An archived `FORGOTTEN` edge revives at the weight it had, never under the forget
/// threshold 0.02, and the archive row goes away.
#[tokio::test]
async fn reinforce_revives_a_forgotten_edge_at_its_weight() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, a, b, c);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        let n = des_notes(&store, Some(p.id), 3).await;
        (a, b, c) = (n[0].id, n[1].id, n[2].id);
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        let noeud = |id: Uuid| {
            db.store()
                .nodes_by_label("Note")
                .into_iter()
                .find(|n| {
                    db.get_node(*n)
                        .unwrap()
                        .get_property("id")
                        .and_then(|v| v.as_str().map(str::to_string))
                        == Some(id.to_string())
                })
                .unwrap()
        };
        db.create_edge_with_props(
            noeud(a),
            noeud(b),
            "FORGOTTEN",
            [("weight", Value::from(0.6_f64))],
        );
        db.create_edge_with_props(
            noeud(c),
            noeud(a),
            "FORGOTTEN",
            [("weight", Value::from(0.001_f64))],
        );
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.reinforce_synapses(&[a, b, c], 0.1).await.unwrap(), 6);
    let sa = store.get_synapses(a).await.unwrap();
    assert!(
        sa.contains(&(b, 0.6)),
        "revived at the archived weight: {sa:?}"
    );
    assert!(
        sa.contains(&(c, 0.02)),
        "never under the forget threshold: {sa:?}"
    );
    store.close().await.unwrap();
    let oublies = base_brute(&base, |db| {
        let mut n = 0;
        for l in ["Note"] {
            for x in db.store().nodes_by_label(l) {
                for (_, e) in db.store().edges_from(x, Direction::Outgoing) {
                    if &*db.get_edge(e).unwrap().edge_type == "FORGOTTEN" {
                        n += 1;
                    }
                }
            }
        }
        n
    })
    .await;
    assert_eq!(oublies, 0, "the archive rows went away");
}

/// Parallel edges in one direction are not what Neo4j's `MERGE` writes: a write that would
/// have to choose refuses, writing nothing; the reads answer what is there.
#[tokio::test]
async fn parallel_edges_refuse_a_write_that_must_choose() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, b);
    {
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        let n = des_notes(&store, Some(p.id), 2).await;
        (a, b) = (n[0].id, n[1].id);
        store.create_synapses(a, &[(b, 0.5)]).await.unwrap();
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        let na = db.store().nodes_by_label("Note")[0];
        let (nb, _) = db.store().edges_from(na, Direction::Outgoing)[0];
        db.create_edge_with_props(na, nb, "SYNAPSE", [("weight", Value::from(0.9_f64))]);
    })
    .await;
    let store = open(&base).await;
    let avant = store.get_synapses(a).await.unwrap();
    assert!(!avant.is_empty());
    for e in [
        store.create_synapses(a, &[(b, 0.1)]).await.unwrap_err(),
        store.create_synapses(b, &[(a, 0.1)]).await.unwrap_err(),
        store.reinforce_synapses(&[a, b], 0.1).await.unwrap_err(),
    ] {
        assert!(format!("{e:#}").contains("parallèles"), "{e:#}");
    }
    assert_eq!(
        store.get_synapses(a).await.unwrap(),
        avant,
        "nothing was written"
    );
}

/// `coactivate_session`: scoped to the given project, a fixed +0.2, two edges per pair, and
/// the rule, measure, date and session recorded on each; `decayed_at` only on the first
/// direction (as the Cypher's two `SET`s).
#[tokio::test]
async fn coactivation_records_its_provenance_on_each_edge() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let a = des_notes(&store, Some(p.id), 2).await;
    let other = une_note(&store, Some(q.id)).await;
    let session = Uuid::new_v4();
    let prov = AccountableProvenance::new("co_activation", "3 notes read together").unwrap();

    assert_eq!(
        store
            .coactivate_session(p.id, &[a[0].id], session, &prov)
            .await
            .unwrap(),
        0,
        "fewer than two notes"
    );
    assert_eq!(
        store
            .coactivate_session(p.id, &[a[0].id, a[1].id, other.id], session, &prov)
            .await
            .unwrap(),
        2,
        "the note of the other project is outside the scope"
    );
    assert!(store.get_synapses(other.id).await.unwrap().is_empty());
    assert_eq!(
        store
            .coactivate_session(p.id, &[a[0].id, a[1].id], session, &prov)
            .await
            .unwrap(),
        2
    );
    assert!(proche(store.get_synapses(a[0].id).await.unwrap()[0].1, 0.7));
    store.close().await.unwrap();

    let b = base_brute(&base, |db| aretes_brutes(db, "SYNAPSE")).await;
    assert_eq!(b.len(), 2);
    let (premier, second) = if b[0].0 == a[0].id.to_string() {
        (&b[0], &b[1])
    } else {
        (&b[1], &b[0])
    };
    for e in [premier, second] {
        assert_eq!(e.2["rule"].as_str(), Some("co_activation"));
        assert_eq!(e.2["measure"].as_str(), Some("3 notes read together"));
        assert_eq!(
            e.2["last_session"].as_str(),
            Some(session.to_string().as_str())
        );
        assert!(e.2.contains_key("at") && e.2.contains_key("last_reinforced_at"));
        assert_eq!(e.2["reinforcement_count"].as_int64(), Some(2));
    }
    assert!(premier.2.contains_key("decayed_at"));
    assert!(!second.2.contains_key("decayed_at"));
}

/// `weaken_node_synapses`: every edge of the node (both directions, a decision too) loses
/// `amount`, those under the threshold are deleted, an edge that crosses two known projects
/// is left alone, and the answer is the number of edges that matched.
#[tokio::test]
async fn weakening_a_node_prunes_and_spares_what_crosses_projects() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let n = des_notes(&store, Some(p.id), 3).await;
    let far = une_note(&store, Some(q.id)).await;
    let free = une_note(&store, None).await;
    let (a, b, c) = (n[0].id, n[1].id, n[2].id);
    store
        .create_synapses(a, &[(b, 0.9), (c, 0.15), (far.id, 0.9), (free.id, 0.5)])
        .await
        .unwrap();

    // 3 edges pairs match (b, c, free) = 6 directed edges; `far` is another project.
    assert_eq!(store.weaken_node_synapses(a, 0.1, 0.1).await.unwrap(), 6);
    let sa = store.get_synapses(a).await.unwrap();
    assert!(sa.iter().any(|s| s.0 == b && proche(s.1, 0.8)), "{sa:?}");
    assert!(
        sa.iter().any(|s| s.0 == far.id && proche(s.1, 0.9)),
        "spared: {sa:?}"
    );
    assert!(
        sa.iter().any(|s| s.0 == free.id && proche(s.1, 0.4)),
        "{sa:?}"
    );
    assert!(sa.iter().all(|s| s.0 != c), "0.05 is under 0.1: deleted");
    assert!(
        store.get_synapses(c).await.unwrap().is_empty(),
        "both directions went"
    );
    // An absent node weakens nothing.
    assert_eq!(
        store
            .weaken_node_synapses(Uuid::new_v4(), 0.1, 0.1)
            .await
            .unwrap(),
        0
    );
    let e = store
        .weaken_node_synapses(a, f64::NAN, 0.1)
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("non fini"), "{e:#}");
}

// ---------------------------------------------------------------------------
// Energy and scars
// ---------------------------------------------------------------------------

/// `boost_energy`: `coalesce(energy, 1.0) + amount` capped at 1.0 (never a lazy decay of the
/// old value), and `last_activated` / `energy_updated_at` set; an absent note is a no-op.
#[tokio::test]
async fn boosting_adds_to_the_stored_energy_and_stamps_the_note() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let mut n = Note::new(None, NoteType::Observation, "c".into(), "t".into());
    n.energy = 0.4;
    // Stale on purpose: a boost must not decay it again.
    n.last_activated = Some(Utc::now() - Duration::days(365));
    store.create_note(&n).await.unwrap();
    store.boost_energy(n.id, 0.3).await.unwrap();
    let lue = store.get_note(n.id).await.unwrap().unwrap();
    assert!(proche(lue.energy, 0.7), "{}", lue.energy);
    assert!(Utc::now() - lue.last_activated.unwrap() < Duration::minutes(1));
    store.boost_energy(n.id, 0.9).await.unwrap();
    assert!(proche(
        store.get_note(n.id).await.unwrap().unwrap().energy,
        1.0
    ));
    // A negative amount lowers it (no floor in the Cypher).
    store.boost_energy(n.id, -0.25).await.unwrap();
    assert!(proche(
        store.get_note(n.id).await.unwrap().unwrap().energy,
        0.75
    ));
    store.boost_energy(Uuid::new_v4(), 0.5).await.unwrap();
    let e = store.boost_energy(n.id, f64::INFINITY).await.unwrap_err();
    assert!(format!("{e:#}").contains("non fini"), "{e:#}");
    store.close().await.unwrap();
    let stampe = base_brute(&base, |db| {
        let n = db.store().nodes_by_label("Note")[0];
        db.get_node(n)
            .unwrap()
            .get_property("energy_updated_at")
            .cloned()
    })
    .await;
    assert!(stampe.is_some(), "energy_updated_at is written by a boost");
}

/// `update_energy_scores`: `energy * exp(-days_since_energy_updated_at / half_life)`, floored
/// to 0.0 under 0.05, only active notes with `energy > 0`; every such note is stamped; a
/// change under 0.001 is not counted; a note with no stamp, or of a dormant project, is only
/// stamped. The three stored forms of the stamp are read.
#[tokio::test]
async fn the_energy_decay_follows_the_stamp_the_half_life_and_the_dormant_projects() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (actif, dormant);
    let (iso, ts, zo, sans, mort, frais, du_dormant);
    {
        let store = open(&base).await;
        actif = projet_actif(&store).await;
        dormant = a_project(); // no sync, no chat: dormant
        store.create_project(&dormant).await.unwrap();
        let n = des_notes(&store, Some(actif.id), 6).await;
        (iso, ts, zo, sans, mort, frais) = (n[0].id, n[1].id, n[2].id, n[3].id, n[4].id, n[5].id);
        du_dormant = une_note(&store, Some(dormant.id)).await.id;
        store.close().await.unwrap();
    }
    let il_y_a = |j: i64| Utc::now() - Duration::days(j);
    let (i, t, z, m, f, d) = (iso, ts, zo, mort, frais, du_dormant);
    base_brute(&base, move |db| {
        let par_id = |id: Uuid| {
            db.store()
                .nodes_by_label("Note")
                .into_iter()
                .find(|n| {
                    db.get_node(*n)
                        .unwrap()
                        .get_property("id")
                        .and_then(|v| v.as_str().map(str::to_string))
                        == Some(id.to_string())
                })
                .unwrap()
        };
        // 90 days idle, tau = 90: energy 1.0 * exp(-1).
        db.set_node_property(
            par_id(i),
            "energy_updated_at",
            Value::from(il_y_a(90).to_rfc3339()),
        );
        db.set_node_property(
            par_id(t),
            "energy_updated_at",
            Value::Timestamp(Timestamp::from_micros(il_y_a(90).timestamp_micros())),
        );
        db.set_node_property(par_id(z), "energy_updated_at", zoned(il_y_a(90)));
        // 400 days idle: exp(-4.44) = 0.0118 < 0.05, floored to 0.
        db.set_node_property(
            par_id(m),
            "energy_updated_at",
            Value::from(il_y_a(400).to_rfc3339()),
        );
        // Stamped one minute ago: the change is under 0.001, not counted.
        db.set_node_property(
            par_id(f),
            "energy_updated_at",
            Value::from((Utc::now() - Duration::minutes(1)).to_rfc3339()),
        );
        // The dormant project's note: 90 idle days, frozen.
        db.set_node_property(
            par_id(d),
            "energy_updated_at",
            Value::from(il_y_a(90).to_rfc3339()),
        );
    })
    .await;
    let store = open(&base).await;
    let n = store.update_energy_scores(90.0).await.unwrap();
    assert_eq!(
        n, 4,
        "iso, timestamp, zoned and the dead one change; the others do not"
    );
    let e = |id: Uuid| {
        let store = &store;
        async move { store.get_note(id).await.unwrap().unwrap().energy }
    };
    let attendu = (-1.0_f64).exp();
    for id in [iso, ts, zo] {
        assert!((e(id).await - attendu).abs() < 2e-3, "{}", e(id).await);
    }
    assert_eq!(e(mort).await, 0.0, "a dead neuron");
    assert!(e(sans).await > 0.99, "no stamp: only stamped");
    assert!(e(frais).await > 0.99);
    assert!(e(du_dormant).await > 0.99, "a dormant project is frozen");
    // Temporally idempotent: the stamp moved to now, a second pass changes nothing.
    assert_eq!(store.update_energy_scores(90.0).await.unwrap(), 0);
    // A dead note (energy 0) is out of the pass for good.
    assert_eq!(e(mort).await, 0.0);
    for bad in [0.0, -1.0, f64::NAN] {
        let err = store.update_energy_scores(bad).await.unwrap_err();
        assert!(
            format!("{err:#}").contains("strictement positif"),
            "{err:#}"
        );
    }
}

/// REGRESSION. A stamp that is present but unreadable refuses the whole pass and writes
/// nothing: it is never read as "no date" (which would silently skip the decay), as 130 713
/// migrated nodes once were. The same pass over readable stamps then works.
#[tokio::test]
async fn an_unreadable_energy_stamp_refuses_the_pass_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (bonne, cassee);
    {
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        let n = des_notes(&store, Some(p.id), 2).await;
        (bonne, cassee) = (n[0].id, n[1].id);
        store.close().await.unwrap();
    }
    let il_y_a = (Utc::now() - Duration::days(90)).to_rfc3339();
    let (b, c) = (bonne, cassee);
    base_brute(&base, move |db| {
        for n in db.store().nodes_by_label("Note") {
            let id = db
                .get_node(n)
                .unwrap()
                .get_property("id")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap();
            if id == b.to_string() {
                db.set_node_property(n, "energy_updated_at", Value::from(il_y_a.clone()));
            } else if id == c.to_string() {
                db.set_node_property(n, "energy_updated_at", Value::from("hier soir"));
            }
        }
    })
    .await;
    let store = open(&base).await;
    let e = store.update_energy_scores(90.0).await.unwrap_err();
    assert!(format!("{e:#}").contains("energy_updated_at"), "{e:#}");
    assert!(
        store.get_note(bonne).await.unwrap().unwrap().energy > 0.99,
        "the readable note was not decayed by a refused pass"
    );
}

/// A note quarantined at open is not decayed by a pass that would miss it: the scans refuse.
#[tokio::test]
async fn a_quarantined_note_makes_the_scans_refuse_and_its_id_is_untouchable() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let id_casse = Uuid::new_v4();
    {
        let store = open(&base).await;
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        // A note with no `content`: unreadable.
        db.create_node_with_props(
            &["Note"],
            vec![
                ("id", Value::from(id_casse.to_string())),
                ("status", Value::from("active")),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
    for e in [
        store.update_energy_scores(90.0).await.unwrap_err(),
        store.init_note_energy(None).await.unwrap_err(),
        store
            .list_notes_needing_synapses(10, 0, None)
            .await
            .unwrap_err(),
        store
            .get_all_synapse_weights(Some(Uuid::new_v4()))
            .await
            .unwrap_err(),
        store
            .get_project_note_synapses(Uuid::new_v4(), 0.0)
            .await
            .unwrap_err(),
    ] {
        assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
    }
    // By id: neither read nor written.
    for e in [
        store.get_synapses(id_casse).await.unwrap_err(),
        store
            .create_synapses(id_casse, &[(Uuid::new_v4(), 0.5)])
            .await
            .unwrap_err(),
        store.boost_energy(id_casse, 0.1).await.unwrap_err(),
        store.heal_scars(id_casse).await.unwrap_err(),
        store.apply_scars(&[id_casse], 0.2).await.unwrap_err(),
        store.track_reactivation(&[id_casse]).await.unwrap_err(),
    ] {
        assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
    }
    // The global weights do not need an index: the unreadable node holds no edge.
    assert!(store
        .get_all_synapse_weights(None)
        .await
        .unwrap()
        .is_empty());
}

/// `init_note_energy`: only a note whose node has NO `energy` property (the decoded note
/// cannot tell: it defaults to 1.0), `energy = 1.0` and `last_activated` copied from
/// `last_confirmed_at`, optionally for one project.
#[tokio::test]
async fn init_note_energy_fills_only_the_notes_with_no_energy_property() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, q);
    let (sans_a, sans_b, avec);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        q = projet_actif(&store).await;
        sans_a = une_note(&store, Some(p.id)).await.id;
        sans_b = une_note(&store, Some(q.id)).await.id;
        avec = une_note(&store, Some(p.id)).await.id;
        store.close().await.unwrap();
    }
    let jadis = zoned(Utc.with_ymd_and_hms(2025, 1, 2, 3, 4, 5).unwrap());
    let (sa, sb) = (sans_a, sans_b);
    base_brute(&base, move |db| {
        for n in db.store().nodes_by_label("Note") {
            let id = db
                .get_node(n)
                .unwrap()
                .get_property("id")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap();
            if id == sa.to_string() || id == sb.to_string() {
                db.remove_node_property(n, "energy");
                db.remove_node_property(n, "last_activated");
                if id == sa.to_string() {
                    db.set_node_property(n, "last_confirmed_at", jadis.clone());
                }
            }
        }
    })
    .await;
    let store = open(&base).await;
    assert_eq!(
        store.init_note_energy(Some(p.id)).await.unwrap(),
        1,
        "p's note only"
    );
    assert_eq!(store.init_note_energy(Some(p.id)).await.unwrap(), 0);
    assert_eq!(store.init_note_energy(None).await.unwrap(), 1, "then q's");
    assert_eq!(store.init_note_energy(None).await.unwrap(), 0);
    let a = store.get_note(sans_a).await.unwrap().unwrap();
    assert_eq!(a.energy, 1.0);
    assert_eq!(
        a.last_activated,
        Some(Utc.with_ymd_and_hms(2025, 1, 2, 3, 4, 5).unwrap()),
        "last_activated is last_confirmed_at, read in its stored form"
    );
    let _ = avec;
}

/// `track_reactivation`: one row per id of the list, so a repeated id counts and adds each
/// time; an absent id is not counted; `last_reactivated` is set.
#[tokio::test]
async fn reactivations_count_one_row_per_id() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let n = des_notes(&store, None, 2).await;
    let ghost = Uuid::new_v4();
    assert_eq!(store.track_reactivation(&[]).await.unwrap(), 0);
    assert_eq!(
        store
            .track_reactivation(&[n[0].id, n[0].id, n[1].id, ghost])
            .await
            .unwrap(),
        3
    );
    let a = store.get_note(n[0].id).await.unwrap().unwrap();
    assert_eq!(a.reactivation_count, 2);
    assert!(a.last_reactivated.is_some());
    assert_eq!(
        store
            .get_note(n[1].id)
            .await
            .unwrap()
            .unwrap()
            .reactivation_count,
        1
    );
}

async fn une_decision(base: &Path) -> Uuid {
    let id = Uuid::new_v4();
    base_brute(base, move |db| {
        db.create_node_with_props(
            &["Decision"],
            vec![
                ("id", Value::from(id.to_string())),
                ("description", Value::from("d")),
                ("rationale", Value::from("r")),
                ("decided_at", Value::from("2026-01-01T00:00:00Z")),
                ("embedding", Value::from("present")),
            ],
        );
    })
    .await;
    id
}

/// `apply_scars` / `heal_scars` work on notes AND decisions; the count is the rows matched
/// (a repeated id scars twice), the scar is capped at 1.0, and `heal_scars` answers whether
/// it found the node (note first).
#[tokio::test]
async fn scars_apply_to_notes_and_decisions_and_heal_answers_whether_it_found() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let n;
    {
        let store = open(&base).await;
        n = une_note(&store, None).await.id;
        store.close().await.unwrap();
    }
    let d = une_decision(&base).await;
    let store = open(&base).await;
    let ghost = Uuid::new_v4();
    assert_eq!(store.apply_scars(&[], 0.2).await.unwrap(), 0);
    assert_eq!(store.apply_scars(&[n, d, ghost], 0.2).await.unwrap(), 2);
    assert!(proche(
        store.get_note(n).await.unwrap().unwrap().scar_intensity,
        0.2
    ));
    assert_eq!(
        store.apply_scars(&[n, n, n, n, n, n], 0.2).await.unwrap(),
        6
    );
    assert!(
        proche(
            store.get_note(n).await.unwrap().unwrap().scar_intensity,
            1.0
        ),
        "capped"
    );
    assert!(proche(
        store.get_decision(d).await.unwrap().unwrap().scar_intensity,
        0.2
    ));
    assert!(store.heal_scars(n).await.unwrap());
    assert_eq!(
        store.get_note(n).await.unwrap().unwrap().scar_intensity,
        0.0
    );
    assert!(store.heal_scars(d).await.unwrap());
    assert_eq!(
        store.get_decision(d).await.unwrap().unwrap().scar_intensity,
        0.0
    );
    assert!(!store.heal_scars(ghost).await.unwrap());
    let e = store.apply_scars(&[n], f64::NAN).await.unwrap_err();
    assert!(format!("{e:#}").contains("non fini"), "{e:#}");
}

// ---------------------------------------------------------------------------
// Cross-entity, lists, project reads
// ---------------------------------------------------------------------------

/// Decision ↔ Note: `create_cross_entity_synapses` links any note or decision, the read
/// answers the neighbour's kind, and `delete_note` (a DETACH DELETE) takes its synapses.
#[tokio::test]
async fn a_decision_and_a_note_are_linked_and_the_note_s_deletion_takes_the_edge() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, b);
    {
        let store = open(&base).await;
        let n = des_notes(&store, None, 2).await;
        (a, b) = (n[0].id, n[1].id);
        store.close().await.unwrap();
    }
    let d = une_decision(&base).await;
    let store = open(&base).await;
    assert_eq!(
        store
            .create_cross_entity_synapses(d, &[(a, 0.7), (b, 0.4), (Uuid::new_v4(), 0.9)])
            .await
            .unwrap(),
        4
    );
    assert_eq!(
        store.get_cross_entity_synapses(d).await.unwrap(),
        vec![(a, 0.7, "Note".to_string()), (b, 0.4, "Note".to_string())]
    );
    assert_eq!(
        store.get_cross_entity_synapses(a).await.unwrap(),
        vec![(d, 0.7, "Decision".to_string())]
    );
    // `get_synapses` is Note-to-Note only.
    assert!(store.get_synapses(a).await.unwrap().is_empty());
    // A decision's weights count in the global distribution, and not in a project's.
    assert_eq!(store.get_all_synapse_weights(None).await.unwrap().len(), 4);
    assert!(store.delete_note(a).await.unwrap());
    assert_eq!(
        store.get_cross_entity_synapses(d).await.unwrap(),
        vec![(b, 0.4, "Note".to_string())]
    );
    assert_eq!(store.get_all_synapse_weights(None).await.unwrap().len(), 2);
    // The decision is scarred, weakened, deleted like any other.
    assert_eq!(store.weaken_node_synapses(d, 0.1, 0.0).await.unwrap(), 2);
    store.delete_decision(d).await.unwrap();
    assert!(store.get_synapses(b).await.unwrap().is_empty());
}

/// `list_notes_needing_synapses`: active or needs-review notes that carry an `embedding`
/// property and no outgoing SYNAPSE, ordered by creation then id, paged, with the total;
/// per project when asked. `list_decisions_needing_synapses` likewise.
#[tokio::test]
async fn notes_and_decisions_needing_synapses_are_listed_in_order_with_their_total() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, q);
    let notes;
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        q = projet_actif(&store).await;
        let mut v = Vec::new();
        for k in 0..4 {
            let mut n = Note::new(
                Some(if k == 3 { q.id } else { p.id }),
                NoteType::Observation,
                format!("c{k}"),
                "t".into(),
            );
            n.created_at = Utc
                .with_ymd_and_hms(2026, 1, 1 + k as u32, 0, 0, 0)
                .unwrap();
            store.create_note(&n).await.unwrap();
            v.push(n.id);
        }
        notes = v;
        // notes[1] has a synapse: it does not need one.
        store
            .create_synapses(notes[1], &[(notes[0], 0.5)])
            .await
            .unwrap();
        store.close().await.unwrap();
    }
    let ns = notes.clone();
    base_brute(&base, move |db| {
        for n in db.store().nodes_by_label("Note") {
            let id = db
                .get_node(n)
                .unwrap()
                .get_property("id")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap();
            // notes[2] is archived: not listed. notes[0], [1], [3] get an embedding.
            if id == ns[2].to_string() {
                db.set_node_property(n, "status", Value::from("archived"));
                db.set_node_property(n, "embedding", Value::from("v"));
            } else {
                db.set_node_property(n, "embedding", Value::from("v"));
            }
        }
    })
    .await;
    let d1 = une_decision(&base).await;
    let store = open(&base).await;
    // notes[0] has an INCOMING synapse only: it still needs an outgoing one? No — the edges
    // are bidirectional, so it has one outgoing. notes[0] and notes[1] are both covered.
    let (liste, total) = store
        .list_notes_needing_synapses(10, 0, None)
        .await
        .unwrap();
    assert_eq!(
        (liste.iter().map(|n| n.id).collect::<Vec<_>>(), total),
        (vec![notes[3]], 1)
    );
    let (liste, total) = store
        .list_notes_needing_synapses(10, 0, Some(p.id))
        .await
        .unwrap();
    assert_eq!((liste.len(), total), (0, 0));
    store.delete_synapses(notes[1]).await.unwrap();
    let (liste, total) = store
        .list_notes_needing_synapses(10, 0, None)
        .await
        .unwrap();
    assert_eq!(
        liste.iter().map(|n| n.id).collect::<Vec<_>>(),
        vec![notes[0], notes[1], notes[3]]
    );
    assert_eq!(total, 3);
    let (liste, total) = store.list_notes_needing_synapses(1, 1, None).await.unwrap();
    assert_eq!(
        (liste.iter().map(|n| n.id).collect::<Vec<_>>(), total),
        (vec![notes[1]], 3)
    );
    let (liste, total) = store.list_notes_needing_synapses(0, 0, None).await.unwrap();
    assert_eq!((liste.len(), total), (0, 3), "limit 0 still counts");
    let (liste, total) = store
        .list_notes_needing_synapses(10, 0, Some(p.id))
        .await
        .unwrap();
    assert_eq!((liste.len(), total), (2, 2));
    let (_, total) = store
        .list_notes_needing_synapses(10, 0, Some(Uuid::new_v4()))
        .await
        .unwrap();
    assert_eq!(total, 0);

    let (liste, total) = store.list_decisions_needing_synapses(10, 0).await.unwrap();
    assert_eq!(
        (liste.iter().map(|d| d.id).collect::<Vec<_>>(), total),
        (vec![d1], 1)
    );
    store
        .create_cross_entity_synapses(d1, &[(notes[0], 0.5)])
        .await
        .unwrap();
    assert_eq!(
        store
            .list_decisions_needing_synapses(10, 0)
            .await
            .unwrap()
            .1,
        0
    );
}

/// The project reads: `get_project_note_synapses` (both ends in the project, weight at least
/// the minimum, one row per pair with the lower id first, heaviest first),
/// `get_all_synapse_weights` (out-edges of the project's notes), `get_synapse_graph`
/// (current notes only, weight strictly over the minimum, no skill-evolution audit note).
#[tokio::test]
async fn the_project_synapse_reads_filter_and_order() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let n = des_notes(&store, Some(p.id), 5).await;
    let far = une_note(&store, Some(q.id)).await;
    let mut audit = Note::new(
        Some(p.id),
        NoteType::Observation,
        "c".into(),
        "skill-evolution".into(),
    );
    audit.created_by = "skill-evolution".into();
    store.create_note(&audit).await.unwrap();
    store
        .create_synapses(
            n[0].id,
            &[
                (n[1].id, 0.9),
                (n[2].id, 0.5),
                (far.id, 0.8),
                (audit.id, 0.7),
            ],
        )
        .await
        .unwrap();
    store
        .create_synapses(n[1].id, &[(n[3].id, 0.3)])
        .await
        .unwrap();
    store
        .create_synapses(n[0].id, &[(n[4].id, 0.95)])
        .await
        .unwrap();
    store.supersede_note(n[4].id, n[2].id).await.unwrap();

    let paires = store.get_project_note_synapses(p.id, 0.5).await.unwrap();
    let attendu_cles: Vec<(String, String)> = {
        let mut v: Vec<(String, String, f64)> = vec![
            (n[0].id.to_string(), n[1].id.to_string(), 0.9),
            (n[0].id.to_string(), n[2].id.to_string(), 0.5),
            (n[0].id.to_string(), audit.id.to_string(), 0.7),
            (n[0].id.to_string(), n[4].id.to_string(), 0.95),
        ];
        // The lower id first in each pair, whichever direction was created first.
        v = v
            .into_iter()
            .map(|(a, b, w)| if a < b { (a, b, w) } else { (b, a, w) })
            .collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2).then(b.0.cmp(&a.0)).then(b.1.cmp(&a.1)));
        v.into_iter().map(|(a, b, _)| (a, b)).collect()
    };
    assert_eq!(
        paires
            .iter()
            .map(|(a, b, _)| (a.clone(), b.clone()))
            .collect::<Vec<_>>(),
        attendu_cles
    );
    assert!(paires.iter().all(|(a, b, _)| a < b));
    assert_eq!(
        paires.iter().map(|x| x.2).collect::<Vec<_>>(),
        vec![0.95, 0.9, 0.7, 0.5]
    );
    // The minimum is inclusive here.
    assert_eq!(
        store
            .get_project_note_synapses(p.id, 0.9)
            .await
            .unwrap()
            .len(),
        2
    );

    // Out-edges of the project's notes to notes: 0↔1, 0↔2, 0↔4, 0↔audit, 1↔3 (2 each = 10),
    // plus `0 → far`; `far → 0` is out of `q`'s.
    let w = store.get_all_synapse_weights(Some(p.id)).await.unwrap();
    assert_eq!(w.len(), 11, "{w:?}");
    assert_eq!(
        store.get_all_synapse_weights(Some(q.id)).await.unwrap(),
        vec![0.8]
    );
    assert_eq!(
        store
            .get_all_synapse_weights(Some(Uuid::new_v4()))
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(store.get_all_synapse_weights(None).await.unwrap().len(), 12);

    // The graph skills read: strictly over 0.5, no audit note, no `far`.
    let g = store.get_synapse_graph(p.id, 0.5).await.unwrap();
    let mut poids: Vec<f64> = g.iter().map(|x| x.2).collect();
    poids.sort_by(|a, b| b.total_cmp(a));
    assert_eq!(
        poids,
        vec![0.9, 0.9],
        "0↔1 in both directions; 0.5 is not over 0.5; audit and superseded left out"
    );
    store.close().await.unwrap();
}

/// `get_project_note_entity_links`: a link to a project or a note is `unknown` (none of the
/// code entities the Cypher names) with its id; a note of another project has none.
#[tokio::test]
async fn note_entity_links_of_a_project_are_listed_as_unknown_ids() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let n = des_notes(&store, Some(p.id), 2).await;
    let other = une_note(&store, Some(q.id)).await;
    store
        .link_note_to_entity(n[0].id, &EntityType::Note, &n[1].id.to_string(), None, None)
        .await
        .unwrap();
    store
        .link_note_to_entity(n[0].id, &EntityType::Project, &p.id.to_string(), None, None)
        .await
        .unwrap();
    store
        .link_note_to_entity(
            other.id,
            &EntityType::Project,
            &q.id.to_string(),
            None,
            None,
        )
        .await
        .unwrap();
    let mut attendu = vec![
        (
            n[0].id.to_string(),
            "unknown".to_string(),
            n[1].id.to_string(),
        ),
        (n[0].id.to_string(), "unknown".to_string(), p.id.to_string()),
    ];
    attendu.sort();
    assert_eq!(
        store.get_project_note_entity_links(p.id).await.unwrap(),
        attendu
    );
    assert!(store
        .get_project_note_entity_links(Uuid::new_v4())
        .await
        .unwrap()
        .is_empty());
}

// ---------------------------------------------------------------------------
// Deep maintenance, persona energy
// ---------------------------------------------------------------------------

/// `mark_deep_maintenance` stamps the project; `get_deep_maintenance_times` lists the stamped
/// ones (any stored form of the date); an absent project is a no-op; an unreadable stamp
/// refuses the read, it is not skipped.
#[tokio::test]
async fn the_deep_maintenance_clock_is_written_read_and_never_skips_an_unreadable_date() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q, r) = (a_project(), a_project(), a_project());
    for x in [&p, &q, &r] {
        store.create_project(x).await.unwrap();
    }
    assert!(store.get_deep_maintenance_times().await.unwrap().is_empty());
    store.mark_deep_maintenance(p.id).await.unwrap();
    store.mark_deep_maintenance(Uuid::new_v4()).await.unwrap();
    let t = store.get_deep_maintenance_times().await.unwrap();
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].0, p.id);
    assert!(Utc::now() - t[0].1 < Duration::minutes(1));
    store.close().await.unwrap();

    let jadis = Utc.with_ymd_and_hms(2025, 5, 6, 7, 8, 9).unwrap();
    let qid = q.id;
    base_brute(&base, move |db| {
        for n in db.store().nodes_by_label("Project") {
            let id = db
                .get_node(n)
                .unwrap()
                .get_property("id")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap();
            if id == qid.to_string() {
                db.set_node_property(n, "last_deep_maintenance_at", zoned(jadis));
            }
        }
    })
    .await;
    let store = open(&base).await;
    let t = store.get_deep_maintenance_times().await.unwrap();
    assert_eq!(t.len(), 2);
    assert!(t.contains(&(q.id, jadis)), "ZonedDatetime read: {t:?}");
    store.close().await.unwrap();

    let rid = r.id;
    base_brute(&base, move |db| {
        for n in db.store().nodes_by_label("Project") {
            let id = db
                .get_node(n)
                .unwrap()
                .get_property("id")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap();
            if id == rid.to_string() {
                db.set_node_property(n, "last_deep_maintenance_at", Value::from("jamais"));
            }
        }
    })
    .await;
    let store = open(&base).await;
    let e = store.get_deep_maintenance_times().await.unwrap_err();
    assert!(
        format!("{e:#}").contains("last_deep_maintenance_at"),
        "{e:#}"
    );
}

fn une_persona(energie: f64, accumule: f64) -> PersonaNode {
    PersonaNode {
        id: Uuid::new_v4(),
        project_id: None,
        name: "p".into(),
        description: "d".into(),
        status: PersonaStatus::Active,
        complexity_default: None,
        timeout_secs: None,
        max_cost_usd: None,
        model_preference: None,
        system_prompt_override: None,
        energy: energie,
        cohesion: 0.0,
        activation_count: 0,
        success_rate: 0.0,
        avg_duration_secs: 0.0,
        last_activated: None,
        energy_boost_accumulated: accumule,
        energy_history: vec![],
        origin: PersonaOrigin::AutoBuild,
        created_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
        updated_at: None,
    }
}

/// `rate_limited_energy_boost`: applied while `accumulated + boost <= max`, adding to the
/// accumulator and to the energy (capped at 1.0); refused (false, nothing written) past it;
/// an absent persona answers false.
#[tokio::test]
async fn a_persona_boost_is_rate_limited_by_its_accumulator() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let p = une_persona(0.5, 0.0);
    store.create_persona(&p).await.unwrap();
    assert!(store
        .rate_limited_energy_boost(p.id, 0.2, 0.5)
        .await
        .unwrap());
    assert!(store
        .rate_limited_energy_boost(p.id, 0.2, 0.5)
        .await
        .unwrap());
    let l = store.get_persona(p.id).await.unwrap().unwrap();
    assert!(
        proche(l.energy, 0.9) && proche(l.energy_boost_accumulated, 0.4),
        "{l:?}"
    );
    assert!(
        !store
            .rate_limited_energy_boost(p.id, 0.2, 0.5)
            .await
            .unwrap(),
        "0.4 + 0.2 > 0.5"
    );
    let l = store.get_persona(p.id).await.unwrap().unwrap();
    assert!(proche(l.energy, 0.9) && proche(l.energy_boost_accumulated, 0.4));
    assert!(store
        .rate_limited_energy_boost(p.id, 0.1, 0.5)
        .await
        .unwrap());
    assert!(
        proche(store.get_persona(p.id).await.unwrap().unwrap().energy, 1.0),
        "capped"
    );
    assert!(!store
        .rate_limited_energy_boost(Uuid::new_v4(), 0.1, 1.0)
        .await
        .unwrap());
}

// ---------------------------------------------------------------------------
// Durability, cuts, refusals
// ---------------------------------------------------------------------------

/// Everything that serves survives a close and a reopen: edges with their properties,
/// energy, scars, the deep-maintenance clock.
#[tokio::test]
async fn the_neural_layer_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, n);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        n = des_notes(&store, Some(p.id), 3).await;
        store
            .reinforce_synapses(&[n[0].id, n[1].id, n[2].id], 0.2)
            .await
            .unwrap();
        store.apply_scars(&[n[0].id], 0.4).await.unwrap();
        store.boost_energy(n[1].id, -0.5).await.unwrap();
        store.track_reactivation(&[n[2].id]).await.unwrap();
        store.mark_deep_maintenance(p.id).await.unwrap();
        store.close().await.unwrap();
    }
    let store = open(&base).await;
    assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
    assert_eq!(store.get_synapses(n[0].id).await.unwrap().len(), 2);
    assert!(proche(
        store
            .get_note(n[0].id)
            .await
            .unwrap()
            .unwrap()
            .scar_intensity,
        0.4
    ));
    assert!(proche(
        store.get_note(n[1].id).await.unwrap().unwrap().energy,
        0.5
    ));
    assert_eq!(
        store
            .get_note(n[2].id)
            .await
            .unwrap()
            .unwrap()
            .reactivation_count,
        1
    );
    assert_eq!(store.get_deep_maintenance_times().await.unwrap().len(), 1);
    assert_eq!(
        store
            .get_all_synapse_weights(Some(p.id))
            .await
            .unwrap()
            .len(),
        6
    );
}

/// A reinforcement cut at every engine write: the store refuses until reopened (what is on
/// disk is a prefix of the writes), the reopened base opens clean, and every state it can be
/// in is readable: either the two edges and counts of before, or those of after, or a prefix
/// whose weight is the old one or the new one and never anything else.
#[tokio::test]
async fn a_reinforcement_cut_at_every_write_leaves_a_readable_prefix() {
    for k in 1..=24 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        let n = des_notes(&store, Some(p.id), 2).await;
        let ids = [n[0].id, n[1].id];
        store.reinforce_synapses(&ids, 0.2).await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.reinforce_synapses(&ids, 0.2).await.is_ok() {
            // The cut reached the k-1 writes before it: at least eight engine writes.
            assert!(k > 8, "la coupure n'a atteint que {} points", k - 1);
            return;
        }
        assert!(
            store.get_synapses(ids[0]).await.is_err(),
            "k={k}: indéterminé"
        );
        drop(store);
        let store = open(&base).await;
        assert!(
            store.rapport_d_ouverture().en_quarantaine.is_empty(),
            "k={k}"
        );
        // One row per distinct weight: the two directions may have been cut apart.
        let s = store.get_synapses(ids[0]).await.unwrap();
        assert!(!s.is_empty() && s.len() <= 2, "k={k}: {s:?}");
        assert!(
            s.iter().all(|x| proche(x.1, 0.5) || proche(x.1, 0.7)),
            "k={k}: a weight is the old or the new one, never a mix: {s:?}"
        );
        store.close().await.unwrap();
        // Every property of every edge, as the engine holds it. The weight is written LAST of
        // an edge's properties: an edge whose weight is the new one carries ALL the new
        // properties; one whose weight is the old one carries a prefix of them (counter and
        // stamps new or still old) and nothing else (source and creation date never move).
        let brutes = base_brute(&base, |db| aretes_brutes(db, "SYNAPSE")).await;
        assert_eq!(brutes.len(), 2, "k={k}: no edge lost or added");
        for e in &brutes {
            let compte = e.2["reinforcement_count"].as_int64().unwrap();
            assert_eq!(e.2["source"].as_str(), Some("coactivation"), "k={k}");
            assert!(e.2.contains_key("created_at"), "k={k}");
            if proche(poids(e), 0.7) {
                assert_eq!(compte, 2, "k={k}: a new weight carries the new counter");
                for c in ["last_reinforced_at", "decayed_at"] {
                    assert!(e.2.contains_key(c), "k={k}: {c}");
                }
            } else {
                assert!(proche(poids(e), 0.5), "k={k}: {}", poids(e));
                assert!(compte == 1 || compte == 2, "k={k}: {compte}");
            }
        }
        // And the base keeps working: the next reinforcement goes through.
        let store = open(&base).await;
        store.reinforce_synapses(&ids, 0.1).await.unwrap();
    }
    panic!("the cut never ran out of writes: raise the bound");
}

/// `rate_limited_energy_boost` cut at every engine write: the accumulator is written BEFORE the
/// energy, so a cut may charge a boost that was not given (a boost lost) and can never leave
/// an energy raised without its charge (the cap exceeded).
#[tokio::test]
async fn a_persona_boost_cut_at_every_write_never_exceeds_the_cap() {
    let mut fin = None;
    for k in 1..=6 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = une_persona(0.5, 0.0);
        store.create_persona(&p).await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store
            .rate_limited_energy_boost(p.id, 0.2, 0.5)
            .await
            .is_ok()
        {
            fin = Some(k);
            break;
        }
        assert!(store.get_persona(p.id).await.is_err(), "k={k}: indéterminé");
        drop(store);
        let store = open(&base).await;
        let l = store.get_persona(p.id).await.unwrap().unwrap();
        assert!(
            proche(l.energy, 0.5) || proche(l.energy, 0.7),
            "k={k}: {l:?}"
        );
        assert!(
            proche(l.energy_boost_accumulated, 0.0) || proche(l.energy_boost_accumulated, 0.2),
            "k={k}: {l:?}"
        );
        if proche(l.energy, 0.7) {
            assert!(
                proche(l.energy_boost_accumulated, 0.2),
                "k={k}: energy raised without its charge: {l:?}"
            );
        }
    }
    assert_eq!(fin, Some(4), "three engine writes");
}

/// `COALESCE(e.path, e.id)`: a target that carries a `path` property answers it.
#[tokio::test]
async fn an_entity_link_answers_the_target_path_when_it_has_one() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, n, q);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        q = projet_actif(&store).await;
        n = des_notes(&store, Some(p.id), 1).await;
        store
            .link_note_to_entity(n[0].id, &EntityType::Project, &q.id.to_string(), None, None)
            .await
            .unwrap();
        store.close().await.unwrap();
    }
    let qid = q.id;
    base_brute(&base, move |db| {
        for x in db.store().nodes_by_label("Project") {
            let id = db
                .get_node(x)
                .unwrap()
                .get_property("id")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap();
            if id == qid.to_string() {
                db.set_node_property(x, "path", Value::from("src/lib.rs"));
            }
        }
    })
    .await;
    let store = open(&base).await;
    assert_eq!(
        store.get_project_note_entity_links(p.id).await.unwrap(),
        vec![(
            n[0].id.to_string(),
            "unknown".to_string(),
            "src/lib.rs".to_string()
        )]
    );
}

/// A boost and a scar cut at every engine write: the note reads back, with an energy that is
/// the old or the new one.
#[tokio::test]
async fn a_boost_cut_at_every_write_keeps_a_valid_energy() {
    let mut sans_coupure = None;
    for k in 1..=6 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let mut note = Note::new(None, NoteType::Observation, "c".into(), "t".into());
        note.energy = 0.4;
        store.create_note(&note).await.unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.boost_energy(note.id, 0.3).await.is_ok() {
            sans_coupure = Some(k);
            break;
        }
        assert!(store.get_note(note.id).await.is_err(), "k={k}: indéterminé");
        drop(store);
        let store = open(&base).await;
        let e = store.get_note(note.id).await.unwrap().unwrap().energy;
        assert!(proche(e, 0.4) || proche(e, 0.7), "k={k}: {e}");
        store.close().await.unwrap();
        // The energy is written LAST: a note with the new energy carries both stamps.
        let stamps = base_brute(&base, |db| {
            let n = db.store().nodes_by_label("Note")[0];
            let node = db.get_node(n).unwrap();
            (
                node.get_property("last_activated").is_some(),
                node.get_property("energy_updated_at").is_some(),
            )
        })
        .await;
        if proche(e, 0.7) {
            assert!(stamps.1, "k={k}: new energy without its stamp");
        }
        let store = open(&base).await;
        store.boost_energy(note.id, 0.3).await.unwrap();
    }
    // The cut reached the writes before the first k that goes through: at least three.
    let k = sans_coupure.expect("le boost n'a jamais abouti : relever la borne");
    assert!(k > 3, "la coupure n'a atteint que {} points", k - 1);
}

/// The sub-lot's own refusals are still named, and the accounting is exact: 23 methods
/// moved from the refused list to the served one.
#[tokio::test]
async fn what_is_not_ported_still_refuses_by_name() {
    let store = ObrainGraphStore::open_temporary().unwrap();
    let id = Uuid::new_v4();
    macro_rules! refuse {
        ($call:expr, $name:literal) => {{
            let e = $call.await.unwrap_err();
            assert_eq!(
                e.downcast_ref::<NonPorte>(),
                Some(&NonPorte($name)),
                "{e:#}"
            );
            assert!(refusees_analytics().contains(&$name));
        }};
    }
    refuse!(
        store.decay_synapses(&Default::default()),
        "AnalyticsStore::decay_synapses"
    );
    refuse!(
        store.decay_project_synapses(id, &Default::default()),
        "AnalyticsStore::decay_project_synapses"
    );
    refuse!(
        store.consolidate_memory(),
        "AnalyticsStore::consolidate_memory"
    );
    refuse!(
        store.get_project_synapse_edges(id),
        "AnalyticsStore::get_project_synapse_edges"
    );
    let servent = methodes_qui_servent();
    for m in [
        "create_synapses",
        "get_synapses",
        "delete_synapses",
        "get_all_synapse_weights",
        "get_project_note_synapses",
        "get_synapse_graph",
        "reinforce_synapses",
        "coactivate_session",
        "weaken_node_synapses",
        "create_cross_entity_synapses",
        "get_cross_entity_synapses",
        "list_notes_needing_synapses",
        "list_decisions_needing_synapses",
        "get_project_note_entity_links",
        "update_energy_scores",
        "boost_energy",
        "track_reactivation",
        "init_note_energy",
        "apply_scars",
        "heal_scars",
        "rate_limited_energy_boost",
        "get_deep_maintenance_times",
        "mark_deep_maintenance",
    ] {
        let nom = format!("AnalyticsStore::{m}");
        assert!(servent.contains(&nom.as_str()), "{nom} must serve");
        assert!(
            !refusees_analytics().contains(&nom.as_str()),
            "{nom} must not be refused"
        );
    }
    assert_eq!(refusees_analytics().len(), 96);
    assert_eq!(
        servent
            .iter()
            .filter(|m| m.starts_with("AnalyticsStore::"))
            .count(),
        49,
        "25 (A1) + 1 (`run_data_migrations`) + 23"
    );
}

/// `update_energy_scores` cut at every engine write: the stamp is written BEFORE the energy, so
/// a cut loses this decay and the retry starts from the new stamp; the same days are never
/// decayed twice (which would also let the 0.05 floor fix 0.0 for good).
#[tokio::test]
async fn an_energy_pass_cut_at_every_write_never_decays_the_same_days_twice() {
    let mut fin = None;
    for k in 1..=6 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let id;
        {
            let store = open(&base).await;
            let p = projet_actif(&store).await;
            id = une_note(&store, Some(p.id)).await.id;
            store.close().await.unwrap();
        }
        let vieux = (Utc::now() - Duration::days(90)).to_rfc3339();
        base_brute(&base, move |db| {
            let n = db.store().nodes_by_label("Note")[0];
            db.set_node_property(n, "energy_updated_at", Value::from(vieux));
        })
        .await;
        let store = open(&base).await;
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.update_energy_scores(90.0).await.is_ok() {
            fin = Some(k);
            break;
        }
        assert!(store.get_note(id).await.is_err(), "k={k}: indéterminé");
        drop(store);
        let store = open(&base).await;
        store.update_energy_scores(90.0).await.unwrap();
        let e = store.get_note(id).await.unwrap().unwrap().energy;
        assert!(
            e >= (-1.0_f64).exp() - 2e-3,
            "k={k}: 90 days decayed more than once: {e}"
        );
    }
    assert_eq!(fin, Some(3), "two engine writes");
}

/// A resurrection cut at every engine write: the archived `FORGOTTEN` edge is deleted AFTER the
/// live edges exist, so a cut never leaves neither; the retry finishes.
#[tokio::test]
async fn a_revival_cut_at_every_write_never_loses_the_archive() {
    let mut fin = None;
    for k in 1..=8 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let (a, b);
        {
            let store = open(&base).await;
            let p = projet_actif(&store).await;
            let n = des_notes(&store, Some(p.id), 2).await;
            (a, b) = (n[0].id, n[1].id);
            store.close().await.unwrap();
        }
        base_brute(&base, move |db| {
            let (na, nb) = (futures_noeud(db, a), futures_noeud(db, b));
            db.create_edge_with_props(na, nb, "FORGOTTEN", [("weight", Value::from(0.6_f64))]);
        })
        .await;
        let store = open(&base).await;
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.reinforce_synapses(&[a, b], 0.1).await.is_ok() {
            fin = Some(k);
            break;
        }
        drop(store);
        let (oubli, vivantes) = base_brute(&base, |db| {
            (
                aretes_brutes(db, "FORGOTTEN").len(),
                aretes_brutes(db, "SYNAPSE").len(),
            )
        })
        .await;
        assert!(
            oubli >= 1 || vivantes == 2,
            "k={k}: the archive is gone and the live edges are not there ({oubli}, {vivantes})"
        );
        let store = open(&base).await;
        store.reinforce_synapses(&[a, b], 0.1).await.unwrap();
        let s = store.get_synapses(a).await.unwrap();
        assert!(!s.is_empty(), "k={k}");
    }
    assert_eq!(fin, Some(4), "three engine writes");
}

fn futures_noeud(db: &obrain_engine::ObrainDB, id: Uuid) -> obrain_common::types::NodeId {
    db.store()
        .nodes_by_label("Note")
        .into_iter()
        .find(|n| {
            db.get_node(*n).and_then(|x| {
                x.get_property("id")
                    .and_then(|v| v.as_str().map(str::to_string))
            }) == Some(id.to_string())
        })
        .unwrap()
}

/// A `SYNAPSE` edge a migrated base left without `weight` (or with a `source` that is not a
/// string) is refused BEFORE the first write: the store is not made indeterminate, the next
/// call works, and nothing was rewritten.
#[tokio::test]
async fn an_unreadable_edge_refuses_a_reinforcement_without_breaking_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, a, b, c);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        let n = des_notes(&store, Some(p.id), 3).await;
        (a, b, c) = (n[0].id, n[1].id, n[2].id);
        store.create_synapses(a, &[(b, 0.5)]).await.unwrap();
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        let na = futures_noeud(db, a);
        let (_, e) = db.store().edges_from(na, Direction::Outgoing)[0];
        db.remove_edge_property(e, "weight");
    })
    .await;
    let store = open(&base).await;
    let prov = AccountableProvenance::new("r", "m").unwrap();
    for e in [
        store.reinforce_synapses(&[a, b, c], 0.1).await.unwrap_err(),
        store
            .coactivate_session(p.id, &[a, b], Uuid::new_v4(), &prov)
            .await
            .unwrap_err(),
    ] {
        assert!(format!("{e:#}").contains("weight"), "{e:#}");
        assert!(!format!("{e:#}").contains("indéterminé"), "{e:#}");
    }
    // Not indeterminate: another call still works, and the pair with `c` was not written.
    store.boost_energy(c, 0.1).await.unwrap();
    assert!(store.get_synapses(c).await.unwrap().is_empty());
    store.close().await.unwrap();

    // `source` that is not a string.
    base_brute(&base, move |db| {
        let na = futures_noeud(db, a);
        let (_, e) = db.store().edges_from(na, Direction::Outgoing)[0];
        db.set_edge_property(e, "weight", Value::from(0.5_f64));
        db.set_edge_property(e, "source", Value::from(5_i64));
    })
    .await;
    let store = open(&base).await;
    let e = store.reinforce_synapses(&[a, b], 0.1).await.unwrap_err();
    assert!(format!("{e:#}").contains("source"), "{e:#}");
    store.boost_energy(c, 0.1).await.unwrap();
}

/// The engine deletes a node's edges with it, so an edge whose neighbour has no labels cannot
/// be built by deleting the neighbour (the guard of `genre_de` against a dangling edge is
/// defensive). What can exist is a neighbour with NO label: it is not a `Note` nor a
/// `Decision`, and the `neighbor:Note` / `neighbor:Note OR neighbor:Decision` of the Cypher
/// leaves it out, which is what the reads do.
#[tokio::test]
async fn a_neighbour_that_is_not_a_note_is_left_out_as_the_cypher_does() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let a;
    {
        let store = open(&base).await;
        a = une_note(&store, None).await.id;
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        let na = futures_noeud(db, a);
        let autre = db.create_node_with_props(&["Other"], vec![("id", Value::from("x"))]);
        db.create_edge_with_props(na, autre, "SYNAPSE", [("weight", Value::from(0.5_f64))]);
        let mort = db.create_node_with_props(&["Note"], vec![("id", Value::from("y"))]);
        db.create_edge_with_props(na, mort, "SYNAPSE", [("weight", Value::from(0.5_f64))]);
        assert!(db.delete_node(mort));
        assert_eq!(
            db.store().edges_from(na, Direction::Outgoing).len(),
            1,
            "cascade"
        );
    })
    .await;
    let store = open(&base).await;
    assert!(store.get_synapses(a).await.unwrap().is_empty());
    assert!(store.get_cross_entity_synapses(a).await.unwrap().is_empty());
}

/// Neo4j drops a link whose entity id is empty: a target with an empty `path` is not a row.
#[tokio::test]
async fn an_entity_link_with_an_empty_path_is_dropped_as_neo4j_does() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, q);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        q = projet_actif(&store).await;
        let n = une_note(&store, Some(p.id)).await;
        store
            .link_note_to_entity(n.id, &EntityType::Project, &q.id.to_string(), None, None)
            .await
            .unwrap();
        store.close().await.unwrap();
    }
    let qid = q.id;
    base_brute(&base, move |db| {
        let x = id_brut_sync(db, "Project", qid);
        db.set_node_property(x, "path", Value::from(""));
    })
    .await;
    let store = open(&base).await;
    assert!(store
        .get_project_note_entity_links(p.id)
        .await
        .unwrap()
        .is_empty());
}

fn id_brut_sync(
    db: &obrain_engine::ObrainDB,
    etiquette: &str,
    id: Uuid,
) -> obrain_common::types::NodeId {
    db.store()
        .nodes_by_label(etiquette)
        .into_iter()
        .find(|n| {
            db.get_node(*n).and_then(|x| {
                x.get_property("id")
                    .and_then(|v| v.as_str().map(str::to_string))
            }) == Some(id.to_string())
        })
        .unwrap()
}
