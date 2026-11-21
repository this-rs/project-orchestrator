//! `AnalyticsStore` on obrain, the memory layer and its maintenance (`src/obrain/shadow.rs`,
//! `src/obrain/memoire.rs`, `ensure_project_canonical_paths` in `project.rs`,
//! `rescope_protocol` in `protocol.rs`): the shadow synapses and their traces, the forgetting
//! passes, the lifecycle sweep, the notes the heuristic wrote, the audit of the gaps.
//!
//! The behaviour shared with the reference stores (mock, Neo4j) is in
//! `tests/store_conformance.rs` (`memoire_checks!`). What is here is what only this backend does and
//! what the mock cannot be the reference for: the expected values are the ones the Cypher of
//! `src/neo4j/note.rs` and `src/neo4j/synapse_shadow.rs` computes, derived by hand and named in
//! each test. What the engine holds that the API does not show (an edge's properties) is read from
//! the closed base.
#![cfg(feature = "obrain-backend")]

use std::collections::HashMap;
use std::path::Path;

use chrono::{DateTime, Duration, TimeZone, Utc};
use obrain_common::types::{Timestamp, Value, ZonedDatetime};
use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use project_orchestrator::neo4j::models::*;
use project_orchestrator::neo4j::traits::*;
use project_orchestrator::neurons::forgetting::DecayPolicy;
use project_orchestrator::neurons::synaptic_shadow::{
    ShadowDecayScope, ShadowSeed, ShadowSelection, ShadowTraceFilter, StopReason,
    SynapseShadowTrace,
};
use project_orchestrator::notes::{EntityType, MemoryHorizon, Note, NoteStatus, NoteType};
use project_orchestrator::obrain::{
    Creation, Interruption, ObrainGraphStore, MARQUEUR_PO, SCHEMA_PO,
};
use project_orchestrator::protocol::Protocol;
use uuid::Uuid;

// ============================================================================
// Harness
// ============================================================================

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
    f: impl FnOnce(&ObrainDB) -> R + Send + 'static,
) -> R {
    let base = base.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let db = ObrainDB::open(&base).unwrap();
        let r = f(&db);
        db.commit_wal().unwrap();
        db.close().unwrap();
        std::fs::write(base.join(MARQUEUR_PO), SCHEMA_PO).unwrap();
        r
    })
    .await
    .unwrap()
}

fn noeud(db: &ObrainDB, label: &str, id: Uuid) -> obrain_common::types::NodeId {
    db.store()
        .nodes_by_label(label)
        .into_iter()
        .find(|n| {
            db.get_node(*n).and_then(|x| {
                x.get_property("id")
                    .and_then(|v| v.as_str().map(str::to_string))
            }) == Some(id.to_string())
        })
        .unwrap_or_else(|| panic!("pas de nœud {label} {id}"))
}

/// Every edge of `type_` from the note `de` to the note `vers`: its properties.
fn aretes(db: &ObrainDB, de: Uuid, vers: Uuid, type_: &str) -> Vec<HashMap<String, Value>> {
    let (nd, nv) = (noeud(db, "Note", de), noeud(db, "Note", vers));
    let mut v = Vec::new();
    for (dst, e) in db.store().edges_from(nd, Direction::Outgoing) {
        let ed = db.get_edge(e).unwrap();
        if dst == nv && &*ed.edge_type == type_ {
            v.push(
                ed.properties
                    .iter()
                    .map(|(k, x)| (k.to_string(), x.clone()))
                    .collect(),
            );
        }
    }
    v
}

/// Set properties on the edges of `type_` from `de` to `vers`.
fn regler(db: &ObrainDB, de: Uuid, vers: Uuid, type_: &str, props: Vec<(&str, Value)>) {
    let (nd, nv) = (noeud(db, "Note", de), noeud(db, "Note", vers));
    for (dst, e) in db.store().edges_from(nd, Direction::Outgoing) {
        let ed = db.get_edge(e).unwrap();
        if dst == nv && &*ed.edge_type == type_ {
            for (k, v) in &props {
                db.set_edge_property(e, k, v.clone());
            }
        }
    }
}

async fn arete_brute(
    base: &Path,
    de: Uuid,
    vers: Uuid,
    type_: &'static str,
) -> Vec<HashMap<String, Value>> {
    base_brute(base, move |db| aretes(db, de, vers, type_)).await
}

fn nombre(p: &HashMap<String, Value>, cle: &str) -> f64 {
    p[cle]
        .as_float64()
        .unwrap_or_else(|| panic!("{cle} : {:?}", p[cle]))
}

fn proche(a: f64, b: f64) -> bool {
    (a - b).abs() < 2e-4
}

fn instant(d: DateTime<Utc>) -> Value {
    Value::from(d.to_rfc3339())
}

fn zoned(d: DateTime<Utc>) -> Value {
    Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
        Timestamp::from_micros(d.timestamp_micros()),
        0,
    ))
}

fn a_project() -> ProjectNode {
    let id = Uuid::new_v4();
    ProjectNode {
        id,
        name: format!("memoire {id}"),
        slug: format!("memoire-{id}"),
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

/// An ACTIVE project: its code was synced now, so it is not dormant.
async fn projet_actif(store: &ObrainGraphStore) -> ProjectNode {
    let p = a_project();
    store.create_project(&p).await.unwrap();
    store.update_project_synced(p.id).await.unwrap();
    p
}

/// A DORMANT project: no sync, no chat.
async fn projet_dormant(store: &ObrainGraphStore) -> ProjectNode {
    let p = a_project();
    store.create_project(&p).await.unwrap();
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

fn politique(demi_vie: f64, similarite: f64, min_degre: usize, fraction: f64) -> DecayPolicy {
    DecayPolicy {
        half_life_days: demi_vie,
        similarity_half_life_days: similarite,
        min_degree: min_degre,
        max_archive_fraction: fraction,
        ..Default::default()
    }
}

// ============================================================================
// Shadow synapses
// ============================================================================

/// `shadow_synapse_edges`: `coalesce(shadow_weight, weight)`, both ends in the project, no
/// self-loop, the target's energy (`1.0` when none), scar, reachability (`active` or
/// `needs_review`), tokens (`ceil(chars / 4)`) and creation date; an unknown id is skipped and a
/// repeated one repeated (`UNWIND`).
#[tokio::test]
async fn shadow_edges_read_the_shadow_weight_and_the_metadata_of_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, c, d);
    let (p, q);
    let b;
    {
        let store = open(&base).await;
        (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
        b = Note::new(
            Some(p.id),
            NoteType::Observation,
            "123456789".into(),
            "t".into(),
        );
        a = une_note(&store, Some(p.id)).await;
        store.create_note(&b).await.unwrap();
        c = une_note(&store, Some(p.id)).await;
        d = une_note(&store, Some(q.id)).await;
        store
            .create_synapses(a.id, &[(b.id, 0.6), (c.id, 0.4), (d.id, 0.9)])
            .await
            .unwrap();
        store.close().await.unwrap();
    }
    let (ai, bi, ci) = (a.id, b.id, c.id);
    base_brute(&base, move |db| {
        regler(
            db,
            ai,
            bi,
            "SYNAPSE",
            vec![("shadow_weight", Value::from(0.9))],
        );
        let (nb, nc, na) = (
            noeud(db, "Note", bi),
            noeud(db, "Note", ci),
            noeud(db, "Note", ai),
        );
        db.set_node_property(nb, "energy", Value::from(0.25));
        db.set_node_property(nb, "scar_intensity", Value::from(0.5));
        db.set_node_property(nc, "status", Value::from("archived"));
        // A self-loop: never listed.
        db.create_edge_with_props(na, na, "SYNAPSE", vec![("weight", Value::from(0.7))]);
    })
    .await;
    let store = open(&base).await;
    let rows = store
        .shadow_synapse_edges(p.id, &[a.id, Uuid::new_v4()])
        .await
        .unwrap();
    assert_eq!(rows.len(), 2, "{rows:?}");
    let ligne = |to: Uuid| rows.iter().find(|r| r.to == to).unwrap();
    assert!(rows.iter().all(|r| r.from == a.id));
    let rb = ligne(b.id);
    assert!(proche(rb.weight, 0.9), "shadow_weight wins over weight");
    assert!(proche(rb.target.energy, 0.25) && proche(rb.target.scar_intensity, 0.5));
    assert!(rb.target.reachable);
    assert_eq!(rb.target.tokens, 3, "ceil(9 / 4)");
    assert_eq!(rb.target.created_at, Some(b.created_at));
    let rc = ligne(c.id);
    assert!(
        proche(rc.weight, 0.4),
        "no shadow weight: the legacy weight"
    );
    assert!(
        proche(rc.target.energy, 1.0) && !rc.target.reachable,
        "archived"
    );
    assert_eq!(rc.target.tokens, 1, "at least one token");
    // `UNWIND`: a repeated id answers twice; no input, no answer; the other project's note and
    // the other project's source are out.
    assert_eq!(
        store
            .shadow_synapse_edges(p.id, &[a.id, a.id])
            .await
            .unwrap()
            .len(),
        4
    );
    assert!(store
        .shadow_synapse_edges(p.id, &[])
        .await
        .unwrap()
        .is_empty());
    assert!(store
        .shadow_synapse_edges(q.id, &[a.id])
        .await
        .unwrap()
        .is_empty());
    assert!(store
        .shadow_synapse_edges(p.id, &[d.id])
        .await
        .unwrap()
        .is_empty());
}

/// `reward_shadow_synapses`: `delta` on the shadow weight (legacy weight to start from) of the
/// existing edges of each unordered pair, both directions, clamped to `[0, 1]`, never writing
/// `weight`; the count is the directed edges written; `shadow_reward_count` counts the rewards and
/// `shadow_decayed_at` is set once.
#[tokio::test]
async fn shadow_rewards_move_the_shadow_weight_only_and_clamp_it() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let (a, b, c, d) = (
        une_note(&store, Some(p.id)).await.id,
        une_note(&store, Some(p.id)).await.id,
        une_note(&store, Some(p.id)).await.id,
        une_note(&store, Some(q.id)).await.id,
    );
    store
        .create_synapses(a, &[(b, 0.6), (d, 0.5)])
        .await
        .unwrap();
    store.create_synapses(c, &[(a, 0.3)]).await.unwrap();
    let t0 = Utc::now() - Duration::hours(2);
    // (a, b): two directed edges. (b, a) is the same unordered pair, a repeated pair counts once.
    let n = store
        .reward_shadow_synapses(p.id, &[(a, b), (b, a), (a, a)], 0.5, t0)
        .await
        .unwrap();
    assert_eq!(n, 2);
    // (a, d) crosses projects: out. (b, c) has no edge: nothing written, nothing created.
    let n = store
        .reward_shadow_synapses(p.id, &[(a, d), (b, c)], 0.5, t0)
        .await
        .unwrap();
    assert_eq!(n, 0);
    // A second reward moves from the SHADOW weight, a negative one clamps at zero.
    let t1 = Utc::now();
    assert_eq!(
        store
            .reward_shadow_synapses(p.id, &[(a, c)], -0.9, t1)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        store
            .reward_shadow_synapses(p.id, &[(a, b)], -0.25, t1)
            .await
            .unwrap(),
        2
    );
    assert!(store
        .reward_shadow_synapses(p.id, &[(a, b)], f64::NAN, t1)
        .await
        .is_err());
    store.close().await.unwrap();
    let ab = arete_brute(&base, a, b, "SYNAPSE").await;
    assert_eq!(ab.len(), 1);
    // 0.6 + 0.5 clamped to 1.0, then - 0.25.
    assert!(proche(nombre(&ab[0], "shadow_weight"), 0.75));
    assert!(
        proche(nombre(&ab[0], "weight"), 0.6),
        "`weight` is never written"
    );
    assert_eq!(ab[0]["shadow_reward_count"].as_int64(), Some(2));
    // The clock of the decay is set by the FIRST reward and kept.
    let horloge = ab[0]["shadow_decayed_at"].as_str().unwrap().to_string();
    assert_eq!(
        horloge.parse::<DateTime<Utc>>().unwrap().timestamp(),
        t0.timestamp()
    );
    let ac = arete_brute(&base, a, c, "SYNAPSE").await;
    // `create_synapses` wrote both directions of `c — a`: one edge each way.
    assert_eq!(ac.len(), 1);
    assert_eq!(nombre(&ac[0], "shadow_weight"), 0.0, "clamped at zero");
    // The cross-project edge was never touched.
    assert!(!arete_brute(&base, a, d, "SYNAPSE").await[0].contains_key("shadow_weight"));
}

/// `decay_shadow_synapses`: `w · 2^(-Δt / half_life)` on the shadow weight (legacy weight to
/// start from), `Δt` from `shadow_decayed_at` in whole seconds (none: no decay, the clock is set);
/// a dormant project is frozen (clock advanced, weight kept, but materialised); a disabled project
/// is out; a clock in the future changes nothing. The three stored date shapes are read.
#[tokio::test]
async fn shadow_decay_follows_the_clock_the_dormancy_and_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (act, dor, off);
    let (a1, a2, d1, d2, e1, e2, g1, g2, h1, h2);
    {
        let store = open(&base).await;
        act = projet_actif(&store).await;
        dor = projet_dormant(&store).await;
        off = projet_actif(&store).await;
        let n = des_notes(&store, Some(act.id), 6).await;
        (a1, a2, g1, g2, h1, h2) = (n[0].id, n[1].id, n[2].id, n[3].id, n[4].id, n[5].id);
        let n = des_notes(&store, Some(dor.id), 2).await;
        (d1, d2) = (n[0].id, n[1].id);
        let n = des_notes(&store, Some(off.id), 2).await;
        (e1, e2) = (n[0].id, n[1].id);
        for (x, y) in [(a1, a2), (d1, d2), (e1, e2), (g1, g2), (h1, h2)] {
            store.create_synapses(x, &[(y, 0.8)]).await.unwrap();
        }
        store.close().await.unwrap();
    }
    let onze = Utc::now() - Duration::days(11);
    let futur = Utc::now() + Duration::hours(1);
    let futur_s = futur.to_rfc3339();
    let fs = futur_s.clone();
    base_brute(&base, move |db| {
        // ISO with a shadow weight, and ZonedDatetime on the legacy weight.
        regler(
            db,
            a1,
            a2,
            "SYNAPSE",
            vec![
                ("shadow_weight", Value::from(1.0)),
                ("shadow_decayed_at", instant(onze)),
            ],
        );
        regler(
            db,
            a2,
            a1,
            "SYNAPSE",
            vec![("shadow_decayed_at", zoned(onze))],
        );
        regler(
            db,
            h1,
            h2,
            "SYNAPSE",
            vec![(
                "shadow_decayed_at",
                Value::Timestamp(Timestamp::from_micros(onze.timestamp_micros())),
            )],
        );
        for (x, y) in [(d1, d2), (d2, d1)] {
            regler(
                db,
                x,
                y,
                "SYNAPSE",
                vec![
                    ("shadow_weight", Value::from(1.0)),
                    ("shadow_decayed_at", instant(onze)),
                ],
            );
        }
        for (x, y) in [(e1, e2), (e2, e1)] {
            regler(
                db,
                x,
                y,
                "SYNAPSE",
                vec![
                    ("shadow_weight", Value::from(1.0)),
                    ("shadow_decayed_at", instant(onze)),
                ],
            );
        }
        regler(
            db,
            g1,
            g2,
            "SYNAPSE",
            vec![
                ("shadow_weight", Value::from(1.0)),
                ("shadow_decayed_at", Value::from(fs)),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    let now = Utc::now();
    let portee = ShadowDecayScope {
        default_enabled: true,
        enabled: vec![],
        disabled: vec![off.id],
    };
    // a1<->a2 (2) + d1<->d2 (2) + g1<->g2 (2) + h1<->h2 (2); the disabled project is out.
    assert_eq!(
        store
            .decay_shadow_synapses(&portee, 11.0, now)
            .await
            .unwrap(),
        8
    );
    store.close().await.unwrap();
    let lire = |x, y| arete_brute(&base, x, y, "SYNAPSE");
    // 11 days idle, half-life 11: half. The zoned and the timestamp clocks are read like the ISO one.
    assert!(proche(nombre(&lire(a1, a2).await[0], "shadow_weight"), 0.5));
    assert!(
        proche(nombre(&lire(a2, a1).await[0], "shadow_weight"), 0.4),
        "0.8 legacy, halved"
    );
    assert!(proche(nombre(&lire(h1, h2).await[0], "shadow_weight"), 0.4));
    assert!(
        proche(nombre(&lire(a1, a2).await[0], "weight"), 0.8),
        "`weight` untouched"
    );
    // The dormant project: weight kept (materialised), clock advanced to `now`.
    let dormant = &lire(d1, d2).await[0];
    assert!(proche(nombre(dormant, "shadow_weight"), 1.0));
    assert_eq!(
        dormant["shadow_decayed_at"]
            .as_str()
            .unwrap()
            .parse::<DateTime<Utc>>()
            .unwrap()
            .timestamp(),
        now.timestamp()
    );
    // The disabled project: nothing written.
    assert!(lire(e1, e2).await[0]
        .get("shadow_weight")
        .is_some_and(|v| v.as_float64() == Some(1.0)));
    assert_eq!(
        lire(e1, e2).await[0]["shadow_decayed_at"]
            .as_str()
            .unwrap()
            .parse::<DateTime<Utc>>()
            .unwrap()
            .timestamp(),
        onze.timestamp()
    );
    // A clock in the future: weight kept, clock kept.
    let futur_edge = &lire(g1, g2).await[0];
    assert!(proche(nombre(futur_edge, "shadow_weight"), 1.0));
    assert_eq!(futur_edge["shadow_decayed_at"].as_str().unwrap(), futur_s);
    // No clock: no decay, the clock is set.
    let sans = &lire(g2, g1).await[0];
    assert!(proche(nombre(sans, "shadow_weight"), 0.8));
    assert!(sans.get("shadow_decayed_at").is_some());
    // Idempotent at the same instant: the clock is `now`, no time has passed.
    let store = open(&base).await;
    assert_eq!(
        store
            .decay_shadow_synapses(&portee, 11.0, now)
            .await
            .unwrap(),
        8
    );
    store.close().await.unwrap();
    assert!(proche(nombre(&lire(a1, a2).await[0], "shadow_weight"), 0.5));
}

/// Nothing is decayed for a scope that enables nothing; a scope that enables one project walks
/// only it; a half-life that is not strictly positive refuses.
#[tokio::test]
async fn shadow_decay_scope_and_half_life_are_checked() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    for pid in [p.id, q.id] {
        let n = des_notes(&store, Some(pid), 2).await;
        store
            .create_synapses(n[0].id, &[(n[1].id, 0.8)])
            .await
            .unwrap();
    }
    let now = Utc::now();
    let rien = ShadowDecayScope::default();
    assert_eq!(
        store.decay_shadow_synapses(&rien, 11.0, now).await.unwrap(),
        0
    );
    let seulp = ShadowDecayScope {
        default_enabled: false,
        enabled: vec![p.id],
        disabled: vec![],
    };
    assert_eq!(
        store
            .decay_shadow_synapses(&seulp, 11.0, now)
            .await
            .unwrap(),
        2
    );
    let tout_sauf_q = ShadowDecayScope {
        default_enabled: true,
        enabled: vec![],
        disabled: vec![q.id],
    };
    assert_eq!(
        store
            .decay_shadow_synapses(&tout_sauf_q, 11.0, now)
            .await
            .unwrap(),
        2
    );
    for mauvaise in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(store
            .decay_shadow_synapses(&seulp, mauvaise, now)
            .await
            .is_err());
    }
}

/// REGRESSION. A shadow clock that is present but unreadable refuses the whole pass and writes
/// nothing: it is never read as "no clock" (Neo4j's `parse_at` turned it into `None`, so the edge
/// was stamped now and its elapsed time lost).
#[tokio::test]
async fn an_unreadable_shadow_clock_refuses_the_pass_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, b, c, d);
    {
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        let n = des_notes(&store, Some(p.id), 4).await;
        (a, b, c, d) = (n[0].id, n[1].id, n[2].id, n[3].id);
        store.create_synapses(a, &[(b, 0.8)]).await.unwrap();
        store.create_synapses(c, &[(d, 0.8)]).await.unwrap();
        store.close().await.unwrap();
    }
    let onze = Utc::now() - Duration::days(11);
    base_brute(&base, move |db| {
        regler(
            db,
            a,
            b,
            "SYNAPSE",
            vec![("shadow_decayed_at", Value::from("pas une date"))],
        );
        regler(
            db,
            c,
            d,
            "SYNAPSE",
            vec![("shadow_decayed_at", instant(onze))],
        );
    })
    .await;
    let store = open(&base).await;
    let portee = ShadowDecayScope {
        default_enabled: true,
        ..Default::default()
    };
    let e = store
        .decay_shadow_synapses(&portee, 11.0, Utc::now())
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("illisible"), "{e:#}");
    store.close().await.unwrap();
    // Nothing written, not even on the readable edge.
    assert!(!arete_brute(&base, c, d, "SYNAPSE").await[0].contains_key("shadow_weight"));
}

fn une_trace(session: Uuid, projet: Uuid, secs: i64, notes: &[Uuid]) -> SynapseShadowTrace {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    SynapseShadowTrace {
        id: Uuid::new_v4(),
        schema_version: 1,
        layer_version: "t8-shadow-v1".into(),
        session_id: session,
        project_id: projet,
        requested_mode: Default::default(),
        effective_mode: Default::default(),
        session_started_at: t0,
        created_at: t0 + Duration::seconds(secs),
        seeds: vec![ShadowSeed {
            note_id: notes[0],
            affinity: 0.5,
        }],
        selected: notes
            .iter()
            .map(|n| ShadowSelection {
                note_id: *n,
                score: 0.7,
                hops: 1,
                via: None,
                tokens: 10,
                explored: false,
                created_at: None,
            })
            .collect(),
        tokens_used: 10 * notes.len(),
        token_budget: 1200,
        stopped_by: StopReason::Exhausted,
        weights_fingerprint: "w".into(),
        config_fingerprint: "c".into(),
        edges_read: 3,
        rewarded: vec![],
    }
}

/// The traces: recorded and read back whole, the latest `keep` of a session kept (ordered by
/// `(created_at, id)`), listed by session oldest first and by note latest first with a limit, a
/// reward claimed once, and all of it after a reopen.
#[tokio::test]
async fn traces_are_recorded_trimmed_listed_and_claimed_once() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (s1, s2, pid) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let (n1, n2) = (Uuid::new_v4(), Uuid::new_v4());
    let t: Vec<SynapseShadowTrace> = vec![
        une_trace(s1, pid, 10, &[n1]),
        une_trace(s1, pid, 20, &[n1, n2]),
        une_trace(s1, pid, 30, &[n2]),
        une_trace(s1, pid, 40, &[n1]),
        une_trace(s2, pid, 5, &[n1]),
    ];
    {
        let store = open(&base).await;
        for x in &t {
            store.record_synapse_shadow_trace(x, 3).await.unwrap();
        }
        // The oldest of the session s1 is gone, the other session is whole.
        assert!(store
            .get_synapse_shadow_trace(t[0].id)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            store
                .get_synapse_shadow_trace(t[1].id)
                .await
                .unwrap()
                .as_ref(),
            Some(&t[1])
        );
        assert_eq!(
            store
                .list_synapse_shadow_traces(&ShadowTraceFilter::Session(s1))
                .await
                .unwrap()
                .iter()
                .map(|x| x.id)
                .collect::<Vec<_>>(),
            vec![t[1].id, t[2].id, t[3].id],
            "oldest first"
        );
        // Latest first, at most `limit`, across sessions.
        let tous = store
            .list_synapse_shadow_traces(&ShadowTraceFilter::ContainingNote {
                note_id: n1,
                limit: 2,
            })
            .await
            .unwrap();
        assert_eq!(
            tous.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![t[3].id, t[1].id]
        );
        // `keep` 0 deletes the new trace itself.
        let z = une_trace(Uuid::new_v4(), pid, 1, &[n1]);
        store.record_synapse_shadow_trace(&z, 0).await.unwrap();
        assert!(store
            .get_synapse_shadow_trace(z.id)
            .await
            .unwrap()
            .is_none());
        // A trace id already held refuses (Neo4j's CREATE would write a second node).
        assert!(store.record_synapse_shadow_trace(&t[2], 3).await.is_err());
        // The claim: once, then never; an unknown trace is `false`.
        assert!(store
            .claim_shadow_trace_reward(t[2].id, "task_closure")
            .await
            .unwrap());
        assert!(!store
            .claim_shadow_trace_reward(t[2].id, "task_closure")
            .await
            .unwrap());
        assert!(store
            .claim_shadow_trace_reward(t[2].id, "token_budget")
            .await
            .unwrap());
        assert!(!store
            .claim_shadow_trace_reward(Uuid::new_v4(), "x")
            .await
            .unwrap());
        store.close().await.unwrap();
    }
    let store = open(&base).await;
    let relue = store
        .get_synapse_shadow_trace(t[2].id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        relue.rewarded,
        vec!["task_closure".to_string(), "token_budget".to_string()]
    );
    assert!(!store
        .claim_shadow_trace_reward(t[2].id, "task_closure")
        .await
        .unwrap());
    assert_eq!(
        store
            .list_synapse_shadow_traces(&ShadowTraceFilter::Session(s1))
            .await
            .unwrap()
            .len(),
        3
    );
}

/// REGRESSION. A `SynapseShadowTrace` node that does not read is quarantined at open: every list
/// refuses (it would miss it), so does a record (it could miss a trace of the session it trims),
/// and the id it names is neither read nor claimed. Another trace's id is still served.
#[tokio::test]
async fn a_quarantined_trace_makes_the_lists_refuse_and_its_id_untouchable() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let casse = Uuid::new_v4();
    {
        let store = open(&base).await;
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        // No payload: unreadable.
        db.create_node_with_props(
            &["SynapseShadowTrace"],
            vec![("id", Value::from(casse.to_string()))],
        );
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
    let (s, pid) = (Uuid::new_v4(), Uuid::new_v4());
    for e in [
        store
            .list_synapse_shadow_traces(&ShadowTraceFilter::Session(s))
            .await
            .unwrap_err(),
        store
            .record_synapse_shadow_trace(&une_trace(s, pid, 1, &[Uuid::new_v4()]), 5)
            .await
            .unwrap_err(),
        store.get_synapse_shadow_trace(casse).await.unwrap_err(),
        store
            .claim_shadow_trace_reward(casse, "k")
            .await
            .unwrap_err(),
    ] {
        assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
    }
    assert!(store
        .get_synapse_shadow_trace(Uuid::new_v4())
        .await
        .unwrap()
        .is_none());
}

/// A record cut at every engine write leaves a base that opens clean: the trace is absent (a
/// draft is deleted at open) or whole, never half, and the base keeps working.
#[tokio::test]
async fn a_trace_record_cut_at_every_write_leaves_it_absent_or_whole() {
    for k in 1..=20 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let (s, pid, n) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let ancienne = une_trace(s, pid, 1, &[n]);
        let nouvelle = une_trace(s, pid, 2, &[n]);
        let store = open(&base).await;
        store
            .record_synapse_shadow_trace(&ancienne, 1)
            .await
            .unwrap();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store
            .record_synapse_shadow_trace(&nouvelle, 1)
            .await
            .is_ok()
        {
            assert!(k > 10, "la coupure n'a atteint que {} points", k - 1);
            return;
        }
        assert!(
            store.get_synapse_shadow_trace(ancienne.id).await.is_err(),
            "k={k}: indéterminé"
        );
        drop(store);
        let store = open(&base).await;
        assert!(
            store.rapport_d_ouverture().en_quarantaine.is_empty(),
            "k={k}"
        );
        let vues = store
            .list_synapse_shadow_traces(&ShadowTraceFilter::Session(s))
            .await
            .unwrap();
        assert!(!vues.is_empty() && vues.len() <= 2, "k={k}: {}", vues.len());
        for v in &vues {
            assert!(
                *v == ancienne || *v == nouvelle,
                "k={k}: une trace à moitié écrite"
            );
        }
        // The next record trims what the cut left.
        let suivante = une_trace(s, pid, 3, &[n]);
        store
            .record_synapse_shadow_trace(&suivante, 1)
            .await
            .unwrap();
        assert_eq!(
            store
                .list_synapse_shadow_traces(&ShadowTraceFilter::Session(s))
                .await
                .unwrap()
                .len(),
            1,
            "k={k}"
        );
    }
    panic!("the cut never ran out of writes: raise the bound");
}

// ============================================================================
// The forgetting passes
// ============================================================================

/// Step 1 of `forgetting_pass`: `w · 2^(-Δt / h)` from the edge's own `decayed_at` (none: stamped,
/// not decayed), `h` by provenance (`coactivation` the half-life, any other the similarity one),
/// the protection floor for an edge reinforced `protected_after` times; the report counts the
/// edges walked, the dormant ones (`< dormancy_threshold`) and the protected ones.
#[tokio::test]
async fn the_forgetting_pass_decays_by_elapsed_time_provenance_and_protection() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let n;
    {
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        n = des_notes(&store, Some(p.id), 6)
            .await
            .into_iter()
            .map(|x| x.id)
            .collect::<Vec<_>>();
        for (x, y, w) in [(0, 1, 0.8), (2, 3, 0.6), (4, 5, 0.5), (0, 2, 0.1)] {
            store.create_synapses(n[x], &[(n[y], w)]).await.unwrap();
        }
        store.close().await.unwrap();
    }
    let il_y_a = |j: i64| instant(Utc::now() - Duration::days(j));
    let n2 = n.clone();
    base_brute(&base, move |db| {
        let n = &n2;
        for (x, y) in [(n[0], n[1]), (n[1], n[0])] {
            regler(
                db,
                x,
                y,
                "SYNAPSE",
                vec![
                    ("source", Value::from("coactivation")),
                    ("decayed_at", il_y_a(10)),
                ],
            );
        }
        for (x, y) in [(n[2], n[3]), (n[3], n[2])] {
            regler(db, x, y, "SYNAPSE", vec![("decayed_at", il_y_a(5))]);
        }
        for (x, y) in [(n[0], n[2]), (n[2], n[0])] {
            regler(
                db,
                x,
                y,
                "SYNAPSE",
                vec![
                    ("source", Value::from("coactivation")),
                    ("reinforcement_count", Value::from(3_i64)),
                    ("decayed_at", il_y_a(100)),
                ],
            );
        }
    })
    .await;
    let store = open(&base).await;
    let r = store
        .decay_synapses(&politique(10.0, 5.0, 3, 0.05))
        .await
        .unwrap();
    assert_eq!(
        (r.decayed, r.dormant, r.protected, r.archived, r.capped),
        (8, 0, 2, 0, false),
        "{r:?}"
    );
    store.close().await.unwrap();
    let w = |x: usize, y: usize| {
        let (x, y) = (n[x], n[y]);
        let base = base.clone();
        async move { arete_brute(&base, x, y, "SYNAPSE").await.remove(0) }
    };
    // 10 days at half-life 10: half. 5 days at the similarity half-life 5: half.
    assert!(proche(nombre(&w(0, 1).await, "weight"), 0.4));
    assert!(proche(nombre(&w(3, 2).await, "weight"), 0.3));
    // Never decayed by a pass: stamped, weight kept.
    let sans = w(4, 5).await;
    assert!(proche(nombre(&sans, "weight"), 0.5));
    assert!(sans.contains_key("decayed_at"));
    // 100 days: ~0, lifted to the protection floor by the three reinforcements.
    assert!(proche(nombre(&w(0, 2).await, "weight"), 0.15));
    // A second pass right after changes (almost) nothing.
    let store = open(&base).await;
    let r2 = store
        .decay_synapses(&politique(10.0, 5.0, 3, 0.05))
        .await
        .unwrap();
    assert_eq!(r2.decayed, 8);
    store.close().await.unwrap();
    assert!(proche(nombre(&w(0, 1).await, "weight"), 0.4));
}

/// A ring of six notes (`i → i+1` and back), every edge `0.01`, created 30 days ago, never
/// reinforced: forgettable. The three weakest directed edges are named, the rest equal.
async fn un_anneau(base: &Path) -> Vec<Uuid> {
    let n: Vec<Uuid>;
    {
        let store = open(base).await;
        let p = projet_actif(&store).await;
        n = des_notes(&store, Some(p.id), 6)
            .await
            .into_iter()
            .map(|x| x.id)
            .collect();
        for i in 0..6 {
            store
                .create_synapses(n[i], &[(n[(i + 1) % 6], 0.01)])
                .await
                .unwrap();
        }
        store.close().await.unwrap();
    }
    let n2 = n.clone();
    base_brute(base, move |db| {
        let vieux = instant(Utc::now() - Duration::days(30));
        for i in 0..6 {
            for (x, y) in [(n2[i], n2[(i + 1) % 6]), (n2[(i + 1) % 6], n2[i])] {
                regler(
                    db,
                    x,
                    y,
                    "SYNAPSE",
                    vec![("created_at", vieux.clone()), ("weight", Value::from(0.01))],
                );
            }
        }
        regler(
            db,
            n2[0],
            n2[1],
            "SYNAPSE",
            vec![("weight", Value::from(0.001))],
        );
        regler(
            db,
            n2[1],
            n2[2],
            "SYNAPSE",
            vec![("weight", Value::from(0.002))],
        );
        regler(
            db,
            n2[2],
            n2[3],
            "SYNAPSE",
            vec![("weight", Value::from(0.003))],
        );
    })
    .await;
    n
}

/// Step 2. REGRESSION on the ORDER. Neo4j's `LIMIT $cap` has no `ORDER BY`; this backend archives
/// the WEAKEST first (then by ids), which the module names. The cap is `ceil(walked × fraction)`,
/// an archived edge keeps all its properties on the `FORGOTTEN` edge with `forgotten_at` and
/// `forgotten_reason = 'decay'`, and the `SYNAPSE` edge is gone.
#[tokio::test]
async fn the_archive_takes_the_weakest_edges_first_under_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let n = un_anneau(&base).await;
    let store = open(&base).await;
    // 12 directed edges, fraction 0.25: a cap of 3. min_degree 1: each node out-degree 2.
    let r = store
        .decay_synapses(&politique(10.0, 5.0, 1, 0.25))
        .await
        .unwrap();
    assert_eq!((r.decayed, r.archived, r.capped), (12, 3, true), "{r:?}");
    store.close().await.unwrap();
    for (x, y, w) in [(0, 1, 0.001), (1, 2, 0.002), (2, 3, 0.003)] {
        assert!(
            arete_brute(&base, n[x], n[y], "SYNAPSE").await.is_empty(),
            "{x}->{y} archived"
        );
        let f = arete_brute(&base, n[x], n[y], "FORGOTTEN").await;
        assert_eq!(f.len(), 1);
        assert!(proche(nombre(&f[0], "weight"), w), "the weight it had");
        assert_eq!(f[0]["forgotten_reason"].as_str(), Some("decay"));
        assert!(f[0].contains_key("forgotten_at"));
        assert_eq!(f[0]["source"].as_str(), Some("cosine"), "provenance kept");
        assert!(f[0].contains_key("created_at"));
    }
    // The others stay.
    for (x, y) in [(1, 0), (3, 2), (3, 4), (5, 0)] {
        assert_eq!(
            arete_brute(&base, n[x], n[y], "SYNAPSE").await.len(),
            1,
            "{x}->{y}"
        );
        assert!(arete_brute(&base, n[x], n[y], "FORGOTTEN").await.is_empty());
    }
}

/// Step 2. REGRESSION on the FLOOR. A complete graph of four notes (each out-degree 3), every edge
/// forgettable, `min_degree` 2, a cap that would take them all: with the degrees read BEFORE the
/// pass (Neo4j's evaluation) all twelve would go and every note would end at degree 0. The live
/// floor lets each note lose at most ONE (3 → 2): every note keeps two.
#[tokio::test]
async fn the_archive_never_takes_a_note_below_the_minimum_degree() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let n: Vec<Uuid>;
    {
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        n = des_notes(&store, Some(p.id), 4)
            .await
            .into_iter()
            .map(|x| x.id)
            .collect();
        for i in 0..4 {
            let autres: Vec<(Uuid, f64)> =
                (0..4).filter(|j| *j != i).map(|j| (n[j], 0.01)).collect();
            store.create_synapses(n[i], &autres).await.unwrap();
        }
        store.close().await.unwrap();
    }
    let n2 = n.clone();
    base_brute(&base, move |db| {
        let vieux = instant(Utc::now() - Duration::days(30));
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    regler(
                        db,
                        n2[i],
                        n2[j],
                        "SYNAPSE",
                        vec![("created_at", vieux.clone())],
                    );
                }
            }
        }
    })
    .await;
    let store = open(&base).await;
    let r = store
        .decay_synapses(&politique(10.0, 5.0, 2, 1.0))
        .await
        .unwrap();
    assert_eq!(r.decayed, 12);
    assert!((1..=4).contains(&r.archived), "at most one per note: {r:?}");
    store.close().await.unwrap();
    let mut total = 0;
    for i in 0..4 {
        let mut restant = 0;
        for j in 0..4 {
            if i != j {
                restant += arete_brute(&base, n[i], n[j], "SYNAPSE").await.len();
            }
        }
        assert!(
            restant >= 2,
            "the note {i} keeps `min_degree` edges ({restant})"
        );
        total += restant;
    }
    assert_eq!(total, 12 - r.archived);
}

/// An edge is archived only when ALL the conditions hold: below the threshold, never proven by use,
/// untouched for `forget_after_days` (`coalesce(last_reinforced_at, created_at)`; no date at all
/// leaves it alone), and the pair's own `FORGOTTEN` edge is replaced, not doubled.
#[tokio::test]
async fn the_archive_needs_every_condition_and_replaces_a_forgotten_edge_of_the_pair() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let n = un_anneau(&base).await;
    let n2 = n.clone();
    base_brute(&base, move |db| {
        let n = &n2;
        // 0 → 1: reinforced three times (protected). 1 → 2: touched yesterday. 2 → 3: a FORGOTTEN
        // edge already sits on the pair. 3 → 4: no date at all.
        regler(
            db,
            n[0],
            n[1],
            "SYNAPSE",
            vec![("reinforcement_count", Value::from(3_i64))],
        );
        regler(
            db,
            n[1],
            n[2],
            "SYNAPSE",
            vec![(
                "last_reinforced_at",
                instant(Utc::now() - Duration::days(1)),
            )],
        );
        let (a, b) = (noeud(db, "Note", n[2]), noeud(db, "Note", n[3]));
        db.create_edge_with_props(
            a,
            b,
            "FORGOTTEN",
            vec![
                ("weight", Value::from(0.9)),
                ("forgotten_reason", Value::from("old")),
            ],
        );
        let (c, d) = (noeud(db, "Note", n[3]), noeud(db, "Note", n[4]));
        for (dst, e) in db.store().edges_from(c, Direction::Outgoing) {
            if dst == d {
                db.remove_edge_property(e, "created_at");
            }
        }
    })
    .await;
    let store = open(&base).await;
    let r = store
        .decay_synapses(&politique(10.0, 5.0, 1, 1.0))
        .await
        .unwrap();
    // 12 edges, three of them held back (protected, touched yesterday, no date). The weakest
    // candidate is `2 → 3` (0.003): it goes first, whatever the ids say.
    assert_eq!(r.decayed, 12);
    assert!(r.archived >= 1, "{r:?}");
    store.close().await.unwrap();
    assert_eq!(
        arete_brute(&base, n[0], n[1], "SYNAPSE").await.len(),
        1,
        "protected"
    );
    assert_eq!(
        arete_brute(&base, n[1], n[2], "SYNAPSE").await.len(),
        1,
        "touched yesterday"
    );
    assert_eq!(
        arete_brute(&base, n[3], n[4], "SYNAPSE").await.len(),
        1,
        "no date: left alone"
    );
    let f = arete_brute(&base, n[2], n[3], "FORGOTTEN").await;
    assert_eq!(f.len(), 1, "replaced, not doubled");
    assert_eq!(f[0]["forgotten_reason"].as_str(), Some("decay"));
    assert!(proche(nombre(&f[0], "weight"), 0.003));
}

/// Two `FORGOTTEN` edges on a pair about to be archived: the pass does not guess which to replace;
/// it refuses and writes nothing.
#[tokio::test]
async fn two_forgotten_edges_on_an_archived_pair_refuse_the_pass() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let n = un_anneau(&base).await;
    let n2 = n.clone();
    base_brute(&base, move |db| {
        let (a, b) = (noeud(db, "Note", n2[0]), noeud(db, "Note", n2[1]));
        for _ in 0..2 {
            db.create_edge_with_props(a, b, "FORGOTTEN", vec![("weight", Value::from(0.9))]);
        }
    })
    .await;
    let store = open(&base).await;
    let e = store
        .decay_synapses(&politique(10.0, 5.0, 1, 0.25))
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("FORGOTTEN"), "{e:#}");
    store.close().await.unwrap();
    assert_eq!(arete_brute(&base, n[0], n[1], "SYNAPSE").await.len(), 1);
    assert!(proche(
        nombre(
            &arete_brute(&base, n[0], n[1], "SYNAPSE").await[0],
            "weight"
        ),
        0.001
    ));
    assert!(
        !arete_brute(&base, n[1], n[0], "SYNAPSE").await[0].contains_key("decayed_at"),
        "nothing written"
    );
}

/// The archive round trip with the sub-lot A2: `reinforce_synapses` revives a pair the pass
/// archived, at the weight it had (never under `0.02`), and the archive rows go away.
#[tokio::test]
async fn an_archived_pair_is_revived_by_a_reinforcement() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let n = un_anneau(&base).await;
    let store = open(&base).await;
    store
        .decay_synapses(&politique(10.0, 5.0, 1, 0.25))
        .await
        .unwrap();
    // 0 → 1 is archived (0.001), 1 → 0 is not.
    store.reinforce_synapses(&[n[0], n[1]], 0.1).await.unwrap();
    store.close().await.unwrap();
    assert!(
        arete_brute(&base, n[0], n[1], "FORGOTTEN").await.is_empty(),
        "the archive row goes"
    );
    let revive = arete_brute(&base, n[0], n[1], "SYNAPSE").await;
    assert_eq!(revive.len(), 1);
    assert!(
        proche(nombre(&revive[0], "weight"), 0.02),
        "never under the forget threshold"
    );
}

/// The scars heal by `0.02 × 0.05` per pass, to zero under `0.001`, for EVERY note and decision,
/// whatever the scope (Neo4j's query has no scope on this step: a project pass heals the scars of
/// all the projects). A dormant project's pass does nothing at all.
#[tokio::test]
async fn the_scars_heal_on_every_pass_whatever_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, b, c, d);
    let (p, dor);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        dor = projet_dormant(&store).await;
        let n = des_notes(&store, Some(dor.id), 2).await;
        (a, b) = (n[0].id, n[1].id);
        c = une_note(&store, None).await.id;
        store.apply_scars(&[a], 0.5).await.unwrap();
        store.apply_scars(&[b], 0.0015).await.unwrap();
        store.apply_scars(&[c], 0.5).await.unwrap();
        store.close().await.unwrap();
    }
    d = {
        let id = Uuid::new_v4();
        base_brute(&base, move |db| {
            db.create_node_with_props(
                &["Decision"],
                vec![
                    ("id", Value::from(id.to_string())),
                    ("description", Value::from("d")),
                    ("rationale", Value::from("r")),
                    ("decided_at", Value::from("2026-01-01T00:00:00Z")),
                    ("scar_intensity", Value::from(0.3)),
                ],
            );
        })
        .await;
        id
    };
    let store = open(&base).await;
    let scar = |id: Uuid| {
        let store = &store;
        async move { store.get_note(id).await.unwrap().unwrap().scar_intensity }
    };
    // The dormant project's pass: nothing, scars included.
    let r = store
        .decay_project_synapses(dor.id, &Default::default())
        .await
        .unwrap();
    assert_eq!((r.decayed, r.archived), (0, 0));
    assert!(proche(scar(a).await, 0.5));
    // An active project's pass heals the scars of the OTHER projects' notes and of decisions.
    store
        .decay_project_synapses(p.id, &Default::default())
        .await
        .unwrap();
    assert!(proche(scar(a).await, 0.499), "{}", scar(a).await);
    assert_eq!(
        scar(b).await,
        0.0,
        "0.0015 - 0.001 = 0.0005 < 0.001: healed"
    );
    assert!(proche(scar(c).await, 0.499));
    let decision = store.get_decision(d).await.unwrap().unwrap();
    assert!(
        proche(decision.scar_intensity, 0.299),
        "{}",
        decision.scar_intensity
    );
    store.decay_synapses(&Default::default()).await.unwrap();
    assert!(proche(scar(a).await, 0.498));
}

/// The scope: a project pass walks the edges leaving the notes of THAT project only; the global
/// pass leaves the edges of a dormant project alone (they are frozen) and walks the notes with
/// no project.
#[tokio::test]
async fn the_passes_walk_their_scope_and_freeze_the_dormant_projects() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let paires: Vec<(Uuid, Uuid)>;
    let (pa, pb);
    {
        let store = open(&base).await;
        pa = projet_actif(&store).await;
        pb = projet_actif(&store).await;
        let dor = projet_dormant(&store).await;
        let mut v = Vec::new();
        for pid in [Some(pa.id), Some(pb.id), Some(dor.id), None] {
            let n = des_notes(&store, pid, 2).await;
            store
                .create_synapses(n[0].id, &[(n[1].id, 0.8)])
                .await
                .unwrap();
            v.push((n[0].id, n[1].id));
        }
        paires = v;
        store.close().await.unwrap();
    }
    let p2 = paires.clone();
    base_brute(&base, move |db| {
        for (x, y) in &p2 {
            for (a, b) in [(*x, *y), (*y, *x)] {
                regler(
                    db,
                    a,
                    b,
                    "SYNAPSE",
                    vec![
                        ("source", Value::from("coactivation")),
                        ("decayed_at", instant(Utc::now() - Duration::days(10))),
                    ],
                );
            }
        }
    })
    .await;
    let store = open(&base).await;
    let pol = politique(10.0, 5.0, 3, 0.05);
    let r = store.decay_project_synapses(pa.id, &pol).await.unwrap();
    assert_eq!(r.decayed, 2, "the two edges of the project only");
    let r = store.decay_synapses(&pol).await.unwrap();
    // pb (2) and the notes without a project (2): the project `pa` was just decayed (its clock is
    // now, it is walked again and unchanged), and the dormant project's edges are not walked.
    assert_eq!(r.decayed, 6, "{r:?}");
    store.close().await.unwrap();
    let poids = |i: usize| {
        let (x, y) = paires[i];
        let base = base.clone();
        async move { nombre(&arete_brute(&base, x, y, "SYNAPSE").await[0], "weight") }
    };
    assert!(proche(poids(0).await, 0.4), "decayed by its own pass");
    assert!(proche(poids(1).await, 0.4), "decayed by the global pass");
    assert!(proche(poids(2).await, 0.8), "dormant: frozen");
    assert!(proche(poids(3).await, 0.4), "no project: walked");
}

/// The pass refuses, writing nothing: an edge with no weight (Neo4j would write a null), a policy
/// that is not a number, a quarantined note (the pass would miss its edges).
#[tokio::test]
async fn the_forgetting_pass_refuses_what_it_cannot_answer_faithfully() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, b, c, d);
    {
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        let n = des_notes(&store, Some(p.id), 4).await;
        (a, b, c, d) = (n[0].id, n[1].id, n[2].id, n[3].id);
        store.create_synapses(a, &[(b, 0.8)]).await.unwrap();
        store.create_synapses(c, &[(d, 0.8)]).await.unwrap();
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        regler(
            db,
            c,
            d,
            "SYNAPSE",
            vec![("decayed_at", instant(Utc::now() - Duration::days(10)))],
        );
        let (na, nb) = (noeud(db, "Note", a), noeud(db, "Note", b));
        db.create_edge_with_props(na, nb, "SYNAPSE", vec![("source", Value::from("cosine"))]);
    })
    .await;
    let store = open(&base).await;
    let e = store
        .decay_synapses(&politique(10.0, 5.0, 3, 0.05))
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("weight"), "{e:#}");
    for mauvaise in [
        politique(f64::NAN, 5.0, 3, 0.05),
        politique(10.0, 0.0, 3, 0.05),
        politique(10.0, 5.0, 3, f64::NAN),
    ] {
        assert!(store.decay_synapses(&mauvaise).await.is_err());
    }
    store.close().await.unwrap();
    assert!(arete_brute(&base, c, d, "SYNAPSE").await[0]
        .get("weight")
        .is_some_and(|v| v.as_float64() == Some(0.8)));
    // A quarantined note.
    base_brute(&base, |db| {
        db.create_node_with_props(
            &["Note"],
            vec![("id", Value::from(Uuid::new_v4().to_string()))],
        );
    })
    .await;
    let store = open(&base).await;
    let e = store
        .decay_project_synapses(Uuid::new_v4(), &Default::default())
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
}

/// A pass cut at every engine write: the base opens clean and NO edge is lost: every directed pair
/// of the ring still has its `SYNAPSE` edge or its `FORGOTTEN` one (the archive is written before
/// the synapse is deleted), and the next pass goes through.
#[tokio::test]
async fn a_forgetting_pass_cut_at_every_write_never_loses_an_edge() {
    for k in 1..=60 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let n = un_anneau(&base).await;
        let store = open(&base).await;
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store
            .decay_synapses(&politique(10.0, 5.0, 1, 0.25))
            .await
            .is_ok()
        {
            assert!(k > 20, "la coupure n'a atteint que {} points", k - 1);
            return;
        }
        drop(store);
        let store = open(&base).await;
        assert!(
            store.rapport_d_ouverture().en_quarantaine.is_empty(),
            "k={k}"
        );
        store.close().await.unwrap();
        for i in 0..6 {
            for (x, y) in [(n[i], n[(i + 1) % 6]), (n[(i + 1) % 6], n[i])] {
                let tenues = arete_brute(&base, x, y, "SYNAPSE").await.len()
                    + arete_brute(&base, x, y, "FORGOTTEN").await.len();
                assert!(tenues >= 1, "k={k}: l'arête {i} a été perdue");
            }
        }
        let store = open(&base).await;
        store
            .decay_synapses(&politique(10.0, 5.0, 1, 0.25))
            .await
            .unwrap();
    }
    panic!("the cut never ran out of writes: raise the bound");
}

// ============================================================================
// The lifecycle sweep
// ============================================================================

fn une_note_de(projet: Uuid, f: impl FnOnce(&mut Note)) -> Note {
    let mut n = Note::new(
        Some(projet),
        NoteType::Observation,
        "contenu".into(),
        "t".into(),
    );
    n.memory_horizon = MemoryHorizon::Operational;
    f(&mut n);
    n
}

/// `consolidate_memory`, case by case (the rules are `NoteLifecycleManager`'s, the usage signal is
/// `max(activation_count, reactivation_count, synapse count)`): a dead note (no activation, 90+
/// days), a depleted one (energy < 0.05, 60+ days), an idle ephemeral (48 h), a confirmed ephemeral
/// promoted to consolidated, a reactivated ephemeral promoted to operational; a consolidated note,
/// an archived one and a note a synapse keeps alive are left alone. Every change appends to the
/// note's history, actor `consolidate_memory`, and a second sweep changes nothing.
#[tokio::test]
async fn consolidate_memory_archives_and_promotes_case_by_case() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let p = projet_actif(&store).await;
    let maintenant = Utc::now();
    let jours = |j: i64| maintenant - Duration::days(j);
    let morte = une_note_de(p.id, |n| n.created_at = jours(100));
    let epuisee = une_note_de(p.id, |n| {
        n.created_at = jours(70);
        n.energy = 0.01;
        n.activation_count = 5;
    });
    let inactive = une_note_de(p.id, |n| {
        n.memory_horizon = MemoryHorizon::Ephemeral;
        n.created_at = jours(4);
        n.last_activated = Some(jours(3));
        n.activation_count = 1;
    });
    let confirmee = une_note_de(p.id, |n| {
        n.memory_horizon = MemoryHorizon::Ephemeral;
        n.human_confirmed_at = Some(maintenant);
        n.human_confirmed_by = Some("moi".into());
    });
    let reactivee = une_note_de(p.id, |n| {
        n.memory_horizon = MemoryHorizon::Ephemeral;
        n.reactivation_count = 2;
    });
    let consolidee = une_note_de(p.id, |n| {
        n.memory_horizon = MemoryHorizon::Consolidated;
        n.created_at = jours(400);
        n.energy = 0.0;
    });
    let gardee = une_note_de(p.id, |n| n.created_at = jours(100));
    let archivee = une_note_de(p.id, |n| {
        n.created_at = jours(100);
        n.status = NoteStatus::Archived;
    });
    for n in [
        &morte,
        &epuisee,
        &inactive,
        &confirmee,
        &reactivee,
        &consolidee,
        &gardee,
        &archivee,
    ] {
        store.create_note(n).await.unwrap();
    }
    // A synapse keeps `gardee` alive: one edge is a usage signal.
    store
        .create_synapses(gardee.id, &[(consolidee.id, 0.5)])
        .await
        .unwrap();

    assert_eq!(store.consolidate_memory().await.unwrap(), (2, 3));
    let lire = |id: Uuid| {
        let store = &store;
        async move { store.get_note(id).await.unwrap().unwrap() }
    };
    for (id, raison) in [
        (morte.id, "dead_note_no_activations_90d"),
        (epuisee.id, "low_energy_60d"),
        (inactive.id, "ephemeral_expired"),
    ] {
        let n = lire(id).await;
        assert_eq!(n.status, NoteStatus::Archived, "{raison}");
        let c = n.changes.last().unwrap();
        assert_eq!(c.actor, "consolidate_memory");
        assert_eq!(c.details.as_ref().unwrap()["reason"], raison);
    }
    let n = lire(confirmee.id).await;
    assert_eq!(
        (n.memory_horizon.clone(), n.status),
        (MemoryHorizon::Consolidated, NoteStatus::Active)
    );
    assert_eq!(
        n.changes.last().unwrap().details.as_ref().unwrap()["from"],
        "ephemeral"
    );
    assert_eq!(
        lire(reactivee.id).await.memory_horizon,
        MemoryHorizon::Operational
    );
    for id in [consolidee.id, gardee.id] {
        let n = lire(id).await;
        assert_eq!(n.status, NoteStatus::Active);
        assert_eq!(
            n.changes.len(),
            1,
            "left alone: only its creation in the history"
        );
    }
    assert_eq!(lire(archivee.id).await.status, NoteStatus::Archived);
    // Nothing more to do.
    assert_eq!(store.consolidate_memory().await.unwrap(), (0, 0));
    // The sweep survives a reopen.
    store.close().await.unwrap();
    let store = open(&base).await;
    assert_eq!(
        store.get_note(morte.id).await.unwrap().unwrap().status,
        NoteStatus::Archived
    );
    assert_eq!(store.consolidate_memory().await.unwrap(), (0, 0));
}

/// REGRESSION. The sweep refuses, writing nothing: when a note was quarantined at open (it would be
/// skipped), and when the history of a note to change is present and unreadable (Neo4j would
/// overwrite it with an empty one plus the new entry).
#[tokio::test]
async fn consolidate_memory_refuses_a_quarantine_and_an_unreadable_history() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (a, b);
    {
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        let jours = |j: i64| Utc::now() - Duration::days(j);
        let (x, y) = (
            une_note_de(p.id, |n| n.created_at = jours(100)),
            une_note_de(p.id, |n| n.created_at = jours(100)),
        );
        store.create_note(&x).await.unwrap();
        store.create_note(&y).await.unwrap();
        (a, b) = (x.id, y.id);
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        db.set_node_property(
            noeud(db, "Note", a),
            "changes_json",
            Value::from("{pas du json"),
        );
    })
    .await;
    let store = open(&base).await;
    let e = store.consolidate_memory().await.unwrap_err();
    assert!(format!("{e:#}").contains("changes_json"), "{e:#}");
    // Nothing written: the readable note is still active.
    assert_eq!(
        store.get_note(b).await.unwrap().unwrap().status,
        NoteStatus::Active
    );
    store.close().await.unwrap();
    base_brute(&base, |db| {
        db.create_node_with_props(
            &["Note"],
            vec![("id", Value::from(Uuid::new_v4().to_string()))],
        );
    })
    .await;
    let store = open(&base).await;
    let e = store.consolidate_memory().await.unwrap_err();
    assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
    assert_eq!(
        store.get_note(b).await.unwrap().unwrap().status,
        NoteStatus::Active
    );
}

/// A sweep cut at every engine write: the history is written BEFORE the effect, so a cut leaves a
/// note active (with, at worst, an entry whose effect is missing), never archived without its trace;
/// the next sweep finishes it.
#[tokio::test]
async fn a_sweep_cut_at_every_write_never_archives_without_a_trace() {
    for k in 1..=6 {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = projet_actif(&store).await;
        let n = une_note_de(p.id, |n| n.created_at = Utc::now() - Duration::days(100));
        store.create_note(&n).await.unwrap();
        let base_len = store.get_note(n.id).await.unwrap().unwrap().changes.len();
        store.interrompre_au_prochain(Interruption::AvantEtape(k));
        if store.consolidate_memory().await.is_ok() {
            assert!(k > 2, "la coupure n'a atteint que {} points", k - 1);
            return;
        }
        drop(store);
        let store = open(&base).await;
        let lue = store.get_note(n.id).await.unwrap().unwrap();
        assert_eq!(lue.status, NoteStatus::Active, "k={k}");
        assert!(lue.changes.len() <= base_len + 1, "k={k}");
        assert_eq!(store.consolidate_memory().await.unwrap(), (0, 1), "k={k}");
        let lue = store.get_note(n.id).await.unwrap().unwrap();
        assert_eq!(lue.status, NoteStatus::Archived);
        assert!(lue.changes.len() > base_len, "the sweep leaves its trace");
    }
    panic!("the cut never ran out of writes: raise the bound");
}

// ============================================================================
// The notes the heuristic wrote
// ============================================================================

fn une_auto(projet: Uuid, par: &str, contenu: &str, f: impl FnOnce(&mut Note)) -> Note {
    let mut n = Note::new(
        Some(projet),
        NoteType::Observation,
        contenu.into(),
        par.into(),
    );
    f(&mut n);
    n
}

/// `auto_observed_report` groups the notes the detectors wrote (by author or by the `auto-observed`
/// tag) by `(status, note_type, importance)`, ordered by count descending then the three strings,
/// with the average content size truncated (`toInteger`); `archive_unconfirmed_auto_observed`
/// archives only the ACTIVE ones no person confirmed, in scope, and counts them.
#[tokio::test]
async fn the_auto_observed_notes_are_reported_by_shape_and_archived_when_unconfirmed() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let det = "system/observation-detector";
    let notes = vec![
        une_auto(p.id, det, "0123456789", |_| {}),
        une_auto(p.id, det, "012345678901234567890", |_| {}),
        une_auto(p.id, "agent", "tagged", |n| {
            n.tags = vec!["auto-observed".into()];
            n.note_type = NoteType::Gotcha;
            n.importance = project_orchestrator::notes::NoteImportance::High;
        }),
        une_auto(p.id, "system/rfc-detector", "arch", |n| {
            n.status = NoteStatus::Archived
        }),
        une_auto(p.id, "person", "humaine", |_| {}),
        une_auto(p.id, det, "confirmee", |n| {
            n.human_confirmed_at = Some(Utc::now())
        }),
        une_auto(q.id, det, "ailleurs", |_| {}),
    ];
    for n in &notes {
        store.create_note(n).await.unwrap();
    }
    let cle = |r: &serde_json::Value| {
        (
            r["status"].as_str().unwrap().to_string(),
            r["note_type"].as_str().unwrap().to_string(),
            r["importance"].as_str().unwrap().to_string(),
            r["notes"].as_i64().unwrap(),
            r["avg_size"].as_i64().unwrap(),
        )
    };
    let tout = store.auto_observed_report(None).await.unwrap();
    // (active, observation, medium): n1, n2, the confirmed one and the other project's = 4, sizes
    // 10 + 21 + 9 + 8 = 48, 48 / 4 = 12. Then the singles, by status: active < archived.
    assert_eq!(
        tout.iter().map(cle).collect::<Vec<_>>(),
        vec![
            (
                "active".into(),
                "observation".into(),
                "medium".into(),
                4,
                12
            ),
            ("active".into(), "gotcha".into(), "high".into(), 1, 6),
            (
                "archived".into(),
                "observation".into(),
                "medium".into(),
                1,
                4
            ),
        ]
    );
    // In scope: the average is truncated, not rounded (10 + 21 + 9 = 40, 40 / 3 = 13).
    let seul = store.auto_observed_report(Some(p.id)).await.unwrap();
    assert_eq!(
        cle(&seul[0]),
        (
            "active".into(),
            "observation".into(),
            "medium".into(),
            3,
            13
        )
    );
    assert!(store
        .auto_observed_report(Some(Uuid::new_v4()))
        .await
        .unwrap()
        .is_empty());

    // Archive: active, unconfirmed, detector or tag; the scope; the count.
    assert_eq!(
        store
            .archive_unconfirmed_auto_observed(Some(p.id))
            .await
            .unwrap(),
        3
    );
    assert_eq!(
        store
            .archive_unconfirmed_auto_observed(Some(p.id))
            .await
            .unwrap(),
        0
    );
    let statut = |i: usize| {
        let (store, id) = (&store, notes[i].id);
        async move { store.get_note(id).await.unwrap().unwrap().status }
    };
    for i in 0..3 {
        assert_eq!(statut(i).await, NoteStatus::Archived, "{i}");
    }
    assert_eq!(statut(4).await, NoteStatus::Active, "a person's note");
    assert_eq!(statut(5).await, NoteStatus::Active, "confirmed by a person");
    assert_eq!(statut(6).await, NoteStatus::Active, "another project");
    assert_eq!(
        store.archive_unconfirmed_auto_observed(None).await.unwrap(),
        1
    );
    assert_eq!(statut(6).await, NoteStatus::Archived);
    let apres = store.auto_observed_report(Some(p.id)).await.unwrap();
    assert_eq!(
        cle(&apres[0]),
        (
            "archived".into(),
            "observation".into(),
            "medium".into(),
            3,
            11
        )
    );
    store.close().await.unwrap();
    let store = open(&base).await;
    assert_eq!(
        store.get_note(notes[0].id).await.unwrap().unwrap().status,
        NoteStatus::Archived
    );
}

/// Both refuse when a note was quarantined (a report or an archive would miss it).
#[tokio::test]
async fn the_auto_observed_methods_refuse_a_quarantined_note() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    {
        let store = open(&base).await;
        store.close().await.unwrap();
    }
    base_brute(&base, |db| {
        db.create_node_with_props(
            &["Note"],
            vec![("id", Value::from(Uuid::new_v4().to_string()))],
        );
    })
    .await;
    let store = open(&base).await;
    for e in [
        store.auto_observed_report(None).await.unwrap_err(),
        store
            .archive_unconfirmed_auto_observed(None)
            .await
            .unwrap_err(),
    ] {
        assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
    }
}

// ============================================================================
// The audit
// ============================================================================

/// `audit_knowledge_gaps`: notes with no outgoing `LINKED_TO`, decisions with no `AFFECTS`, commits
/// with no `TOUCHES`, skills with no `MEMBER_OF` from a note nor `MEMBER_OF_SKILL` from a decision,
/// scoped to the project, with the true totals, the samples (`id[..8] — left(text, 80)`), and the
/// census of every relationship type of the base (compared with a count made here, from the raw
/// edges).
#[tokio::test]
async fn the_audit_finds_the_four_kinds_of_gap_and_counts_every_relationship() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let (p, q);
    let (n1, n2, n3, nq);
    {
        let store = open(&base).await;
        p = projet_actif(&store).await;
        q = projet_actif(&store).await;
        n1 = une_note(&store, Some(p.id)).await;
        n2 = Note::new(
            Some(p.id),
            NoteType::Observation,
            "x".repeat(100),
            "t".into(),
        );
        n3 = une_note(&store, Some(p.id)).await;
        nq = une_note(&store, Some(q.id)).await;
        store.create_note(&n2).await.unwrap();
        store
            .link_note_to_entity(n1.id, &EntityType::Project, &p.id.to_string(), None, None)
            .await
            .unwrap();
        store.create_synapses(n1.id, &[(n3.id, 0.5)]).await.unwrap();
        store.close().await.unwrap();
    }
    let (pid, qid, n2i, n1i) = (p.id.to_string(), q.id.to_string(), n2.id, n1.id);
    let recensement = base_brute(&base, move |db| {
        let mk =
            |label: &str, props: Vec<(&str, Value)>| db.create_node_with_props(&[label], props);
        let (d1, d2) = (Uuid::new_v4(), Uuid::new_v4());
        let decision = |id: Uuid, projet: &str, desc: &str| {
            mk(
                "Decision",
                vec![
                    ("id", Value::from(id.to_string())),
                    ("description", Value::from(desc.to_string())),
                    ("rationale", Value::from("r")),
                    ("decided_at", Value::from("2026-01-01T00:00:00Z")),
                    ("project_id", Value::from(projet.to_string())),
                ],
            )
        };
        let (nd1, nd2) = (
            decision(d1, &pid, "sans lien"),
            decision(d2, &pid, "avec lien"),
        );
        decision(Uuid::new_v4(), &qid, "ailleurs");
        db.create_edge(nd2, noeud(db, "Note", n1i), "AFFECTS");
        let commit = |hash: &str, msg: &str, projet: &str| {
            mk(
                "Commit",
                vec![
                    ("hash", Value::from(hash.to_string())),
                    ("message", Value::from(msg.to_string())),
                    ("project_id", Value::from(projet.to_string())),
                ],
            )
        };
        commit("abcdef0123456789", &"m".repeat(100), &pid);
        let c2 = commit("fedcba9876543210", "touche", &pid);
        commit("0000000000000000", "ailleurs", &qid);
        let fichier = mk("File", vec![("path", Value::from("a.rs"))]);
        db.create_edge(c2, fichier, "TOUCHES");
        let skill = |nom: &str, projet: &str| {
            mk(
                "Skill",
                vec![
                    ("id", Value::from(Uuid::new_v4().to_string())),
                    ("name", Value::from(nom.to_string())),
                    ("project_id", Value::from(projet.to_string())),
                ],
            )
        };
        let (_s1, s2, s3, s4) = (
            skill("vide", &pid),
            skill("par note", &pid),
            skill("par fichier", &pid),
            skill("par decision", &pid),
        );
        db.create_edge(noeud(db, "Note", n2i), s2, "MEMBER_OF");
        db.create_edge(fichier, s3, "MEMBER_OF");
        db.create_edge(nd1, s4, "MEMBER_OF_SKILL");
        skill("ailleurs", &qid);
        // The census, counted here from the raw edges.
        let mut m: std::collections::BTreeMap<String, i64> = Default::default();
        for n in db.store().all_node_ids() {
            for (_, e) in db.store().edges_from(n, Direction::Outgoing) {
                *m.entry(db.get_edge(e).unwrap().edge_type.to_string())
                    .or_insert(0) += 1;
            }
        }
        m
    })
    .await;
    let store = open(&base).await;
    let r = store.audit_knowledge_gaps(p.id).await.unwrap();
    // `n2` and `n3` have no LINKED_TO (the other project's note is out).
    assert_eq!(r.orphan_notes_total, 2);
    let mut paires = [
        (n2.id.to_string(), "x".repeat(80)),
        (n3.id.to_string(), "c".to_string()),
    ];
    paires.sort();
    let attendu: Vec<String> = paires
        .iter()
        .map(|(i, c)| format!("{} — {c}", &i[..8]))
        .collect();
    assert_eq!(r.orphan_notes, attendu, "sorted by id, `left(content, 80)`");
    // The decision with no AFFECTS: the one with `d2` is out, the other project's too.
    assert_eq!(r.decisions_without_affects_total, 1);
    assert!(
        r.decisions_without_affects[0].ends_with("— sans lien"),
        "{:?}",
        r.decisions_without_affects
    );
    assert_eq!(r.commits_without_touches_total, 1);
    assert_eq!(
        r.commits_without_touches,
        vec![format!("abcdef01 — {}", "m".repeat(80))]
    );
    // Skills: `vide` and `par fichier` (a MEMBER_OF from a File is not a member).
    assert_eq!(r.skills_without_members_total, 2);
    let mut noms: Vec<&str> = r
        .skills_without_members
        .iter()
        .map(|l| l.split(" — ").nth(1).unwrap())
        .collect();
    noms.sort();
    assert_eq!(noms, vec!["par fichier", "vide"]);
    assert_eq!(r.total_gaps, 2 + 1 + 1 + 2);
    // The census: every relationship type, ordered by name, the counts of the raw base.
    let obtenu: Vec<(String, i64)> = r
        .relationship_type_counts
        .iter()
        .map(|x| (x.rel_type.clone(), x.count))
        .collect();
    assert_eq!(obtenu, recensement.clone().into_iter().collect::<Vec<_>>());
    let compte = |t: &str| recensement.get(t).copied().unwrap_or(0);
    assert_eq!(
        (compte("LINKED_TO"), compte("AFFECTS"), compte("TOUCHES")),
        (1, 1, 1)
    );
    assert_eq!(
        (
            compte("MEMBER_OF"),
            compte("MEMBER_OF_SKILL"),
            compte("SYNAPSE")
        ),
        (2, 1, 2)
    );
    // The other project sees its own gaps only.
    let rq = store.audit_knowledge_gaps(q.id).await.unwrap();
    assert_eq!(
        (
            rq.orphan_notes_total,
            rq.decisions_without_affects_total,
            rq.commits_without_touches_total,
            rq.skills_without_members_total
        ),
        (1, 1, 1, 1)
    );
    let _ = nq;
    assert_eq!(
        store
            .audit_knowledge_gaps(Uuid::new_v4())
            .await
            .unwrap()
            .total_gaps,
        0
    );
}

/// The samples are capped (100; the true total is not) and ordered by id.
#[tokio::test]
async fn the_audit_caps_its_samples_and_not_its_totals() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let p = projet_actif(&store).await;
    let mut ids: Vec<String> = Vec::new();
    for _ in 0..105 {
        ids.push(une_note(&store, Some(p.id)).await.id.to_string());
    }
    ids.sort();
    let r = store.audit_knowledge_gaps(p.id).await.unwrap();
    assert_eq!(
        (r.orphan_notes_total, r.orphan_notes.len(), r.total_gaps),
        (105, 100, 105)
    );
    let attendu: Vec<String> = ids[..100]
        .iter()
        .map(|i| format!("{} — c", &i[..8]))
        .collect();
    assert_eq!(r.orphan_notes, attendu);
}

// ============================================================================
// Maintenance writes of a project and of a protocol
// ============================================================================

/// `ensure_project_canonical_paths`: a project whose `canonical_path` differs from its resolved
/// root is rewritten (`(scanned, updated)`), one whose root cannot be resolved this time keeps
/// what it had, one with no root is not scanned, and a second call writes nothing.
#[tokio::test]
async fn the_canonical_paths_are_repaired_once_and_never_from_an_unresolvable_root() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let reel = dir.path().join("reel");
    std::fs::create_dir(&reel).unwrap();
    let lien = dir.path().join("lien");
    std::os::unix::fs::symlink(&reel, &lien).unwrap();
    let canonique = std::fs::canonicalize(&reel)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let (p1, p2, p3);
    {
        let store = open(&base).await;
        let mut a = a_project();
        a.root_path = lien.to_string_lossy().into_owned();
        let mut b = a_project();
        b.root_path = "/definitivement/absent/ici".into();
        let c = a_project();
        for x in [&a, &b, &c] {
            store.create_project(x).await.unwrap();
        }
        (p1, p2, p3) = (a.id, b.id, c.id);
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        db.set_node_property(
            noeud(db, "Project", p1),
            "canonical_path",
            Value::from("/ancien"),
        );
        db.set_node_property(
            noeud(db, "Project", p2),
            "canonical_path",
            Value::from("/valeur/precedente"),
        );
    })
    .await;
    let lire = |base: std::path::PathBuf| async move {
        base_brute(&base, move |db| {
            let c = |id| {
                db.get_node(noeud(db, "Project", id))
                    .unwrap()
                    .get_property("canonical_path")
                    .and_then(|v| v.as_str().map(str::to_string))
            };
            (c(p1), c(p2), c(p3))
        })
        .await
    };
    let store = open(&base).await;
    assert_eq!(
        store.ensure_project_canonical_paths().await.unwrap(),
        (2, 1)
    );
    assert_eq!(
        store.ensure_project_canonical_paths().await.unwrap(),
        (2, 0),
        "idempotent"
    );
    store.close().await.unwrap();
    let (c1, c2, c3) = lire(base.clone()).await;
    assert_eq!(c1.as_deref(), Some(canonique.as_str()));
    assert_eq!(
        c2.as_deref(),
        Some("/valeur/precedente"),
        "unresolvable: what it had is kept"
    );
    assert_eq!(c3.as_deref().unwrap_or(""), "", "no root: not scanned");
}

/// A cut of the write leaves the stale value, and the next call completes it. A project node the
/// index does not serve (quarantined) refuses the call.
#[tokio::test]
async fn the_canonical_paths_survive_a_cut_and_refuse_a_quarantined_project() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let reel = dir.path().join("reel");
    std::fs::create_dir(&reel).unwrap();
    let id;
    {
        let store = open(&base).await;
        let mut a = a_project();
        a.root_path = reel.to_string_lossy().into_owned();
        store.create_project(&a).await.unwrap();
        id = a.id;
        store.close().await.unwrap();
    }
    base_brute(&base, move |db| {
        db.set_node_property(
            noeud(db, "Project", id),
            "canonical_path",
            Value::from("/ancien"),
        );
    })
    .await;
    let store = open(&base).await;
    store.interrompre_au_prochain(Interruption::AvantEtape(1));
    assert!(store.ensure_project_canonical_paths().await.is_err());
    drop(store);
    let store = open(&base).await;
    assert_eq!(
        store.ensure_project_canonical_paths().await.unwrap(),
        (1, 1)
    );
    store.close().await.unwrap();
    base_brute(&base, |db| {
        db.create_node_with_props(
            &["Project"],
            vec![("id", Value::from(Uuid::new_v4().to_string()))],
        );
    })
    .await;
    let store = open(&base).await;
    let e = store.ensure_project_canonical_paths().await.unwrap_err();
    assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
}

/// `rescope_protocol`: attaches, makes transverse, is idempotent, survives a reopen, changes
/// nothing else; a protocol or a project that does not exist refuses and writes nothing.
#[tokio::test]
async fn a_protocol_is_rescoped_to_a_project_or_made_transverse() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let store = open(&base).await;
    let (p, q) = (projet_actif(&store).await, projet_actif(&store).await);
    let proto = Protocol::new(Some(p.id), "proto", Uuid::new_v4());
    store.upsert_protocol(&proto).await.unwrap();
    let avant = store.get_protocol(proto.id).await.unwrap().unwrap();
    store.rescope_protocol(proto.id, Some(q.id)).await.unwrap();
    let lu = store.get_protocol(proto.id).await.unwrap().unwrap();
    assert_eq!(lu.project_id, Some(q.id));
    assert_eq!(lu.updated_at, avant.updated_at, "only the scope moves");
    assert_eq!(lu.name, "proto");
    store.rescope_protocol(proto.id, Some(q.id)).await.unwrap();
    store.rescope_protocol(proto.id, None).await.unwrap();
    assert_eq!(
        store
            .get_protocol(proto.id)
            .await
            .unwrap()
            .unwrap()
            .project_id,
        None
    );
    // Transverse: listed for every project.
    assert_eq!(
        store
            .list_protocols(Some(p.id), None, 10, 0)
            .await
            .unwrap()
            .1,
        1
    );
    // Refused, nothing written.
    store.rescope_protocol(proto.id, Some(p.id)).await.unwrap();
    let e = store
        .rescope_protocol(proto.id, Some(Uuid::new_v4()))
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("n'existe pas"), "{e:#}");
    assert_eq!(
        store
            .get_protocol(proto.id)
            .await
            .unwrap()
            .unwrap()
            .project_id,
        Some(p.id)
    );
    let e = store
        .rescope_protocol(Uuid::new_v4(), None)
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("not found"), "{e:#}");
    store.rescope_protocol(proto.id, Some(q.id)).await.unwrap();
    store.close().await.unwrap();
    let store = open(&base).await;
    assert_eq!(
        store
            .get_protocol(proto.id)
            .await
            .unwrap()
            .unwrap()
            .project_id,
        Some(q.id)
    );
}
