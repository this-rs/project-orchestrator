//! PORT-DATES — a migrated date is READ, not replaced by the time of the read.
//!
//! The migrator writes `ZonedDatetime` (faithful to Neo4j's zoned `DateTime`),
//! while the portage used to read dates with `as_str()` only: a `ZonedDatetime`
//! was therefore indistinguishable from a missing property. Two consequences,
//! both silent: `created_at` "absent" quarantined every migrated node, and
//! `decided_at` "absent" was replaced by `Utc::now()` — the decision appeared to
//! have been taken at the moment of the read.
//!
//! These tests write the date the way the migrator does, in the base, then
//! reopen through the portage. They fail on the `as_str()` shape and pass on the
//! single decoder (`obrain::date_propriete`).
#![cfg(feature = "obrain-backend")]

use std::path::Path;

use chrono::{TimeZone, Utc};
use obrain_common::types::{Timestamp, Value, ZonedDatetime};
use project_orchestrator::neo4j::traits::*;
use project_orchestrator::obrain::{Creation, ObrainGraphStore, MARQUEUR_PO, SCHEMA_PO};
use uuid::Uuid;

async fn open(path: &Path) -> ObrainGraphStore {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, Creation::Permise))
        .await
        .unwrap()
        .unwrap()
}

/// Write straight into a closed base, the way the migrator does.
async fn write_foreign(base: &Path, f: impl FnOnce(&obrain_engine::ObrainDB) + Send + 'static) {
    let base = base.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let db = obrain_engine::ObrainDB::open(&base).unwrap();
        f(&db);
        db.commit_wal().unwrap();
        db.close().unwrap();
        std::fs::write(base.join(MARQUEUR_PO), SCHEMA_PO).unwrap();
    })
    .await
    .unwrap();
}

/// The instant the migrator would have written, to the microsecond.
fn quand() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2021, 3, 4, 5, 6, 7).unwrap() + chrono::Duration::microseconds(123_456)
}

fn zoned(dt: chrono::DateTime<Utc>) -> Value {
    Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
        Timestamp::from_micros(dt.timestamp_micros()),
        0,
    ))
}

#[tokio::test]
async fn a_decision_migrated_with_a_zoned_datetime_keeps_its_instant() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    {
        // Creates the base and its schema marker.
        open(&base).await.close().await.unwrap();
    }
    let id = Uuid::new_v4();
    let attendu = quand();
    write_foreign(&base, move |db| {
        db.create_node_with_props(
            &["Decision"],
            vec![
                ("id", Value::from(id.to_string())),
                ("description", Value::from("migrated decision")),
                ("rationale", Value::from("written by the migrator")),
                ("decided_at", zoned(attendu)),
            ],
        );
    })
    .await;

    let store = open(&base).await;
    // Nothing to quarantine: the date is READABLE, it was only written in a
    // shape the old reader did not know.
    let quarantaine = &store.rapport_d_ouverture().en_quarantaine;
    assert!(
        quarantaine.is_empty(),
        "un nœud lisible a été mis en quarantaine : {quarantaine:?}"
    );
    let decision = store
        .get_decision(id)
        .await
        .unwrap()
        .expect("la décision migrée est lisible");
    assert_eq!(
        decision.decided_at.timestamp_micros(),
        attendu.timestamp_micros(),
        "decided_at a été remplacé par une autre date (le temps de la lecture ?)"
    );
}

/// A date nobody can read, written the way a foreign base may hold it.
fn illisible() -> Value {
    Value::from("pas une date")
}

/// An OPTIONAL date that is present but unreadable is read ABSENT (with a
/// warning): the project and the persona stay served instead of being quarantined.
#[tokio::test]
async fn an_unreadable_optional_date_is_read_absent_and_the_node_stays_served() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    open(&base).await.close().await.unwrap();
    let (pid, tid) = (Uuid::new_v4(), Uuid::new_v4());
    let cree = quand();
    write_foreign(&base, move |db| {
        db.create_node_with_props(
            &["Project"],
            vec![
                ("id", Value::from(pid.to_string())),
                ("name", Value::from("p")),
                ("slug", Value::from("p")),
                ("created_at", zoned(cree)),
                ("last_synced", illisible()),
            ],
        );
        db.create_node_with_props(
            &["Persona"],
            vec![
                ("id", Value::from(tid.to_string())),
                ("name", Value::from("pe")),
                ("status", Value::from("active")),
                ("energy", Value::from(0.5)),
                ("created_at", zoned(cree)),
                ("updated_at", illisible()),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    let quarantaine = &store.rapport_d_ouverture().en_quarantaine;
    assert!(
        quarantaine.is_empty(),
        "mis en quarantaine : {quarantaine:?}"
    );
    let p = store.get_project(pid).await.unwrap().expect("projet servi");
    assert!(p.last_synced.is_none());
    assert_eq!(p.created_at.timestamp_micros(), cree.timestamp_micros());
    let pe = store
        .get_persona(tid)
        .await
        .unwrap()
        .expect("persona servie");
    assert!(pe.updated_at.is_none());
}

/// A REQUIRED date unreadable stays a named quarantine (never replaced).
#[tokio::test]
async fn an_unreadable_required_date_is_still_quarantined() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    open(&base).await.close().await.unwrap();
    let pid = Uuid::new_v4();
    write_foreign(&base, move |db| {
        db.create_node_with_props(
            &["Project"],
            vec![
                ("id", Value::from(pid.to_string())),
                ("name", Value::from("p")),
                ("slug", Value::from("p")),
                ("created_at", illisible()),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
    assert!(store.get_project(pid).await.unwrap().is_none());
}

/// SECURITY: a signed tombstone whose `issued_at` is unreadable is KEPT and the
/// hash stays revoked (fail-closed). Quarantined, `is_tombstoned` answered an
/// error where Neo4j answers "revoked": a revocation lost to a bad date.
#[tokio::test]
async fn a_tombstone_with_an_unreadable_issued_at_stays_a_revocation() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    open(&base).await.close().await.unwrap();
    write_foreign(&base, move |db| {
        db.create_node_with_props(
            &["Tombstone"],
            vec![
                ("content_hash", Value::from("h-revoque")),
                ("issuer_did", Value::from("did:key:abc")),
                ("signature_hex", Value::from("ab12cd34")),
                ("issued_at", illisible()),
                ("reason", Value::from("")),
            ],
        );
    })
    .await;
    let store = open(&base).await;
    assert!(
        store.is_tombstoned("h-revoque").await.unwrap(),
        "la révocation a été perdue"
    );
    let tombes = store.list_tombstones().await.unwrap();
    assert_eq!(tombes.len(), 1);
    assert_eq!(tombes[0].issued_at, chrono::DateTime::UNIX_EPOCH);
}
