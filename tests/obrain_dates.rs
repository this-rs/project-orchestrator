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
