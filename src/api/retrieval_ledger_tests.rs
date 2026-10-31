//! `GET /api/retrieval-ledger` — le registre de récupération est LISIBLE.
//!
//! Sans lecteur, un registre en mémoire est un compteur que personne ne lit : la
//! PR1 du plan 42d49a31 existe pour rendre la récupération mesurable, pas pour
//! incrémenter des atomes.

use crate::notes::retrieval_ledger::{self, RetrievedItem, Source, Trigger};
use crate::test_helpers::{router_with_auth, test_bearer_token};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

fn get(uri: &str) -> Request<Body> {
    Request::get(uri)
        .header("authorization", test_bearer_token())
        .body(Body::empty())
        .unwrap()
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1_000_000)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

/// Le registre est global : on écrit sous une requête unique, puis on la retrouve
/// dans la réponse. Une assertion par delta serait brouillée par les autres tests.
// Le registre est global : ce test le sérialise et agrandit le journal, pour
// qu'aucun événement voisin n'évince celui qu'il vient d'écrire. Le garde est
// tenu à travers l'appel observé : c'est son rôle.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn le_registre_est_lisible_et_porte_les_sept_champs() {
    let _serial = retrieval_ledger::serial_guard();
    let saved_capacity = retrieval_ledger::journal_capacity();
    retrieval_ledger::set_journal_capacity(4096);

    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let app = router_with_auth(graph).await;

    let query = format!("lisible-{}", uuid::Uuid::new_v4());
    let id = retrieval_ledger::record(
        Source::Semantic,
        Trigger::Uncertainty,
        &query,
        vec![RetrievedItem::new("note-x", "note", Some(0.77))],
        true,
        9,
    )
    .expect("le journal accepte l'événement");

    let response = app
        .oneshot(get("/api/retrieval-ledger?limit=200"))
        .await
        .unwrap();
    retrieval_ledger::set_journal_capacity(saved_capacity);
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;

    // L'agrégat.
    assert!(
        body["snapshot"]["calls"].as_u64().unwrap_or(0) >= 1,
        "l'appel est compté : {body}"
    );
    assert!(body["snapshot"]["dropped"].is_u64(), "le manque est exposé");
    assert!(body["snapshot"]["uncertainty_calls"].is_u64());

    // Les ratios : présents, et `null` plutôt que zéro quand ils n'ont pas de sens.
    assert!(
        body["derived"]["cost_per_correct"].is_null()
            || body["derived"]["cost_per_correct"].is_f64(),
        "cost_per_correct est un nombre ou null, jamais 0 par défaut"
    );

    // L'événement, avec la requête exacte, l'identifiant et le score.
    let events = body["recent"].as_array().expect("une liste d'événements");
    let mine = events
        .iter()
        .find(|e| e["id"].as_u64() == Some(id))
        .unwrap_or_else(|| panic!("l'événement {id} doit être dans la réponse : {body}"));
    assert_eq!(mine["query"], query);
    assert_eq!(mine["trigger"], "uncertainty");
    assert_eq!(mine["source"], "semantic");
    assert_eq!(mine["results"][0]["id"], "note-x");
    assert_eq!(mine["results"][0]["score"], 0.77);
    assert!(
        mine["usage"].is_null(),
        "le tour qui récupère ne peut pas savoir ce qu'il en fera"
    );
}

/// `limit` est borné : la route ne peut pas être transformée en extraction de
/// tout le journal par un paramètre.
#[tokio::test]
async fn la_route_borne_le_nombre_d_evenements() {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let app = router_with_auth(graph).await;
    let response = app
        .oneshot(get("/api/retrieval-ledger?limit=10000"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert!(
        body["recent"].as_array().unwrap().len() <= 200,
        "le plafond de 200 s'applique"
    );
}
