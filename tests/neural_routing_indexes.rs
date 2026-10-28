//! The NN router answers a query with
//! `CALL db.index.vector.queryNodes('trajectory_query_embedding_idx', …)` on
//! `Trajectory.query_embedding`. A Neo4j VECTOR index is not implicit: if
//! nothing creates it, every call throws
//! `There is no such vector schema index: trajectory_query_embedding_idx`,
//! `NNRouter::route_internal` returns early, and the dashboard shows
//! `hits: 0` while `total_queries` climbs — the exact state measured on
//! 17/10/2026 (290 trajectories stored, index absent, `hits` 0).
//!
//! Two tests:
//! 1. the startup path creates the indexes (offline, deterministic — same
//!    source-reading idiom as `src/boot.rs`'s phase test);
//! 2. against a real Neo4j, a stored trajectory is found above the 0.7
//!    threshold once the indexes exist. Skipped when no Neo4j answers, like
//!    `tests/harness_neo4j.rs`; `HARNESS_NEO4J_REQUIRED=1` — the switch the
//!    integration-tests CI job already exports for every Neo4j suite — turns
//!    that skip into a failure.
//!
//!   docker run -d --rm --name po-nn -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test neural_routing_indexes

use std::sync::Arc;
use std::time::Duration;

use neural_routing_runtime::{Trajectory, TrajectoryNode, TrajectoryStore};
use project_orchestrator::neo4j::trajectory_store::Neo4jTrajectoryStore;
use uuid::Uuid;

/// The startup path must create the vector index the router queries.
///
/// Offline and deterministic: `AppState::new` needs live databases, so what is
/// checked is that the code says so.
#[test]
fn startup_creates_the_vector_index_the_nn_router_queries() {
    let lib = include_str!("../src/lib.rs");
    assert!(
        lib.contains("ensure_indexes"),
        "the startup path never calls Neo4jTrajectoryStore::ensure_indexes: the NN router's \
         `db.index.vector.queryNodes('trajectory_query_embedding_idx', …)` then throws \
         `There is no such vector schema index`, every route is a miss and `hits` stays 0 forever"
    );

    // Tie the two halves together: a rename of the index can never silently
    // disconnect the query from its creation.
    let store = include_str!("../src/neo4j/trajectory_store.rs");
    assert!(
        store.contains("trajectory_query_embedding_idx"),
        "the index the router queries must be the one ensure_indexes creates"
    );
}

// ===========================================================================
// Behavioural test against a real Neo4j
// ===========================================================================

/// Whether something is listening on the Bolt port of `uri`.
///
/// Deliberately a socket connect and NOT a Cypher round-trip: every query of
/// `neo4rs` is wrapped in a retrying backoff (`Graph::run` / `Graph::execute`
/// read "Retries happen with an exponential backoff until a retry delay exceeds
/// 60s"), so a probe that asks the DATABASE does not fail fast when nothing
/// answers — it sits in the retry loop. Measured on 17/10/2026 with the port
/// closed: still retrying after 444 s, where this socket probe decides in
/// milliseconds. A gate that costs minutes to say "skip" is not a gate.
/// The socket answers the only question this helper asks: is anyone there?
async fn bolt_reachable(uri: &str) -> bool {
    let authority = uri
        .split_once("://")
        .map_or(uri, |(_, rest)| rest)
        .split(['/', '?'])
        .next()
        .unwrap_or_default();
    tokio::time::timeout(
        Duration::from_secs(2),
        tokio::net::TcpStream::connect(authority),
    )
    .await
    .is_ok_and(|connected| connected.is_ok())
}

/// Connect to the throwaway Neo4j named by the environment, or `None` when none
/// answers — the same contract as `tests/harness_neo4j.rs` (skip by default,
/// hard failure under `HARNESS_NEO4J_REQUIRED`, the switch the integration-tests
/// CI job exports for every Neo4j suite).
async fn store() -> Option<Arc<Neo4jTrajectoryStore>> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());

    let connected = if bolt_reachable(&uri).await {
        neo4rs::Graph::new(&uri, &user, &password)
            .await
            .ok()
            .map(|graph| Arc::new(Neo4jTrajectoryStore::new(Arc::new(graph))))
    } else {
        None
    };

    if connected.is_none() {
        assert!(
            std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
            "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}"
        );
        eprintln!("Skipping neural-routing index tests: no Neo4j at {uri}");
    }
    connected
}

/// A 256d unit vector (the index is declared with `vector.dimensions: 256`).
fn unit_embedding() -> Vec<f32> {
    let v = 1.0 / (256.0f32).sqrt();
    vec![v; 256]
}

#[tokio::test]
async fn a_stored_trajectory_is_found_once_the_index_exists() {
    let Some(store) = store().await else { return };

    // The production startup step: without it `search_similar` errors out.
    store
        .ensure_indexes()
        .await
        .expect("ensure_indexes must succeed against a writable Neo4j");

    let embedding = unit_embedding();
    let id = Uuid::new_v4();
    let session = format!("nn-index-test-{id}");
    let trajectory = Trajectory {
        id,
        session_id: session.clone(),
        query_embedding: embedding.clone(),
        total_reward: 0.8,
        step_count: 1,
        duration_ms: 10,
        nodes: vec![TrajectoryNode {
            id: Uuid::new_v4(),
            context_embedding: embedding.clone(),
            action_type: "routing.select_sections".to_string(),
            action_params: serde_json::Value::Null,
            alternatives_count: 3,
            chosen_index: 0,
            confidence: 0.9,
            local_reward: 0.8,
            cumulative_reward: 0.8,
            delta_ms: 10,
            order: 0,
        }],
        created_at: chrono::Utc::now(),
        protocol_run_id: None,
    };

    store.store_trajectory(&trajectory).await.expect("store");

    // The index may still be back-filling; retry briefly rather than flake.
    let mut found = false;
    for _ in 0..20 {
        let hits = store
            .search_similar(&embedding, 5, 0.7)
            .await
            .expect("search_similar must not error once the index exists");
        if hits.iter().any(|(t, _)| t.id == id) {
            found = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let _ = store.delete_trajectory(&id).await;
    assert!(
        found,
        "a trajectory stored with the same 256d embedding must be returned by \
         search_similar(min_similarity = 0.7)"
    );
}
