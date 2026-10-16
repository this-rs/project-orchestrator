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
//!    `tests/harness_neo4j.rs`; `NEURAL_NEO4J_REQUIRED=1` turns the skip into
//!    a failure.
//!
//!   docker run -d --rm --name po-nn -p 17687:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
//!   NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword cargo test --test neural_routing_indexes

use std::sync::Arc;

use neural_routing_runtime::{Neo4jTrajectoryStore, Trajectory, TrajectoryNode, TrajectoryStore};
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
    let store = include_str!("../crates/neural-routing-core/src/store.rs");
    assert!(
        store.contains("trajectory_query_embedding_idx"),
        "the index the router queries must be the one ensure_indexes creates"
    );
}

// ===========================================================================
// Behavioural test against a real Neo4j
// ===========================================================================

async fn store() -> Option<Arc<Neo4jTrajectoryStore>> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());

    match neo4rs::Graph::new(&uri, &user, &password).await {
        Ok(graph) => Some(Arc::new(Neo4jTrajectoryStore::new(Arc::new(graph)))),
        Err(err) => {
            assert!(
                std::env::var("NEURAL_NEO4J_REQUIRED").is_err(),
                "NEURAL_NEO4J_REQUIRED is set but no Neo4j answers at {uri}: {err}"
            );
            eprintln!("Skipping neural-routing index tests: no Neo4j at {uri}");
            None
        }
    }
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
