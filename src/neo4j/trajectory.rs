//! Trajectory store for the neural-routing domain.
//!
//! It keeps its own connection pool, so everything that depends on the driver
//! lives here rather than in the composition root: the adapter owns its
//! connections and the schema it needs.

use std::sync::Arc;

use super::trajectory_store::Neo4jTrajectoryStore;
use anyhow::Result;

/// Opens the trajectory store on the configured server and makes sure the
/// vector indexes it queries actually exist.
///
/// The NN router answers with `CALL db.index.vector.queryNodes(
/// 'trajectory_query_embedding_idx', …)`; a VECTOR index is not implicit.
/// Without it the call throws `There is no such vector schema index`,
/// `NNRouter::route_internal` returns early, and the dashboard reads `hits: 0`
/// while `total_queries` climbs — the state measured on 17/10/2026 (290
/// trajectories, no index). `IF NOT EXISTS` makes it idempotent and back-fills
/// what is already stored.
pub async fn open(uri: &str, user: &str, password: &str) -> Result<Arc<Neo4jTrajectoryStore>> {
    let graph = Arc::new(neo4rs::Graph::new(uri, user, password).await?);
    let store = Arc::new(Neo4jTrajectoryStore::new(graph));

    if let Err(e) = store.ensure_indexes().await {
        tracing::warn!(
            error = %e,
            "neural-routing vector indexes not ensured — the NN router will miss on every \
             query (hits stays 0)"
        );
    }

    Ok(store)
}
