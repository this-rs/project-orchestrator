//! API handlers for trajectory storage and retrieval.
//!
//! Endpoints:
//! - GET  /api/trajectories          — list with filters
//! - GET  /api/trajectories/stats    — statistics
//! - GET  /api/trajectories/:id      — get by ID with nodes
//! - POST /api/trajectories/similar  — vector similarity search

use super::handlers::{AppError, OrchestratorState};
use axum::{
    extract::{Path, Query, State},
    Json,
};
use neural_routing_runtime::{Trajectory, TrajectoryFilter, TrajectoryStats};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Request / Response types
// ============================================================================

/// Query parameters for GET /api/trajectories
#[derive(Debug, Deserialize)]
pub struct ListTrajectoriesQuery {
    pub session_id: Option<String>,
    pub min_reward: Option<f64>,
    pub max_reward: Option<f64>,
    pub min_steps: Option<usize>,
    pub max_steps: Option<usize>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Request body for POST /api/trajectories/similar
#[derive(Debug, Deserialize)]
pub struct SimilarSearchRequest {
    /// Query embedding (256d).
    pub embedding: Vec<f32>,
    /// Number of results to return (default: 10).
    pub top_k: Option<usize>,
    /// Minimum cosine similarity threshold (default: 0.7).
    pub min_similarity: Option<f32>,
}

/// Response for similarity search
#[derive(Debug, Serialize)]
pub struct SimilarResult {
    pub trajectory: Trajectory,
    pub similarity: f64,
}

/// Wrapper for list response
#[derive(Debug, Serialize)]
pub struct TrajectoryListResponse {
    pub trajectories: Vec<Trajectory>,
    pub count: usize,
}

/// Wrapper for single trajectory
#[derive(Debug, Serialize)]
pub struct TrajectoryDetailResponse {
    pub trajectory: Trajectory,
}

/// Wrapper for stats
#[derive(Debug, Serialize)]
pub struct TrajectoryStatsResponse {
    pub stats: TrajectoryStats,
}

/// Wrapper for similar results
#[derive(Debug, Serialize)]
pub struct SimilarSearchResponse {
    pub results: Vec<SimilarResult>,
    pub count: usize,
}

// ============================================================================
// Helper
// ============================================================================

fn get_store(
    state: &OrchestratorState,
) -> Result<&dyn neural_routing_runtime::TrajectoryStore, AppError> {
    state
        .trajectory_store
        .as_ref()
        .map(|s| s.as_ref())
        .ok_or_else(|| {
            AppError::BadRequest(
                "Trajectory store not available (neural routing may be disabled)".to_string(),
            )
        })
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /api/trajectories
///
/// List trajectories with optional filters.
pub async fn list_trajectories(
    State(state): State<OrchestratorState>,
    Query(query): Query<ListTrajectoriesQuery>,
) -> Result<Json<TrajectoryListResponse>, AppError> {
    let store = get_store(&state)?;

    let filter = TrajectoryFilter {
        session_id: query.session_id,
        min_reward: query.min_reward,
        max_reward: query.max_reward,
        from_date: None,
        to_date: None,
        min_steps: query.min_steps,
        max_steps: query.max_steps,
        limit: Some(query.limit.unwrap_or(50).min(200)),
        offset: query.offset,
    };

    let trajectories = store
        .list_trajectories(&filter)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to list trajectories: {e}")))?;

    let count = trajectories.len();
    Ok(Json(TrajectoryListResponse {
        trajectories,
        count,
    }))
}

/// GET /api/trajectories/stats
///
/// Get trajectory statistics.
pub async fn get_stats(
    State(state): State<OrchestratorState>,
) -> Result<Json<TrajectoryStatsResponse>, AppError> {
    let store = get_store(&state)?;

    let stats = store
        .get_stats()
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to get trajectory stats: {e}")))?;

    Ok(Json(TrajectoryStatsResponse { stats }))
}

/// GET /api/trajectories/:id
///
/// Get a single trajectory with all its nodes.
pub async fn get_trajectory(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<Json<TrajectoryDetailResponse>, AppError> {
    let store = get_store(&state)?;

    let trajectory = store
        .get_trajectory(&id)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to get trajectory: {e}")))?
        .ok_or_else(|| AppError::NotFound(format!("Trajectory {id} not found")))?;

    Ok(Json(TrajectoryDetailResponse { trajectory }))
}

/// POST /api/trajectories/similar
///
/// Find trajectories similar to a given embedding.
pub async fn search_similar(
    State(state): State<OrchestratorState>,
    Json(req): Json<SimilarSearchRequest>,
) -> Result<Json<SimilarSearchResponse>, AppError> {
    let store = get_store(&state)?;

    // Validate embedding dimension (must be 256d)
    if req.embedding.len() != neural_routing_runtime::TOTAL_DIM {
        return Err(AppError::BadRequest(format!(
            "Embedding must be {}d, got {}d",
            neural_routing_runtime::TOTAL_DIM,
            req.embedding.len()
        )));
    }

    let top_k = req.top_k.unwrap_or(10).min(100);
    let min_similarity = req.min_similarity.unwrap_or(0.7);

    let results = store
        .search_similar(&req.embedding, top_k, min_similarity)
        .await
        .map_err(|e| {
            AppError::Internal(anyhow::anyhow!(
                "Failed to search similar trajectories: {e}"
            ))
        })?;

    let count = results.len();
    let results: Vec<SimilarResult> = results
        .into_iter()
        .map(|(trajectory, similarity)| SimilarResult {
            trajectory,
            similarity,
        })
        .collect();

    Ok(Json(SimilarSearchResponse { results, count }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, mock_neural_router, mock_trajectory_store};
    use std::sync::Arc;

    /// Build a minimal `OrchestratorState` with trajectory_store = None.
    async fn test_state_no_store() -> OrchestratorState {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 0,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: None,
        })
    }

    /// Build a minimal `OrchestratorState` with a NoopStore trajectory store.
    async fn test_state_with_store() -> OrchestratorState {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 0,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: Some(mock_trajectory_store()),
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: None,
        })
    }

    #[tokio::test]
    async fn test_get_store_returns_error_when_none() {
        let state = test_state_no_store().await;
        let result = get_store(&state);
        assert!(result.is_err());
        match result {
            Err(AppError::BadRequest(msg)) => {
                assert!(msg.contains("Trajectory store not available"));
            }
            Err(other) => panic!("Expected BadRequest, got {:?}", other),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[tokio::test]
    async fn test_get_store_succeeds_with_store() {
        let state = test_state_with_store().await;
        let result = get_store(&state);
        assert!(
            result.is_ok(),
            "get_store should succeed when store is present"
        );
    }

    #[tokio::test]
    async fn test_list_trajectories_default_params() {
        let state = test_state_with_store().await;

        // Call with default/empty query params
        let query = ListTrajectoriesQuery {
            session_id: None,
            min_reward: None,
            max_reward: None,
            min_steps: None,
            max_steps: None,
            limit: None,
            offset: None,
        };

        let result = list_trajectories(State(state), Query(query)).await;
        assert!(
            result.is_ok(),
            "list_trajectories should succeed with default params"
        );

        let resp = result.unwrap().0;
        // NoopStore returns empty vec
        assert_eq!(resp.count, 0);
        assert!(resp.trajectories.is_empty());
    }

    #[tokio::test]
    async fn test_list_trajectories_no_store() {
        let state = test_state_no_store().await;

        let query = ListTrajectoriesQuery {
            session_id: None,
            min_reward: None,
            max_reward: None,
            min_steps: None,
            max_steps: None,
            limit: Some(10),
            offset: Some(0),
        };

        let result = list_trajectories(State(state), Query(query)).await;
        assert!(result.is_err(), "should fail when store is None");
    }

    #[tokio::test]
    async fn test_search_similar_validates_embedding_dim() {
        let state = test_state_with_store().await;

        // Send a wrong-dimension embedding (e.g. 10d instead of TOTAL_DIM=256)
        let req = SimilarSearchRequest {
            embedding: vec![0.0f32; 10],
            top_k: None,
            min_similarity: None,
        };

        let result = search_similar(State(state), Json(req)).await;
        assert!(result.is_err(), "wrong dimension should return an error");

        match result.unwrap_err() {
            AppError::BadRequest(msg) => {
                assert!(
                    msg.contains("256d")
                        || msg.contains(&neural_routing_runtime::TOTAL_DIM.to_string())
                );
                assert!(msg.contains("10d"));
            }
            other => panic!("Expected BadRequest, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_search_similar_correct_dim() {
        let state = test_state_with_store().await;

        let req = SimilarSearchRequest {
            embedding: vec![0.0f32; neural_routing_runtime::TOTAL_DIM],
            top_k: Some(5),
            min_similarity: Some(0.5),
        };

        let result = search_similar(State(state), Json(req)).await;
        assert!(result.is_ok(), "correct dimension should succeed");

        let resp = result.unwrap().0;
        // NoopStore returns empty results
        assert_eq!(resp.count, 0);
        assert!(resp.results.is_empty());
    }

    #[tokio::test]
    async fn test_get_stats_with_store() {
        let state = test_state_with_store().await;

        let result = get_stats(State(state)).await;
        assert!(
            result.is_ok(),
            "get_stats should succeed with store present"
        );

        let resp = result.unwrap().0;
        assert_eq!(resp.stats.total_count, 0);
    }

    #[tokio::test]
    async fn test_get_stats_no_store() {
        let state = test_state_no_store().await;

        let result = get_stats(State(state)).await;
        assert!(result.is_err(), "get_stats should fail when store is None");
    }
}
