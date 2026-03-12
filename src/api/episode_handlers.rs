//! API handlers for Episodic Memory operations.
//!
//! Provides endpoints:
//! - `POST /api/episodes/collect` — Collect an episode from a completed ProtocolRun
//! - `GET /api/episodes` — List episodes for a project
//! - `POST /api/episodes/anonymize` — Convert an episode to portable format

use crate::api::handlers::{AppError, OrchestratorState};
use crate::episodes::collector;
use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Request / Response types
// ============================================================================

/// Request body for `POST /api/episodes/collect`.
#[derive(Debug, Deserialize)]
pub struct CollectEpisodeRequest {
    /// The ProtocolRun ID to collect an episode from.
    pub run_id: Uuid,
    /// The project ID.
    pub project_id: Uuid,
}

/// Response for `POST /api/episodes/collect`.
#[derive(Debug, Serialize)]
pub struct CollectEpisodeResponse {
    /// The collected episode (None if the run was not found).
    pub episode: Option<crate::episodes::Episode>,
}

/// Query parameters for `GET /api/episodes`.
#[derive(Debug, Deserialize)]
pub struct ListEpisodesQuery {
    /// The project ID to list episodes for.
    pub project_id: Uuid,
    /// Maximum number of episodes to return (default: 20).
    pub limit: Option<usize>,
}

/// Response for `GET /api/episodes`.
#[derive(Debug, Serialize)]
pub struct ListEpisodesResponse {
    /// The collected episodes.
    pub episodes: Vec<crate::episodes::Episode>,
    /// Total number returned.
    pub count: usize,
}

/// Request body for `POST /api/episodes/anonymize`.
#[derive(Debug, Deserialize)]
pub struct AnonymizeEpisodeRequest {
    /// The ProtocolRun ID to collect and anonymize.
    pub run_id: Uuid,
    /// The project ID.
    pub project_id: Uuid,
}

/// Response for `POST /api/episodes/anonymize`.
#[derive(Debug, Serialize)]
pub struct AnonymizeEpisodeResponse {
    /// The portable (anonymized) episode.
    pub portable_episode: Option<crate::episodes::PortableEpisode>,
}

// ============================================================================
// Handlers
// ============================================================================

/// Collect an episode from a completed ProtocolRun.
///
/// POST /api/episodes/collect
///
/// Queries the knowledge graph to assemble a complete Episode from:
/// - ProtocolRun (stimulus + states_visited)
/// - PRODUCED_DURING relations (outcome: notes + decisions)
/// - REASONING_FOR relation (process: reasoning tree)
/// - Run status (validation)
pub async fn collect_episode(
    State(state): State<OrchestratorState>,
    Json(body): Json<CollectEpisodeRequest>,
) -> Result<Json<CollectEpisodeResponse>, AppError> {
    let episode =
        collector::collect_episode(state.orchestrator.neo4j(), body.run_id, body.project_id)
            .await
            .map_err(AppError::Internal)?;

    Ok(Json(CollectEpisodeResponse { episode }))
}

/// List episodes for a project.
///
/// GET /api/episodes?project_id=...&limit=20
///
/// Finds completed ProtocolRuns linked to the project and collects each one.
pub async fn list_episodes(
    State(state): State<OrchestratorState>,
    Query(query): Query<ListEpisodesQuery>,
) -> Result<Json<ListEpisodesResponse>, AppError> {
    let limit = query.limit.unwrap_or(20).min(100);

    let episodes = collector::list_episodes(state.orchestrator.neo4j(), query.project_id, limit)
        .await
        .map_err(AppError::Internal)?;

    let count = episodes.len();
    Ok(Json(ListEpisodesResponse { episodes, count }))
}

/// Collect and anonymize an episode into portable format.
///
/// POST /api/episodes/anonymize
///
/// Collects the episode from the graph, then converts it to a
/// PortableEpisode (no UUIDs, no absolute paths).
pub async fn anonymize_episode(
    State(state): State<OrchestratorState>,
    Json(body): Json<AnonymizeEpisodeRequest>,
) -> Result<Json<AnonymizeEpisodeResponse>, AppError> {
    let episode =
        collector::collect_episode(state.orchestrator.neo4j(), body.run_id, body.project_id)
            .await
            .map_err(AppError::Internal)?;

    let portable_episode = episode.map(|ep| ep.to_portable());

    Ok(Json(AnonymizeEpisodeResponse { portable_episode }))
}
