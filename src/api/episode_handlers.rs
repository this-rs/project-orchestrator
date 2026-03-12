//! API handlers for Episodic Memory operations.
//!
//! Provides endpoints:
//! - `POST /api/episodes/collect` — Collect an episode from a completed ProtocolRun
//! - `GET /api/episodes` — List episodes for a project
//! - `POST /api/episodes/anonymize` — Convert an episode to portable format
//! - `POST /api/episodes/export-artifact` — Export enriched artifact (structure + episodes)

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

// ============================================================================
// Enriched Artifact Export
// ============================================================================

/// Request body for `POST /api/episodes/export-artifact`.
#[derive(Debug, Deserialize)]
pub struct ExportArtifactRequest {
    /// The project ID to export an artifact for.
    pub project_id: Uuid,
    /// Maximum number of episodes to include (default: 50).
    pub max_episodes: Option<usize>,
    /// Whether to include structural edges (co-change, SYNAPSE weights).
    #[serde(default = "default_true")]
    pub include_structure: bool,
}

fn default_true() -> bool {
    true
}

/// A structural edge in the exported artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub weight: f64,
}

/// The enriched artifact — hybrid structure + episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichedArtifact {
    /// Schema version.
    pub schema_version: u32,
    /// When the artifact was exported.
    pub exported_at: chrono::DateTime<chrono::Utc>,
    /// Source project slug.
    pub source_project: String,
    /// Structural edges (co-change, SYNAPSE).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub structure: Vec<ArtifactEdge>,
    /// Portable episodes.
    pub episodes: Vec<crate::episodes::PortableEpisode>,
    /// Stats.
    pub stats: ArtifactStats,
}

/// Stats about the exported artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStats {
    pub edge_count: usize,
    pub episode_count: usize,
    pub episodes_with_lessons: usize,
    pub total_notes_produced: usize,
    pub total_decisions_made: usize,
}

/// Export an enriched artifact for a project.
///
/// POST /api/episodes/export-artifact
///
/// Produces a JSON artifact combining:
/// 1. Structural edges (co-change graph, SYNAPSE weights)
/// 2. Portable episodes (anonymized cognitive episodes)
///
/// This is the M2 deliverable — proving episodic enrichment adds value.
pub async fn export_artifact(
    State(state): State<OrchestratorState>,
    Json(body): Json<ExportArtifactRequest>,
) -> Result<Json<EnrichedArtifact>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let max_episodes = body.max_episodes.unwrap_or(50).min(200);

    // 1. Get project info
    let project = neo4j
        .get_project(body.project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound("Project not found".to_string()))?;

    // 2. Collect episodes and convert to portable
    let episodes = collector::list_episodes(neo4j, body.project_id, max_episodes)
        .await
        .map_err(AppError::Internal)?;

    let portable_episodes: Vec<crate::episodes::PortableEpisode> =
        episodes.iter().map(|ep| ep.to_portable()).collect();

    // 3. Collect structural edges if requested
    let mut structure = Vec::new();
    if body.include_structure {
        // Co-change edges (files that change together)
        if let Ok(co_changes) = neo4j.get_co_change_graph(body.project_id, 2, 100).await {
            for pair in co_changes {
                structure.push(ArtifactEdge {
                    source: pair.file_a,
                    target: pair.file_b,
                    relation: "CO_CHANGED".to_string(),
                    weight: pair.count as f64,
                });
            }
        }
    }

    // 4. Compute stats
    let episodes_with_lessons = portable_episodes
        .iter()
        .filter(|ep| ep.lesson.is_some())
        .count();
    let total_notes_produced: usize = portable_episodes
        .iter()
        .map(|ep| ep.outcome.notes_produced)
        .sum();
    let total_decisions_made: usize = portable_episodes
        .iter()
        .map(|ep| ep.outcome.decisions_made)
        .sum();

    let edge_count = structure.len();
    let episode_count = episodes.len();

    let artifact = EnrichedArtifact {
        schema_version: 1,
        exported_at: chrono::Utc::now(),
        source_project: project.slug,
        structure,
        episodes: portable_episodes,
        stats: ArtifactStats {
            edge_count,
            episode_count,
            episodes_with_lessons,
            total_notes_produced,
            total_decisions_made,
        },
    };

    Ok(Json(artifact))
}
