//! API handlers for Episodic Memory operations.
//!
//! Provides endpoints:
//! - `POST /api/episodes/collect` — Collect an episode from a completed ProtocolRun
//! - `GET /api/episodes` — List episodes for a project
//! - `POST /api/episodes/anonymize` — Convert an episode to portable format
//! - `POST /api/episodes/export-artifact` — Export enriched artifact (structure + episodes)

use crate::api::handlers::{AppError, OrchestratorState};
use crate::episodes::collector;
use crate::events::EventEmitter;
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

    if episode.is_some() {
        state.event_bus.emit_created(
            crate::events::EntityType::Episode,
            &body.run_id.to_string(),
            serde_json::json!({
                "run_id": body.run_id.to_string(),
                "project_id": body.project_id.to_string(),
            }),
            Some(body.project_id.to_string()),
        );
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_collect_episode_request_deserialization() {
        let json = json!({
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "project_id": "660e8400-e29b-41d4-a716-446655440000"
        });
        let req: CollectEpisodeRequest = serde_json::from_value(json).unwrap();
        assert_eq!(
            req.run_id.to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
        assert_eq!(
            req.project_id.to_string(),
            "660e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[test]
    fn test_list_episodes_query_deserialization() {
        let json = json!({
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "limit": 50
        });
        let query: ListEpisodesQuery = serde_json::from_value(json).unwrap();
        assert_eq!(query.limit, Some(50));
    }

    #[test]
    fn test_list_episodes_query_default_limit() {
        let json = json!({
            "project_id": "550e8400-e29b-41d4-a716-446655440000"
        });
        let query: ListEpisodesQuery = serde_json::from_value(json).unwrap();
        assert!(query.limit.is_none());
    }

    #[test]
    fn test_anonymize_request_deserialization() {
        let json = json!({
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "project_id": "660e8400-e29b-41d4-a716-446655440000"
        });
        let req: AnonymizeEpisodeRequest = serde_json::from_value(json).unwrap();
        assert_eq!(
            req.run_id.to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[test]
    fn test_export_artifact_request_defaults() {
        let json = json!({"project_id": "550e8400-e29b-41d4-a716-446655440000"});
        let req: ExportArtifactRequest = serde_json::from_value(json).unwrap();
        assert!(req.include_structure); // default_true()
        assert!(req.max_episodes.is_none());
    }

    #[test]
    fn test_export_artifact_request_override_include_structure() {
        let json = json!({
            "project_id": "550e8400-e29b-41d4-a716-446655440000",
            "include_structure": false,
            "max_episodes": 10
        });
        let req: ExportArtifactRequest = serde_json::from_value(json).unwrap();
        assert!(!req.include_structure);
        assert_eq!(req.max_episodes, Some(10));
    }

    #[test]
    fn test_default_true_helper() {
        assert!(default_true());
    }

    #[test]
    fn test_artifact_edge_serialization_roundtrip() {
        let edge = ArtifactEdge {
            source: "src/auth/mod.rs".to_string(),
            target: "src/auth/google.rs".to_string(),
            relation: "CO_CHANGED".to_string(),
            weight: 5.0,
        };
        let json = serde_json::to_value(&edge).unwrap();
        assert_eq!(json["source"], "src/auth/mod.rs");
        assert_eq!(json["weight"], 5.0);

        let deserialized: ArtifactEdge = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.source, edge.source);
        assert_eq!(deserialized.weight, edge.weight);
    }

    #[test]
    fn test_artifact_stats_serialization() {
        let stats = ArtifactStats {
            edge_count: 42,
            episode_count: 10,
            episodes_with_lessons: 7,
            total_notes_produced: 25,
            total_decisions_made: 8,
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["edge_count"], 42);
        assert_eq!(json["episodes_with_lessons"], 7);
        assert_eq!(json["total_decisions_made"], 8);
    }

    #[test]
    fn test_enriched_artifact_empty_structure_skipped() {
        let artifact = EnrichedArtifact {
            schema_version: 1,
            exported_at: chrono::Utc::now(),
            source_project: "test-project".to_string(),
            structure: vec![],
            episodes: vec![],
            stats: ArtifactStats {
                edge_count: 0,
                episode_count: 0,
                episodes_with_lessons: 0,
                total_notes_produced: 0,
                total_decisions_made: 0,
            },
        };
        let json = serde_json::to_value(&artifact).unwrap();
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["source_project"], "test-project");
        // Empty structure should be skipped (skip_serializing_if = "Vec::is_empty")
        assert!(json.get("structure").is_none());
    }

    #[test]
    fn test_enriched_artifact_with_structure() {
        let artifact = EnrichedArtifact {
            schema_version: 1,
            exported_at: chrono::Utc::now(),
            source_project: "test-project".to_string(),
            structure: vec![ArtifactEdge {
                source: "a.rs".to_string(),
                target: "b.rs".to_string(),
                relation: "CO_CHANGED".to_string(),
                weight: 3.0,
            }],
            episodes: vec![],
            stats: ArtifactStats {
                edge_count: 1,
                episode_count: 0,
                episodes_with_lessons: 0,
                total_notes_produced: 0,
                total_decisions_made: 0,
            },
        };
        let json = serde_json::to_value(&artifact).unwrap();
        // Non-empty structure should be present
        assert!(json.get("structure").is_some());
        assert_eq!(json["structure"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_list_episodes_response_serialization() {
        let resp = ListEpisodesResponse {
            episodes: vec![],
            count: 0,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["count"], 0);
        assert!(json["episodes"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_collect_episode_response_serialization() {
        let resp = CollectEpisodeResponse { episode: None };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["episode"].is_null());
    }
}
