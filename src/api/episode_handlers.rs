//! API handlers for Episodic Memory operations.
//!
//! Provides endpoints:
//! - `POST /api/episodes/collect` — Collect an episode from a completed ProtocolRun
//! - `GET /api/episodes` — List episodes for a project
//! - `POST /api/episodes/anonymize` — Convert an episode to portable format
//! - `POST /api/episodes/export-artifact` — Export enriched artifact (structure + episodes)
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use crate::api::handlers::{AppError, OrchestratorState};
use crate::episodes::collector;
use crate::events::EventEmitter;

#[allow(unused_imports)]
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
    /// Which episodes: `run` (from protocol runs, the default and the only kind before
    /// session episodes existed), `session` (left by closed chat sessions) or `all`.
    pub source: Option<String>,
}

/// When an episode ended: a session's at its close (`collected_at`, written then), a
/// run's at its completion (start + duration), else when it started.
pub fn ended_at(e: &crate::episodes::Episode) -> chrono::DateTime<chrono::Utc> {
    if e.session.is_some() {
        return e.collected_at;
    }
    match e.process.duration_ms {
        Some(ms) => e.stimulus.timestamp + chrono::Duration::milliseconds(ms),
        None => e.stimulus.timestamp,
    }
}

/// The episodes `GET /api/episodes` lists (its `source` parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeSource {
    Run,
    Session,
    All,
}

impl EpisodeSource {
    /// `None` (absent) is `Run`: a caller that never asked keeps what it always had.
    pub fn parse(raw: Option<&str>) -> Result<Self, AppError> {
        match raw {
            None | Some("run") => Ok(Self::Run),
            Some("session") => Ok(Self::Session),
            Some("all") => Ok(Self::All),
            Some(other) => Err(AppError::BadRequest(format!(
                "unknown episode source `{other}`: expected run, session or all"
            ))),
        }
    }
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
    let source = EpisodeSource::parse(query.source.as_deref())?;
    let graph = state.orchestrator.neo4j();

    let mut episodes = Vec::new();
    if source != EpisodeSource::Session {
        episodes = collector::list_episodes(graph, query.project_id, limit)
            .await
            .map_err(AppError::Internal)?;
    }
    if source != EpisodeSource::Run {
        let sessions = graph
            .list_session_episodes(query.project_id, limit)
            .await
            .map_err(AppError::Internal)?;
        episodes.extend(sessions);
    }
    if source == EpisodeSource::All {
        // Both kinds, most recent END first: the key each source was cut on (runs by
        // completion, sessions by close), so the merge keeps the true latest `limit`.
        episodes.sort_by_key(|e| std::cmp::Reverse(ended_at(e)));
        episodes.truncate(limit);
    }

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
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use serde_json::{json, Value};

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

    /// `source=all` merges on the key each source was cut on: a session that started
    /// first but closed last is the most recent; a run counts from its completion.
    #[test]
    fn episodes_are_merged_by_when_they_ended() {
        use crate::episodes::collector::tests_support::episode_at;
        let t0 = chrono::Utc::now() - chrono::Duration::hours(10);
        // A session opened at t0, closed at t0+9h.
        let long_session = episode_at(t0, None, Some(t0 + chrono::Duration::hours(9)));
        // A session opened at t0+1h, closed at t0+2h.
        let short_session = episode_at(
            t0 + chrono::Duration::hours(1),
            None,
            Some(t0 + chrono::Duration::hours(2)),
        );
        // A run started at t0+3h that lasted 1h.
        let run = episode_at(t0 + chrono::Duration::hours(3), Some(3_600_000), None);
        let mut all = [short_session.clone(), run.clone(), long_session.clone()];
        all.sort_by_key(|e| std::cmp::Reverse(ended_at(e)));
        assert_eq!(
            all.iter().map(|e| e.id).collect::<Vec<_>>(),
            vec![long_session.id, run.id, short_session.id]
        );
    }

    #[test]
    fn an_absent_source_lists_run_episodes_as_before() {
        assert_eq!(EpisodeSource::parse(None).unwrap(), EpisodeSource::Run);
        assert_eq!(
            EpisodeSource::parse(Some("run")).unwrap(),
            EpisodeSource::Run
        );
        assert_eq!(
            EpisodeSource::parse(Some("session")).unwrap(),
            EpisodeSource::Session
        );
        assert_eq!(
            EpisodeSource::parse(Some("all")).unwrap(),
            EpisodeSource::All
        );
        assert!(matches!(
            EpisodeSource::parse(Some("sessions")),
            Err(AppError::BadRequest(_))
        ));
    }

    /// Through the router: the episode a closed session left is listed with
    /// `source=session` and `source=all`, never by default (run episodes only).
    #[tokio::test]
    async fn session_episodes_are_listed_only_when_asked_for() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let graph = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let project = crate::test_helpers::test_project_named("alpha");
        graph.create_project(&project).await.unwrap();
        let mut session = crate::test_helpers::test_chat_session(Some("alpha"));
        session.preview = Some("first message".into());
        session.message_count = 2;
        let sid = session.id;
        graph.create_chat_session(&session).await.unwrap();
        let episode = collector::collect_session_episode(graph.as_ref(), sid, "closed")
            .await
            .unwrap()
            .unwrap();
        assert!(graph.save_session_episode(&episode).await.unwrap());
        let app = crate::test_helpers::router_with_auth(graph.clone()).await;

        let list = |source: Option<&str>| {
            let mut uri = format!("/api/episodes?project_id={}", project.id);
            if let Some(s) = source {
                uri.push_str(&format!("&source={s}"));
            }
            let app = app.clone();
            async move {
                let resp = app
                    .oneshot(
                        Request::builder()
                            .uri(uri)
                            .header("authorization", crate::test_helpers::test_bearer_token())
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let status = resp.status();
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                    .await
                    .unwrap();
                (
                    status,
                    serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null),
                )
            }
        };

        let (status, body) = list(None).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["count"], 0, "default stays run episodes only: {body}");

        for source in ["session", "all"] {
            let (status, body) = list(Some(source)).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["count"], 1, "{source}: {body}");
            assert_eq!(
                body["episodes"][0]["session"]["session_id"],
                sid.to_string()
            );
            assert_eq!(body["episodes"][0]["stimulus"]["request"], "first message");
        }

        let (status, _) = list(Some("bogus")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}
