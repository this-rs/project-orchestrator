//! Project API handlers

use crate::api::{PaginatedResponse, PaginationParams, SearchFilter};
use crate::neo4j::models::ProjectNode;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::expand_tilde;

use super::graph_types::{
    GraphQuery, ProjectGraphResponse, IntelligenceSummaryResponse, parse_layers,
};
use super::handlers::{AppError, OrchestratorState};

// ============================================================================
// Request/Response types
// ============================================================================

#[derive(Deserialize)]
pub struct CreateProjectRequest {
    pub name: String,
    pub slug: Option<String>,
    pub root_path: String,
    pub description: Option<String>,
}

#[derive(Serialize)]
pub struct ProjectResponse {
    pub id: String,
    pub name: String,
    pub slug: String,
    pub root_path: String,
    pub description: Option<String>,
    pub created_at: String,
    pub last_synced: Option<String>,
    pub file_count: usize,
    pub plan_count: usize,
}

#[derive(Serialize)]
pub struct ProjectListResponse {
    pub projects: Vec<ProjectResponse>,
    pub total: usize,
}

// ============================================================================
// Handlers
// ============================================================================

/// Query parameters for listing projects
#[derive(Debug, Deserialize, Default)]
pub struct ProjectsListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub search_filter: SearchFilter,
}

/// List all projects with optional pagination and search
pub async fn list_projects(
    State(state): State<OrchestratorState>,
    Query(query): Query<ProjectsListQuery>,
) -> Result<Json<PaginatedResponse<ProjectResponse>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let (projects, total) = state
        .orchestrator
        .neo4j()
        .list_projects_filtered(
            query.search_filter.search.as_deref(),
            query.pagination.validated_limit(),
            query.pagination.offset,
            query.pagination.sort_by.as_deref(),
            &query.pagination.sort_order,
        )
        .await?;

    let mut responses = Vec::new();
    for project in &projects {
        let file_count = state
            .orchestrator
            .neo4j()
            .count_project_files(project.id)
            .await
            .unwrap_or(0);
        let plan_count = state
            .orchestrator
            .neo4j()
            .count_project_plans(project.id)
            .await
            .unwrap_or(0);

        responses.push(ProjectResponse {
            id: project.id.to_string(),
            name: project.name.clone(),
            slug: project.slug.clone(),
            root_path: project.root_path.clone(),
            description: project.description.clone(),
            created_at: project.created_at.to_rfc3339(),
            last_synced: project.last_synced.map(|dt| dt.to_rfc3339()),
            file_count: file_count as usize,
            plan_count: plan_count as usize,
        });
    }

    Ok(Json(PaginatedResponse::new(
        responses,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// Create a new project
pub async fn create_project(
    State(state): State<OrchestratorState>,
    Json(req): Json<CreateProjectRequest>,
) -> Result<Json<ProjectResponse>, AppError> {
    // Generate slug from name if not provided
    let slug = req.slug.unwrap_or_else(|| slugify(&req.name));

    // Check if slug already exists
    if state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .is_some()
    {
        return Err(AppError::BadRequest(format!(
            "Project with slug '{}' already exists",
            slug
        )));
    }

    let project = ProjectNode {
        id: Uuid::new_v4(),
        name: req.name,
        slug: slug.clone(),
        root_path: expand_tilde(&req.root_path),
        description: req.description,
        created_at: chrono::Utc::now(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
    };

    state.orchestrator.create_project(&project).await?;
    // Auto-registration on the file watcher is handled by the ProjectWatcherBridge
    // which listens to CrudEvent::Created events emitted by the orchestrator.

    Ok(Json(ProjectResponse {
        id: project.id.to_string(),
        name: project.name,
        slug: project.slug,
        root_path: project.root_path,
        description: project.description,
        created_at: project.created_at.to_rfc3339(),
        last_synced: None,
        file_count: 0,
        plan_count: 0,
    }))
}

/// Get a project by slug
pub async fn get_project(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<ProjectResponse>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    let file_count = state
        .orchestrator
        .neo4j()
        .count_project_files(project.id)
        .await
        .unwrap_or(0);
    let plan_count = state
        .orchestrator
        .neo4j()
        .count_project_plans(project.id)
        .await
        .unwrap_or(0);

    Ok(Json(ProjectResponse {
        id: project.id.to_string(),
        name: project.name,
        slug: project.slug,
        root_path: project.root_path,
        description: project.description,
        created_at: project.created_at.to_rfc3339(),
        last_synced: project.last_synced.map(|dt| dt.to_rfc3339()),
        file_count: file_count as usize,
        plan_count: plan_count as usize,
    }))
}

/// Request to update a project
#[derive(Deserialize)]
pub struct UpdateProjectRequest {
    pub name: Option<String>,
    pub description: Option<Option<String>>,
    pub root_path: Option<String>,
}

/// Update a project
pub async fn update_project(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(req): Json<UpdateProjectRequest>,
) -> Result<StatusCode, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    // Use orchestrator.update_project() instead of neo4j() directly so that
    // a CrudEvent::Updated is emitted — the ProjectWatcherBridge listens for
    // root_path changes to re-register the project on the file watcher.
    state
        .orchestrator
        .update_project(project.id, req.name, req.description, req.root_path)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Delete a project
pub async fn delete_project(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<StatusCode, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    // Auto-unregistration from the file watcher is handled by the ProjectWatcherBridge
    // which listens to CrudEvent::Deleted events emitted by the orchestrator.
    state.orchestrator.delete_project(project.id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Sync a project's codebase
#[derive(Serialize)]
pub struct SyncProjectResponse {
    pub files_synced: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub symbols_deleted: usize,
    pub errors: usize,
}

/// Query parameters for sync_project
#[derive(Debug, Deserialize, Default)]
pub struct SyncProjectQuery {
    pub force: Option<bool>,
}

pub async fn sync_project(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Query(query): Query<SyncProjectQuery>,
) -> Result<Json<SyncProjectResponse>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    let force = query.force.unwrap_or(false);
    let expanded = expand_tilde(&project.root_path);
    let path = std::path::Path::new(&expanded);
    let result = state
        .orchestrator
        .sync_directory_for_project_with_options(path, Some(project.id), Some(&project.slug), force)
        .await?;

    // Update last_synced timestamp
    state
        .orchestrator
        .neo4j()
        .update_project_synced(project.id)
        .await?;

    // Compute graph analytics (PageRank, communities, etc.) — best-effort, background
    state.orchestrator.spawn_analyze_project(project.id);

    // Refresh auto-built feature graphs in background (best-effort)
    state.orchestrator.spawn_refresh_feature_graphs(project.id);

    // Spawn event-triggered protocol runs (post_sync)
    crate::protocol::hooks::spawn_event_triggered_protocols(
        state.orchestrator.neo4j_arc(),
        project.id,
        "post_sync",
    );

    Ok(Json(SyncProjectResponse {
        files_synced: result.files_synced,
        files_skipped: result.files_skipped,
        files_deleted: result.files_deleted,
        symbols_deleted: result.symbols_deleted,
        errors: result.errors,
    }))
}

/// Query parameters for list_project_plans
#[derive(Debug, Deserialize, Default)]
pub struct ProjectPlansQuery {
    pub status: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// List plans for a project with optional status filter and pagination
pub async fn list_project_plans(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Query(query): Query<ProjectPlansQuery>,
) -> Result<Json<PaginatedResponse<crate::neo4j::models::PlanNode>>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    let limit = query.limit.unwrap_or(50);
    let offset = query.offset.unwrap_or(0);
    let status_filter: Option<Vec<String>> = query
        .status
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());

    let (plans, total) = state
        .orchestrator
        .neo4j()
        .list_plans_for_project(project.id, status_filter, limit, offset)
        .await?;

    Ok(Json(PaginatedResponse::new(plans, total, limit, offset)))
}

/// Search code in a project
#[derive(Deserialize)]
pub struct ProjectCodeSearchQuery {
    pub q: String,
    pub limit: Option<usize>,
    pub language: Option<String>,
}

pub async fn search_project_code(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    axum::extract::Query(query): axum::extract::Query<ProjectCodeSearchQuery>,
) -> Result<Json<Vec<crate::meilisearch::indexes::CodeDocument>>, AppError> {
    // Verify project exists
    let _project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    let results = state
        .orchestrator
        .meili()
        .search_code_in_project(
            &query.q,
            query.limit.unwrap_or(10),
            query.language.as_deref(),
            Some(&slug),
        )
        .await?;

    Ok(Json(results))
}

// ============================================================================
// Graph Visualization Endpoint
// ============================================================================

/// GET /api/projects/:slug/graph — Multi-layer graph export for visualization
///
/// Returns a unified {nodes, edges, communities, stats} response.
/// Delegates to `build_project_graph_data()` in graph_types.rs.
pub async fn get_project_graph(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Query(params): Query<GraphQuery>,
) -> Result<Json<ProjectGraphResponse>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    let project = neo4j
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?;

    let requested_layers = parse_layers(&params.layers);
    let limit = params.limit.unwrap_or(5000);

    let (nodes, edges, communities, stats) = super::graph_types::build_project_graph_data(
        &*neo4j,
        &project,
        &requested_layers,
        limit,
        params.community,
    )
    .await?;

    Ok(Json(ProjectGraphResponse {
        nodes,
        edges,
        communities,
        stats,
    }))
}

// ============================================================================
// Intelligence Summary endpoint
// ============================================================================

/// GET /api/projects/:slug/intelligence/summary
///
/// Returns an aggregated dashboard of intelligence metrics across all 6 layers.
/// Delegates to `build_intelligence_summary()` in graph_types.rs.
pub async fn get_intelligence_summary(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<IntelligenceSummaryResponse>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    let project = neo4j
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?;

    let summary = super::graph_types::build_intelligence_summary(
        &*neo4j,
        project.id,
    )
    .await?;

    Ok(Json(summary))
}

// ============================================================================
// Embeddings 2D projection (UMAP)
// ============================================================================

/// Query parameters for the embeddings projection endpoint
#[derive(Debug, Deserialize)]
pub struct ProjectionQuery {
    /// Projection dimensions: 2 (default) or 3
    pub dimensions: Option<usize>,
}

/// A point in the UMAP projection (note or decision)
#[derive(Serialize)]
pub struct ProjectionPoint {
    pub id: String,
    #[serde(rename = "type")]
    pub point_type: String,
    pub x: f64,
    pub y: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f64>,
    pub energy: f64,
    pub importance: String,
    pub tags: Vec<String>,
    pub content_preview: String,
}

/// A synapse edge between two notes
#[derive(Serialize)]
pub struct ProjectionSynapse {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

/// A skill cluster with centroid coordinates
#[derive(Serialize)]
pub struct ProjectionSkill {
    pub id: String,
    pub name: String,
    pub member_ids: Vec<String>,
    pub centroid_x: f64,
    pub centroid_y: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub centroid_z: Option<f64>,
}

/// Response for GET /api/projects/:slug/embeddings/projection
#[derive(Serialize)]
pub struct EmbeddingsProjectionResponse {
    pub points: Vec<ProjectionPoint>,
    pub synapses: Vec<ProjectionSynapse>,
    pub skills: Vec<ProjectionSkill>,
    pub dimensions: usize,
    pub projection_dimensions: usize,
    pub method: String,
}

/// GET /api/projects/:slug/embeddings/projection?dimensions=2|3
///
/// Projects note embeddings from high-dimensional space (768d) to 2D or 3D
/// using UMAP for visualization. Returns points, synapses, and skill clusters.
pub async fn get_embeddings_projection(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Query(query): Query<ProjectionQuery>,
) -> Result<Json<EmbeddingsProjectionResponse>, AppError> {
    let proj_dims = query.dimensions.unwrap_or(2);
    if proj_dims != 2 && proj_dims != 3 {
        return Err(AppError::BadRequest(
            "dimensions must be 2 or 3".to_string(),
        ));
    }
    let neo4j = state.orchestrator.neo4j();

    // 1. Resolve project
    let project = neo4j
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", slug)))?;
    let pid = project.id;

    // 2. Get all note embeddings for the project
    let embedding_points = neo4j.get_note_embeddings_for_project(pid).await?;

    if embedding_points.is_empty() {
        return Ok(Json(EmbeddingsProjectionResponse {
            points: vec![],
            synapses: vec![],
            skills: vec![],
            dimensions: 0,
            projection_dimensions: proj_dims,
            method: "none".to_string(),
        }));
    }

    // 3. Extract embeddings as Vec<Vec<f64>> for UMAP
    let embeddings_f64: Vec<Vec<f64>> = embedding_points
        .iter()
        .map(|p| p.embedding.iter().map(|&x| x as f64).collect())
        .collect();

    let dimensions = embeddings_f64.first().map(|v| v.len()).unwrap_or(0);

    // 4. Run UMAP projection (requires >= 4 points for n_neighbors=15)
    let circular_fallback = |n: usize, dims: usize| -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                let mut coords = vec![angle.cos() * 10.0, angle.sin() * 10.0];
                if dims == 3 {
                    coords.push(0.0);
                }
                coords
            })
            .collect()
    };

    let projected = if embedding_points.len() < 4 {
        // Too few points for UMAP — use simple circular layout
        circular_fallback(embedding_points.len(), proj_dims)
    } else if proj_dims == 3 {
        match rag_umap::convert_to_3d(embeddings_f64) {
            Ok(coords) => coords,
            Err(e) => {
                tracing::warn!(
                    "UMAP 3D projection failed, falling back to circular layout: {}",
                    e
                );
                circular_fallback(embedding_points.len(), 3)
            }
        }
    } else {
        match rag_umap::convert_to_2d(embeddings_f64) {
            Ok(coords) => coords,
            Err(e) => {
                tracing::warn!(
                    "UMAP 2D projection failed, falling back to circular layout: {}",
                    e
                );
                circular_fallback(embedding_points.len(), 2)
            }
        }
    };

    // 4b. Validate UMAP output — NaN/Inf coordinates crash serde_json serialization (500)
    let projected = if projected
        .iter()
        .any(|coords| coords.iter().any(|c| !c.is_finite()))
    {
        tracing::warn!("UMAP produced NaN/Inf coordinates, falling back to circular layout");
        circular_fallback(embedding_points.len(), proj_dims)
    } else {
        projected
    };

    // 5. Build projection points
    let points: Vec<ProjectionPoint> = embedding_points
        .iter()
        .zip(projected.iter())
        .map(|(ep, coords)| ProjectionPoint {
            id: ep.id.to_string(),
            point_type: "note".to_string(),
            x: coords.first().copied().unwrap_or(0.0),
            y: coords.get(1).copied().unwrap_or(0.0),
            z: if proj_dims == 3 {
                Some(coords.get(2).copied().unwrap_or(0.0))
            } else {
                None
            },
            energy: ep.energy,
            importance: ep.importance.clone(),
            tags: ep.tags.clone(),
            content_preview: ep.content_preview.clone(),
        })
        .collect();

    // 6. Get synapses and skills in parallel
    let point_ids: HashSet<String> = points.iter().map(|p| p.id.clone()).collect();

    let (synapse_res, skills_res) = tokio::join!(
        neo4j.get_synapse_graph(pid, 0.0),
        neo4j.list_skills(pid, None, 1000, 0),
    );

    // 7. Build synapses (filter to only include projected points)
    let synapses: Vec<ProjectionSynapse> = synapse_res
        .unwrap_or_default()
        .into_iter()
        .filter(|(src, tgt, _)| point_ids.contains(src) && point_ids.contains(tgt))
        .map(|(source, target, weight)| ProjectionSynapse {
            source,
            target,
            weight,
        })
        .collect();

    // 8. Build skills with centroids
    let id_to_coords: HashMap<String, (f64, f64, Option<f64>)> = points
        .iter()
        .map(|p| (p.id.clone(), (p.x, p.y, p.z)))
        .collect();

    let all_skills = skills_res.map(|(s, _)| s).unwrap_or_default();
    let mut skills = Vec::new();

    for skill in &all_skills {
        // Get member note IDs for this skill
        let members = neo4j.get_skill_members(skill.id).await;
        let member_notes = members.map(|(notes, _decs)| notes).unwrap_or_default();
        let member_ids: Vec<String> = member_notes
            .iter()
            .map(|n| n.id.to_string())
            .filter(|id| point_ids.contains(id))
            .collect();

        if member_ids.is_empty() {
            continue;
        }

        // Compute centroid from projected coordinates
        let (sum_x, sum_y, sum_z, count) = member_ids.iter().fold(
            (0.0_f64, 0.0_f64, 0.0_f64, 0usize),
            |(sx, sy, sz, c), id| {
                if let Some(&(x, y, z)) = id_to_coords.get(id) {
                    (sx + x, sy + y, sz + z.unwrap_or(0.0), c + 1)
                } else {
                    (sx, sy, sz, c)
                }
            },
        );

        let centroid_x = if count > 0 { sum_x / count as f64 } else { 0.0 };
        let centroid_y = if count > 0 { sum_y / count as f64 } else { 0.0 };
        let centroid_z = if proj_dims == 3 && count > 0 {
            Some(sum_z / count as f64)
        } else {
            None
        };

        skills.push(ProjectionSkill {
            id: skill.id.to_string(),
            name: skill.name.clone(),
            member_ids,
            centroid_x,
            centroid_y,
            centroid_z,
        });
    }

    Ok(Json(EmbeddingsProjectionResponse {
        points,
        synapses,
        skills,
        dimensions,
        projection_dimensions: proj_dims,
        method: if embedding_points.len() < 4 {
            "circular_fallback".to_string()
        } else {
            "umap".to_string()
        },
    }))
}

// ============================================================================
// Utilities
// ============================================================================

/// Convert a name to a URL-safe slug
fn slugify(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("My Project"), "my-project");
        assert_eq!(slugify("Test  Project!"), "test-project");
        assert_eq!(slugify("embryon-neural"), "embryon-neural");
        assert_eq!(slugify("Project 123"), "project-123");
    }

    #[test]
    fn test_update_project_request_all_fields() {
        let json = r#"{"name":"New Name","description":"New desc","root_path":"/new"}"#;
        let req: UpdateProjectRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, Some("New Name".to_string()));
        assert_eq!(req.description, Some(Some("New desc".to_string())));
        assert_eq!(req.root_path, Some("/new".to_string()));
    }

    #[test]
    fn test_update_project_request_empty() {
        let json = r#"{}"#;
        let req: UpdateProjectRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, None);
        assert_eq!(req.description, None);
        assert_eq!(req.root_path, None);
    }

    #[test]
    fn test_update_project_request_null_description() {
        // With default serde, null on Option<Option<String>> deserializes to None (absent)
        // To distinguish "field absent" from "field = null", a custom deserializer is needed
        let json = r#"{"description":null}"#;
        let req: UpdateProjectRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.description, None);
    }

    #[test]
    fn test_update_project_request_explicit_description() {
        // Explicit string value -> Some(Some("..."))
        let json = r#"{"description":"hello"}"#;
        let req: UpdateProjectRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.description, Some(Some("hello".to_string())));
    }

    #[test]
    fn test_update_project_request_only_name() {
        let json = r#"{"name":"Renamed"}"#;
        let req: UpdateProjectRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, Some("Renamed".to_string()));
        assert_eq!(req.description, None);
        assert_eq!(req.root_path, None);
    }

    // ====================================================================
    // Handler integration tests (axum + mock backends)
    // ====================================================================

    use axum::{
        body::Body,
        http::{Request, StatusCode as AxumStatus},
    };
    use tower::ServiceExt;

    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::neo4j::models::FileNode;
    use crate::orchestrator::watcher::FileWatcher;
    use crate::orchestrator::Orchestrator;
    use crate::test_helpers::{
        mock_app_state, test_auth_config, test_bearer_token, test_project_named,
    };

    /// Build a mock OrchestratorState for project handler tests
    async fn mock_server_state() -> super::OrchestratorState {
        let app_state = mock_app_state();
        let orchestrator = std::sync::Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = std::sync::Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        std::sync::Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: std::sync::Arc::new(crate::events::HybridEmitter::new(std::sync::Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: std::sync::Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
        })
    }

    fn authed_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn authed_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .header("authorization", test_bearer_token())
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn test_list_projects_empty() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let resp = app.oneshot(authed_get("/api/projects")).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 0);
        assert_eq!(json["items"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_list_projects_with_counts() {
        let app_state = mock_app_state();

        // Seed a project with files and plans
        let project = test_project_named("my-proj");
        app_state.neo4j.create_project(&project).await.unwrap();

        // Add 2 files
        for i in 0..2 {
            let path = format!("/tmp/my-proj/file_{}.rs", i);
            let file = FileNode {
                path: path.clone(),
                language: "rust".to_string(),
                hash: format!("h{}", i),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project.id),
            };
            app_state.neo4j.upsert_file(&file).await.unwrap();
            app_state
                .neo4j
                .link_file_to_project(&path, project.id)
                .await
                .unwrap();
        }

        // Add 1 plan (Draft)
        let plan = crate::test_helpers::test_plan_for_project(project.id);
        app_state.neo4j.create_plan(&plan).await.unwrap();
        app_state
            .neo4j
            .link_plan_to_project(plan.id, project.id)
            .await
            .unwrap();

        let orchestrator = std::sync::Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = std::sync::Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = std::sync::Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: std::sync::Arc::new(crate::events::HybridEmitter::new(std::sync::Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: std::sync::Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
        });
        let app = create_router(state);

        let resp = app.oneshot(authed_get("/api/projects")).await.unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total"], 1);
        let proj = &json["items"][0];
        assert_eq!(proj["file_count"], 2);
        assert_eq!(proj["plan_count"], 1);
    }

    #[tokio::test]
    async fn test_get_project_with_counts() {
        let app_state = mock_app_state();

        let project = test_project_named("detail-proj");
        app_state.neo4j.create_project(&project).await.unwrap();

        // Add 3 files
        for i in 0..3 {
            let path = format!("/tmp/detail-proj/f{}.rs", i);
            let file = FileNode {
                path: path.clone(),
                language: "rust".to_string(),
                hash: format!("d{}", i),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project.id),
            };
            app_state.neo4j.upsert_file(&file).await.unwrap();
            app_state
                .neo4j
                .link_file_to_project(&path, project.id)
                .await
                .unwrap();
        }

        let orchestrator = std::sync::Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = std::sync::Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = std::sync::Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: std::sync::Arc::new(crate::events::HybridEmitter::new(std::sync::Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: std::sync::Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
        });
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(&format!("/api/projects/{}", project.slug)))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["file_count"], 3);
        assert_eq!(json["plan_count"], 0);
        assert_eq!(json["slug"], "detail-proj");
    }

    #[tokio::test]
    async fn test_get_project_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get("/api/projects/nonexistent"))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_create_project_handler() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_post(
                "/api/projects",
                serde_json::json!({
                    "name": "New Project",
                    "root_path": "/tmp/new-proj"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["name"], "New Project");
        assert_eq!(json["slug"], "new-project");
        assert_eq!(json["file_count"], 0);
        assert_eq!(json["plan_count"], 0);
    }

    // ====================================================================
    // GET /api/projects/:slug/graph tests
    // ====================================================================

    use crate::graph::models::FileAnalyticsUpdate;

    /// Helper: seed a project with files, import edges, and analytics
    async fn seed_graph_project(app_state: &crate::AppState) -> ProjectNode {
        let project = test_project_named("graph-proj");
        app_state.neo4j.create_project(&project).await.unwrap();

        // Create 3 files
        let paths = [
            "/tmp/graph-proj/src/main.rs",
            "/tmp/graph-proj/src/lib.rs",
            "/tmp/graph-proj/src/utils.rs",
        ];
        for (i, path) in paths.iter().enumerate() {
            let file = FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: format!("gh{}", i),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project.id),
            };
            app_state.neo4j.upsert_file(&file).await.unwrap();
            app_state
                .neo4j
                .link_file_to_project(path, project.id)
                .await
                .unwrap();
        }

        // Create import edges: main.rs -> lib.rs, main.rs -> utils.rs
        app_state
            .neo4j
            .create_import_relationship(paths[0], paths[1], "crate::lib")
            .await
            .unwrap();
        app_state
            .neo4j
            .create_import_relationship(paths[0], paths[2], "crate::utils")
            .await
            .unwrap();

        // Seed file analytics (pagerank, betweenness, community)
        let analytics = vec![
            FileAnalyticsUpdate {
                path: paths[0].to_string(),
                pagerank: 0.5,
                betweenness: 0.3,
                community_id: 0,
                community_label: "core".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: paths[1].to_string(),
                pagerank: 0.3,
                betweenness: 0.1,
                community_id: 0,
                community_label: "core".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
            FileAnalyticsUpdate {
                path: paths[2].to_string(),
                pagerank: 0.2,
                betweenness: 0.05,
                community_id: 1,
                community_label: "utilities".to_string(),
                clustering_coefficient: 0.0,
                component_id: 0,
            },
        ];
        app_state
            .neo4j
            .batch_update_file_analytics(&analytics)
            .await
            .unwrap();

        project
    }

    /// Helper: build OrchestratorState from pre-seeded AppState
    async fn state_from_app(app_state: crate::AppState) -> super::OrchestratorState {
        let orchestrator = std::sync::Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = std::sync::Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        std::sync::Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: std::sync::Arc::new(crate::events::HybridEmitter::new(std::sync::Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: std::sync::Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
        })
    }

    #[tokio::test]
    async fn test_project_graph_default_code_layer() {
        let app_state = mock_app_state();
        let project = seed_graph_project(&app_state).await;
        let state = state_from_app(app_state).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(&format!("/api/projects/{}/graph", project.slug)))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Should have 3 file nodes (code layer)
        let nodes = json["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 3);

        // All nodes should be file type, code layer
        for node in nodes {
            assert_eq!(node["type"], "file");
            assert_eq!(node["layer"], "code");
        }

        // Should have 2 IMPORTS edges (main->lib, main->utils)
        let edges = json["edges"].as_array().unwrap();
        assert_eq!(edges.len(), 2);
        for edge in edges {
            assert_eq!(edge["type"], "IMPORTS");
            assert_eq!(edge["layer"], "code");
        }

        // Stats should show code layer
        let stats = &json["stats"];
        assert_eq!(stats["code"]["nodes"], 3);
        assert_eq!(stats["code"]["edges"], 2);

        // Communities should be present (2 communities: core + utilities)
        let communities = json["communities"].as_array().unwrap();
        assert_eq!(communities.len(), 2);

        // Check node attributes include analytics
        let main_node = nodes
            .iter()
            .find(|n| n["id"].as_str().unwrap().ends_with("main.rs"))
            .unwrap();
        let attrs = &main_node["attributes"];
        assert_eq!(attrs["language"], "rust");
        assert_eq!(attrs["pagerank"], 0.5);
        assert_eq!(attrs["community_id"], 0);
    }

    #[tokio::test]
    async fn test_project_graph_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get("/api/projects/nonexistent/graph"))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_project_graph_community_filter() {
        let app_state = mock_app_state();
        let project = seed_graph_project(&app_state).await;
        let state = state_from_app(app_state).await;
        let app = create_router(state);

        // Filter to community 0 (core) — should get main.rs + lib.rs only
        let resp = app
            .oneshot(authed_get(&format!(
                "/api/projects/{}/graph?community=0",
                project.slug
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let nodes = json["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 2);
        for node in nodes {
            assert_eq!(node["attributes"]["community_id"], 0);
        }

        // Edges: only main->lib should remain (main->utils filtered out because utils is community 1)
        let edges = json["edges"].as_array().unwrap();
        assert_eq!(edges.len(), 1);
        assert!(edges[0]["target"].as_str().unwrap().ends_with("lib.rs"));
    }

    #[tokio::test]
    async fn test_project_graph_multiple_layers() {
        let app_state = mock_app_state();
        let project = seed_graph_project(&app_state).await;
        let state = state_from_app(app_state).await;
        let app = create_router(state);

        // Request code + fabric layers
        let resp = app
            .oneshot(authed_get(&format!(
                "/api/projects/{}/graph?layers=code,fabric",
                project.slug
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Stats should have both code and fabric layers
        let stats = &json["stats"];
        assert!(stats["code"].is_object());
        assert!(stats["fabric"].is_object());
        // Fabric edges: mock returns empty vec, so 0 edges
        assert_eq!(stats["fabric"]["edges"], 0);
    }

    #[tokio::test]
    async fn test_project_graph_invalid_layer_ignored() {
        let app_state = mock_app_state();
        let project = seed_graph_project(&app_state).await;
        let state = state_from_app(app_state).await;
        let app = create_router(state);

        // "bogus" should be silently ignored, only "code" processed
        let resp = app
            .oneshot(authed_get(&format!(
                "/api/projects/{}/graph?layers=code,bogus",
                project.slug
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Only code layer should be in stats
        let stats = &json["stats"];
        assert!(stats["code"].is_object());
        assert!(stats["bogus"].is_null());
    }

    // ====================================================================
    // GET /api/projects/:slug/intelligence/summary tests
    // ====================================================================

    #[tokio::test]
    async fn test_intelligence_summary_empty_project() {
        let app_state = mock_app_state();
        let project = test_project_named("intel-proj");
        app_state.neo4j.create_project(&project).await.unwrap();

        let state = state_from_app(app_state).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(&format!(
                "/api/projects/{}/intelligence/summary",
                project.slug
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Empty project should have all zeros
        assert_eq!(json["code"]["files"], 0);
        assert_eq!(json["code"]["communities"], 0);
        assert_eq!(json["code"]["orphans"], 0);
        assert_eq!(json["knowledge"]["notes"], 0);
        assert_eq!(json["knowledge"]["stale_count"], 0);
        assert_eq!(json["fabric"]["co_changed_pairs"], 0);
        assert_eq!(json["neural"]["active_synapses"], 0);
        assert_eq!(json["neural"]["avg_energy"], 0.0);
        assert_eq!(json["skills"]["total"], 0);
        assert_eq!(json["skills"]["active"], 0);
        assert_eq!(json["skills"]["avg_cohesion"], 0.0);
    }

    #[tokio::test]
    async fn test_intelligence_summary_with_data() {
        let app_state = mock_app_state();
        let project = seed_graph_project(&app_state).await;

        let state = state_from_app(app_state).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(&format!(
                "/api/projects/{}/intelligence/summary",
                project.slug
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Code layer should reflect seeded data
        assert_eq!(json["code"]["files"], 3);
        assert_eq!(json["code"]["communities"], 2);

        // All 5 top-level sections should be present
        assert!(json["code"].is_object());
        assert!(json["knowledge"].is_object());
        assert!(json["fabric"].is_object());
        assert!(json["neural"].is_object());
        assert!(json["skills"].is_object());
    }

    #[tokio::test]
    async fn test_intelligence_summary_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get("/api/projects/nonexistent/intelligence/summary"))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }

    // ====================================================================
    // Embeddings projection tests
    // ====================================================================

    use crate::notes::NoteType;
    use crate::test_helpers::test_note;

    /// Helper: seed a project with notes that have embeddings
    async fn seed_projection_project(app_state: &crate::AppState) -> ProjectNode {
        let project = test_project_named("embed-proj");
        app_state.neo4j.create_project(&project).await.unwrap();

        // Create 5 notes with fake 4-dimensional embeddings (small for testing)
        for i in 0..5 {
            let note = test_note(
                project.id,
                NoteType::Context,
                &format!("Note content {} about some topic", i),
            );
            app_state.neo4j.create_note(&note).await.unwrap();

            // Generate a simple embedding: [i, i+1, i+2, i+3] normalized
            let base = i as f32;
            let emb = vec![base, base + 1.0, base + 2.0, base + 3.0];
            app_state
                .neo4j
                .set_note_embedding(note.id, &emb, "test-model")
                .await
                .unwrap();
        }

        project
    }

    #[tokio::test]
    async fn test_embeddings_projection_empty_project() {
        let app_state = mock_app_state();
        let project = test_project_named("empty-embed");
        app_state.neo4j.create_project(&project).await.unwrap();

        let state = state_from_app(app_state).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(&format!(
                "/api/projects/{}/embeddings/projection",
                project.slug
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["points"].as_array().unwrap().len(), 0);
        assert_eq!(json["method"], "none");
    }

    #[tokio::test]
    async fn test_embeddings_projection_with_notes() {
        let app_state = mock_app_state();
        let project = seed_projection_project(&app_state).await;

        let state = state_from_app(app_state).await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(&format!(
                "/api/projects/{}/embeddings/projection",
                project.slug
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Should have 5 projected points
        let points = json["points"].as_array().unwrap();
        assert_eq!(points.len(), 5);

        // Each point should have x, y coordinates and metadata
        for point in points {
            assert!(point["x"].is_f64() || point["x"].is_i64());
            assert!(point["y"].is_f64() || point["y"].is_i64());
            assert_eq!(point["type"], "note");
            assert!(point["content_preview"]
                .as_str()
                .unwrap()
                .contains("Note content"));
        }

        // Method should be umap (5 points > 4 threshold)
        assert_eq!(json["method"], "umap");
        assert_eq!(json["dimensions"], 4);

        // Synapses array should exist (empty since mock has none)
        assert!(json["synapses"].is_array());

        // Skills array should exist
        assert!(json["skills"].is_array());
    }

    #[tokio::test]
    async fn test_embeddings_projection_not_found() {
        let state = mock_server_state().await;
        let app = create_router(state);

        let resp = app
            .oneshot(authed_get(
                "/api/projects/nonexistent/embeddings/projection",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), AxumStatus::NOT_FOUND);
    }
}
