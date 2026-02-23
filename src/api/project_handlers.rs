//! Project API handlers

use crate::api::{PaginatedResponse, PaginationParams, SearchFilter};
use crate::neo4j::models::ProjectNode;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::expand_tilde;

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

pub async fn sync_project(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<SyncProjectResponse>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    let expanded = expand_tilde(&project.root_path);
    let path = std::path::Path::new(&expanded);
    let result = state
        .orchestrator
        .sync_directory_for_project(path, Some(project.id), Some(&project.slug))
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

    Ok(Json(SyncProjectResponse {
        files_synced: result.files_synced,
        files_skipped: result.files_skipped,
        files_deleted: result.files_deleted,
        symbols_deleted: result.symbols_deleted,
        errors: result.errors,
    }))
}

/// List plans for a project
pub async fn list_project_plans(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<crate::neo4j::models::PlanNode>>, AppError> {
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project '{}' not found", slug)))?;

    let plans = state
        .orchestrator
        .neo4j()
        .list_project_plans(project.id)
        .await?;

    Ok(Json(plans))
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
}
