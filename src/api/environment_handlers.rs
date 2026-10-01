//! REST handlers for project environments and deployments.
//!
//! - `GET/POST   /api/projects/{project_id}/environments`
//! - `GET/PATCH/DELETE /api/environments/{id}`
//! - `GET/POST   /api/environments/{id}/deployments`
//! - `PATCH      /api/deployments/{id}`
//! - `GET        /api/projects/{project_id}/deployment-matrix`
//!
//! Enum-valued fields (`kind`, `status`) are received as plain strings and parsed
//! here so that an invalid value yields a `400 Bad Request` (rather than the
//! extractor's generic `422`). A duplicate environment name yields `409 Conflict`.

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams};
use crate::events::{EntityType, EventEmitter};
use crate::neo4j::models::{
    DeploymentMatrixEntry, DeploymentNode, DeploymentStatus, EnvironmentKind, EnvironmentNode,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use uuid::Uuid;

// ============================================================================
// Helpers
// ============================================================================

/// Normalise the free-form `config` field to a JSON string.
///
/// Accepts either a JSON object/array/etc. (serialised) or a string that itself
/// contains valid JSON. An empty string is passed through (it clears the value on
/// update). Anything else is a `400`.
fn normalize_config(value: serde_json::Value) -> Result<String, AppError> {
    match value {
        serde_json::Value::String(s) => {
            if s.trim().is_empty() {
                return Ok(String::new());
            }
            serde_json::from_str::<serde_json::Value>(&s)
                .map_err(|e| AppError::BadRequest(format!("config must be valid JSON: {}", e)))?;
            Ok(s)
        }
        serde_json::Value::Null => Ok(String::new()),
        other => Ok(other.to_string()),
    }
}

fn parse_kind(s: &str) -> Result<EnvironmentKind, AppError> {
    s.parse().map_err(AppError::BadRequest)
}

fn parse_status(s: &str) -> Result<DeploymentStatus, AppError> {
    s.parse().map_err(AppError::BadRequest)
}

fn clean_name(name: &str) -> Result<String, AppError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(AppError::BadRequest("name must not be empty".into()));
    }
    Ok(name.to_string())
}

async fn require_environment(
    state: &OrchestratorState,
    id: Uuid,
) -> Result<EnvironmentNode, AppError> {
    state
        .orchestrator
        .neo4j()
        .get_environment(id)
        .await?
        .ok_or_else(|| AppError::NotFound("Environment not found".into()))
}

async fn require_project(state: &OrchestratorState, project_id: Uuid) -> Result<(), AppError> {
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await?
        .map(|_| ())
        .ok_or_else(|| AppError::NotFound("Project not found".into()))
}

/// Fail with `409` when `name` is already used by another environment of the project.
async fn ensure_name_available(
    state: &OrchestratorState,
    project_id: Uuid,
    name: &str,
    except: Option<Uuid>,
) -> Result<(), AppError> {
    let taken = state
        .orchestrator
        .neo4j()
        .list_project_environments(project_id)
        .await?
        .iter()
        .any(|e| e.name == name && Some(e.id) != except);
    if taken {
        return Err(AppError::Conflict(format!(
            "An environment named '{}' already exists in this project",
            name
        )));
    }
    Ok(())
}

// ============================================================================
// Environments
// ============================================================================

/// Request to create an environment
#[derive(Debug, Deserialize)]
pub struct CreateEnvironmentRequest {
    pub name: String,
    /// dev | staging | production | other (default: other)
    pub kind: Option<String>,
    pub url: Option<String>,
    pub description: Option<String>,
    /// Free-form JSON (object or JSON string): host, region, runtime, ...
    pub config: Option<serde_json::Value>,
}

/// `GET /api/projects/{project_id}/environments`
pub async fn list_environments(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
) -> Result<Json<Vec<EnvironmentNode>>, AppError> {
    let envs = state
        .orchestrator
        .neo4j()
        .list_project_environments(project_id)
        .await?;
    Ok(Json(envs))
}

/// `POST /api/projects/{project_id}/environments`
pub async fn create_environment(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
    Json(req): Json<CreateEnvironmentRequest>,
) -> Result<Json<EnvironmentNode>, AppError> {
    let name = clean_name(&req.name)?;
    let kind = match req.kind.as_deref() {
        Some(k) => parse_kind(k)?,
        None => EnvironmentKind::Other,
    };
    let config = match req.config {
        Some(v) => Some(normalize_config(v)?).filter(|s| !s.is_empty()),
        None => None,
    };

    require_project(&state, project_id).await?;
    ensure_name_available(&state, project_id, &name, None).await?;

    let env = EnvironmentNode {
        id: Uuid::new_v4(),
        project_id,
        name,
        kind,
        url: req.url.filter(|s| !s.is_empty()),
        description: req.description.filter(|s| !s.is_empty()),
        config,
        created_at: Utc::now(),
    };
    state.orchestrator.neo4j().create_environment(&env).await?;
    state.event_bus.emit_created(
        EntityType::Environment,
        &env.id.to_string(),
        serde_json::json!({"name": &env.name, "kind": env.kind}),
        Some(project_id.to_string()),
    );
    Ok(Json(env))
}

/// `GET /api/environments/{id}`
pub async fn get_environment(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<Json<EnvironmentNode>, AppError> {
    Ok(Json(require_environment(&state, id).await?))
}

/// Request to update an environment (all fields optional).
///
/// An empty string clears `url`, `description` and `config`.
#[derive(Debug, Deserialize)]
pub struct UpdateEnvironmentRequest {
    pub name: Option<String>,
    pub kind: Option<String>,
    pub url: Option<String>,
    pub description: Option<String>,
    pub config: Option<serde_json::Value>,
}

/// `PATCH /api/environments/{id}` — returns the updated environment
pub async fn update_environment(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateEnvironmentRequest>,
) -> Result<Json<EnvironmentNode>, AppError> {
    let name = req.name.as_deref().map(clean_name).transpose()?;
    let kind = req.kind.as_deref().map(parse_kind).transpose()?;
    let config = req.config.map(normalize_config).transpose()?;

    let existing = require_environment(&state, id).await?;
    if let Some(ref n) = name {
        if *n != existing.name {
            ensure_name_available(&state, existing.project_id, n, Some(id)).await?;
        }
    }

    state
        .orchestrator
        .neo4j()
        .update_environment(id, name, kind, req.url, req.description, config)
        .await?;
    let updated = require_environment(&state, id).await?;
    state.event_bus.emit_updated(
        EntityType::Environment,
        &id.to_string(),
        serde_json::json!({"name": &updated.name, "kind": updated.kind}),
        Some(updated.project_id.to_string()),
    );
    Ok(Json(updated))
}

/// `DELETE /api/environments/{id}` — also deletes its deployments
pub async fn delete_environment(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    let existing = require_environment(&state, id).await?;
    state.orchestrator.neo4j().delete_environment(id).await?;
    state.event_bus.emit_deleted(
        EntityType::Environment,
        &id.to_string(),
        Some(existing.project_id.to_string()),
    );
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Deployments
// ============================================================================

/// Request to record a deployment
#[derive(Debug, Deserialize)]
pub struct CreateDeploymentRequest {
    pub version: Option<String>,
    pub commit_sha: Option<String>,
    /// pending | running | succeeded | failed | rolled_back (default: pending)
    pub status: Option<String>,
    pub notes: Option<String>,
    /// Who/what triggered the deployment (default: "api")
    pub created_by: Option<String>,
}

/// `GET /api/environments/{id}/deployments` — newest first, paginated
pub async fn list_deployments(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
    Query(pagination): Query<PaginationParams>,
) -> Result<Json<PaginatedResponse<DeploymentNode>>, AppError> {
    pagination.validate().map_err(AppError::BadRequest)?;
    require_environment(&state, id).await?;

    let (items, total) = state
        .orchestrator
        .neo4j()
        .list_environment_deployments(id, pagination.validated_limit(), pagination.offset)
        .await?;
    Ok(Json(PaginatedResponse::new(
        items,
        total,
        pagination.validated_limit(),
        pagination.offset,
    )))
}

/// `POST /api/environments/{id}/deployments`
pub async fn create_deployment(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
    Json(req): Json<CreateDeploymentRequest>,
) -> Result<Json<DeploymentNode>, AppError> {
    let status = match req.status.as_deref() {
        Some(s) => parse_status(s)?,
        None => DeploymentStatus::Pending,
    };
    let env = require_environment(&state, id).await?;

    let now = Utc::now();
    let deployment = DeploymentNode {
        id: Uuid::new_v4(),
        environment_id: id,
        version: req.version.filter(|s| !s.trim().is_empty()),
        commit_sha: req.commit_sha.filter(|s| !s.trim().is_empty()),
        status,
        notes: req.notes.filter(|s| !s.is_empty()),
        created_by: req
            .created_by
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "api".to_string()),
        started_at: now,
        finished_at: status.is_terminal().then_some(now),
    };
    state
        .orchestrator
        .neo4j()
        .create_deployment(&deployment)
        .await?;
    state.event_bus.emit_created(
        EntityType::Deployment,
        &deployment.id.to_string(),
        serde_json::json!({
            "environment_id": id,
            "version": &deployment.version,
            "commit_sha": &deployment.commit_sha,
            "status": deployment.status,
        }),
        Some(env.project_id.to_string()),
    );
    Ok(Json(deployment))
}

/// Request to update a deployment
#[derive(Debug, Deserialize)]
pub struct UpdateDeploymentRequest {
    pub status: Option<String>,
    pub finished_at: Option<DateTime<Utc>>,
    pub notes: Option<String>,
}

/// `PATCH /api/deployments/{id}` — returns the updated deployment.
///
/// Moving to a terminal status (succeeded/failed/rolled_back) stamps
/// `finished_at` with the current time unless one is given or already set.
pub async fn update_deployment(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateDeploymentRequest>,
) -> Result<Json<DeploymentNode>, AppError> {
    let new_status = req.status.as_deref().map(parse_status).transpose()?;
    let existing = state
        .orchestrator
        .neo4j()
        .get_deployment(id)
        .await?
        .ok_or_else(|| AppError::NotFound("Deployment not found".into()))?;

    let finished_at = match (req.finished_at, new_status) {
        (Some(f), _) => Some(f),
        (None, Some(s)) if s.is_terminal() && existing.finished_at.is_none() => Some(Utc::now()),
        _ => None,
    };

    state
        .orchestrator
        .neo4j()
        .update_deployment(id, new_status, finished_at, req.notes)
        .await?;
    let updated = state
        .orchestrator
        .neo4j()
        .get_deployment(id)
        .await?
        .ok_or_else(|| AppError::NotFound("Deployment not found".into()))?;

    let project_id = state
        .orchestrator
        .neo4j()
        .get_environment(updated.environment_id)
        .await?
        .map(|e| e.project_id.to_string());
    if new_status.is_some() && existing.status != updated.status {
        state.event_bus.emit_status_changed(
            EntityType::Deployment,
            &id.to_string(),
            existing.status.as_str(),
            updated.status.as_str(),
            project_id,
        );
    } else {
        state.event_bus.emit_updated(
            EntityType::Deployment,
            &id.to_string(),
            serde_json::json!({"notes": &updated.notes}),
            project_id,
        );
    }
    Ok(Json(updated))
}

// ============================================================================
// Deployment matrix
// ============================================================================

/// `GET /api/projects/{project_id}/deployment-matrix` — environments x latest
/// deployment, in one aggregate query.
pub async fn get_deployment_matrix(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
) -> Result<Json<Vec<DeploymentMatrixEntry>>, AppError> {
    require_project(&state, project_id).await?;
    let matrix = state
        .orchestrator
        .neo4j()
        .get_deployment_matrix(project_id)
        .await?;
    Ok(Json(matrix))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::events::CrudAction;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::traits::GraphStore;
    use crate::orchestrator::watcher::FileWatcher;
    use crate::orchestrator::Orchestrator;
    use crate::test_helpers::{
        mock_app_state_with_graph, test_auth_config, test_bearer_token, test_commit, test_project,
    };
    use axum::body::Body;
    use axum::http::Request;
    use axum::Router;
    use std::sync::Arc;
    use tower::ServiceExt;

    struct Harness {
        app: Router,
        graph: Arc<MockGraphStore>,
        events: tokio::sync::broadcast::Receiver<crate::events::CrudEvent>,
        project_id: Uuid,
    }

    async fn harness() -> Harness {
        let graph = Arc::new(MockGraphStore::new());
        let project = test_project();
        graph.create_project(&project).await.unwrap();

        let orchestrator = Arc::new(
            Orchestrator::new(mock_app_state_with_graph(graph.clone()))
                .await
                .unwrap(),
        );
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let event_bus = Arc::new(crate::events::HybridEmitter::new(Arc::new(
            crate::events::EventBus::default(),
        )));
        let events = event_bus.subscribe();
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus,
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            remote_mcp: crate::RemoteMcpConfig::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
        });
        Harness {
            app: create_router(state),
            graph,
            events,
            project_id: project.id,
        }
    }

    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let mut builder = Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", test_bearer_token());
        let body = match body {
            Some(b) => {
                builder = builder.header("content-type", "application/json");
                Body::from(b.to_string())
            }
            None => Body::empty(),
        };
        let resp = app
            .clone()
            .oneshot(builder.body(body).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    async fn create_env(h: &Harness, name: &str, kind: &str) -> serde_json::Value {
        let (status, json) = call(
            &h.app,
            "POST",
            &format!("/api/projects/{}/environments", h.project_id),
            Some(serde_json::json!({"name": name, "kind": kind})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{json}");
        json
    }

    async fn create_dep(
        h: &Harness,
        env_id: &str,
        version: &str,
        status: &str,
    ) -> serde_json::Value {
        let (st, json) = call(
            &h.app,
            "POST",
            &format!("/api/environments/{env_id}/deployments"),
            Some(serde_json::json!({"version": version, "status": status})),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{json}");
        json
    }

    // ---------------------------------------------------------------- environments

    #[tokio::test]
    async fn test_create_and_list_environments() {
        let h = harness().await;
        let (status, json) = call(
            &h.app,
            "POST",
            &format!("/api/projects/{}/environments", h.project_id),
            Some(serde_json::json!({
                "name": "  prod  ",
                "kind": "production",
                "url": "https://example.com",
                "description": "live",
                "config": {"region": "eu-west-1", "host": "h1"}
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["name"], "prod");
        assert_eq!(json["kind"], "production");
        assert_eq!(json["project_id"], h.project_id.to_string());
        assert_eq!(json["url"], "https://example.com");
        let config: serde_json::Value =
            serde_json::from_str(json["config"].as_str().unwrap()).unwrap();
        assert_eq!(config["region"], "eu-west-1");

        let (status, list) = call(
            &h.app,
            "GET",
            &format!("/api/projects/{}/environments", h.project_id),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(list.as_array().unwrap().len(), 1);
        assert_eq!(list[0]["id"], json["id"]);
    }

    #[tokio::test]
    async fn test_create_environment_defaults_kind_to_other() {
        let h = harness().await;
        let (status, json) = call(
            &h.app,
            "POST",
            &format!("/api/projects/{}/environments", h.project_id),
            Some(serde_json::json!({"name": "sandbox"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["kind"], "other");
        assert!(json["url"].is_null());
        assert!(json["config"].is_null());
    }

    #[tokio::test]
    async fn test_create_environment_bad_kind_is_400() {
        let h = harness().await;
        let (status, json) = call(
            &h.app,
            "POST",
            &format!("/api/projects/{}/environments", h.project_id),
            Some(serde_json::json!({"name": "x", "kind": "prod"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(json["error"].as_str().unwrap().contains("kind"));
    }

    #[tokio::test]
    async fn test_create_environment_bad_input_is_400() {
        let h = harness().await;
        let uri = format!("/api/projects/{}/environments", h.project_id);
        let (status, _) = call(
            &h.app,
            "POST",
            &uri,
            Some(serde_json::json!({"name": "  "})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = call(
            &h.app,
            "POST",
            &uri,
            Some(serde_json::json!({"name": "x", "config": "{not json"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_environment_unknown_project_is_404() {
        let h = harness().await;
        let (status, _) = call(
            &h.app,
            "POST",
            &format!("/api/projects/{}/environments", Uuid::new_v4()),
            Some(serde_json::json!({"name": "x"})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_create_environment_duplicate_name_is_409() {
        let h = harness().await;
        create_env(&h, "staging", "staging").await;
        let (status, json) = call(
            &h.app,
            "POST",
            &format!("/api/projects/{}/environments", h.project_id),
            Some(serde_json::json!({"name": "staging", "kind": "dev"})),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(json["error"].as_str().unwrap().contains("staging"));
    }

    #[tokio::test]
    async fn test_same_name_allowed_in_other_project() {
        let h = harness().await;
        create_env(&h, "prod", "production").await;
        let other = crate::test_helpers::test_project_named("other");
        h.graph.create_project(&other).await.unwrap();
        let (status, _) = call(
            &h.app,
            "POST",
            &format!("/api/projects/{}/environments", other.id),
            Some(serde_json::json!({"name": "prod", "kind": "production"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_environment_and_404() {
        let h = harness().await;
        let env = create_env(&h, "dev", "dev").await;
        let id = env["id"].as_str().unwrap();

        let (status, json) = call(&h.app, "GET", &format!("/api/environments/{id}"), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["name"], "dev");

        let (status, _) = call(
            &h.app,
            "GET",
            &format!("/api/environments/{}", Uuid::new_v4()),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_update_environment() {
        let h = harness().await;
        let env = create_env(&h, "dev", "dev").await;
        let id = env["id"].as_str().unwrap();
        let uri = format!("/api/environments/{id}");

        let (status, json) = call(
            &h.app,
            "PATCH",
            &uri,
            Some(serde_json::json!({
                "name": "development", "kind": "other", "url": "http://dev", "config": {"a": 1}
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["name"], "development");
        assert_eq!(json["kind"], "other");
        assert_eq!(json["url"], "http://dev");
        assert_eq!(json["config"], "{\"a\":1}");

        // Empty string clears an optional field; renaming to itself is not a conflict
        let (status, json) = call(
            &h.app,
            "PATCH",
            &uri,
            Some(serde_json::json!({"url": "", "name": "development"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(json["url"].is_null());
    }

    #[tokio::test]
    async fn test_update_environment_errors() {
        let h = harness().await;
        create_env(&h, "prod", "production").await;
        let dev = create_env(&h, "dev", "dev").await;
        let uri = format!("/api/environments/{}", dev["id"].as_str().unwrap());

        let (status, _) = call(
            &h.app,
            "PATCH",
            &uri,
            Some(serde_json::json!({"kind": "nope"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = call(
            &h.app,
            "PATCH",
            &uri,
            Some(serde_json::json!({"name": "prod"})),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        let (status, _) = call(
            &h.app,
            "PATCH",
            &format!("/api/environments/{}", Uuid::new_v4()),
            Some(serde_json::json!({"name": "z"})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_delete_environment_cascades_deployments() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let id = env["id"].as_str().unwrap();
        let dep = create_dep(&h, id, "1.0.0", "succeeded").await;

        let (status, _) = call(&h.app, "DELETE", &format!("/api/environments/{id}"), None).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        let (status, _) = call(&h.app, "GET", &format!("/api/environments/{id}"), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(h
            .graph
            .get_deployment(dep["id"].as_str().unwrap().parse().unwrap())
            .await
            .unwrap()
            .is_none());

        let (status, _) = call(&h.app, "DELETE", &format!("/api/environments/{id}"), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    // ------------------------------------------------------------------ deployments

    #[tokio::test]
    async fn test_create_deployment_defaults() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let (status, json) = call(
            &h.app,
            "POST",
            &format!(
                "/api/environments/{}/deployments",
                env["id"].as_str().unwrap()
            ),
            Some(serde_json::json!({"version": "1.2.3", "commit_sha": "abc123"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["status"], "pending");
        assert_eq!(json["version"], "1.2.3");
        assert_eq!(json["commit_sha"], "abc123");
        assert_eq!(json["created_by"], "api");
        assert!(json["finished_at"].is_null());
        assert_eq!(json["environment_id"], env["id"]);
    }

    #[tokio::test]
    async fn test_create_deployment_terminal_status_sets_finished_at() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let dep = create_dep(&h, env["id"].as_str().unwrap(), "1", "succeeded").await;
        assert!(dep["finished_at"].is_string());
    }

    #[tokio::test]
    async fn test_create_deployment_errors() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let uri = format!(
            "/api/environments/{}/deployments",
            env["id"].as_str().unwrap()
        );
        let (status, json) = call(
            &h.app,
            "POST",
            &uri,
            Some(serde_json::json!({"version": "1", "status": "done"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(json["error"].as_str().unwrap().contains("status"));

        let (status, _) = call(
            &h.app,
            "POST",
            &format!("/api/environments/{}/deployments", Uuid::new_v4()),
            Some(serde_json::json!({"version": "1"})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_create_deployment_with_unknown_commit_does_not_fail() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        // No Commit node exists for this sha: the deployment is still recorded.
        let (status, _) = call(
            &h.app,
            "POST",
            &format!(
                "/api/environments/{}/deployments",
                env["id"].as_str().unwrap()
            ),
            Some(serde_json::json!({"commit_sha": "deadbeef"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        // ... and with an existing commit
        h.graph
            .create_commit(&test_commit("cafe01", "msg"))
            .await
            .unwrap();
        let (status, _) = call(
            &h.app,
            "POST",
            &format!(
                "/api/environments/{}/deployments",
                env["id"].as_str().unwrap()
            ),
            Some(serde_json::json!({"commit_sha": "cafe01"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_deployments_newest_first_and_paginated() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let id = env["id"].as_str().unwrap();
        for v in ["1", "2", "3"] {
            create_dep(&h, id, v, "succeeded").await;
            tokio::time::sleep(std::time::Duration::from_millis(3)).await;
        }
        let (status, json) = call(
            &h.app,
            "GET",
            &format!("/api/environments/{id}/deployments?limit=2"),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["total"], 3);
        assert_eq!(json["has_more"], true);
        let items = json["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["version"], "3");
        assert_eq!(items[1]["version"], "2");

        let (_, json) = call(
            &h.app,
            "GET",
            &format!("/api/environments/{id}/deployments?limit=2&offset=2"),
            None,
        )
        .await;
        assert_eq!(json["items"][0]["version"], "1");

        let (status, _) = call(
            &h.app,
            "GET",
            &format!("/api/environments/{id}/deployments?limit=500"),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, _) = call(
            &h.app,
            "GET",
            &format!("/api/environments/{}/deployments", Uuid::new_v4()),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_update_deployment() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let dep = create_dep(&h, env["id"].as_str().unwrap(), "1", "running").await;
        let uri = format!("/api/deployments/{}", dep["id"].as_str().unwrap());

        let (status, json) = call(
            &h.app,
            "PATCH",
            &uri,
            Some(serde_json::json!({"status": "failed", "notes": "db migration timed out"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["status"], "failed");
        assert_eq!(json["notes"], "db migration timed out");
        assert!(
            json["finished_at"].is_string(),
            "terminal status stamps finished_at"
        );

        // Explicit finished_at wins
        let (_, json) = call(
            &h.app,
            "PATCH",
            &uri,
            Some(
                serde_json::json!({"status": "rolled_back", "finished_at": "2030-01-01T00:00:00Z"}),
            ),
        )
        .await;
        assert_eq!(json["status"], "rolled_back");
        assert!(json["finished_at"]
            .as_str()
            .unwrap()
            .starts_with("2030-01-01"));
    }

    #[tokio::test]
    async fn test_update_deployment_errors() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let dep = create_dep(&h, env["id"].as_str().unwrap(), "1", "running").await;

        let (status, _) = call(
            &h.app,
            "PATCH",
            &format!("/api/deployments/{}", dep["id"].as_str().unwrap()),
            Some(serde_json::json!({"status": "exploded"})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, _) = call(
            &h.app,
            "PATCH",
            &format!("/api/deployments/{}", Uuid::new_v4()),
            Some(serde_json::json!({"status": "failed"})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    // ----------------------------------------------------------------------- matrix

    #[tokio::test]
    async fn test_matrix_empty_and_unknown_project() {
        let h = harness().await;
        let (status, json) = call(
            &h.app,
            "GET",
            &format!("/api/projects/{}/deployment-matrix", h.project_id),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json.as_array().unwrap().len(), 0);

        let (status, _) = call(
            &h.app,
            "GET",
            &format!("/api/projects/{}/deployment-matrix", Uuid::new_v4()),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_matrix_aggregation() {
        let h = harness().await;
        let prod = create_env(&h, "prod", "production").await;
        let stg = create_env(&h, "staging", "staging").await;
        let _dev = create_env(&h, "dev", "dev").await; // never deployed

        // 7 deployments to prod: only the 5 newest statuses are reported
        let prod_id = prod["id"].as_str().unwrap();
        let statuses = [
            "failed",
            "succeeded",
            "succeeded",
            "rolled_back",
            "succeeded",
            "failed",
            "running",
        ];
        for (i, st) in statuses.iter().enumerate() {
            create_dep(&h, prod_id, &format!("v{i}"), st).await;
            tokio::time::sleep(std::time::Duration::from_millis(3)).await;
        }
        create_dep(&h, stg["id"].as_str().unwrap(), "v9", "succeeded").await;

        let (status, json) = call(
            &h.app,
            "GET",
            &format!("/api/projects/{}/deployment-matrix", h.project_id),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let rows = json.as_array().unwrap();
        assert_eq!(rows.len(), 3);

        let row = |name: &str| {
            rows.iter()
                .find(|r| r["environment"]["name"] == name)
                .unwrap_or_else(|| panic!("row {name}"))
        };
        let p = row("prod");
        assert_eq!(p["latest_deployment"]["version"], "v6");
        assert_eq!(
            p["recent_statuses"],
            serde_json::json!(["running", "failed", "succeeded", "rolled_back", "succeeded"])
        );
        let s = row("staging");
        assert_eq!(s["latest_deployment"]["version"], "v9");
        assert_eq!(s["recent_statuses"], serde_json::json!(["succeeded"]));
        let d = row("dev");
        assert!(d["latest_deployment"].is_null());
        assert_eq!(d["recent_statuses"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn test_matrix_is_scoped_to_project() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        create_dep(&h, env["id"].as_str().unwrap(), "1", "succeeded").await;

        let other = crate::test_helpers::test_project_named("other");
        h.graph.create_project(&other).await.unwrap();
        let (_, json) = call(
            &h.app,
            "GET",
            &format!("/api/projects/{}/deployment-matrix", other.id),
            None,
        )
        .await;
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_project_delete_cascades_environments() {
        let h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let dep = create_dep(&h, env["id"].as_str().unwrap(), "1", "succeeded").await;
        h.graph.delete_project(h.project_id, "test").await.unwrap();
        assert!(h
            .graph
            .get_environment(env["id"].as_str().unwrap().parse().unwrap())
            .await
            .unwrap()
            .is_none());
        assert!(h
            .graph
            .get_deployment(dep["id"].as_str().unwrap().parse().unwrap())
            .await
            .unwrap()
            .is_none());
    }

    // ----------------------------------------------------------------------- events

    #[tokio::test]
    async fn test_events_emitted() {
        let mut h = harness().await;
        let env = create_env(&h, "prod", "production").await;
        let ev = h.events.try_recv().unwrap();
        assert_eq!(ev.entity_type, EntityType::Environment);
        assert_eq!(ev.action, CrudAction::Created);
        assert_eq!(
            ev.project_id.as_deref(),
            Some(h.project_id.to_string().as_str())
        );

        let dep = create_dep(&h, env["id"].as_str().unwrap(), "1", "running").await;
        let ev = h.events.try_recv().unwrap();
        assert_eq!(ev.entity_type, EntityType::Deployment);
        assert_eq!(ev.action, CrudAction::Created);

        call(
            &h.app,
            "PATCH",
            &format!("/api/deployments/{}", dep["id"].as_str().unwrap()),
            Some(serde_json::json!({"status": "succeeded"})),
        )
        .await;
        let ev = h.events.try_recv().unwrap();
        assert_eq!(ev.entity_type, EntityType::Deployment);
        assert_eq!(ev.action, CrudAction::StatusChanged);

        call(
            &h.app,
            "DELETE",
            &format!("/api/environments/{}", env["id"].as_str().unwrap()),
            None,
        )
        .await;
        let ev = h.events.try_recv().unwrap();
        assert_eq!(ev.entity_type, EntityType::Environment);
        assert_eq!(ev.action, CrudAction::Deleted);
    }

    // ------------------------------------------------------------------- unit tests

    #[test]
    fn test_kind_and_status_parsing() {
        assert_eq!(
            "Production".parse::<EnvironmentKind>().unwrap(),
            EnvironmentKind::Production
        );
        assert!("prod".parse::<EnvironmentKind>().is_err());
        for s in ["pending", "running", "succeeded", "failed", "rolled_back"] {
            let parsed: DeploymentStatus = s.parse().unwrap();
            assert_eq!(parsed.as_str(), s);
            assert_eq!(serde_json::to_value(parsed).unwrap(), s);
        }
        assert!("rolled-back".parse::<DeploymentStatus>().is_err());
        assert!(DeploymentStatus::Failed.is_terminal());
        assert!(!DeploymentStatus::Running.is_terminal());
    }

    #[test]
    fn test_normalize_config() {
        assert_eq!(
            normalize_config(serde_json::json!({"a": 1})).unwrap(),
            "{\"a\":1}"
        );
        assert_eq!(
            normalize_config(serde_json::json!("{\"a\":1}")).unwrap(),
            "{\"a\":1}"
        );
        assert_eq!(normalize_config(serde_json::json!("")).unwrap(), "");
        assert!(normalize_config(serde_json::json!("nope")).is_err());
    }
}
