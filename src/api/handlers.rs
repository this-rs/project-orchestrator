//! API request handlers

use crate::api::{
    PaginatedResponse, PaginationParams, PriorityFilter, SearchFilter, StatusFilter, TagsFilter,
};
use crate::chat::ChatManager;
use crate::events::{EventEmitter, HybridEmitter, NatsEmitter};
use crate::neo4j::models::{
    CommitNode, ConstraintNode, DecisionNode, MilestoneNode, MilestoneStatus, PlanNode, PlanStatus,
    ReleaseNode, ReleaseStatus, StepNode, TaskNode, TaskWithPlan,
};
use crate::orchestrator::{FileWatcher, Orchestrator};
use crate::plan::models::*;
use crate::AuthConfig;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Shared server state
pub struct ServerState {
    pub orchestrator: Arc<Orchestrator>,
    pub watcher: Arc<RwLock<FileWatcher>>,
    pub chat_manager: Option<Arc<ChatManager>>,
    pub event_bus: Arc<HybridEmitter>,
    /// NATS emitter for inter-process chat events and interrupts.
    /// None when NATS is not configured (local-only mode).
    pub nats_emitter: Option<Arc<NatsEmitter>>,
    /// Auth config — None means deny-by-default
    pub auth_config: Option<AuthConfig>,
    /// Whether the app has been fully configured (setup wizard completed).
    /// When false, the frontend should show the setup wizard.
    pub setup_completed: bool,
    /// Whether to serve the frontend static files (SPA fallback)
    pub serve_frontend: bool,
    /// Path to the frontend dist/ directory
    pub frontend_path: String,
    /// Server port (used for building the localhost origin in OAuth whitelist)
    pub server_port: u16,
    /// Public URL for reverse-proxy setups (e.g. https://ffs.dev).
    /// Used for CORS and OAuth origin whitelist when both desktop + web access is needed.
    pub public_url: Option<String>,
    /// In-memory store for ephemeral WebSocket auth tickets.
    /// Used as a fallback when cookies are not sent on WS upgrades (WKWebView).
    pub ws_ticket_store: Arc<super::ws_auth::WsTicketStore>,
}

/// Shared orchestrator state
pub type OrchestratorState = Arc<ServerState>;

impl ServerState {
    /// Build the list of allowed origins for OAuth redirect_uri validation and CORS.
    ///
    /// Always includes:
    /// - `http://localhost:{server_port}` (desktop / Tauri app)
    /// - `tauri://localhost` (Tauri webview custom scheme — Linux/Windows)
    /// - `https://tauri.localhost` (Tauri 2 WKWebView on macOS — WebKit transforms
    ///   `tauri://localhost` → `https://tauri.localhost` for the Origin header)
    ///
    /// Optionally includes:
    /// - `public_url` from server config (e.g. `https://ffs.dev`)
    /// - `frontend_url` from auth config (backward compat — many configs use this instead of public_url)
    pub fn allowed_origins(&self) -> Vec<String> {
        let mut origins = vec![
            format!("http://localhost:{}", self.server_port),
            format!("http://127.0.0.1:{}", self.server_port),
            "tauri://localhost".to_string(),
            "https://tauri.localhost".to_string(),
        ];

        // Helper to add an origin if not already present
        let mut add = |url: &str| {
            let trimmed = url.trim_end_matches('/').to_string();
            if !trimmed.is_empty() && !origins.contains(&trimmed) {
                origins.push(trimmed);
            }
        };

        // public_url from server config
        if let Some(ref url) = self.public_url {
            add(url);
        }

        // frontend_url from auth config (backward compat: existing configs have
        // `auth.frontend_url: https://ffs.dev` but no `server.public_url`)
        if let Some(ref auth_config) = self.auth_config {
            if let Some(ref frontend_url) = auth_config.frontend_url {
                add(frontend_url);
            }
        }

        origins
    }

    /// Validate an `origin` parameter and return the **redirect-safe** origin
    /// to use for constructing OAuth `redirect_uri`.
    ///
    /// - If `origin` is `Some`, check it against [`allowed_origins`].
    /// - Special case: `tauri://localhost` is a valid *browser* origin but OAuth
    ///   providers only accept `http://` or `https://` redirect URIs, so we map
    ///   it to `http://localhost:{server_port}` for the redirect_uri.
    /// - If `origin` is `None`, return `None` — the caller should fall back to the
    ///   static `redirect_uri` from config for backward compatibility.
    pub fn validate_origin(&self, origin: Option<&str>) -> Result<Option<String>, AppError> {
        match origin {
            None => Ok(None),
            Some(raw) => {
                let trimmed = raw.trim_end_matches('/');
                let allowed = self.allowed_origins();
                if allowed.iter().any(|a| a == trimmed) {
                    // tauri://localhost and https://tauri.localhost are valid for
                    // CORS but OAuth providers only accept standard http/https
                    // redirect URIs → map both to http://localhost:{port}
                    if trimmed == "tauri://localhost" || trimmed == "https://tauri.localhost" {
                        Ok(Some(format!("http://localhost:{}", self.server_port)))
                    } else {
                        Ok(Some(trimmed.to_string()))
                    }
                } else {
                    Err(AppError::BadRequest(format!(
                        "Unknown origin: {}. Allowed: {}",
                        trimmed,
                        allowed.join(", ")
                    )))
                }
            }
        }
    }
}

// ============================================================================
// Health check
// ============================================================================

/// Per-service health status in the health response
#[derive(Serialize)]
pub struct ServiceHealthStatus {
    pub neo4j: String,
    pub meilisearch: String,
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub services: Option<ServiceHealthStatus>,
}

/// Health check handler — verifies actual connectivity to Neo4j and Meilisearch.
///
/// Returns:
/// - 200 + `"ok"` if both Neo4j and Meilisearch are connected
/// - 200 + `"degraded"` if Neo4j is connected but Meilisearch is not
/// - 503 + `"unhealthy"` if Neo4j is disconnected (critical dependency)
pub async fn health(State(state): State<OrchestratorState>) -> (StatusCode, Json<HealthResponse>) {
    let neo4j_ok = state
        .orchestrator
        .neo4j()
        .health_check()
        .await
        .unwrap_or(false);
    let meili_ok = state
        .orchestrator
        .meili()
        .health_check()
        .await
        .unwrap_or(false);

    let status = if neo4j_ok && meili_ok {
        "ok"
    } else if neo4j_ok {
        "degraded"
    } else {
        "unhealthy"
    };

    let http_status = if neo4j_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        http_status,
        Json(HealthResponse {
            status: status.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            services: Some(ServiceHealthStatus {
                neo4j: if neo4j_ok {
                    "connected".to_string()
                } else {
                    "disconnected".to_string()
                },
                meilisearch: if meili_ok {
                    "connected".to_string()
                } else {
                    "disconnected".to_string()
                },
            }),
        }),
    )
}

// ============================================================================
// Version info
// ============================================================================

/// Feature flags exposed in the version endpoint
#[derive(Serialize)]
pub struct VersionFeatures {
    pub embedded_frontend: bool,
    pub serve_frontend: bool,
}

/// Build metadata exposed in the version endpoint
#[derive(Serialize)]
pub struct VersionBuild {
    pub target: String,
    pub profile: &'static str,
}

/// Full version response
#[derive(Serialize)]
pub struct VersionResponse {
    pub version: &'static str,
    pub features: VersionFeatures,
    pub build: VersionBuild,
}

/// GET /api/version — public endpoint returning server version and build info.
pub async fn get_version(State(state): State<OrchestratorState>) -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION"),
        features: VersionFeatures {
            embedded_frontend: cfg!(feature = "embedded-frontend"),
            serve_frontend: state.serve_frontend,
        },
        build: VersionBuild {
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
        },
    })
}

// ============================================================================
// Setup status
// ============================================================================

/// Setup status response
#[derive(Serialize)]
pub struct SetupStatusResponse {
    /// Whether the setup wizard has been completed
    pub configured: bool,
}

/// GET /api/setup-status — public endpoint to check if the app needs initial setup.
pub async fn setup_status(State(state): State<OrchestratorState>) -> Json<SetupStatusResponse> {
    Json(SetupStatusResponse {
        configured: state.setup_completed,
    })
}

// ============================================================================
// Plans
// ============================================================================

/// Query parameters for listing plans
#[derive(Debug, Deserialize, Default)]
pub struct PlansListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub status_filter: StatusFilter,
    #[serde(flatten)]
    pub priority_filter: PriorityFilter,
    #[serde(flatten)]
    pub search_filter: SearchFilter,
    /// Filter by project UUID
    pub project_id: Option<String>,
    /// Filter by workspace slug (plans of projects in this workspace)
    pub workspace_slug: Option<String>,
}

/// List all plans with optional pagination and filters
pub async fn list_plans(
    State(state): State<OrchestratorState>,
    Query(query): Query<PlansListQuery>,
) -> Result<Json<PaginatedResponse<PlanNode>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let project_id = query
        .project_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| {
            uuid::Uuid::parse_str(s)
                .map_err(|_| AppError::BadRequest("Invalid project_id UUID".to_string()))
        })
        .transpose()?;

    let (plans, total) = state
        .orchestrator
        .neo4j()
        .list_plans_filtered(
            project_id,
            query.workspace_slug.as_deref(),
            query.status_filter.to_vec(),
            query.priority_filter.priority_min,
            query.priority_filter.priority_max,
            query.search_filter.search.as_deref(),
            query.pagination.validated_limit(),
            query.pagination.offset,
            query.pagination.sort_by.as_deref(),
            &query.pagination.sort_order,
        )
        .await?;

    Ok(Json(PaginatedResponse::new(
        plans,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// Create a new plan
pub async fn create_plan(
    State(state): State<OrchestratorState>,
    Json(req): Json<CreatePlanRequest>,
) -> Result<Json<PlanNode>, AppError> {
    let plan = state
        .orchestrator
        .plan_manager()
        .create_plan(req, "orchestrator")
        .await?;
    Ok(Json(plan))
}

/// Get plan details
pub async fn get_plan(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<PlanDetails>, AppError> {
    let details = state
        .orchestrator
        .plan_manager()
        .get_plan_details(plan_id)
        .await?
        .ok_or(AppError::NotFound("Plan not found".into()))?;
    Ok(Json(details))
}

/// Update plan status
#[derive(Deserialize)]
pub struct UpdatePlanStatusRequest {
    pub status: PlanStatus,
}

/// Request to link a plan to a project
#[derive(Deserialize)]
pub struct LinkPlanToProjectRequest {
    pub project_id: Uuid,
}

/// Link a plan to a project
pub async fn link_plan_to_project(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(req): Json<LinkPlanToProjectRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .link_plan_to_project(plan_id, req.project_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Unlink a plan from its project
pub async fn unlink_plan_from_project(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.unlink_plan_from_project(plan_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn update_plan_status(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(req): Json<UpdatePlanStatusRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .plan_manager()
        .update_plan_status(plan_id, req.status)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a plan and all its related data
pub async fn delete_plan(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .plan_manager()
        .delete_plan(plan_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Tasks
// ============================================================================

/// Add a task to a plan
pub async fn add_task(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(req): Json<CreateTaskRequest>,
) -> Result<Json<TaskNode>, AppError> {
    let task = state
        .orchestrator
        .plan_manager()
        .add_task(plan_id, req)
        .await?;
    Ok(Json(task))
}

/// Get task details
pub async fn get_task(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<TaskDetails>, AppError> {
    let details = state
        .orchestrator
        .plan_manager()
        .get_task_details(task_id)
        .await?
        .ok_or(AppError::NotFound("Task not found".into()))?;
    Ok(Json(details))
}

/// Delete a task and all its related data
pub async fn delete_task(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .plan_manager()
        .delete_task(task_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Update task status
pub async fn update_task(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
    Json(req): Json<UpdateTaskRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .plan_manager()
        .update_task(task_id, req)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get next available task
pub async fn get_next_task(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<Option<TaskNode>>, AppError> {
    let task = state
        .orchestrator
        .plan_manager()
        .get_next_available_task(plan_id)
        .await?;
    Ok(Json(task))
}

/// Query parameters for listing all tasks
#[derive(Debug, Deserialize, Default)]
pub struct TasksListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub status_filter: StatusFilter,
    #[serde(flatten)]
    pub priority_filter: PriorityFilter,
    #[serde(flatten)]
    pub tags_filter: TagsFilter,
    /// Filter by plan ID
    pub plan_id: Option<Uuid>,
    /// Filter by assigned agent/user
    pub assigned_to: Option<String>,
    /// Filter by project UUID (tasks belonging to plans of this project)
    pub project_id: Option<Uuid>,
    /// Filter by workspace slug (tasks belonging to plans of projects in this workspace)
    pub workspace_slug: Option<String>,
}

/// List all tasks across all plans with optional filters
pub async fn list_all_tasks(
    State(state): State<OrchestratorState>,
    Query(query): Query<TasksListQuery>,
) -> Result<Json<PaginatedResponse<TaskWithPlan>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let (tasks, total) = state
        .orchestrator
        .neo4j()
        .list_all_tasks_filtered(
            query.plan_id,
            query.project_id,
            query.workspace_slug.as_deref(),
            query.status_filter.to_vec(),
            query.priority_filter.priority_min,
            query.priority_filter.priority_max,
            query.tags_filter.to_vec(),
            query.assigned_to.as_deref(),
            query.pagination.validated_limit(),
            query.pagination.offset,
            query.pagination.sort_by.as_deref(),
            &query.pagination.sort_order,
        )
        .await?;

    Ok(Json(PaginatedResponse::new(
        tasks,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

// ============================================================================
// Context
// ============================================================================

/// Get context for a task
pub async fn get_task_context(
    State(state): State<OrchestratorState>,
    Path((plan_id, task_id)): Path<(Uuid, Uuid)>,
) -> Result<Json<AgentContext>, AppError> {
    let context = state
        .orchestrator
        .context_builder()
        .build_context(task_id, plan_id)
        .await?;
    Ok(Json(context))
}

/// Get generated prompt for a task
#[derive(Serialize)]
pub struct PromptResponse {
    pub prompt: String,
}

pub async fn get_task_prompt(
    State(state): State<OrchestratorState>,
    Path((plan_id, task_id)): Path<(Uuid, Uuid)>,
) -> Result<Json<PromptResponse>, AppError> {
    let context = state
        .orchestrator
        .context_builder()
        .build_context(task_id, plan_id)
        .await?;
    let prompt = state
        .orchestrator
        .context_builder()
        .generate_prompt(&context);
    Ok(Json(PromptResponse { prompt }))
}

// ============================================================================
// Decisions
// ============================================================================

/// Add a decision to a task
pub async fn add_decision(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
    Json(req): Json<CreateDecisionRequest>,
) -> Result<Json<DecisionNode>, AppError> {
    let decision = state
        .orchestrator
        .plan_manager()
        .add_decision(task_id, req, "agent")
        .await?;
    Ok(Json(decision))
}

/// Get a single decision by ID
pub async fn get_decision(
    State(state): State<OrchestratorState>,
    Path(decision_id): Path<Uuid>,
) -> Result<Json<DecisionNode>, AppError> {
    let decision = state
        .orchestrator
        .neo4j()
        .get_decision(decision_id)
        .await?
        .ok_or(AppError::NotFound("Decision not found".into()))?;
    Ok(Json(decision))
}

/// Request to update a decision
#[derive(Deserialize)]
pub struct UpdateDecisionRequest {
    pub description: Option<String>,
    pub rationale: Option<String>,
    pub chosen_option: Option<String>,
}

/// Update a decision
pub async fn update_decision(
    State(state): State<OrchestratorState>,
    Path(decision_id): Path<Uuid>,
    Json(req): Json<UpdateDecisionRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .update_decision(
            decision_id,
            req.description,
            req.rationale,
            req.chosen_option,
        )
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a decision
pub async fn delete_decision(
    State(state): State<OrchestratorState>,
    Path(decision_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_decision(decision_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Search decisions
#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    pub limit: Option<usize>,
}

pub async fn search_decisions(
    State(state): State<OrchestratorState>,
    axum::extract::Query(query): axum::extract::Query<SearchQuery>,
) -> Result<Json<Vec<DecisionNode>>, AppError> {
    let decisions = state
        .orchestrator
        .plan_manager()
        .search_decisions(&query.q, query.limit.unwrap_or(10))
        .await?;
    Ok(Json(decisions))
}

// ============================================================================
// Sync
// ============================================================================

/// Sync request
#[derive(Deserialize)]
pub struct SyncRequest {
    pub path: String,
    pub project_id: Option<String>,
}

/// Sync response
#[derive(Serialize)]
pub struct SyncResponse {
    pub files_synced: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub symbols_deleted: usize,
    pub errors: usize,
}

/// Sync a directory to the knowledge base
pub async fn sync_directory(
    State(state): State<OrchestratorState>,
    Json(req): Json<SyncRequest>,
) -> Result<Json<SyncResponse>, AppError> {
    let path = std::path::Path::new(&req.path);

    // Resolve project context when project_id is provided
    let (project_id, project_slug) = if let Some(ref pid_str) = req.project_id {
        let pid = uuid::Uuid::parse_str(pid_str)
            .map_err(|_| AppError::BadRequest(format!("Invalid project_id: {}", pid_str)))?;
        let project = state
            .orchestrator
            .neo4j()
            .get_project(pid)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", pid_str)))?;
        (Some(project.id), Some(project.slug))
    } else {
        (None, None)
    };

    let result = state
        .orchestrator
        .sync_directory_for_project(path, project_id, project_slug.as_deref())
        .await?;

    // Update last_synced when project context is available
    if let Some(pid) = project_id {
        state
            .orchestrator
            .neo4j()
            .update_project_synced(pid)
            .await?;
    }

    Ok(Json(SyncResponse {
        files_synced: result.files_synced,
        files_skipped: result.files_skipped,
        files_deleted: result.files_deleted,
        symbols_deleted: result.symbols_deleted,
        errors: result.errors,
    }))
}

// ============================================================================
// Webhooks
// ============================================================================

/// Wake callback from an agent
#[derive(Deserialize)]
pub struct WakeRequest {
    pub task_id: Uuid,
    pub success: bool,
    pub summary: String,
    pub files_modified: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct WakeResponse {
    pub acknowledged: bool,
}

/// Handle wake callback from agent
pub async fn wake(
    State(state): State<OrchestratorState>,
    Json(req): Json<WakeRequest>,
) -> Result<Json<WakeResponse>, AppError> {
    state
        .orchestrator
        .handle_task_completion(
            req.task_id,
            req.success,
            &req.summary,
            &req.files_modified.unwrap_or_default(),
        )
        .await?;

    Ok(Json(WakeResponse { acknowledged: true }))
}

// ============================================================================
// Internal Events (MCP → HTTP bridge) — DEPRECATED, use NATS instead
// ============================================================================

/// POST /internal/events — Receive a CrudEvent from the MCP server and broadcast it.
///
/// **DEPRECATED**: This endpoint was used by `EventNotifier` (HTTP bridge) before
/// NATS-based inter-process sync was added. Use `NATS_URL` for cross-instance
/// event delivery instead. This endpoint will be removed in a future version.
pub async fn receive_event(
    State(state): State<OrchestratorState>,
    Json(event): Json<crate::events::CrudEvent>,
) -> Result<StatusCode, AppError> {
    tracing::debug!(
        entity_type = ?event.entity_type,
        action = ?event.action,
        entity_id = %event.entity_id,
        "Received internal event from MCP"
    );
    state.event_bus.emit(event);
    Ok(StatusCode::OK)
}

// ============================================================================
// File Watcher
// ============================================================================

/// Watch request
#[derive(Deserialize)]
pub struct WatchRequest {
    pub path: String,
    pub project_id: Option<String>,
}

/// Watch status response
#[derive(Serialize)]
pub struct WatchStatusResponse {
    pub running: bool,
    pub watched_paths: Vec<String>,
}

/// Start watching a directory
pub async fn start_watch(
    State(state): State<OrchestratorState>,
    Json(req): Json<WatchRequest>,
) -> Result<Json<WatchStatusResponse>, AppError> {
    let expanded = crate::expand_tilde(&req.path);
    let path = std::path::Path::new(&expanded);
    let mut watcher = state.watcher.write().await;

    // Register project context if project_id is provided (multi-project aware)
    if let Some(ref pid_str) = req.project_id {
        if let Ok(pid) = uuid::Uuid::parse_str(pid_str) {
            if let Some(project) = state.orchestrator.neo4j().get_project(pid).await? {
                watcher
                    .register_project(path, project.id, project.slug)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to register project: {}", e))?;
            }
        }
    } else {
        // Legacy: watch without project context
        watcher.watch(path).await?;
    }

    watcher.start().await?;

    let paths = watcher.watched_paths().await;
    Ok(Json(WatchStatusResponse {
        running: true,
        watched_paths: paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
    }))
}

/// Stop the file watcher
pub async fn stop_watch(
    State(state): State<OrchestratorState>,
) -> Result<Json<WatchStatusResponse>, AppError> {
    let mut watcher = state.watcher.write().await;
    watcher.stop().await;

    Ok(Json(WatchStatusResponse {
        running: false,
        watched_paths: vec![],
    }))
}

/// Get watcher status
pub async fn watch_status(
    State(state): State<OrchestratorState>,
) -> Result<Json<WatchStatusResponse>, AppError> {
    let watcher = state.watcher.read().await;
    let paths = watcher.watched_paths().await;

    Ok(Json(WatchStatusResponse {
        running: !paths.is_empty(),
        watched_paths: paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
    }))
}

// ============================================================================
// Steps
// ============================================================================

/// Add a step to a task
pub async fn add_step(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
    Json(req): Json<AddStepRequest>,
) -> Result<Json<StepNode>, AppError> {
    // Get current step count to determine order
    let steps = state.orchestrator.neo4j().get_task_steps(task_id).await?;
    let order = steps.len() as u32;

    let step = StepNode::new(order, req.description, req.verification);
    state
        .orchestrator
        .plan_manager()
        .add_step(task_id, &step)
        .await?;

    Ok(Json(step))
}

/// Get steps for a task
pub async fn get_task_steps(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<Vec<StepNode>>, AppError> {
    let steps = state.orchestrator.neo4j().get_task_steps(task_id).await?;
    Ok(Json(steps))
}

/// Get a single step by ID
pub async fn get_step(
    State(state): State<OrchestratorState>,
    Path(step_id): Path<Uuid>,
) -> Result<Json<StepNode>, AppError> {
    let step = state
        .orchestrator
        .neo4j()
        .get_step(step_id)
        .await?
        .ok_or(AppError::NotFound("Step not found".into()))?;
    Ok(Json(step))
}

/// Delete a step
pub async fn delete_step(
    State(state): State<OrchestratorState>,
    Path(step_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_step(step_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Update a step
pub async fn update_step(
    State(state): State<OrchestratorState>,
    Path(step_id): Path<Uuid>,
    Json(req): Json<UpdateStepRequest>,
) -> Result<StatusCode, AppError> {
    if let Some(status) = req.status {
        state
            .orchestrator
            .plan_manager()
            .update_step_status(step_id, status)
            .await?;
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Step progress response
#[derive(Serialize)]
pub struct StepProgressResponse {
    pub completed: u32,
    pub total: u32,
    pub percentage: f32,
}

/// Get step progress for a task
pub async fn get_step_progress(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<StepProgressResponse>, AppError> {
    let (completed, total) = state
        .orchestrator
        .neo4j()
        .get_task_step_progress(task_id)
        .await?;

    let percentage = if total > 0 {
        (completed as f32 / total as f32) * 100.0
    } else {
        0.0
    };

    Ok(Json(StepProgressResponse {
        completed,
        total,
        percentage,
    }))
}

// ============================================================================
// Constraints
// ============================================================================

/// Add a constraint to a plan
pub async fn add_constraint(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(req): Json<AddConstraintRequest>,
) -> Result<Json<ConstraintNode>, AppError> {
    let constraint = ConstraintNode::new(req.constraint_type, req.description, req.enforced_by);
    state
        .orchestrator
        .plan_manager()
        .add_constraint(plan_id, &constraint)
        .await?;

    Ok(Json(constraint))
}

/// Get constraints for a plan
pub async fn get_plan_constraints(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<Vec<ConstraintNode>>, AppError> {
    let constraints = state
        .orchestrator
        .neo4j()
        .get_plan_constraints(plan_id)
        .await?;
    Ok(Json(constraints))
}

/// Get a single constraint by ID
pub async fn get_constraint(
    State(state): State<OrchestratorState>,
    Path(constraint_id): Path<Uuid>,
) -> Result<Json<ConstraintNode>, AppError> {
    let constraint = state
        .orchestrator
        .neo4j()
        .get_constraint(constraint_id)
        .await?
        .ok_or(AppError::NotFound("Constraint not found".into()))?;
    Ok(Json(constraint))
}

/// Request to update a constraint
#[derive(Deserialize)]
pub struct UpdateConstraintRequest {
    pub description: Option<String>,
    pub constraint_type: Option<crate::neo4j::models::ConstraintType>,
    pub enforced_by: Option<String>,
}

/// Update a constraint
pub async fn update_constraint(
    State(state): State<OrchestratorState>,
    Path(constraint_id): Path<Uuid>,
    Json(req): Json<UpdateConstraintRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .update_constraint(
            constraint_id,
            req.description,
            req.constraint_type,
            req.enforced_by,
        )
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a constraint
pub async fn delete_constraint(
    State(state): State<OrchestratorState>,
    Path(constraint_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_constraint(constraint_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Meilisearch maintenance
// ============================================================================

/// Response for cleanup operations
#[derive(Serialize)]
pub struct CleanupResponse {
    pub success: bool,
    pub message: String,
}

/// Response for index stats
#[derive(Serialize)]
pub struct MeiliStatsResponse {
    pub code_documents: usize,
    pub is_indexing: bool,
}

/// Delete orphan documents from Meilisearch (documents without project_id)
pub async fn delete_meilisearch_orphans(
    State(state): State<OrchestratorState>,
) -> Result<Json<CleanupResponse>, AppError> {
    state
        .orchestrator
        .meili()
        .delete_orphan_code_documents()
        .await?;

    Ok(Json(CleanupResponse {
        success: true,
        message: "Orphan documents deleted".to_string(),
    }))
}

/// Get Meilisearch code index statistics
pub async fn get_meilisearch_stats(
    State(state): State<OrchestratorState>,
) -> Result<Json<MeiliStatsResponse>, AppError> {
    let stats = state.orchestrator.meili().get_code_stats().await?;

    Ok(Json(MeiliStatsResponse {
        code_documents: stats.total_documents,
        is_indexing: stats.is_indexing,
    }))
}

// ============================================================================
// Commits
// ============================================================================

/// Request to create a commit
#[derive(Deserialize)]
pub struct CreateCommitRequest {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

/// Create a new commit
pub async fn create_commit(
    State(state): State<OrchestratorState>,
    Json(req): Json<CreateCommitRequest>,
) -> Result<Json<CommitNode>, AppError> {
    let commit = CommitNode {
        hash: req.hash,
        message: req.message,
        author: req.author,
        timestamp: req.timestamp.unwrap_or_else(chrono::Utc::now),
    };

    state.orchestrator.create_commit(&commit).await?;
    Ok(Json(commit))
}

/// Request to link a commit to a task
#[derive(Deserialize)]
pub struct LinkCommitRequest {
    pub commit_hash: String,
}

/// Link a commit to a task
pub async fn link_commit_to_task(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
    Json(req): Json<LinkCommitRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .link_commit_to_task(&req.commit_hash, task_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get commits for a task
pub async fn get_task_commits(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<Vec<CommitNode>>, AppError> {
    let commits = state.orchestrator.neo4j().get_task_commits(task_id).await?;
    Ok(Json(commits))
}

/// Link a commit to a plan
pub async fn link_commit_to_plan(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(req): Json<LinkCommitRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .link_commit_to_plan(&req.commit_hash, plan_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get commits for a plan
pub async fn get_plan_commits(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<Vec<CommitNode>>, AppError> {
    let commits = state.orchestrator.neo4j().get_plan_commits(plan_id).await?;
    Ok(Json(commits))
}

// ============================================================================
// Releases
// ============================================================================

/// Request to create a release
#[derive(Deserialize)]
pub struct CreateReleaseRequest {
    pub version: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub target_date: Option<chrono::DateTime<chrono::Utc>>,
}

/// Create a release for a project
pub async fn create_release(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
    Json(req): Json<CreateReleaseRequest>,
) -> Result<Json<ReleaseNode>, AppError> {
    let release = ReleaseNode {
        id: Uuid::new_v4(),
        version: req.version,
        title: req.title,
        description: req.description,
        status: ReleaseStatus::Planned,
        target_date: req.target_date,
        released_at: None,
        created_at: chrono::Utc::now(),
        project_id,
    };

    state.orchestrator.create_release(&release).await?;
    Ok(Json(release))
}

/// List releases for a project
/// Query parameters for listing releases
#[derive(Debug, Deserialize, Default)]
pub struct ReleasesListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub status_filter: StatusFilter,
}

pub async fn list_releases(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
    Query(query): Query<ReleasesListQuery>,
) -> Result<Json<PaginatedResponse<ReleaseNode>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let (releases, total) = state
        .orchestrator
        .neo4j()
        .list_releases_filtered(
            project_id,
            query.status_filter.to_vec(),
            query.pagination.validated_limit(),
            query.pagination.offset,
            query.pagination.sort_by.as_deref(),
            &query.pagination.sort_order,
        )
        .await?;

    Ok(Json(PaginatedResponse::new(
        releases,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// Request to update a release
#[derive(Deserialize)]
pub struct UpdateReleaseRequest {
    pub status: Option<ReleaseStatus>,
    pub target_date: Option<chrono::DateTime<chrono::Utc>>,
    pub released_at: Option<chrono::DateTime<chrono::Utc>>,
    pub title: Option<String>,
    pub description: Option<String>,
}

/// Update a release
pub async fn update_release(
    State(state): State<OrchestratorState>,
    Path(release_id): Path<Uuid>,
    Json(req): Json<UpdateReleaseRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .update_release(
            release_id,
            req.status,
            req.target_date,
            req.released_at,
            req.title,
            req.description,
        )
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a release
pub async fn delete_release(
    State(state): State<OrchestratorState>,
    Path(release_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_release(release_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Request to add a task to a release
#[derive(Deserialize)]
pub struct AddTaskToReleaseRequest {
    pub task_id: Uuid,
}

/// Add a task to a release
pub async fn add_task_to_release(
    State(state): State<OrchestratorState>,
    Path(release_id): Path<Uuid>,
    Json(req): Json<AddTaskToReleaseRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .add_task_to_release(release_id, req.task_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Request to add a commit to a release
#[derive(Deserialize)]
pub struct AddCommitToReleaseRequest {
    pub commit_hash: String,
}

/// Add a commit to a release
pub async fn add_commit_to_release(
    State(state): State<OrchestratorState>,
    Path(release_id): Path<Uuid>,
    Json(req): Json<AddCommitToReleaseRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .add_commit_to_release(release_id, &req.commit_hash)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Remove a commit from a release
pub async fn remove_commit_from_release(
    State(state): State<OrchestratorState>,
    Path((release_id, commit_sha)): Path<(Uuid, String)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .remove_commit_from_release(release_id, &commit_sha)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Release details response
#[derive(Serialize)]
pub struct ReleaseDetailsResponse {
    pub release: ReleaseNode,
    pub tasks: Vec<TaskNode>,
    pub commits: Vec<CommitNode>,
}

/// Get release details
pub async fn get_release(
    State(state): State<OrchestratorState>,
    Path(release_id): Path<Uuid>,
) -> Result<Json<ReleaseDetailsResponse>, AppError> {
    let details = state
        .orchestrator
        .neo4j()
        .get_release_details(release_id)
        .await?
        .ok_or(AppError::NotFound("Release not found".into()))?;

    Ok(Json(ReleaseDetailsResponse {
        release: details.0,
        tasks: details.1,
        commits: details.2,
    }))
}

// ============================================================================
// Milestones
// ============================================================================

/// Request to create a milestone
#[derive(Deserialize)]
pub struct CreateMilestoneRequest {
    pub title: String,
    pub description: Option<String>,
    pub target_date: Option<chrono::DateTime<chrono::Utc>>,
}

/// Create a milestone for a project
pub async fn create_milestone(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
    Json(req): Json<CreateMilestoneRequest>,
) -> Result<Json<MilestoneNode>, AppError> {
    let milestone = MilestoneNode {
        id: Uuid::new_v4(),
        title: req.title,
        description: req.description,
        status: MilestoneStatus::Open,
        target_date: req.target_date,
        closed_at: None,
        created_at: chrono::Utc::now(),
        project_id,
    };

    state.orchestrator.create_milestone(&milestone).await?;
    Ok(Json(milestone))
}

/// List milestones for a project
/// Query parameters for listing milestones
#[derive(Debug, Deserialize, Default)]
pub struct MilestonesListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub status_filter: StatusFilter,
}

pub async fn list_milestones(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
    Query(query): Query<MilestonesListQuery>,
) -> Result<Json<PaginatedResponse<MilestoneNode>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let (milestones, total) = state
        .orchestrator
        .neo4j()
        .list_milestones_filtered(
            project_id,
            query.status_filter.to_vec(),
            query.pagination.validated_limit(),
            query.pagination.offset,
            query.pagination.sort_by.as_deref(),
            &query.pagination.sort_order,
        )
        .await?;

    Ok(Json(PaginatedResponse::new(
        milestones,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// Request to update a milestone
#[derive(Deserialize)]
pub struct UpdateMilestoneRequest {
    pub status: Option<MilestoneStatus>,
    pub target_date: Option<chrono::DateTime<chrono::Utc>>,
    pub closed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub title: Option<String>,
    pub description: Option<String>,
}

/// Update a milestone
pub async fn update_milestone(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
    Json(req): Json<UpdateMilestoneRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .update_milestone(
            milestone_id,
            req.status,
            req.target_date,
            req.closed_at,
            req.title,
            req.description,
        )
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a milestone
pub async fn delete_milestone(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_milestone(milestone_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Request to add a task to a milestone
#[derive(Deserialize)]
pub struct AddTaskToMilestoneRequest {
    pub task_id: Uuid,
}

/// Add a task to a milestone
pub async fn add_task_to_milestone(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
    Json(req): Json<AddTaskToMilestoneRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .add_task_to_milestone(milestone_id, req.task_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Request body for linking a plan to a milestone
#[derive(Deserialize)]
pub struct LinkPlanToMilestoneRequest {
    pub plan_id: Uuid,
}

/// Link a plan to a project milestone
pub async fn link_plan_to_milestone(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
    Json(req): Json<LinkPlanToMilestoneRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .link_plan_to_milestone(req.plan_id, milestone_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Unlink a plan from a project milestone
pub async fn unlink_plan_from_milestone(
    State(state): State<OrchestratorState>,
    Path((milestone_id, plan_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .unlink_plan_from_milestone(plan_id, milestone_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Milestone details response
#[derive(Serialize)]
pub struct MilestoneDetailsResponse {
    pub milestone: MilestoneNode,
    pub tasks: Vec<TaskNode>,
}

/// Get milestone details
pub async fn get_milestone(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
) -> Result<Json<MilestoneDetailsResponse>, AppError> {
    let details = state
        .orchestrator
        .neo4j()
        .get_milestone_details(milestone_id)
        .await?
        .ok_or(AppError::NotFound("Milestone not found".into()))?;

    Ok(Json(MilestoneDetailsResponse {
        milestone: details.0,
        tasks: details.1,
    }))
}

/// Milestone progress response
#[derive(Serialize)]
pub struct MilestoneProgressResponse {
    pub completed: u32,
    pub total: u32,
    pub percentage: f32,
}

/// Get milestone progress
pub async fn get_milestone_progress(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
) -> Result<Json<MilestoneProgressResponse>, AppError> {
    let (completed, total) = state
        .orchestrator
        .neo4j()
        .get_milestone_progress(milestone_id)
        .await?;

    let percentage = if total > 0 {
        (completed as f32 / total as f32) * 100.0
    } else {
        0.0
    };

    Ok(Json(MilestoneProgressResponse {
        completed,
        total,
        percentage,
    }))
}

// ============================================================================
// Task Dependencies
// ============================================================================

/// Request to add dependencies to a task
#[derive(Deserialize)]
pub struct AddDependenciesRequest {
    pub depends_on: Vec<Uuid>,
}

/// Add dependencies to a task
pub async fn add_task_dependencies(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
    Json(req): Json<AddDependenciesRequest>,
) -> Result<StatusCode, AppError> {
    for dep_id in req.depends_on {
        state
            .orchestrator
            .add_task_dependency(task_id, dep_id)
            .await?;
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Remove a dependency from a task
pub async fn remove_task_dependency(
    State(state): State<OrchestratorState>,
    Path((task_id, dep_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .remove_task_dependency(task_id, dep_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get tasks that block this task (uncompleted dependencies)
pub async fn get_task_blockers(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<Vec<TaskNode>>, AppError> {
    let blockers = state
        .orchestrator
        .neo4j()
        .get_task_blockers(task_id)
        .await?;
    Ok(Json(blockers))
}

/// Get tasks blocked by this task
pub async fn get_tasks_blocked_by(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<Vec<TaskNode>>, AppError> {
    let blocked = state
        .orchestrator
        .neo4j()
        .get_tasks_blocked_by(task_id)
        .await?;
    Ok(Json(blocked))
}

/// Dependency graph node for visualization
#[derive(Serialize)]
pub struct DependencyGraphNode {
    pub id: Uuid,
    pub title: Option<String>,
    pub description: String,
    pub status: String,
    pub priority: Option<i32>,
}

/// Dependency graph edge
#[derive(Serialize)]
pub struct DependencyGraphEdge {
    pub from: Uuid,
    pub to: Uuid,
}

/// Dependency graph response
#[derive(Serialize)]
pub struct DependencyGraphResponse {
    pub nodes: Vec<DependencyGraphNode>,
    pub edges: Vec<DependencyGraphEdge>,
}

/// Get dependency graph for a plan
pub async fn get_plan_dependency_graph(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<DependencyGraphResponse>, AppError> {
    let (tasks, edges) = state
        .orchestrator
        .neo4j()
        .get_plan_dependency_graph(plan_id)
        .await?;

    let nodes: Vec<DependencyGraphNode> = tasks
        .into_iter()
        .map(|t| DependencyGraphNode {
            id: t.id,
            title: t.title,
            description: t.description,
            status: format!("{:?}", t.status),
            priority: t.priority,
        })
        .collect();

    let edges: Vec<DependencyGraphEdge> = edges
        .into_iter()
        .map(|(from, to)| DependencyGraphEdge { from, to })
        .collect();

    Ok(Json(DependencyGraphResponse { nodes, edges }))
}

/// Critical path response
#[derive(Serialize)]
pub struct CriticalPathResponse {
    pub tasks: Vec<TaskNode>,
    pub length: usize,
}

/// Get critical path for a plan
pub async fn get_plan_critical_path(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<CriticalPathResponse>, AppError> {
    let tasks = state
        .orchestrator
        .neo4j()
        .get_plan_critical_path(plan_id)
        .await?;

    let length = tasks.len();
    Ok(Json(CriticalPathResponse { tasks, length }))
}

// ============================================================================
// Roadmap
// ============================================================================

/// Milestone with tasks for roadmap view
#[derive(Serialize)]
pub struct RoadmapMilestone {
    pub milestone: MilestoneNode,
    pub tasks: Vec<TaskNode>,
    pub progress: MilestoneProgressResponse,
}

/// Release with tasks and commits for roadmap view
#[derive(Serialize)]
pub struct RoadmapRelease {
    pub release: ReleaseNode,
    pub tasks: Vec<TaskNode>,
    pub commits: Vec<CommitNode>,
}

/// Project progress stats
#[derive(Serialize)]
pub struct ProjectProgress {
    pub total_tasks: u32,
    pub completed_tasks: u32,
    pub in_progress_tasks: u32,
    pub pending_tasks: u32,
    pub percentage: f32,
}

/// Roadmap response
#[derive(Serialize)]
pub struct RoadmapResponse {
    pub milestones: Vec<RoadmapMilestone>,
    pub releases: Vec<RoadmapRelease>,
    pub progress: ProjectProgress,
    pub dependency_graph: DependencyGraphResponse,
}

/// Get project roadmap
pub async fn get_project_roadmap(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
) -> Result<Json<RoadmapResponse>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    // Get milestones with their tasks and progress
    let milestone_nodes = neo4j.list_project_milestones(project_id).await?;
    let mut milestones = Vec::new();
    for m in milestone_nodes {
        let tasks = neo4j.get_milestone_tasks(m.id).await?;
        let (completed, total) = neo4j.get_milestone_progress(m.id).await?;
        let percentage = if total > 0 {
            (completed as f32 / total as f32) * 100.0
        } else {
            0.0
        };
        milestones.push(RoadmapMilestone {
            milestone: m,
            tasks,
            progress: MilestoneProgressResponse {
                completed,
                total,
                percentage,
            },
        });
    }

    // Get releases with their tasks and commits
    let release_nodes = neo4j.list_project_releases(project_id).await?;
    let mut releases = Vec::new();
    for r in release_nodes {
        let details = neo4j.get_release_details(r.id).await?;
        if let Some((release, tasks, commits)) = details {
            releases.push(RoadmapRelease {
                release,
                tasks,
                commits,
            });
        }
    }

    // Get project progress
    let (total, completed, in_progress, pending) = neo4j.get_project_progress(project_id).await?;
    let percentage = if total > 0 {
        (completed as f32 / total as f32) * 100.0
    } else {
        0.0
    };
    let progress = ProjectProgress {
        total_tasks: total,
        completed_tasks: completed,
        in_progress_tasks: in_progress,
        pending_tasks: pending,
        percentage,
    };

    // Get dependency graph for all tasks in the project
    let all_tasks = neo4j.get_project_tasks(project_id).await?;
    let edges = neo4j.get_project_task_dependencies(project_id).await?;

    let nodes: Vec<DependencyGraphNode> = all_tasks
        .into_iter()
        .map(|t| DependencyGraphNode {
            id: t.id,
            title: t.title,
            description: t.description,
            status: format!("{:?}", t.status),
            priority: t.priority,
        })
        .collect();

    let edges: Vec<DependencyGraphEdge> = edges
        .into_iter()
        .map(|(from, to)| DependencyGraphEdge { from, to })
        .collect();

    let dependency_graph = DependencyGraphResponse { nodes, edges };

    Ok(Json(RoadmapResponse {
        milestones,
        releases,
        progress,
        dependency_graph,
    }))
}

// ============================================================================
// Error handling
// ============================================================================

/// Application error type
#[derive(Debug)]
pub enum AppError {
    Internal(anyhow::Error),
    NotFound(String),
    BadRequest(String),
    Unauthorized(String),
    Forbidden(String),
    Conflict(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            AppError::Internal(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            AppError::Forbidden(msg) => (StatusCode::FORBIDDEN, msg),
            AppError::Conflict(msg) => (StatusCode::CONFLICT, msg),
        };

        let body = Json(serde_json::json!({
            "error": message
        }));

        (status, body).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::Internal(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_decision_request_all_fields() {
        let json = r#"{"description":"new desc","rationale":"new reason","chosen_option":"B"}"#;
        let req: UpdateDecisionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.description, Some("new desc".to_string()));
        assert_eq!(req.rationale, Some("new reason".to_string()));
        assert_eq!(req.chosen_option, Some("B".to_string()));
    }

    #[test]
    fn test_update_decision_request_partial() {
        let json = r#"{"description":"only desc"}"#;
        let req: UpdateDecisionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.description, Some("only desc".to_string()));
        assert_eq!(req.rationale, None);
        assert_eq!(req.chosen_option, None);
    }

    #[test]
    fn test_update_decision_request_empty() {
        let json = r#"{}"#;
        let req: UpdateDecisionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.description, None);
        assert_eq!(req.rationale, None);
        assert_eq!(req.chosen_option, None);
    }

    #[test]
    fn test_update_constraint_request_with_enum() {
        let json = r#"{"constraint_type":"performance","description":"Must be fast"}"#;
        let req: UpdateConstraintRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.constraint_type,
            Some(crate::neo4j::models::ConstraintType::Performance)
        );
        assert_eq!(req.description, Some("Must be fast".to_string()));
        assert_eq!(req.enforced_by, None);
    }

    #[test]
    fn test_update_constraint_request_all_enum_variants() {
        for variant in ["performance", "security", "style", "compatibility", "other"] {
            let json = format!(r#"{{"constraint_type":"{}"}}"#, variant);
            let req: UpdateConstraintRequest = serde_json::from_str(&json).unwrap();
            assert!(
                req.constraint_type.is_some(),
                "Failed to parse constraint_type: {}",
                variant
            );
        }
    }

    #[test]
    fn test_update_constraint_request_empty() {
        let json = r#"{}"#;
        let req: UpdateConstraintRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.description, None);
        assert_eq!(req.constraint_type, None);
        assert_eq!(req.enforced_by, None);
    }

    #[test]
    fn test_plans_list_query_defaults() {
        let json = r#"{}"#;
        let query: PlansListQuery = serde_json::from_str(json).unwrap();
        assert!(query.project_id.is_none());
        assert!(query.status_filter.status.is_none());
        assert!(query.search_filter.search.is_none());
    }

    #[test]
    fn test_plans_list_query_with_project_id() {
        let json = r#"{"project_id":"e83b0663-9600-450d-9f63-234e857394df"}"#;
        let query: PlansListQuery = serde_json::from_str(json).unwrap();
        assert_eq!(
            query.project_id,
            Some("e83b0663-9600-450d-9f63-234e857394df".to_string())
        );
    }

    #[test]
    fn test_plans_list_query_with_all_filters() {
        let json = r#"{"project_id":"e83b0663-9600-450d-9f63-234e857394df","status":"draft,in_progress","priority_min":"5","search":"auth"}"#;
        let query: PlansListQuery = serde_json::from_str(json).unwrap();
        assert_eq!(
            query.project_id,
            Some("e83b0663-9600-450d-9f63-234e857394df".to_string())
        );
        assert_eq!(
            query.status_filter.status,
            Some("draft,in_progress".to_string())
        );
        assert_eq!(query.priority_filter.priority_min, Some(5));
        assert_eq!(query.search_filter.search, Some("auth".to_string()));
    }

    #[test]
    fn test_plans_list_query_project_id_uuid_validation() {
        // Valid UUID
        let json = r#"{"project_id":"e83b0663-9600-450d-9f63-234e857394df"}"#;
        let query: PlansListQuery = serde_json::from_str(json).unwrap();
        let parsed = query
            .project_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(uuid::Uuid::parse_str);
        assert!(parsed.is_some());
        assert!(parsed.unwrap().is_ok());
    }

    #[test]
    fn test_plans_list_query_project_id_invalid_uuid() {
        let json = r#"{"project_id":"not-valid"}"#;
        let query: PlansListQuery = serde_json::from_str(json).unwrap();
        let parsed = query
            .project_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(uuid::Uuid::parse_str);
        assert!(parsed.is_some());
        assert!(parsed.unwrap().is_err());
    }

    #[test]
    fn test_plans_list_query_project_id_empty_string() {
        let json = r#"{"project_id":""}"#;
        let query: PlansListQuery = serde_json::from_str(json).unwrap();
        // Empty string should be filtered out
        let parsed = query
            .project_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(uuid::Uuid::parse_str);
        assert!(parsed.is_none());
    }

    // ================================================================
    // Origin validation tests
    // ================================================================

    /// Build a ServerState for origin validation tests.
    fn make_origin_test_state(
        server_port: u16,
        public_url: Option<&str>,
        auth_config: Option<crate::AuthConfig>,
    ) -> ServerState {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let state = crate::test_helpers::mock_app_state();
        let event_bus = Arc::new(crate::events::HybridEmitter::new(Arc::new(
            crate::events::EventBus::default(),
        )));
        let orchestrator = rt.block_on(async {
            Arc::new(
                crate::orchestrator::Orchestrator::with_event_bus(state, event_bus.clone())
                    .await
                    .unwrap(),
            )
        });
        let watcher = crate::orchestrator::FileWatcher::new(orchestrator.clone());
        ServerState {
            orchestrator,
            watcher: Arc::new(tokio::sync::RwLock::new(watcher)),
            chat_manager: None,
            event_bus,
            nats_emitter: None,
            auth_config,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port,
            public_url: public_url.map(|s| s.to_string()),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        }
    }

    #[test]
    fn test_allowed_origins_without_public_url() {
        let state = make_origin_test_state(6600, None, None);
        let origins = state.allowed_origins();
        assert_eq!(
            origins,
            vec![
                "http://localhost:6600",
                "http://127.0.0.1:6600",
                "tauri://localhost",
                "https://tauri.localhost",
            ]
        );
    }

    #[test]
    fn test_allowed_origins_with_public_url() {
        let state = make_origin_test_state(6600, Some("https://ffs.dev"), None);
        let origins = state.allowed_origins();
        assert_eq!(
            origins,
            vec![
                "http://localhost:6600",
                "http://127.0.0.1:6600",
                "tauri://localhost",
                "https://tauri.localhost",
                "https://ffs.dev"
            ]
        );
    }

    #[test]
    fn test_allowed_origins_with_frontend_url_from_auth_config() {
        // Simulates the real config: auth.frontend_url is set but server.public_url is not
        let auth_config = crate::AuthConfig {
            jwt_secret: "test".to_string(),
            access_token_expiry_secs: 3600,
            refresh_token_expiry_secs: 604800,
            allowed_email_domain: None,
            allowed_emails: None,
            frontend_url: Some("https://ffs.dev".to_string()),
            allow_registration: false,
            root_account: None,
            oidc: None,
            google_client_id: None,
            google_client_secret: None,
            google_redirect_uri: None,
        };
        let state = make_origin_test_state(6600, None, Some(auth_config));
        let origins = state.allowed_origins();
        assert!(
            origins.contains(&"https://ffs.dev".to_string()),
            "frontend_url from auth config should be in allowed origins: {:?}",
            origins
        );
    }

    #[test]
    fn test_allowed_origins_public_url_trailing_slash() {
        let state = make_origin_test_state(6600, Some("https://ffs.dev/"), None);
        let origins = state.allowed_origins();
        assert!(origins.contains(&"https://ffs.dev".to_string()));
    }

    #[test]
    fn test_validate_origin_valid_localhost() {
        let state = make_origin_test_state(6600, Some("https://ffs.dev"), None);
        let result = state.validate_origin(Some("http://localhost:6600"));
        assert_eq!(result.unwrap(), Some("http://localhost:6600".to_string()));
    }

    #[test]
    fn test_validate_origin_valid_public_url() {
        let state = make_origin_test_state(6600, Some("https://ffs.dev"), None);
        let result = state.validate_origin(Some("https://ffs.dev"));
        assert_eq!(result.unwrap(), Some("https://ffs.dev".to_string()));
    }

    #[test]
    fn test_validate_origin_tauri_maps_to_localhost() {
        // tauri://localhost is allowed but gets mapped to http://localhost:{port}
        // because OAuth providers only accept http/https redirect URIs
        let state = make_origin_test_state(6600, None, None);
        let result = state.validate_origin(Some("tauri://localhost"));
        assert_eq!(
            result.unwrap(),
            Some("http://localhost:6600".to_string()),
            "tauri://localhost should be mapped to http://localhost:6600 for OAuth redirect_uri"
        );
    }

    #[test]
    fn test_validate_origin_invalid_returns_error() {
        let state = make_origin_test_state(6600, Some("https://ffs.dev"), None);
        let result = state.validate_origin(Some("https://evil.com"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            AppError::BadRequest(msg) => {
                assert!(msg.contains("Unknown origin"));
                assert!(msg.contains("evil.com"));
            }
            _ => panic!("Expected BadRequest, got {:?}", err),
        }
    }

    #[test]
    fn test_validate_origin_none_returns_none() {
        let state = make_origin_test_state(6600, Some("https://ffs.dev"), None);
        let result = state.validate_origin(None);
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn test_validate_origin_strips_trailing_slash() {
        let state = make_origin_test_state(6600, Some("https://ffs.dev"), None);
        let result = state.validate_origin(Some("https://ffs.dev/"));
        assert_eq!(result.unwrap(), Some("https://ffs.dev".to_string()));
    }
}
