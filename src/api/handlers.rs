//! API request handlers

use crate::api::{
    workspace_handlers::{MilestonePlanSummary, MilestoneStepSummary, MilestoneTaskSummary},
    PaginatedResponse, PaginationParams, PriorityFilter, SearchFilter, StatusFilter, TagsFilter,
};
use crate::chat::ChatManager;
use crate::events::{EventEmitter, HybridEmitter, NatsEmitter};
use crate::graph::algorithms::add_thermal_noise;
use crate::identity::InstanceIdentity;
use crate::neo4j::models::{
    AffectsRelation, CommitNode, ConstraintNode, DecisionNode, DecisionStatus,
    DecisionTimelineEntry, MilestoneNode, MilestoneStatus, PlanNode, PlanStatus, ReleaseNode,
    ReleaseStatus, StepNode, TaskNode, TaskWithPlan,
};
use crate::neo4j::plan::{compute_file_conflicts, WaveComputationResult};
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
    /// Remote skill registry URL (optional — enables cross-instance skill search).
    /// When set, registry search merges local + remote results.
    pub registry_remote_url: Option<String>,
    /// Instance identity for Ed25519 signing (skill packages, P2P messages).
    /// Loaded or generated at startup.
    pub identity: Option<Arc<InstanceIdentity>>,
    /// Pre-built OIDC client — constructed once at server startup.
    /// None when OIDC is not configured (legacy Google-only or no auth).
    pub oidc_client: Option<Arc<crate::auth::oidc::OidcClient>>,
    /// Neural routing — DualTrack router (NN + policy net fallback).
    /// Always present (build full), but only active when config.neural_routing.enabled = true.
    pub neural_router: Arc<tokio::sync::RwLock<neural_routing_runtime::DualTrackRouter>>,
    /// Trajectory collector — wrapped in RwLock for lazy runtime initialization.
    pub trajectory_collector:
        std::sync::RwLock<Option<Arc<neural_routing_runtime::TrajectoryCollector>>>,
    /// Concrete Neo4j trajectory store — needed to create the collector at runtime.
    pub trajectory_store_neo4j: Option<Arc<neural_routing_runtime::Neo4jTrajectoryStore>>,
    /// Trajectory store — Neo4j CRUD + vector search for stored trajectories.
    pub trajectory_store: Option<Arc<dyn neural_routing_runtime::TrajectoryStore>>,
    /// EventReactor counters for the /api/reactor/status endpoint.
    /// Initialized once after reactor is built via `OnceLock`.
    pub reactor_counters: std::sync::OnceLock<Arc<crate::events::ReactorCounters>>,
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
            // additional_origins from auth config (e.g. dev frontend on a different port)
            for origin in &auth_config.additional_origins {
                add(origin);
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
    /// NATS connection status: "connected", "disconnected", or "disabled".
    pub nats: String,
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub services: Option<ServiceHealthStatus>,
}

/// Health check handler — verifies actual connectivity to Neo4j, Meilisearch, and NATS.
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

    // NATS: check connection state if configured, otherwise "disabled"
    let nats_status = match &state.nats_emitter {
        Some(emitter) => {
            let conn_state = emitter.client().connection_state();
            if conn_state == async_nats::connection::State::Connected {
                "connected"
            } else {
                "disconnected"
            }
        }
        None => "disabled",
    };

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
                nats: nats_status.to_string(),
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
    state.event_bus.emit_created(
        crate::events::EntityType::Plan,
        &plan.id.to_string(),
        serde_json::json!({"title": &plan.title, "status": format!("{:?}", plan.status)}),
        plan.project_id.map(|id| id.to_string()),
    );
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

/// Update plan status (kept for backwards compatibility with status-only requests)
#[derive(Deserialize)]
pub struct UpdatePlanStatusRequest {
    pub status: PlanStatus,
}

/// Full plan update request (title, description, priority, and optionally status)
#[derive(Deserialize)]
pub struct UpdatePlanFullRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(default)]
    pub status: Option<PlanStatus>,
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
    Json(req): Json<UpdatePlanFullRequest>,
) -> Result<StatusCode, AppError> {
    // Handle status change if provided
    if let Some(status) = req.status {
        // Get old status before mutation for StatusChanged event
        let old_status = state
            .orchestrator
            .neo4j()
            .get_plan(plan_id)
            .await
            .ok()
            .flatten()
            .map(|p| format!("{:?}", p.status))
            .unwrap_or_default();
        state
            .orchestrator
            .plan_manager()
            .update_plan_status(plan_id, status.clone())
            .await?;
        state.event_bus.emit_status_changed(
            crate::events::EntityType::Plan,
            &plan_id.to_string(),
            &old_status,
            &format!("{:?}", status),
            None,
        );
    }

    // Handle field updates (title, description, priority)
    let plan_update = UpdatePlanRequest {
        title: req.title,
        description: req.description,
        priority: req.priority,
    };
    if plan_update.title.is_some()
        || plan_update.description.is_some()
        || plan_update.priority.is_some()
    {
        state
            .orchestrator
            .plan_manager()
            .update_plan(plan_id, plan_update)
            .await?;
        state.event_bus.emit_updated(
            crate::events::EntityType::Plan,
            &plan_id.to_string(),
            serde_json::json!({}),
            None,
        );
    }

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
    state
        .event_bus
        .emit_deleted(crate::events::EntityType::Plan, &plan_id.to_string(), None);
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
    state.event_bus.emit_created(
        crate::events::EntityType::Task,
        &task.id.to_string(),
        serde_json::json!({"title": &task.title, "plan_id": plan_id}),
        None,
    );
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
    state
        .event_bus
        .emit_deleted(crate::events::EntityType::Task, &task_id.to_string(), None);
    Ok(StatusCode::NO_CONTENT)
}

/// Update task status
pub async fn update_task(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
    Json(req): Json<UpdateTaskRequest>,
) -> Result<StatusCode, AppError> {
    let status_change = if let Some(ref new_status_val) = req.status {
        let old_status = state
            .orchestrator
            .neo4j()
            .get_task(task_id)
            .await
            .ok()
            .flatten()
            .map(|t| format!("{:?}", t.status))
            .unwrap_or_default();
        let new_status = format!("{:?}", new_status_val);
        Some((old_status, new_status))
    } else {
        None
    };
    state
        .orchestrator
        .plan_manager()
        .update_task(task_id, req)
        .await?;
    if let Some((old_status, new_status)) = status_change {
        state.event_bus.emit_status_changed(
            crate::events::EntityType::Task,
            &task_id.to_string(),
            &old_status,
            &new_status,
            None,
        );
    } else {
        state.event_bus.emit_updated(
            crate::events::EntityType::Task,
            &task_id.to_string(),
            serde_json::json!({}),
            None,
        );
    }
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

/// Request body for build_prompt
#[derive(Debug, Deserialize)]
pub struct BuildPromptRequest {
    /// Optional custom sections to append
    #[serde(default)]
    pub custom_sections: Vec<String>,
}

/// Build an enriched prompt for a task via PromptBuilder + EnrichmentPipeline.
///
/// Returns a structured prompt with individual sections (for review) and
/// the fully rendered prompt string.
pub async fn build_task_prompt(
    State(state): State<OrchestratorState>,
    Path((plan_id, task_id)): Path<(Uuid, Uuid)>,
    Json(req): Json<BuildPromptRequest>,
) -> Result<Json<crate::runner::prompt::StructuredPrompt>, AppError> {
    let structured = state
        .orchestrator
        .context_builder()
        .build_enriched_context(
            task_id,
            plan_id,
            None, // no pipeline in HTTP context (stages require DI)
            None, // project_slug
            None, // project_id
            req.custom_sections,
        )
        .await?;
    Ok(Json(structured))
}

// ============================================================================
// Task Delegation — sub-agent orchestration from the conversational agent
// ============================================================================

/// Request body for delegate_task.
///
/// ## Delegation workflow
///
/// The conversational agent orchestrates sub-agents via this endpoint:
///
/// 1. `skill(activate)` — activate relevant skills for the task's domain
/// 2. `note(get_propagated)` — retrieve propagated knowledge notes
/// 3. `code(analyze_impact)` — analyze affected files for context
/// 4. Compose a custom prompt with `custom_sections`
/// 5. `plan(delegate_task)` — spawn a sub-agent with the enriched prompt
///
/// The sub-agent runs asynchronously. Track progress via `chat(get_session_tree)`.
/// Results (tools_used, files_modified, commits) are available on the AgentExecution
/// node once the agent completes.
#[derive(Debug, Deserialize)]
pub struct DelegateTaskRequest {
    /// Working directory for the spawned agent
    pub cwd: String,
    /// Optional project slug (scopes MCP operations)
    #[serde(default)]
    pub project_slug: Option<String>,
    /// Optional custom prompt sections to append
    #[serde(default)]
    pub custom_sections: Vec<String>,
    /// Parent session ID — if set, creates a SPAWNED_BY relation in Neo4j
    /// so the delegation is visible in the session tree.
    #[serde(default)]
    pub parent_session_id: Option<String>,
}

/// Response for a successfully delegated task.
#[derive(Debug, Serialize)]
pub struct DelegateTaskResponse {
    /// Session ID of the spawned sub-agent
    pub session_id: String,
    /// Task ID that was delegated
    pub task_id: Uuid,
    /// Plan ID the task belongs to
    pub plan_id: Uuid,
    /// First 200 characters of the prompt (for preview/debugging)
    pub prompt_preview: String,
}

/// POST /api/plans/:plan_id/tasks/:task_id/delegate — Delegate a task to a sub-agent.
///
/// Builds an enriched prompt via `ContextBuilder.build_enriched_context()` with
/// optional custom sections, spawns a Claude Code agent via `ChatManager`, and
/// creates a SPAWNED_BY relation if `parent_session_id` is provided.
///
/// Returns 202 Accepted with the session_id. The agent executes asynchronously.
/// Track progress via `chat(get_session_tree)` or `chat(get_children)`.
/// Retrieve results from the AgentExecution node after completion.
pub async fn delegate_task(
    State(state): State<OrchestratorState>,
    Path((plan_id, task_id)): Path<(Uuid, Uuid)>,
    Json(req): Json<DelegateTaskRequest>,
) -> Result<(StatusCode, Json<DelegateTaskResponse>), AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    // Validate that the task exists
    let graph = state.orchestrator.neo4j_arc();
    let task_node = graph
        .get_task(task_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Task {} not found", task_id)))?;

    let task_title = task_node
        .title
        .clone()
        .unwrap_or_else(|| "Untitled task".to_string());

    // Step 1: Build enriched prompt via ContextBuilder + EnrichmentPipeline
    let pipeline = chat_manager.enrichment_pipeline.clone();
    let structured = state
        .orchestrator
        .context_builder()
        .build_enriched_context(
            task_id,
            plan_id,
            Some(&pipeline),
            req.project_slug.as_deref(),
            None, // project_id resolved from slug if needed
            req.custom_sections,
        )
        .await
        .map_err(AppError::Internal)?;

    let prompt = structured.rendered.clone();
    let prompt_preview = {
        let end = prompt
            .char_indices()
            .nth(200)
            .map(|(i, _)| i)
            .unwrap_or(prompt.len());
        prompt[..end].to_string()
    };

    // Step 2: Resolve scaffolding level from project for inheritance
    let scaffolding_override = if let Some(ref slug) = req.project_slug {
        match graph.get_project_by_slug(slug).await {
            Ok(Some(project)) => {
                match graph
                    .compute_scaffolding_level(project.id, project.scaffolding_override)
                    .await
                {
                    Ok(level) => Some(level.level),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    } else {
        None
    };

    // Step 3: Spawn sub-agent via ChatManager
    let spawned_by_json = serde_json::json!({
        "type": "delegation",
        "plan_id": plan_id.to_string(),
        "task_id": task_id.to_string(),
        "parent_session_id": req.parent_session_id,
        "scaffolding_level": scaffolding_override,
    });

    let chat_request_cwd = req.cwd.clone();
    let chat_request = crate::chat::types::ChatRequest {
        message: String::new(), // prompt sent via send_message
        session_id: None,
        cwd: req.cwd,
        project_slug: req.project_slug,
        model: None,
        permission_mode: Some("bypassPermissions".to_string()),
        add_dirs: None,
        workspace_slug: None,
        user_claims: Some(crate::auth::jwt::Claims::service_account(
            &format!("delegate-agent:{}", task_id),
        )),
        spawned_by: Some(spawned_by_json.to_string()),
        task_context: Some(task_title.clone()),
        scaffolding_override,
        runner_context: None, // TODO: populate for delegate_task
    };

    let session = chat_manager
        .create_session(&chat_request)
        .await
        .map_err(AppError::Internal)?;
    let session_id = session.session_id.clone();

    // Step 3: Create AgentExecution node for tracking (fire-and-forget)
    let session_uuid = session_id.parse::<Uuid>().ok();
    {
        use crate::neo4j::agent_execution::{AgentExecutionNode, AgentExecutionStatus};
        let ae = AgentExecutionNode {
            id: Uuid::new_v4(),
            run_id: Uuid::nil(), // no PlanRun — this is a direct delegation
            task_id,
            session_id: session_uuid,
            started_at: chrono::Utc::now(),
            completed_at: None,
            cost_usd: 0.0,
            duration_secs: 0.0,
            status: AgentExecutionStatus::Running,
            tools_used: "{}".to_string(),
            files_modified: vec![],
            commits: vec![],
            persona_profile: "delegation".to_string(),
            vector_json: None,
            report_json: None,
            execution_type: Default::default(),
        };
        let graph_clone = graph.clone();
        tokio::spawn(async move {
            if let Err(e) = graph_clone.create_agent_execution(&ae).await {
                tracing::warn!("Failed to create AgentExecution for delegation: {}", e);
            }
        });
    }

    // Step 4: Send the enriched prompt to the agent (async — don't block)
    let cm = chat_manager.clone();
    let sid = session_id.clone();
    let task_title_clone = task_title.clone();

    // Emit TaskStarted event via WebSocket for real-time tracking
    let event_bus = state.event_bus.clone();
    let ev_task_id = task_id;
    let ev_plan_id = plan_id;
    let ev_session_id = session_id.clone();

    tokio::spawn(async move {
        // Send the prompt
        if let Err(e) = cm.send_message(&sid, &prompt).await {
            tracing::error!(
                session_id = %sid,
                task_id = %ev_task_id,
                "Failed to send delegation prompt: {}", e
            );
            return;
        }

        tracing::info!(
            session_id = %ev_session_id,
            task_id = %ev_task_id,
            plan_id = %ev_plan_id,
            task_title = %task_title_clone,
            "Delegation started: sub-agent spawned with enriched prompt"
        );

        // Listen for completion and emit TaskCompleted event
        let rx = match cm.subscribe(&sid).await {
            Ok(rx) => rx,
            Err(e) => {
                tracing::warn!("Failed to subscribe to delegated session {}: {}", sid, e);
                return;
            }
        };

        let delegation_cwd = Some(chat_request_cwd.clone());
        listen_delegation_result(rx, ev_task_id, &task_title_clone, event_bus, delegation_cwd)
            .await;
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(DelegateTaskResponse {
            session_id,
            task_id,
            plan_id,
            prompt_preview,
        }),
    ))
}

/// Listen for a delegated sub-agent's Result event and emit a RunnerEvent-compatible
/// CrudEvent so the frontend is notified via WebSocket.
/// Also collects commits from agent worktrees after completion (prevents ghost completions).
async fn listen_delegation_result(
    mut rx: tokio::sync::broadcast::Receiver<crate::chat::types::ChatEvent>,
    task_id: Uuid,
    task_title: &str,
    event_bus: std::sync::Arc<crate::events::HybridEmitter>,
    cwd: Option<String>,
) {
    use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};

    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(3600); // 1h safety net

    loop {
        let remaining = timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            tracing::warn!(task_id = %task_id, "Delegation listener timed out after 1h");
            break;
        }

        match tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv()).await {
            Ok(Ok(crate::chat::types::ChatEvent::Result {
                cost_usd, is_error, ..
            })) => {
                let duration_secs = start.elapsed().as_secs_f64();
                let cost = cost_usd.unwrap_or(0.0);

                // Emit a CrudEvent so WebSocket clients see the completion
                let payload = serde_json::json!({
                    "event": if is_error { "delegation_failed" } else { "delegation_completed" },
                    "task_id": task_id,
                    "task_title": task_title,
                    "cost_usd": cost,
                    "duration_secs": duration_secs,
                    "is_error": is_error,
                });

                event_bus.emit(CrudEvent {
                    entity_type: EntityType::Task,
                    action: CrudAction::Updated,
                    entity_id: task_id.to_string(),
                    related: None,
                    payload,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    project_id: None,
                });

                tracing::info!(
                    task_id = %task_id,
                    cost_usd = cost,
                    duration_secs = duration_secs,
                    is_error = is_error,
                    "Delegation completed"
                );

                // Collect commits from agent worktrees (prevents ghost completions)
                if let Some(ref cwd) = cwd {
                    if let Ok(run_branch) = crate::runner::git::current_branch(cwd).await {
                        if !run_branch.is_empty() {
                            match crate::runner::git::collect_worktree_commits(cwd, &run_branch)
                                .await
                            {
                                Ok(wt_result) => {
                                    if !wt_result.recovered_commits.is_empty() {
                                        tracing::info!(
                                            task_id = %task_id,
                                            commits = wt_result.recovered_commits.len(),
                                            "Delegation: recovered commits from agent worktrees"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        task_id = %task_id,
                                        "Delegation worktree collection failed (non-fatal): {}", e
                                    );
                                }
                            }
                        }
                    }
                }

                break;
            }
            Ok(Ok(_)) => {
                // Other event types — keep listening
            }
            Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(n))) => {
                tracing::warn!("Delegation listener lagged by {} events", n);
            }
            Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => {
                tracing::warn!(task_id = %task_id, "Delegation channel closed");
                break;
            }
            Err(_) => {
                // 10s poll timeout — loop again
            }
        }
    }
}

/// Enrich all tasks in a plan (pre-build context, profile persona, cache prompts).
pub async fn enrich_plan(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<crate::orchestrator::context::PlanEnrichmentResult>, AppError> {
    let result = state
        .orchestrator
        .context_builder()
        .enrich_plan(plan_id)
        .await?;
    Ok(Json(result))
}

/// Enrich a single task (pre-build context, profile persona, cache prompt).
pub async fn enrich_task(
    State(state): State<OrchestratorState>,
    Path((plan_id, task_id)): Path<(Uuid, Uuid)>,
) -> Result<Json<crate::orchestrator::context::EnrichmentResult>, AppError> {
    let result = state
        .orchestrator
        .context_builder()
        .enrich_task(task_id, plan_id)
        .await?;
    Ok(Json(result))
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
    let explicit_run_id = req.run_id;
    let decision = state
        .orchestrator
        .plan_manager()
        .add_decision(task_id, req, "agent")
        .await?;

    // Resolve run_id: explicit > auto-detect from active run
    let run_id = match explicit_run_id {
        Some(rid) => Some(rid),
        None => {
            // Auto-detect: resolve project_id from task, then find active run
            match state
                .orchestrator
                .neo4j()
                .get_project_for_task(task_id)
                .await
            {
                Ok(Some(project)) => state
                    .orchestrator
                    .neo4j()
                    .find_active_run_for_project(project.id)
                    .await
                    .unwrap_or(None),
                _ => None,
            }
        }
    };

    // Create PRODUCED_DURING relation if we have a run_id
    if let Some(rid) = run_id {
        if let Err(e) = state
            .orchestrator
            .neo4j()
            .create_produced_during("Decision", decision.id, rid)
            .await
        {
            tracing::warn!(
                decision_id = %decision.id,
                run_id = %rid,
                error = %e,
                "Failed to create PRODUCED_DURING relation for decision"
            );
        }
    }

    // Growth hook: auto-link decision to RELEVANT personas only (not all active ones).
    // Relevance = persona KNOWS files that overlap with decision's AFFECTS entities.
    {
        let neo4j = state.orchestrator.neo4j_arc();
        let decision_id = decision.id;
        tokio::spawn(async move {
            // Resolve project from task
            if let Ok(Some(project)) = neo4j.get_project_for_task(task_id).await {
                match neo4j
                    .find_relevant_personas_for_decision(decision_id, project.id)
                    .await
                {
                    Ok(relevant) if !relevant.is_empty() => {
                        for (persona_id, avg_weight) in &relevant {
                            let link_weight = (*avg_weight * 0.8).max(0.3);
                            if let Err(e) = neo4j
                                .auto_link_decision_to_persona(
                                    *persona_id,
                                    decision_id,
                                    link_weight,
                                )
                                .await
                            {
                                tracing::debug!(
                                    persona_id = %persona_id,
                                    decision_id = %decision_id,
                                    error = %e,
                                    "Growth hook: auto_link_decision failed (non-fatal)"
                                );
                            }
                        }
                        tracing::debug!(
                            decision_id = %decision_id,
                            count = relevant.len(),
                            "Growth hook: linked decision to relevant personas"
                        );
                    }
                    Ok(_) => {
                        tracing::debug!(
                            decision_id = %decision_id,
                            "Growth hook: no relevant personas found for decision (skipped)"
                        );
                    }
                    Err(e) => {
                        tracing::debug!(
                            project_id = %project.id,
                            error = %e,
                            "Growth hook: find_relevant_personas_for_decision failed (non-fatal)"
                        );
                    }
                }
            }
        });
    }

    state.event_bus.emit_created(
        crate::events::EntityType::Decision,
        &decision.id.to_string(),
        serde_json::json!({"task_id": task_id, "description": &decision.description}),
        None,
    );

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
    pub status: Option<DecisionStatus>,
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
            req.status,
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
    pub project_slug: Option<String>,
    /// Filter by workspace slug (searches all projects in the workspace)
    pub workspace_slug: Option<String>,
}

pub async fn search_decisions(
    State(state): State<OrchestratorState>,
    axum::extract::Query(query): axum::extract::Query<SearchQuery>,
) -> Result<Json<Vec<DecisionNode>>, AppError> {
    let limit = query.limit.unwrap_or(10);

    // If project_slug is given, use it directly (takes precedence)
    if query.project_slug.is_some() {
        let decisions = state
            .orchestrator
            .plan_manager()
            .search_decisions(&query.q, limit, query.project_slug.as_deref())
            .await?;
        return Ok(Json(decisions));
    }

    // If workspace_slug is given, resolve to project slugs and filter
    if let Some(ref ws_slug) = query.workspace_slug {
        let workspace = state
            .orchestrator
            .neo4j()
            .get_workspace_by_slug(ws_slug)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("Workspace not found: {}", ws_slug)))?;

        let projects = state
            .orchestrator
            .neo4j()
            .list_workspace_projects(workspace.id)
            .await?;

        let project_slugs: Vec<String> = projects.iter().map(|p| p.slug.clone()).collect();

        let decisions = state
            .orchestrator
            .plan_manager()
            .search_decisions_in_workspace(&query.q, limit, &project_slugs)
            .await?;
        return Ok(Json(decisions));
    }

    // No filter — global search
    let decisions = state
        .orchestrator
        .plan_manager()
        .search_decisions(&query.q, limit, None)
        .await?;
    Ok(Json(decisions))
}

/// Semantic search for decisions using vector embeddings
#[derive(Deserialize)]
pub struct SearchDecisionsSemanticQuery {
    pub query: String,
    pub limit: Option<usize>,
    pub project_id: Option<String>,
    /// Thermal noise temperature (0.0 - 1.0) for stochastic exploration.
    /// Inspired by Langevin dynamics: adds T × N(0, σ) Gaussian noise to scores.
    pub temperature: Option<f64>,
    /// Intent-adaptive analysis profile name (e.g., "debug", "explore", "impact", "plan").
    /// When provided, weights the search results according to the profile's configuration.
    pub profile: Option<String>,
}

pub async fn search_decisions_semantic(
    State(state): State<OrchestratorState>,
    axum::extract::Query(params): axum::extract::Query<SearchDecisionsSemanticQuery>,
) -> Result<Json<Vec<DecisionSearchHit>>, AppError> {
    let mut results = state
        .orchestrator
        .plan_manager()
        .search_decisions_semantic(
            &params.query,
            params.limit.unwrap_or(10),
            params.project_id.as_deref(),
        )
        .await?;

    // Apply Langevin thermal noise for stochastic exploration
    if let Some(temperature) = params.temperature {
        if temperature > 0.0 {
            let mut scored: Vec<(DecisionSearchHit, f64)> = results
                .into_iter()
                .map(|h| {
                    let s = h.score;
                    (h, s)
                })
                .collect();
            add_thermal_noise(&mut scored, temperature);
            results = scored
                .into_iter()
                .map(|(mut h, s)| {
                    h.score = s;
                    h
                })
                .collect();
        }
    }

    Ok(Json(results))
}

// ============================================================================
// Decision Affects
// ============================================================================

/// Query params for getting decisions that affect an entity
#[derive(Deserialize)]
pub struct DecisionsAffectingQuery {
    pub entity_type: String,
    pub entity_id: String,
    pub status: Option<String>,
}

/// Get decisions that affect a given entity (reverse AFFECTS lookup)
pub async fn get_decisions_affecting(
    State(state): State<OrchestratorState>,
    Query(params): Query<DecisionsAffectingQuery>,
) -> Result<Json<Vec<DecisionNode>>, AppError> {
    let results = state
        .orchestrator
        .neo4j()
        .get_decisions_affecting(
            &params.entity_type,
            &params.entity_id,
            params.status.as_deref(),
        )
        .await?;
    Ok(Json(results))
}

/// Request to add an AFFECTS relation from a decision to an entity
#[derive(Deserialize)]
pub struct AddAffectsRequest {
    pub entity_type: String,
    pub entity_id: String,
    pub impact_description: Option<String>,
}

/// Add an AFFECTS relation from a decision to an entity
pub async fn add_decision_affects(
    State(state): State<OrchestratorState>,
    Path(decision_id): Path<Uuid>,
    Json(req): Json<AddAffectsRequest>,
) -> Result<StatusCode, AppError> {
    let neo4j = state.orchestrator.neo4j();
    neo4j
        .add_decision_affects(
            decision_id,
            &req.entity_type,
            &req.entity_id,
            req.impact_description.as_deref(),
        )
        .await?;

    // Emit GraphEvent for the new AFFECTS edge (best-effort, don't fail the request)
    if let Ok(Some(pid)) = neo4j.get_decision_project_id(decision_id).await {
        state.event_bus.emit_graph(crate::events::GraphEvent::edge(
            crate::events::graph::GraphEventType::EdgeCreated,
            crate::events::graph::GraphLayer::Knowledge,
            decision_id.to_string(),
            &req.entity_id,
            "AFFECTS",
            pid,
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Remove an AFFECTS relation from a decision to an entity.
///
/// Supports two URL formats:
/// - `DELETE /api/decisions/{id}/affects/{entity_type}/{entity_id}` (simple entity_ids)
/// - `DELETE /api/decisions/{id}/affects?entity_type=File&entity_id=/path/to/file.rs` (paths with slashes)
pub async fn remove_decision_affects(
    State(state): State<OrchestratorState>,
    Path((decision_id, entity_type, entity_id)): Path<(Uuid, String, String)>,
) -> Result<StatusCode, AppError> {
    let neo4j = state.orchestrator.neo4j();
    neo4j
        .remove_decision_affects(decision_id, &entity_type, &entity_id)
        .await?;

    // Emit GraphEvent for the removed AFFECTS edge (best-effort)
    if let Ok(Some(pid)) = neo4j.get_decision_project_id(decision_id).await {
        state.event_bus.emit_graph(crate::events::GraphEvent::edge(
            crate::events::graph::GraphEventType::EdgeRemoved,
            crate::events::graph::GraphLayer::Knowledge,
            decision_id.to_string(),
            &entity_id,
            "AFFECTS",
            pid,
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Query params for the query-param variant of remove_decision_affects.
#[derive(Deserialize)]
pub struct RemoveAffectsQuery {
    pub entity_type: String,
    pub entity_id: String,
}

/// DELETE /api/decisions/{id}/affects?entity_type=...&entity_id=...
///
/// Alternative to the path-based route, for entity_ids that contain slashes (e.g. file paths).
pub async fn remove_decision_affects_query(
    State(state): State<OrchestratorState>,
    Path(decision_id): Path<Uuid>,
    Query(query): Query<RemoveAffectsQuery>,
) -> Result<StatusCode, AppError> {
    let neo4j = state.orchestrator.neo4j();
    neo4j
        .remove_decision_affects(decision_id, &query.entity_type, &query.entity_id)
        .await?;

    // Emit GraphEvent for the removed AFFECTS edge (best-effort)
    if let Ok(Some(pid)) = neo4j.get_decision_project_id(decision_id).await {
        state.event_bus.emit_graph(crate::events::GraphEvent::edge(
            crate::events::graph::GraphEventType::EdgeRemoved,
            crate::events::graph::GraphLayer::Knowledge,
            decision_id.to_string(),
            &query.entity_id,
            "AFFECTS",
            pid,
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

/// List all entities affected by a decision
pub async fn list_decision_affects(
    State(state): State<OrchestratorState>,
    Path(decision_id): Path<Uuid>,
) -> Result<Json<Vec<AffectsRelation>>, AppError> {
    let affects = state
        .orchestrator
        .neo4j()
        .list_decision_affects(decision_id)
        .await?;
    Ok(Json(affects))
}

/// Mark a decision as superseded by a newer decision
pub async fn supersede_decision(
    State(state): State<OrchestratorState>,
    Path((new_id, old_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .supersede_decision(new_id, old_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Decision Timeline
// ============================================================================

/// Query parameters for the decision timeline endpoint
#[derive(Deserialize)]
pub struct DecisionTimelineQuery {
    pub task_id: Option<Uuid>,
    pub from: Option<String>,
    pub to: Option<String>,
}

/// Get a timeline of decisions with supersession chains
pub async fn get_decision_timeline(
    State(state): State<OrchestratorState>,
    Query(params): Query<DecisionTimelineQuery>,
) -> Result<Json<Vec<DecisionTimelineEntry>>, AppError> {
    let entries = state
        .orchestrator
        .neo4j()
        .get_decision_timeline(params.task_id, params.from.as_deref(), params.to.as_deref())
        .await?;
    Ok(Json(entries))
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

        // Spawn knowledge link reconstruction in background: link notes + decisions
        // to newly synced files (cross-project notes + decision AFFECTS)
        let neo4j = state.orchestrator.neo4j_arc();
        let neo4j_topo = neo4j.clone();
        tokio::spawn(async move {
            match crate::skills::activation::reconstruct_knowledge_links(neo4j.as_ref(), pid).await
            {
                Ok(r) if r.notes_linked > 0 || r.affects_created > 0 => {
                    tracing::info!(
                        %pid,
                        notes_linked = r.notes_linked,
                        affects_created = r.affects_created,
                        elapsed_ms = r.elapsed_ms,
                        "Post-sync knowledge reconstruction: {} note links, {} decision affects",
                        r.notes_linked,
                        r.affects_created
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(%pid, "Post-sync knowledge reconstruction failed: {}", e);
                }
            }
        });

        // Spawn topology firewall check in background: detect violations and create gotcha notes
        tokio::spawn(async move {
            match crate::orchestrator::topology_hook::check_topology_post_sync(neo4j_topo, pid)
                .await
            {
                Ok(r) if r.notes_created > 0 => {
                    tracing::info!(
                        %pid,
                        violations = r.violations_found,
                        new_notes = r.notes_created,
                        skipped = r.already_captured,
                        "Post-sync topology check: {} violations, {} new gotcha notes",
                        r.violations_found,
                        r.notes_created,
                    );
                }
                Ok(r) if r.violations_found > 0 => {
                    tracing::debug!(
                        %pid,
                        violations = r.violations_found,
                        "Post-sync topology check: {} violations (all already captured)",
                        r.violations_found,
                    );
                }
                Ok(_) => {} // no violations
                Err(e) => {
                    tracing::warn!(%pid, "Post-sync topology check failed: {}", e);
                }
            }
        });

        // Spawn event-triggered protocol runs (post_sync)
        crate::protocol::hooks::spawn_event_triggered_protocols(
            state.orchestrator.neo4j_arc(),
            pid,
            "post_sync",
            Some(state.event_bus.clone() as std::sync::Arc<dyn crate::events::EventEmitter>),
        );
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

/// GET /api/reactor/status — EventReactor health and statistics.
///
/// Returns the reactor's running state and event processing counters.
pub async fn reactor_status(State(state): State<OrchestratorState>) -> Json<serde_json::Value> {
    match state.reactor_counters.get() {
        Some(counters) => {
            let stats = counters.snapshot(0, 0);
            Json(serde_json::json!({
                "running": stats.running,
                "events_received": stats.events_received,
                "events_matched": stats.events_matched,
                "handlers_invoked": stats.handlers_invoked,
                "handler_errors": stats.handler_errors,
            }))
        }
        None => Json(serde_json::json!({
            "running": false,
            "error": "reactor not initialized",
        })),
    }
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
    state
        .event_bus
        .emit_deleted(crate::events::EntityType::Step, &step_id.to_string(), None);
    Ok(StatusCode::NO_CONTENT)
}

/// Update a step
pub async fn update_step(
    State(state): State<OrchestratorState>,
    Path(step_id): Path<Uuid>,
    Json(req): Json<UpdateStepRequest>,
) -> Result<StatusCode, AppError> {
    if let Some(status) = req.status.clone() {
        state
            .orchestrator
            .plan_manager()
            .update_step_status(step_id, status.clone())
            .await?;

        // Biomimicry: Frustration decay — completing a step reduces frustration on parent task
        if status == crate::neo4j::models::StepStatus::Completed {
            // Find parent task and decrement frustration by 0.1
            if let Ok(Some(task_id)) = state
                .orchestrator
                .neo4j()
                .get_step_parent_task_id(step_id)
                .await
            {
                let _ = state
                    .orchestrator
                    .neo4j()
                    .decrement_frustration(task_id, 0.1)
                    .await;
            }
        }
    }
    if req.description.is_some() || req.verification.is_some() {
        state
            .orchestrator
            .plan_manager()
            .update_step(step_id, &req)
            .await?;
    }
    state.event_bus.emit_updated(
        crate::events::EntityType::Step,
        &step_id.to_string(),
        serde_json::json!({}),
        None,
    );
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

    state.event_bus.emit_created(
        crate::events::EntityType::Constraint,
        &constraint.id.to_string(),
        serde_json::json!({"plan_id": plan_id, "constraint_type": format!("{:?}", constraint.constraint_type)}),
        None,
    );

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

    state.event_bus.emit_updated(
        crate::events::EntityType::Constraint,
        &constraint_id.to_string(),
        serde_json::json!({"constraint_id": constraint_id}),
        None,
    );

    Ok(StatusCode::NO_CONTENT)
}

/// Delete a constraint
pub async fn delete_constraint(
    State(state): State<OrchestratorState>,
    Path(constraint_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_constraint(constraint_id).await?;

    state.event_bus.emit_deleted(
        crate::events::EntityType::Constraint,
        &constraint_id.to_string(),
        None,
    );

    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Lifecycle Hooks
// ============================================================================

/// GET /api/lifecycle-hooks — List lifecycle hooks, optionally filtered by project_id
pub async fn list_lifecycle_hooks(
    State(state): State<OrchestratorState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<crate::lifecycle::LifecycleHook>>, AppError> {
    let project_id = params
        .get("project_id")
        .and_then(|s| Uuid::parse_str(s).ok());
    let hooks = state
        .orchestrator
        .neo4j()
        .list_lifecycle_hooks(project_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(hooks))
}

/// POST /api/lifecycle-hooks — Create a lifecycle hook
pub async fn create_lifecycle_hook(
    State(state): State<OrchestratorState>,
    Json(req): Json<crate::lifecycle::CreateLifecycleHookRequest>,
) -> Result<Json<crate::lifecycle::LifecycleHook>, AppError> {
    let mut hook = crate::lifecycle::LifecycleHook::new(
        req.name,
        req.scope,
        req.on_status,
        req.action_type,
        req.action_config.unwrap_or(serde_json::json!({})),
    );
    hook.description = req.description;
    if let Some(p) = req.priority {
        hook.priority = p;
    }
    if let Some(pid_str) = &req.project_id {
        hook.project_id = Uuid::parse_str(pid_str).ok();
    }

    state
        .orchestrator
        .neo4j()
        .create_lifecycle_hook(&hook)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_created(
        crate::events::EntityType::LifecycleHook,
        &hook.id.to_string(),
        serde_json::json!({"name": hook.name, "scope": format!("{:?}", hook.scope)}),
        hook.project_id.map(|id| id.to_string()),
    );

    Ok(Json(hook))
}

/// GET /api/lifecycle-hooks/:id — Get a lifecycle hook by ID
pub async fn get_lifecycle_hook(
    State(state): State<OrchestratorState>,
    Path(hook_id): Path<Uuid>,
) -> Result<Json<crate::lifecycle::LifecycleHook>, AppError> {
    let hook = state
        .orchestrator
        .neo4j()
        .get_lifecycle_hook(hook_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound("Lifecycle hook not found".into()))?;
    Ok(Json(hook))
}

/// PATCH /api/lifecycle-hooks/:id — Update a lifecycle hook
pub async fn update_lifecycle_hook(
    State(state): State<OrchestratorState>,
    Path(hook_id): Path<Uuid>,
    Json(req): Json<crate::lifecycle::UpdateLifecycleHookRequest>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .update_lifecycle_hook(hook_id, &req)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_updated(
        crate::events::EntityType::LifecycleHook,
        &hook_id.to_string(),
        serde_json::json!({"hook_id": hook_id}),
        None,
    );

    Ok(StatusCode::OK)
}

/// DELETE /api/lifecycle-hooks/:id — Delete a lifecycle hook (builtin hooks cannot be deleted)
pub async fn delete_lifecycle_hook(
    State(state): State<OrchestratorState>,
    Path(hook_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .delete_lifecycle_hook(hook_id)
        .await
        .map_err(|e| {
            if e.to_string().contains("builtin") {
                AppError::Conflict(e.to_string())
            } else {
                AppError::Internal(e)
            }
        })?;

    state.event_bus.emit_deleted(
        crate::events::EntityType::LifecycleHook,
        &hook_id.to_string(),
        None,
    );

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

/// POST /api/admin/cleanup-cross-project-calls — Remove spurious CALLS relationships between different projects
pub async fn cleanup_cross_project_calls(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .cleanup_cross_project_calls()
        .await?;
    Ok(Json(serde_json::json!({ "deleted_count": deleted })))
}

/// POST /api/admin/cleanup-builtin-calls — Delete CALLS relationships targeting built-in functions
pub async fn cleanup_builtin_calls(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = state.orchestrator.neo4j().cleanup_builtin_calls().await?;
    Ok(Json(serde_json::json!({ "deleted_count": deleted })))
}

/// POST /api/admin/migrate-calls-confidence — Add confidence/reason to existing CALLS relationships
pub async fn migrate_calls_confidence(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let updated = state
        .orchestrator
        .neo4j()
        .migrate_calls_confidence()
        .await?;
    Ok(Json(serde_json::json!({ "updated_count": updated })))
}

/// POST /api/admin/cleanup-sync-data — Delete all sync data (File/Function/Struct nodes) and Meilisearch code index
pub async fn cleanup_sync_data(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = state.orchestrator.neo4j().cleanup_sync_data().await?;
    // Also clean ALL Meilisearch code documents to avoid stale/duplicate entries
    if let Err(e) = state.orchestrator.meili().delete_all_code().await {
        tracing::warn!("Failed to clean Meilisearch code index: {}", e);
    }
    Ok(Json(serde_json::json!({
        "deleted_count": deleted,
        "message": "Sync data and Meilisearch code index cleaned. Run sync_project to rebuild."
    })))
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
    /// Files changed in this commit (enables incremental sync + TOUCHES relations).
    /// Accepts either simple strings `["a.rs"]` or objects `[{"path": "a.rs", "additions": 10}]`.
    #[serde(
        default,
        deserialize_with = "crate::neo4j::models::deserialize_files_changed"
    )]
    pub files_changed: Option<Vec<crate::neo4j::models::FileChangedInfo>>,
    /// Project UUID (enables incremental sync + analytics)
    pub project_id: Option<String>,
}

/// Response for create_commit
#[derive(Serialize)]
pub struct CreateCommitResponse {
    #[serde(flatten)]
    pub commit: CommitNode,
    pub sync_triggered: bool,
}

/// Create a new commit
///
/// Side-effects (when `files_changed` + `project_id` are provided):
/// 1. Incremental sync of changed files (background)
/// 2. Analytics debounce trigger (background)
/// 3. Hebbian energy boost on notes linked to committed files (background)
pub async fn create_commit(
    State(state): State<OrchestratorState>,
    Json(req): Json<CreateCommitRequest>,
) -> Result<Json<CreateCommitResponse>, AppError> {
    let commit = CommitNode {
        hash: req.hash,
        message: req.message,
        author: req.author,
        timestamp: req.timestamp.unwrap_or_else(chrono::Utc::now),
    };

    state.orchestrator.create_commit(&commit).await?;

    let files_changed = req.files_changed.unwrap_or_default();
    let project_id = req
        .project_id
        .as_deref()
        .and_then(|s| uuid::Uuid::parse_str(s).ok());

    let sync_triggered = !files_changed.is_empty() && project_id.is_some();

    // Resolve project to get root_path (for path resolution) and slug (for MeiliSearch).
    // File nodes in Neo4j use absolute paths, so TOUCHES relations require absolute paths.
    let project_info = if let Some(pid) = project_id {
        state
            .orchestrator
            .neo4j()
            .get_project(pid)
            .await
            .ok()
            .flatten()
    } else {
        None
    };
    let project_root = project_info
        .as_ref()
        .map(|p| std::path::PathBuf::from(crate::expand_tilde(&p.root_path)));
    let project_slug = project_info.as_ref().map(|p| p.slug.clone());

    // Resolve relative paths to absolute using project root_path
    let files_changed: Vec<crate::neo4j::models::FileChangedInfo> = files_changed
        .into_iter()
        .map(|mut f| {
            if let Some(ref root) = project_root {
                let p = std::path::Path::new(&f.path);
                if p.is_relative() {
                    f.path = root.join(p).to_string_lossy().to_string();
                }
            }
            f
        })
        .collect();

    // Side-effect: Create TOUCHES relations (Commit→File) — synchronous for consistency
    if !files_changed.is_empty() {
        let commit_hash = commit.hash.clone();
        if let Err(e) = state
            .orchestrator
            .neo4j()
            .create_commit_touches(&commit_hash, &files_changed)
            .await
        {
            tracing::warn!(
                commit = %commit_hash, error = %e,
                "Failed to create TOUCHES relations"
            );
        }
    }

    // Side-effect: Freshness ping — fire-and-forget update of freshness_pinged_at
    // on all notes LINKED_TO touched files (regardless of project_id).
    if !files_changed.is_empty() {
        let paths_for_freshness: Vec<String> =
            files_changed.iter().map(|f| f.path.clone()).collect();
        let neo4j_freshness = state.orchestrator.neo4j_arc();
        tokio::spawn(async move {
            match neo4j_freshness
                .ping_freshness_for_files(&paths_for_freshness)
                .await
            {
                Ok(n) if n > 0 => {
                    tracing::debug!(pinged = n, "Freshness ping: updated notes");
                }
                Err(e) => {
                    tracing::warn!("Freshness ping failed: {}", e);
                }
                _ => {}
            }
        });
    }

    if sync_triggered {
        let pid = project_id.unwrap();
        let orchestrator = state.orchestrator.clone();
        // Extract file paths for sync and boost operations
        let file_paths: Vec<String> = files_changed.iter().map(|f| f.path.clone()).collect();
        let paths_for_boost = file_paths.clone();

        // Side-effect 0: Invalidate context cards for changed files + 1-hop neighbors
        let paths_for_invalidate = file_paths.clone();
        let orch_invalidate = orchestrator.clone();
        let pid_str = pid.to_string();
        tokio::spawn(async move {
            if let Err(e) = orch_invalidate
                .neo4j()
                .invalidate_context_cards(&paths_for_invalidate, &pid_str)
                .await
            {
                tracing::warn!(
                    project_id = %pid,
                    "Failed to invalidate context cards: {}", e
                );
            }
        });

        // Side-effect 1 & 2: Incremental sync + analytics debounce
        let orch2 = orchestrator.clone();
        tokio::spawn(async move {
            for file_path in &file_paths {
                let path = std::path::Path::new(file_path.as_str());
                if path.exists() {
                    if let Err(e) = orch2
                        .sync_file_for_project(path, Some(pid), project_slug.as_deref())
                        .await
                    {
                        tracing::warn!("Incremental sync failed for {}: {}", file_path, e);
                    }
                }
            }
            if let Err(e) = orch2.neo4j().update_project_synced(pid).await {
                tracing::warn!("Failed to update last_synced: {}", e);
            }
            orch2.analytics_debouncer().trigger(pid);
            orch2.co_change_debouncer().trigger(pid);

            // Reconstruct knowledge links for newly synced files
            match crate::skills::activation::reconstruct_knowledge_links(orch2.neo4j(), pid).await {
                Ok(r) if r.notes_linked > 0 || r.affects_created > 0 => {
                    tracing::info!(
                        %pid,
                        notes_linked = r.notes_linked,
                        affects_created = r.affects_created,
                        "Post-commit knowledge reconstruction: {} note links, {} decision affects",
                        r.notes_linked,
                        r.affects_created
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(%pid, "Post-commit knowledge reconstruction failed: {}", e);
                }
            }
        });

        // Side-effect 3: Debounced Hebbian energy boost + synapse reinforcement.
        // Instead of inline processing, delegate to the NeuralReinforcementDebouncer
        // which batches file paths across rapid-fire commits (e.g., during git
        // checkout/rebase) and performs a single pass after a 5s quiet period.
        if orchestrator.auto_reinforcement_config().enabled && !paths_for_boost.is_empty() {
            orchestrator.neural_reinforcement_debouncer().trigger(
                crate::graph::ReinforcementPayload {
                    project_id: pid,
                    file_paths: paths_for_boost,
                },
            );
        }

        // Side-effect 4: Event-triggered protocol runs (post_sync)
        crate::protocol::hooks::spawn_event_triggered_protocols(
            orchestrator.neo4j_arc(),
            pid,
            "post_sync",
            None,
        );
    }

    state.event_bus.emit_created(
        crate::events::EntityType::Commit,
        &commit.hash,
        serde_json::json!({"message": &commit.message, "author": &commit.author}),
        project_id.map(|p| p.to_string()),
    );

    Ok(Json(CreateCommitResponse {
        commit,
        sync_triggered,
    }))
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
    state.event_bus.emit_linked(
        crate::events::EntityType::Commit,
        &req.commit_hash,
        crate::events::EntityType::Task,
        &task_id.to_string(),
        None,
    );
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
    state.event_bus.emit_linked(
        crate::events::EntityType::Commit,
        &req.commit_hash,
        crate::events::EntityType::Plan,
        &plan_id.to_string(),
        None,
    );
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
// TOUCHES — Commit ↔ File queries
// ============================================================================

/// Get files touched by a commit (via TOUCHES relations)
pub async fn get_commit_files(
    State(state): State<OrchestratorState>,
    Path(commit_sha): Path<String>,
) -> Result<Json<Vec<crate::neo4j::models::CommitFileInfo>>, AppError> {
    let files = state
        .orchestrator
        .neo4j()
        .get_commit_files(&commit_sha)
        .await?;
    Ok(Json(files))
}

/// Query parameters for file history
#[derive(Deserialize)]
pub struct FileHistoryQuery {
    pub path: String,
    pub limit: Option<i64>,
}

/// Get commit history for a file (via TOUCHES relations)
pub async fn get_file_history(
    State(state): State<OrchestratorState>,
    Query(query): Query<FileHistoryQuery>,
) -> Result<Json<Vec<crate::neo4j::models::FileHistoryEntry>>, AppError> {
    let history = state
        .orchestrator
        .neo4j()
        .get_file_history(&query.path, query.limit)
        .await?;
    Ok(Json(history))
}

/// Query parameters for co-change graph
#[derive(Deserialize)]
pub struct CoChangeGraphQuery {
    pub min_count: Option<i64>,
    pub limit: Option<i64>,
}

/// Get the co-change graph for a project
pub async fn get_co_change_graph(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
    Query(query): Query<CoChangeGraphQuery>,
) -> Result<Json<Vec<crate::neo4j::models::CoChangePair>>, AppError> {
    let pairs = state
        .orchestrator
        .neo4j()
        .get_co_change_graph(
            project_id,
            query.min_count.unwrap_or(3),
            query.limit.unwrap_or(100),
        )
        .await?;
    Ok(Json(pairs))
}

/// Query parameters for file co-changers
#[derive(Deserialize)]
pub struct FileCoChangersQuery {
    pub path: String,
    pub min_count: Option<i64>,
    pub limit: Option<i64>,
}

/// Get files that co-change with a given file
pub async fn get_file_co_changers(
    State(state): State<OrchestratorState>,
    Query(query): Query<FileCoChangersQuery>,
) -> Result<Json<Vec<crate::neo4j::models::CoChanger>>, AppError> {
    let changers = state
        .orchestrator
        .neo4j()
        .get_file_co_changers(
            &query.path,
            query.min_count.unwrap_or(3),
            query.limit.unwrap_or(50),
        )
        .await?;
    Ok(Json(changers))
}

/// Backfill TOUCHES relations from git history for a project
pub async fn backfill_commit_touches(
    State(state): State<OrchestratorState>,
    Path(project_slug): Path<String>,
) -> Result<Json<crate::orchestrator::BackfillResult>, AppError> {
    // Resolve project
    let project = state
        .orchestrator
        .neo4j()
        .get_project_by_slug(&project_slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_slug)))?;

    let root_path = std::path::PathBuf::from(crate::expand_tilde(&project.root_path));
    let result = state
        .orchestrator
        .backfill_commit_touches(project.id, &root_path)
        .await?;

    Ok(Json(result))
}

/// POST /api/admin/reindex-decisions — Reindex all decisions from Neo4j into MeiliSearch.
///
/// Reads all Decision nodes from Neo4j and upserts them into MeiliSearch.
/// Useful after MeiliSearch data loss or rebuild.
pub async fn reindex_decisions(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let (total, indexed) = state
        .orchestrator
        .plan_manager()
        .reindex_decisions()
        .await?;

    Ok(Json(serde_json::json!({
        "decisions_processed": total,
        "decisions_indexed": indexed,
    })))
}

/// Backfill embeddings for all decisions that don't have one yet.
///
/// Returns synchronously with the count of decisions processed.
pub async fn backfill_decision_embeddings(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let (total, created) = state
        .orchestrator
        .plan_manager()
        .backfill_decision_embeddings()
        .await?;

    Ok(Json(serde_json::json!({
        "decisions_processed": total,
        "embeddings_created": created,
    })))
}

/// POST /api/admin/backfill-decision-project-slugs — Backfill project_slug on DecisionDocuments in Meilisearch
pub async fn backfill_decision_project_slugs(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let (total, updated) = state
        .orchestrator
        .plan_manager()
        .backfill_decision_project_slugs()
        .await?;

    Ok(Json(serde_json::json!({
        "decisions_processed": total,
        "decisions_updated": updated,
    })))
}

/// POST /api/admin/backfill-discussed — Backfill DISCUSSED relations on existing sessions
pub async fn backfill_discussed(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let (sessions, entities, relations) = state
        .orchestrator
        .neo4j()
        .backfill_discussed()
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "sessions_processed": sessions,
        "entities_found": entities,
        "relations_created": relations,
    })))
}

// ============================================================================
// Knowledge Fabric Admin
// ============================================================================

/// Request body for update-fabric-scores and bootstrap-knowledge-fabric
#[derive(Deserialize)]
pub struct FabricProjectRequest {
    pub project_id: Uuid,
}

/// POST /api/admin/update-fabric-scores
///
/// Orchestrates the full fabric analytics pipeline: extracts the multi-layer
/// fabric graph (IMPORTS + CO_CHANGED + SYNAPSE) and computes PageRank,
/// Louvain, and Betweenness scores on it.
pub async fn update_fabric_scores(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    let _project = state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(120);

    // Compute fabric graph analytics with timeout (IMPORTS + CO_CHANGED + SYNAPSE layers)
    let weights = crate::graph::models::FabricWeights::default();
    let analytics = tokio::time::timeout(timeout, async {
        state
            .orchestrator
            .analytics()
            .analyze_fabric_graph(project_id, &weights)
            .await
    })
    .await
    .map_err(|_| {
        AppError::Internal(anyhow::anyhow!(
            "Fabric scores computation timed out after 120s"
        ))
    })?
    .map_err(AppError::Internal)?;

    let computation_ms = start.elapsed().as_millis() as u64;

    if computation_ms > 10_000 {
        tracing::warn!(
            project_id = %project_id,
            nodes = analytics.metrics.len(),
            elapsed_ms = computation_ms,
            "Large graph: fabric scores computation took >10s"
        );
    }

    // Also compute churn, knowledge_density, and risk scores
    let neo = state.orchestrator.neo4j();
    let churn = neo
        .compute_churn_scores(project_id)
        .await
        .unwrap_or_default();
    let churn_count = churn.len();
    let _ = neo.batch_update_churn_scores(&churn).await;

    let density = neo
        .compute_knowledge_density(project_id)
        .await
        .unwrap_or_default();
    let density_count = density.len();
    let _ = neo.batch_update_knowledge_density(&density).await;

    let risk = neo
        .compute_risk_scores(project_id)
        .await
        .unwrap_or_default();
    let risk_count = risk.len();
    let _ = neo.batch_update_risk_scores(&risk).await;

    Ok(Json(serde_json::json!({
        "nodes_updated": analytics.metrics.len(),
        "computation_ms": start.elapsed().as_millis() as u64,
        "fabric_scores_computed": true,
        "communities": analytics.communities.len(),
        "components": analytics.components.len(),
        "churn_scores_computed": churn_count,
        "knowledge_density_computed": density_count,
        "risk_scores_computed": risk_count,
    })))
}

/// POST /api/admin/audit-gaps
///
/// Audit the knowledge graph for a project and return a structured report
/// of gaps: orphan notes, decisions without AFFECTS, commits without TOUCHES,
/// skills without members, and relationship type inventory.
/// Used by the system-inference protocol (AUDIT_GAPS state).
pub async fn audit_gaps(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let report = state
        .orchestrator
        .neo4j()
        .audit_knowledge_gaps(project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(report).unwrap()))
}

/// POST /api/admin/persist-health-report
///
/// Run health diagnostics (health + knowledge_gaps + risk_assessment) and
/// persist the result as a Note (type: observation, tags: health-check).
/// Compares with previous health-check note for delta analysis.
/// Used by the system-inference protocol (HEALTH_CHECK state).
pub async fn persist_health_report(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    let project = state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    // 1. Collect health data
    let health = state
        .orchestrator
        .neo4j()
        .get_code_health_report(project_id, 200)
        .await
        .map_err(AppError::Internal)?;

    let gaps = state
        .orchestrator
        .neo4j()
        .audit_knowledge_gaps(project_id)
        .await
        .map_err(AppError::Internal)?;

    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

    // 2. Build report content
    let report = serde_json::json!({
        "date": today,
        "project": project.name,
        "health": {
            "god_functions": health.god_functions.len(),
            "orphan_files": health.orphan_files.len(),
            "coupling": health.coupling_metrics,
        },
        "gaps": {
            "total": gaps.total_gaps,
            "orphan_notes": gaps.orphan_notes.len(),
            "decisions_without_affects": gaps.decisions_without_affects.len(),
            "commits_without_touches": gaps.commits_without_touches.len(),
            "skills_without_members": gaps.skills_without_members.len(),
        },
        "relationship_types": gaps.relationship_type_counts.len(),
    });

    let content = format!(
        "## Health Report — {}\n\n### Code Health\n- God functions: {}\n- Orphan files: {}\n\n### Knowledge Gaps\n- Total gaps: {}\n- Orphan notes (no links): {}\n- Decisions without AFFECTS: {}\n- Commits without TOUCHES: {}\n- Skills without members: {}\n\n### Raw Data\n```json\n{}\n```",
        today,
        health.god_functions.len(),
        health.orphan_files.len(),
        gaps.total_gaps,
        gaps.orphan_notes.len(),
        gaps.decisions_without_affects.len(),
        gaps.commits_without_touches.len(),
        gaps.skills_without_members.len(),
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    );

    // 3. Search for previous health-check note to compute delta
    use crate::notes::models::{
        CreateNoteRequest, EntityType, LinkNoteRequest, NoteFilters, NoteImportance, NoteType,
    };
    let filters = NoteFilters {
        tags: Some(vec!["health-check".to_string()]),
        ..Default::default()
    };
    let previous_notes = state
        .orchestrator
        .note_manager()
        .search_notes("health-check", &filters)
        .await
        .unwrap_or_default();

    // Extract previous metrics from the Raw Data JSON block in the previous note
    let (delta, delta_section) = if let Some(prev_hit) = previous_notes.first() {
        // Parse the JSON from the ```json ... ``` block in the previous note
        let prev_json = prev_hit
            .note
            .content
            .split("```json")
            .nth(1)
            .and_then(|s| s.split("```").next())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s.trim()).ok());

        if let Some(prev_data) = prev_json {
            let prev_god = prev_data["health"]["god_functions"].as_u64().unwrap_or(0) as usize;
            let prev_orphans = prev_data["health"]["orphan_files"].as_u64().unwrap_or(0) as usize;
            let prev_gaps_total = prev_data["gaps"]["total"].as_u64().unwrap_or(0) as usize;
            let prev_coupling = prev_data["health"]["coupling"]["avg_clustering_coefficient"]
                .as_f64()
                .unwrap_or(0.0);

            let cur_god = health.god_functions.len();
            let cur_orphans = health.orphan_files.len();
            let cur_coupling = health
                .coupling_metrics
                .as_ref()
                .map(|c| c.avg_clustering_coefficient)
                .unwrap_or(0.0);

            // Compute deltas with semantic direction
            let compute_delta =
                |metric: &str, cur: f64, prev: f64, increase_is_bad: bool| -> String {
                    let change_pct = if prev == 0.0 {
                        if cur == 0.0 {
                            0.0
                        } else {
                            100.0
                        }
                    } else {
                        (cur - prev) / prev * 100.0
                    };

                    let (emoji, direction) = if change_pct.abs() < 5.0 {
                        ("→", "Stable")
                    } else if increase_is_bad {
                        if cur > prev {
                            ("⚠️", "Degrading")
                        } else {
                            ("✅", "Improving")
                        }
                    } else if cur < prev {
                        ("✅", "Improving")
                    } else {
                        ("⚠️", "Degrading")
                    };

                    format!(
                        "- {}: {} → {} ({:+.1}%) {} {}",
                        metric, prev, cur, change_pct, emoji, direction
                    )
                };

            let prev_date = prev_hit
                .note
                .tags
                .iter()
                .find(|t| t.starts_with("20"))
                .cloned()
                .unwrap_or_else(|| "previous".to_string());

            let mut lines = Vec::new();
            lines.push(format!("\n## Δ vs previous ({})\n", prev_date));
            lines.push(compute_delta(
                "avg_coupling",
                cur_coupling,
                prev_coupling,
                true,
            ));
            lines.push(compute_delta(
                "god_functions",
                cur_god as f64,
                prev_god as f64,
                true,
            ));
            lines.push(compute_delta(
                "orphan_files",
                cur_orphans as f64,
                prev_orphans as f64,
                true,
            ));
            lines.push(compute_delta(
                "total_gaps",
                gaps.total_gaps as f64,
                prev_gaps_total as f64,
                true,
            ));

            let section = lines.join("\n");

            let delta_json = serde_json::json!({
                "previous_date": prev_date,
                "metrics": {
                    "coupling": {
                        "previous": prev_coupling,
                        "current": cur_coupling,
                    },
                    "god_functions": {
                        "previous": prev_god,
                        "current": cur_god,
                    },
                    "orphan_files": {
                        "previous": prev_orphans,
                        "current": cur_orphans,
                    },
                    "total_gaps": {
                        "previous": prev_gaps_total,
                        "current": gaps.total_gaps,
                    },
                },
            });

            (Some(delta_json), section)
        } else {
            // Fallback: try simple text parsing for gaps only
            let prev_gaps: Option<usize> = prev_hit
                .note
                .content
                .lines()
                .find(|l| l.contains("Total gaps:"))
                .and_then(|l| l.split(':').next_back())
                .and_then(|s| s.trim().parse().ok());

            let delta_json = prev_gaps.map(|prev| serde_json::json!({
                "previous_total_gaps": prev,
                "current_total_gaps": gaps.total_gaps,
                "delta": gaps.total_gaps as i64 - prev as i64,
                "trend": if gaps.total_gaps < prev { "improving" } else if gaps.total_gaps > prev { "degrading" } else { "stable" },
            }));

            (delta_json, String::new())
        }
    } else {
        (None, String::new())
    };

    // Append delta section to content
    let content = if delta_section.is_empty() {
        content
    } else {
        format!("{}{}", content, delta_section)
    };

    // 4. Create the note
    let create_req = CreateNoteRequest {
        project_id: Some(project_id),
        note_type: NoteType::Observation,
        content,
        importance: Some(NoteImportance::Medium),
        scope: None,
        tags: Some(vec![
            "health-check".to_string(),
            "auto-generated".to_string(),
            today.clone(),
        ]),
        anchors: None,
        assertion_rule: None,
        run_id: None,
    };
    let note = state
        .orchestrator
        .note_manager()
        .create_note(create_req, "system-inference")
        .await
        .map_err(AppError::Internal)?;

    // 5. Link note to project
    let link_req = LinkNoteRequest {
        entity_type: EntityType::Project,
        entity_id: project_id.to_string(),
    };
    let _ = state
        .orchestrator
        .note_manager()
        .link_note_to_entity(note.id, &link_req)
        .await;

    Ok(Json(serde_json::json!({
        "note_id": note.id.to_string(),
        "report": report,
        "delta": delta,
    })))
}

/// POST /api/admin/bootstrap-knowledge-fabric
///
/// Chains ALL knowledge fabric backfill steps in order, then computes
/// fabric scores. Each step is best-effort (continues on failure).
/// Returns a report of completed and failed steps.
pub async fn bootstrap_knowledge_fabric(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    let project = state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let start = std::time::Instant::now();
    let mut completed = Vec::new();
    let mut failed = Vec::new();

    // Step 1: Backfill commit touches
    let root_path = std::path::PathBuf::from(crate::expand_tilde(&project.root_path));
    match state
        .orchestrator
        .backfill_commit_touches(project_id, &root_path)
        .await
    {
        Ok(result) => completed.push(serde_json::json!({
            "step": "backfill_touches",
            "commits_parsed": result.commits_parsed,
            "commits_backfilled": result.commits_backfilled,
            "touches_created": result.touches_created,
        })),
        Err(e) => failed.push(serde_json::json!({
            "step": "backfill_touches",
            "error": e.to_string(),
        })),
    }

    // Step 2: Reindex decisions from Neo4j into MeiliSearch
    match state.orchestrator.plan_manager().reindex_decisions().await {
        Ok((total, indexed)) => completed.push(serde_json::json!({
            "step": "reindex_decisions",
            "decisions_processed": total,
            "decisions_indexed": indexed,
        })),
        Err(e) => failed.push(serde_json::json!({
            "step": "reindex_decisions",
            "error": e.to_string(),
        })),
    }

    // Step 2b: Backfill decision embeddings
    match state
        .orchestrator
        .plan_manager()
        .backfill_decision_embeddings()
        .await
    {
        Ok((total, created)) => completed.push(serde_json::json!({
            "step": "backfill_decision_embeddings",
            "decisions_processed": total,
            "embeddings_created": created,
        })),
        Err(e) => failed.push(serde_json::json!({
            "step": "backfill_decision_embeddings",
            "error": e.to_string(),
        })),
    }

    // Step 2c: Backfill decision project_slugs in Meilisearch
    match state
        .orchestrator
        .plan_manager()
        .backfill_decision_project_slugs()
        .await
    {
        Ok((total, updated)) => completed.push(serde_json::json!({
            "step": "backfill_decision_project_slugs",
            "decisions_processed": total,
            "decisions_updated": updated,
        })),
        Err(e) => failed.push(serde_json::json!({
            "step": "backfill_decision_project_slugs",
            "error": e.to_string(),
        })),
    }

    // Step 3: Backfill DISCUSSED relations
    match state.orchestrator.neo4j().backfill_discussed().await {
        Ok((sessions, entities, relations)) => completed.push(serde_json::json!({
            "step": "backfill_discussed",
            "sessions_processed": sessions,
            "entities_found": entities,
            "relations_created": relations,
        })),
        Err(e) => failed.push(serde_json::json!({
            "step": "backfill_discussed",
            "error": e.to_string(),
        })),
    }

    // Step 4: Update fabric scores (the final analytics computation)
    let weights = crate::graph::models::FabricWeights::default();
    let timeout = std::time::Duration::from_secs(120);
    match tokio::time::timeout(
        timeout,
        state
            .orchestrator
            .analytics()
            .analyze_fabric_graph(project_id, &weights),
    )
    .await
    {
        Ok(Ok(analytics)) => completed.push(serde_json::json!({
            "step": "update_fabric_scores",
            "nodes_updated": analytics.metrics.len(),
            "communities": analytics.communities.len(),
        })),
        Ok(Err(e)) => failed.push(serde_json::json!({
            "step": "update_fabric_scores",
            "error": e.to_string(),
        })),
        Err(_) => failed.push(serde_json::json!({
            "step": "update_fabric_scores",
            "error": "Timed out after 120s",
        })),
    }

    // Step 5: Compute churn, knowledge density, and risk scores
    let neo = state.orchestrator.neo4j();
    if let Ok(churn) = neo.compute_churn_scores(project_id).await {
        let count = churn.len();
        let _ = neo.batch_update_churn_scores(&churn).await;
        completed.push(serde_json::json!({"step": "churn_scores", "files_scored": count}));
    }
    if let Ok(density) = neo.compute_knowledge_density(project_id).await {
        let count = density.len();
        let _ = neo.batch_update_knowledge_density(&density).await;
        completed.push(serde_json::json!({"step": "knowledge_density", "files_scored": count}));
    }
    if let Ok(risk) = neo.compute_risk_scores(project_id).await {
        let count = risk.len();
        let _ = neo.batch_update_risk_scores(&risk).await;
        completed.push(serde_json::json!({"step": "risk_scores", "files_scored": count}));
    }

    let total_time_ms = start.elapsed().as_millis() as u64;

    Ok(Json(serde_json::json!({
        "steps_completed": completed,
        "steps_failed": failed,
        "total_time_ms": total_time_ms,
    })))
}

// ============================================================================
// Isomorphic Synapse Reinforcement
// ============================================================================

/// POST /api/admin/reinforce-isomorphic — Reinforce synapses between notes linked to
/// structurally isomorphic files (same WL hash).
pub async fn reinforce_isomorphic_synapses(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    let _project = state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let neo4j = state.orchestrator.neo4j();

    // Find isomorphic groups (files sharing same WL hash)
    let groups = neo4j
        .find_isomorphic_groups(&project_id.to_string(), 2)
        .await
        .map_err(AppError::Internal)?;

    let mut total_groups = 0usize;
    let mut total_synapses_reinforced = 0u64;
    let synapse_boost = 0.05;

    for group in &groups {
        // Collect note IDs linked to files in this group
        let mut note_ids: Vec<uuid::Uuid> = Vec::new();
        let entity_type = crate::notes::models::EntityType::File;
        for file_path in &group.members {
            if let Ok(notes) = neo4j.get_notes_for_entity(&entity_type, file_path).await {
                for note in notes {
                    if !note_ids.contains(&note.id) {
                        note_ids.push(note.id);
                    }
                }
            }
        }

        // Reinforce synapses if >= 2 notes
        if note_ids.len() >= 2 {
            // Boost energy for each note
            for nid in &note_ids {
                let _ = neo4j.boost_energy(*nid, 0.1).await;
                // Emit reinforcement graph event
                state
                    .event_bus
                    .emit_graph(crate::events::GraphEvent::reinforcement(
                        nid.to_string(),
                        0.1,
                        project_id.to_string(),
                    ));
            }
            // Reinforce synapses between all note pairs
            match neo4j.reinforce_synapses(&note_ids, synapse_boost).await {
                Ok(count) => {
                    total_synapses_reinforced += count as u64;
                    total_groups += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        wl_hash = group.wl_hash,
                        error = %e,
                        "Failed to reinforce synapses for isomorphic group"
                    );
                }
            }
        }
    }

    Ok(Json(serde_json::json!({
        "isomorphic_groups_found": groups.len(),
        "groups_with_notes": total_groups,
        "synapses_reinforced": total_synapses_reinforced,
        "synapse_boost": synapse_boost,
    })))
}

// ============================================================================
// Skill Detection
// ============================================================================

/// Request to run skill detection pipeline
#[derive(Deserialize)]
pub struct DetectSkillsRequest {
    pub project_id: Uuid,
    /// When true, delete all existing skills before re-detecting from scratch.
    /// Default: false (incremental deduplication via Jaccard overlap).
    #[serde(default)]
    pub force: bool,
}

/// Run the full skill detection pipeline for a project
pub async fn detect_skills(
    State(state): State<OrchestratorState>,
    Json(body): Json<DetectSkillsRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    // Force mode: delete all existing skills before re-detecting from scratch
    if body.force {
        let existing_skills = state
            .orchestrator
            .neo4j()
            .get_skills_for_project(project_id)
            .await
            .map_err(AppError::Internal)?;
        for skill in &existing_skills {
            let _ = state.orchestrator.neo4j().delete_skill(skill.id).await;
        }
        tracing::info!(
            %project_id,
            deleted = existing_skills.len(),
            "Force mode: deleted {} existing skills before re-detection",
            existing_skills.len()
        );
    }

    let config = crate::skills::SkillDetectionConfig::default();
    let start = std::time::Instant::now();

    let result =
        crate::skills::detect_skills_pipeline(state.orchestrator.neo4j(), project_id, &config)
            .await
            .map_err(AppError::Internal)?;

    let elapsed_ms = start.elapsed().as_millis() as u64;

    // Invalidate hook activation cache — skills have changed
    super::hook_handlers::skill_cache()
        .invalidate_project(&project_id)
        .await;

    Ok(Json(serde_json::json!({
        "status": format!("{:?}", result.status),
        "skills_detected": result.skills_detected,
        "skills_created": result.skills_created,
        "skills_updated": result.skills_updated,
        "total_notes": result.total_notes,
        "total_synapses": result.total_synapses,
        "modularity": result.modularity,
        "message": result.message,
        "skill_ids": result.skill_ids,
        "elapsed_ms": elapsed_ms,
    })))
}

// ============================================================================
// Skill Fission / Fusion Detection (read-only inspection)
// ============================================================================

/// POST /api/admin/detect-skill-fission
///
/// Detect skills that are candidates for splitting (fission).
/// Read-only — does not modify the graph.
pub async fn detect_skill_fission(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let start = std::time::Instant::now();
    let candidates = crate::skills::evolution::detect_skill_fission(
        state.orchestrator.neo4j(),
        project_id,
        0.5, // overlap threshold for sub-cluster matching
    )
    .await
    .map_err(AppError::Internal)?;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(Json(serde_json::json!({
        "candidates": candidates,
        "count": candidates.len(),
        "elapsed_ms": elapsed_ms,
    })))
}

/// POST /api/admin/detect-skill-fusion
///
/// Detect pairs of skills that are candidates for merging (fusion).
/// Read-only — does not modify the graph.
pub async fn detect_skill_fusion(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let start = std::time::Instant::now();
    let candidates = crate::skills::evolution::detect_skill_fusion(
        state.orchestrator.neo4j(),
        project_id,
        0.5, // overlap threshold for fusion detection
    )
    .await
    .map_err(AppError::Internal)?;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(Json(serde_json::json!({
        "candidates": candidates,
        "count": candidates.len(),
        "elapsed_ms": elapsed_ms,
    })))
}

// ============================================================================
// Auto-anchor Notes
// ============================================================================

/// POST /api/admin/auto-anchor-notes
///
/// Scan all notes for a project, extract file paths from their content,
/// and create LINKED_TO relations to matching File nodes in the graph.
/// Idempotent — safe to run multiple times.
pub async fn auto_anchor_notes(
    State(state): State<OrchestratorState>,
    Json(body): Json<FabricProjectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let start = std::time::Instant::now();

    let result = crate::skills::activation::auto_anchor_notes_for_project(
        state.orchestrator.neo4j(),
        project_id,
    )
    .await
    .map_err(AppError::Internal)?;

    let elapsed_ms = start.elapsed().as_millis() as u64;

    Ok(Json(serde_json::json!({
        "notes_processed": result.notes_processed,
        "anchors_created": result.anchors_created,
        "elapsed_ms": elapsed_ms,
        "root_path_resolved": result.root_path_resolved,
    })))
}

// ============================================================================
// Reconstruct Knowledge Links
// ============================================================================

/// POST /api/admin/reconstruct-knowledge
///
/// Reconstruct all knowledge links (LINKED_TO, AFFECTS) for a project.
/// Processes all notes (cross-project) and all decisions for the project,
/// creating file anchors based on path mentions in content.
/// Idempotent — safe to run multiple times (uses MERGE).
///
/// If `project_id` is provided, reconstructs for that project only.
/// If omitted, iterates all projects and reconstructs for each.
pub async fn reconstruct_knowledge(
    State(state): State<OrchestratorState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, AppError> {
    let start = std::time::Instant::now();

    let project_id = body
        .get("project_id")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<uuid::Uuid>().ok());

    if let Some(pid) = project_id {
        // Single project
        state
            .orchestrator
            .neo4j()
            .get_project(pid)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", pid)))?;

        let report =
            crate::skills::activation::reconstruct_knowledge_links(state.orchestrator.neo4j(), pid)
                .await
                .map_err(AppError::Internal)?;

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(Json(serde_json::json!({
            "projects_processed": 1,
            "notes_processed": report.notes_processed,
            "notes_linked": report.notes_linked,
            "decisions_processed": report.decisions_processed,
            "affects_created": report.affects_created,
            "elapsed_ms": elapsed_ms,
        })))
    } else {
        // All projects
        let projects = state
            .orchestrator
            .neo4j()
            .list_projects()
            .await
            .map_err(AppError::Internal)?;

        let mut total_notes_processed = 0usize;
        let mut total_notes_linked = 0usize;
        let mut total_decisions_processed = 0usize;
        let mut total_affects_created = 0usize;
        let projects_count = projects.len();

        for project in &projects {
            match crate::skills::activation::reconstruct_knowledge_links(
                state.orchestrator.neo4j(),
                project.id,
            )
            .await
            {
                Ok(report) => {
                    total_notes_processed += report.notes_processed;
                    total_notes_linked += report.notes_linked;
                    total_decisions_processed += report.decisions_processed;
                    total_affects_created += report.affects_created;
                }
                Err(e) => {
                    tracing::warn!(
                        project_id = %project.id,
                        error = %e,
                        "reconstruct_knowledge failed for project"
                    );
                }
            }
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(Json(serde_json::json!({
            "projects_processed": projects_count,
            "notes_processed": total_notes_processed,
            "notes_linked": total_notes_linked,
            "decisions_processed": total_decisions_processed,
            "affects_created": total_affects_created,
            "elapsed_ms": elapsed_ms,
        })))
    }
}

// ============================================================================
// Skill Maintenance
// ============================================================================

/// Request to run skill maintenance
#[derive(Deserialize)]
pub struct SkillMaintenanceRequest {
    pub project_id: Uuid,
    #[serde(default = "default_maintenance_level")]
    pub level: String,
}

fn default_maintenance_level() -> String {
    "daily".to_string()
}

/// Run skill maintenance for a project
pub async fn skill_maintenance(
    State(state): State<OrchestratorState>,
    Json(body): Json<SkillMaintenanceRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = body.project_id;

    // Verify project exists
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let level = body.level.as_str();
    if !matches!(level, "hourly" | "daily" | "weekly" | "full") {
        return Err(AppError::BadRequest(format!(
            "Invalid maintenance level: '{}'. Expected: hourly, daily, weekly, full",
            level
        )));
    }

    let config = crate::skills::maintenance::SkillMaintenanceConfig::default();

    let (result, report) = crate::skills::maintenance::run_maintenance_with_tracking(
        state.orchestrator.neo4j(),
        project_id,
        level,
        &config,
    )
    .await
    .map_err(AppError::Internal)?;

    // Invalidate hook activation cache after maintenance
    super::hook_handlers::skill_cache()
        .invalidate_project(&project_id)
        .await;

    Ok(Json(serde_json::json!({
        "level": result.level,
        "lifecycle": result.lifecycle,
        "synapses_decayed": result.synapses_decayed,
        "synapses_pruned": result.synapses_pruned,
        "evolution": result.evolution,
        "skills_detected": result.skills_detected,
        "warnings": result.warnings,
        "elapsed_ms": report.duration_ms,
        "tracking": {
            "before": report.before,
            "after": report.after,
            "delta_health_score": report.delta_health_score,
            "delta_active_synapses": report.delta_active_synapses,
            "delta_mean_energy": report.delta_mean_energy,
            "delta_skill_count": report.delta_skill_count,
            "delta_note_count": report.delta_note_count,
            "success_rate": report.success_rate,
        },
    })))
}

// ============================================================================
// Stagnation Detection & Deep Maintenance (biomimicry T12)
// ============================================================================

/// Detect global stagnation across a project.
pub async fn detect_stagnation(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let report = state
        .orchestrator
        .neo4j()
        .detect_global_stagnation(project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(&report).unwrap_or_default()))
}

/// Run deep maintenance (aggressive cleanup for stagnating projects).
pub async fn run_deep_maintenance(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let config = crate::skills::maintenance::SkillMaintenanceConfig::default();

    let report = crate::skills::maintenance::deep_maintenance(
        state.orchestrator.neo4j(),
        project_id,
        &config,
    )
    .await
    .map_err(AppError::Internal)?;

    // Invalidate hook activation cache after deep maintenance
    super::hook_handlers::skill_cache()
        .invalidate_project(&project_id)
        .await;

    Ok(Json(serde_json::to_value(&report).unwrap_or_default()))
}

/// Seed prompt fragments for the 5 critical protocols.
///
/// Populates `prompt_fragment`, `available_tools`, and `forbidden_actions`
/// on existing protocol states. Idempotent — safe to run multiple times.
pub async fn seed_prompt_fragments(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .orchestrator
        .neo4j()
        .get_project(project_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Project not found: {}", project_id)))?;

    let result =
        crate::protocol::seed::seed_prompt_fragments(state.orchestrator.neo4j(), project_id)
            .await
            .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(&result).unwrap_or_default()))
}

// ============================================================================
// Alerts (HeartbeatEngine)
// ============================================================================

/// Query parameters for listing alerts.
#[derive(Debug, Deserialize, Default)]
pub struct AlertsListQuery {
    /// Filter by project ID.
    pub project_id: Option<Uuid>,
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

/// List alerts (optionally filtered by project).
pub async fn list_alerts(
    State(state): State<OrchestratorState>,
    Query(query): Query<AlertsListQuery>,
) -> Result<Json<PaginatedResponse<crate::neo4j::models::AlertNode>>, AppError> {
    let limit = query.pagination.limit.min(200);
    let offset = query.pagination.offset;

    let (alerts, total) = state
        .orchestrator
        .neo4j()
        .list_alerts(query.project_id, limit, offset)
        .await
        .map_err(AppError::Internal)?;

    let has_more = offset + alerts.len() < total;
    Ok(Json(PaginatedResponse {
        items: alerts,
        total,
        limit,
        offset,
        has_more,
    }))
}

/// Request body for acknowledging an alert.
#[derive(Debug, Deserialize)]
pub struct AcknowledgeAlertRequest {
    /// Who is acknowledging (username or "system").
    pub acknowledged_by: String,
}

/// Acknowledge (dismiss) an alert.
pub async fn acknowledge_alert(
    State(state): State<OrchestratorState>,
    Path(alert_id): Path<Uuid>,
    Json(req): Json<AcknowledgeAlertRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Verify the alert exists
    state
        .orchestrator
        .neo4j()
        .get_alert(alert_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Alert not found: {}", alert_id)))?;

    state
        .orchestrator
        .neo4j()
        .acknowledge_alert(alert_id, &req.acknowledged_by)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "id": alert_id,
        "acknowledged": true,
        "acknowledged_by": req.acknowledged_by,
    })))
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
    state.event_bus.emit_created(
        crate::events::EntityType::Release,
        &release.id.to_string(),
        serde_json::json!({"version": &release.version, "title": &release.title}),
        Some(project_id.to_string()),
    );
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
    let status_change = if let Some(ref new_status_val) = req.status {
        let old_status = state
            .orchestrator
            .neo4j()
            .get_release(release_id)
            .await
            .ok()
            .flatten()
            .map(|r| format!("{:?}", r.status))
            .unwrap_or_default();
        let new_status = format!("{:?}", new_status_val);
        Some((old_status, new_status))
    } else {
        None
    };
    let payload = serde_json::json!({
        "title": &req.title,
        "description": &req.description,
    });
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
    if let Some((old_status, new_status)) = status_change {
        state.event_bus.emit_status_changed(
            crate::events::EntityType::Release,
            &release_id.to_string(),
            &old_status,
            &new_status,
            None,
        );
    } else {
        state.event_bus.emit_updated(
            crate::events::EntityType::Release,
            &release_id.to_string(),
            payload,
            None,
        );
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a release
pub async fn delete_release(
    State(state): State<OrchestratorState>,
    Path(release_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_release(release_id).await?;
    state.event_bus.emit_deleted(
        crate::events::EntityType::Release,
        &release_id.to_string(),
        None,
    );
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
    state.event_bus.emit_linked(
        crate::events::EntityType::Release,
        &release_id.to_string(),
        crate::events::EntityType::Task,
        &req.task_id.to_string(),
        None,
    );
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
    state.event_bus.emit_created(
        crate::events::EntityType::Milestone,
        &milestone.id.to_string(),
        serde_json::json!({"title": &milestone.title, "description": &milestone.description}),
        Some(project_id.to_string()),
    );
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
    let status_change = if let Some(ref new_status_val) = req.status {
        let old_status = state
            .orchestrator
            .neo4j()
            .get_milestone(milestone_id)
            .await
            .ok()
            .flatten()
            .map(|m| format!("{:?}", m.status))
            .unwrap_or_default();
        let new_status = format!("{:?}", new_status_val);
        Some((old_status, new_status))
    } else {
        None
    };
    let payload = serde_json::json!({
        "title": &req.title,
        "description": &req.description,
    });
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
    if let Some((old_status, new_status)) = status_change {
        state.event_bus.emit_status_changed(
            crate::events::EntityType::Milestone,
            &milestone_id.to_string(),
            &old_status,
            &new_status,
            None,
        );
    } else {
        state.event_bus.emit_updated(
            crate::events::EntityType::Milestone,
            &milestone_id.to_string(),
            payload,
            None,
        );
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a milestone
pub async fn delete_milestone(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state.orchestrator.delete_milestone(milestone_id).await?;
    state.event_bus.emit_deleted(
        crate::events::EntityType::Milestone,
        &milestone_id.to_string(),
        None,
    );
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
    state.event_bus.emit_linked(
        crate::events::EntityType::Milestone,
        &milestone_id.to_string(),
        crate::events::EntityType::Task,
        &req.task_id.to_string(),
        None,
    );
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
    state.event_bus.emit_linked(
        crate::events::EntityType::Milestone,
        &milestone_id.to_string(),
        crate::events::EntityType::Plan,
        &req.plan_id.to_string(),
        None,
    );
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
    pub plans: Vec<MilestonePlanSummary>,
    pub tasks: Vec<TaskNode>,
    pub progress: MilestoneProgressResponse,
}

/// Query parameters for GET /api/milestones/:id
#[derive(Deserialize)]
pub struct MilestoneGetQuery {
    /// Include the flat top-level tasks list in the response (default: true for backward compat).
    /// The hierarchical plans → tasks → steps are always returned regardless.
    pub include_tasks: Option<bool>,
}

/// Get milestone details
pub async fn get_milestone(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
    Query(query): Query<MilestoneGetQuery>,
) -> Result<Json<MilestoneDetailsResponse>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let include_tasks = query.include_tasks.unwrap_or(true);

    // Fetch milestone node — use lightweight query when flat tasks are not needed
    let (milestone, flat_tasks) = if include_tasks {
        let (m, tasks) = neo4j
            .get_milestone_details(milestone_id)
            .await?
            .ok_or(AppError::NotFound("Milestone not found".into()))?;
        (m, tasks)
    } else {
        let m = neo4j
            .get_milestone(milestone_id)
            .await?
            .ok_or(AppError::NotFound("Milestone not found".into()))?;
        (m, vec![])
    };

    // Always load plans → tasks → steps hierarchy
    let tasks_with_plan = neo4j.get_milestone_tasks_with_plans(milestone_id).await?;

    let (total, completed, in_progress, pending) =
        neo4j.get_milestone_progress(milestone_id).await?;

    let percentage = if total > 0 {
        (completed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let mut steps_map = neo4j.get_milestone_steps_batch(milestone_id).await?;

    let mut plan_order: Vec<Uuid> = Vec::new();
    let mut plan_map: std::collections::HashMap<
        Uuid,
        (String, Option<String>, Vec<MilestoneTaskSummary>),
    > = std::collections::HashMap::new();

    for twp in tasks_with_plan {
        let steps = steps_map
            .remove(&twp.task.id)
            .unwrap_or_default()
            .into_iter()
            .map(|s| MilestoneStepSummary {
                id: s.id.to_string(),
                order: s.order,
                description: s.description,
                status: serde_json::to_value(&s.status)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string(),
                verification: s.verification,
            })
            .collect();

        let task_summary = MilestoneTaskSummary {
            id: twp.task.id.to_string(),
            title: twp.task.title,
            description: twp.task.description,
            status: serde_json::to_value(&twp.task.status)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
            priority: twp.task.priority,
            tags: twp.task.tags,
            created_at: twp.task.created_at.to_rfc3339(),
            completed_at: twp.task.completed_at.map(|dt| dt.to_rfc3339()),
            steps,
        };

        let entry = plan_map.entry(twp.plan_id).or_insert_with(|| {
            plan_order.push(twp.plan_id);
            (twp.plan_title.clone(), twp.plan_status.clone(), vec![])
        });
        entry.2.push(task_summary);
    }

    let plans: Vec<MilestonePlanSummary> = plan_order
        .into_iter()
        .filter_map(|pid| {
            let (title, status, tasks) = plan_map.remove(&pid)?;
            Some(MilestonePlanSummary {
                id: pid.to_string(),
                title,
                status,
                tasks,
            })
        })
        .collect();

    Ok(Json(MilestoneDetailsResponse {
        milestone,
        plans,
        tasks: flat_tasks,
        progress: MilestoneProgressResponse {
            total,
            completed,
            in_progress,
            pending,
            percentage,
        },
    }))
}

/// Milestone progress response
#[derive(Serialize)]
pub struct MilestoneProgressResponse {
    pub total: u32,
    pub completed: u32,
    pub in_progress: u32,
    pub pending: u32,
    pub percentage: f64,
}

/// Get milestone progress
pub async fn get_milestone_progress(
    State(state): State<OrchestratorState>,
    Path(milestone_id): Path<Uuid>,
) -> Result<Json<MilestoneProgressResponse>, AppError> {
    let (total, completed, in_progress, pending) = state
        .orchestrator
        .neo4j()
        .get_milestone_progress(milestone_id)
        .await?;

    let percentage = if total > 0 {
        (completed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    Ok(Json(MilestoneProgressResponse {
        total,
        completed,
        in_progress,
        pending,
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

/// Step summary for dependency graph visualization
#[derive(Serialize)]
pub struct DependencyGraphStep {
    pub id: String,
    pub order: u32,
    pub description: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification: Option<String>,
}

/// Chat session summary for dependency graph visualization
#[derive(Serialize)]
pub struct DependencyGraphSession {
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub is_active: bool,
    pub child_count: usize,
}

/// Discussed file for dependency graph visualization
#[derive(Serialize)]
pub struct DependencyGraphDiscussedFile {
    pub file_path: String,
    pub mention_count: i64,
}

/// Feature graph summary for dependency graph response
#[derive(Serialize)]
pub struct FeatureGraphSummary {
    pub id: String,
    pub name: String,
    pub entity_count: usize,
}

/// Dependency graph node for visualization (enriched to match Wave view)
#[derive(Serialize)]
pub struct DependencyGraphNode {
    pub id: Uuid,
    pub title: Option<String>,
    pub description: String,
    pub status: String,
    pub priority: Option<i32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub affected_files: Vec<String>,
    pub assigned_to: Option<String>,
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    /// Number of steps for this task
    pub step_count: usize,
    /// Number of completed steps
    pub completed_step_count: usize,
    /// Number of notes linked to this task
    pub note_count: usize,
    /// Number of decisions linked to this task
    pub decision_count: usize,
    /// Individual step details (ordered)
    #[serde(default)]
    pub steps: Vec<DependencyGraphStep>,
    /// Number of chat sessions linked to this task
    #[serde(default)]
    pub session_count: usize,
    /// Number of currently active (streaming) sessions
    #[serde(default)]
    pub active_session_count: usize,
    /// Total number of child sessions (sub-discussions)
    #[serde(default)]
    pub child_session_count: usize,
    /// Files discussed in linked chat sessions
    #[serde(default)]
    pub discussed_files: Vec<DependencyGraphDiscussedFile>,
}

/// Dependency graph edge
#[derive(Serialize)]
pub struct DependencyGraphEdge {
    pub from: Uuid,
    pub to: Uuid,
}

/// Dependency graph response (enriched to match Wave view)
#[derive(Serialize)]
pub struct DependencyGraphResponse {
    pub nodes: Vec<DependencyGraphNode>,
    pub edges: Vec<DependencyGraphEdge>,
    /// File conflicts between tasks (same as Wave view)
    #[serde(default)]
    pub conflicts: Vec<crate::neo4j::plan::FileConflict>,
    /// Feature graphs linked to the plan's project
    #[serde(default)]
    pub feature_graphs: Vec<FeatureGraphSummary>,
}

/// Get dependency graph for a plan (enriched with steps, notes, decisions counts + conflicts)
pub async fn get_plan_dependency_graph(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<DependencyGraphResponse>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let (tasks, edges) = neo4j.get_plan_dependency_graph(plan_id).await?;

    // Batch-fetch enrichment data (counts + step details) for all tasks
    let task_ids: Vec<String> = tasks.iter().map(|t| t.id.to_string()).collect();
    let enrichment = neo4j
        .get_task_enrichment_data(&task_ids)
        .await
        .unwrap_or_default();

    let nodes: Vec<DependencyGraphNode> = tasks
        .into_iter()
        .map(|t| {
            let tid = t.id.to_string();
            let data = enrichment.get(&tid).cloned().unwrap_or_default();
            DependencyGraphNode {
                id: t.id,
                title: t.title,
                description: t.description,
                status: format!("{:?}", t.status),
                priority: t.priority,
                tags: t.tags,
                affected_files: t.affected_files,
                assigned_to: t.assigned_to,
                acceptance_criteria: t.acceptance_criteria,
                step_count: data.counts.step_count,
                completed_step_count: data.counts.completed_step_count,
                note_count: data.counts.note_count,
                decision_count: data.counts.decision_count,
                steps: data
                    .steps
                    .into_iter()
                    .map(|s| DependencyGraphStep {
                        id: s.id,
                        order: s.order,
                        description: s.description,
                        status: s.status,
                        verification: s.verification,
                    })
                    .collect(),
                session_count: data.sessions.len(),
                active_session_count: data.sessions.iter().filter(|s| s.is_active).count(),
                child_session_count: data.sessions.iter().map(|s| s.child_count).sum(),
                discussed_files: data
                    .discussed_files
                    .into_iter()
                    .map(|f| DependencyGraphDiscussedFile {
                        file_path: f.file_path,
                        mention_count: f.mention_count,
                    })
                    .collect(),
            }
        })
        .collect();

    // Compute file conflicts (same logic as Wave view)
    let conflict_items: Vec<(uuid::Uuid, &[String])> = nodes
        .iter()
        .map(|n| (n.id, n.affected_files.as_slice()))
        .collect();
    let conflicts = compute_file_conflicts(&conflict_items);

    let edges: Vec<DependencyGraphEdge> = edges
        .into_iter()
        .map(|(from, to)| DependencyGraphEdge { from, to })
        .collect();

    // Fetch feature graphs for the plan's project
    let feature_graphs = if let Ok(Some(project_id)) = neo4j
        .get_plan(plan_id)
        .await
        .map(|p| p.and_then(|pl| pl.project_id))
    {
        let fgs = neo4j
            .list_feature_graphs(Some(project_id))
            .await
            .unwrap_or_default();
        fgs.into_iter()
            .map(|fg| {
                FeatureGraphSummary {
                    id: fg.id.to_string(),
                    name: fg.name,
                    entity_count: 0, // lightweight — don't count entities for each FG
                }
            })
            .collect()
    } else {
        vec![]
    };

    Ok(Json(DependencyGraphResponse {
        nodes,
        edges,
        conflicts,
        feature_graphs,
    }))
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

/// Get execution waves for a plan (topological sort + level grouping)
pub async fn get_plan_waves(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<WaveComputationResult>, AppError> {
    let result = state.orchestrator.neo4j().compute_waves(plan_id).await?;

    Ok(Json(result))
}

// ============================================================================
// Runner — Plan execution API
// ============================================================================

/// Request body for starting a plan run.
#[derive(Deserialize)]
pub struct RunPlanRequest {
    /// Working directory for the runner (where to execute commands)
    pub cwd: String,
    /// Optional project slug (for scoping MCP operations)
    pub project_slug: Option<String>,
    /// How this run was triggered (manual, chat, schedule, webhook, event).
    /// Defaults to "manual" if not specified.
    #[serde(default)]
    pub triggered_by: Option<String>,
    /// Optional budget limit in USD. Overrides the default ($10).
    /// When omitted, falls back to RunnerConfig::default().max_cost_usd.
    pub max_cost_usd: Option<f64>,
}

/// Response for a successfully started plan run.
#[derive(Serialize)]
pub struct RunPlanResponse {
    pub run_id: Uuid,
    pub plan_id: Uuid,
    pub total_waves: usize,
    pub total_tasks: usize,
}

/// POST /api/plans/:id/run — Start executing a plan.
///
/// Returns 202 Accepted with the run_id. Execution happens in background.
/// Returns 409 Conflict if the plan already has an active run.
pub async fn run_plan(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(req): Json<RunPlanRequest>,
) -> Result<(StatusCode, Json<RunPlanResponse>), AppError> {
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let graph = state.orchestrator.neo4j_arc();
    let context_builder = state.orchestrator.context_builder().clone();
    let mut config = state.orchestrator.runner_config();
    // Override budget if the caller specified one
    if let Some(budget) = req.max_cost_usd {
        config.max_cost_usd = budget;
    }

    // Create a broadcast channel for RunnerEvents
    let (event_tx, _) = tokio::sync::broadcast::channel(256);

    let mut runner = crate::runner::PlanRunner::new(
        chat_manager.clone(),
        graph,
        context_builder,
        config,
        event_tx,
    );

    // Bridge RunnerEvents to CrudEvent for WebSocket delivery
    runner =
        runner.with_event_emitter(state.event_bus.clone() as Arc<dyn crate::events::EventEmitter>);

    let runner = Arc::new(runner);

    // Parse trigger source from request (default: Manual)
    let trigger_source = match req.triggered_by.as_deref() {
        Some("chat") => crate::runner::TriggerSource::Chat { session_id: None },
        Some("schedule") => crate::runner::TriggerSource::Schedule {
            trigger_id: uuid::Uuid::nil(),
        },
        _ => crate::runner::TriggerSource::Manual,
    };

    let start_result = runner
        .start(plan_id, trigger_source, req.cwd, req.project_slug)
        .await
        .map_err(|e| {
            if e.to_string().contains("already has an active run") {
                AppError::Conflict(e.to_string())
            } else {
                AppError::Internal(e)
            }
        })?;

    Ok((
        StatusCode::ACCEPTED,
        Json(RunPlanResponse {
            run_id: start_result.run_id,
            plan_id: start_result.plan_id,
            total_waves: start_result.total_waves,
            total_tasks: start_result.total_tasks,
        }),
    ))
}

/// GET /api/plans/:id/run/status — Get current runner status.
pub async fn get_run_status(
    State(_state): State<OrchestratorState>,
    Path(_plan_id): Path<Uuid>,
) -> Result<Json<crate::runner::RunStatus>, AppError> {
    let status = crate::runner::PlanRunner::status().await;
    Ok(Json(status))
}

/// POST /api/plans/:id/run/cancel — Cancel an active plan run.
pub async fn cancel_run(
    State(_state): State<OrchestratorState>,
    Path(_plan_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Get the current run_id to cancel
    let status = crate::runner::PlanRunner::status().await;
    let run_id = status
        .run_id
        .ok_or_else(|| AppError::NotFound("No active run".to_string()))?;

    crate::runner::PlanRunner::cancel(run_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "cancelled": true,
        "run_id": run_id,
    })))
}

/// POST /api/plans/{plan_id}/run/force-cancel — Force-cancel a stuck run.
/// Unlike regular cancel (which sets a flag and waits for the execution loop to
/// check it), force-cancel immediately clears the global runner state and persists
/// the run as cancelled. Use when agents are stuck in "spawning" and never respond
/// to the graceful cancel flag.
pub async fn force_cancel_run(
    State(state): State<OrchestratorState>,
    Path(_plan_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let status = crate::runner::PlanRunner::status().await;
    let run_id = status
        .run_id
        .ok_or_else(|| AppError::NotFound("No active run".to_string()))?;

    crate::runner::PlanRunner::force_cancel(run_id, state.orchestrator.neo4j_arc())
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "force_cancelled": true,
        "run_id": run_id,
    })))
}

/// PATCH /api/plans/{plan_id}/run/budget — Update the budget of a running execution.
pub async fn update_run_budget(
    State(_state): State<OrchestratorState>,
    Path(_plan_id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, AppError> {
    let max_cost_usd = body
        .get("max_cost_usd")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| {
            AppError::BadRequest(
                "Missing or invalid 'max_cost_usd' (must be a positive number)".to_string(),
            )
        })?;

    if max_cost_usd <= 0.0 {
        return Err(AppError::BadRequest(
            "max_cost_usd must be positive".to_string(),
        ));
    }

    // Get the current run_id
    let status = crate::runner::PlanRunner::status().await;
    let run_id = status
        .run_id
        .ok_or_else(|| AppError::NotFound("No active run".to_string()))?;

    crate::runner::PlanRunner::update_budget(run_id, max_cost_usd)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "updated": true,
        "run_id": run_id,
        "max_cost_usd": max_cost_usd,
    })))
}

/// POST /api/plans/:id/run/auto-pr — Create a PR from a completed plan.
///
/// Verifies the plan is completed, collects commits, generates a PR body,
/// and creates the PR via `gh pr create`.
pub async fn create_auto_pr(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();

    // 1. Verify the plan is completed
    let plan = graph
        .get_plan(plan_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Plan {} not found", plan_id)))?;

    use crate::neo4j::models::PlanStatus;
    if plan.status != PlanStatus::Completed {
        return Err(AppError::BadRequest(format!(
            "Plan is not completed (status: {:?}). Cannot create PR.",
            plan.status
        )));
    }

    // 2. Get tasks for the plan
    let tasks = graph
        .get_plan_tasks(plan_id)
        .await
        .map_err(AppError::Internal)?;

    // 3. Get commits linked to the plan
    let commits = graph
        .get_plan_commits(plan_id)
        .await
        .map_err(AppError::Internal)?;

    // 4. Generate PR body
    let plan_title = &plan.title;
    let mut body = format!("## {}\n\n", plan_title);

    if !plan.description.is_empty() {
        body.push_str(&format!("{}\n\n", plan.description));
    }

    body.push_str("### Tasks\n\n");
    for task in &tasks {
        use crate::neo4j::models::TaskStatus;
        let status_emoji = match task.status {
            TaskStatus::Completed => "✅",
            TaskStatus::Failed => "❌",
            TaskStatus::Blocked => "🚫",
            _ => "⬜",
        };
        body.push_str(&format!(
            "- {} {}\n",
            status_emoji,
            task.title.as_deref().unwrap_or("Untitled")
        ));
    }

    if !commits.is_empty() {
        body.push_str("\n### Commits\n\n");
        for commit in &commits {
            body.push_str(&format!(
                "- `{}` {}\n",
                &commit.hash[..7.min(commit.hash.len())],
                commit.message
            ));
        }
    }

    body.push_str("\n---\n🤖 Generated by PlanRunner\n");

    // 5. Create PR via gh CLI
    let pr_title = format!("[PlanRunner] {}", &plan.title);
    let output = tokio::process::Command::new("gh")
        .args(["pr", "create", "--title", &pr_title, "--body", &body])
        .output()
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to run gh: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::Internal(anyhow::anyhow!(
            "gh pr create failed: {}",
            stderr
        )));
    }

    let pr_url = String::from_utf8_lossy(&output.stdout).trim().to_string();

    Ok(Json(serde_json::json!({
        "pr_url": pr_url,
        "plan_id": plan_id,
        "title": pr_title,
        "tasks_count": tasks.len(),
        "commits_count": commits.len(),
    })))
}

// ============================================================================
// Triggers
// ============================================================================

/// POST /api/plans/:id/triggers — Create a trigger for a plan.
pub async fn create_trigger(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::runner::{Trigger, TriggerType};

    let trigger_type_str = body
        .get("trigger_type")
        .and_then(|v| v.as_str())
        .unwrap_or("schedule");
    let trigger_type = match trigger_type_str {
        "schedule" => TriggerType::Schedule,
        "webhook" => TriggerType::Webhook,
        "event" => TriggerType::Event,
        "chat" => TriggerType::Chat,
        other => {
            return Err(AppError::BadRequest(format!(
                "Invalid trigger_type: {}. Must be schedule, webhook, event, or chat.",
                other
            )));
        }
    };
    let config = body.get("config").cloned().unwrap_or(serde_json::json!({}));
    let cooldown_secs = body
        .get("cooldown_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let trigger = Trigger {
        id: Uuid::new_v4(),
        plan_id,
        trigger_type,
        config,
        enabled: true,
        cooldown_secs,
        last_fired: None,
        fire_count: 0,
        created_at: chrono::Utc::now(),
    };

    let graph = state.orchestrator.neo4j_arc();
    let created = graph
        .create_trigger(&trigger)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_created(
        crate::events::EntityType::Trigger,
        &trigger.id.to_string(),
        serde_json::json!({"plan_id": plan_id, "trigger_type": trigger_type_str}),
        None,
    );

    Ok(Json(serde_json::to_value(&created).unwrap_or_default()))
}

/// GET /api/plans/:id/triggers — List triggers for a plan.
pub async fn list_triggers(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    let triggers = graph
        .list_triggers(plan_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(&triggers).unwrap_or_default()))
}

/// DELETE /api/triggers/:id — Delete a trigger.
pub async fn delete_trigger(
    State(state): State<OrchestratorState>,
    Path(trigger_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    graph
        .delete_trigger(trigger_id)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_deleted(
        crate::events::EntityType::Trigger,
        &trigger_id.to_string(),
        None,
    );

    Ok(Json(
        serde_json::json!({ "deleted": true, "trigger_id": trigger_id }),
    ))
}

/// PATCH /api/triggers/:id/enable — Enable a trigger.
pub async fn enable_trigger(
    State(state): State<OrchestratorState>,
    Path(trigger_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    let trigger = graph
        .update_trigger(trigger_id, Some(true), None, None)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Trigger {} not found", trigger_id)))?;

    state.event_bus.emit_updated(
        crate::events::EntityType::Trigger,
        &trigger_id.to_string(),
        serde_json::json!({"enabled": true}),
        None,
    );

    Ok(Json(serde_json::to_value(&trigger).unwrap_or_default()))
}

/// PATCH /api/triggers/:id/disable — Disable a trigger.
pub async fn disable_trigger(
    State(state): State<OrchestratorState>,
    Path(trigger_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    let trigger = graph
        .update_trigger(trigger_id, Some(false), None, None)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Trigger {} not found", trigger_id)))?;

    state.event_bus.emit_updated(
        crate::events::EntityType::Trigger,
        &trigger_id.to_string(),
        serde_json::json!({"enabled": false}),
        None,
    );

    Ok(Json(serde_json::to_value(&trigger).unwrap_or_default()))
}

/// GET /api/triggers/:id/firings — Get trigger firing history.
pub async fn list_trigger_firings(
    State(state): State<OrchestratorState>,
    Path(trigger_id): Path<Uuid>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let limit = params
        .get("limit")
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(20);
    let graph = state.orchestrator.neo4j_arc();
    let firings = graph
        .list_trigger_firings(trigger_id, limit)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(&firings).unwrap_or_default()))
}

// ============================================================================
// Webhooks
// ============================================================================

/// POST /api/webhooks/:trigger_id — Receive external webhook payload.
///
/// Validates HMAC-SHA256 signature (if configured), filters by event type
/// and branch pattern, then evaluates the trigger.
pub async fn receive_webhook(
    State(state): State<OrchestratorState>,
    Path(trigger_id): Path<Uuid>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();

    // 1. Load the trigger
    let trigger = graph
        .get_trigger(trigger_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Trigger {} not found", trigger_id)))?;

    if trigger.trigger_type != crate::runner::TriggerType::Webhook {
        return Err(AppError::BadRequest(format!(
            "Trigger {} is not a webhook trigger (type: {:?})",
            trigger_id, trigger.trigger_type
        )));
    }

    // 2. Validate HMAC signature if secret is configured
    if let Some(secret) = trigger.config.get("secret").and_then(|v| v.as_str()) {
        let signature = headers
            .get("x-hub-signature-256")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !crate::runner::providers::webhook::validate_github_signature(secret, &body, signature) {
            return Err(AppError::Unauthorized(
                "Invalid webhook signature".to_string(),
            ));
        }
    }

    // 3. Parse payload
    let payload: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON payload: {}", e)))?;

    // 4. Filter by event type
    if let Some(event_filter) = trigger
        .config
        .get("event_filter")
        .and_then(|v| v.as_array())
    {
        let github_event = headers
            .get("x-github-event")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let event_types: Vec<&str> = event_filter.iter().filter_map(|v| v.as_str()).collect();
        if !event_types.is_empty() && !event_types.contains(&github_event) {
            return Ok(Json(serde_json::json!({
                "status": "filtered",
                "reason": format!("Event '{}' not in filter {:?}", github_event, event_types)
            })));
        }
    }

    // 5. Filter by branch pattern
    if let Some(branch_pattern) = trigger
        .config
        .get("branch_pattern")
        .and_then(|v| v.as_str())
    {
        let branch = payload.get("ref").and_then(|v| v.as_str()).unwrap_or("");
        // Extract branch name from "refs/heads/main" format
        let branch_name = branch.strip_prefix("refs/heads/").unwrap_or(branch);
        if let Ok(re) = regex::Regex::new(branch_pattern) {
            if !re.is_match(branch_name) {
                return Ok(Json(serde_json::json!({
                    "status": "filtered",
                    "reason": format!("Branch '{}' does not match pattern '{}'", branch_name, branch_pattern)
                })));
            }
        }
    }

    // 6. Evaluate trigger
    let engine = crate::runner::TriggerEngine::new(graph.clone());
    match engine.evaluate_and_prepare(&trigger).await {
        Ok(Some(source)) => {
            engine
                .record_fire(&trigger, None, Some(payload))
                .await
                .map_err(AppError::Internal)?;
            Ok(Json(serde_json::json!({
                "status": "fired",
                "trigger_id": trigger_id,
                "source": format!("{:?}", source),
            })))
        }
        Ok(None) => Ok(Json(serde_json::json!({
            "status": "skipped",
            "reason": "Trigger guards not met (disabled, cooldown, or active run)"
        }))),
        Err(e) => Err(AppError::Internal(e)),
    }
}

// ============================================================================
// Plan Runs — List, Get, Compare, Predict
// ============================================================================

/// Enrich a list of RunnerState with `plan_title` from the associated Plan nodes.
/// Does a single batch query to Neo4j to resolve all unique plan_ids → titles.
async fn enrich_runs_with_plan_titles(
    graph: &std::sync::Arc<dyn crate::neo4j::traits::GraphStore>,
    runs: Vec<crate::runner::state::RunnerState>,
) -> serde_json::Value {
    use std::collections::HashMap;

    // Collect unique plan_ids
    let plan_ids: Vec<uuid::Uuid> = runs
        .iter()
        .map(|r| r.plan_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Batch fetch plan titles
    let mut title_map: HashMap<uuid::Uuid, String> = HashMap::new();
    for pid in &plan_ids {
        if let Ok(Some(plan)) = graph.get_plan(*pid).await {
            title_map.insert(*pid, plan.title);
        }
    }

    // Serialize runs and inject plan_title
    let mut result = Vec::with_capacity(runs.len());
    for run in &runs {
        let mut val = serde_json::to_value(run).unwrap_or_default();
        if let Some(obj) = val.as_object_mut() {
            let title = title_map.get(&run.plan_id).cloned().unwrap_or_default();
            obj.insert("plan_title".to_string(), serde_json::Value::String(title));
        }
        result.push(val);
    }

    serde_json::Value::Array(result)
}

/// GET /api/runs — List all plan runs across all plans.
/// Supports optional `workspace_slug` query param to scope runs to a workspace.
///
/// Each run in the response is enriched with a `plan_title` field
/// fetched from the associated Plan node, avoiding N+1 queries on the frontend.
pub async fn list_all_plan_runs(
    State(state): State<OrchestratorState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let limit = params
        .get("limit")
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(50);
    let offset = params
        .get("offset")
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(0);
    let status = params.get("status").map(|s| s.as_str());
    let workspace_slug = params.get("workspace_slug").map(|s| s.as_str());
    let graph = state.orchestrator.neo4j_arc();
    let runs = graph
        .list_all_plan_runs(limit, offset, status, workspace_slug)
        .await
        .map_err(AppError::Internal)?;

    // Enrich runs with plan titles (batch fetch to avoid N+1)
    let enriched = enrich_runs_with_plan_titles(&graph, runs).await;
    Ok(Json(enriched))
}

/// GET /api/plans/:plan_id/runs — List historical plan runs.
pub async fn list_plan_runs(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let limit = params
        .get("limit")
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(20);
    let graph = state.orchestrator.neo4j_arc();
    let runs = graph
        .list_plan_runs(plan_id, limit)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(serde_json::to_value(&runs).unwrap_or_default()))
}

/// GET /api/runs/:run_id/agent-executions — Get agent execution records for a plan run.
pub async fn get_run_agent_executions(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<Vec<crate::neo4j::agent_execution::AgentExecutionNode>>, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    let executions = graph
        .get_agent_executions_for_run(run_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(executions))
}

/// GET /api/runs/:run_id — Get a specific plan run.
pub async fn get_plan_run(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    let run = graph
        .get_plan_run(run_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Run {} not found", run_id)))?;
    Ok(Json(serde_json::to_value(&run).unwrap_or_default()))
}

/// POST /api/plans/:plan_id/runs/compare — Compare multiple plan runs.
///
/// Body: { "run_ids": ["uuid1", "uuid2", ...] }
/// Returns dimension-level comparison with trend analysis.
pub async fn compare_plan_runs(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, AppError> {
    let run_ids: Vec<Uuid> = body
        .get("run_ids")
        .and_then(|v| v.as_array())
        .ok_or_else(|| AppError::BadRequest("run_ids array required".to_string()))?
        .iter()
        .filter_map(|v| v.as_str().and_then(|s| s.parse().ok()))
        .collect();

    if run_ids.len() < 2 {
        return Err(AppError::BadRequest(
            "At least 2 run_ids required for comparison".to_string(),
        ));
    }

    let graph = state.orchestrator.neo4j_arc();

    // Load all runs and build vectors
    let mut vectors = Vec::new();
    for run_id in &run_ids {
        let run = graph
            .get_plan_run(*run_id)
            .await
            .map_err(AppError::Internal)?
            .ok_or_else(|| AppError::NotFound(format!("Run {} not found", run_id)))?;

        // Verify the run belongs to this plan
        if run.plan_id != plan_id {
            return Err(AppError::BadRequest(format!(
                "Run {} belongs to plan {}, not {}",
                run_id, run.plan_id, plan_id
            )));
        }

        vectors.push(crate::runner::vector::ExecutionVector::from_runner_state(
            &run,
        ));
    }

    let result = crate::runner::vector::compare_vectors(&vectors);
    Ok(Json(serde_json::to_value(&result).unwrap_or_default()))
}

/// POST /api/plans/:plan_id/runs/predict — Predict next run based on history.
///
/// Uses exponential weighted average on historical runs.
pub async fn predict_plan_run(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let graph = state.orchestrator.neo4j_arc();

    // Load all runs for this plan (most recent last)
    let runs = graph
        .list_plan_runs(plan_id, 50)
        .await
        .map_err(AppError::Internal)?;

    // list_plan_runs returns DESC order, reverse for chronological
    let vectors: Vec<_> = runs
        .iter()
        .rev()
        .map(crate::runner::vector::ExecutionVector::from_runner_state)
        .collect();

    let prediction = crate::runner::vector::predict_run(&vectors);
    Ok(Json(serde_json::to_value(&prediction).unwrap_or_default()))
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
        let (total, completed, in_progress, pending) = neo4j.get_milestone_progress(m.id).await?;
        let percentage = if total > 0 {
            (completed as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        milestones.push(RoadmapMilestone {
            milestone: m,
            tasks,
            progress: MilestoneProgressResponse {
                total,
                completed,
                in_progress,
                pending,
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
            tags: t.tags,
            affected_files: t.affected_files,
            assigned_to: t.assigned_to,
            acceptance_criteria: t.acceptance_criteria,
            step_count: 0,
            completed_step_count: 0,
            note_count: 0,
            decision_count: 0,
            steps: vec![],
            session_count: 0,
            active_session_count: 0,
            child_session_count: 0,
            discussed_files: vec![],
        })
        .collect();

    let edges: Vec<DependencyGraphEdge> = edges
        .into_iter()
        .map(|(from, to)| DependencyGraphEdge { from, to })
        .collect();

    let dependency_graph = DependencyGraphResponse {
        nodes,
        edges,
        conflicts: vec![],
        feature_graphs: vec![],
    };

    Ok(Json(RoadmapResponse {
        milestones,
        releases,
        progress,
        dependency_graph,
    }))
}

// ============================================================================
// Pipeline — Gate Results & Progress Score
// ============================================================================

/// GET /api/runs/:run_id/gates — Return gate results for a pipeline run.
///
/// Returns an array of `GateResult` records. Currently returns the empty array
/// when no gates have been persisted yet (pipeline V1 placeholder).
pub async fn get_run_gates(
    State(_state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    // V1: return empty gate results. The pipeline runner will persist these
    // in a future iteration and we'll query them from Neo4j here.
    let response = serde_json::json!({
        "run_id": run_id.to_string(),
        "gates": [],
    });
    Ok(Json(response))
}

/// GET /api/runs/:run_id/progress — Return progress score for a pipeline run.
///
/// Returns a `ProgressScore` with default values when the pipeline has not
/// recorded any checkpoints yet (V1 placeholder).
pub async fn get_run_progress(
    State(_state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    // V1: return a default progress summary. Real data will come from
    // ProgressOracle checkpoints once the pipeline runner persists them.
    let response = serde_json::json!({
        "run_id": run_id.to_string(),
        "score": 0.0,
        "delta": null,
        "dimensions": {
            "build": 0.0,
            "tests": 0.0,
            "coverage": 0.0,
            "steps": 0.0,
        },
        "trend": "unknown",
        "total_checkpoints": 0,
        "best_score": 0.0,
        "worst_score": 0.0,
    });
    Ok(Json(response))
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
    // GET /api/milestones/{id} — enriched milestone detail
    // ================================================================

    use crate::api::routes::create_router;
    use crate::neo4j::models::MilestoneStatus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{
        mock_app_state, test_bearer_token, test_milestone, test_plan, test_project, test_step,
        test_task_titled,
    };
    use axum::body::Body;
    use axum::http::{Request, StatusCode as HttpStatus};
    use std::sync::Arc;
    use tower::ServiceExt;

    /// Create an authenticated GET request
    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    /// Build a test router pre-seeded with a project milestone + plan + tasks + steps
    async fn test_app_with_project_milestone() -> (axum::Router, uuid::Uuid, uuid::Uuid, uuid::Uuid)
    {
        let app_state = mock_app_state();

        // Create a project
        let project = test_project();
        app_state.neo4j.create_project(&project).await.unwrap();

        // Create a project milestone
        let milestone = test_milestone(project.id, "Q1 Release");
        app_state.neo4j.create_milestone(&milestone).await.unwrap();

        // Create a plan linked to project
        let plan = test_plan();
        app_state.neo4j.create_plan(&plan).await.unwrap();

        // Create two tasks and link them to the milestone
        let task1 = test_task_titled("Backend API");
        let task2 = test_task_titled("Frontend UI");
        app_state.neo4j.create_task(plan.id, &task1).await.unwrap();
        app_state.neo4j.create_task(plan.id, &task2).await.unwrap();
        app_state
            .neo4j
            .add_task_to_milestone(milestone.id, task1.id)
            .await
            .unwrap();
        app_state
            .neo4j
            .add_task_to_milestone(milestone.id, task2.id)
            .await
            .unwrap();

        // Add steps to task1
        let step1 = test_step(0, "Create endpoint");
        let step2 = test_step(1, "Add tests");
        app_state.neo4j.create_step(task1.id, &step1).await.unwrap();
        app_state.neo4j.create_step(task1.id, &step2).await.unwrap();

        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        });
        (create_router(state), milestone.id, task1.id, task2.id)
    }

    #[tokio::test]
    async fn test_get_milestone_returns_enriched_response() {
        let (app, milestone_id, task1_id, task2_id) = test_app_with_project_milestone().await;
        let uri = format!("/api/milestones/{}", milestone_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Milestone field
        assert_eq!(json["milestone"]["id"], milestone_id.to_string());
        assert_eq!(json["milestone"]["title"], "Q1 Release");

        // Plans array — tasks are grouped under their plan
        let plans = json["plans"].as_array().unwrap();
        assert_eq!(plans.len(), 1);
        assert!(plans[0]["id"].is_string());
        assert!(plans[0]["title"].is_string());

        // Each plan has its tasks
        let plan_tasks = plans[0]["tasks"].as_array().unwrap();
        assert_eq!(plan_tasks.len(), 2);
        let task_ids: Vec<String> = plan_tasks
            .iter()
            .map(|t| t["id"].as_str().unwrap().to_string())
            .collect();
        assert!(task_ids.contains(&task1_id.to_string()));
        assert!(task_ids.contains(&task2_id.to_string()));

        // Steps are included in task1
        let t1 = plan_tasks
            .iter()
            .find(|t| t["id"].as_str().unwrap() == task1_id.to_string())
            .unwrap();
        let steps = t1["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 2);
        assert!(steps[0]["description"].is_string());

        // Flat tasks array for backward compat
        let flat_tasks = json["tasks"].as_array().unwrap();
        assert_eq!(flat_tasks.len(), 2);

        // Progress
        assert_eq!(json["progress"]["total"], 2);
        assert_eq!(json["progress"]["completed"], 0);
        assert_eq!(json["progress"]["in_progress"], 0);
        assert_eq!(json["progress"]["pending"], 2);
        assert_eq!(json["progress"]["percentage"], 0.0);
    }

    #[tokio::test]
    async fn test_get_milestone_not_found() {
        let (app, _, _, _) = test_app_with_project_milestone().await;
        let uri = format!("/api/milestones/{}", uuid::Uuid::new_v4());
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_milestone_invalid_uuid() {
        let (app, _, _, _) = test_app_with_project_milestone().await;
        let resp = app
            .oneshot(auth_get("/api/milestones/not-a-uuid"))
            .await
            .unwrap();

        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_milestone_progress_endpoint() {
        let (app, milestone_id, _, _) = test_app_with_project_milestone().await;
        let uri = format!("/api/milestones/{}/progress", milestone_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["total"], 2);
        assert_eq!(json["completed"], 0);
        assert_eq!(json["in_progress"], 0);
        assert_eq!(json["pending"], 2);
        assert_eq!(json["percentage"], 0.0);
    }

    #[tokio::test]
    async fn test_get_milestone_progress_empty_milestone() {
        let (app, _, _, _) = test_app_with_project_milestone().await;
        // Non-existent milestone returns 0/0 progress (not an error)
        let uri = format!("/api/milestones/{}/progress", uuid::Uuid::new_v4());
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["total"], 0);
        assert_eq!(json["percentage"], 0.0);
    }

    #[test]
    fn test_milestone_details_response_serialization() {
        let milestone = MilestoneNode {
            id: uuid::Uuid::new_v4(),
            project_id: uuid::Uuid::new_v4(),
            title: "Test".to_string(),
            description: Some("Description".to_string()),
            status: MilestoneStatus::Open,
            target_date: None,
            closed_at: None,
            created_at: chrono::Utc::now(),
        };
        let response = MilestoneDetailsResponse {
            milestone,
            plans: vec![],
            tasks: vec![],
            progress: MilestoneProgressResponse {
                total: 5,
                completed: 3,
                in_progress: 1,
                pending: 1,
                percentage: 60.0,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["progress"]["total"], 5);
        assert_eq!(json["progress"]["completed"], 3);
        assert_eq!(json["progress"]["in_progress"], 1);
        assert_eq!(json["progress"]["pending"], 1);
        assert_eq!(json["progress"]["percentage"], 60.0);
        assert!(json["plans"].is_array());
        assert!(json["tasks"].is_array());
    }

    #[test]
    fn test_milestone_progress_response_fields() {
        let response = MilestoneProgressResponse {
            total: 10,
            completed: 4,
            in_progress: 3,
            pending: 3,
            percentage: 40.0,
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["total"], 10);
        assert_eq!(json["completed"], 4);
        assert_eq!(json["in_progress"], 3);
        assert_eq!(json["pending"], 3);
        assert_eq!(json["percentage"], 40.0);
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
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
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
            additional_origins: vec![],
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

    // ================================================================
    // Knowledge Fabric — Handler Tests
    // ================================================================

    /// Build a simple test router (no seeded data)
    async fn test_app() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        });
        create_router(state)
    }

    /// Create an authenticated POST request with JSON body
    fn auth_post_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    /// Create an authenticated DELETE request
    fn auth_delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    /// Parse response body as JSON
    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ----------------------------------------------------------------
    // Decision semantic search
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_search_decisions_semantic_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/decisions/search-semantic?query=authentication",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_search_decisions_semantic_with_limit() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/decisions/search-semantic?query=test&limit=5",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }

    // ----------------------------------------------------------------
    // Decisions affecting
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_decisions_affecting_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/decisions/affecting?entity_type=File&entity_id=src/main.rs",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_decisions_affecting_with_status_filter() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/decisions/affecting?entity_type=File&entity_id=src/main.rs&status=accepted",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }

    // ----------------------------------------------------------------
    // Decision affects CRUD
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_decision_affects() {
        let app = test_app().await;
        let decision_id = uuid::Uuid::new_v4();
        let uri = format!("/api/decisions/{}/affects", decision_id);
        let body = serde_json::json!({
            "entity_type": "File",
            "entity_id": "src/lib.rs",
            "impact_description": "Changes authentication flow"
        });
        let resp = app.oneshot(auth_post_json(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_add_decision_affects_without_description() {
        let app = test_app().await;
        let decision_id = uuid::Uuid::new_v4();
        let uri = format!("/api/decisions/{}/affects", decision_id);
        let body = serde_json::json!({
            "entity_type": "Function",
            "entity_id": "handle_request"
        });
        let resp = app.oneshot(auth_post_json(&uri, body)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_list_decision_affects_empty() {
        let app = test_app().await;
        let decision_id = uuid::Uuid::new_v4();
        let uri = format!("/api/decisions/{}/affects", decision_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_remove_decision_affects_path() {
        let app = test_app().await;
        let decision_id = uuid::Uuid::new_v4();
        let uri = format!("/api/decisions/{}/affects/Function/my_func", decision_id);
        let resp = app.oneshot(auth_delete(&uri)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_remove_decision_affects_query_params() {
        let app = test_app().await;
        let decision_id = uuid::Uuid::new_v4();
        let uri = format!(
            "/api/decisions/{}/affects?entity_type=File&entity_id=src%2Flib.rs",
            decision_id
        );
        let resp = app.oneshot(auth_delete(&uri)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Decision supersede
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_supersede_decision() {
        let app = test_app().await;
        let new_id = uuid::Uuid::new_v4();
        let old_id = uuid::Uuid::new_v4();
        let uri = format!("/api/decisions/{}/supersedes/{}", new_id, old_id);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&uri)
                    .header("authorization", test_bearer_token())
                    .header("content-type", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Decision timeline
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_decision_timeline_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/decisions/timeline"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_decision_timeline_with_task_filter() {
        let app = test_app().await;
        let task_id = uuid::Uuid::new_v4();
        let uri = format!("/api/decisions/timeline?task_id={}", task_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }

    #[tokio::test]
    async fn test_get_decision_timeline_with_date_range() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/decisions/timeline?from=2024-01-01&to=2025-01-01",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }

    // ----------------------------------------------------------------
    // Commit files (TOUCHES)
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_commit_files_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/commits/abc123/files"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    // ----------------------------------------------------------------
    // File history
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_file_history_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/files/history?path=%2Fhome%2Fuser%2Fproject%2Fsrc%2Fmain.rs",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_file_history_with_limit() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/files/history?path=src%2Flib.rs&limit=5"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }

    #[tokio::test]
    async fn test_get_file_history_missing_path() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/files/history")).await.unwrap();
        // path is a required query param
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // Co-change graph
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_co_change_graph_empty() {
        let app = test_app().await;
        let project_id = uuid::Uuid::new_v4();
        let uri = format!("/api/projects/{}/co-changes", project_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_co_change_graph_with_params() {
        let app = test_app().await;
        let project_id = uuid::Uuid::new_v4();
        let uri = format!(
            "/api/projects/{}/co-changes?min_count=5&limit=50",
            project_id
        );
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }

    // ----------------------------------------------------------------
    // File co-changers
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_file_co_changers_empty() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/files/co-changers?path=src%2Flib.rs"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_file_co_changers_with_params() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/files/co-changers?path=src%2Flib.rs&min_count=2&limit=20",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
    }

    #[tokio::test]
    async fn test_get_file_co_changers_missing_path() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/files/co-changers"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // Backfill decision embeddings
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_backfill_decision_embeddings() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/admin/backfill-decision-embeddings")
                    .header("authorization", test_bearer_token())
                    .header("content-type", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json["decisions_processed"].is_number());
        assert!(json["embeddings_created"].is_number());
    }

    // ----------------------------------------------------------------
    // Backfill decision project slugs
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_backfill_decision_project_slugs() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/admin/backfill-decision-project-slugs")
                    .header("authorization", test_bearer_token())
                    .header("content-type", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json["decisions_processed"].is_number());
        assert!(json["decisions_updated"].is_number());
    }

    // ----------------------------------------------------------------
    // Backfill discussed
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_backfill_discussed() {
        let app = test_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/admin/backfill-discussed")
                    .header("authorization", test_bearer_token())
                    .header("content-type", "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["sessions_processed"], 0);
        assert_eq!(json["entities_found"], 0);
        assert_eq!(json["relations_created"], 0);
    }

    // ----------------------------------------------------------------
    // Request/Response serialization tests
    // ----------------------------------------------------------------

    #[test]
    fn test_search_decisions_semantic_query_deserialize() {
        let json = r#"{"query":"auth flow","limit":5,"project_id":"e83b0663-9600-450d-9f63-234e857394df"}"#;
        let q: SearchDecisionsSemanticQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.query, "auth flow");
        assert_eq!(q.limit, Some(5));
        assert_eq!(
            q.project_id,
            Some("e83b0663-9600-450d-9f63-234e857394df".to_string())
        );
    }

    #[test]
    fn test_search_decisions_semantic_query_minimal() {
        let json = r#"{"query":"test"}"#;
        let q: SearchDecisionsSemanticQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.query, "test");
        assert_eq!(q.limit, None);
        assert_eq!(q.project_id, None);
    }

    #[test]
    fn test_decisions_affecting_query_deserialize() {
        let json = r#"{"entity_type":"File","entity_id":"src/main.rs","status":"accepted"}"#;
        let q: DecisionsAffectingQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.entity_type, "File");
        assert_eq!(q.entity_id, "src/main.rs");
        assert_eq!(q.status, Some("accepted".to_string()));
    }

    #[test]
    fn test_decisions_affecting_query_no_status() {
        let json = r#"{"entity_type":"Function","entity_id":"handle_request"}"#;
        let q: DecisionsAffectingQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.entity_type, "Function");
        assert_eq!(q.status, None);
    }

    #[test]
    fn test_add_affects_request_deserialize() {
        let json = r#"{"entity_type":"File","entity_id":"src/lib.rs","impact_description":"Modifies auth"}"#;
        let r: AddAffectsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(r.entity_type, "File");
        assert_eq!(r.entity_id, "src/lib.rs");
        assert_eq!(r.impact_description, Some("Modifies auth".to_string()));
    }

    #[test]
    fn test_add_affects_request_no_description() {
        let json = r#"{"entity_type":"Function","entity_id":"foo"}"#;
        let r: AddAffectsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(r.impact_description, None);
    }

    #[test]
    fn test_remove_affects_query_deserialize() {
        let json = r#"{"entity_type":"File","entity_id":"src/lib.rs"}"#;
        let q: RemoveAffectsQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.entity_type, "File");
        assert_eq!(q.entity_id, "src/lib.rs");
    }

    #[test]
    fn test_decision_timeline_query_all_fields() {
        let json = r#"{"task_id":"e83b0663-9600-450d-9f63-234e857394df","from":"2024-01-01","to":"2025-01-01"}"#;
        let q: DecisionTimelineQuery = serde_json::from_str(json).unwrap();
        assert!(q.task_id.is_some());
        assert_eq!(q.from, Some("2024-01-01".to_string()));
        assert_eq!(q.to, Some("2025-01-01".to_string()));
    }

    #[test]
    fn test_decision_timeline_query_empty() {
        let json = r#"{}"#;
        let q: DecisionTimelineQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.task_id, None);
        assert_eq!(q.from, None);
        assert_eq!(q.to, None);
    }

    #[test]
    fn test_file_history_query_deserialize() {
        let json = r#"{"path":"src/main.rs","limit":10}"#;
        let q: FileHistoryQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.path, "src/main.rs");
        assert_eq!(q.limit, Some(10));
    }

    #[test]
    fn test_file_history_query_no_limit() {
        let json = r#"{"path":"src/main.rs"}"#;
        let q: FileHistoryQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.path, "src/main.rs");
        assert_eq!(q.limit, None);
    }

    #[test]
    fn test_co_change_graph_query_defaults() {
        let json = r#"{}"#;
        let q: CoChangeGraphQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.min_count, None);
        assert_eq!(q.limit, None);
    }

    #[test]
    fn test_co_change_graph_query_with_values() {
        let json = r#"{"min_count":5,"limit":50}"#;
        let q: CoChangeGraphQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.min_count, Some(5));
        assert_eq!(q.limit, Some(50));
    }

    #[test]
    fn test_file_co_changers_query_full() {
        let json = r#"{"path":"src/lib.rs","min_count":2,"limit":20}"#;
        let q: FileCoChangersQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.path, "src/lib.rs");
        assert_eq!(q.min_count, Some(2));
        assert_eq!(q.limit, Some(20));
    }

    #[test]
    fn test_file_co_changers_query_minimal() {
        let json = r#"{"path":"src/lib.rs"}"#;
        let q: FileCoChangersQuery = serde_json::from_str(json).unwrap();
        assert_eq!(q.path, "src/lib.rs");
        assert_eq!(q.min_count, None);
        assert_eq!(q.limit, None);
    }

    #[test]
    fn test_fabric_project_request_deserialize() {
        let json = r#"{"project_id":"e83b0663-9600-450d-9f63-234e857394df"}"#;
        let r: FabricProjectRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            r.project_id.to_string(),
            "e83b0663-9600-450d-9f63-234e857394df"
        );
    }

    // ================================================================
    // Dependency graph — enriched structs serialization
    // ================================================================

    #[test]
    fn test_dependency_graph_step_serialization() {
        let step = DependencyGraphStep {
            id: "step-1".to_string(),
            order: 0,
            description: "Create endpoint".to_string(),
            status: "Pending".to_string(),
            verification: Some("cargo test".to_string()),
        };
        let json = serde_json::to_value(&step).unwrap();
        assert_eq!(json["id"], "step-1");
        assert_eq!(json["order"], 0);
        assert_eq!(json["description"], "Create endpoint");
        assert_eq!(json["status"], "Pending");
        assert_eq!(json["verification"], "cargo test");
    }

    #[test]
    fn test_dependency_graph_step_no_verification() {
        let step = DependencyGraphStep {
            id: "step-2".to_string(),
            order: 1,
            description: "Review code".to_string(),
            status: "Completed".to_string(),
            verification: None,
        };
        let json = serde_json::to_value(&step).unwrap();
        assert!(json.get("verification").is_none()); // skip_serializing_if
    }

    #[test]
    fn test_dependency_graph_session_serialization() {
        let session = DependencyGraphSession {
            session_id: "sess-1".to_string(),
            title: Some("Planning session".to_string()),
            is_active: true,
            child_count: 3,
        };
        let json = serde_json::to_value(&session).unwrap();
        assert_eq!(json["session_id"], "sess-1");
        assert_eq!(json["title"], "Planning session");
        assert_eq!(json["is_active"], true);
        assert_eq!(json["child_count"], 3);
    }

    #[test]
    fn test_dependency_graph_session_no_title() {
        let session = DependencyGraphSession {
            session_id: "sess-2".to_string(),
            title: None,
            is_active: false,
            child_count: 0,
        };
        let json = serde_json::to_value(&session).unwrap();
        assert!(json.get("title").is_none()); // skip_serializing_if
    }

    #[test]
    fn test_dependency_graph_discussed_file_serialization() {
        let f = DependencyGraphDiscussedFile {
            file_path: "src/main.rs".to_string(),
            mention_count: 5,
        };
        let json = serde_json::to_value(&f).unwrap();
        assert_eq!(json["file_path"], "src/main.rs");
        assert_eq!(json["mention_count"], 5);
    }

    #[test]
    fn test_feature_graph_summary_serialization() {
        let fg = FeatureGraphSummary {
            id: "fg-1".to_string(),
            name: "Auth flow".to_string(),
            entity_count: 12,
        };
        let json = serde_json::to_value(&fg).unwrap();
        assert_eq!(json["id"], "fg-1");
        assert_eq!(json["name"], "Auth flow");
        assert_eq!(json["entity_count"], 12);
    }

    #[test]
    fn test_dependency_graph_node_enriched_serialization() {
        let node = DependencyGraphNode {
            id: uuid::Uuid::new_v4(),
            title: Some("Implement API".to_string()),
            description: "Build the REST endpoint".to_string(),
            status: "Pending".to_string(),
            priority: Some(80),
            tags: vec!["api".to_string(), "backend".to_string()],
            affected_files: vec!["src/api.rs".to_string()],
            assigned_to: Some("agent-1".to_string()),
            acceptance_criteria: vec!["Tests pass".to_string()],
            step_count: 3,
            completed_step_count: 1,
            note_count: 2,
            decision_count: 1,
            steps: vec![DependencyGraphStep {
                id: "s1".to_string(),
                order: 0,
                description: "Step 1".to_string(),
                status: "Completed".to_string(),
                verification: None,
            }],
            session_count: 1,
            active_session_count: 0,
            child_session_count: 2,
            discussed_files: vec![DependencyGraphDiscussedFile {
                file_path: "src/api.rs".to_string(),
                mention_count: 3,
            }],
        };
        let json = serde_json::to_value(&node).unwrap();
        assert_eq!(json["step_count"], 3);
        assert_eq!(json["completed_step_count"], 1);
        assert_eq!(json["note_count"], 2);
        assert_eq!(json["decision_count"], 1);
        assert_eq!(json["session_count"], 1);
        assert_eq!(json["active_session_count"], 0);
        assert_eq!(json["child_session_count"], 2);
        assert_eq!(json["steps"].as_array().unwrap().len(), 1);
        assert_eq!(json["discussed_files"].as_array().unwrap().len(), 1);
        assert_eq!(json["tags"].as_array().unwrap().len(), 2);
        assert_eq!(json["affected_files"].as_array().unwrap().len(), 1);
        assert_eq!(json["assigned_to"], "agent-1");
        assert_eq!(json["acceptance_criteria"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_dependency_graph_response_serialization() {
        let resp = DependencyGraphResponse {
            nodes: vec![],
            edges: vec![],
            conflicts: vec![],
            feature_graphs: vec![FeatureGraphSummary {
                id: "fg-1".to_string(),
                name: "Auth".to_string(),
                entity_count: 0,
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert!(json["nodes"].is_array());
        assert!(json["edges"].is_array());
        assert!(json["conflicts"].is_array());
        assert_eq!(json["feature_graphs"].as_array().unwrap().len(), 1);
    }

    // ================================================================
    // GET /api/plans/{id}/dependency-graph — enriched handler test
    // ================================================================

    /// Build a test router with a plan containing tasks with steps and dependencies
    async fn test_app_with_dependency_graph() -> (axum::Router, uuid::Uuid, uuid::Uuid, uuid::Uuid)
    {
        let app_state = mock_app_state();

        // Create plan
        let plan = test_plan();
        app_state.neo4j.create_plan(&plan).await.unwrap();

        // Create two tasks with affected_files (for conflict detection)
        let mut task1 = test_task_titled("Backend API");
        task1.affected_files = vec!["src/api.rs".to_string(), "src/shared.rs".to_string()];
        let mut task2 = test_task_titled("Frontend UI");
        task2.affected_files = vec!["src/ui.rs".to_string(), "src/shared.rs".to_string()];
        app_state.neo4j.create_task(plan.id, &task1).await.unwrap();
        app_state.neo4j.create_task(plan.id, &task2).await.unwrap();

        // Add dependency: task2 depends on task1
        app_state
            .neo4j
            .add_task_dependency(task2.id, task1.id)
            .await
            .unwrap();

        // Add steps to task1
        let step1 = test_step(0, "Create endpoint");
        let step2 = test_step(1, "Add tests");
        app_state.neo4j.create_step(task1.id, &step1).await.unwrap();
        app_state.neo4j.create_step(task1.id, &step2).await.unwrap();

        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        });
        (create_router(state), plan.id, task1.id, task2.id)
    }

    #[tokio::test]
    async fn test_get_dependency_graph_enriched() {
        let (app, plan_id, task1_id, task2_id) = test_app_with_dependency_graph().await;
        let uri = format!("/api/plans/{}/dependency-graph", plan_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Nodes
        let nodes = json["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 2);

        // Find task1 node — should have enriched step data
        let t1 = nodes
            .iter()
            .find(|n| n["id"].as_str().unwrap() == task1_id.to_string())
            .unwrap();
        assert_eq!(t1["step_count"], 2);
        assert_eq!(t1["completed_step_count"], 0); // none completed yet
        assert!(t1["steps"].is_array());
        assert_eq!(t1["steps"].as_array().unwrap().len(), 2);

        // Step details
        let step_descs: Vec<&str> = t1["steps"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["description"].as_str())
            .collect();
        assert!(step_descs.contains(&"Create endpoint"));
        assert!(step_descs.contains(&"Add tests"));

        // task2 has no steps
        let t2 = nodes
            .iter()
            .find(|n| n["id"].as_str().unwrap() == task2_id.to_string())
            .unwrap();
        assert_eq!(t2["step_count"], 0);
        assert_eq!(t2["steps"].as_array().unwrap().len(), 0);

        // Enrichment fields present
        assert!(t1["tags"].is_array());
        assert!(t1["affected_files"].is_array());
        assert!(t1["acceptance_criteria"].is_array());
        assert!(t1["note_count"].is_number());
        assert!(t1["decision_count"].is_number());
        assert!(t1["session_count"].is_number());
        assert!(t1["discussed_files"].is_array());

        // Edges — task2 depends on task1
        let edges = json["edges"].as_array().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0]["from"].as_str().unwrap(), task2_id.to_string());
        assert_eq!(edges[0]["to"].as_str().unwrap(), task1_id.to_string());

        // Conflicts — src/shared.rs is in both tasks
        let conflicts = json["conflicts"].as_array().unwrap();
        assert!(
            !conflicts.is_empty(),
            "Should detect file conflict on src/shared.rs"
        );

        // Feature graphs — empty (no project linked)
        assert!(json["feature_graphs"].is_array());
    }

    #[tokio::test]
    async fn test_get_dependency_graph_empty_plan() {
        let app_state = mock_app_state();
        let plan = test_plan();
        app_state.neo4j.create_plan(&plan).await.unwrap();

        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
        });
        let app = create_router(state);

        let uri = format!("/api/plans/{}/dependency-graph", plan.id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["nodes"].as_array().unwrap().len(), 0);
        assert_eq!(json["edges"].as_array().unwrap().len(), 0);
        assert_eq!(json["conflicts"].as_array().unwrap().len(), 0);
        assert_eq!(json["feature_graphs"].as_array().unwrap().len(), 0);
    }

    // ================================================================
    // GET /api/runs — list_all_plan_runs
    // ================================================================

    #[tokio::test]
    async fn test_list_all_plan_runs_empty() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/runs")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_list_all_plan_runs_with_limit() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/runs?limit=10")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn test_list_all_plan_runs_with_offset() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/runs?offset=5")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn test_list_all_plan_runs_with_limit_and_offset() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/runs?limit=10&offset=5"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn test_list_all_plan_runs_with_status_filter() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/runs?status=running"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_list_all_plan_runs_with_all_params() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/runs?limit=5&offset=0&status=completed"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn test_list_all_plan_runs_invalid_limit_uses_default() {
        let app = test_app().await;
        // Non-numeric limit should be ignored, falling back to default (50)
        let resp = app.oneshot(auth_get("/api/runs?limit=abc")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    #[tokio::test]
    async fn test_list_all_plan_runs_invalid_offset_uses_default() {
        let app = test_app().await;
        // Non-numeric offset should be ignored, falling back to default (0)
        let resp = app.oneshot(auth_get("/api/runs?offset=xyz")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert!(json.is_array());
    }

    // ================================================================
    // GET /api/reactor/status — reactor_status
    // ================================================================

    #[tokio::test]
    async fn test_reactor_status_not_initialized() {
        // test_app() creates a ServerState with an empty OnceLock (reactor not set)
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/reactor/status")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["running"], false);
        assert_eq!(json["error"], "reactor not initialized");
    }

    #[tokio::test]
    async fn test_reactor_status_with_counters() {
        // Build a state where reactor_counters is populated
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let reactor_counters = std::sync::OnceLock::new();
        let counters = Arc::new(crate::events::ReactorCounters::default());
        // Mark running
        counters
            .running
            .store(true, std::sync::atomic::Ordering::Relaxed);
        // Simulate some events
        counters
            .events_received
            .store(42, std::sync::atomic::Ordering::Relaxed);
        counters
            .events_matched
            .store(10, std::sync::atomic::Ordering::Relaxed);
        counters
            .handlers_invoked
            .store(8, std::sync::atomic::Ordering::Relaxed);
        counters
            .handler_errors
            .store(1, std::sync::atomic::Ordering::Relaxed);
        let _ = reactor_counters.set(counters);

        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters,
        });
        let app = create_router(state);

        let resp = app.oneshot(auth_get("/api/reactor/status")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["running"], true);
        assert_eq!(json["events_received"], 42);
        assert_eq!(json["events_matched"], 10);
        assert_eq!(json["handlers_invoked"], 8);
        assert_eq!(json["handler_errors"], 1);
    }

    #[tokio::test]
    async fn test_reactor_status_with_zero_counters() {
        // Reactor initialized but no events processed yet
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let reactor_counters = std::sync::OnceLock::new();
        let counters = Arc::new(crate::events::ReactorCounters::default());
        let _ = reactor_counters.set(counters);

        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                crate::events::EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(crate::test_helpers::test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters,
        });
        let app = create_router(state);

        let resp = app.oneshot(auth_get("/api/reactor/status")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["running"], false);
        assert_eq!(json["events_received"], 0);
        assert_eq!(json["events_matched"], 0);
        assert_eq!(json["handlers_invoked"], 0);
        assert_eq!(json["handler_errors"], 0);
        // Should NOT have the error field when reactor is initialized
        assert!(json.get("error").is_none());
    }
}
