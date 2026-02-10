//! API request handlers

use crate::api::{
    PaginatedResponse, PaginationParams, PriorityFilter, SearchFilter, StatusFilter, TagsFilter,
};
use crate::chat::ChatManager;
use crate::events::{EventBus, EventEmitter};
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
    pub event_bus: Arc<EventBus>,
    /// Auth config — None means deny-by-default
    pub auth_config: Option<AuthConfig>,
    /// Whether to serve the frontend static files (SPA fallback)
    pub serve_frontend: bool,
    /// Path to the frontend dist/ directory
    pub frontend_path: String,
}

/// Shared orchestrator state
pub type OrchestratorState = Arc<ServerState>;

// ============================================================================
// Health check
// ============================================================================

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Health check handler
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
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
}

/// Sync response
#[derive(Serialize)]
pub struct SyncResponse {
    pub files_synced: usize,
    pub files_skipped: usize,
    pub errors: usize,
}

/// Sync a directory to the knowledge base
pub async fn sync_directory(
    State(state): State<OrchestratorState>,
    Json(req): Json<SyncRequest>,
) -> Result<Json<SyncResponse>, AppError> {
    let path = std::path::Path::new(&req.path);
    let result = state.orchestrator.sync_directory(path).await?;
    Ok(Json(SyncResponse {
        files_synced: result.files_synced,
        files_skipped: result.files_skipped,
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
// Internal Events (MCP → HTTP bridge)
// ============================================================================

/// POST /internal/events — Receive a CrudEvent from the MCP server and broadcast it
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
    let path = std::path::Path::new(&req.path);
    let mut watcher = state.watcher.write().await;

    watcher.watch(path).await?;
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
}
