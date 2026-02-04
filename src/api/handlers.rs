//! API request handlers

use crate::neo4j::models::{
    CommitNode, ConstraintNode, DecisionNode, PlanNode, PlanStatus, StepNode, TaskNode,
};
use crate::orchestrator::{FileWatcher, Orchestrator};
use crate::plan::models::*;
use axum::{
    extract::{Path, State},
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
// Plans
// ============================================================================

/// List all active plans
pub async fn list_plans(
    State(state): State<OrchestratorState>,
) -> Result<Json<Vec<PlanNode>>, AppError> {
    let plans = state
        .orchestrator
        .plan_manager()
        .list_active_plans()
        .await?;
    Ok(Json(plans))
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
        .neo4j()
        .link_plan_to_project(plan_id, req.project_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Unlink a plan from its project
pub async fn unlink_plan_from_project(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .unlink_plan_from_project(plan_id)
        .await?;
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

/// Delete a constraint
pub async fn delete_constraint(
    State(state): State<OrchestratorState>,
    Path(constraint_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .delete_constraint(constraint_id)
        .await?;
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

    state.orchestrator.neo4j().create_commit(&commit).await?;
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
        .neo4j()
        .link_commit_to_task(&req.commit_hash, task_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get commits for a task
pub async fn get_task_commits(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<Vec<CommitNode>>, AppError> {
    let commits = state
        .orchestrator
        .neo4j()
        .get_task_commits(task_id)
        .await?;
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
        .neo4j()
        .link_commit_to_plan(&req.commit_hash, plan_id)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Get commits for a plan
pub async fn get_plan_commits(
    State(state): State<OrchestratorState>,
    Path(plan_id): Path<Uuid>,
) -> Result<Json<Vec<CommitNode>>, AppError> {
    let commits = state
        .orchestrator
        .neo4j()
        .get_plan_commits(plan_id)
        .await?;
    Ok(Json(commits))
}

// ============================================================================
// Error handling
// ============================================================================

/// Application error type
pub enum AppError {
    Internal(anyhow::Error),
    NotFound(String),
    BadRequest(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            AppError::Internal(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
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
