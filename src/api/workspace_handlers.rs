//! Workspace API handlers

use crate::api::{PaginatedResponse, PaginationParams, StatusFilter};
use crate::neo4j::models::*;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::handlers::{AppError, OrchestratorState};

// ============================================================================
// Request/Response types - Workspace
// ============================================================================

#[derive(Deserialize)]
pub struct CreateWorkspaceRequest {
    pub name: String,
    pub slug: Option<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Deserialize)]
pub struct UpdateWorkspaceRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct WorkspaceResponse {
    pub id: String,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub created_at: String,
    pub updated_at: Option<String>,
    pub metadata: serde_json::Value,
}

impl From<WorkspaceNode> for WorkspaceResponse {
    fn from(w: WorkspaceNode) -> Self {
        Self {
            id: w.id.to_string(),
            name: w.name,
            slug: w.slug,
            description: w.description,
            created_at: w.created_at.to_rfc3339(),
            updated_at: w.updated_at.map(|dt| dt.to_rfc3339()),
            metadata: w.metadata,
        }
    }
}

#[derive(Serialize)]
pub struct WorkspaceOverviewResponse {
    pub workspace: WorkspaceResponse,
    pub projects: Vec<ProjectSummary>,
    pub milestones: Vec<WorkspaceMilestoneResponse>,
    pub resources: Vec<ResourceResponse>,
    pub components: Vec<ComponentResponse>,
}

#[derive(Serialize)]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub slug: String,
}

// ============================================================================
// Request/Response types - Workspace Milestone
// ============================================================================

#[derive(Deserialize)]
pub struct CreateWorkspaceMilestoneRequest {
    pub title: String,
    pub description: Option<String>,
    pub target_date: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Deserialize)]
pub struct UpdateWorkspaceMilestoneRequest {
    pub title: Option<String>,
    pub description: Option<String>,
    pub status: Option<String>,
    pub target_date: Option<String>,
}

#[derive(Serialize)]
pub struct WorkspaceMilestoneResponse {
    pub id: String,
    pub workspace_id: String,
    pub title: String,
    pub description: Option<String>,
    pub status: String,
    pub target_date: Option<String>,
    pub closed_at: Option<String>,
    pub created_at: String,
    pub tags: Vec<String>,
}

impl From<WorkspaceMilestoneNode> for WorkspaceMilestoneResponse {
    fn from(m: WorkspaceMilestoneNode) -> Self {
        Self {
            id: m.id.to_string(),
            workspace_id: m.workspace_id.to_string(),
            title: m.title,
            description: m.description,
            status: serde_json::to_value(&m.status)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
            target_date: m.target_date.map(|dt| dt.to_rfc3339()),
            closed_at: m.closed_at.map(|dt| dt.to_rfc3339()),
            created_at: m.created_at.to_rfc3339(),
            tags: m.tags,
        }
    }
}

#[derive(Serialize)]
pub struct MilestoneProgressResponse {
    pub total: u32,
    pub completed: u32,
    pub in_progress: u32,
    pub pending: u32,
    pub percentage: f64,
}

/// Full milestone detail: milestone → plans → tasks → steps + progress
#[derive(Serialize)]
pub struct MilestoneDetailResponse {
    #[serde(flatten)]
    pub milestone: WorkspaceMilestoneResponse,
    pub plans: Vec<MilestonePlanSummary>,
    pub progress: MilestoneProgressResponse,
}

/// A plan summary within a milestone detail
#[derive(Serialize)]
pub struct MilestonePlanSummary {
    pub id: String,
    pub title: String,
    pub status: Option<String>,
    pub tasks: Vec<MilestoneTaskSummary>,
}

/// A task summary within a milestone plan
#[derive(Serialize)]
pub struct MilestoneTaskSummary {
    pub id: String,
    pub title: Option<String>,
    pub description: String,
    pub status: String,
    pub priority: Option<i32>,
    pub tags: Vec<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub steps: Vec<MilestoneStepSummary>,
}

/// A step summary within a milestone task
#[derive(Serialize)]
pub struct MilestoneStepSummary {
    pub id: String,
    pub order: u32,
    pub description: String,
    pub status: String,
    pub verification: Option<String>,
}

// ============================================================================
// Request/Response types - Resource
// ============================================================================

#[derive(Deserialize)]
pub struct CreateResourceRequest {
    pub name: String,
    pub resource_type: String,
    pub file_path: String,
    pub url: Option<String>,
    pub format: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Serialize)]
pub struct ResourceResponse {
    pub id: String,
    pub workspace_id: Option<String>,
    pub project_id: Option<String>,
    pub name: String,
    pub resource_type: String,
    pub file_path: String,
    pub url: Option<String>,
    pub format: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub created_at: String,
    pub metadata: serde_json::Value,
}

impl From<ResourceNode> for ResourceResponse {
    fn from(r: ResourceNode) -> Self {
        Self {
            id: r.id.to_string(),
            workspace_id: r.workspace_id.map(|id| id.to_string()),
            project_id: r.project_id.map(|id| id.to_string()),
            name: r.name,
            resource_type: format!("{:?}", r.resource_type),
            file_path: r.file_path,
            url: r.url,
            format: r.format,
            version: r.version,
            description: r.description,
            created_at: r.created_at.to_rfc3339(),
            metadata: r.metadata,
        }
    }
}

#[derive(Deserialize)]
pub struct LinkResourceRequest {
    pub project_id: String,
    pub relation: String, // "implements" or "uses"
}

// ============================================================================
// Request/Response types - Component
// ============================================================================

#[derive(Deserialize)]
pub struct CreateComponentRequest {
    pub name: String,
    pub component_type: String,
    pub description: Option<String>,
    pub runtime: Option<String>,
    #[serde(default)]
    pub config: serde_json::Value,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Serialize)]
pub struct ComponentResponse {
    pub id: String,
    pub workspace_id: String,
    pub name: String,
    pub component_type: String,
    pub description: Option<String>,
    pub runtime: Option<String>,
    pub config: serde_json::Value,
    pub created_at: String,
    pub tags: Vec<String>,
}

impl From<ComponentNode> for ComponentResponse {
    fn from(c: ComponentNode) -> Self {
        Self {
            id: c.id.to_string(),
            workspace_id: c.workspace_id.to_string(),
            name: c.name,
            component_type: format!("{:?}", c.component_type),
            description: c.description,
            runtime: c.runtime,
            config: c.config,
            created_at: c.created_at.to_rfc3339(),
            tags: c.tags,
        }
    }
}

#[derive(Deserialize)]
pub struct AddDependencyRequest {
    pub depends_on_id: String,
    pub protocol: Option<String>,
    #[serde(default = "default_true")]
    pub required: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
pub struct MapToProjectRequest {
    pub project_id: String,
}

#[derive(Serialize)]
pub struct TopologyResponse {
    pub components: Vec<TopologyComponent>,
}

#[derive(Serialize)]
pub struct TopologyComponent {
    pub component: ComponentResponse,
    pub project_name: Option<String>,
    pub dependencies: Vec<TopologyDependency>,
}

#[derive(Serialize)]
pub struct TopologyDependency {
    pub to_id: String,
    pub protocol: Option<String>,
    pub required: bool,
}

// ============================================================================
// Workspace Handlers
// ============================================================================

/// List all workspaces
pub async fn list_workspaces(
    State(state): State<OrchestratorState>,
    Query(query): Query<PaginationParams>,
) -> Result<Json<PaginatedResponse<WorkspaceResponse>>, AppError> {
    let workspaces = state.orchestrator.neo4j().list_workspaces().await?;

    let total = workspaces.len() as i64;
    let limit = query.validated_limit();
    let offset = query.offset;

    let items: Vec<WorkspaceResponse> = workspaces
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(WorkspaceResponse::from)
        .collect();

    Ok(Json(PaginatedResponse {
        items,
        total: total as usize,
        limit,
        offset,
        has_more: (offset + limit) < total as usize,
    }))
}

/// Create a new workspace
pub async fn create_workspace(
    State(state): State<OrchestratorState>,
    Json(req): Json<CreateWorkspaceRequest>,
) -> Result<(StatusCode, Json<WorkspaceResponse>), AppError> {
    let slug = req.slug.unwrap_or_else(|| {
        req.name
            .to_lowercase()
            .replace(' ', "-")
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-')
            .collect()
    });

    let workspace = WorkspaceNode {
        id: Uuid::new_v4(),
        name: req.name,
        slug,
        description: req.description,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: req.metadata,
    };

    state.orchestrator.create_workspace(&workspace).await?;

    Ok((
        StatusCode::CREATED,
        Json(WorkspaceResponse::from(workspace)),
    ))
}

/// Get workspace by slug
pub async fn get_workspace(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<WorkspaceResponse>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    Ok(Json(WorkspaceResponse::from(workspace)))
}

/// Update workspace
pub async fn update_workspace(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(req): Json<UpdateWorkspaceRequest>,
) -> Result<Json<WorkspaceResponse>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    state
        .orchestrator
        .update_workspace(workspace.id, req.name, req.description, req.metadata)
        .await?;

    let updated = state
        .orchestrator
        .neo4j()
        .get_workspace(workspace.id)
        .await?
        .unwrap();

    Ok(Json(WorkspaceResponse::from(updated)))
}

/// Delete workspace
pub async fn delete_workspace(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<StatusCode, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    state.orchestrator.delete_workspace(workspace.id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Get workspace overview
pub async fn get_workspace_overview(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<WorkspaceOverviewResponse>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let projects = state
        .orchestrator
        .neo4j()
        .list_workspace_projects(workspace.id)
        .await?;

    let milestones = state
        .orchestrator
        .neo4j()
        .list_workspace_milestones(workspace.id)
        .await?;

    let resources = state
        .orchestrator
        .neo4j()
        .list_workspace_resources(workspace.id)
        .await?;

    let components = state
        .orchestrator
        .neo4j()
        .list_components(workspace.id)
        .await?;

    Ok(Json(WorkspaceOverviewResponse {
        workspace: WorkspaceResponse::from(workspace),
        projects: projects
            .into_iter()
            .map(|p| ProjectSummary {
                id: p.id.to_string(),
                name: p.name,
                slug: p.slug,
            })
            .collect(),
        milestones: milestones
            .into_iter()
            .map(WorkspaceMilestoneResponse::from)
            .collect(),
        resources: resources.into_iter().map(ResourceResponse::from).collect(),
        components: components
            .into_iter()
            .map(ComponentResponse::from)
            .collect(),
    }))
}

// ============================================================================
// Workspace Project Handlers
// ============================================================================

#[derive(Deserialize)]
pub struct AddProjectRequest {
    pub project_id: String,
}

/// List projects in workspace
pub async fn list_workspace_projects(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<ProjectNode>>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let projects = state
        .orchestrator
        .neo4j()
        .list_workspace_projects(workspace.id)
        .await?;

    Ok(Json(projects))
}

/// Add project to workspace
pub async fn add_project_to_workspace(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(req): Json<AddProjectRequest>,
) -> Result<StatusCode, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let project_id: Uuid = req
        .project_id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid project ID".to_string()))?;

    state
        .orchestrator
        .add_project_to_workspace(workspace.id, project_id)
        .await?;

    Ok(StatusCode::CREATED)
}

/// Remove project from workspace
pub async fn remove_project_from_workspace(
    State(state): State<OrchestratorState>,
    Path((slug, project_id)): Path<(String, String)>,
) -> Result<StatusCode, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let project_id: Uuid = project_id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid project ID".to_string()))?;

    state
        .orchestrator
        .remove_project_from_workspace(workspace.id, project_id)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Workspace Milestone Handlers
// ============================================================================

/// Query parameters for listing workspace milestones
#[derive(Debug, Deserialize, Default)]
pub struct WorkspaceMilestonesListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub status_filter: StatusFilter,
}

/// List workspace milestones with pagination and status filter
pub async fn list_workspace_milestones(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Query(query): Query<WorkspaceMilestonesListQuery>,
) -> Result<Json<PaginatedResponse<WorkspaceMilestoneResponse>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let status_str = query.status_filter.status.as_deref();

    let (milestones, total) = state
        .orchestrator
        .neo4j()
        .list_workspace_milestones_filtered(
            workspace.id,
            status_str,
            query.pagination.validated_limit(),
            query.pagination.offset,
        )
        .await?;

    let items: Vec<WorkspaceMilestoneResponse> = milestones
        .into_iter()
        .map(WorkspaceMilestoneResponse::from)
        .collect();

    Ok(Json(PaginatedResponse::new(
        items,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

/// Create workspace milestone
pub async fn create_workspace_milestone(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(req): Json<CreateWorkspaceMilestoneRequest>,
) -> Result<(StatusCode, Json<WorkspaceMilestoneResponse>), AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let target_date = req
        .target_date
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    let milestone = WorkspaceMilestoneNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        title: req.title,
        description: req.description,
        status: MilestoneStatus::Open,
        target_date,
        closed_at: None,
        created_at: chrono::Utc::now(),
        tags: req.tags,
    };

    state
        .orchestrator
        .create_workspace_milestone(&milestone)
        .await?;

    Ok((
        StatusCode::CREATED,
        Json(WorkspaceMilestoneResponse::from(milestone)),
    ))
}

/// Get workspace milestone with full detail: plans → tasks → steps + progress
pub async fn get_workspace_milestone(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<MilestoneDetailResponse>, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid milestone ID".to_string()))?;

    let neo4j = state.orchestrator.neo4j();

    // 1. Get milestone node
    let milestone = neo4j
        .get_workspace_milestone(id)
        .await?
        .ok_or_else(|| AppError::NotFound("Milestone not found".to_string()))?;

    // 2. Get tasks with plan info (plan_id, plan_title, plan_status)
    let tasks_with_plan = neo4j.get_workspace_milestone_tasks(id).await?;

    // 3. Get progress stats
    let (total, completed, in_progress, pending) =
        neo4j.get_workspace_milestone_progress(id).await?;

    let percentage = if total > 0 {
        (completed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    // 4. Get all steps in one batch query
    let mut steps_map = neo4j.get_workspace_milestone_steps(id).await?;

    // 5. Group tasks by plan and build hierarchical response
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

    Ok(Json(MilestoneDetailResponse {
        milestone: WorkspaceMilestoneResponse::from(milestone),
        plans,
        progress: MilestoneProgressResponse {
            total,
            completed,
            in_progress,
            pending,
            percentage,
        },
    }))
}

/// Update workspace milestone
pub async fn update_workspace_milestone(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateWorkspaceMilestoneRequest>,
) -> Result<Json<WorkspaceMilestoneResponse>, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid milestone ID".to_string()))?;

    let status = req
        .status
        .map(|s| match s.to_lowercase().as_str() {
            "planned" => Ok(MilestoneStatus::Planned),
            "open" => Ok(MilestoneStatus::Open),
            "in_progress" => Ok(MilestoneStatus::InProgress),
            "completed" => Ok(MilestoneStatus::Completed),
            "closed" => Ok(MilestoneStatus::Closed),
            other => Err(AppError::BadRequest(format!(
                "Invalid milestone status '{}'. Valid: planned, open, in_progress, completed, closed",
                other
            ))),
        })
        .transpose()?;

    let target_date = req
        .target_date
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    state
        .orchestrator
        .update_workspace_milestone(id, req.title, req.description, status, target_date)
        .await?;

    let updated = state
        .orchestrator
        .neo4j()
        .get_workspace_milestone(id)
        .await?
        .unwrap();

    Ok(Json(WorkspaceMilestoneResponse::from(updated)))
}

/// Delete workspace milestone
pub async fn delete_workspace_milestone(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid milestone ID".to_string()))?;

    state.orchestrator.delete_workspace_milestone(id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Add task to workspace milestone
#[derive(Deserialize)]
pub struct AddTaskRequest {
    pub task_id: String,
}

pub async fn add_task_to_workspace_milestone(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
    Json(req): Json<AddTaskRequest>,
) -> Result<StatusCode, AppError> {
    let milestone_id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid milestone ID".to_string()))?;

    let task_id: Uuid = req
        .task_id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid task ID".to_string()))?;

    state
        .orchestrator
        .add_task_to_workspace_milestone(milestone_id, task_id)
        .await?;

    Ok(StatusCode::CREATED)
}

/// List tasks linked to a workspace milestone (with plan info)
pub async fn list_workspace_milestone_tasks(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<Vec<TaskWithPlan>>, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid milestone ID".to_string()))?;

    let tasks = state
        .orchestrator
        .neo4j()
        .get_workspace_milestone_tasks(id)
        .await?;

    Ok(Json(tasks))
}

/// Get workspace milestone progress
pub async fn get_workspace_milestone_progress(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<MilestoneProgressResponse>, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid milestone ID".to_string()))?;

    let (total, completed, in_progress, pending) = state
        .orchestrator
        .neo4j()
        .get_workspace_milestone_progress(id)
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
// Global Workspace Milestones
// ============================================================================

/// Response for workspace milestone with workspace info
#[derive(Serialize)]
pub struct WorkspaceMilestoneWithWorkspace {
    #[serde(flatten)]
    pub milestone: WorkspaceMilestoneResponse,
    pub workspace_id: String,
    pub workspace_name: String,
    pub workspace_slug: String,
}

/// Query params for global workspace milestones listing
#[derive(Debug, Deserialize, Default)]
pub struct AllWorkspaceMilestonesQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    #[serde(flatten)]
    pub status_filter: StatusFilter,
    pub workspace_id: Option<String>,
}

/// List all workspace milestones across all workspaces
pub async fn list_all_workspace_milestones(
    State(state): State<OrchestratorState>,
    Query(query): Query<AllWorkspaceMilestonesQuery>,
) -> Result<Json<PaginatedResponse<WorkspaceMilestoneWithWorkspace>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let workspace_id = query
        .workspace_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| {
            uuid::Uuid::parse_str(s)
                .map_err(|_| AppError::BadRequest("Invalid workspace_id UUID".to_string()))
        })
        .transpose()?;

    let status_str = query.status_filter.status.as_deref();

    let total = state
        .orchestrator
        .neo4j()
        .count_all_workspace_milestones(workspace_id, status_str)
        .await?;

    let results = state
        .orchestrator
        .neo4j()
        .list_all_workspace_milestones_filtered(
            workspace_id,
            status_str,
            query.pagination.validated_limit(),
            query.pagination.offset,
        )
        .await?;

    let items: Vec<WorkspaceMilestoneWithWorkspace> = results
        .into_iter()
        .map(|(m, wid, wname, wslug)| WorkspaceMilestoneWithWorkspace {
            milestone: WorkspaceMilestoneResponse::from(m),
            workspace_id: wid,
            workspace_name: wname,
            workspace_slug: wslug,
        })
        .collect();

    Ok(Json(PaginatedResponse::new(
        items,
        total,
        query.pagination.validated_limit(),
        query.pagination.offset,
    )))
}

// ============================================================================
// Resource Handlers
// ============================================================================

/// List workspace resources
pub async fn list_resources(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<ResourceResponse>>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let resources = state
        .orchestrator
        .neo4j()
        .list_workspace_resources(workspace.id)
        .await?;

    Ok(Json(
        resources.into_iter().map(ResourceResponse::from).collect(),
    ))
}

/// Create resource
pub async fn create_resource(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(req): Json<CreateResourceRequest>,
) -> Result<(StatusCode, Json<ResourceResponse>), AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let resource_type = match req.resource_type.to_lowercase().as_str() {
        "apicontract" | "api_contract" => ResourceType::ApiContract,
        "protobuf" => ResourceType::Protobuf,
        "graphqlschema" | "graphql_schema" => ResourceType::GraphqlSchema,
        "jsonschema" | "json_schema" => ResourceType::JsonSchema,
        "databaseschema" | "database_schema" => ResourceType::DatabaseSchema,
        "sharedtypes" | "shared_types" => ResourceType::SharedTypes,
        "config" => ResourceType::Config,
        "documentation" => ResourceType::Documentation,
        _ => ResourceType::Other,
    };

    let resource = ResourceNode {
        id: Uuid::new_v4(),
        workspace_id: Some(workspace.id),
        project_id: None,
        name: req.name,
        resource_type,
        file_path: req.file_path,
        url: req.url,
        format: req.format,
        version: req.version,
        description: req.description,
        created_at: chrono::Utc::now(),
        updated_at: None,
        metadata: req.metadata,
    };

    state.orchestrator.create_resource(&resource).await?;

    Ok((StatusCode::CREATED, Json(ResourceResponse::from(resource))))
}

/// Get resource
pub async fn get_resource(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<ResourceResponse>, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid resource ID".to_string()))?;

    let resource = state
        .orchestrator
        .neo4j()
        .get_resource(id)
        .await?
        .ok_or_else(|| AppError::NotFound("Resource not found".to_string()))?;

    Ok(Json(ResourceResponse::from(resource)))
}

/// Request to update a resource
#[derive(Deserialize)]
pub struct UpdateResourceRequest {
    pub name: Option<String>,
    pub file_path: Option<String>,
    pub url: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
}

/// Update resource
pub async fn update_resource(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateResourceRequest>,
) -> Result<StatusCode, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid resource ID".to_string()))?;

    state
        .orchestrator
        .update_resource(
            id,
            req.name,
            req.file_path,
            req.url,
            req.version,
            req.description,
        )
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Delete resource
pub async fn delete_resource(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid resource ID".to_string()))?;

    state.orchestrator.delete_resource(id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Link resource to project
pub async fn link_resource_to_project(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
    Json(req): Json<LinkResourceRequest>,
) -> Result<StatusCode, AppError> {
    let resource_id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid resource ID".to_string()))?;

    let project_id: Uuid = req
        .project_id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid project ID".to_string()))?;

    match req.relation.to_lowercase().as_str() {
        "implements" => {
            state
                .orchestrator
                .link_project_implements_resource(project_id, resource_id)
                .await?;
        }
        "uses" => {
            state
                .orchestrator
                .link_project_uses_resource(project_id, resource_id)
                .await?;
        }
        _ => {
            return Err(AppError::BadRequest(
                "Invalid relation type. Use 'implements' or 'uses'".to_string(),
            ));
        }
    }

    Ok(StatusCode::CREATED)
}

// ============================================================================
// Component Handlers
// ============================================================================

/// List components
pub async fn list_components(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<ComponentResponse>>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let components = state
        .orchestrator
        .neo4j()
        .list_components(workspace.id)
        .await?;

    Ok(Json(
        components
            .into_iter()
            .map(ComponentResponse::from)
            .collect(),
    ))
}

/// Create component
pub async fn create_component(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(req): Json<CreateComponentRequest>,
) -> Result<(StatusCode, Json<ComponentResponse>), AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let component_type = match req.component_type.to_lowercase().as_str() {
        "service" => ComponentType::Service,
        "frontend" => ComponentType::Frontend,
        "worker" => ComponentType::Worker,
        "database" => ComponentType::Database,
        "messagequeue" | "message_queue" => ComponentType::MessageQueue,
        "cache" => ComponentType::Cache,
        "gateway" => ComponentType::Gateway,
        "external" => ComponentType::External,
        _ => ComponentType::Other,
    };

    let component = ComponentNode {
        id: Uuid::new_v4(),
        workspace_id: workspace.id,
        name: req.name,
        component_type,
        description: req.description,
        runtime: req.runtime,
        config: req.config,
        created_at: chrono::Utc::now(),
        tags: req.tags,
    };

    state.orchestrator.create_component(&component).await?;

    Ok((
        StatusCode::CREATED,
        Json(ComponentResponse::from(component)),
    ))
}

/// Get component
pub async fn get_component(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<ComponentResponse>, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid component ID".to_string()))?;

    let component = state
        .orchestrator
        .neo4j()
        .get_component(id)
        .await?
        .ok_or_else(|| AppError::NotFound("Component not found".to_string()))?;

    Ok(Json(ComponentResponse::from(component)))
}

/// Request to update a component
#[derive(Deserialize)]
pub struct UpdateComponentRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub runtime: Option<String>,
    pub config: Option<serde_json::Value>,
    pub tags: Option<Vec<String>>,
}

/// Update component
pub async fn update_component(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateComponentRequest>,
) -> Result<StatusCode, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid component ID".to_string()))?;

    state
        .orchestrator
        .update_component(
            id,
            req.name,
            req.description,
            req.runtime,
            req.config,
            req.tags,
        )
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Delete component
pub async fn delete_component(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid component ID".to_string()))?;

    state.orchestrator.delete_component(id).await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Add component dependency
pub async fn add_component_dependency(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
    Json(req): Json<AddDependencyRequest>,
) -> Result<StatusCode, AppError> {
    let component_id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid component ID".to_string()))?;

    let depends_on_id: Uuid = req
        .depends_on_id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid depends_on_id".to_string()))?;

    state
        .orchestrator
        .add_component_dependency(component_id, depends_on_id, req.protocol, req.required)
        .await?;

    Ok(StatusCode::CREATED)
}

/// Remove component dependency
pub async fn remove_component_dependency(
    State(state): State<OrchestratorState>,
    Path((id, dep_id)): Path<(String, String)>,
) -> Result<StatusCode, AppError> {
    let component_id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid component ID".to_string()))?;

    let depends_on_id: Uuid = dep_id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid dependency ID".to_string()))?;

    state
        .orchestrator
        .remove_component_dependency(component_id, depends_on_id)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Map component to project
pub async fn map_component_to_project(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
    Json(req): Json<MapToProjectRequest>,
) -> Result<StatusCode, AppError> {
    let component_id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid component ID".to_string()))?;

    let project_id: Uuid = req
        .project_id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid project ID".to_string()))?;

    state
        .orchestrator
        .map_component_to_project(component_id, project_id)
        .await?;

    Ok(StatusCode::OK)
}

/// Get workspace topology
pub async fn get_workspace_topology(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<TopologyResponse>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let topology = state
        .orchestrator
        .neo4j()
        .get_workspace_topology(workspace.id)
        .await?;

    let components: Vec<TopologyComponent> = topology
        .into_iter()
        .map(|(component, project_name, deps)| TopologyComponent {
            component: ComponentResponse::from(component),
            project_name,
            dependencies: deps
                .into_iter()
                .map(|d| TopologyDependency {
                    to_id: d.to_id.to_string(),
                    protocol: d.protocol,
                    required: d.required,
                })
                .collect(),
        })
        .collect();

    Ok(Json(TopologyResponse { components }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::neo4j::models::{MilestoneStatus, WorkspaceMilestoneNode};
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{
        mock_app_state, test_bearer_token, test_plan, test_task_titled, test_workspace,
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

    /// Build a test router with mock backends
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
        });
        create_router(state)
    }

    /// Build a test router pre-seeded with workspace milestone + tasks
    async fn test_app_with_milestone_tasks() -> (axum::Router, Uuid, Uuid, Uuid) {
        let app_state = mock_app_state();

        // Create workspace
        let ws = test_workspace();
        app_state.neo4j.create_workspace(&ws).await.unwrap();

        // Create workspace milestone
        let milestone_id = Uuid::new_v4();
        let milestone = WorkspaceMilestoneNode {
            id: milestone_id,
            workspace_id: ws.id,
            title: "Test Milestone".to_string(),
            description: Some("Milestone for testing".to_string()),
            status: MilestoneStatus::Open,
            target_date: None,
            closed_at: None,
            created_at: chrono::Utc::now(),
            tags: vec!["test".to_string()],
        };
        app_state
            .neo4j
            .create_workspace_milestone(&milestone)
            .await
            .unwrap();

        // Create a plan (needed to create tasks)
        let plan = test_plan();
        app_state.neo4j.create_plan(&plan).await.unwrap();

        // Create two tasks and link them to the milestone
        let task1 = test_task_titled("Task Alpha");
        let task2 = test_task_titled("Task Beta");
        app_state.neo4j.create_task(plan.id, &task1).await.unwrap();
        app_state.neo4j.create_task(plan.id, &task2).await.unwrap();
        app_state
            .neo4j
            .add_task_to_workspace_milestone(milestone_id, task1.id)
            .await
            .unwrap();
        app_state
            .neo4j
            .add_task_to_workspace_milestone(milestone_id, task2.id)
            .await
            .unwrap();

        // Create a third task NOT linked to the milestone
        let task3 = test_task_titled("Task Gamma (not linked)");
        app_state.neo4j.create_task(plan.id, &task3).await.unwrap();

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
        });
        (create_router(state), milestone_id, task1.id, task2.id)
    }

    // ====================================================================
    // GET /api/workspace-milestones/{id}/tasks
    // ====================================================================

    #[tokio::test]
    async fn test_list_workspace_milestone_tasks_returns_linked_tasks() {
        let (app, milestone_id, task1_id, task2_id) = test_app_with_milestone_tasks().await;
        let uri = format!("/api/workspace-milestones/{}/tasks", milestone_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let tasks: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

        // Should return exactly 2 linked tasks, not the unlinked one
        assert_eq!(tasks.len(), 2);
        let ids: Vec<String> = tasks
            .iter()
            .map(|t| t["id"].as_str().unwrap().to_string())
            .collect();
        assert!(ids.contains(&task1_id.to_string()));
        assert!(ids.contains(&task2_id.to_string()));
    }

    #[tokio::test]
    async fn test_list_workspace_milestone_tasks_empty_milestone() {
        let app = test_app().await;
        // Use a random UUID for a milestone with no tasks
        let milestone_id = Uuid::new_v4();
        let uri = format!("/api/workspace-milestones/{}/tasks", milestone_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let tasks: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn test_list_workspace_milestone_tasks_invalid_id() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/workspace-milestones/not-a-uuid/tasks"))
            .await
            .unwrap();

        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_list_workspace_milestone_tasks_response_contains_task_fields() {
        let (app, milestone_id, _, _) = test_app_with_milestone_tasks().await;
        let uri = format!("/api/workspace-milestones/{}/tasks", milestone_id);
        let resp = app.oneshot(auth_get(&uri)).await.unwrap();

        assert_eq!(resp.status(), HttpStatus::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let tasks: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();

        // Verify task structure includes expected fields (now returns TaskWithPlan)
        for task in &tasks {
            assert!(task["id"].is_string());
            assert!(task["description"].is_string());
            assert!(task["status"].is_string());
            assert!(task.get("created_at").is_some());
            // TaskWithPlan adds plan_id and plan_title
            assert!(task["plan_id"].is_string());
            assert!(task["plan_title"].is_string());
        }
    }

    // ====================================================================
    // Existing serialization tests
    // ====================================================================

    #[test]
    fn test_update_resource_request_all_fields() {
        let json = r#"{"name":"API v2","file_path":"/api.yaml","url":"https://x.com","version":"2.0","description":"Updated"}"#;
        let req: UpdateResourceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, Some("API v2".to_string()));
        assert_eq!(req.file_path, Some("/api.yaml".to_string()));
        assert_eq!(req.url, Some("https://x.com".to_string()));
        assert_eq!(req.version, Some("2.0".to_string()));
        assert_eq!(req.description, Some("Updated".to_string()));
    }

    #[test]
    fn test_update_resource_request_empty() {
        let json = r#"{}"#;
        let req: UpdateResourceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, None);
        assert_eq!(req.file_path, None);
        assert_eq!(req.url, None);
        assert_eq!(req.version, None);
        assert_eq!(req.description, None);
    }

    #[test]
    fn test_update_resource_request_partial() {
        let json = r#"{"version":"3.0"}"#;
        let req: UpdateResourceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.version, Some("3.0".to_string()));
        assert_eq!(req.name, None);
    }

    #[test]
    fn test_update_component_request_all_fields() {
        let json = r#"{"name":"Auth","description":"Auth service","runtime":"rust","config":{"port":8080},"tags":["auth","core"]}"#;
        let req: UpdateComponentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, Some("Auth".to_string()));
        assert_eq!(req.description, Some("Auth service".to_string()));
        assert_eq!(req.runtime, Some("rust".to_string()));
        assert!(req.config.is_some());
        assert_eq!(req.config.as_ref().unwrap()["port"], 8080);
        assert_eq!(req.tags, Some(vec!["auth".to_string(), "core".to_string()]));
    }

    #[test]
    fn test_update_component_request_empty() {
        let json = r#"{}"#;
        let req: UpdateComponentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, None);
        assert_eq!(req.description, None);
        assert_eq!(req.runtime, None);
        assert_eq!(req.config, None);
        assert_eq!(req.tags, None);
    }

    #[test]
    fn test_update_component_request_with_json_config() {
        let json = r#"{"config":{"workers":4,"timeout":30,"nested":{"key":"val"}}}"#;
        let req: UpdateComponentRequest = serde_json::from_str(json).unwrap();
        let config = req.config.unwrap();
        assert_eq!(config["workers"], 4);
        assert_eq!(config["timeout"], 30);
        assert_eq!(config["nested"]["key"], "val");
    }

    #[test]
    fn test_update_component_request_empty_tags() {
        let json = r#"{"tags":[]}"#;
        let req: UpdateComponentRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.tags, Some(vec![]));
    }

    #[test]
    fn test_workspace_milestones_list_query_defaults() {
        let json = r#"{}"#;
        let query: WorkspaceMilestonesListQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.pagination.limit, 50);
        assert_eq!(query.pagination.offset, 0);
        assert!(query.status_filter.status.is_none());
    }

    #[test]
    fn test_workspace_milestones_list_query_with_status() {
        let json = r#"{"status":"open","limit":"10"}"#;
        let query: WorkspaceMilestonesListQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.status_filter.status, Some("open".to_string()));
        assert_eq!(query.pagination.limit, 10);
    }

    #[test]
    fn test_all_workspace_milestones_query_defaults() {
        let json = r#"{}"#;
        let query: AllWorkspaceMilestonesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.pagination.limit, 50);
        assert!(query.status_filter.status.is_none());
        assert!(query.workspace_id.is_none());
    }

    #[test]
    fn test_all_workspace_milestones_query_with_filters() {
        let json = r#"{"status":"open","workspace_id":"b37351e3-6c90-4a53-bc4f-8cbd024cecb7","limit":"5"}"#;
        let query: AllWorkspaceMilestonesQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.status_filter.status, Some("open".to_string()));
        assert_eq!(
            query.workspace_id,
            Some("b37351e3-6c90-4a53-bc4f-8cbd024cecb7".to_string())
        );
        assert_eq!(query.pagination.limit, 5);
    }

    #[test]
    fn test_workspace_milestone_with_workspace_serialization() {
        let resp = WorkspaceMilestoneWithWorkspace {
            milestone: WorkspaceMilestoneResponse {
                id: "test-id".to_string(),
                workspace_id: "ws-id".to_string(),
                title: "Test".to_string(),
                description: None,
                status: "open".to_string(),
                target_date: None,
                closed_at: None,
                created_at: "2026-01-01T00:00:00Z".to_string(),
                tags: vec![],
            },
            workspace_id: "ws-uuid".to_string(),
            workspace_name: "My Workspace".to_string(),
            workspace_slug: "my-workspace".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["title"], "Test");
        assert_eq!(json["workspace_name"], "My Workspace");
        assert_eq!(json["workspace_slug"], "my-workspace");
    }

    #[test]
    fn test_workspace_milestone_with_workspace_flatten() {
        // Verify flatten merges milestone fields into top-level
        let resp = WorkspaceMilestoneWithWorkspace {
            milestone: WorkspaceMilestoneResponse {
                id: "m-123".to_string(),
                workspace_id: "ws-inner".to_string(),
                title: "Cross-project milestone".to_string(),
                description: Some("Important milestone".to_string()),
                status: "open".to_string(),
                target_date: Some("2026-06-01T00:00:00Z".to_string()),
                closed_at: None,
                created_at: "2026-01-15T10:00:00Z".to_string(),
                tags: vec!["release".to_string(), "q2".to_string()],
            },
            workspace_id: "ws-outer".to_string(),
            workspace_name: "Production".to_string(),
            workspace_slug: "production".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();

        // Flattened milestone fields
        assert_eq!(json["id"], "m-123");
        assert_eq!(json["title"], "Cross-project milestone");
        assert_eq!(json["description"], "Important milestone");
        assert_eq!(json["status"], "open");
        assert_eq!(json["target_date"], "2026-06-01T00:00:00Z");
        assert!(json["closed_at"].is_null());
        assert_eq!(json["created_at"], "2026-01-15T10:00:00Z");
        assert_eq!(json["tags"], serde_json::json!(["release", "q2"]));

        // Extra workspace fields
        assert_eq!(json["workspace_id"], "ws-outer");
        assert_eq!(json["workspace_name"], "Production");
        assert_eq!(json["workspace_slug"], "production");
    }

    #[test]
    fn test_all_workspace_milestones_query_workspace_id_validation() {
        // Valid UUID workspace_id
        let json = r#"{"workspace_id":"b37351e3-6c90-4a53-bc4f-8cbd024cecb7"}"#;
        let query: AllWorkspaceMilestonesQuery = serde_json::from_str(json).unwrap();
        let parsed = query
            .workspace_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(uuid::Uuid::parse_str);
        assert!(parsed.is_some());
        assert!(parsed.unwrap().is_ok());
    }

    #[test]
    fn test_all_workspace_milestones_query_invalid_workspace_id() {
        let json = r#"{"workspace_id":"not-a-uuid"}"#;
        let query: AllWorkspaceMilestonesQuery = serde_json::from_str(json).unwrap();
        let parsed = query
            .workspace_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(uuid::Uuid::parse_str);
        assert!(parsed.is_some());
        assert!(parsed.unwrap().is_err());
    }

    #[test]
    fn test_all_workspace_milestones_query_empty_workspace_id() {
        let json = r#"{"workspace_id":""}"#;
        let query: AllWorkspaceMilestonesQuery = serde_json::from_str(json).unwrap();
        // Empty string should be filtered out
        let parsed = query
            .workspace_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(uuid::Uuid::parse_str);
        assert!(parsed.is_none());
    }

    #[test]
    fn test_workspace_milestones_list_query_pagination_values() {
        let json = r#"{"limit":"25","offset":"10","status":"closed"}"#;
        let query: WorkspaceMilestonesListQuery = serde_json::from_str(json).unwrap();
        assert_eq!(query.pagination.limit, 25);
        assert_eq!(query.pagination.offset, 10);
        assert_eq!(query.status_filter.status, Some("closed".to_string()));
    }
}
