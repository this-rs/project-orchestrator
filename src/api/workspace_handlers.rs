//! Workspace API handlers

use crate::api::{PaginatedResponse, PaginationParams};
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
            status: format!("{:?}", m.status),
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

    state
        .orchestrator
        .neo4j()
        .create_workspace(&workspace)
        .await?;

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
        .neo4j()
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

    state
        .orchestrator
        .neo4j()
        .delete_workspace(workspace.id)
        .await?;

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
) -> Result<Json<Vec<ProjectSummary>>, AppError> {
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

    Ok(Json(
        projects
            .into_iter()
            .map(|p| ProjectSummary {
                id: p.id.to_string(),
                name: p.name,
                slug: p.slug,
            })
            .collect(),
    ))
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
        .neo4j()
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
        .neo4j()
        .remove_project_from_workspace(workspace.id, project_id)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Workspace Milestone Handlers
// ============================================================================

/// List workspace milestones
pub async fn list_workspace_milestones(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<Vec<WorkspaceMilestoneResponse>>, AppError> {
    let workspace = state
        .orchestrator
        .neo4j()
        .get_workspace_by_slug(&slug)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Workspace '{}' not found", slug)))?;

    let milestones = state
        .orchestrator
        .neo4j()
        .list_workspace_milestones(workspace.id)
        .await?;

    Ok(Json(
        milestones
            .into_iter()
            .map(WorkspaceMilestoneResponse::from)
            .collect(),
    ))
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
        .neo4j()
        .create_workspace_milestone(&milestone)
        .await?;

    Ok((
        StatusCode::CREATED,
        Json(WorkspaceMilestoneResponse::from(milestone)),
    ))
}

/// Get workspace milestone
pub async fn get_workspace_milestone(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<Json<WorkspaceMilestoneResponse>, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid milestone ID".to_string()))?;

    let milestone = state
        .orchestrator
        .neo4j()
        .get_workspace_milestone(id)
        .await?
        .ok_or_else(|| AppError::NotFound("Milestone not found".to_string()))?;

    Ok(Json(WorkspaceMilestoneResponse::from(milestone)))
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

    let status = req.status.map(|s| match s.to_lowercase().as_str() {
        "closed" => MilestoneStatus::Closed,
        _ => MilestoneStatus::Open,
    });

    let target_date = req
        .target_date
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    state
        .orchestrator
        .neo4j()
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

    state
        .orchestrator
        .neo4j()
        .delete_workspace_milestone(id)
        .await?;

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
        .neo4j()
        .add_task_to_workspace_milestone(milestone_id, task_id)
        .await?;

    Ok(StatusCode::CREATED)
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

    state
        .orchestrator
        .neo4j()
        .create_resource(&resource)
        .await?;

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

/// Delete resource
pub async fn delete_resource(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid resource ID".to_string()))?;

    state.orchestrator.neo4j().delete_resource(id).await?;

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
                .neo4j()
                .link_project_implements_resource(project_id, resource_id)
                .await?;
        }
        "uses" => {
            state
                .orchestrator
                .neo4j()
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

    state
        .orchestrator
        .neo4j()
        .create_component(&component)
        .await?;

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

/// Delete component
pub async fn delete_component(
    State(state): State<OrchestratorState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let id: Uuid = id
        .parse()
        .map_err(|_| AppError::BadRequest("Invalid component ID".to_string()))?;

    state.orchestrator.neo4j().delete_component(id).await?;

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
        .neo4j()
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
        .neo4j()
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
        .neo4j()
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
