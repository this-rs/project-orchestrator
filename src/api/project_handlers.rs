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
    query
        .pagination
        .validate()
        .map_err(|e| AppError::BadRequest(e))?;

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
        let files = state
            .orchestrator
            .neo4j()
            .list_project_files(project.id)
            .await
            .unwrap_or_default();
        let plans = state
            .orchestrator
            .neo4j()
            .list_project_plans(project.id)
            .await
            .unwrap_or_default();

        responses.push(ProjectResponse {
            id: project.id.to_string(),
            name: project.name.clone(),
            slug: project.slug.clone(),
            root_path: project.root_path.clone(),
            description: project.description.clone(),
            created_at: project.created_at.to_rfc3339(),
            last_synced: project.last_synced.map(|dt| dt.to_rfc3339()),
            file_count: files.len(),
            plan_count: plans.len(),
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
        root_path: req.root_path,
        description: req.description,
        created_at: chrono::Utc::now(),
        last_synced: None,
    };

    state.orchestrator.neo4j().create_project(&project).await?;

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

    let files = state
        .orchestrator
        .neo4j()
        .list_project_files(project.id)
        .await
        .unwrap_or_default();
    let plans = state
        .orchestrator
        .neo4j()
        .list_project_plans(project.id)
        .await
        .unwrap_or_default();

    Ok(Json(ProjectResponse {
        id: project.id.to_string(),
        name: project.name,
        slug: project.slug,
        root_path: project.root_path,
        description: project.description,
        created_at: project.created_at.to_rfc3339(),
        last_synced: project.last_synced.map(|dt| dt.to_rfc3339()),
        file_count: files.len(),
        plan_count: plans.len(),
    }))
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

    state
        .orchestrator
        .neo4j()
        .delete_project(project.id)
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Sync a project's codebase
#[derive(Serialize)]
pub struct SyncProjectResponse {
    pub files_synced: usize,
    pub files_skipped: usize,
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

    let path = std::path::Path::new(&project.root_path);
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

    Ok(Json(SyncProjectResponse {
        files_synced: result.files_synced,
        files_skipped: result.files_skipped,
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
}
