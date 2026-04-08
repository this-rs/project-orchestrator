//! API handlers for Blueprints
//!
//! CRUD for blueprint nodes, relations (DEPENDS_ON, PAIRS_WITH),
//! and project linking (APPLIES_TO).

use super::handlers::{AppError, OrchestratorState};
use crate::blueprint::{
    BlueprintCategory, BlueprintDifficulty, BlueprintRelationType, BlueprintScope, BlueprintStatus,
    CreateBlueprintRequest, ListBlueprintsQuery, UpdateBlueprintRequest,
};
use crate::events::EventEmitter;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;

// ============================================================================
// Request / Query DTOs
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BlueprintListQuery {
    pub scope: Option<String>,
    pub category: Option<String>,
    pub status: Option<String>,
    pub stack: Option<String>,
    pub search: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct CreateBlueprintBody {
    pub slug: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub scope: String,
    pub category: String,
    #[serde(default)]
    pub difficulty: Option<String>,
    #[serde(default)]
    pub estimated_time: Option<String>,
    #[serde(default)]
    pub stack: Option<Vec<String>>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub tier1_content: Option<String>,
    #[serde(default)]
    pub tier2_content: Option<String>,
    #[serde(default)]
    pub tier3_content: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub content_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateBlueprintBody {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub difficulty: Option<String>,
    #[serde(default)]
    pub estimated_time: Option<String>,
    #[serde(default)]
    pub stack: Option<Vec<String>>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub tier1_content: Option<String>,
    #[serde(default)]
    pub tier2_content: Option<String>,
    #[serde(default)]
    pub tier3_content: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub content_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RelationBody {
    pub from_slug: String,
    pub to_slug: String,
    pub relation_type: String,
}

#[derive(Debug, Deserialize)]
pub struct ProjectLinkBody {
    pub project_id: String,
    #[serde(default)]
    pub relevance: Option<f64>,
}

// ============================================================================
// CRUD Handlers
// ============================================================================

/// List blueprints with optional filters.
///
/// GET /api/blueprints?scope=...&category=...&status=...&stack=...&search=...&limit=...&offset=...
pub async fn list_blueprints(
    State(state): State<OrchestratorState>,
    Query(query): Query<BlueprintListQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let scope = query
        .scope
        .as_deref()
        .map(|s| s.parse::<BlueprintScope>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid scope: {}", e)))?;
    let category = query
        .category
        .as_deref()
        .map(|s| s.parse::<BlueprintCategory>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid category: {}", e)))?;
    let status = query
        .status
        .as_deref()
        .map(|s| s.parse::<BlueprintStatus>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid status: {}", e)))?;

    let list_query = ListBlueprintsQuery {
        scope,
        category,
        status,
        stack: query.stack.clone(),
        search: query.search.clone(),
        tier: 1, // list always returns tier 1 (catalog)
        limit: query.limit.unwrap_or(50) as usize,
        offset: query.offset.unwrap_or(0) as usize,
    };

    let blueprints = state
        .orchestrator
        .neo4j()
        .list_blueprints(&list_query)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(blueprints).unwrap()))
}

/// Create a new blueprint.
///
/// POST /api/blueprints
pub async fn create_blueprint(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateBlueprintBody>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    if body.slug.trim().is_empty() {
        return Err(AppError::BadRequest("slug cannot be empty".to_string()));
    }
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest("name cannot be empty".to_string()));
    }
    if body.name.len() > crate::blueprint::MAX_BLUEPRINT_NAME_LEN {
        return Err(AppError::BadRequest(format!(
            "name must be {} characters or less",
            crate::blueprint::MAX_BLUEPRINT_NAME_LEN
        )));
    }

    let scope = body.scope.parse::<BlueprintScope>().ok();
    let category = body.category.parse::<BlueprintCategory>().ok();
    let difficulty = body
        .difficulty
        .as_deref()
        .map(|d| d.parse::<BlueprintDifficulty>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid difficulty: {}", e)))?;

    let req = CreateBlueprintRequest {
        slug: body.slug,
        name: body.name,
        description: body.description,
        scope,
        category,
        difficulty,
        estimated_time: body.estimated_time,
        stack: body.stack,
        tags: body.tags,
        tier1: body.tier1_content,
        tier2: body.tier2_content,
        tier3_content: body.tier3_content,
        source_file: body.source_file,
        content_hash: body.content_hash,
    };

    let blueprint = state
        .orchestrator
        .neo4j()
        .create_blueprint(&req)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_created(
        crate::events::EntityType::Blueprint,
        &blueprint.id.to_string(),
        serde_json::json!({
            "slug": blueprint.slug,
            "name": blueprint.name,
            "scope": blueprint.scope.to_string(),
        }),
        None,
    );

    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(blueprint).unwrap()),
    ))
}

/// Get a single blueprint by ID or slug.
///
/// GET /api/blueprints/:id_or_slug
pub async fn get_blueprint(
    State(state): State<OrchestratorState>,
    Path(id_or_slug): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let tier: i32 = params.get("tier").and_then(|t| t.parse().ok()).unwrap_or(3);

    // Try by UUID first, then by slug
    let blueprint = if id_or_slug.contains('-') && id_or_slug.len() > 30 {
        state
            .orchestrator
            .neo4j()
            .get_blueprint(&id_or_slug)
            .await
            .map_err(AppError::Internal)?
    } else {
        state
            .orchestrator
            .neo4j()
            .get_blueprint_by_slug(&id_or_slug)
            .await
            .map_err(AppError::Internal)?
    };

    let blueprint = blueprint
        .ok_or_else(|| AppError::NotFound(format!("Blueprint '{}' not found", id_or_slug)))?;

    let response = blueprint.to_response(tier);
    Ok(Json(serde_json::to_value(response).unwrap()))
}

/// Update a blueprint.
///
/// PATCH /api/blueprints/:id
pub async fn update_blueprint(
    State(state): State<OrchestratorState>,
    Path(blueprint_id): Path<String>,
    Json(body): Json<UpdateBlueprintBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let scope = body
        .scope
        .as_deref()
        .map(|s| s.parse::<BlueprintScope>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid scope: {}", e)))?;
    let category = body
        .category
        .as_deref()
        .map(|s| s.parse::<BlueprintCategory>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid category: {}", e)))?;
    let difficulty = body
        .difficulty
        .as_deref()
        .map(|d| d.parse::<BlueprintDifficulty>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid difficulty: {}", e)))?;
    let status = body
        .status
        .as_deref()
        .map(|s| s.parse::<BlueprintStatus>())
        .transpose()
        .map_err(|e| AppError::BadRequest(format!("invalid status: {}", e)))?;

    let req = UpdateBlueprintRequest {
        name: body.name,
        description: body.description,
        scope,
        category,
        difficulty,
        estimated_time: body.estimated_time,
        stack: body.stack,
        tags: body.tags,
        status,
        tier1: body.tier1_content,
        tier2: body.tier2_content,
        tier3_content: body.tier3_content,
        source_file: body.source_file,
        content_hash: body.content_hash,
    };

    let blueprint = state
        .orchestrator
        .neo4j()
        .update_blueprint(&blueprint_id, &req)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_updated(
        crate::events::EntityType::Blueprint,
        &blueprint.id.to_string(),
        serde_json::json!({
            "slug": blueprint.slug,
            "name": blueprint.name,
        }),
        None,
    );

    Ok(Json(serde_json::to_value(blueprint).unwrap()))
}

/// Delete a blueprint.
///
/// DELETE /api/blueprints/:id
pub async fn delete_blueprint(
    State(state): State<OrchestratorState>,
    Path(blueprint_id): Path<String>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .delete_blueprint(&blueprint_id)
        .await
        .map_err(AppError::Internal)?;

    state
        .event_bus
        .emit_deleted(crate::events::EntityType::Blueprint, &blueprint_id, None);

    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation Handlers
// ============================================================================

/// Add a relation between two blueprints (DEPENDS_ON or PAIRS_WITH).
///
/// POST /api/blueprints/relations
pub async fn add_blueprint_relation(
    State(state): State<OrchestratorState>,
    Json(body): Json<RelationBody>,
) -> Result<StatusCode, AppError> {
    let relation_type: BlueprintRelationType = body
        .relation_type
        .parse()
        .map_err(|e: String| AppError::BadRequest(format!("invalid relation_type: {}", e)))?;

    state
        .orchestrator
        .neo4j()
        .add_blueprint_relation(&body.from_slug, &body.to_slug, relation_type)
        .await
        .map_err(AppError::Internal)?;

    Ok(StatusCode::CREATED)
}

/// Remove a relation between two blueprints.
///
/// DELETE /api/blueprints/relations
pub async fn remove_blueprint_relation(
    State(state): State<OrchestratorState>,
    Json(body): Json<RelationBody>,
) -> Result<StatusCode, AppError> {
    let relation_type: BlueprintRelationType = body
        .relation_type
        .parse()
        .map_err(|e: String| AppError::BadRequest(format!("invalid relation_type: {}", e)))?;

    state
        .orchestrator
        .neo4j()
        .remove_blueprint_relation(&body.from_slug, &body.to_slug, relation_type)
        .await
        .map_err(AppError::Internal)?;

    Ok(StatusCode::NO_CONTENT)
}

/// Get relations for a blueprint (dependencies, dependents, pairs).
///
/// GET /api/blueprints/:slug/relations
pub async fn get_blueprint_relations(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deps = state
        .orchestrator
        .neo4j()
        .get_blueprint_dependencies(&slug)
        .await
        .map_err(AppError::Internal)?;
    let dependents = state
        .orchestrator
        .neo4j()
        .get_blueprint_dependents(&slug)
        .await
        .map_err(AppError::Internal)?;
    let pairs = state
        .orchestrator
        .neo4j()
        .get_blueprint_pairs(&slug)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "dependencies": deps,
        "dependents": dependents,
        "pairs_with": pairs,
    })))
}

// ============================================================================
// Project Linking Handlers
// ============================================================================

/// Link a blueprint to a project (APPLIES_TO).
///
/// POST /api/blueprints/:slug/projects
pub async fn link_blueprint_to_project(
    State(state): State<OrchestratorState>,
    Path(slug): Path<String>,
    Json(body): Json<ProjectLinkBody>,
) -> Result<StatusCode, AppError> {
    let relevance = body.relevance.unwrap_or(1.0);

    state
        .orchestrator
        .neo4j()
        .link_blueprint_to_project(&slug, &body.project_id, relevance)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_linked(
        crate::events::EntityType::Blueprint,
        &slug,
        crate::events::EntityType::Project,
        &body.project_id,
        Some(body.project_id.clone()),
    );

    Ok(StatusCode::CREATED)
}

/// Unlink a blueprint from a project.
///
/// DELETE /api/blueprints/:slug/projects/:project_id
pub async fn unlink_blueprint_from_project(
    State(state): State<OrchestratorState>,
    Path((slug, project_id)): Path<(String, String)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .unlink_blueprint_from_project(&slug, &project_id)
        .await
        .map_err(AppError::Internal)?;

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Blueprint,
        &slug,
        crate::events::EntityType::Project,
        &project_id,
        Some(project_id.clone()),
    );

    Ok(StatusCode::NO_CONTENT)
}

/// Get blueprints linked to a project.
///
/// GET /api/projects/:project_id/blueprints
pub async fn get_project_blueprints(
    State(state): State<OrchestratorState>,
    Path(project_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let blueprints = state
        .orchestrator
        .neo4j()
        .get_project_blueprints(&project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::to_value(blueprints).unwrap()))
}
