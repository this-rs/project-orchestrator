//! API handlers for Neural Skills

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams};
use crate::skills::{ActivatedSkillContext, SkillNode, SkillStatus, SkillTrigger};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use uuid::Uuid;

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for listing skills
#[derive(Debug, Deserialize, Default)]
pub struct SkillsListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    /// Required: filter skills by project
    pub project_id: Uuid,
    /// Optional: filter by status (emerging, active, dormant, archived, imported)
    pub status: Option<String>,
}

// ============================================================================
// Request Bodies
// ============================================================================

/// Request body for creating a skill
#[derive(Debug, Deserialize)]
pub struct CreateSkillBody {
    pub project_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub trigger_patterns: Option<Vec<SkillTrigger>>,
    pub context_template: Option<String>,
}

/// Request body for updating a skill
#[derive(Debug, Deserialize)]
pub struct UpdateSkillBody {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<String>,
    pub tags: Option<Vec<String>>,
    pub trigger_patterns: Option<Vec<SkillTrigger>>,
    pub context_template: Option<String>,
    pub energy: Option<f64>,
    pub cohesion: Option<f64>,
}

/// Request body for adding a member to a skill
#[derive(Debug, Deserialize)]
pub struct AddMemberBody {
    /// "note" or "decision"
    pub entity_type: String,
    pub entity_id: Uuid,
}

/// Response for skill members
#[derive(Debug, serde::Serialize)]
pub struct SkillMembersResponse {
    pub notes: Vec<crate::notes::Note>,
    pub decisions: Vec<crate::neo4j::models::DecisionNode>,
}

/// Request body for activating a skill
#[derive(Debug, Deserialize)]
pub struct ActivateSkillBody {
    pub query: String,
}

// ============================================================================
// Handlers — CRUD
// ============================================================================

/// List skills for a project
///
/// GET /api/skills?project_id=...&status=...&limit=...&offset=...
pub async fn list_skills(
    State(state): State<OrchestratorState>,
    Query(query): Query<SkillsListQuery>,
) -> Result<Json<PaginatedResponse<SkillNode>>, AppError> {
    query
        .pagination
        .validate()
        .map_err(AppError::BadRequest)?;

    let status_filter = query
        .status
        .as_ref()
        .and_then(|s| s.parse::<SkillStatus>().ok());

    let limit = query.pagination.validated_limit();
    let offset = query.pagination.offset;

    let (skills, total) = state
        .orchestrator
        .neo4j()
        .list_skills(query.project_id, status_filter, limit, offset)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(PaginatedResponse::new(skills, total, limit, offset)))
}

/// Create a new skill
///
/// POST /api/skills
pub async fn create_skill(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateSkillBody>,
) -> Result<(StatusCode, Json<SkillNode>), AppError> {
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest("name cannot be empty".to_string()));
    }

    let mut skill = SkillNode::new(body.project_id, body.name);
    if let Some(desc) = body.description {
        skill.description = desc;
    }
    if let Some(tags) = body.tags {
        skill.tags = tags;
    }
    if let Some(triggers) = body.trigger_patterns {
        skill.trigger_patterns = triggers;
    }
    skill.context_template = body.context_template;

    state
        .orchestrator
        .neo4j()
        .create_skill(&skill)
        .await
        .map_err(AppError::Internal)?;

    Ok((StatusCode::CREATED, Json(skill)))
}

/// Get a skill by ID
///
/// GET /api/skills/:id
pub async fn get_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
) -> Result<Json<SkillNode>, AppError> {
    let skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", skill_id)))?;

    Ok(Json(skill))
}

/// Update a skill
///
/// PUT /api/skills/:id
pub async fn update_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Json(body): Json<UpdateSkillBody>,
) -> Result<Json<SkillNode>, AppError> {
    let mut skill = state
        .orchestrator
        .neo4j()
        .get_skill(skill_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Skill {} not found", skill_id)))?;

    // Apply updates
    if let Some(name) = body.name {
        if name.trim().is_empty() {
            return Err(AppError::BadRequest("name cannot be empty".to_string()));
        }
        skill.name = name;
    }
    if let Some(description) = body.description {
        skill.description = description;
    }
    if let Some(status_str) = body.status {
        skill.status = status_str
            .parse()
            .map_err(|e: String| AppError::BadRequest(e))?;
    }
    if let Some(tags) = body.tags {
        skill.tags = tags;
    }
    if let Some(triggers) = body.trigger_patterns {
        skill.trigger_patterns = triggers;
    }
    if let Some(template) = body.context_template {
        skill.context_template = Some(template);
    }
    if let Some(energy) = body.energy {
        skill.energy = energy.clamp(0.0, 1.0);
    }
    if let Some(cohesion) = body.cohesion {
        skill.cohesion = cohesion.clamp(0.0, 1.0);
    }
    skill.updated_at = Utc::now();

    state
        .orchestrator
        .neo4j()
        .update_skill(&skill)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(skill))
}

/// Delete a skill
///
/// DELETE /api/skills/:id
pub async fn delete_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_skill(skill_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "Skill {} not found",
            skill_id
        )))
    }
}

// ============================================================================
// Handlers — Membership
// ============================================================================

/// Get members (notes + decisions) of a skill
///
/// GET /api/skills/:id/members
pub async fn get_skill_members(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
) -> Result<Json<SkillMembersResponse>, AppError> {
    let (notes, decisions) = state
        .orchestrator
        .neo4j()
        .get_skill_members(skill_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(SkillMembersResponse { notes, decisions }))
}

/// Add a member (note or decision) to a skill
///
/// POST /api/skills/:id/members
pub async fn add_skill_member(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Json(body): Json<AddMemberBody>,
) -> Result<StatusCode, AppError> {
    if !["note", "decision"].contains(&body.entity_type.as_str()) {
        return Err(AppError::BadRequest(
            "entity_type must be 'note' or 'decision'".to_string(),
        ));
    }

    state
        .orchestrator
        .neo4j()
        .add_skill_member(skill_id, &body.entity_type, body.entity_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(StatusCode::CREATED)
}

/// Remove a member from a skill
///
/// DELETE /api/skills/:id/members/:entity_type/:entity_id
pub async fn remove_skill_member(
    State(state): State<OrchestratorState>,
    Path((skill_id, entity_type, entity_id)): Path<(Uuid, String, Uuid)>,
) -> Result<StatusCode, AppError> {
    if !["note", "decision"].contains(&entity_type.as_str()) {
        return Err(AppError::BadRequest(
            "entity_type must be 'note' or 'decision'".to_string(),
        ));
    }

    let removed = state
        .orchestrator
        .neo4j()
        .remove_skill_member(skill_id, &entity_type, entity_id)
        .await
        .map_err(AppError::Internal)?;

    if removed {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "Member {} ({}) not found in skill {}",
            entity_id, entity_type, skill_id
        )))
    }
}

// ============================================================================
// Handlers — Activation
// ============================================================================

/// Activate a skill — retrieve enriched context from its members
///
/// POST /api/skills/:id/activate
pub async fn activate_skill(
    State(state): State<OrchestratorState>,
    Path(skill_id): Path<Uuid>,
    Json(body): Json<ActivateSkillBody>,
) -> Result<Json<ActivatedSkillContext>, AppError> {
    if body.query.trim().is_empty() {
        return Err(AppError::BadRequest("query cannot be empty".to_string()));
    }

    let context = state
        .orchestrator
        .neo4j()
        .activate_skill(skill_id, &body.query)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(context))
}
