//! API handlers for Living Personas

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams};
use crate::neo4j::models::{PersonaNode, PersonaOrigin, PersonaStatus, PersonaSubgraph};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use uuid::Uuid;

// ============================================================================
// Validation constants
// ============================================================================

const MAX_NAME_LEN: usize = 200;
const MAX_DESCRIPTION_LEN: usize = 10_000;
const MAX_SYSTEM_PROMPT_LEN: usize = 50_000;

// ============================================================================
// Request / query types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct PersonaListQuery {
    pub project_id: Uuid,
    pub status: Option<String>,
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

#[derive(Debug, Deserialize)]
pub struct GlobalPersonaListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
}

#[derive(Debug, Deserialize)]
pub struct CreatePersonaBody {
    pub project_id: Option<Uuid>,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub complexity_default: Option<String>,
    pub timeout_secs: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub model_preference: Option<String>,
    pub system_prompt_override: Option<String>,
    pub origin: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdatePersonaBody {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<String>,
    pub complexity_default: Option<String>,
    pub timeout_secs: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub model_preference: Option<String>,
    pub system_prompt_override: Option<String>,
    pub energy: Option<f64>,
    pub cohesion: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct AddFileBody {
    pub file_path: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
pub struct RemoveFileBody {
    pub file_path: String,
}

#[derive(Debug, Deserialize)]
pub struct AddFunctionBody {
    pub function_name: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
pub struct RemoveFunctionBody {
    pub function_name: String,
}

#[derive(Debug, Deserialize)]
pub struct AddWeightedBody {
    #[serde(default = "default_weight")]
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
pub struct FindForFileQuery {
    pub file_path: String,
    pub project_id: Uuid,
}

fn default_weight() -> f64 {
    1.0
}

// ============================================================================
// CRUD handlers
// ============================================================================

/// List personas for a project
///
/// GET /api/personas?project_id=...&status=...&limit=...&offset=...
pub async fn list_personas(
    State(state): State<OrchestratorState>,
    Query(query): Query<PersonaListQuery>,
) -> Result<Json<PaginatedResponse<PersonaNode>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let status_filter = match &query.status {
        Some(s) => Some(s.parse::<PersonaStatus>().map_err(AppError::BadRequest)?),
        None => None,
    };

    let limit = query.pagination.validated_limit();
    let offset = query.pagination.offset;

    let (personas, total) = state
        .orchestrator
        .neo4j()
        .list_personas(query.project_id, status_filter, limit, offset)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(PaginatedResponse::new(personas, total, limit, offset)))
}

/// List global personas (project_id IS NULL)
///
/// GET /api/personas/global?limit=...&offset=...
pub async fn list_global_personas(
    State(state): State<OrchestratorState>,
    Query(_query): Query<GlobalPersonaListQuery>,
) -> Result<Json<Vec<PersonaNode>>, AppError> {
    let personas = state
        .orchestrator
        .neo4j()
        .list_global_personas()
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(personas))
}

/// Create a new persona
///
/// POST /api/personas
pub async fn create_persona(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreatePersonaBody>,
) -> Result<(StatusCode, Json<PersonaNode>), AppError> {
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest("name cannot be empty".to_string()));
    }
    if body.name.len() > MAX_NAME_LEN {
        return Err(AppError::BadRequest(format!(
            "name too long ({} > {} chars)",
            body.name.len(),
            MAX_NAME_LEN
        )));
    }
    if body.description.len() > MAX_DESCRIPTION_LEN {
        return Err(AppError::BadRequest(format!(
            "description too long ({} > {} chars)",
            body.description.len(),
            MAX_DESCRIPTION_LEN
        )));
    }
    if let Some(ref sp) = body.system_prompt_override {
        if sp.len() > MAX_SYSTEM_PROMPT_LEN {
            return Err(AppError::BadRequest(format!(
                "system_prompt_override too long ({} > {} chars)",
                sp.len(),
                MAX_SYSTEM_PROMPT_LEN
            )));
        }
    }

    let origin = match body.origin.as_deref() {
        Some(s) => s.parse::<PersonaOrigin>().map_err(AppError::BadRequest)?,
        None => PersonaOrigin::Manual,
    };

    let persona = PersonaNode {
        id: Uuid::new_v4(),
        project_id: body.project_id,
        name: body.name,
        description: body.description,
        status: PersonaStatus::Emerging,
        complexity_default: body.complexity_default,
        timeout_secs: body.timeout_secs,
        max_cost_usd: body.max_cost_usd,
        model_preference: body.model_preference,
        system_prompt_override: body.system_prompt_override,
        energy: 0.5,
        cohesion: 0.0,
        activation_count: 0,
        success_rate: 0.0,
        avg_duration_secs: 0.0,
        last_activated: None,
        origin,
        created_at: Utc::now(),
        updated_at: None,
    };

    state
        .orchestrator
        .neo4j()
        .create_persona(&persona)
        .await
        .map_err(AppError::Internal)?;

    Ok((StatusCode::CREATED, Json(persona)))
}

/// Get a persona by ID
///
/// GET /api/personas/:id
pub async fn get_persona(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
) -> Result<Json<PersonaNode>, AppError> {
    let persona = state
        .orchestrator
        .neo4j()
        .get_persona(persona_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Persona {} not found", persona_id)))?;

    Ok(Json(persona))
}

/// Update a persona
///
/// PUT /api/personas/:id
pub async fn update_persona(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
    Json(body): Json<UpdatePersonaBody>,
) -> Result<Json<PersonaNode>, AppError> {
    let mut persona = state
        .orchestrator
        .neo4j()
        .get_persona(persona_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Persona {} not found", persona_id)))?;

    if let Some(name) = body.name {
        if name.trim().is_empty() {
            return Err(AppError::BadRequest("name cannot be empty".to_string()));
        }
        if name.len() > MAX_NAME_LEN {
            return Err(AppError::BadRequest(format!(
                "name too long ({} > {} chars)",
                name.len(),
                MAX_NAME_LEN
            )));
        }
        persona.name = name;
    }
    if let Some(description) = body.description {
        if description.len() > MAX_DESCRIPTION_LEN {
            return Err(AppError::BadRequest(format!(
                "description too long ({} > {} chars)",
                description.len(),
                MAX_DESCRIPTION_LEN
            )));
        }
        persona.description = description;
    }
    if let Some(status_str) = body.status {
        persona.status = status_str
            .parse::<PersonaStatus>()
            .map_err(AppError::BadRequest)?;
    }
    if let Some(complexity) = body.complexity_default {
        persona.complexity_default = Some(complexity);
    }
    if let Some(timeout) = body.timeout_secs {
        persona.timeout_secs = Some(timeout);
    }
    if let Some(cost) = body.max_cost_usd {
        persona.max_cost_usd = Some(cost);
    }
    if let Some(model) = body.model_preference {
        persona.model_preference = Some(model);
    }
    if let Some(sp) = body.system_prompt_override {
        if sp.len() > MAX_SYSTEM_PROMPT_LEN {
            return Err(AppError::BadRequest(format!(
                "system_prompt_override too long ({} > {} chars)",
                sp.len(),
                MAX_SYSTEM_PROMPT_LEN
            )));
        }
        persona.system_prompt_override = Some(sp);
    }
    if let Some(energy) = body.energy {
        persona.energy = energy.clamp(0.0, 1.0);
    }
    if let Some(cohesion) = body.cohesion {
        persona.cohesion = cohesion.clamp(0.0, 1.0);
    }
    persona.updated_at = Some(Utc::now());

    state
        .orchestrator
        .neo4j()
        .update_persona(&persona)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(persona))
}

/// Delete a persona
///
/// DELETE /api/personas/:id
pub async fn delete_persona(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_persona(persona_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "Persona {} not found",
            persona_id
        )))
    }
}

// ============================================================================
// Relation handlers — Skills (MASTERS)
// ============================================================================

/// POST /api/personas/:persona_id/skills/:skill_id
pub async fn add_skill(
    State(state): State<OrchestratorState>,
    Path((persona_id, skill_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .add_persona_skill(persona_id, skill_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/skills/:skill_id
pub async fn remove_skill(
    State(state): State<OrchestratorState>,
    Path((persona_id, skill_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_skill(persona_id, skill_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation handlers — Protocols (FOLLOWS)
// ============================================================================

/// POST /api/personas/:persona_id/protocols/:protocol_id
pub async fn add_protocol(
    State(state): State<OrchestratorState>,
    Path((persona_id, protocol_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .add_persona_protocol(persona_id, protocol_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/protocols/:protocol_id
pub async fn remove_protocol(
    State(state): State<OrchestratorState>,
    Path((persona_id, protocol_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_protocol(persona_id, protocol_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation handlers — FeatureGraph (SCOPED_TO)
// ============================================================================

/// POST /api/personas/:persona_id/feature-graphs/:feature_graph_id
pub async fn scope_to_feature_graph(
    State(state): State<OrchestratorState>,
    Path((persona_id, feature_graph_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .set_persona_feature_graph(persona_id, feature_graph_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/feature-graphs/:feature_graph_id
pub async fn unscope_feature_graph(
    State(state): State<OrchestratorState>,
    Path((persona_id, _feature_graph_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_feature_graph(persona_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation handlers — Files (KNOWS)
// ============================================================================

/// POST /api/personas/:persona_id/files
pub async fn add_file(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
    Json(body): Json<AddFileBody>,
) -> Result<StatusCode, AppError> {
    if body.file_path.trim().is_empty() {
        return Err(AppError::BadRequest(
            "file_path cannot be empty".to_string(),
        ));
    }
    let weight = body.weight.clamp(0.0, 1.0);
    state
        .orchestrator
        .neo4j()
        .add_persona_file(persona_id, &body.file_path, weight)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/files
pub async fn remove_file(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
    Json(body): Json<RemoveFileBody>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_file(persona_id, &body.file_path)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation handlers — Functions (KNOWS)
// ============================================================================

/// POST /api/personas/:persona_id/functions
pub async fn add_function(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
    Json(body): Json<AddFunctionBody>,
) -> Result<StatusCode, AppError> {
    if body.function_name.trim().is_empty() {
        return Err(AppError::BadRequest(
            "function_name cannot be empty".to_string(),
        ));
    }
    let weight = body.weight.clamp(0.0, 1.0);
    state
        .orchestrator
        .neo4j()
        .add_persona_function(persona_id, &body.function_name, weight)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/functions
pub async fn remove_function(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
    Json(body): Json<RemoveFunctionBody>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_function(persona_id, &body.function_name)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation handlers — Notes (USES)
// ============================================================================

/// POST /api/personas/:persona_id/notes/:note_id
pub async fn add_note(
    State(state): State<OrchestratorState>,
    Path((persona_id, note_id)): Path<(Uuid, Uuid)>,
    Json(body): Json<AddWeightedBody>,
) -> Result<StatusCode, AppError> {
    let weight = body.weight.clamp(0.0, 1.0);
    state
        .orchestrator
        .neo4j()
        .add_persona_note(persona_id, note_id, weight)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/notes/:note_id
pub async fn remove_note(
    State(state): State<OrchestratorState>,
    Path((persona_id, note_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_note(persona_id, note_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation handlers — Decisions (USES)
// ============================================================================

/// POST /api/personas/:persona_id/decisions/:decision_id
pub async fn add_decision(
    State(state): State<OrchestratorState>,
    Path((persona_id, decision_id)): Path<(Uuid, Uuid)>,
    Json(body): Json<AddWeightedBody>,
) -> Result<StatusCode, AppError> {
    let weight = body.weight.clamp(0.0, 1.0);
    state
        .orchestrator
        .neo4j()
        .add_persona_decision(persona_id, decision_id, weight)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/decisions/:decision_id
pub async fn remove_decision(
    State(state): State<OrchestratorState>,
    Path((persona_id, decision_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_decision(persona_id, decision_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Relation handlers — Extends (EXTENDS)
// ============================================================================

/// POST /api/personas/:persona_id/extends/:parent_persona_id
pub async fn add_extends(
    State(state): State<OrchestratorState>,
    Path((persona_id, parent_persona_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    if persona_id == parent_persona_id {
        return Err(AppError::BadRequest(
            "A persona cannot extend itself".to_string(),
        ));
    }
    state
        .orchestrator
        .neo4j()
        .add_persona_extends(persona_id, parent_persona_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

/// DELETE /api/personas/:persona_id/extends/:parent_persona_id
pub async fn remove_extends(
    State(state): State<OrchestratorState>,
    Path((persona_id, parent_persona_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    state
        .orchestrator
        .neo4j()
        .remove_persona_extends(persona_id, parent_persona_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(StatusCode::NO_CONTENT)
}

// ============================================================================
// Subgraph & discovery
// ============================================================================

/// Get the full knowledge subgraph for a persona
///
/// GET /api/personas/:id/subgraph
pub async fn get_subgraph(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
) -> Result<Json<PersonaSubgraph>, AppError> {
    let subgraph = state
        .orchestrator
        .neo4j()
        .get_persona_subgraph(persona_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(subgraph))
}

/// Find personas that KNOW a given file
///
/// GET /api/personas/find-for-file?file_path=...&project_id=...
pub async fn find_for_file(
    State(state): State<OrchestratorState>,
    Query(query): Query<FindForFileQuery>,
) -> Result<Json<Vec<PersonaMatch>>, AppError> {
    let matches = state
        .orchestrator
        .neo4j()
        .find_personas_for_file(&query.file_path, query.project_id)
        .await
        .map_err(AppError::Internal)?;

    let result: Vec<PersonaMatch> = matches
        .into_iter()
        .map(|(persona, weight)| PersonaMatch { persona, weight })
        .collect();

    Ok(Json(result))
}

/// A persona match with its relevance weight for a file query.
#[derive(Debug, serde::Serialize)]
pub struct PersonaMatch {
    pub persona: PersonaNode,
    pub weight: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weight() {
        assert!((default_weight() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_persona_origin_parse() {
        assert_eq!("manual".parse::<PersonaOrigin>().unwrap(), PersonaOrigin::Manual);
        assert_eq!("auto_build".parse::<PersonaOrigin>().unwrap(), PersonaOrigin::AutoBuild);
        assert_eq!("imported".parse::<PersonaOrigin>().unwrap(), PersonaOrigin::Imported);
        assert!("unknown".parse::<PersonaOrigin>().is_err());
    }

    #[test]
    fn test_persona_status_parse() {
        assert_eq!("active".parse::<PersonaStatus>().unwrap(), PersonaStatus::Active);
        assert_eq!("dormant".parse::<PersonaStatus>().unwrap(), PersonaStatus::Dormant);
        assert_eq!("emerging".parse::<PersonaStatus>().unwrap(), PersonaStatus::Emerging);
        assert_eq!("archived".parse::<PersonaStatus>().unwrap(), PersonaStatus::Archived);
        assert!("invalid".parse::<PersonaStatus>().is_err());
    }

    #[test]
    fn test_create_body_deserialization() {
        let json = serde_json::json!({
            "project_id": "00000000-0000-0000-0000-000000000001",
            "name": "test-persona",
            "description": "A test persona",
            "origin": "manual"
        });
        let body: CreatePersonaBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.name, "test-persona");
        assert_eq!(body.origin.as_deref(), Some("manual"));
        assert!(body.project_id.is_some());
    }

    #[test]
    fn test_create_body_minimal() {
        let json = serde_json::json!({
            "name": "minimal"
        });
        let body: CreatePersonaBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.name, "minimal");
        assert!(body.project_id.is_none());
        assert!(body.origin.is_none());
    }

    #[test]
    fn test_update_body_partial() {
        let json = serde_json::json!({
            "energy": 0.8,
            "status": "active"
        });
        let body: UpdatePersonaBody = serde_json::from_value(json).unwrap();
        assert!(body.name.is_none());
        assert_eq!(body.energy, Some(0.8));
        assert_eq!(body.status.as_deref(), Some("active"));
    }

    #[test]
    fn test_add_file_body_default_weight() {
        let json = serde_json::json!({
            "file_path": "src/main.rs"
        });
        let body: AddFileBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.file_path, "src/main.rs");
        assert!((body.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_weighted_body_default() {
        let json = serde_json::json!({});
        let body: AddWeightedBody = serde_json::from_value(json).unwrap();
        assert!((body.weight - 1.0).abs() < f64::EPSILON);
    }
}
