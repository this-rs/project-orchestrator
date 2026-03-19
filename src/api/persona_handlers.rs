//! API handlers for Living Personas

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams};
use crate::events::EventEmitter;
use crate::neo4j::models::{
    DecisionNode, PersonaImportResult, PersonaNode, PersonaOrigin, PersonaPackage,
    PersonaPackageSource, PersonaStatus, PersonaSubgraph, PortablePersona, PortablePersonaDecision,
    PortablePersonaNote,
};
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
        energy_boost_accumulated: 0.0,
        energy_history: vec![],
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

    state.event_bus.emit_created(
        crate::events::EntityType::Persona,
        &persona.id.to_string(),
        serde_json::json!({
            "name": persona.name,
            "project_id": persona.project_id,
            "origin": format!("{:?}", persona.origin),
        }),
        persona.project_id.map(|pid| pid.to_string()),
    );

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

    let old_status = format!("{:?}", persona.status);

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

    let new_status = format!("{:?}", persona.status);
    if old_status != new_status {
        state.event_bus.emit_status_changed(
            crate::events::EntityType::Persona,
            &persona.id.to_string(),
            &old_status,
            &new_status,
            persona.project_id.map(|pid| pid.to_string()),
        );
    } else {
        state.event_bus.emit_updated(
            crate::events::EntityType::Persona,
            &persona.id.to_string(),
            serde_json::json!({
                "name": &persona.name,
                "energy": persona.energy,
                "cohesion": persona.cohesion,
            }),
            persona.project_id.map(|pid| pid.to_string()),
        );
    }

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
        state.event_bus.emit_deleted(
            crate::events::EntityType::Persona,
            &persona_id.to_string(),
            None,
        );
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

    state.event_bus.emit_linked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Skill,
        &skill_id.to_string(),
        None,
    );

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

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Skill,
        &skill_id.to_string(),
        None,
    );

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

    state.event_bus.emit_linked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Protocol,
        &protocol_id.to_string(),
        None,
    );

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

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Protocol,
        &protocol_id.to_string(),
        None,
    );

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

    state.event_bus.emit_linked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Resource,
        &body.file_path,
        None,
    );

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

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Resource,
        &body.file_path,
        None,
    );

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

    state.event_bus.emit_linked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Resource,
        &body.function_name,
        None,
    );

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

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Resource,
        &body.function_name,
        None,
    );

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

    state.event_bus.emit_linked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Note,
        &note_id.to_string(),
        None,
    );

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

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Note,
        &note_id.to_string(),
        None,
    );

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

    state.event_bus.emit_linked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Decision,
        &decision_id.to_string(),
        None,
    );

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

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Decision,
        &decision_id.to_string(),
        None,
    );

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

    state.event_bus.emit_linked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Persona,
        &parent_persona_id.to_string(),
        None,
    );

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

    state.event_bus.emit_unlinked(
        crate::events::EntityType::Persona,
        &persona_id.to_string(),
        crate::events::EntityType::Persona,
        &parent_persona_id.to_string(),
        None,
    );

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
///
/// Fallback chain (same as PreToolUse hook):
/// 1. Exact KNOWS match (direct persona-file relation)
/// 2. Directory-prefix match (persona KNOWS ≥ 2 files in same dir, weight = avg * 0.7)
/// 3. Community match via SCOPED_TO FeatureGraph (weight 0.3)
pub async fn find_for_file(
    State(state): State<OrchestratorState>,
    Query(query): Query<FindForFileQuery>,
) -> Result<Json<Vec<PersonaMatch>>, AppError> {
    use crate::chat::skill_hook::{PersonaFileIndex, PersonaMatchSource, SkillActivationHook};
    use std::collections::HashMap;

    // Step 1: Try exact KNOWS + community match via Cypher
    let matches = state
        .orchestrator
        .neo4j()
        .find_personas_for_file(&query.file_path, query.project_id)
        .await
        .map_err(AppError::Internal)?;

    if !matches.is_empty() {
        let result: Vec<PersonaMatch> = matches
            .into_iter()
            .map(|(persona, weight)| PersonaMatch { persona, weight })
            .collect();
        return Ok(Json(result));
    }

    // Step 2: No exact/community match — try directory-prefix fallback
    // Load ALL KNOWS for the project and build dir_index
    let all_knows = state
        .orchestrator
        .neo4j()
        .get_all_persona_knows(query.project_id)
        .await
        .map_err(AppError::Internal)?;

    if all_knows.is_empty() {
        return Ok(Json(vec![]));
    }

    // Build entries map from all KNOWS
    let mut entries: HashMap<String, Vec<(uuid::Uuid, String, f64, PersonaMatchSource)>> =
        HashMap::new();
    for (persona, path, weight) in &all_knows {
        entries.entry(path.clone()).or_default().push((
            persona.id,
            persona.name.clone(),
            *weight,
            PersonaMatchSource::DirectKnows,
        ));
    }

    let dir_index = SkillActivationHook::build_dir_index(&entries);
    let index = PersonaFileIndex {
        entries,
        dir_index,
        loaded_at: std::time::Instant::now(),
    };

    if let Some((persona_id, _name, weight)) =
        SkillActivationHook::directory_prefix_match(&index, &query.file_path, 2)
    {
        // Fetch the full PersonaNode for the matched persona
        if let Ok(Some(persona)) = state.orchestrator.neo4j().get_persona(persona_id).await {
            return Ok(Json(vec![PersonaMatch { persona, weight }]));
        }
    }

    Ok(Json(vec![]))
}

/// A persona match with its relevance weight for a file query.
#[derive(Debug, serde::Serialize)]
pub struct PersonaMatch {
    pub persona: PersonaNode,
    pub weight: f64,
}

// ============================================================================
// Export / Import
// ============================================================================

/// Query params for export
#[derive(Debug, Deserialize)]
pub struct ExportPersonaQuery {
    pub source_project_name: Option<String>,
}

/// Body for importing a PersonaPackage
#[derive(Debug, Deserialize)]
pub struct ImportPersonaBody {
    pub project_id: Uuid,
    pub package: PersonaPackage,
    pub conflict_strategy: Option<String>,
}

/// Body for auto-build
#[derive(Debug, Deserialize)]
pub struct AutoBuildPersonaBody {
    pub project_id: Uuid,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Glob-like pattern to match files (e.g. "src/api/**")
    pub file_pattern: Option<String>,
    /// Entry function to trace call graph from
    pub entry_function: Option<String>,
    /// Depth for call graph traversal (default: 3)
    pub depth: Option<usize>,
}

/// Export a persona as a portable PersonaPackage.
///
/// GET /api/personas/:id/export?source_project_name=...
///
/// Assembles the full subgraph (notes content, decisions content, skill names)
/// into a portable package. Files/functions are project-specific and NOT included.
pub async fn export_persona(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
    Query(query): Query<ExportPersonaQuery>,
) -> Result<Json<PersonaPackage>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    // 1. Load persona
    let persona = neo4j
        .get_persona(persona_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Persona {} not found", persona_id)))?;

    // 2. Load full subgraph
    let subgraph = neo4j
        .get_persona_subgraph(persona_id)
        .await
        .map_err(AppError::Internal)?;

    // 3. Resolve notes → portable format (fetch content + weight)
    let mut portable_notes = Vec::new();
    for rel in &subgraph.notes {
        let note_id = match rel.entity_id.parse::<Uuid>() {
            Ok(id) => id,
            Err(_) => continue,
        };
        if let Ok(Some(note)) = neo4j.get_note(note_id).await {
            portable_notes.push(PortablePersonaNote {
                note_type: note.note_type.to_string(),
                content: note.content,
                importance: note.importance.to_string(),
                tags: note.tags,
                weight: rel.weight,
            });
        }
    }

    // 4. Resolve decisions → portable format (fetch content + weight)
    let mut portable_decisions = Vec::new();
    for rel in &subgraph.decisions {
        let decision_id = match rel.entity_id.parse::<Uuid>() {
            Ok(id) => id,
            Err(_) => continue,
        };
        if let Ok(Some(decision)) = neo4j.get_decision(decision_id).await {
            portable_decisions.push(PortablePersonaDecision {
                description: decision.description,
                rationale: decision.rationale,
                chosen_option: decision.chosen_option.unwrap_or_default(),
                weight: rel.weight,
            });
        }
    }

    // 5. Resolve skill IDs → names for portability
    let mut skill_names = Vec::new();
    for rel in &subgraph.skills {
        if let Ok(skill_uuid) = rel.entity_id.parse::<Uuid>() {
            if let Ok(Some(skill)) = neo4j.get_skill(skill_uuid).await {
                skill_names.push(skill.name);
            }
        }
    }

    // 6. Assemble package
    let package = PersonaPackage {
        schema_version: 1,
        persona: PortablePersona {
            name: persona.name,
            description: persona.description,
            complexity_default: persona.complexity_default,
            timeout_secs: persona.timeout_secs,
            max_cost_usd: persona.max_cost_usd,
            model_preference: persona.model_preference,
            system_prompt_override: persona.system_prompt_override,
            energy: persona.energy,
            cohesion: persona.cohesion,
            activation_count: persona.activation_count,
            success_rate: persona.success_rate,
        },
        notes: portable_notes,
        decisions: portable_decisions,
        skill_names,
        source: Some(PersonaPackageSource {
            project_name: query.source_project_name,
            exported_at: Utc::now(),
        }),
    };

    Ok(Json(package))
}

/// Import a PersonaPackage into a target project.
///
/// POST /api/personas/import
///
/// Creates the persona, re-creates notes/decisions, rebuilds USES relations,
/// and attempts to re-link skills by name.
pub async fn import_persona(
    State(state): State<OrchestratorState>,
    Json(body): Json<ImportPersonaBody>,
) -> Result<(StatusCode, Json<PersonaImportResult>), AppError> {
    let neo4j = state.orchestrator.neo4j();

    let conflict_strategy = body.conflict_strategy.as_deref().unwrap_or("skip");

    // Check for name conflict
    let (existing, _) = neo4j
        .list_personas(body.project_id, None, 1000, 0)
        .await
        .map_err(AppError::Internal)?;

    let existing_match = existing
        .iter()
        .find(|p| p.name == body.package.persona.name);

    if let Some(existing_persona) = existing_match {
        match conflict_strategy {
            "skip" => {
                return Ok((
                    StatusCode::OK,
                    Json(PersonaImportResult {
                        persona_id: existing_persona.id,
                        persona_name: existing_persona.name.clone(),
                        notes_imported: 0,
                        decisions_imported: 0,
                        skills_linked: 0,
                    }),
                ));
            }
            "replace" => {
                // Delete existing, then fall through to create
                let _ = neo4j.delete_persona(existing_persona.id).await;
            }
            // "merge" or unknown → fall through, create with new ID
            _ => {}
        }
    }

    let pp = &body.package.persona;

    // 1. Create persona
    let persona = PersonaNode {
        id: Uuid::new_v4(),
        project_id: Some(body.project_id),
        name: pp.name.clone(),
        description: pp.description.clone(),
        status: PersonaStatus::Emerging,
        complexity_default: pp.complexity_default.clone(),
        timeout_secs: pp.timeout_secs,
        max_cost_usd: pp.max_cost_usd,
        model_preference: pp.model_preference.clone(),
        system_prompt_override: pp.system_prompt_override.clone(),
        energy: pp.energy,
        cohesion: pp.cohesion,
        activation_count: 0, // Reset counters on import
        success_rate: 0.0,
        avg_duration_secs: 0.0,
        last_activated: None,
        energy_boost_accumulated: 0.0,
        energy_history: vec![],
        origin: PersonaOrigin::Imported,
        created_at: Utc::now(),
        updated_at: None,
    };

    neo4j
        .create_persona(&persona)
        .await
        .map_err(AppError::Internal)?;

    // 2. Import notes → create Note + USES relation
    let mut notes_imported = 0;
    for pn in &body.package.notes {
        let note_type = pn
            .note_type
            .parse()
            .unwrap_or(crate::notes::models::NoteType::Observation);
        let importance = pn
            .importance
            .parse()
            .unwrap_or(crate::notes::models::NoteImportance::Medium);

        let mut note = crate::notes::models::Note::new(
            Some(body.project_id),
            note_type,
            pn.content.clone(),
            "persona-import".to_string(),
        );
        note.importance = importance;
        note.tags = pn.tags.clone();

        if neo4j.create_note(&note).await.is_ok() {
            let _ = neo4j.add_persona_note(persona.id, note.id, pn.weight).await;
            notes_imported += 1;
        }
    }

    // 3. Import decisions → create DecisionNode + USES relation
    // Decisions need a task_id — we create them unlinked via a sentinel approach:
    // We create the DecisionNode directly without a task link, then add the USES relation.
    let mut decisions_imported = 0;
    for pd in &body.package.decisions {
        let decision = DecisionNode {
            id: Uuid::new_v4(),
            description: pd.description.clone(),
            rationale: pd.rationale.clone(),
            alternatives: vec![],
            chosen_option: Some(pd.chosen_option.clone()),
            decided_by: "persona-import".to_string(),
            decided_at: Utc::now(),
            status: crate::neo4j::models::DecisionStatus::Accepted,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
        };

        // Create decision with a nil task_id (the trait requires it, but it just creates the node)
        if neo4j.create_decision(Uuid::nil(), &decision).await.is_ok() {
            let _ = neo4j
                .add_persona_decision(persona.id, decision.id, pd.weight)
                .await;
            decisions_imported += 1;
        }
    }

    // 4. Re-link skills by name
    let mut skills_linked = 0;
    let (all_skills, _) = neo4j
        .list_skills(body.project_id, None, 1000, 0)
        .await
        .unwrap_or_default();

    for skill_name in &body.package.skill_names {
        if let Some(skill) = all_skills.iter().find(|s| &s.name == skill_name) {
            if neo4j.add_persona_skill(persona.id, skill.id).await.is_ok() {
                skills_linked += 1;
            }
        }
    }

    Ok((
        StatusCode::CREATED,
        Json(PersonaImportResult {
            persona_id: persona.id,
            persona_name: persona.name,
            notes_imported,
            decisions_imported,
            skills_linked,
        }),
    ))
}

// ============================================================================
// Activate
// ============================================================================

/// Rendered activation context for a persona.
#[derive(Debug, serde::Serialize)]
pub struct PersonaActivation {
    pub persona_id: Uuid,
    pub persona_name: String,
    /// Human-readable rendered context (notes + decisions + system prompt)
    pub rendered_context: String,
    /// Number of notes included
    pub notes_count: usize,
    /// Number of decisions included
    pub decisions_count: usize,
    /// Files this persona knows
    pub files: Vec<String>,
    /// Functions this persona knows
    pub functions: Vec<String>,
}

/// Activate a persona: load its subgraph, render a text prompt context.
///
/// POST /api/personas/:id/activate
///
/// Returns a rendered text block suitable for injection into a system prompt,
/// plus metadata about the persona's scope.
pub async fn activate_persona(
    State(state): State<OrchestratorState>,
    Path(persona_id): Path<Uuid>,
) -> Result<Json<PersonaActivation>, AppError> {
    let neo4j = state.orchestrator.neo4j();

    // 1. Load persona
    let persona = neo4j
        .get_persona(persona_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Persona {} not found", persona_id)))?;

    // 2. Load subgraph
    let subgraph = neo4j
        .get_persona_subgraph(persona_id)
        .await
        .map_err(AppError::Internal)?;

    // 3. Resolve notes content
    let mut note_texts = Vec::new();
    for rel in &subgraph.notes {
        let note_id = match rel.entity_id.parse::<Uuid>() {
            Ok(id) => id,
            Err(_) => continue,
        };
        if let Ok(Some(note)) = neo4j.get_note(note_id).await {
            note_texts.push(format!(
                "[{} | {} | w={:.1}] {}",
                note.note_type, note.importance, rel.weight, note.content
            ));
        }
    }

    // 4. Resolve decisions content
    let mut decision_texts = Vec::new();
    for rel in &subgraph.decisions {
        let decision_id = match rel.entity_id.parse::<Uuid>() {
            Ok(id) => id,
            Err(_) => continue,
        };
        if let Ok(Some(decision)) = neo4j.get_decision(decision_id).await {
            decision_texts.push(format!(
                "[decision | w={:.1}] {} — rationale: {} — chosen: {}",
                rel.weight,
                decision.description,
                decision.rationale,
                decision.chosen_option.as_deref().unwrap_or("n/a")
            ));
        }
    }

    let files: Vec<String> = subgraph.files.iter().map(|r| r.entity_id.clone()).collect();
    let functions: Vec<String> = subgraph
        .functions
        .iter()
        .map(|r| r.entity_id.clone())
        .collect();

    // 5. Render context
    let mut ctx = String::new();
    ctx.push_str(&format!("# Persona: {}\n", persona.name));
    if !persona.description.is_empty() {
        ctx.push_str(&format!("{}\n", persona.description));
    }
    ctx.push('\n');

    if let Some(ref sp) = persona.system_prompt_override {
        ctx.push_str("## System Prompt Override\n");
        ctx.push_str(sp);
        ctx.push_str("\n\n");
    }

    if !files.is_empty() {
        ctx.push_str("## Known Files\n");
        for f in &files {
            ctx.push_str(&format!("- {}\n", f));
        }
        ctx.push('\n');
    }

    if !functions.is_empty() {
        ctx.push_str("## Known Functions\n");
        for f in &functions {
            ctx.push_str(&format!("- {}\n", f));
        }
        ctx.push('\n');
    }

    if !note_texts.is_empty() {
        ctx.push_str("## Notes\n");
        for n in &note_texts {
            ctx.push_str(&format!("- {}\n", n));
        }
        ctx.push('\n');
    }

    if !decision_texts.is_empty() {
        ctx.push_str("## Decisions\n");
        for d in &decision_texts {
            ctx.push_str(&format!("- {}\n", d));
        }
        ctx.push('\n');
    }

    // 6. Update activation metrics
    let mut updated_persona = persona.clone();
    updated_persona.activation_count += 1;
    updated_persona.last_activated = Some(Utc::now());
    updated_persona.updated_at = Some(Utc::now());
    let _ = neo4j.update_persona(&updated_persona).await;

    Ok(Json(PersonaActivation {
        persona_id,
        persona_name: persona.name,
        rendered_context: ctx,
        notes_count: note_texts.len(),
        decisions_count: decision_texts.len(),
        files,
        functions,
    }))
}

// ============================================================================
// Auto-build
// ============================================================================

/// Auto-build a persona from a file pattern or entry function.
///
/// POST /api/personas/auto-build
///
/// Discovers relevant files (via glob match on the code graph) and optionally
/// traces the call graph from an entry function, then creates a persona with
/// KNOWS relations to all matched files/functions.
pub async fn auto_build_persona(
    State(state): State<OrchestratorState>,
    Json(body): Json<AutoBuildPersonaBody>,
) -> Result<(StatusCode, Json<PersonaNode>), AppError> {
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest("name cannot be empty".to_string()));
    }
    if body.file_pattern.is_none() && body.entry_function.is_none() {
        return Err(AppError::BadRequest(
            "at least one of file_pattern or entry_function is required".to_string(),
        ));
    }

    let neo4j = state.orchestrator.neo4j();
    let depth = body.depth.unwrap_or(3);

    // 1. Create the persona
    let persona = PersonaNode {
        id: Uuid::new_v4(),
        project_id: Some(body.project_id),
        name: body.name.clone(),
        description: if body.description.is_empty() {
            format!(
                "Auto-built persona{}{}",
                body.file_pattern
                    .as_deref()
                    .map(|p| format!(" from pattern '{}'", p))
                    .unwrap_or_default(),
                body.entry_function
                    .as_deref()
                    .map(|f| format!(" + entry '{}'", f))
                    .unwrap_or_default(),
            )
        } else {
            body.description.clone()
        },
        status: PersonaStatus::Emerging,
        complexity_default: None,
        timeout_secs: None,
        max_cost_usd: None,
        model_preference: None,
        system_prompt_override: None,
        energy: 0.5,
        cohesion: 0.0,
        activation_count: 0,
        success_rate: 0.0,
        avg_duration_secs: 0.0,
        last_activated: None,
        energy_boost_accumulated: 0.0,
        energy_history: vec![],
        origin: PersonaOrigin::AutoBuild,
        created_at: Utc::now(),
        updated_at: None,
    };

    neo4j
        .create_persona(&persona)
        .await
        .map_err(AppError::Internal)?;

    // 2. Match files by pattern (prefix/contains on project File nodes)
    if let Some(ref pattern) = body.file_pattern {
        let all_files = neo4j
            .get_project_file_paths(body.project_id)
            .await
            .unwrap_or_default();

        // Simple glob-like matching: if pattern contains '*', split on it;
        // otherwise treat as substring match.
        let pattern_lower = pattern.to_lowercase();
        let matched: Vec<&String> = if pattern.contains('*') {
            let parts: Vec<&str> = pattern_lower.split('*').collect();
            all_files
                .iter()
                .filter(|f| {
                    let fl = f.to_lowercase();
                    parts
                        .iter()
                        .all(|part| part.is_empty() || fl.contains(part))
                })
                .collect()
        } else {
            all_files
                .iter()
                .filter(|f| f.to_lowercase().contains(&pattern_lower))
                .collect()
        };

        for file_path in matched {
            let _ = neo4j.add_persona_file(persona.id, file_path, 1.0).await;
        }
    }

    // 3. Trace call graph from entry function
    if let Some(ref entry_fn) = body.entry_function {
        if let Ok(callees) = neo4j.get_callees(entry_fn, depth as u32).await {
            for func in &callees {
                let _ = neo4j
                    .add_persona_function(persona.id, &func.name, 0.8)
                    .await;
            }
            // Also add the entry function itself
            let _ = neo4j.add_persona_function(persona.id, entry_fn, 1.0).await;
        }
    }

    Ok((StatusCode::CREATED, Json(persona)))
}

// ============================================================================
// Maintenance & Detection
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ProjectIdQuery {
    pub project_id: Uuid,
}

/// Maintain personas: decay weights, prune dead relations, recalculate cohesion
/// POST /api/personas/maintain?project_id=<uuid>
pub async fn maintain_personas(
    State(state): State<OrchestratorState>,
    Query(query): Query<ProjectIdQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let (decayed, pruned, updated) = neo4j
        .maintain_personas(query.project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!({
        "decayed": decayed,
        "pruned": pruned,
        "cohesion_updated": updated,
        "project_id": query.project_id,
    })))
}

/// Detect persona candidates from code communities (Louvain)
/// POST /api/personas/detect?project_id=<uuid>
pub async fn detect_personas(
    State(state): State<OrchestratorState>,
    Query(query): Query<ProjectIdQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let proposals = neo4j
        .detect_personas(query.project_id)
        .await
        .map_err(AppError::Internal)?;

    let proposals_json: Vec<serde_json::Value> = proposals
        .iter()
        .map(|p| {
            serde_json::json!({
                "suggested_name": p.suggested_name,
                "sample_files": p.sample_files,
                "file_count": p.file_count,
                "community_id": p.community_id,
                "confidence": p.confidence,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "proposals": proposals_json,
        "count": proposals.len(),
        "project_id": query.project_id,
    })))
}

// ============================================================================
// Learning Health
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct LearningHealthQuery {
    pub project_id: Option<Uuid>,
    pub project_slug: Option<String>,
}

/// Get learning health metrics for a project's persona system.
///
/// GET /api/personas/learning-health?project_id=...
/// GET /api/personas/learning-health?project_slug=...
pub async fn get_learning_health(
    State(state): State<OrchestratorState>,
    Query(query): Query<LearningHealthQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let project_id = match (query.project_id, &query.project_slug) {
        (Some(id), _) => id,
        (None, Some(slug)) => {
            let project = state
                .orchestrator
                .neo4j()
                .get_project_by_slug(slug)
                .await
                .map_err(AppError::Internal)?
                .ok_or_else(|| AppError::NotFound(format!("Project not found: {slug}")))?;
            project.id
        }
        (None, None) => {
            return Err(AppError::BadRequest(
                "Either project_id or project_slug is required".to_string(),
            ));
        }
    };

    let report = state
        .orchestrator
        .neo4j()
        .get_learning_health(project_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(serde_json::json!(report)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // Unit tests (deserialization / parsing)
    // ================================================================

    #[test]
    fn test_default_weight() {
        assert!((default_weight() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_persona_origin_parse() {
        assert_eq!(
            "manual".parse::<PersonaOrigin>().unwrap(),
            PersonaOrigin::Manual
        );
        assert_eq!(
            "auto_build".parse::<PersonaOrigin>().unwrap(),
            PersonaOrigin::AutoBuild
        );
        assert_eq!(
            "imported".parse::<PersonaOrigin>().unwrap(),
            PersonaOrigin::Imported
        );
        assert!("unknown".parse::<PersonaOrigin>().is_err());
    }

    #[test]
    fn test_persona_status_parse() {
        assert_eq!(
            "active".parse::<PersonaStatus>().unwrap(),
            PersonaStatus::Active
        );
        assert_eq!(
            "dormant".parse::<PersonaStatus>().unwrap(),
            PersonaStatus::Dormant
        );
        assert_eq!(
            "emerging".parse::<PersonaStatus>().unwrap(),
            PersonaStatus::Emerging
        );
        assert_eq!(
            "archived".parse::<PersonaStatus>().unwrap(),
            PersonaStatus::Archived
        );
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
    fn test_create_body_all_fields() {
        let json = serde_json::json!({
            "project_id": "00000000-0000-0000-0000-000000000001",
            "name": "full-persona",
            "description": "Full description",
            "complexity_default": "complex",
            "timeout_secs": 300,
            "max_cost_usd": 5.0,
            "model_preference": "opus",
            "system_prompt_override": "You are an expert.",
            "origin": "auto_build"
        });
        let body: CreatePersonaBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.name, "full-persona");
        assert_eq!(body.complexity_default.as_deref(), Some("complex"));
        assert_eq!(body.timeout_secs, Some(300));
        assert_eq!(body.max_cost_usd, Some(5.0));
        assert_eq!(body.model_preference.as_deref(), Some("opus"));
        assert_eq!(
            body.system_prompt_override.as_deref(),
            Some("You are an expert.")
        );
        assert_eq!(body.origin.as_deref(), Some("auto_build"));
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
    fn test_update_body_all_fields() {
        let json = serde_json::json!({
            "name": "Updated",
            "description": "New desc",
            "status": "dormant",
            "complexity_default": "simple",
            "timeout_secs": 120,
            "max_cost_usd": 2.5,
            "model_preference": "sonnet",
            "system_prompt_override": "Override",
            "energy": 0.9,
            "cohesion": 0.7
        });
        let body: UpdatePersonaBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.name.as_deref(), Some("Updated"));
        assert_eq!(body.description.as_deref(), Some("New desc"));
        assert_eq!(body.status.as_deref(), Some("dormant"));
        assert_eq!(body.complexity_default.as_deref(), Some("simple"));
        assert_eq!(body.timeout_secs, Some(120));
        assert_eq!(body.max_cost_usd, Some(2.5));
        assert_eq!(body.model_preference.as_deref(), Some("sonnet"));
        assert_eq!(body.system_prompt_override.as_deref(), Some("Override"));
        assert_eq!(body.energy, Some(0.9));
        assert_eq!(body.cohesion, Some(0.7));
    }

    #[test]
    fn test_update_body_empty() {
        let json = serde_json::json!({});
        let body: UpdatePersonaBody = serde_json::from_value(json).unwrap();
        assert!(body.name.is_none());
        assert!(body.description.is_none());
        assert!(body.status.is_none());
        assert!(body.energy.is_none());
        assert!(body.cohesion.is_none());
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
    fn test_add_file_body_custom_weight() {
        let json = serde_json::json!({
            "file_path": "src/lib.rs",
            "weight": 0.75
        });
        let body: AddFileBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.file_path, "src/lib.rs");
        assert!((body.weight - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remove_file_body() {
        let json = serde_json::json!({
            "file_path": "src/old.rs"
        });
        let body: RemoveFileBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.file_path, "src/old.rs");
    }

    #[test]
    fn test_add_function_body_default_weight() {
        let json = serde_json::json!({
            "function_name": "my_func"
        });
        let body: AddFunctionBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.function_name, "my_func");
        assert!((body.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_function_body_custom_weight() {
        let json = serde_json::json!({
            "function_name": "my_func",
            "weight": 0.5
        });
        let body: AddFunctionBody = serde_json::from_value(json).unwrap();
        assert!((body.weight - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remove_function_body() {
        let json = serde_json::json!({
            "function_name": "old_func"
        });
        let body: RemoveFunctionBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.function_name, "old_func");
    }

    #[test]
    fn test_add_weighted_body_default() {
        let json = serde_json::json!({});
        let body: AddWeightedBody = serde_json::from_value(json).unwrap();
        assert!((body.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_weighted_body_custom() {
        let json = serde_json::json!({"weight": 0.3});
        let body: AddWeightedBody = serde_json::from_value(json).unwrap();
        assert!((body.weight - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_for_file_query() {
        let json = serde_json::json!({
            "file_path": "src/api/mod.rs",
            "project_id": "00000000-0000-0000-0000-000000000001"
        });
        let query: FindForFileQuery = serde_json::from_value(json).unwrap();
        assert_eq!(query.file_path, "src/api/mod.rs");
    }

    #[test]
    fn test_export_persona_query() {
        let json = serde_json::json!({
            "source_project_name": "my-project"
        });
        let query: ExportPersonaQuery = serde_json::from_value(json).unwrap();
        assert_eq!(query.source_project_name.as_deref(), Some("my-project"));
    }

    #[test]
    fn test_export_persona_query_empty() {
        let json = serde_json::json!({});
        let query: ExportPersonaQuery = serde_json::from_value(json).unwrap();
        assert!(query.source_project_name.is_none());
    }

    #[test]
    fn test_import_persona_body_deserialization() {
        let json = serde_json::json!({
            "project_id": "00000000-0000-0000-0000-000000000001",
            "package": {
                "schema_version": 1,
                "persona": {
                    "name": "imported-persona",
                    "description": "Imported desc",
                    "energy": 0.5,
                    "cohesion": 0.3,
                    "activation_count": 10,
                    "success_rate": 0.9
                },
                "notes": [],
                "decisions": [],
                "skill_names": ["neo4j-expert"]
            },
            "conflict_strategy": "skip"
        });
        let body: ImportPersonaBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.package.persona.name, "imported-persona");
        assert_eq!(body.conflict_strategy.as_deref(), Some("skip"));
        assert_eq!(body.package.skill_names, vec!["neo4j-expert"]);
    }

    #[test]
    fn test_auto_build_body_deserialization() {
        let json = serde_json::json!({
            "project_id": "00000000-0000-0000-0000-000000000001",
            "name": "api-specialist",
            "file_pattern": "src/api/**",
            "entry_function": "main",
            "depth": 5
        });
        let body: AutoBuildPersonaBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.name, "api-specialist");
        assert_eq!(body.file_pattern.as_deref(), Some("src/api/**"));
        assert_eq!(body.entry_function.as_deref(), Some("main"));
        assert_eq!(body.depth, Some(5));
    }

    #[test]
    fn test_auto_build_body_minimal() {
        let json = serde_json::json!({
            "project_id": "00000000-0000-0000-0000-000000000001",
            "name": "minimal"
        });
        let body: AutoBuildPersonaBody = serde_json::from_value(json).unwrap();
        assert!(body.file_pattern.is_none());
        assert!(body.entry_function.is_none());
        assert!(body.depth.is_none());
    }

    #[test]
    fn test_project_id_query() {
        let json = serde_json::json!({
            "project_id": "00000000-0000-0000-0000-000000000001"
        });
        let query: ProjectIdQuery = serde_json::from_value(json).unwrap();
        assert_eq!(
            query.project_id,
            "00000000-0000-0000-0000-000000000001"
                .parse::<Uuid>()
                .unwrap()
        );
    }

    #[test]
    fn test_global_persona_list_query() {
        let json = serde_json::json!({});
        let _query: GlobalPersonaListQuery = serde_json::from_value(json).unwrap();
    }

    // ================================================================
    // Async integration tests (mock backends)
    // ================================================================

    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_bearer_token, test_project};
    use axum::body::Body;
    use axum::http::{Request, StatusCode as HttpStatus};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn auth_post_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    fn auth_put_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("PUT")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    fn auth_delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn auth_delete_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

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
            trajectory_collector: None,
            trajectory_store: None,
            identity: None,
        });
        create_router(state)
    }

    async fn test_app_with_project() -> (axum::Router, uuid::Uuid) {
        let app_state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        app_state.neo4j.create_project(&project).await.unwrap();
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
            trajectory_collector: None,
            trajectory_store: None,
            identity: None,
        });
        (create_router(state), project_id)
    }

    /// Build an app with a project and a pre-seeded persona, returning
    /// (router, project_id, persona_id).
    async fn test_app_with_persona() -> (axum::Router, uuid::Uuid, uuid::Uuid) {
        let app_state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        app_state.neo4j.create_project(&project).await.unwrap();

        let persona = PersonaNode {
            id: Uuid::new_v4(),
            project_id: Some(project_id),
            name: "Test Persona".to_string(),
            description: "A test persona".to_string(),
            status: PersonaStatus::Emerging,
            complexity_default: None,
            timeout_secs: None,
            max_cost_usd: None,
            model_preference: None,
            system_prompt_override: None,
            energy: 0.5,
            cohesion: 0.0,
            activation_count: 0,
            success_rate: 0.0,
            avg_duration_secs: 0.0,
            last_activated: None,
            energy_boost_accumulated: 0.0,
            energy_history: vec![],
            origin: PersonaOrigin::Manual,
            created_at: Utc::now(),
            updated_at: None,
        };
        let persona_id = persona.id;
        app_state.neo4j.create_persona(&persona).await.unwrap();

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
            trajectory_collector: None,
            trajectory_store: None,
            identity: None,
        });
        (create_router(state), project_id, persona_id)
    }

    // ----------------------------------------------------------------
    // CRUD: create_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_create_persona() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "Neo4j Expert",
                    "description": "Knows all about Neo4j"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::CREATED);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "Neo4j Expert");
        assert_eq!(json["description"], "Knows all about Neo4j");
        assert_eq!(json["status"], "emerging");
        assert_eq!(json["origin"], "manual");
    }

    #[tokio::test]
    async fn test_create_persona_with_origin() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "Auto Persona",
                    "origin": "auto_build"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::CREATED);
        let json = body_json(resp).await;
        assert_eq!(json["origin"], "auto_build");
    }

    #[tokio::test]
    async fn test_create_persona_global_no_project() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "name": "Global Persona",
                    "description": "Available everywhere"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::CREATED);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "Global Persona");
        assert!(json["project_id"].is_null());
    }

    #[tokio::test]
    async fn test_create_persona_empty_name_fails() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": ""
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_persona_whitespace_name_fails() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "   "
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_persona_name_too_long_fails() {
        let (app, project_id) = test_app_with_project().await;
        let long_name = "a".repeat(MAX_NAME_LEN + 1);
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": long_name
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_persona_description_too_long_fails() {
        let (app, project_id) = test_app_with_project().await;
        let long_desc = "a".repeat(MAX_DESCRIPTION_LEN + 1);
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "Valid Name",
                    "description": long_desc
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_persona_system_prompt_too_long_fails() {
        let (app, project_id) = test_app_with_project().await;
        let long_sp = "a".repeat(MAX_SYSTEM_PROMPT_LEN + 1);
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "Valid Name",
                    "system_prompt_override": long_sp
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_persona_invalid_origin_fails() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "Valid",
                    "origin": "bogus_origin"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // CRUD: get_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_persona() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_get(&format!("/api/personas/{}", persona_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "Test Persona");
        assert_eq!(json["id"], persona_id.to_string());
    }

    #[tokio::test]
    async fn test_get_persona_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_get(&format!("/api/personas/{}", fake_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }

    // ----------------------------------------------------------------
    // CRUD: list_personas
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_list_personas_empty() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas?project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total"], 0);
        assert_eq!(json["items"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_list_personas_with_results() {
        let (app, project_id, _persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas?project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total"], 1);
        assert_eq!(json["items"].as_array().unwrap().len(), 1);
        assert_eq!(json["items"][0]["name"], "Test Persona");
    }

    // ----------------------------------------------------------------
    // CRUD: update_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_update_persona() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", persona_id),
                serde_json::json!({
                    "name": "Updated Persona",
                    "description": "Updated description",
                    "status": "active",
                    "energy": 0.9,
                    "cohesion": 0.7
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "Updated Persona");
        assert_eq!(json["description"], "Updated description");
        assert_eq!(json["status"], "active");
    }

    #[tokio::test]
    async fn test_update_persona_partial() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", persona_id),
                serde_json::json!({
                    "energy": 0.8
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        // Name should be unchanged
        assert_eq!(json["name"], "Test Persona");
    }

    #[tokio::test]
    async fn test_update_persona_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", fake_id),
                serde_json::json!({"name": "Nope"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_update_persona_empty_name_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", persona_id),
                serde_json::json!({"name": ""}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_persona_name_too_long_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let long_name = "a".repeat(MAX_NAME_LEN + 1);
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", persona_id),
                serde_json::json!({"name": long_name}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_persona_description_too_long_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let long_desc = "a".repeat(MAX_DESCRIPTION_LEN + 1);
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", persona_id),
                serde_json::json!({"description": long_desc}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_persona_system_prompt_too_long_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let long_sp = "a".repeat(MAX_SYSTEM_PROMPT_LEN + 1);
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", persona_id),
                serde_json::json!({"system_prompt_override": long_sp}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_update_persona_invalid_status_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_put_json(
                &format!("/api/personas/{}", persona_id),
                serde_json::json!({"status": "nonexistent"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    // ----------------------------------------------------------------
    // CRUD: delete_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_delete_persona() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_delete(&format!("/api/personas/{}", persona_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_delete_persona_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_delete(&format!("/api/personas/{}", fake_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }

    // ----------------------------------------------------------------
    // Relations: add/remove skill
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_skill_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let skill_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/skills/{}", persona_id, skill_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_remove_skill_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let skill_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_delete(&format!(
                "/api/personas/{}/skills/{}",
                persona_id, skill_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Relations: add/remove protocol
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_protocol_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let protocol_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/protocols/{}", persona_id, protocol_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_remove_protocol_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let protocol_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_delete(&format!(
                "/api/personas/{}/protocols/{}",
                persona_id, protocol_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Relations: scope/unscope feature graph
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_scope_feature_graph() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let fg_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/feature-graphs/{}", persona_id, fg_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_unscope_feature_graph() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let fg_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_delete(&format!(
                "/api/personas/{}/feature-graphs/{}",
                persona_id, fg_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Relations: add/remove file
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_file() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/files", persona_id),
                serde_json::json!({
                    "file_path": "src/main.rs",
                    "weight": 0.8
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_add_file_empty_path_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/files", persona_id),
                serde_json::json!({
                    "file_path": "  "
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_remove_file() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_delete_json(
                &format!("/api/personas/{}/files", persona_id),
                serde_json::json!({
                    "file_path": "src/main.rs"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Relations: add/remove function
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_function() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/functions", persona_id),
                serde_json::json!({
                    "function_name": "handle_request",
                    "weight": 0.9
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_add_function_empty_name_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/functions", persona_id),
                serde_json::json!({
                    "function_name": "  "
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_remove_function() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_delete_json(
                &format!("/api/personas/{}/functions", persona_id),
                serde_json::json!({
                    "function_name": "handle_request"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Relations: add/remove note
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_note_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let note_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/notes/{}", persona_id, note_id),
                serde_json::json!({"weight": 0.7}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_add_note_relation_default_weight() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let note_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/notes/{}", persona_id, note_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_remove_note_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let note_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_delete(&format!(
                "/api/personas/{}/notes/{}",
                persona_id, note_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Relations: add/remove decision
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_decision_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let decision_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/decisions/{}", persona_id, decision_id),
                serde_json::json!({"weight": 0.6}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_remove_decision_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let decision_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_delete(&format!(
                "/api/personas/{}/decisions/{}",
                persona_id, decision_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Relations: add/remove extends
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_add_extends_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let parent_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/extends/{}", persona_id, parent_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_add_extends_self_fails() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/extends/{}", persona_id, persona_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_remove_extends_relation() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let parent_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_delete(&format!(
                "/api/personas/{}/extends/{}",
                persona_id, parent_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NO_CONTENT);
    }

    // ----------------------------------------------------------------
    // Subgraph
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_subgraph_empty() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_get(&format!("/api/personas/{}/subgraph", persona_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["persona_id"], persona_id.to_string());
        assert_eq!(json["persona_name"], "Test Persona");
        assert_eq!(json["files"].as_array().unwrap().len(), 0);
        assert_eq!(json["functions"].as_array().unwrap().len(), 0);
        assert_eq!(json["notes"].as_array().unwrap().len(), 0);
        assert_eq!(json["decisions"].as_array().unwrap().len(), 0);
        assert_eq!(json["skills"].as_array().unwrap().len(), 0);
        assert_eq!(json["protocols"].as_array().unwrap().len(), 0);
    }

    // ----------------------------------------------------------------
    // find_for_file
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_find_for_file_empty() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/find-for-file?file_path=src/main.rs&project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    // ----------------------------------------------------------------
    // list_global_personas
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_list_global_personas_empty() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/personas/global")).await.unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    // ----------------------------------------------------------------
    // activate_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_activate_persona() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/activate", persona_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["persona_id"], persona_id.to_string());
        assert_eq!(json["persona_name"], "Test Persona");
        assert!(json["rendered_context"]
            .as_str()
            .unwrap()
            .contains("Test Persona"));
        assert_eq!(json["notes_count"], 0);
        assert_eq!(json["decisions_count"], 0);
    }

    #[tokio::test]
    async fn test_activate_persona_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/{}/activate", fake_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }

    // ----------------------------------------------------------------
    // export_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_export_persona() {
        let (app, _project_id, persona_id) = test_app_with_persona().await;
        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/{}/export?source_project_name=test-project",
                persona_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["persona"]["name"], "Test Persona");
        assert_eq!(json["source"]["project_name"], "test-project");
        // notes, decisions, skill_names are skipped when empty (skip_serializing_if)
        assert!(json.get("notes").is_none() || json["notes"].as_array().unwrap().is_empty());
        assert!(
            json.get("decisions").is_none() || json["decisions"].as_array().unwrap().is_empty()
        );
        assert!(
            json.get("skill_names").is_none() || json["skill_names"].as_array().unwrap().is_empty()
        );
    }

    #[tokio::test]
    async fn test_export_persona_not_found() {
        let app = test_app().await;
        let fake_id = Uuid::new_v4();
        let resp = app
            .oneshot(auth_get(&format!("/api/personas/{}/export", fake_id)))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }

    // ----------------------------------------------------------------
    // import_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_import_persona() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas/import",
                serde_json::json!({
                    "project_id": project_id,
                    "package": {
                        "schema_version": 1,
                        "persona": {
                            "name": "Imported Expert",
                            "description": "Imported from another project",
                            "energy": 0.7,
                            "cohesion": 0.5,
                            "activation_count": 42,
                            "success_rate": 0.95
                        },
                        "notes": [],
                        "decisions": [],
                        "skill_names": []
                    }
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::CREATED);
        let json = body_json(resp).await;
        assert_eq!(json["persona_name"], "Imported Expert");
        assert_eq!(json["notes_imported"], 0);
        assert_eq!(json["decisions_imported"], 0);
        assert_eq!(json["skills_linked"], 0);
    }

    #[tokio::test]
    async fn test_import_persona_skip_conflict() {
        let (app, project_id, _persona_id) = test_app_with_persona().await;
        // Import with the same name "Test Persona" — should skip
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas/import",
                serde_json::json!({
                    "project_id": project_id,
                    "package": {
                        "schema_version": 1,
                        "persona": {
                            "name": "Test Persona",
                            "description": "Duplicate",
                            "energy": 0.5,
                            "cohesion": 0.0,
                            "activation_count": 0,
                            "success_rate": 0.0
                        },
                        "notes": [],
                        "decisions": [],
                        "skill_names": []
                    },
                    "conflict_strategy": "skip"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["persona_name"], "Test Persona");
        assert_eq!(json["notes_imported"], 0);
    }

    // ----------------------------------------------------------------
    // auto_build_persona
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_auto_build_persona_empty_name_fails() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas/auto-build",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "",
                    "file_pattern": "src/**"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_auto_build_persona_no_pattern_no_function_fails() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas/auto-build",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "Empty Builder"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_auto_build_persona_with_pattern() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                "/api/personas/auto-build",
                serde_json::json!({
                    "project_id": project_id,
                    "name": "API Expert",
                    "description": "Auto-built from api pattern",
                    "file_pattern": "src/api/**"
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::CREATED);
        let json = body_json(resp).await;
        assert_eq!(json["name"], "API Expert");
        assert_eq!(json["origin"], "auto_build");
    }

    // ----------------------------------------------------------------
    // maintain_personas
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_maintain_personas() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/maintain?project_id={}", project_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["project_id"], project_id.to_string());
    }

    // ----------------------------------------------------------------
    // detect_personas
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_detect_personas() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_post_json(
                &format!("/api/personas/detect?project_id={}", project_id),
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["project_id"], project_id.to_string());
        assert!(json["proposals"].as_array().unwrap().is_empty());
        assert_eq!(json["count"], 0);
    }

    // ----------------------------------------------------------------
    // find_for_file — fallback chain integration tests
    // ----------------------------------------------------------------

    /// Helper: build app with persona + KNOWS files seeded via mock
    async fn test_app_with_persona_files(
        files: Vec<(&str, f64)>,
    ) -> (axum::Router, uuid::Uuid, uuid::Uuid) {
        let app_state = mock_app_state();
        let project = test_project();
        let project_id = project.id;
        app_state.neo4j.create_project(&project).await.unwrap();

        let persona = PersonaNode {
            id: Uuid::new_v4(),
            project_id: Some(project_id),
            name: "Neo4j Expert".to_string(),
            description: "Knows neo4j files".to_string(),
            status: PersonaStatus::Active,
            complexity_default: None,
            timeout_secs: None,
            max_cost_usd: None,
            model_preference: None,
            system_prompt_override: None,
            energy: 0.8,
            cohesion: 0.5,
            activation_count: 10,
            success_rate: 0.9,
            avg_duration_secs: 30.0,
            last_activated: None,
            energy_boost_accumulated: 0.0,
            energy_history: vec![],
            origin: PersonaOrigin::Manual,
            created_at: Utc::now(),
            updated_at: None,
        };
        let persona_id = persona.id;
        app_state.neo4j.create_persona(&persona).await.unwrap();

        for (path, weight) in files {
            app_state
                .neo4j
                .add_persona_file(persona_id, path, weight)
                .await
                .unwrap();
        }

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
            trajectory_collector: None,
            trajectory_store: None,
            identity: None,
        });
        (create_router(state), project_id, persona_id)
    }

    #[tokio::test]
    async fn test_find_for_file_exact_knows_match() {
        // Persona KNOWS exactly "src/neo4j/persona.rs" → exact match
        let (app, project_id, _persona_id) = test_app_with_persona_files(vec![
            ("src/neo4j/persona.rs", 0.95),
            ("src/neo4j/client.rs", 0.8),
        ])
        .await;

        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/find-for-file?file_path=src/neo4j/persona.rs&project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        let matches = json.as_array().unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["persona"]["name"], "Neo4j Expert");
        let weight = matches[0]["weight"].as_f64().unwrap();
        assert!(
            (weight - 0.95).abs() < 0.01,
            "expected exact weight 0.95, got {}",
            weight
        );
    }

    #[tokio::test]
    async fn test_find_for_file_directory_prefix_fallback() {
        // Persona KNOWS 2 files in src/neo4j/ but NOT src/neo4j/analytics.rs
        // → directory-prefix match should kick in
        let (app, project_id, _persona_id) = test_app_with_persona_files(vec![
            ("src/neo4j/persona.rs", 0.9),
            ("src/neo4j/client.rs", 0.8),
        ])
        .await;

        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/find-for-file?file_path=src/neo4j/analytics.rs&project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        let matches = json.as_array().unwrap();
        assert_eq!(matches.len(), 1, "expected dir-prefix fallback match");
        assert_eq!(matches[0]["persona"]["name"], "Neo4j Expert");
        // Weight should be avg(0.9, 0.8) * 0.7 = 0.595
        let weight = matches[0]["weight"].as_f64().unwrap();
        assert!(
            weight > 0.5 && weight < 0.7,
            "expected dir-prefix weight ~0.595, got {}",
            weight
        );
    }

    #[tokio::test]
    async fn test_find_for_file_no_match() {
        // Persona KNOWS files in src/neo4j/ only
        // Query for a completely different directory → no match
        let (app, project_id, _persona_id) = test_app_with_persona_files(vec![
            ("src/neo4j/persona.rs", 0.9),
            ("src/neo4j/client.rs", 0.8),
        ])
        .await;

        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/find-for-file?file_path=src/api/routes.rs&project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_find_for_file_dir_prefix_needs_min_2_files() {
        // Persona KNOWS only 1 file in src/api/ → dir-prefix needs >= 2 files
        // Should NOT match via dir-prefix
        let (app, project_id, _persona_id) =
            test_app_with_persona_files(vec![("src/api/routes.rs", 0.9)]).await;

        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/find-for-file?file_path=src/api/handlers.rs&project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(
            json.as_array().unwrap().len(),
            0,
            "should not match with only 1 file in directory"
        );
    }

    #[tokio::test]
    async fn test_find_for_file_exact_takes_priority_over_dir_prefix() {
        // Persona KNOWS both the exact file AND other files in the same dir
        // Exact match should be returned (higher weight)
        let (app, project_id, _persona_id) = test_app_with_persona_files(vec![
            ("src/neo4j/persona.rs", 0.95),
            ("src/neo4j/client.rs", 0.8),
            ("src/neo4j/mock.rs", 0.7),
        ])
        .await;

        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/find-for-file?file_path=src/neo4j/persona.rs&project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        let matches = json.as_array().unwrap();
        assert_eq!(matches.len(), 1);
        // Should be exact weight (0.95), not dir-prefix weight
        let weight = matches[0]["weight"].as_f64().unwrap();
        assert!(
            (weight - 0.95).abs() < 0.01,
            "exact match should take priority, got weight {}",
            weight
        );
    }

    // ----------------------------------------------------------------
    // Learning Health
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_get_learning_health_with_project_id() {
        let (app, project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_get(&format!(
                "/api/personas/learning-health?project_id={}",
                project_id
            )))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["persona_count"], 0);
        assert_eq!(json["knows_convergence"], 0.0);
    }

    #[tokio::test]
    async fn test_get_learning_health_with_slug() {
        let (app, _project_id) = test_app_with_project().await;
        let resp = app
            .oneshot(auth_get(
                "/api/personas/learning-health?project_slug=test-project",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::OK);
        let json = body_json(resp).await;
        assert_eq!(json["persona_count"], 0);
    }

    #[tokio::test]
    async fn test_get_learning_health_missing_params() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/personas/learning-health"))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_get_learning_health_slug_not_found() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get(
                "/api/personas/learning-health?project_slug=nonexistent",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), HttpStatus::NOT_FOUND);
    }
}
