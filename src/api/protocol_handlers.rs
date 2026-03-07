//! API handlers for Protocol (Pattern Federation)

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams};
use crate::events::EventEmitter;
use crate::protocol::{
    self, Protocol, ProtocolCategory, ProtocolRun, ProtocolState, ProtocolTransition, RunStatus,
    TransitionResult,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Validation constants
// ============================================================================

const MAX_NAME_LEN: usize = 200;
const MAX_DESCRIPTION_LEN: usize = 5000;
const MAX_TRIGGER_LEN: usize = 500;

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for listing protocols
#[derive(Debug, Deserialize, Default)]
pub struct ProtocolsListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    /// Required: filter by project
    pub project_id: Uuid,
    /// Optional: filter by category (system, business)
    pub category: Option<String>,
}

// ============================================================================
// Request Bodies
// ============================================================================

/// Request body for creating a protocol
#[derive(Debug, Deserialize)]
pub struct CreateProtocolBody {
    pub project_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub skill_id: Option<Uuid>,
    pub protocol_category: Option<String>,
    /// Optional inline states to create with the protocol
    pub states: Option<Vec<CreateStateInline>>,
    /// Optional inline transitions to create with the protocol
    pub transitions: Option<Vec<CreateTransitionInline>>,
}

/// Inline state definition for protocol creation
#[derive(Debug, Deserialize)]
pub struct CreateStateInline {
    pub name: String,
    pub description: Option<String>,
    pub state_type: Option<String>,
    pub action: Option<String>,
}

/// Inline transition definition for protocol creation
#[derive(Debug, Deserialize)]
pub struct CreateTransitionInline {
    pub from_state: Uuid,
    pub to_state: Uuid,
    pub trigger: String,
    pub guard: Option<String>,
}

/// Request body for updating a protocol
#[derive(Debug, Deserialize)]
pub struct UpdateProtocolBody {
    pub name: Option<String>,
    pub description: Option<String>,
    pub protocol_category: Option<String>,
}

/// Request body for adding a state
#[derive(Debug, Deserialize)]
pub struct AddStateBody {
    pub name: String,
    pub description: Option<String>,
    pub state_type: Option<String>,
    pub action: Option<String>,
}

/// Request body for adding a transition
#[derive(Debug, Deserialize)]
pub struct AddTransitionBody {
    pub from_state: Uuid,
    pub to_state: Uuid,
    pub trigger: String,
    pub guard: Option<String>,
}

/// Request body for linking a protocol to a skill
#[derive(Debug, Deserialize)]
pub struct LinkToSkillBody {
    pub skill_id: Uuid,
}

/// Request body to start a protocol run
#[derive(Debug, Deserialize)]
pub struct StartRunBody {
    pub plan_id: Option<Uuid>,
    pub task_id: Option<Uuid>,
}

/// Request body to fire a transition
#[derive(Debug, Deserialize)]
pub struct FireTransitionBody {
    pub trigger: String,
}

/// Query parameters for listing protocol runs
#[derive(Debug, Deserialize, Default)]
pub struct RunsListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    pub status: Option<String>,
}

// ============================================================================
// Response types
// ============================================================================

/// Full protocol response with states and transitions
#[derive(Debug, Serialize)]
pub struct ProtocolDetail {
    #[serde(flatten)]
    pub protocol: Protocol,
    pub states: Vec<ProtocolState>,
    pub transitions: Vec<ProtocolTransition>,
}

// ============================================================================
// Handlers — CRUD
// ============================================================================

/// List protocols for a project
///
/// GET /api/protocols?project_id=...&category=...&limit=...&offset=...
pub async fn list_protocols(
    State(state): State<OrchestratorState>,
    Query(query): Query<ProtocolsListQuery>,
) -> Result<Json<PaginatedResponse<Protocol>>, AppError> {
    query.pagination.validate().map_err(AppError::BadRequest)?;

    let category_filter = match &query.category {
        Some(s) => Some(
            s.parse::<ProtocolCategory>()
                .map_err(|e| AppError::BadRequest(e.to_string()))?,
        ),
        None => None,
    };

    let limit = query.pagination.validated_limit();
    let offset = query.pagination.offset;

    let (protocols, total) = state
        .orchestrator
        .neo4j()
        .list_protocols(query.project_id, category_filter, limit, offset)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(PaginatedResponse::new(protocols, total, limit, offset)))
}

/// Create a new protocol (optionally with inline states and transitions)
///
/// POST /api/protocols
pub async fn create_protocol(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateProtocolBody>,
) -> Result<(StatusCode, Json<ProtocolDetail>), AppError> {
    // Validate
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
    if let Some(ref desc) = body.description {
        if desc.len() > MAX_DESCRIPTION_LEN {
            return Err(AppError::BadRequest(format!(
                "description too long ({} > {} chars)",
                desc.len(),
                MAX_DESCRIPTION_LEN
            )));
        }
    }

    // Parse category
    let category = match &body.protocol_category {
        Some(s) => s
            .parse::<ProtocolCategory>()
            .map_err(|e| AppError::BadRequest(e.to_string()))?,
        None => ProtocolCategory::default(),
    };

    // Create protocol — placeholder entry_state, will be updated if inline states are provided
    let placeholder_entry = Uuid::new_v4();
    let mut protocol = Protocol::new(body.project_id, &body.name, placeholder_entry);
    protocol.description = body.description.unwrap_or_default();
    protocol.skill_id = body.skill_id;
    protocol.protocol_category = category;

    // Create inline states
    let mut created_states = Vec::new();
    if let Some(ref inline_states) = body.states {
        for s in inline_states {
            if s.name.trim().is_empty() {
                return Err(AppError::BadRequest(
                    "state name cannot be empty".to_string(),
                ));
            }
            let state_type = match &s.state_type {
                Some(st) => st
                    .parse()
                    .map_err(|e: String| AppError::BadRequest(e))?,
                None => crate::protocol::StateType::default(),
            };
            let ps = ProtocolState {
                id: Uuid::new_v4(),
                protocol_id: protocol.id,
                name: s.name.clone(),
                description: s.description.clone().unwrap_or_default(),
                action: s.action.clone(),
                state_type,
            };
            created_states.push(ps);
        }

        // Find the entry state (first Start state, or first state)
        if let Some(start) = created_states
            .iter()
            .find(|s| s.state_type == crate::protocol::StateType::Start)
        {
            protocol.entry_state = start.id;
        } else if let Some(first) = created_states.first() {
            protocol.entry_state = first.id;
        }

        // Collect terminal states
        protocol.terminal_states = created_states
            .iter()
            .filter(|s| s.state_type == crate::protocol::StateType::Terminal)
            .map(|s| s.id)
            .collect();
    }

    // Upsert protocol
    state
        .orchestrator
        .neo4j()
        .upsert_protocol(&protocol)
        .await
        .map_err(AppError::Internal)?;

    // Upsert states
    for ps in &created_states {
        state
            .orchestrator
            .neo4j()
            .upsert_protocol_state(ps)
            .await
            .map_err(AppError::Internal)?;
    }

    // Create inline transitions
    let mut created_transitions = Vec::new();
    if let Some(ref inline_transitions) = body.transitions {
        for t in inline_transitions {
            if t.trigger.trim().is_empty() {
                return Err(AppError::BadRequest(
                    "transition trigger cannot be empty".to_string(),
                ));
            }
            if t.trigger.len() > MAX_TRIGGER_LEN {
                return Err(AppError::BadRequest(format!(
                    "trigger too long ({} > {} chars)",
                    t.trigger.len(),
                    MAX_TRIGGER_LEN
                )));
            }
            let pt = ProtocolTransition {
                id: Uuid::new_v4(),
                protocol_id: protocol.id,
                from_state: t.from_state,
                to_state: t.to_state,
                trigger: t.trigger.clone(),
                guard: t.guard.clone(),
            };
            created_transitions.push(pt);
        }

        for pt in &created_transitions {
            state
                .orchestrator
                .neo4j()
                .upsert_protocol_transition(pt)
                .await
                .map_err(AppError::Internal)?;
        }
    }

    Ok((
        StatusCode::CREATED,
        Json(ProtocolDetail {
            protocol,
            states: created_states,
            transitions: created_transitions,
        }),
    ))
}

/// Get a protocol by ID (with states and transitions)
///
/// GET /api/protocols/:id
pub async fn get_protocol(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
) -> Result<Json<ProtocolDetail>, AppError> {
    let protocol = state
        .orchestrator
        .neo4j()
        .get_protocol(protocol_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Protocol {} not found", protocol_id)))?;

    let states = state
        .orchestrator
        .neo4j()
        .get_protocol_states(protocol_id)
        .await
        .map_err(AppError::Internal)?;

    let transitions = state
        .orchestrator
        .neo4j()
        .get_protocol_transitions(protocol_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(ProtocolDetail {
        protocol,
        states,
        transitions,
    }))
}

/// Update a protocol
///
/// PUT /api/protocols/:id
pub async fn update_protocol(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
    Json(body): Json<UpdateProtocolBody>,
) -> Result<Json<Protocol>, AppError> {
    let mut protocol = state
        .orchestrator
        .neo4j()
        .get_protocol(protocol_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Protocol {} not found", protocol_id)))?;

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
        protocol.name = name;
    }
    if let Some(description) = body.description {
        if description.len() > MAX_DESCRIPTION_LEN {
            return Err(AppError::BadRequest(format!(
                "description too long ({} > {} chars)",
                description.len(),
                MAX_DESCRIPTION_LEN
            )));
        }
        protocol.description = description;
    }
    if let Some(category_str) = body.protocol_category {
        protocol.protocol_category = category_str
            .parse::<ProtocolCategory>()
            .map_err(|e| AppError::BadRequest(e.to_string()))?;
    }
    protocol.updated_at = chrono::Utc::now();

    state
        .orchestrator
        .neo4j()
        .upsert_protocol(&protocol)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(protocol))
}

/// Delete a protocol and all its states/transitions
///
/// DELETE /api/protocols/:id
pub async fn delete_protocol(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_protocol(protocol_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "Protocol {} not found",
            protocol_id
        )))
    }
}

// ============================================================================
// Handlers — States
// ============================================================================

/// Add a state to a protocol
///
/// POST /api/protocols/:id/states
pub async fn add_state(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
    Json(body): Json<AddStateBody>,
) -> Result<(StatusCode, Json<ProtocolState>), AppError> {
    // Verify protocol exists
    state
        .orchestrator
        .neo4j()
        .get_protocol(protocol_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Protocol {} not found", protocol_id)))?;

    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest(
            "state name cannot be empty".to_string(),
        ));
    }

    let state_type = match &body.state_type {
        Some(st) => st
            .parse()
            .map_err(|e: String| AppError::BadRequest(e))?,
        None => crate::protocol::StateType::default(),
    };

    let ps = ProtocolState {
        id: Uuid::new_v4(),
        protocol_id,
        name: body.name,
        description: body.description.unwrap_or_default(),
        action: body.action,
        state_type,
    };

    state
        .orchestrator
        .neo4j()
        .upsert_protocol_state(&ps)
        .await
        .map_err(AppError::Internal)?;

    Ok((StatusCode::CREATED, Json(ps)))
}

/// Get all states for a protocol
///
/// GET /api/protocols/:id/states
pub async fn list_states(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
) -> Result<Json<Vec<ProtocolState>>, AppError> {
    let states = state
        .orchestrator
        .neo4j()
        .get_protocol_states(protocol_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(states))
}

/// Delete a state
///
/// DELETE /api/protocols/:protocol_id/states/:state_id
pub async fn delete_state(
    State(state): State<OrchestratorState>,
    Path((_protocol_id, state_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_protocol_state(state_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "ProtocolState {} not found",
            state_id
        )))
    }
}

// ============================================================================
// Handlers — Transitions
// ============================================================================

/// Add a transition to a protocol
///
/// POST /api/protocols/:id/transitions
pub async fn add_transition(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
    Json(body): Json<AddTransitionBody>,
) -> Result<(StatusCode, Json<ProtocolTransition>), AppError> {
    // Verify protocol exists
    state
        .orchestrator
        .neo4j()
        .get_protocol(protocol_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Protocol {} not found", protocol_id)))?;

    if body.trigger.trim().is_empty() {
        return Err(AppError::BadRequest(
            "trigger cannot be empty".to_string(),
        ));
    }
    if body.trigger.len() > MAX_TRIGGER_LEN {
        return Err(AppError::BadRequest(format!(
            "trigger too long ({} > {} chars)",
            body.trigger.len(),
            MAX_TRIGGER_LEN
        )));
    }

    let pt = ProtocolTransition {
        id: Uuid::new_v4(),
        protocol_id,
        from_state: body.from_state,
        to_state: body.to_state,
        trigger: body.trigger,
        guard: body.guard,
    };

    state
        .orchestrator
        .neo4j()
        .upsert_protocol_transition(&pt)
        .await
        .map_err(AppError::Internal)?;

    Ok((StatusCode::CREATED, Json(pt)))
}

/// Get all transitions for a protocol
///
/// GET /api/protocols/:id/transitions
pub async fn list_transitions(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
) -> Result<Json<Vec<ProtocolTransition>>, AppError> {
    let transitions = state
        .orchestrator
        .neo4j()
        .get_protocol_transitions(protocol_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(transitions))
}

/// Delete a transition
///
/// DELETE /api/protocols/:protocol_id/transitions/:transition_id
pub async fn delete_transition(
    State(state): State<OrchestratorState>,
    Path((_protocol_id, transition_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_protocol_transition(transition_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "ProtocolTransition {} not found",
            transition_id
        )))
    }
}

// ============================================================================
// Handlers — Link to Skill
// ============================================================================

/// Link a protocol to a skill
///
/// POST /api/protocols/:id/link-skill
pub async fn link_to_skill(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
    Json(body): Json<LinkToSkillBody>,
) -> Result<Json<Protocol>, AppError> {
    let mut protocol = state
        .orchestrator
        .neo4j()
        .get_protocol(protocol_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Protocol {} not found", protocol_id)))?;

    protocol.skill_id = Some(body.skill_id);
    protocol.updated_at = chrono::Utc::now();

    state
        .orchestrator
        .neo4j()
        .upsert_protocol(&protocol)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(protocol))
}

// ============================================================================
// Handlers — Protocol Runs (FSM Runtime)
// ============================================================================

/// Start a new protocol run
///
/// POST /api/protocols/:id/runs
pub async fn start_run(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
    Json(body): Json<StartRunBody>,
) -> Result<(StatusCode, Json<ProtocolRun>), AppError> {
    let run = protocol::engine::start_run(
        state.orchestrator.neo4j(),
        protocol_id,
        body.plan_id,
        body.task_id,
    )
    .await
    .map_err(|e| {
        if e.to_string().contains("not found") {
            AppError::NotFound(e.to_string())
        } else {
            AppError::Internal(e)
        }
    })?;

    // Emit event
    state.event_bus.emit_created(
        crate::events::EntityType::ProtocolRun,
        &run.id.to_string(),
        serde_json::json!({
            "protocol_id": run.protocol_id,
            "current_state": run.current_state,
            "status": run.status.to_string(),
        }),
        Some(run.protocol_id.to_string()),
    );

    Ok((StatusCode::CREATED, Json(run)))
}

/// Fire a transition on a running protocol
///
/// POST /api/protocols/runs/:run_id/transition
pub async fn fire_transition(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
    Json(body): Json<FireTransitionBody>,
) -> Result<Json<TransitionResult>, AppError> {
    let result = protocol::engine::fire_transition(
        state.orchestrator.neo4j(),
        run_id,
        &body.trigger,
    )
    .await
    .map_err(|e| {
        if e.to_string().contains("not found") {
            AppError::NotFound(e.to_string())
        } else {
            AppError::Internal(e)
        }
    })?;

    // Emit event for successful transitions
    if result.success {
        state.event_bus.emit_updated(
            crate::events::EntityType::ProtocolRun,
            &run_id.to_string(),
            serde_json::json!({
                "trigger": body.trigger,
                "current_state": result.current_state,
                "current_state_name": result.current_state_name,
                "run_completed": result.run_completed,
                "status": result.status.to_string(),
            }),
            None,
        );
    }

    Ok(Json(result))
}

/// Get a protocol run by ID
///
/// GET /api/protocols/runs/:run_id
pub async fn get_run(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<ProtocolRun>, AppError> {
    let run = state
        .orchestrator
        .neo4j()
        .get_protocol_run(run_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("ProtocolRun {} not found", run_id)))?;

    Ok(Json(run))
}

/// List runs for a protocol
///
/// GET /api/protocols/:id/runs
pub async fn list_runs(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
    Query(query): Query<RunsListQuery>,
) -> Result<Json<PaginatedResponse<ProtocolRun>>, AppError> {
    let status_filter: Option<RunStatus> = query
        .status
        .as_deref()
        .and_then(|s| s.parse().ok());

    let limit = query.pagination.limit.min(100);
    let offset = query.pagination.offset;

    let (runs, total) = state
        .orchestrator
        .neo4j()
        .list_protocol_runs(protocol_id, status_filter, limit, offset)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(PaginatedResponse {
        items: runs,
        total,
        limit,
        offset,
        has_more: offset + limit < total,
    }))
}

/// Cancel a running protocol
///
/// POST /api/protocols/runs/:run_id/cancel
pub async fn cancel_run(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<ProtocolRun>, AppError> {
    let run = protocol::engine::cancel_run(state.orchestrator.neo4j(), run_id)
        .await
        .map_err(|e| {
            if e.to_string().contains("not found") {
                AppError::NotFound(e.to_string())
            } else {
                AppError::BadRequest(e.to_string())
            }
        })?;

    state.event_bus.emit_updated(
        crate::events::EntityType::ProtocolRun,
        &run_id.to_string(),
        serde_json::json!({
            "status": "cancelled",
        }),
        None,
    );

    Ok(Json(run))
}

/// Fail a running protocol
///
/// POST /api/protocols/runs/:run_id/fail
pub async fn fail_run(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<ProtocolRun>, AppError> {
    let error_msg = body
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown error");

    let run =
        protocol::engine::fail_run(state.orchestrator.neo4j(), run_id, error_msg)
            .await
            .map_err(|e| {
                if e.to_string().contains("not found") {
                    AppError::NotFound(e.to_string())
                } else {
                    AppError::BadRequest(e.to_string())
                }
            })?;

    state.event_bus.emit_updated(
        crate::events::EntityType::ProtocolRun,
        &run_id.to_string(),
        serde_json::json!({
            "status": "failed",
            "error": error_msg,
        }),
        None,
    );

    Ok(Json(run))
}

/// Delete a protocol run
///
/// DELETE /api/protocols/runs/:run_id
pub async fn delete_run(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_protocol_run(run_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        state.event_bus.emit_deleted(
            crate::events::EntityType::ProtocolRun,
            &run_id.to_string(),
            None,
        );
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(AppError::NotFound(format!(
            "ProtocolRun {} not found",
            run_id
        )))
    }
}
