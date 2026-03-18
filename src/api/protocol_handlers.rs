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
// Validation helpers (pure functions — unit-testable)
// ============================================================================

/// Validate that a transition does not originate from a terminal state or target a start state.
///
/// Looks up `from_state_id` and `to_state_id` in the provided `states` slice.
/// Returns `Ok(())` if valid, or `Err(message)` describing the violation.
/// If a state ID is not found in the slice, the check is silently skipped
/// (the caller may validate existence separately).
fn validate_transition_state_types(
    states: &[ProtocolState],
    from_state_id: Uuid,
    to_state_id: Uuid,
) -> Result<(), String> {
    if let Some(from_s) = states.iter().find(|s| s.id == from_state_id) {
        if from_s.state_type == crate::protocol::StateType::Terminal {
            return Err(format!(
                "Cannot add transition from terminal state '{}'",
                from_s.name
            ));
        }
    }
    if let Some(to_s) = states.iter().find(|s| s.id == to_state_id) {
        if to_s.state_type == crate::protocol::StateType::Start {
            return Err(format!(
                "Cannot add transition to start state '{}'",
                to_s.name
            ));
        }
    }
    Ok(())
}

/// Map an `anyhow::Error` from `fire_transition` to the appropriate `AppError` variant.
///
/// - Messages containing "not found" → `AppError::NotFound`
/// - Messages containing "OptimisticLockError" → `AppError::Conflict`
/// - Everything else → `AppError::Internal`
fn map_fire_transition_error(e: anyhow::Error) -> AppError {
    let msg = e.to_string();
    if msg.contains("not found") {
        AppError::NotFound(msg)
    } else if msg.contains("OptimisticLockError") {
        AppError::Conflict(msg)
    } else {
        AppError::Internal(e)
    }
}

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
    /// Trigger mode (manual, event, scheduled, auto)
    pub trigger_mode: Option<String>,
    /// Trigger configuration (events, schedule, conditions)
    pub trigger_config: Option<crate::protocol::TriggerConfig>,
    /// Multi-dimensional relevance profile for context-aware routing
    pub relevance_vector: Option<crate::protocol::routing::RelevanceVector>,
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
    pub trigger_mode: Option<String>,
    pub trigger_config: Option<crate::protocol::TriggerConfig>,
    pub relevance_vector: Option<crate::protocol::routing::RelevanceVector>,
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

/// Query parameters for routing protocols
#[derive(Debug, Deserialize)]
pub struct RouteProtocolsQuery {
    /// Required: project_id to scope the protocols
    pub project_id: Uuid,
    /// Optional: plan_id to auto-build context from plan metrics
    pub plan_id: Option<Uuid>,
    /// Optional: explicit phase override (warmup, planning, execution, review, closure)
    pub phase: Option<String>,
    /// Optional: explicit domain relevance (0.0-1.0)
    pub domain: Option<f64>,
    /// Optional: explicit resource availability (0.0-1.0)
    pub resource: Option<f64>,
    /// Optional: explicit structure complexity override (0.0-1.0)
    pub structure: Option<f64>,
    /// Optional: explicit lifecycle position override (0.0-1.0)
    pub lifecycle: Option<f64>,
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

    Ok(Json(PaginatedResponse::new(
        protocols, total, limit, offset,
    )))
}

/// Route protocols: rank all protocols in a project by affinity to the current context.
///
/// GET /api/protocols/route?project_id=...&plan_id=...&phase=...&domain=...&resource=...
///
/// If `plan_id` is provided, the context vector is auto-built from plan metrics
/// (task count, dependency count, affected files, completion %). Explicit query
/// params (`phase`, `domain`, `resource`) override the auto-built values.
pub async fn route_protocols(
    State(state): State<OrchestratorState>,
    Query(query): Query<RouteProtocolsQuery>,
) -> Result<Json<crate::protocol::routing::RouteResponse>, AppError> {
    use crate::protocol::routing::{ContextVector, DimensionWeights};

    // Fetch all protocols for this project (no pagination — routing needs all)
    let (protocols, _) = state
        .orchestrator
        .neo4j()
        .list_protocols(query.project_id, None, 200, 0)
        .await
        .map_err(AppError::Internal)?;

    if protocols.is_empty() {
        return Ok(Json(crate::protocol::routing::RouteResponse {
            context: ContextVector::default(),
            weights: DimensionWeights::default(),
            results: vec![],
            total_evaluated: 0,
        }));
    }

    // Build context vector
    let context = if let Some(plan_id) = query.plan_id {
        // Auto-build from plan metrics
        let plan = state
            .orchestrator
            .neo4j()
            .get_plan(plan_id)
            .await
            .map_err(AppError::Internal)?;

        let (tasks, edges) = state
            .orchestrator
            .neo4j()
            .get_plan_dependency_graph(plan_id)
            .await
            .map_err(AppError::Internal)?;

        let task_count = tasks.len();
        let dependency_count = edges.len();
        let affected_files_count: usize = tasks
            .iter()
            .flat_map(|t| t.affected_files.iter())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let completed_count = tasks
            .iter()
            .filter(|t| t.status == crate::neo4j::models::TaskStatus::Completed)
            .count();
        let completion_pct = if task_count > 0 {
            completed_count as f64 / task_count as f64
        } else {
            0.0
        };

        let plan_status_str = plan
            .as_ref()
            .map(|p| {
                // PlanStatus serializes via serde rename_all = "snake_case"
                serde_json::to_string(&p.status)
                    .unwrap_or_else(|_| "\"execution\"".to_string())
                    .trim_matches('"')
                    .to_string()
            })
            .unwrap_or_else(|| "execution".to_string());

        let mut ctx = ContextVector::from_plan_context(
            &plan_status_str,
            task_count,
            dependency_count,
            affected_files_count,
            completion_pct,
        );

        // Allow explicit overrides
        if let Some(phase) = &query.phase {
            ctx.phase = match phase.as_str() {
                "warmup" | "draft" => 0.0,
                "planning" | "approved" => 0.25,
                "execution" | "in_progress" => 0.5,
                "review" | "testing" => 0.75,
                "closure" | "completed" => 1.0,
                _ => ctx.phase,
            };
        }
        if let Some(domain) = query.domain {
            ctx.domain = domain.clamp(0.0, 1.0);
        }
        if let Some(resource) = query.resource {
            ctx.resource = resource.clamp(0.0, 1.0);
        }
        if let Some(structure) = query.structure {
            ctx.structure = structure.clamp(0.0, 1.0);
        }
        if let Some(lifecycle) = query.lifecycle {
            ctx.lifecycle = lifecycle.clamp(0.0, 1.0);
        }
        ctx
    } else {
        // Manual context — use defaults with overrides
        let mut ctx = ContextVector::default();
        if let Some(phase) = &query.phase {
            ctx.phase = match phase.as_str() {
                "warmup" | "draft" => 0.0,
                "planning" | "approved" => 0.25,
                "execution" | "in_progress" => 0.5,
                "review" | "testing" => 0.75,
                "closure" | "completed" => 1.0,
                _ => ctx.phase,
            };
        }
        if let Some(domain) = query.domain {
            ctx.domain = domain.clamp(0.0, 1.0);
        }
        if let Some(resource) = query.resource {
            ctx.resource = resource.clamp(0.0, 1.0);
        }
        if let Some(structure) = query.structure {
            ctx.structure = structure.clamp(0.0, 1.0);
        }
        if let Some(lifecycle) = query.lifecycle {
            ctx.lifecycle = lifecycle.clamp(0.0, 1.0);
        }
        ctx
    };

    let weights = DimensionWeights::default();
    let response = crate::protocol::routing::rank_protocols(&context, &protocols, &weights);

    Ok(Json(response))
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

    // Check name uniqueness within the project
    if let Some(existing_id) = state
        .orchestrator
        .neo4j()
        .get_protocol_by_name_and_project(&body.name, body.project_id)
        .await
        .map_err(AppError::Internal)?
    {
        return Err(AppError::Conflict(format!(
            "Protocol '{}' already exists in this project (id: {})",
            body.name, existing_id
        )));
    }

    // Create protocol — placeholder entry_state, will be updated if inline states are provided
    let placeholder_entry = Uuid::new_v4();
    let mut protocol = Protocol::new(body.project_id, &body.name, placeholder_entry);
    protocol.description = body.description.unwrap_or_default();
    protocol.skill_id = body.skill_id;
    protocol.protocol_category = category;
    if let Some(ref tm) = body.trigger_mode {
        protocol.trigger_mode = tm
            .parse::<crate::protocol::TriggerMode>()
            .map_err(AppError::BadRequest)?;
    }
    protocol.trigger_config = body.trigger_config.clone();
    protocol.relevance_vector = body.relevance_vector.clone();

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
                Some(st) => st.parse().map_err(|e: String| AppError::BadRequest(e))?,
                None => crate::protocol::StateType::default(),
            };
            let ps = ProtocolState {
                id: Uuid::new_v4(),
                protocol_id: protocol.id,
                name: s.name.clone(),
                description: s.description.clone().unwrap_or_default(),
                action: s.action.clone(),
                state_type,
                sub_protocol_id: None,
                completion_strategy: None,
                on_failure_strategy: None,
                generator_config: None,
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
            // Validate state types: no transition FROM terminal or TO start
            validate_transition_state_types(&created_states, t.from_state, t.to_state)
                .map_err(AppError::BadRequest)?;
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
    if let Some(trigger_mode_str) = body.trigger_mode {
        protocol.trigger_mode = trigger_mode_str
            .parse::<crate::protocol::TriggerMode>()
            .map_err(AppError::BadRequest)?;
    }
    if let Some(trigger_config) = body.trigger_config {
        protocol.trigger_config = Some(trigger_config);
    }
    if let Some(relevance_vector) = body.relevance_vector {
        protocol.relevance_vector = Some(relevance_vector);
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
        Some(st) => st.parse().map_err(|e: String| AppError::BadRequest(e))?,
        None => crate::protocol::StateType::default(),
    };

    let ps = ProtocolState {
        id: Uuid::new_v4(),
        protocol_id,
        name: body.name,
        description: body.description.unwrap_or_default(),
        action: body.action,
        state_type,
        sub_protocol_id: None,
        completion_strategy: None,
        on_failure_strategy: None,
        generator_config: None,
    };

    state
        .orchestrator
        .neo4j()
        .upsert_protocol_state(&ps)
        .await
        .map_err(AppError::Internal)?;

    // Update protocol's entry_state / terminal_states when adding Start or Terminal states
    if ps.state_type == crate::protocol::StateType::Start
        || ps.state_type == crate::protocol::StateType::Terminal
    {
        if let Some(mut protocol) = state
            .orchestrator
            .neo4j()
            .get_protocol(protocol_id)
            .await
            .map_err(AppError::Internal)?
        {
            if ps.state_type == crate::protocol::StateType::Start {
                protocol.entry_state = ps.id;
            }
            if ps.state_type == crate::protocol::StateType::Terminal
                && !protocol.terminal_states.contains(&ps.id)
            {
                protocol.terminal_states.push(ps.id);
            }
            state
                .orchestrator
                .neo4j()
                .upsert_protocol(&protocol)
                .await
                .map_err(AppError::Internal)?;
        }
    }

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
        return Err(AppError::BadRequest("trigger cannot be empty".to_string()));
    }
    if body.trigger.len() > MAX_TRIGGER_LEN {
        return Err(AppError::BadRequest(format!(
            "trigger too long ({} > {} chars)",
            body.trigger.len(),
            MAX_TRIGGER_LEN
        )));
    }

    // Validate state types: cannot transition FROM a terminal state or TO a start state
    let states = state
        .orchestrator
        .neo4j()
        .get_protocol_states(protocol_id)
        .await
        .map_err(AppError::Internal)?;

    // Verify both states exist in the protocol
    if !states.iter().any(|s| s.id == body.from_state) {
        return Err(AppError::BadRequest(format!(
            "from_state {} not found in protocol",
            body.from_state
        )));
    }
    if !states.iter().any(|s| s.id == body.to_state) {
        return Err(AppError::BadRequest(format!(
            "to_state {} not found in protocol",
            body.to_state
        )));
    }

    validate_transition_state_types(&states, body.from_state, body.to_state)
        .map_err(AppError::BadRequest)?;

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
        None, // manual trigger
    )
    .await
    .map_err(|e| {
        let msg = e.to_string();
        if msg.contains("not found") {
            AppError::NotFound(msg)
        } else if msg.contains("concurrent run") {
            AppError::Conflict(msg)
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
    let result =
        protocol::engine::fire_transition(state.orchestrator.neo4j(), run_id, &body.trigger)
            .await
            .map_err(map_fire_transition_error)?;

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

        // Emit child_completed / child_failed event when a child run reaches terminal state
        if let Some(ref info) = result.child_completion {
            let event_action = if info.status == RunStatus::Failed {
                "child_failed"
            } else {
                "child_completed"
            };
            state.event_bus.emit_updated(
                crate::events::EntityType::ProtocolRun,
                &info.parent_run_id.to_string(),
                serde_json::json!({
                    "event": event_action,
                    "parent_run_id": info.parent_run_id,
                    "child_run_id": info.child_run_id,
                    "protocol_id": info.protocol_id,
                    "child_status": info.status.to_string(),
                }),
                None,
            );
        }
    }

    // ── Trajectory collection: record protocol transition decision ────────
    if let Some(ref collector) = state.trajectory_collector {
        collector.record_decision(neural_routing_runtime::DecisionRecord {
            session_id: format!("protocol-run:{}", run_id),
            context_embedding: vec![],
            action_type: "protocol.transition".to_string(),
            action_params: serde_json::json!({
                "run_id": run_id,
                "trigger": body.trigger,
                "success": result.success,
                "current_state_name": result.current_state_name,
                "run_completed": result.run_completed,
            }),
            alternatives_count: 1,
            chosen_index: 0,
            confidence: if result.success { 0.9 } else { 0.1 },
            protocol_run_id: Some(run_id),
            protocol_state: Some(result.current_state_name.clone()),
            tool_usages: vec![neural_routing_runtime::ToolUsage {
                tool_name: "protocol".to_string(),
                action: "transition".to_string(),
                params_hash: format!("trigger:{}", body.trigger),
                duration_ms: None,
                success: result.success,
            }],
            touched_entities: vec![neural_routing_runtime::TouchedEntity {
                entity_type: "ProtocolRun".to_string(),
                entity_id: run_id.to_string(),
                access_mode: "write".to_string(),
                relevance: Some(1.0),
            }],
            timestamp_ms: 0,
            query_embedding: vec![],
            node_features: vec![],
        });
    }

    Ok(Json(result))
}

/// Enriched protocol run response with hierarchy metadata
#[derive(Debug, Serialize)]
pub struct EnrichedProtocolRun {
    #[serde(flatten)]
    pub run: ProtocolRun,
    /// Number of direct child runs
    pub children_count: usize,
}

/// Get a protocol run by ID (enriched with children_count)
///
/// GET /api/protocols/runs/:run_id
pub async fn get_run(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<EnrichedProtocolRun>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let run = neo4j
        .get_protocol_run(run_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("ProtocolRun {} not found", run_id)))?;

    let children_count = neo4j
        .count_child_runs(run_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(EnrichedProtocolRun {
        run,
        children_count,
    }))
}

/// Get direct children of a protocol run
///
/// GET /api/protocols/runs/:run_id/children
pub async fn get_run_children(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<Vec<ProtocolRun>>, AppError> {
    // Verify parent exists
    let neo4j = state.orchestrator.neo4j();
    neo4j
        .get_protocol_run(run_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("ProtocolRun {} not found", run_id)))?;

    let children = neo4j
        .list_child_runs(run_id)
        .await
        .map_err(AppError::Internal)?;

    Ok(Json(children))
}

/// Recursive tree node for protocol run hierarchy
#[derive(Debug, Serialize)]
pub struct RunTreeNode {
    #[serde(flatten)]
    pub run: ProtocolRun,
    pub children: Vec<RunTreeNode>,
}

/// Get the full run tree starting from a root run (recursive)
///
/// GET /api/protocols/runs/:run_id/tree
pub async fn get_run_tree(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<RunTreeNode>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let root = neo4j
        .get_protocol_run(run_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("ProtocolRun {} not found", run_id)))?;

    // Build tree recursively (max depth 10 to prevent infinite loops)
    async fn build_tree(
        neo4j: &dyn crate::neo4j::traits::GraphStore,
        run: ProtocolRun,
        depth: usize,
    ) -> Result<RunTreeNode, AppError> {
        if depth > 10 {
            return Ok(RunTreeNode {
                run,
                children: vec![],
            });
        }
        let child_runs = neo4j
            .list_child_runs(run.id)
            .await
            .map_err(AppError::Internal)?;
        let mut children = Vec::with_capacity(child_runs.len());
        for child in child_runs {
            children.push(Box::pin(build_tree(neo4j, child, depth + 1)).await?);
        }
        Ok(RunTreeNode { run, children })
    }

    let tree = build_tree(neo4j, root, 0).await?;
    Ok(Json(tree))
}

/// List runs for a protocol
///
/// GET /api/protocols/:id/runs
pub async fn list_runs(
    State(state): State<OrchestratorState>,
    Path(protocol_id): Path<Uuid>,
    Query(query): Query<RunsListQuery>,
) -> Result<Json<PaginatedResponse<ProtocolRun>>, AppError> {
    let status_filter: Option<RunStatus> = query.status.as_deref().and_then(|s| s.parse().ok());

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

    let run = protocol::engine::fail_run(state.orchestrator.neo4j(), run_id, error_msg)
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

    // Emit child_failed event if this run has a parent
    if let Some(parent_run_id) = run.parent_run_id {
        state.event_bus.emit_updated(
            crate::events::EntityType::ProtocolRun,
            &parent_run_id.to_string(),
            serde_json::json!({
                "event": "child_failed",
                "parent_run_id": parent_run_id,
                "child_run_id": run.id,
                "protocol_id": run.protocol_id,
                "child_status": run.status.to_string(),
            }),
            None,
        );
    }

    Ok(Json(run))
}

/// Report progress on a running protocol state
///
/// POST /api/protocols/runs/:run_id/progress
///
/// Emits a WebSocket progress event for the FSM Viewer to display.
/// Only accepted for runs with status `running`.
pub async fn report_progress(
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
    Json(body): Json<ReportProgressBody>,
) -> Result<StatusCode, AppError> {
    // Verify the run exists and is active
    let run = state
        .orchestrator
        .neo4j()
        .get_protocol_run(run_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("ProtocolRun {} not found", run_id)))?;

    if !run.is_active() {
        return Err(AppError::BadRequest(format!(
            "Cannot report progress on a {} run",
            run.status
        )));
    }

    let progress = protocol::ProtocolRunProgress::new(
        run_id,
        &body.state_name,
        &body.sub_action,
        body.processed,
        body.total,
        body.elapsed_ms,
    );

    // Emit progress event via WebSocket
    state.event_bus.emit_progress(
        crate::events::EntityType::ProtocolRun,
        &run_id.to_string(),
        serde_json::to_value(&progress).unwrap_or_default(),
        None,
    );

    Ok(StatusCode::ACCEPTED)
}

#[derive(Debug, Deserialize)]
pub struct ReportProgressBody {
    /// Current state name (e.g., "BACKFILL")
    pub state_name: String,
    /// Current sub-action (e.g., "backfill_synapses")
    pub sub_action: String,
    /// Number of sub-actions completed so far
    pub processed: usize,
    /// Total number of sub-actions
    pub total: usize,
    /// Milliseconds elapsed since the state was entered
    #[serde(default)]
    pub elapsed_ms: u64,
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

// ============================================================================
// Pattern Composer — Compose protocol from notes + FSM
// ============================================================================

/// A note-to-state binding in a composition
#[derive(Debug, Deserialize)]
pub struct NoteStateBinding {
    pub note_id: Uuid,
    pub state_name: String,
}

/// Inline state for composition (uses name-based references instead of UUIDs)
#[derive(Debug, Deserialize)]
pub struct ComposeStateInline {
    pub name: String,
    pub description: Option<String>,
    pub state_type: Option<String>,
    pub action: Option<String>,
    /// Optional sub-protocol to spawn when entering this state
    pub sub_protocol_id: Option<Uuid>,
}

/// Inline transition for composition (uses state names instead of UUIDs)
#[derive(Debug, Deserialize)]
pub struct ComposeTransitionInline {
    pub from_state: String,
    pub to_state: String,
    pub trigger: String,
    pub guard: Option<String>,
}

/// Request body for composing a protocol from notes + FSM definition
#[derive(Debug, Deserialize)]
pub struct ComposeProtocolBody {
    pub project_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub category: Option<String>,
    /// Notes to bind to states (optional, defaults to empty)
    #[serde(default)]
    pub notes: Vec<NoteStateBinding>,
    /// FSM states
    pub states: Vec<ComposeStateInline>,
    /// FSM transitions (using state names)
    pub transitions: Vec<ComposeTransitionInline>,
    /// Relevance vector for context routing
    pub relevance_vector: Option<crate::protocol::routing::RelevanceVector>,
    /// Trigger patterns to register on the auto-created skill
    pub triggers: Option<Vec<crate::skills::SkillTrigger>>,
}

/// Response from composition
#[derive(Debug, Serialize)]
pub struct ComposeResponse {
    pub protocol_id: Uuid,
    pub skill_id: Uuid,
    pub states_created: usize,
    pub transitions_created: usize,
    pub notes_linked: usize,
}

/// Compose a protocol from notes + FSM + triggers in one shot.
///
/// POST /api/protocols/compose
///
/// Creates: Skill + Protocol + States + Transitions + Note→State links
pub async fn compose_protocol(
    State(state): State<OrchestratorState>,
    Json(body): Json<ComposeProtocolBody>,
) -> Result<(StatusCode, Json<ComposeResponse>), AppError> {
    // ── Validation ──────────────────────────────────────────────────
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
    if body.states.is_empty() {
        return Err(AppError::BadRequest(
            "at least one state is required".to_string(),
        ));
    }

    // Validate states have unique names
    let state_names: Vec<&str> = body.states.iter().map(|s| s.name.as_str()).collect();
    let unique_names: std::collections::HashSet<&str> = state_names.iter().copied().collect();
    if unique_names.len() != state_names.len() {
        return Err(AppError::BadRequest(
            "duplicate state names found".to_string(),
        ));
    }

    // Validate transitions reference existing state names
    for t in &body.transitions {
        if !unique_names.contains(t.from_state.as_str()) {
            return Err(AppError::BadRequest(format!(
                "transition from_state '{}' not found in states",
                t.from_state
            )));
        }
        if !unique_names.contains(t.to_state.as_str()) {
            return Err(AppError::BadRequest(format!(
                "transition to_state '{}' not found in states",
                t.to_state
            )));
        }
    }

    // Validate note bindings reference existing state names
    for nb in &body.notes {
        if !unique_names.contains(nb.state_name.as_str()) {
            return Err(AppError::BadRequest(format!(
                "note binding state_name '{}' not found in states",
                nb.state_name
            )));
        }
    }

    let category = match &body.category {
        Some(s) => s
            .parse::<ProtocolCategory>()
            .map_err(|e| AppError::BadRequest(e.to_string()))?,
        None => ProtocolCategory::default(),
    };

    let neo4j = state.orchestrator.neo4j();

    // Check name uniqueness within the project
    if let Some(existing_id) = neo4j
        .get_protocol_by_name_and_project(&body.name, body.project_id)
        .await
        .map_err(AppError::Internal)?
    {
        return Err(AppError::Conflict(format!(
            "Protocol '{}' already exists in this project (id: {})",
            body.name, existing_id
        )));
    }

    // ── 1. Create Skill ─────────────────────────────────────────────
    let skill_id = Uuid::new_v4();
    let skill = crate::skills::SkillNode {
        id: skill_id,
        project_id: body.project_id,
        name: format!("Skill: {}", body.name),
        description: body
            .description
            .clone()
            .unwrap_or_else(|| format!("Auto-generated skill for protocol '{}'", body.name)),
        status: crate::skills::SkillStatus::Active,
        trigger_patterns: body.triggers.clone().unwrap_or_default(),
        context_template: None,
        energy: 0.8,
        cohesion: 0.7,
        coverage: 0,
        note_count: 0,
        decision_count: 0,
        activation_count: 0,
        hit_rate: 0.0,
        last_activated: None,
        version: 1,
        fingerprint: None,
        imported_at: None,
        is_validated: false,
        tags: vec![],
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    neo4j
        .create_skill(&skill)
        .await
        .map_err(AppError::Internal)?;

    // ── 2. Create Protocol ──────────────────────────────────────────
    let placeholder_entry = Uuid::new_v4();
    let mut proto = Protocol::new(body.project_id, &body.name, placeholder_entry);
    proto.description = body.description.unwrap_or_default();
    proto.protocol_category = category;
    proto.skill_id = Some(skill_id);
    proto.relevance_vector = body.relevance_vector;

    // Create states with UUID mapping
    let mut name_to_id: std::collections::HashMap<String, Uuid> = std::collections::HashMap::new();
    let mut created_states = Vec::new();

    for s in &body.states {
        let state_type = match &s.state_type {
            Some(st) => st.parse().map_err(|e: String| AppError::BadRequest(e))?,
            None => crate::protocol::StateType::default(),
        };
        let ps = ProtocolState {
            id: Uuid::new_v4(),
            protocol_id: proto.id,
            name: s.name.clone(),
            description: s.description.clone().unwrap_or_default(),
            action: s.action.clone(),
            state_type,
            sub_protocol_id: s.sub_protocol_id,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
        };
        name_to_id.insert(s.name.clone(), ps.id);
        created_states.push(ps);
    }

    // Set entry state and terminal states
    if let Some(start) = created_states
        .iter()
        .find(|s| s.state_type == crate::protocol::StateType::Start)
    {
        proto.entry_state = start.id;
    } else if let Some(first) = created_states.first() {
        proto.entry_state = first.id;
    }

    proto.terminal_states = created_states
        .iter()
        .filter(|s| s.state_type == crate::protocol::StateType::Terminal)
        .map(|s| s.id)
        .collect();

    // Upsert protocol
    neo4j
        .upsert_protocol(&proto)
        .await
        .map_err(AppError::Internal)?;

    // Upsert states
    for ps in &created_states {
        neo4j
            .upsert_protocol_state(ps)
            .await
            .map_err(AppError::Internal)?;
    }

    // ── 3. Create Transitions ───────────────────────────────────────
    let mut created_transitions = Vec::new();
    for t in &body.transitions {
        let from_id = name_to_id
            .get(&t.from_state)
            .ok_or_else(|| AppError::BadRequest(format!("state '{}' not found", t.from_state)))?;
        let to_id = name_to_id
            .get(&t.to_state)
            .ok_or_else(|| AppError::BadRequest(format!("state '{}' not found", t.to_state)))?;

        // Validate state types: no transition FROM terminal or TO start
        validate_transition_state_types(&created_states, *from_id, *to_id)
            .map_err(AppError::BadRequest)?;

        let pt = ProtocolTransition {
            id: Uuid::new_v4(),
            protocol_id: proto.id,
            from_state: *from_id,
            to_state: *to_id,
            trigger: t.trigger.clone(),
            guard: t.guard.clone(),
        };
        created_transitions.push(pt);
    }

    for pt in &created_transitions {
        neo4j
            .upsert_protocol_transition(pt)
            .await
            .map_err(AppError::Internal)?;
    }

    // ── 4. Link notes to the skill as members ──────────────────────
    // Notes become members of the auto-created skill (HAS_MEMBER relationship),
    // which provides the knowledge base for this composed protocol.
    let mut notes_linked = 0;
    for nb in &body.notes {
        if name_to_id.contains_key(&nb.state_name) {
            neo4j
                .add_skill_member(skill_id, "note", nb.note_id)
                .await
                .map_err(AppError::Internal)?;
            notes_linked += 1;
        }
    }

    Ok((
        StatusCode::CREATED,
        Json(ComposeResponse {
            protocol_id: proto.id,
            skill_id,
            states_created: created_states.len(),
            transitions_created: created_transitions.len(),
            notes_linked,
        }),
    ))
}

// ============================================================================
// Pattern Composer — Simulate activation (dry-run routing)
// ============================================================================

/// Request body for simulating protocol activation
#[derive(Debug, Deserialize)]
pub struct SimulateActivationBody {
    pub protocol_id: Uuid,
    /// Explicit context vector (if provided, overrides plan-based auto-build)
    pub context: Option<crate::protocol::routing::ContextVector>,
    /// Optional plan_id to auto-build context from
    pub plan_id: Option<Uuid>,
}

/// Response from simulation
#[derive(Debug, Serialize)]
pub struct SimulateResponse {
    pub score: f64,
    pub dimensions: Vec<crate::protocol::routing::DimensionScore>,
    pub would_activate: bool,
    pub explanation: String,
    pub context_used: crate::protocol::routing::ContextVector,
}

/// Simulate protocol activation without side effects.
///
/// POST /api/protocols/simulate
///
/// Computes affinity score for a given protocol + context without creating a run.
pub async fn simulate_activation(
    State(state): State<OrchestratorState>,
    Json(body): Json<SimulateActivationBody>,
) -> Result<Json<SimulateResponse>, AppError> {
    use crate::protocol::routing::{compute_affinity, ContextVector, DimensionWeights};

    let neo4j = state.orchestrator.neo4j();

    // Fetch protocol
    let proto = neo4j
        .get_protocol(body.protocol_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("Protocol {} not found", body.protocol_id)))?;

    // Build context
    let context = if let Some(ctx) = body.context {
        ctx
    } else if let Some(plan_id) = body.plan_id {
        // Auto-build from plan metrics (same logic as route_protocols)
        let plan = neo4j.get_plan(plan_id).await.map_err(AppError::Internal)?;
        let (tasks, edges) = neo4j
            .get_plan_dependency_graph(plan_id)
            .await
            .map_err(AppError::Internal)?;

        let task_count = tasks.len();
        let dependency_count = edges.len();
        let affected_files_count: usize = tasks
            .iter()
            .flat_map(|t| t.affected_files.iter())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let completed_count = tasks
            .iter()
            .filter(|t| t.status == crate::neo4j::models::TaskStatus::Completed)
            .count();
        let completion_pct = if task_count > 0 {
            completed_count as f64 / task_count as f64
        } else {
            0.0
        };

        let plan_status_str = plan
            .as_ref()
            .map(|p| {
                serde_json::to_string(&p.status)
                    .unwrap_or_else(|_| "\"execution\"".to_string())
                    .trim_matches('"')
                    .to_string()
            })
            .unwrap_or_else(|| "execution".to_string());

        ContextVector::from_plan_context(
            &plan_status_str,
            task_count,
            dependency_count,
            affected_files_count,
            completion_pct,
        )
    } else {
        ContextVector::default()
    };

    let relevance = proto.relevance_vector.clone().unwrap_or_default();
    let weights = DimensionWeights::default();

    let affinity = compute_affinity(&context, &relevance, &weights);

    // Activation threshold: 0.6 (60%)
    let would_activate = affinity.score >= 0.6;

    Ok(Json(SimulateResponse {
        score: affinity.score,
        dimensions: affinity.dimensions,
        would_activate,
        explanation: affinity.explanation,
        context_used: context,
    }))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::StateType;

    /// Helper: build a minimal ProtocolState for testing.
    fn make_state(name: &str, state_type: StateType) -> ProtocolState {
        ProtocolState {
            id: Uuid::new_v4(),
            protocol_id: Uuid::new_v4(),
            name: name.to_string(),
            description: String::new(),
            action: None,
            state_type,
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
        }
    }

    // ----------------------------------------------------------------
    // validate_transition_state_types
    // ----------------------------------------------------------------

    #[test]
    fn test_validate_transition_ok_start_to_intermediate() {
        let start = make_state("init", StateType::Start);
        let mid = make_state("work", StateType::Intermediate);
        let states = vec![start.clone(), mid.clone()];
        assert!(validate_transition_state_types(&states, start.id, mid.id).is_ok());
    }

    #[test]
    fn test_validate_transition_ok_intermediate_to_terminal() {
        let mid = make_state("work", StateType::Intermediate);
        let end = make_state("done", StateType::Terminal);
        let states = vec![mid.clone(), end.clone()];
        assert!(validate_transition_state_types(&states, mid.id, end.id).is_ok());
    }

    #[test]
    fn test_validate_transition_rejects_from_terminal() {
        let end = make_state("done", StateType::Terminal);
        let mid = make_state("work", StateType::Intermediate);
        let states = vec![end.clone(), mid.clone()];
        let err = validate_transition_state_types(&states, end.id, mid.id).unwrap_err();
        assert!(err.contains("Cannot add transition from terminal state"));
        assert!(err.contains("done"));
    }

    #[test]
    fn test_validate_transition_rejects_to_start() {
        let start = make_state("init", StateType::Start);
        let mid = make_state("work", StateType::Intermediate);
        let states = vec![start.clone(), mid.clone()];
        let err = validate_transition_state_types(&states, mid.id, start.id).unwrap_err();
        assert!(err.contains("Cannot add transition to start state"));
        assert!(err.contains("init"));
    }

    #[test]
    fn test_validate_transition_unknown_ids_pass() {
        // IDs not in the states slice → silently pass (existence validated elsewhere)
        let states = vec![make_state("x", StateType::Intermediate)];
        let unknown_from = Uuid::new_v4();
        let unknown_to = Uuid::new_v4();
        assert!(validate_transition_state_types(&states, unknown_from, unknown_to).is_ok());
    }

    #[test]
    fn test_validate_transition_from_terminal_to_start_both_errors() {
        // When both violations exist, the FROM-terminal error is returned first
        let start = make_state("init", StateType::Start);
        let end = make_state("done", StateType::Terminal);
        let states = vec![start.clone(), end.clone()];
        let err = validate_transition_state_types(&states, end.id, start.id).unwrap_err();
        assert!(err.contains("terminal"));
    }

    #[test]
    fn test_validate_transition_empty_states() {
        // Empty states slice → both lookups miss → Ok
        assert!(validate_transition_state_types(&[], Uuid::new_v4(), Uuid::new_v4()).is_ok());
    }

    #[test]
    fn test_validate_transition_intermediate_to_intermediate() {
        let a = make_state("step-a", StateType::Intermediate);
        let b = make_state("step-b", StateType::Intermediate);
        let states = vec![a.clone(), b.clone()];
        assert!(validate_transition_state_types(&states, a.id, b.id).is_ok());
    }

    #[test]
    fn test_validate_transition_start_to_terminal() {
        let start = make_state("begin", StateType::Start);
        let end = make_state("finish", StateType::Terminal);
        let states = vec![start.clone(), end.clone()];
        assert!(validate_transition_state_types(&states, start.id, end.id).is_ok());
    }

    // ----------------------------------------------------------------
    // map_fire_transition_error
    // ----------------------------------------------------------------

    #[test]
    fn test_map_error_not_found() {
        let e = anyhow::anyhow!("ProtocolRun abc123 not found");
        match map_fire_transition_error(e) {
            AppError::NotFound(msg) => assert!(msg.contains("not found")),
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_optimistic_lock() {
        let e = anyhow::anyhow!(
            "OptimisticLockError: protocol run xyz was modified concurrently (expected version 3, stale)"
        );
        match map_fire_transition_error(e) {
            AppError::Conflict(msg) => assert!(msg.contains("OptimisticLockError")),
            other => panic!("Expected Conflict, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_internal_fallback() {
        let e = anyhow::anyhow!("Neo4j connection refused");
        match map_fire_transition_error(e) {
            AppError::Internal(err) => assert!(err.to_string().contains("connection refused")),
            other => panic!("Expected Internal, got {:?}", other),
        }
    }

    #[test]
    fn test_map_error_not_found_takes_priority_over_optimistic() {
        // If both keywords appear, "not found" is checked first
        let e = anyhow::anyhow!("OptimisticLockError: not found something");
        match map_fire_transition_error(e) {
            AppError::NotFound(_) => {} // Correct: "not found" matched first
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    // ================================================================
    // Handler integration tests (with mock OrchestratorState)
    // ================================================================

    use crate::events::{EventBus, HybridEmitter};
    use crate::meilisearch::mock::MockSearchStore;
    use crate::neo4j::mock::MockGraphStore;
    use crate::orchestrator::watcher::FileWatcher;
    use crate::orchestrator::Orchestrator;
    use crate::test_helpers::{mock_app_state_with, test_project};
    use axum::extract::State;
    use std::sync::Arc;

    /// Build a server state from a MockGraphStore (shared helper).
    async fn build_server_state(mock: MockGraphStore) -> OrchestratorState {
        let app_state = mock_app_state_with(mock, MockSearchStore::new());
        let orchestrator = Orchestrator::new(app_state).await.unwrap();
        let orc = Arc::new(orchestrator);
        let watcher = FileWatcher::new(orc.clone());
        let event_bus = HybridEmitter::new(Arc::new(EventBus::new(100)));
        Arc::new(super::super::handlers::ServerState {
            orchestrator: orc,
            watcher: Arc::new(tokio::sync::RwLock::new(watcher)),
            chat_manager: None,
            event_bus: Arc::new(event_bus),
            nats_emitter: None,
            auth_config: None,
            setup_completed: true,
            serve_frontend: false,
            frontend_path: String::new(),
            server_port: 0,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            identity: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: None,
            trajectory_store: None,
        })
    }

    /// Build a minimal OrchestratorState for handler tests.
    async fn make_test_server_state() -> OrchestratorState {
        build_server_state(MockGraphStore::new()).await
    }

    /// Build a test state with a pre-seeded project and protocol.
    async fn make_state_with_protocol() -> (OrchestratorState, Uuid, Uuid) {
        let mock = MockGraphStore::new();
        let mut project = test_project();
        let project_id = project.id;
        project.name = "test-project".to_string();
        project.slug = "test-project".to_string();
        mock.projects.write().await.insert(project_id, project);

        // Seed a protocol with states
        let start_id = Uuid::new_v4();
        let mid_id = Uuid::new_v4();
        let end_id = Uuid::new_v4();
        let proto = crate::protocol::Protocol::new(project_id, "existing-proto", start_id);
        let proto_id = proto.id;
        mock.protocols.write().await.insert(proto_id, proto);

        // Seed states
        for (id, name, st) in [
            (start_id, "start", StateType::Start),
            (mid_id, "work", StateType::Intermediate),
            (end_id, "done", StateType::Terminal),
        ] {
            let ps = ProtocolState {
                id,
                protocol_id: proto_id,
                name: name.to_string(),
                description: String::new(),
                action: None,
                state_type: st,
                sub_protocol_id: None,
                completion_strategy: None,
                on_failure_strategy: None,
                generator_config: None,
            };
            mock.protocol_states.write().await.insert(id, ps);
        }

        let state = build_server_state(mock).await;
        (state, project_id, proto_id)
    }

    #[tokio::test]
    async fn test_handler_create_protocol_name_uniqueness() {
        let (state, project_id, _proto_id) = make_state_with_protocol().await;

        // Try to create a protocol with the same name → should Conflict
        let body = CreateProtocolBody {
            project_id,
            name: "existing-proto".to_string(),
            description: None,
            skill_id: None,
            protocol_category: None,
            trigger_mode: None,
            trigger_config: None,
            relevance_vector: None,
            states: None,
            transitions: None,
        };
        let result = create_protocol(State(state.clone()), Json(body)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::Conflict(msg) => {
                assert!(msg.contains("existing-proto"));
                assert!(msg.contains("already exists"));
            }
            other => panic!("Expected Conflict, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handler_create_protocol_with_new_name_succeeds() {
        let (state, project_id, _) = make_state_with_protocol().await;

        let body = CreateProtocolBody {
            project_id,
            name: "brand-new-proto".to_string(),
            description: Some("A new protocol".to_string()),
            skill_id: None,
            protocol_category: None,
            trigger_mode: None,
            trigger_config: None,
            relevance_vector: None,
            states: None,
            transitions: None,
        };
        let result = create_protocol(State(state.clone()), Json(body)).await;
        assert!(result.is_ok(), "Create should succeed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_handler_add_transition_rejects_from_terminal() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;

        // Get the state IDs
        let states = state
            .orchestrator
            .neo4j()
            .get_protocol_states(proto_id)
            .await
            .unwrap();
        let terminal_id = states.iter().find(|s| s.name == "done").unwrap().id;
        let mid_id = states.iter().find(|s| s.name == "work").unwrap().id;

        let body = AddTransitionBody {
            from_state: terminal_id,
            to_state: mid_id,
            trigger: "go_back".to_string(),
            guard: None,
        };
        let result = add_transition(State(state.clone()), Path(proto_id), Json(body)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::BadRequest(msg) => {
                assert!(msg.contains("terminal state"));
            }
            other => panic!("Expected BadRequest, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handler_add_transition_rejects_to_start() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;

        let states = state
            .orchestrator
            .neo4j()
            .get_protocol_states(proto_id)
            .await
            .unwrap();
        let start_id = states.iter().find(|s| s.name == "start").unwrap().id;
        let mid_id = states.iter().find(|s| s.name == "work").unwrap().id;

        let body = AddTransitionBody {
            from_state: mid_id,
            to_state: start_id,
            trigger: "restart".to_string(),
            guard: None,
        };
        let result = add_transition(State(state.clone()), Path(proto_id), Json(body)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::BadRequest(msg) => {
                assert!(msg.contains("start state"));
            }
            other => panic!("Expected BadRequest, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handler_add_transition_succeeds() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;

        let states = state
            .orchestrator
            .neo4j()
            .get_protocol_states(proto_id)
            .await
            .unwrap();
        let start_id = states.iter().find(|s| s.name == "start").unwrap().id;
        let mid_id = states.iter().find(|s| s.name == "work").unwrap().id;

        let body = AddTransitionBody {
            from_state: start_id,
            to_state: mid_id,
            trigger: "begin".to_string(),
            guard: None,
        };
        let result = add_transition(State(state.clone()), Path(proto_id), Json(body)).await;
        assert!(result.is_ok(), "Should succeed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_handler_fire_transition_run_not_found() {
        let state = make_test_server_state().await;
        let fake_run_id = Uuid::new_v4();

        let body = FireTransitionBody {
            trigger: "go".to_string(),
        };
        let result = fire_transition(State(state), Path(fake_run_id), Json(body)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::NotFound(msg) => {
                assert!(msg.contains("not found"));
            }
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_handler_fire_transition_succeeds_with_event_emission() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;

        // Add a transition first
        let states = state
            .orchestrator
            .neo4j()
            .get_protocol_states(proto_id)
            .await
            .unwrap();
        let start_id = states.iter().find(|s| s.name == "start").unwrap().id;
        let mid_id = states.iter().find(|s| s.name == "work").unwrap().id;

        let transition = ProtocolTransition {
            id: Uuid::new_v4(),
            protocol_id: proto_id,
            from_state: start_id,
            to_state: mid_id,
            trigger: "begin".to_string(),
            guard: None,
        };
        state
            .orchestrator
            .neo4j()
            .upsert_protocol_transition(&transition)
            .await
            .unwrap();

        // Start a run
        let run = crate::protocol::engine::start_run(
            state.orchestrator.neo4j(),
            proto_id,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        // Fire transition via handler (covers event emission + trajectory collection)
        let body = FireTransitionBody {
            trigger: "begin".to_string(),
        };
        let result = fire_transition(State(state.clone()), Path(run.id), Json(body)).await;
        assert!(
            result.is_ok(),
            "Fire transition should succeed: {:?}",
            result.err()
        );
        let response = result.unwrap();
        assert!(response.success);
        assert_eq!(response.current_state_name, "work");
    }
}
