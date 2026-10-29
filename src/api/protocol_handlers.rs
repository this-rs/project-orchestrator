//! API handlers for Protocol (Pattern Federation)
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use super::handlers::{AppError, OrchestratorState};
use super::{PaginatedResponse, PaginationParams};
use crate::events::EventEmitter;
#[allow(unused_imports)]
use crate::protocol::{
    self, validate_protocol_structure, Protocol, ProtocolCategory, ProtocolRun, ProtocolState,
    ProtocolTransition, RunStatus, TransitionResult, ValidatableState, ValidatableTransition,
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
    /// Re-scope this protocol. Absent: leave it alone. `null`: make it
    /// transverse. A uuid: attach it to that project.
    #[serde(default, deserialize_with = "crate::api::double_option")]
    pub project_id: Option<Option<Uuid>>,
}

/// Request body for adding a state
#[derive(Debug, Deserialize)]
pub struct AddStateBody {
    pub name: String,
    pub description: Option<String>,
    pub state_type: Option<String>,
    pub action: Option<String>,
    /// Contextual prompt fragment injected when the agent is in this state.
    pub prompt_fragment: Option<String>,
    /// Whitelist of MCP tools allowed in this state.
    pub available_tools: Option<Vec<String>>,
    /// Actions explicitly forbidden in this state.
    pub forbidden_actions: Option<Vec<String>>,
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

/// Query parameters for listing runs across all protocols
#[derive(Debug, Deserialize, Default)]
pub struct AllRunsListQuery {
    #[serde(flatten)]
    pub pagination: PaginationParams,
    /// running | completed | failed | cancelled (omitted = every status)
    pub status: Option<String>,
    /// Restrict to the protocols of one project (wins over `workspace_slug`).
    pub project_id: Option<Uuid>,
    /// Restrict to the protocols of one workspace's projects.
    pub workspace_slug: Option<String>,
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

/// Query parameters for validating stored protocols
#[derive(Debug, Deserialize, Default)]
pub struct ValidateProtocolsQuery {
    /// Optional: restrict validation to a single project
    pub project_id: Option<Uuid>,
}

/// A protocol validation result
#[derive(Debug, Serialize, Deserialize)]
pub struct ProtocolValidationResult {
    pub protocol_id: Uuid,
    pub protocol_name: String,
    pub project_id: Option<Uuid>,
    pub violations: Vec<crate::protocol::ProtocolViolation>,
}

/// Response for protocol validation
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidateProtocolsResponse {
    /// Total number of protocols validated
    pub total_validated: usize,
    /// Number of protocols with violations
    pub invalid_count: usize,
    /// Number of protocols that passed validation
    pub valid_count: usize,
    /// Details for protocols with violations (empty if all valid)
    pub invalid_protocols: Vec<ProtocolValidationResult>,
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
        .list_protocols(Some(query.project_id), category_filter, limit, offset)
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
        .list_protocols(Some(query.project_id), None, 200, 0)
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
    let mut protocol = Protocol::new(Some(body.project_id), &body.name, placeholder_entry);
    protocol.description = body.description.unwrap_or_default();
    protocol.skill_id = body.skill_id;
    protocol.protocol_category = category;
    if let Some(ref tm) = body.trigger_mode {
        protocol.trigger_mode = tm
            .parse::<crate::protocol::TriggerMode>()
            .map_err(AppError::BadRequest)?;
    }
    if let Some(ref cfg) = body.trigger_config {
        cfg.validate().map_err(AppError::BadRequest)?;
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
                prompt_fragment: None,
                available_tools: None,
                forbidden_actions: None,
                state_timeout_secs: None,
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

    // Emit event for protocol creation
    state.event_bus.emit_created(
        crate::events::EntityType::Protocol,
        &protocol.id.to_string(),
        serde_json::json!({
            "name": protocol.name,
            "project_id": protocol.project_id,
            "category": protocol.protocol_category.to_string(),
            "states_count": created_states.len(),
            "transitions_count": created_transitions.len(),
        }),
        protocol.project_id.map(|id| id.to_string()),
    );

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
        trigger_config.validate().map_err(AppError::BadRequest)?;
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

    // Re-scope: absent leaves the scope alone, `null` makes it transverse, a
    // uuid attaches it. Same contract as the skill update.
    if let Some(scope) = body.project_id {
        state
            .orchestrator
            .neo4j()
            .rescope_protocol(protocol.id, scope)
            .await
            .map_err(AppError::Internal)?;
        protocol.project_id = scope;
    }

    // Emit event for protocol update
    state.event_bus.emit_updated(
        crate::events::EntityType::Protocol,
        &protocol.id.to_string(),
        serde_json::json!({
            "name": protocol.name,
            "project_id": protocol.project_id,
        }),
        protocol.project_id.map(|id| id.to_string()),
    );

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
        state.event_bus.emit_deleted(
            crate::events::EntityType::Protocol,
            &protocol_id.to_string(),
            None,
        );
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
        prompt_fragment: body.prompt_fragment,
        available_tools: body.available_tools,
        forbidden_actions: body.forbidden_actions,
        state_timeout_secs: None,
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

    // Emit event for state addition
    state.event_bus.emit_linked(
        crate::events::EntityType::Protocol,
        &protocol_id.to_string(),
        crate::events::EntityType::Protocol,
        &ps.id.to_string(),
        None,
    );

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
    Path((protocol_id, state_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_protocol_state(state_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        // Emit event for state removal
        state.event_bus.emit_unlinked(
            crate::events::EntityType::Protocol,
            &protocol_id.to_string(),
            crate::events::EntityType::Protocol,
            &state_id.to_string(),
            None,
        );
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

    // Emit event for transition addition
    state.event_bus.emit_linked(
        crate::events::EntityType::Protocol,
        &protocol_id.to_string(),
        crate::events::EntityType::Protocol,
        &pt.id.to_string(),
        None,
    );

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
    Path((protocol_id, transition_id)): Path<(Uuid, Uuid)>,
) -> Result<StatusCode, AppError> {
    let deleted = state
        .orchestrator
        .neo4j()
        .delete_protocol_transition(transition_id)
        .await
        .map_err(AppError::Internal)?;

    if deleted {
        // Emit event for transition removal
        state.event_bus.emit_unlinked(
            crate::events::EntityType::Protocol,
            &protocol_id.to_string(),
            crate::events::EntityType::Protocol,
            &transition_id.to_string(),
            None,
        );
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

    // Emit event for skill linking
    state.event_bus.emit_linked(
        crate::events::EntityType::Protocol,
        &protocol_id.to_string(),
        crate::events::EntityType::Skill,
        &body.skill_id.to_string(),
        protocol.project_id.map(|id| id.to_string()),
    );

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
    if let Some(ref collector) = *state.trajectory_collector.read().unwrap() {
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

/// List runs across ALL protocols (no protocol id needed)
///
/// GET /api/protocols/runs?status=running[&workspace_slug=..|&project_id=..]
pub async fn list_all_runs(
    State(state): State<OrchestratorState>,
    Query(query): Query<AllRunsListQuery>,
) -> Result<Json<PaginatedResponse<ProtocolRun>>, AppError> {
    // Unlike the per-protocol list, an unknown status is an error: silently
    // dropping it would answer "every run" to a request for "running" ones.
    let status_filter: Option<RunStatus> = match query
        .status
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(s) => Some(
            s.parse()
                .map_err(|_| AppError::BadRequest(format!("Invalid run status: {}", s)))?,
        ),
        None => None,
    };

    let limit = query.pagination.limit.clamp(1, 100);
    let offset = query.pagination.offset;

    let (runs, total) = state
        .orchestrator
        .neo4j()
        .list_all_protocol_runs(
            status_filter,
            query.project_id,
            super::handlers::normalize_slug(query.workspace_slug.as_deref()),
            limit,
            offset,
        )
        .await
        .map_err(AppError::Internal)?;

    let has_more = offset + runs.len() < total;
    Ok(Json(PaginatedResponse {
        items: runs,
        total,
        limit,
        offset,
        has_more,
    }))
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
    /// Contextual prompt fragment injected when the agent is in this state.
    pub prompt_fragment: Option<String>,
    /// Whitelist of MCP tools allowed in this state.
    pub available_tools: Option<Vec<String>>,
    /// Actions explicitly forbidden in this state.
    pub forbidden_actions: Option<Vec<String>>,
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

    // ── PYSR-4: Structural validation (via protocol::validate) ────────────
    // Convert request body to validatable types
    let validatable_states: Vec<ValidatableState<'_>> = body
        .states
        .iter()
        .map(|s| ValidatableState {
            name: s.name.as_str(),
            state_type: s
                .state_type
                .as_ref()
                .and_then(|t| t.parse().ok())
                .unwrap_or_default(),
        })
        .collect();

    let validatable_transitions: Vec<ValidatableTransition<'_>> = body
        .transitions
        .iter()
        .map(|t| ValidatableTransition {
            from_state: t.from_state.as_str(),
            to_state: t.to_state.as_str(),
            trigger: t.trigger.as_str(),
        })
        .collect();

    // Run structural validation — returns first violation as error for backward compat
    let violations = validate_protocol_structure(&validatable_states, &validatable_transitions);
    if let Some(violation) = violations.first() {
        return Err(AppError::BadRequest(violation.to_string()));
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
        project_id: Some(body.project_id),
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
    let mut proto = Protocol::new(Some(body.project_id), &body.name, placeholder_entry);
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
            prompt_fragment: s.prompt_fragment.clone(),
            available_tools: s.available_tools.clone(),
            forbidden_actions: s.forbidden_actions.clone(),
            state_timeout_secs: None,
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

    // Emit event for composed protocol creation
    state.event_bus.emit_created(
        crate::events::EntityType::Protocol,
        &proto.id.to_string(),
        serde_json::json!({
            "name": proto.name,
            "project_id": proto.project_id,
            "skill_id": skill_id,
            "states_created": created_states.len(),
            "transitions_created": created_transitions.len(),
            "notes_linked": notes_linked,
            "composed": true,
        }),
        proto.project_id.map(|id| id.to_string()),
    );

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

/// Validate all stored protocols for structural integrity.
///
/// GET /api/protocols/validate?project_id=...
///
/// Validates all protocols (or just those in the specified project) against
/// the structural rules:
/// - No duplicate transitions (same from_state + trigger)
/// - At least one terminal state
/// - All states reachable from start
///
/// Returns a summary with counts and details for any invalid protocols.
/// This is a read-only admin operation — it does not modify any protocols.
pub async fn validate_all_protocols(
    State(state): State<OrchestratorState>,
    Query(query): Query<ValidateProtocolsQuery>,
) -> Result<Json<ValidateProtocolsResponse>, AppError> {
    use crate::protocol::{ProtocolViolation, ValidatableState, ValidatableTransition};

    let neo4j = state.orchestrator.neo4j();

    // Collect all protocols to validate
    let protocols_to_validate: Vec<(Uuid, String, Option<Uuid>)> =
        if let Some(project_id) = query.project_id {
            // Single project
            let (protocols, _) = neo4j
                .list_protocols(Some(project_id), None, 1000, 0)
                .await
                .map_err(AppError::Internal)?;
            protocols
                .into_iter()
                .map(|p| (p.id, p.name, p.project_id))
                .collect()
        } else {
            // All projects
            let projects = neo4j.list_projects().await.map_err(AppError::Internal)?;
            let mut all_protocols = Vec::new();
            for project in projects {
                let (protocols, _) = neo4j
                    .list_protocols(Some(project.id), None, 1000, 0)
                    .await
                    .map_err(AppError::Internal)?;
                for p in protocols {
                    all_protocols.push((p.id, p.name, p.project_id));
                }
            }
            all_protocols
        };

    let total_validated = protocols_to_validate.len();
    let mut invalid_protocols = Vec::new();

    for (protocol_id, protocol_name, project_id) in protocols_to_validate {
        // Load states and transitions
        let states = neo4j
            .get_protocol_states(protocol_id)
            .await
            .map_err(AppError::Internal)?;
        let transitions = neo4j
            .get_protocol_transitions(protocol_id)
            .await
            .map_err(AppError::Internal)?;

        // Convert to validatable types
        let validatable_states: Vec<ValidatableState<'_>> = states
            .iter()
            .map(|s| ValidatableState {
                name: s.name.as_str(),
                state_type: s.state_type,
            })
            .collect();

        let validatable_transitions: Vec<ValidatableTransition<'_>> = transitions
            .iter()
            .map(|t| {
                // For stored protocols, transitions use UUIDs — we need to map to names
                let from_name = states
                    .iter()
                    .find(|s| s.id == t.from_state)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?");
                let to_name = states
                    .iter()
                    .find(|s| s.id == t.to_state)
                    .map(|s| s.name.as_str())
                    .unwrap_or("?");
                ValidatableTransition {
                    from_state: from_name,
                    to_state: to_name,
                    trigger: t.trigger.as_str(),
                }
            })
            .collect();

        // Run validation
        let violations: Vec<ProtocolViolation> =
            validate_protocol_structure(&validatable_states, &validatable_transitions);

        if !violations.is_empty() {
            invalid_protocols.push(ProtocolValidationResult {
                protocol_id,
                protocol_name,
                project_id,
                violations,
            });
        }
    }

    let invalid_count = invalid_protocols.len();
    let valid_count = total_validated - invalid_count;

    Ok(Json(ValidateProtocolsResponse {
        total_validated,
        invalid_count,
        valid_count,
        invalid_protocols,
    }))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
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
            prompt_fragment: None,
            available_tools: None,
            forbidden_actions: None,
            state_timeout_secs: None,
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
            remote_mcp: crate::RemoteMcpConfig::default(),
            ingress: Default::default(),
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
            vault: crate::vault::VaultService::ephemeral(),
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
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
        let proto = crate::protocol::Protocol::new(Some(project_id), "existing-proto", start_id);
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
                prompt_fragment: None,
                available_tools: None,
                forbidden_actions: None,
                state_timeout_secs: None,
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

    // ----------------------------------------------------------------
    // Handler coverage batch — CRUD handlers via direct calls against the
    // mock GraphStore (no DB). Each test gets its own isolated state.
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn test_handler_list_protocols_ok() {
        let (state, project_id, _) = make_state_with_protocol().await;
        let q = ProtocolsListQuery {
            pagination: PaginationParams::default(),
            project_id,
            category: None,
        };
        let res = list_protocols(State(state), axum::extract::Query(q)).await;
        assert!(res.is_ok(), "list_protocols failed: {:?}", res.err());
    }

    #[tokio::test]
    async fn test_handler_get_protocol_ok_and_not_found() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        assert!(get_protocol(State(state.clone()), Path(proto_id))
            .await
            .is_ok());
        let missing = get_protocol(State(state), Path(Uuid::new_v4())).await;
        assert!(matches!(missing, Err(AppError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_handler_list_states_returns_seeded_three() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let states = list_states(State(state), Path(proto_id)).await.unwrap().0;
        assert_eq!(states.len(), 3);
        assert!(states.iter().any(|s| s.name == "start"));
        assert!(states.iter().any(|s| s.name == "done"));
    }

    #[tokio::test]
    async fn test_handler_list_transitions_ok_empty() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let transitions = list_transitions(State(state), Path(proto_id))
            .await
            .unwrap()
            .0;
        assert!(transitions.is_empty());
    }

    #[tokio::test]
    async fn test_handler_add_state_then_listed() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let body = AddStateBody {
            name: "review".to_string(),
            description: Some("review step".to_string()),
            state_type: Some("intermediate".to_string()),
            action: None,
            prompt_fragment: None,
            available_tools: None,
            forbidden_actions: None,
        };
        let added = add_state(State(state.clone()), Path(proto_id), Json(body)).await;
        assert!(added.is_ok(), "add_state failed: {:?}", added.err());
        let states = list_states(State(state), Path(proto_id)).await.unwrap().0;
        assert_eq!(states.len(), 4);
        assert!(states.iter().any(|s| s.name == "review"));
    }

    #[tokio::test]
    async fn test_handler_delete_state_ok() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let states = list_states(State(state.clone()), Path(proto_id))
            .await
            .unwrap()
            .0;
        let target = states.iter().find(|s| s.name == "work").unwrap().id;
        let res = delete_state(State(state.clone()), Path((proto_id, target))).await;
        assert!(res.is_ok(), "delete_state failed: {:?}", res.err());
        let after = list_states(State(state), Path(proto_id)).await.unwrap().0;
        assert_eq!(after.len(), 2);
    }

    /// Same tri-state on the protocol side: `null` makes it transverse,
    /// absent leaves it attached. Both share `double_option`.
    #[tokio::test]
    async fn a_null_project_id_makes_the_protocol_transverse() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let body = UpdateProtocolBody {
            project_id: Some(None),
            name: None,
            description: None,
            protocol_category: None,
            trigger_mode: None,
            trigger_config: None,
            relevance_vector: None,
        };
        let res = update_protocol(State(state), Path(proto_id), Json(body)).await;
        let protocol = res.expect("update_protocol failed").0;
        assert!(
            protocol.project_id.is_none(),
            "null must make the protocol transverse"
        );
    }

    #[tokio::test]
    async fn an_absent_project_id_leaves_the_protocol_attached() {
        let (state, project_id, proto_id) = make_state_with_protocol().await;
        let body = UpdateProtocolBody {
            project_id: None,
            name: Some("renamed-proto".to_string()),
            description: None,
            protocol_category: None,
            trigger_mode: None,
            trigger_config: None,
            relevance_vector: None,
        };
        let res = update_protocol(State(state), Path(proto_id), Json(body)).await;
        let protocol = res.expect("update_protocol failed").0;
        assert_eq!(
            protocol.project_id,
            Some(project_id),
            "an absent project_id must not detach the protocol"
        );
    }

    #[tokio::test]
    async fn test_handler_update_protocol_ok() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let body = UpdateProtocolBody {
            project_id: None,
            name: Some("renamed-proto".to_string()),
            description: Some("updated".to_string()),
            protocol_category: None,
            trigger_mode: None,
            trigger_config: None,
            relevance_vector: None,
        };
        let res = update_protocol(State(state), Path(proto_id), Json(body)).await;
        assert!(res.is_ok(), "update_protocol failed: {:?}", res.err());
        assert_eq!(res.unwrap().0.name, "renamed-proto");
    }

    #[tokio::test]
    async fn test_handler_delete_protocol_ok() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let res = delete_protocol(State(state.clone()), Path(proto_id)).await;
        assert!(res.is_ok(), "delete_protocol failed: {:?}", res.err());
        // After deletion the protocol is gone.
        let gone = get_protocol(State(state), Path(proto_id)).await;
        assert!(matches!(gone, Err(AppError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_handler_get_run_not_found() {
        let (state, _project_id, _proto_id) = make_state_with_protocol().await;
        let res = get_run(State(state), Path(Uuid::new_v4())).await;
        assert!(matches!(res, Err(AppError::NotFound(_))));
    }

    #[tokio::test]
    async fn test_handler_list_runs_ok_empty() {
        let (state, _project_id, proto_id) = make_state_with_protocol().await;
        let q = RunsListQuery {
            pagination: PaginationParams::default(),
            status: None,
        };
        let res = list_runs(State(state), Path(proto_id), axum::extract::Query(q)).await;
        assert!(res.is_ok(), "list_runs failed: {:?}", res.err());
    }

    // ----------------------------------------------------------------
    // schedule validation (REST)
    // ----------------------------------------------------------------

    async fn send(
        app: &axum::Router,
        method: &str,
        uri: &str,
        body: serde_json::Value,
    ) -> (axum::http::StatusCode, serde_json::Value) {
        use tower::ServiceExt;
        let resp = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("authorization", crate::test_helpers::test_bearer_token())
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test]
    async fn create_protocol_refuses_an_unknown_schedule_with_a_400_listing_the_values() {
        let graph = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;
        let (status, body) = send(
            &app,
            "POST",
            "/api/protocols",
            serde_json::json!({
                "project_id": Uuid::new_v4(),
                "name": "fsm-watchdog",
                "trigger_mode": "scheduled",
                "trigger_config": {"schedule": "6h"},
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{body}");
        let text = body.to_string();
        assert!(text.contains("6h"), "{text}");
        for accepted in ["hourly", "daily", "weekly"] {
            assert!(text.contains(accepted), "{text}");
        }
    }

    #[tokio::test]
    async fn create_protocol_accepts_a_known_schedule() {
        let graph = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;
        let (status, body) = send(
            &app,
            "POST",
            "/api/protocols",
            serde_json::json!({
                "project_id": Uuid::new_v4(),
                "name": "nightly",
                "trigger_mode": "scheduled",
                "trigger_config": {"schedule": "daily"},
            }),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::CREATED, "{body}");
    }

    #[tokio::test]
    async fn update_protocol_refuses_an_unknown_schedule_and_keeps_the_stored_one() {
        let graph = std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let project_id = Uuid::new_v4();
        let mut p = Protocol::new(Some(project_id), "nightly", Uuid::new_v4());
        p.trigger_config = Some(crate::protocol::TriggerConfig {
            schedule: Some("daily".to_string()),
            ..Default::default()
        });
        graph.upsert_protocol(&p).await.unwrap();
        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;

        let (status, body) = send(
            &app,
            "PUT",
            &format!("/api/protocols/{}", p.id),
            serde_json::json!({"trigger_config": {"schedule": "6h"}}),
        )
        .await;

        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{body}");
        let stored = graph.get_protocol(p.id).await.unwrap().unwrap();
        assert_eq!(
            stored.trigger_config.unwrap().schedule.as_deref(),
            Some("daily")
        );
    }

    // ================================================================
    // PYSR-4: RED TESTS — missing structural validation in compose_protocol
    //
    // These tests document CURRENT BROKEN BEHAVIOR: invalid protocols
    // are accepted when they should be rejected. When validation is
    // added (step 3), these tests WILL FAIL and need to change from
    // assert_eq!(status, CREATED) to assert_eq!(status, BAD_REQUEST).
    // ================================================================

    /// RED TEST: A state unreachable from the initial state should be rejected.
    /// Current behavior: accepted (no reachability check).
    #[tokio::test]
    async fn compose_protocol_rejects_unreachable_state() {
        let graph = std::sync::Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();
        let mut project = test_project();
        project.id = project_id;
        graph.projects.write().await.insert(project_id, project);
        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;

        // Protocol: start → done, but "orphan" has no incoming transition
        // (unreachable from start)
        let (status, body) = send(
            &app,
            "POST",
            "/api/protocols/compose",
            serde_json::json!({
                "project_id": project_id,
                "name": "unreachable-state-proto",
                "states": [
                    {"name": "start", "state_type": "start"},
                    {"name": "done", "state_type": "terminal"},
                    {"name": "orphan", "state_type": "intermediate"}  // unreachable!
                ],
                "transitions": [
                    {"from_state": "start", "to_state": "done", "trigger": "finish"}
                ]
            }),
        )
        .await;

        // PYSR-4 step 3: validation now rejects unreachable states
        assert_eq!(
            status,
            axum::http::StatusCode::BAD_REQUEST,
            "unreachable state 'orphan' should be rejected: {body}"
        );
        assert!(
            body.to_string().contains("unreachable"),
            "error should mention 'unreachable': {body}"
        );
    }

    /// Protocol with no terminal state is rejected.
    /// PYSR-4: terminal state validation added.
    #[tokio::test]
    async fn compose_protocol_rejects_no_terminal_state() {
        let graph = std::sync::Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();
        let mut project = test_project();
        project.id = project_id;
        graph.projects.write().await.insert(project_id, project);
        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;

        // Protocol: start → work1 → work2 (chain with no terminal state)
        // All states are intermediate or start — no terminal to end the protocol.
        let (status, body) = send(
            &app,
            "POST",
            "/api/protocols/compose",
            serde_json::json!({
                "project_id": project_id,
                "name": "no-terminal-proto",
                "states": [
                    {"name": "start", "state_type": "start"},
                    {"name": "work1", "state_type": "intermediate"},
                    {"name": "work2", "state_type": "intermediate"}
                ],
                "transitions": [
                    {"from_state": "start", "to_state": "work1", "trigger": "begin"},
                    {"from_state": "work1", "to_state": "work2", "trigger": "proceed"}
                ]
            }),
        )
        .await;

        // PYSR-4 step 3: validation now requires at least one terminal state
        assert_eq!(
            status,
            axum::http::StatusCode::BAD_REQUEST,
            "protocol without terminal state should be rejected: {body}"
        );
        assert!(
            body.to_string().contains("terminal"),
            "error should mention 'terminal': {body}"
        );
    }

    /// Duplicate transitions (same from_state + trigger) are rejected.
    /// PYSR-4: duplicate transition validation added.
    #[tokio::test]
    async fn compose_protocol_rejects_duplicate_transitions() {
        let graph = std::sync::Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();
        let mut project = test_project();
        project.id = project_id;
        graph.projects.write().await.insert(project_id, project);
        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;

        // Protocol: from "start" with trigger "go" → both "a" AND "b"
        // This is non-deterministic: which state do we go to on "go"?
        let (status, body) = send(
            &app,
            "POST",
            "/api/protocols/compose",
            serde_json::json!({
                "project_id": project_id,
                "name": "duplicate-transitions-proto",
                "states": [
                    {"name": "start", "state_type": "start"},
                    {"name": "a", "state_type": "intermediate"},
                    {"name": "b", "state_type": "intermediate"},
                    {"name": "done", "state_type": "terminal"}
                ],
                "transitions": [
                    {"from_state": "start", "to_state": "a", "trigger": "go"},  // dup!
                    {"from_state": "start", "to_state": "b", "trigger": "go"},  // dup!
                    {"from_state": "a", "to_state": "done", "trigger": "finish"},
                    {"from_state": "b", "to_state": "done", "trigger": "finish"}
                ]
            }),
        )
        .await;

        // PYSR-4 step 3: validation now rejects duplicate transitions
        assert_eq!(
            status,
            axum::http::StatusCode::BAD_REQUEST,
            "duplicate transitions should be rejected: {body}"
        );
        assert!(
            body.to_string().contains("duplicate"),
            "error should mention 'duplicate': {body}"
        );
    }

    // ----------------------------------------------------------------
    // validate_all_protocols (PYSR-4 criterion 4)
    // ----------------------------------------------------------------

    /// Stored protocol with unreachable state appears in validation report.
    /// PYSR-4 criterion 4: validate stored protocols.
    #[tokio::test]
    async fn validate_all_reports_unreachable_state() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::{Protocol, ProtocolState, ProtocolTransition};

        let graph = std::sync::Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();
        let mut project = test_project();
        project.id = project_id;
        graph.projects.write().await.insert(project_id, project);

        // Create state IDs first (needed for Protocol::new)
        let start_id = Uuid::new_v4();
        let done_id = Uuid::new_v4();
        let orphan_id = Uuid::new_v4();

        // Create a protocol with an unreachable state directly in the store
        let mut protocol = Protocol::new(Some(project_id), "invalid-proto", start_id);
        protocol.description = "Protocol with unreachable state".to_string();
        protocol.terminal_states = vec![done_id];
        let protocol_id = protocol.id;
        graph.protocols.write().await.insert(protocol_id, protocol);

        // Add states: start, done (terminal), orphan (unreachable)
        let start_state = ProtocolState {
            id: start_id,
            protocol_id,
            name: "start".to_string(),
            description: String::new(),
            action: None,
            state_type: StateType::Start,
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
            prompt_fragment: None,
            available_tools: None,
            forbidden_actions: None,
            state_timeout_secs: None,
        };
        let done_state = ProtocolState {
            id: done_id,
            protocol_id,
            name: "done".to_string(),
            state_type: StateType::Terminal,
            ..start_state.clone()
        };
        let orphan_state = ProtocolState {
            id: orphan_id,
            protocol_id,
            name: "orphan".to_string(),
            state_type: StateType::Intermediate,
            ..start_state.clone()
        };

        {
            let mut states = graph.protocol_states.write().await;
            states.insert(start_id, start_state);
            states.insert(done_id, done_state);
            states.insert(orphan_id, orphan_state);
        }

        // Add transition: start → done (orphan is unreachable)
        let transition = ProtocolTransition {
            id: Uuid::new_v4(),
            protocol_id,
            from_state: start_id,
            to_state: done_id,
            trigger: "finish".to_string(),
            guard: None,
        };
        graph
            .protocol_transitions
            .write()
            .await
            .insert(transition.id, transition);

        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;

        // Call validate endpoint
        let (status, body) = send(
            &app,
            "GET",
            &format!("/api/protocols/validate?project_id={}", project_id),
            serde_json::json!({}),
        )
        .await;

        assert_eq!(
            status,
            axum::http::StatusCode::OK,
            "validate should succeed: {body}"
        );

        let response: ValidateProtocolsResponse =
            serde_json::from_value(body.clone()).expect("should parse response");

        assert_eq!(response.total_validated, 1, "should validate 1 protocol");
        assert_eq!(response.invalid_count, 1, "should find 1 invalid protocol");
        assert_eq!(response.valid_count, 0, "should find 0 valid protocols");
        assert_eq!(response.invalid_protocols.len(), 1);

        let invalid = &response.invalid_protocols[0];
        assert_eq!(invalid.protocol_name, "invalid-proto");
        assert!(
            invalid
                .violations
                .iter()
                .any(|v| matches!(v, crate::protocol::ProtocolViolation::UnreachableState { state } if state == "orphan")),
            "should report 'orphan' as unreachable: {:?}",
            invalid.violations
        );
    }

    /// Valid stored protocol does NOT appear in validation report.
    #[tokio::test]
    async fn validate_all_valid_protocol_not_in_report() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::protocol::{Protocol, ProtocolState, ProtocolTransition};

        let graph = std::sync::Arc::new(MockGraphStore::new());
        let project_id = Uuid::new_v4();
        let mut project = test_project();
        project.id = project_id;
        graph.projects.write().await.insert(project_id, project);

        // Create state IDs first (needed for Protocol::new)
        let start_id = Uuid::new_v4();
        let work_id = Uuid::new_v4();
        let done_id = Uuid::new_v4();

        // Create a valid protocol
        let mut protocol = Protocol::new(Some(project_id), "valid-proto", start_id);
        protocol.description = "Valid protocol".to_string();
        protocol.terminal_states = vec![done_id];
        let protocol_id = protocol.id;
        graph.protocols.write().await.insert(protocol_id, protocol);

        // Add states: start → work → done (all reachable, has terminal)

        let start_state = ProtocolState {
            id: start_id,
            protocol_id,
            name: "start".to_string(),
            description: String::new(),
            action: None,
            state_type: StateType::Start,
            sub_protocol_id: None,
            completion_strategy: None,
            on_failure_strategy: None,
            generator_config: None,
            prompt_fragment: None,
            available_tools: None,
            forbidden_actions: None,
            state_timeout_secs: None,
        };
        let work_state = ProtocolState {
            id: work_id,
            protocol_id,
            name: "work".to_string(),
            state_type: StateType::Intermediate,
            ..start_state.clone()
        };
        let done_state = ProtocolState {
            id: done_id,
            protocol_id,
            name: "done".to_string(),
            state_type: StateType::Terminal,
            ..start_state.clone()
        };

        {
            let mut states = graph.protocol_states.write().await;
            states.insert(start_id, start_state);
            states.insert(work_id, work_state);
            states.insert(done_id, done_state);
        }

        // Add transitions: start → work → done
        let t1 = ProtocolTransition {
            id: Uuid::new_v4(),
            protocol_id,
            from_state: start_id,
            to_state: work_id,
            trigger: "begin".to_string(),
            guard: None,
        };
        let t2 = ProtocolTransition {
            id: Uuid::new_v4(),
            protocol_id,
            from_state: work_id,
            to_state: done_id,
            trigger: "finish".to_string(),
            guard: None,
        };
        {
            let mut transitions = graph.protocol_transitions.write().await;
            transitions.insert(t1.id, t1);
            transitions.insert(t2.id, t2);
        }

        let app = crate::api::list_routes_tests::router_over(graph.clone()).await;

        // Call validate endpoint
        let (status, body) = send(
            &app,
            "GET",
            &format!("/api/protocols/validate?project_id={}", project_id),
            serde_json::json!({}),
        )
        .await;

        assert_eq!(
            status,
            axum::http::StatusCode::OK,
            "validate should succeed: {body}"
        );

        let response: ValidateProtocolsResponse =
            serde_json::from_value(body.clone()).expect("should parse response");

        assert_eq!(response.total_validated, 1, "should validate 1 protocol");
        assert_eq!(response.invalid_count, 0, "should find 0 invalid protocols");
        assert_eq!(response.valid_count, 1, "should find 1 valid protocol");
        assert!(
            response.invalid_protocols.is_empty(),
            "valid protocol should not appear in invalid list"
        );
    }
}
