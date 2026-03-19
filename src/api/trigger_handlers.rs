//! API handlers for EventTriggers

use super::handlers::{AppError, OrchestratorState};
use crate::events::trigger::EventTrigger;
use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
use axum::{
    extract::{Path, Query, State},
    Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for listing triggers
#[derive(Debug, Deserialize, Default)]
pub struct TriggersListQuery {
    /// Optional project_id filter
    pub project_id: Option<Uuid>,
}

// ============================================================================
// Request / Response types
// ============================================================================

/// Request body for creating a trigger
#[derive(Debug, Deserialize)]
pub struct CreateTriggerRequest {
    pub name: String,
    pub protocol_id: Uuid,
    pub entity_type_pattern: Option<String>,
    pub action_pattern: Option<String>,
    pub payload_conditions: Option<serde_json::Value>,
    #[serde(default)]
    pub cooldown_secs: u32,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub project_scope: Option<Uuid>,
}

fn default_true() -> bool {
    true
}

/// Request body for updating a trigger
#[derive(Debug, Deserialize)]
pub struct UpdateTriggerRequest {
    pub name: Option<String>,
    pub enabled: Option<bool>,
    pub entity_type_pattern: Option<Option<String>>,
    pub action_pattern: Option<Option<String>>,
    pub payload_conditions: Option<Option<serde_json::Value>>,
    pub cooldown_secs: Option<u32>,
    pub project_scope: Option<Option<Uuid>>,
}

/// Trigger stats response
#[derive(Debug, Serialize)]
pub struct TriggerStatsResponse {
    pub total: usize,
    pub enabled: usize,
    pub disabled: usize,
    pub by_entity_type: Vec<EntityTypeCount>,
}

#[derive(Debug, Serialize)]
pub struct EntityTypeCount {
    pub entity_type: String,
    pub count: usize,
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /api/triggers — list all event triggers (with optional ?project_id= filter)
pub async fn list_triggers(
    State(state): State<OrchestratorState>,
    Query(query): Query<TriggersListQuery>,
) -> Result<Json<Vec<EventTrigger>>, AppError> {
    let triggers = state
        .orchestrator
        .neo4j()
        .list_event_triggers(query.project_id, false)
        .await?;
    Ok(Json(triggers))
}

/// POST /api/triggers — create a new event trigger
pub async fn create_trigger(
    State(state): State<OrchestratorState>,
    Json(body): Json<CreateTriggerRequest>,
) -> Result<Json<EventTrigger>, AppError> {
    let now = Utc::now();
    let trigger = EventTrigger {
        id: Uuid::new_v4(),
        name: body.name,
        protocol_id: body.protocol_id,
        entity_type_pattern: body.entity_type_pattern,
        action_pattern: body.action_pattern,
        payload_conditions: body.payload_conditions,
        cooldown_secs: body.cooldown_secs,
        enabled: body.enabled,
        project_scope: body.project_scope,
        created_at: now,
        updated_at: now,
    };

    state
        .orchestrator
        .neo4j()
        .create_event_trigger(&trigger)
        .await?;

    // Emit CRUD event
    state.event_bus.emit(
        CrudEvent::new(
            EntityType::Trigger,
            CrudAction::Created,
            trigger.id.to_string(),
        )
        .with_payload(serde_json::json!({
            "name": trigger.name,
            "protocol_id": trigger.protocol_id.to_string(),
        })),
    );

    Ok(Json(trigger))
}

/// GET /api/triggers/:id — get a trigger by ID
pub async fn get_trigger(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<Json<EventTrigger>, AppError> {
    let trigger = state
        .orchestrator
        .neo4j()
        .get_event_trigger(id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Trigger {} not found", id)))?;
    Ok(Json(trigger))
}

/// PUT /api/triggers/:id — update a trigger (enable/disable, modify patterns)
pub async fn update_trigger(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
    Json(body): Json<UpdateTriggerRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let updated = state
        .orchestrator
        .neo4j()
        .update_event_trigger(
            id,
            body.enabled,
            body.name,
            body.entity_type_pattern,
            body.action_pattern,
            body.payload_conditions,
            body.cooldown_secs,
            body.project_scope,
        )
        .await?;

    if !updated {
        return Err(AppError::NotFound(format!("Trigger {} not found", id)));
    }

    // Emit CRUD event
    state.event_bus.emit(CrudEvent::new(
        EntityType::Trigger,
        CrudAction::Updated,
        id.to_string(),
    ));

    Ok(Json(serde_json::json!({ "ok": true })))
}

/// DELETE /api/triggers/:id — delete a trigger
pub async fn delete_trigger(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = state.orchestrator.neo4j().delete_event_trigger(id).await?;

    if !deleted {
        return Err(AppError::NotFound(format!("Trigger {} not found", id)));
    }

    // Emit CRUD event
    state.event_bus.emit(CrudEvent::new(
        EntityType::Trigger,
        CrudAction::Deleted,
        id.to_string(),
    ));

    Ok(Json(serde_json::json!({ "ok": true })))
}

/// POST /api/triggers/:id/enable — enable a trigger
pub async fn enable_trigger(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let updated = state
        .orchestrator
        .neo4j()
        .update_event_trigger(id, Some(true), None, None, None, None, None, None)
        .await?;

    if !updated {
        return Err(AppError::NotFound(format!("Trigger {} not found", id)));
    }

    state.event_bus.emit(
        CrudEvent::new(EntityType::Trigger, CrudAction::Updated, id.to_string())
            .with_payload(serde_json::json!({ "enabled": true })),
    );

    Ok(Json(serde_json::json!({ "ok": true, "enabled": true })))
}

/// POST /api/triggers/:id/disable — disable a trigger
pub async fn disable_trigger(
    State(state): State<OrchestratorState>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let updated = state
        .orchestrator
        .neo4j()
        .update_event_trigger(id, Some(false), None, None, None, None, None, None)
        .await?;

    if !updated {
        return Err(AppError::NotFound(format!("Trigger {} not found", id)));
    }

    state.event_bus.emit(
        CrudEvent::new(EntityType::Trigger, CrudAction::Updated, id.to_string())
            .with_payload(serde_json::json!({ "enabled": false })),
    );

    Ok(Json(serde_json::json!({ "ok": true, "enabled": false })))
}

/// GET /api/triggers/stats — get trigger activation stats
pub async fn trigger_stats(
    State(state): State<OrchestratorState>,
) -> Result<Json<TriggerStatsResponse>, AppError> {
    let triggers = state
        .orchestrator
        .neo4j()
        .list_event_triggers(None, false)
        .await?;

    let total = triggers.len();
    let enabled = triggers.iter().filter(|t| t.enabled).count();
    let disabled = total - enabled;

    // Count by entity_type_pattern
    let mut type_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for t in &triggers {
        let key = t
            .entity_type_pattern
            .clone()
            .unwrap_or_else(|| "*".to_string());
        *type_counts.entry(key).or_insert(0) += 1;
    }

    let by_entity_type: Vec<EntityTypeCount> = type_counts
        .into_iter()
        .map(|(entity_type, count)| EntityTypeCount { entity_type, count })
        .collect();

    Ok(Json(TriggerStatsResponse {
        total,
        enabled,
        disabled,
        by_entity_type,
    }))
}
