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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::mock_app_state;
    use axum::extract::State;
    use std::sync::Arc;

    /// Build a minimal `OrchestratorState` for handler-level tests.
    async fn test_state() -> OrchestratorState {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: None,
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 0,
            public_url: None,
            ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
            registry_remote_url: None,
            oidc_client: None,
            neural_router: crate::test_helpers::mock_neural_router(),
            trajectory_collector: std::sync::RwLock::new(None),
            trajectory_store_neo4j: None,
            trajectory_store: None,
            identity: None,
            reactor_counters: std::sync::OnceLock::new(),
            confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
        })
    }

    // ── Serialization / Deserialization tests ────────────────────────────

    #[test]
    fn test_create_trigger_request_full() {
        let project_id = Uuid::new_v4();
        let protocol_id = Uuid::new_v4();
        let json = serde_json::json!({
            "name": "on-task-created",
            "protocol_id": protocol_id.to_string(),
            "entity_type_pattern": "task",
            "action_pattern": "created",
            "payload_conditions": { "priority": "high" },
            "cooldown_secs": 60,
            "enabled": false,
            "project_scope": project_id.to_string(),
        });
        let req: CreateTriggerRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.name, "on-task-created");
        assert_eq!(req.protocol_id, protocol_id);
        assert_eq!(req.entity_type_pattern.as_deref(), Some("task"));
        assert_eq!(req.action_pattern.as_deref(), Some("created"));
        assert!(req.payload_conditions.is_some());
        assert_eq!(req.cooldown_secs, 60);
        assert!(!req.enabled);
        assert_eq!(req.project_scope, Some(project_id));
    }

    #[test]
    fn test_create_trigger_request_minimal() {
        let protocol_id = Uuid::new_v4();
        let json = serde_json::json!({
            "name": "my-trigger",
            "protocol_id": protocol_id.to_string(),
        });
        let req: CreateTriggerRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.name, "my-trigger");
        assert_eq!(req.protocol_id, protocol_id);
        assert!(req.entity_type_pattern.is_none());
        assert!(req.action_pattern.is_none());
        assert!(req.payload_conditions.is_none());
        assert_eq!(req.cooldown_secs, 0); // serde default
        assert!(req.enabled); // default_true
        assert!(req.project_scope.is_none());
    }

    #[test]
    fn test_create_trigger_request_missing_name() {
        let json = serde_json::json!({
            "protocol_id": Uuid::new_v4().to_string(),
        });
        assert!(serde_json::from_value::<CreateTriggerRequest>(json).is_err());
    }

    #[test]
    fn test_create_trigger_request_missing_protocol_id() {
        let json = serde_json::json!({
            "name": "my-trigger",
        });
        assert!(serde_json::from_value::<CreateTriggerRequest>(json).is_err());
    }

    #[test]
    fn test_update_trigger_request_full() {
        let scope_id = Uuid::new_v4();
        let json = serde_json::json!({
            "name": "renamed",
            "enabled": false,
            "entity_type_pattern": "project",
            "action_pattern": "synced",
            "payload_conditions": { "status": "done" },
            "cooldown_secs": 120,
            "project_scope": scope_id.to_string(),
        });
        let req: UpdateTriggerRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.name.as_deref(), Some("renamed"));
        assert_eq!(req.enabled, Some(false));
        assert_eq!(req.entity_type_pattern, Some(Some("project".to_string())));
        assert_eq!(req.action_pattern, Some(Some("synced".to_string())));
        assert!(req.payload_conditions.is_some());
        assert_eq!(req.cooldown_secs, Some(120));
        assert_eq!(req.project_scope, Some(Some(scope_id)));
    }

    #[test]
    fn test_update_trigger_request_null_clears_field() {
        // JSON null on an Option<Option<T>> field deserializes to None (absent),
        // not Some(None) (explicitly cleared). This matches serde's default behavior.
        let json = serde_json::json!({
            "action_pattern": null,
            "entity_type_pattern": null,
        });
        let req: UpdateTriggerRequest = serde_json::from_value(json).unwrap();
        // serde treats null as None for the outer Option
        assert!(req.action_pattern.is_none());
        assert!(req.entity_type_pattern.is_none());
    }

    #[test]
    fn test_update_trigger_request_empty() {
        let json = serde_json::json!({});
        let req: UpdateTriggerRequest = serde_json::from_value(json).unwrap();
        assert!(req.name.is_none());
        assert!(req.enabled.is_none());
        assert!(req.entity_type_pattern.is_none());
        assert!(req.action_pattern.is_none());
        assert!(req.payload_conditions.is_none());
        assert!(req.cooldown_secs.is_none());
        assert!(req.project_scope.is_none());
    }

    #[test]
    fn test_triggers_list_query_defaults() {
        let q: TriggersListQuery = serde_json::from_str("{}").unwrap();
        assert!(q.project_id.is_none());
    }

    #[test]
    fn test_triggers_list_query_with_project_id() {
        let id = Uuid::new_v4();
        let json = format!(r#"{{"project_id":"{}"}}"#, id);
        let q: TriggersListQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.project_id, Some(id));
    }

    #[test]
    fn test_trigger_stats_response_serialization() {
        let resp = TriggerStatsResponse {
            total: 5,
            enabled: 3,
            disabled: 2,
            by_entity_type: vec![
                EntityTypeCount {
                    entity_type: "task".to_string(),
                    count: 3,
                },
                EntityTypeCount {
                    entity_type: "*".to_string(),
                    count: 2,
                },
            ],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["total"], 5);
        assert_eq!(json["enabled"], 3);
        assert_eq!(json["disabled"], 2);
        let by_type = json["by_entity_type"].as_array().unwrap();
        assert_eq!(by_type.len(), 2);
        assert_eq!(by_type[0]["entity_type"], "task");
        assert_eq!(by_type[0]["count"], 3);
    }

    #[test]
    fn test_entity_type_count_serialization() {
        let item = EntityTypeCount {
            entity_type: "milestone".to_string(),
            count: 7,
        };
        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["entity_type"], "milestone");
        assert_eq!(json["count"], 7);
    }

    // ── Handler tests ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_triggers_empty() {
        let state = test_state().await;
        let query = TriggersListQuery { project_id: None };
        let result = list_triggers(State(state), Query(query)).await;
        assert!(result.is_ok());
        let triggers = result.unwrap().0;
        assert!(triggers.is_empty());
    }

    #[tokio::test]
    async fn test_list_triggers_with_project_filter() {
        let state = test_state().await;
        let query = TriggersListQuery {
            project_id: Some(Uuid::new_v4()),
        };
        let result = list_triggers(State(state), Query(query)).await;
        assert!(result.is_ok());
        assert!(result.unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn test_create_trigger_ok() {
        let state = test_state().await;
        let protocol_id = Uuid::new_v4();
        let body = CreateTriggerRequest {
            name: "test-trigger".to_string(),
            protocol_id,
            entity_type_pattern: Some("task".to_string()),
            action_pattern: Some("created".to_string()),
            payload_conditions: None,
            cooldown_secs: 30,
            enabled: true,
            project_scope: None,
        };
        let result = create_trigger(State(state), Json(body)).await;
        assert!(result.is_ok());
        let trigger = result.unwrap().0;
        assert_eq!(trigger.name, "test-trigger");
        assert_eq!(trigger.protocol_id, protocol_id);
        assert_eq!(trigger.entity_type_pattern.as_deref(), Some("task"));
        assert_eq!(trigger.action_pattern.as_deref(), Some("created"));
        assert_eq!(trigger.cooldown_secs, 30);
        assert!(trigger.enabled);
        assert!(trigger.project_scope.is_none());
        // id should be a valid UUID
        assert!(!trigger.id.is_nil());
        // timestamps should be set
        assert!(trigger.created_at <= Utc::now());
        assert_eq!(trigger.created_at, trigger.updated_at);
    }

    #[tokio::test]
    async fn test_create_trigger_minimal_fields() {
        let state = test_state().await;
        let body = CreateTriggerRequest {
            name: "minimal".to_string(),
            protocol_id: Uuid::new_v4(),
            entity_type_pattern: None,
            action_pattern: None,
            payload_conditions: None,
            cooldown_secs: 0,
            enabled: true,
            project_scope: None,
        };
        let result = create_trigger(State(state), Json(body)).await;
        assert!(result.is_ok());
        let trigger = result.unwrap().0;
        assert_eq!(trigger.name, "minimal");
        assert!(trigger.entity_type_pattern.is_none());
        assert!(trigger.action_pattern.is_none());
        assert!(trigger.payload_conditions.is_none());
        assert_eq!(trigger.cooldown_secs, 0);
    }

    #[tokio::test]
    async fn test_get_trigger_not_found() {
        // MockGraphStore.get_event_trigger always returns None
        let state = test_state().await;
        let id = Uuid::new_v4();
        let result = get_trigger(State(state), Path(id)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::NotFound(msg) => {
                assert!(msg.contains(&id.to_string()));
            }
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_update_trigger_not_found() {
        // MockGraphStore.update_event_trigger always returns Ok(false)
        let state = test_state().await;
        let id = Uuid::new_v4();
        let body = UpdateTriggerRequest {
            name: Some("new-name".to_string()),
            enabled: None,
            entity_type_pattern: None,
            action_pattern: None,
            payload_conditions: None,
            cooldown_secs: None,
            project_scope: None,
        };
        let result = update_trigger(State(state), Path(id), Json(body)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::NotFound(msg) => {
                assert!(msg.contains(&id.to_string()));
            }
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_delete_trigger_not_found() {
        // MockGraphStore.delete_event_trigger always returns Ok(false)
        let state = test_state().await;
        let id = Uuid::new_v4();
        let result = delete_trigger(State(state), Path(id)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::NotFound(msg) => {
                assert!(msg.contains(&id.to_string()));
            }
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_enable_trigger_not_found() {
        // MockGraphStore.update_event_trigger always returns Ok(false)
        let state = test_state().await;
        let id = Uuid::new_v4();
        let result = enable_trigger(State(state), Path(id)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::NotFound(msg) => {
                assert!(msg.contains(&id.to_string()));
            }
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_disable_trigger_not_found() {
        // MockGraphStore.update_event_trigger always returns Ok(false)
        let state = test_state().await;
        let id = Uuid::new_v4();
        let result = disable_trigger(State(state), Path(id)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AppError::NotFound(msg) => {
                assert!(msg.contains(&id.to_string()));
            }
            other => panic!("Expected NotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_trigger_stats_empty() {
        // MockGraphStore.list_event_triggers returns empty vec
        let state = test_state().await;
        let result = trigger_stats(State(state)).await;
        assert!(result.is_ok());
        let stats = result.unwrap().0;
        assert_eq!(stats.total, 0);
        assert_eq!(stats.enabled, 0);
        assert_eq!(stats.disabled, 0);
        assert!(stats.by_entity_type.is_empty());
    }

    // ── Stats aggregation unit tests ────────────────────────────────────

    #[test]
    fn test_stats_aggregation_logic() {
        // Test the aggregation logic used in trigger_stats by replicating it
        let triggers = vec![
            EventTrigger {
                id: Uuid::new_v4(),
                name: "t1".to_string(),
                protocol_id: Uuid::new_v4(),
                entity_type_pattern: Some("task".to_string()),
                action_pattern: Some("created".to_string()),
                payload_conditions: None,
                cooldown_secs: 0,
                enabled: true,
                project_scope: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            EventTrigger {
                id: Uuid::new_v4(),
                name: "t2".to_string(),
                protocol_id: Uuid::new_v4(),
                entity_type_pattern: Some("task".to_string()),
                action_pattern: None,
                payload_conditions: None,
                cooldown_secs: 0,
                enabled: false,
                project_scope: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            EventTrigger {
                id: Uuid::new_v4(),
                name: "t3".to_string(),
                protocol_id: Uuid::new_v4(),
                entity_type_pattern: None,
                action_pattern: None,
                payload_conditions: None,
                cooldown_secs: 10,
                enabled: true,
                project_scope: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            EventTrigger {
                id: Uuid::new_v4(),
                name: "t4".to_string(),
                protocol_id: Uuid::new_v4(),
                entity_type_pattern: Some("milestone".to_string()),
                action_pattern: Some("updated".to_string()),
                payload_conditions: None,
                cooldown_secs: 0,
                enabled: true,
                project_scope: Some(Uuid::new_v4()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        ];

        let total = triggers.len();
        let enabled = triggers.iter().filter(|t| t.enabled).count();
        let disabled = total - enabled;

        assert_eq!(total, 4);
        assert_eq!(enabled, 3);
        assert_eq!(disabled, 1);

        // Replicate the by_entity_type aggregation
        let mut type_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for t in &triggers {
            let key = t
                .entity_type_pattern
                .clone()
                .unwrap_or_else(|| "*".to_string());
            *type_counts.entry(key).or_insert(0) += 1;
        }

        assert_eq!(type_counts.len(), 3);
        assert_eq!(type_counts["task"], 2);
        assert_eq!(type_counts["*"], 1); // None maps to "*"
        assert_eq!(type_counts["milestone"], 1);
    }

    #[test]
    fn test_stats_aggregation_all_enabled() {
        let triggers: Vec<EventTrigger> = (0..3)
            .map(|i| EventTrigger {
                id: Uuid::new_v4(),
                name: format!("t{}", i),
                protocol_id: Uuid::new_v4(),
                entity_type_pattern: Some("task".to_string()),
                action_pattern: None,
                payload_conditions: None,
                cooldown_secs: 0,
                enabled: true,
                project_scope: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            })
            .collect();

        let total = triggers.len();
        let enabled = triggers.iter().filter(|t| t.enabled).count();
        let disabled = total - enabled;

        assert_eq!(total, 3);
        assert_eq!(enabled, 3);
        assert_eq!(disabled, 0);
    }

    #[test]
    fn test_stats_aggregation_all_disabled() {
        let triggers: Vec<EventTrigger> = (0..2)
            .map(|i| EventTrigger {
                id: Uuid::new_v4(),
                name: format!("t{}", i),
                protocol_id: Uuid::new_v4(),
                entity_type_pattern: None,
                action_pattern: None,
                payload_conditions: None,
                cooldown_secs: 0,
                enabled: false,
                project_scope: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            })
            .collect();

        let total = triggers.len();
        let enabled = triggers.iter().filter(|t| t.enabled).count();
        let disabled = total - enabled;

        assert_eq!(total, 2);
        assert_eq!(enabled, 0);
        assert_eq!(disabled, 2);

        // All have None entity_type_pattern, so all map to "*"
        let mut type_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for t in &triggers {
            let key = t
                .entity_type_pattern
                .clone()
                .unwrap_or_else(|| "*".to_string());
            *type_counts.entry(key).or_insert(0) += 1;
        }
        assert_eq!(type_counts.len(), 1);
        assert_eq!(type_counts["*"], 2);
    }
}
