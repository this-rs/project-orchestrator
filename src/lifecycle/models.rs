use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Scope of a LifecycleHook — which entity type it watches
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleScope {
    Task,
    Plan,
    Step,
    Milestone,
}

/// Action type a LifecycleHook can execute
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleActionType {
    CascadeChildren,
    McpCall,
    CreateNote,
    EmitAlert,
    StartProtocol,
}

/// A LifecycleHook defines an automatic action to execute when an entity changes status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleHook {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub scope: LifecycleScope,
    pub on_status: String,
    pub action_type: LifecycleActionType,
    pub action_config: serde_json::Value,
    pub priority: i32,
    pub enabled: bool,
    pub builtin: bool,
    pub project_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl LifecycleHook {
    pub fn new(
        name: String,
        scope: LifecycleScope,
        on_status: String,
        action_type: LifecycleActionType,
        action_config: serde_json::Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            description: None,
            scope,
            on_status,
            action_type,
            action_config,
            priority: 100,
            enabled: true,
            builtin: false,
            project_id: None,
            created_at: Utc::now(),
            updated_at: None,
        }
    }
}

/// Request to create a lifecycle hook
#[derive(Debug, Deserialize)]
pub struct CreateLifecycleHookRequest {
    pub name: String,
    pub description: Option<String>,
    pub scope: LifecycleScope,
    pub on_status: String,
    pub action_type: LifecycleActionType,
    pub action_config: Option<serde_json::Value>,
    pub priority: Option<i32>,
    pub project_id: Option<String>,
}

/// Request to update a lifecycle hook
#[derive(Debug, Deserialize)]
pub struct UpdateLifecycleHookRequest {
    pub name: Option<String>,
    pub description: Option<Option<String>>,
    pub on_status: Option<String>,
    pub action_config: Option<serde_json::Value>,
    pub priority: Option<i32>,
    pub enabled: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_hook_new_defaults() {
        let hook = LifecycleHook::new(
            "test-hook".to_string(),
            LifecycleScope::Task,
            "completed".to_string(),
            LifecycleActionType::EmitAlert,
            serde_json::json!({"level": "info"}),
        );

        assert_eq!(hook.name, "test-hook");
        assert_eq!(hook.scope, LifecycleScope::Task);
        assert_eq!(hook.on_status, "completed");
        assert_eq!(hook.action_type, LifecycleActionType::EmitAlert);
        assert_eq!(hook.priority, 100);
        assert!(hook.enabled);
        assert!(!hook.builtin);
        assert!(hook.project_id.is_none());
        assert!(hook.description.is_none());
        assert!(hook.updated_at.is_none());
    }

    #[test]
    fn test_lifecycle_scope_serde_roundtrip() {
        let scopes = vec![
            (LifecycleScope::Task, "\"task\""),
            (LifecycleScope::Plan, "\"plan\""),
            (LifecycleScope::Step, "\"step\""),
            (LifecycleScope::Milestone, "\"milestone\""),
        ];
        for (scope, expected_json) in scopes {
            let serialized = serde_json::to_string(&scope).unwrap();
            assert_eq!(serialized, expected_json);
            let deserialized: LifecycleScope = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, scope);
        }
    }

    #[test]
    fn test_lifecycle_action_type_serde_roundtrip() {
        let types = vec![
            (LifecycleActionType::CascadeChildren, "\"cascade_children\""),
            (LifecycleActionType::McpCall, "\"mcp_call\""),
            (LifecycleActionType::CreateNote, "\"create_note\""),
            (LifecycleActionType::EmitAlert, "\"emit_alert\""),
            (LifecycleActionType::StartProtocol, "\"start_protocol\""),
        ];
        for (action, expected_json) in types {
            let serialized = serde_json::to_string(&action).unwrap();
            assert_eq!(serialized, expected_json);
            let deserialized: LifecycleActionType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, action);
        }
    }

    #[test]
    fn test_lifecycle_hook_serde_roundtrip() {
        let hook = LifecycleHook::new(
            "my-hook".to_string(),
            LifecycleScope::Plan,
            "in_progress".to_string(),
            LifecycleActionType::CreateNote,
            serde_json::json!({"note_type": "observation"}),
        );

        let json = serde_json::to_string(&hook).unwrap();
        let deserialized: LifecycleHook = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, hook.name);
        assert_eq!(deserialized.scope, hook.scope);
        assert_eq!(deserialized.on_status, hook.on_status);
        assert_eq!(deserialized.action_type, hook.action_type);
        assert_eq!(deserialized.priority, hook.priority);
        assert_eq!(deserialized.enabled, hook.enabled);
        assert_eq!(deserialized.builtin, hook.builtin);
    }

    #[test]
    fn test_create_request_deserialization() {
        let json = serde_json::json!({
            "name": "test",
            "scope": "task",
            "on_status": "completed",
            "action_type": "emit_alert",
            "action_config": {"level": "warning"},
            "priority": 50,
            "project_id": "550e8400-e29b-41d4-a716-446655440000"
        });
        let req: CreateLifecycleHookRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.name, "test");
        assert_eq!(req.scope, LifecycleScope::Task);
        assert_eq!(req.priority, Some(50));
        assert!(req.project_id.is_some());
    }

    #[test]
    fn test_create_request_minimal_deserialization() {
        let json = serde_json::json!({
            "name": "minimal",
            "scope": "plan",
            "on_status": "completed",
            "action_type": "create_note"
        });
        let req: CreateLifecycleHookRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.name, "minimal");
        assert!(req.description.is_none());
        assert!(req.action_config.is_none());
        assert!(req.priority.is_none());
        assert!(req.project_id.is_none());
    }

    #[test]
    fn test_update_request_deserialization() {
        let json = serde_json::json!({
            "name": "updated-name",
            "enabled": false,
            "priority": 10
        });
        let req: UpdateLifecycleHookRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.name.as_deref(), Some("updated-name"));
        assert_eq!(req.enabled, Some(false));
        assert_eq!(req.priority, Some(10));
        assert!(req.on_status.is_none());
        assert!(req.description.is_none());
        assert!(req.action_config.is_none());
    }
}
