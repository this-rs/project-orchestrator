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
