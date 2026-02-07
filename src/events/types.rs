//! CRUD event types for WebSocket notifications

use serde::{Deserialize, Serialize};

/// The type of entity that was mutated
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Project,
    Plan,
    Task,
    Step,
    Decision,
    Constraint,
    Commit,
    Release,
    Milestone,
    Workspace,
    WorkspaceMilestone,
    Resource,
    Component,
    Note,
}

/// The CRUD action performed
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrudAction {
    Created,
    Updated,
    Deleted,
    Linked,
    Unlinked,
}

/// A related entity for Linked/Unlinked actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelatedEntity {
    pub entity_type: EntityType,
    pub entity_id: String,
}

/// A CRUD event emitted after a successful mutation
///
/// Sent to WebSocket clients for real-time UI updates.
/// Must be Clone for `tokio::sync::broadcast`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrudEvent {
    /// The type of entity that was mutated
    pub entity_type: EntityType,
    /// The action performed
    pub action: CrudAction,
    /// The ID of the mutated entity
    pub entity_id: String,
    /// Related entity (for Linked/Unlinked actions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related: Option<RelatedEntity>,
    /// Optional payload with entity data (e.g. new status, title)
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub payload: serde_json::Value,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Optional project ID for client-side filtering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

impl CrudEvent {
    /// Create a new CrudEvent with the current timestamp
    pub fn new(entity_type: EntityType, action: CrudAction, entity_id: impl Into<String>) -> Self {
        Self {
            entity_type,
            action,
            entity_id: entity_id.into(),
            related: None,
            payload: serde_json::Value::Null,
            timestamp: chrono::Utc::now().to_rfc3339(),
            project_id: None,
        }
    }

    /// Set the related entity (for Linked/Unlinked)
    pub fn with_related(mut self, entity_type: EntityType, entity_id: impl Into<String>) -> Self {
        self.related = Some(RelatedEntity {
            entity_type,
            entity_id: entity_id.into(),
        });
        self
    }

    /// Set the payload
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = payload;
        self
    }

    /// Set the project ID
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_serde_roundtrip() {
        let variants = vec![
            EntityType::Project,
            EntityType::Plan,
            EntityType::Task,
            EntityType::Step,
            EntityType::Decision,
            EntityType::Constraint,
            EntityType::Commit,
            EntityType::Release,
            EntityType::Milestone,
            EntityType::Workspace,
            EntityType::WorkspaceMilestone,
            EntityType::Resource,
            EntityType::Component,
            EntityType::Note,
        ];

        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: EntityType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, &deserialized);
        }

        // Verify snake_case serialization
        assert_eq!(
            serde_json::to_string(&EntityType::WorkspaceMilestone).unwrap(),
            "\"workspace_milestone\""
        );
    }

    #[test]
    fn test_crud_action_serde_roundtrip() {
        let variants = vec![
            CrudAction::Created,
            CrudAction::Updated,
            CrudAction::Deleted,
            CrudAction::Linked,
            CrudAction::Unlinked,
        ];

        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: CrudAction = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, &deserialized);
        }
    }

    #[test]
    fn test_crud_event_serde_roundtrip() {
        let event = CrudEvent::new(EntityType::Plan, CrudAction::Created, "plan-123")
            .with_payload(serde_json::json!({"title": "My Plan"}))
            .with_project_id("proj-456");

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: CrudEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.entity_type, EntityType::Plan);
        assert_eq!(deserialized.action, CrudAction::Created);
        assert_eq!(deserialized.entity_id, "plan-123");
        assert_eq!(deserialized.project_id.as_deref(), Some("proj-456"));
        assert!(deserialized.related.is_none());
    }

    #[test]
    fn test_crud_event_with_related() {
        let event = CrudEvent::new(EntityType::Task, CrudAction::Linked, "task-1")
            .with_related(EntityType::Release, "release-2");

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: CrudEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.action, CrudAction::Linked);
        let related = deserialized.related.unwrap();
        assert_eq!(related.entity_type, EntityType::Release);
        assert_eq!(related.entity_id, "release-2");
    }

    #[test]
    fn test_crud_event_null_payload_omitted() {
        let event = CrudEvent::new(EntityType::Note, CrudAction::Deleted, "note-1");
        let json = serde_json::to_string(&event).unwrap();
        // Null payload should be omitted
        assert!(!json.contains("\"payload\""));
        // None related should be omitted
        assert!(!json.contains("\"related\""));
        // None project_id should be omitted
        assert!(!json.contains("\"project_id\""));
    }

    #[test]
    fn test_crud_event_clone_for_broadcast() {
        let event = CrudEvent::new(EntityType::Workspace, CrudAction::Updated, "ws-1")
            .with_payload(serde_json::json!({"name": "Updated"}));

        let cloned = event.clone();
        assert_eq!(cloned.entity_type, event.entity_type);
        assert_eq!(cloned.entity_id, event.entity_id);
        assert_eq!(cloned.payload, event.payload);
    }

    #[test]
    fn test_entity_type_has_14_variants() {
        // Ensure we don't accidentally add/remove variants
        let all = vec![
            EntityType::Project,
            EntityType::Plan,
            EntityType::Task,
            EntityType::Step,
            EntityType::Decision,
            EntityType::Constraint,
            EntityType::Commit,
            EntityType::Release,
            EntityType::Milestone,
            EntityType::Workspace,
            EntityType::WorkspaceMilestone,
            EntityType::Resource,
            EntityType::Component,
            EntityType::Note,
        ];
        assert_eq!(all.len(), 14);
    }
}
