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
    ChatSession,
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

/// Trait for emitting CRUD events.
///
/// Implemented by both `EventBus` (in-process broadcast) and `EventNotifier` (HTTP bridge).
/// Consumers (PlanManager, NoteManager, Orchestrator) hold `Option<Arc<dyn EventEmitter>>`
/// so the same code works for both the HTTP server and the MCP server.
pub trait EventEmitter: Send + Sync {
    /// Emit a CrudEvent (fire-and-forget)
    fn emit(&self, event: CrudEvent);

    /// Emit a Created event
    fn emit_created(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        payload: serde_json::Value,
        project_id: Option<String>,
    ) {
        let mut event =
            CrudEvent::new(entity_type, CrudAction::Created, entity_id).with_payload(payload);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit an Updated event
    fn emit_updated(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        payload: serde_json::Value,
        project_id: Option<String>,
    ) {
        let mut event =
            CrudEvent::new(entity_type, CrudAction::Updated, entity_id).with_payload(payload);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit a Deleted event
    fn emit_deleted(&self, entity_type: EntityType, entity_id: &str, project_id: Option<String>) {
        let mut event = CrudEvent::new(entity_type, CrudAction::Deleted, entity_id);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit a Linked event
    fn emit_linked(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        related_type: EntityType,
        related_id: &str,
        project_id: Option<String>,
    ) {
        let mut event = CrudEvent::new(entity_type, CrudAction::Linked, entity_id)
            .with_related(related_type, related_id);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit an Unlinked event
    fn emit_unlinked(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        related_type: EntityType,
        related_id: &str,
        project_id: Option<String>,
    ) {
        let mut event = CrudEvent::new(entity_type, CrudAction::Unlinked, entity_id)
            .with_related(related_type, related_id);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }
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
            EntityType::ChatSession,
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
        assert_eq!(
            serde_json::to_string(&EntityType::ChatSession).unwrap(),
            "\"chat_session\""
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
    fn test_entity_type_has_15_variants() {
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
            EntityType::ChatSession,
        ];
        assert_eq!(all.len(), 15);
    }

    // ================================================================
    // EventEmitter trait tests
    // ================================================================

    use std::sync::{Arc, Mutex};

    /// A test-only EventEmitter that captures emitted events
    struct RecordingEmitter {
        events: Mutex<Vec<CrudEvent>>,
    }

    impl RecordingEmitter {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn take_events(&self) -> Vec<CrudEvent> {
            std::mem::take(&mut *self.events.lock().unwrap())
        }
    }

    impl EventEmitter for RecordingEmitter {
        fn emit(&self, event: CrudEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    #[test]
    fn test_emit_created_default_method() {
        let emitter = RecordingEmitter::new();
        emitter.emit_created(
            EntityType::Plan,
            "plan-1",
            serde_json::json!({"title": "My Plan"}),
            Some("proj-1".into()),
        );

        let events = emitter.take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].entity_type, EntityType::Plan);
        assert_eq!(events[0].action, CrudAction::Created);
        assert_eq!(events[0].entity_id, "plan-1");
        assert_eq!(events[0].payload["title"], "My Plan");
        assert_eq!(events[0].project_id.as_deref(), Some("proj-1"));
    }

    #[test]
    fn test_emit_updated_default_method() {
        let emitter = RecordingEmitter::new();
        emitter.emit_updated(
            EntityType::Task,
            "task-1",
            serde_json::json!({"status": "completed"}),
            None,
        );

        let events = emitter.take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, CrudAction::Updated);
        assert_eq!(events[0].entity_id, "task-1");
        assert!(events[0].project_id.is_none());
    }

    #[test]
    fn test_emit_deleted_default_method() {
        let emitter = RecordingEmitter::new();
        emitter.emit_deleted(EntityType::Note, "note-1", Some("proj-2".into()));

        let events = emitter.take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, CrudAction::Deleted);
        assert_eq!(events[0].entity_id, "note-1");
        assert_eq!(events[0].project_id.as_deref(), Some("proj-2"));
    }

    #[test]
    fn test_emit_linked_default_method() {
        let emitter = RecordingEmitter::new();
        emitter.emit_linked(
            EntityType::Task,
            "task-1",
            EntityType::Release,
            "release-1",
            None,
        );

        let events = emitter.take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, CrudAction::Linked);
        let related = events[0].related.as_ref().unwrap();
        assert_eq!(related.entity_type, EntityType::Release);
        assert_eq!(related.entity_id, "release-1");
    }

    #[test]
    fn test_emit_unlinked_default_method() {
        let emitter = RecordingEmitter::new();
        emitter.emit_unlinked(
            EntityType::Note,
            "note-1",
            EntityType::Task,
            "task-2",
            Some("proj-1".into()),
        );

        let events = emitter.take_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, CrudAction::Unlinked);
        assert_eq!(events[0].project_id.as_deref(), Some("proj-1"));
        let related = events[0].related.as_ref().unwrap();
        assert_eq!(related.entity_type, EntityType::Task);
        assert_eq!(related.entity_id, "task-2");
    }

    #[test]
    fn test_event_emitter_as_dyn_trait_object() {
        // Verify the trait is dyn-compatible
        let emitter: Arc<dyn EventEmitter> = Arc::new(RecordingEmitter::new());
        emitter.emit_created(EntityType::Workspace, "ws-1", serde_json::Value::Null, None);
        // If this compiles and runs, the trait is dyn-compatible
    }
}
