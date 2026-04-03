//! CRUD event types for WebSocket notifications.
//!
//! Defines the core types used by the event system:
//! - [`EntityType`] — all entity kinds that can emit events (27 variants)
//! - [`CrudAction`] — the mutation action performed (8 variants)
//! - [`CrudEvent`] — the event payload sent over WebSocket/NATS
//! - [`EventEmitter`] — trait with convenience methods for emitting events

use super::graph::GraphEvent;
use serde::{Deserialize, Serialize};

/// The type of entity that was mutated.
///
/// Each variant corresponds to a Neo4j node label. Serialized as `snake_case`
/// for JSON (e.g., `FeatureGraph` → `"feature_graph"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// Tracked codebase — emitters: handlers.rs, project_handlers.rs
    Project,
    /// Development objective — emitters: handlers.rs (Created, Updated, Deleted, StatusChanged)
    Plan,
    /// Unit of work within a plan — emitters: handlers.rs (Created, Updated, Deleted, StatusChanged)
    Task,
    /// Atomic sub-step within a task — emitters: handlers.rs (Updated, Deleted)
    Step,
    /// Architectural decision record — emitters: handlers.rs (Created, Linked via AFFECTS)
    Decision,
    /// Plan constraint (performance, security, etc.) — emitters: handlers.rs
    Constraint,
    /// Git commit record — emitters: handlers.rs (Created, Linked to Task/Plan)
    Commit,
    /// Deliverable version — emitters: handlers.rs (Created, Updated, Deleted, StatusChanged, Linked)
    Release,
    /// Progress marker — emitters: handlers.rs (Created, Updated, Deleted, StatusChanged, Linked)
    Milestone,
    /// Multi-project container
    Workspace,
    /// Cross-project milestone
    WorkspaceMilestone,
    /// Shared API contract or schema
    Resource,
    /// Service/library/database in workspace topology
    Component,
    /// Knowledge note (guideline, gotcha, pattern, etc.) — emitters: note_handlers.rs
    Note,
    /// Conversation session — emitters: chat_handlers.rs
    ChatSession,
    /// FSM protocol execution instance — emitters: protocol_handlers.rs
    ProtocolRun,
    /// Autonomous plan execution agent — emitters: runner.rs
    Runner,
    /// System-generated alert
    Alert,
    /// Adaptive knowledge agent — emitters: persona_handlers.rs (full CRUD + 7 link pairs + StatusChanged)
    Persona,
    /// Emergent knowledge cluster — emitters: skill_handlers.rs (CRUD + member links + StatusChanged + split/merge)
    Skill,
    /// FSM definition (states + transitions) — emitters: protocol_handlers.rs (CRUD + state/transition/skill links)
    Protocol,
    /// Code feature subgraph — emitters: code_handlers.rs (Created, Deleted)
    FeatureGraph,
    /// Cognitive snapshot from protocol run — emitters: episode_handlers.rs (Created)
    Episode,
    /// Edge/fusion weight preset — emitters: profile_handlers.rs (Created, Deleted)
    AnalysisProfile,
    /// Automated plan trigger (schedule/webhook/event) — emitters: handlers.rs (Created, Updated, Deleted)
    Trigger,
    /// Graph topology constraint rule — emitters: code_handlers.rs (Created, Deleted)
    TopologyRule,
    /// Lifecycle hook (automatic action on status change) — emitters: handlers.rs (Created, Updated, Deleted)
    LifecycleHook,
    /// Learning system — emitters: reactions.rs (PatternsDetected after episode analysis)
    Learning,
}

/// The CRUD action performed on an entity.
///
/// Serialized as `snake_case` for JSON. Each action has specific semantics
/// for the frontend and the EventReactor system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrudAction {
    /// Entity was created — frontend should add it to lists/caches
    Created,
    /// Entity fields were updated (NOT status) — frontend should refresh the entity
    Updated,
    /// Entity was deleted — frontend should remove it from lists/caches
    Deleted,
    /// Entity was linked to another entity — payload contains `related` field
    Linked,
    /// Entity was unlinked from another entity — payload contains `related` field
    Unlinked,
    /// In-flight progress update for long-running operations (e.g., protocol state execution).
    /// Payload contains `processed`, `total`, `elapsed_ms`.
    Progress,
    /// Entity has been synchronized from an external source (e.g., project sync from filesystem).
    /// Payload contains sync stats: `files_parsed`, `duration_ms`, `is_first_sync`.
    Synced,
    /// Entity status has changed (distinct from a field update).
    /// Payload contains `old_status` and `new_status` strings.
    /// Used for lifecycle transitions (plan: draft→approved, task: pending→in_progress, etc.).
    StatusChanged,
    /// An episode has been collected from a completed run.
    /// Payload contains `episode_id`, `run_id`, `project_id`.
    /// Used by the learning loop to trigger pattern detection (T2).
    Collected,
    /// Patterns have been detected from collected episodes.
    /// Payload contains `project_id`, `patterns_count`.
    /// Used by the learning loop to trigger materialization (T3).
    PatternsDetected,
    /// Feedback patterns detected from runner observation notes.
    /// Payload contains `project_id`, `patterns_count`, `suggestion_note_ids`.
    /// Emitted by the FeedbackAnalyzer reaction for frontend notification.
    FeedbackPatternsDetected,
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
/// Implemented by `HybridEmitter` (local broadcast + optional NATS), `NatsEmitter`
/// (NATS-only), and `EventBus` (local-only). Consumers (PlanManager, NoteManager,
/// Orchestrator) hold `Option<Arc<dyn EventEmitter>>` for polymorphic dispatch.
pub trait EventEmitter: Send + Sync {
    /// Emit a CrudEvent (fire-and-forget)
    fn emit(&self, event: CrudEvent);

    /// Emit a GraphEvent for real-time visualization (fire-and-forget).
    ///
    /// Default implementation is a no-op. Only `HybridEmitter` overrides this
    /// to broadcast graph events to WebSocket clients.
    fn emit_graph(&self, _event: GraphEvent) {}

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

    /// Emit a Synced event after a successful synchronization.
    ///
    /// Used after project sync from filesystem. Payload should include
    /// stats like `files_parsed`, `duration_ms`, `is_first_sync`.
    fn emit_synced(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        payload: serde_json::Value,
        project_id: Option<String>,
    ) {
        let mut event =
            CrudEvent::new(entity_type, CrudAction::Synced, entity_id).with_payload(payload);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit a StatusChanged event for lifecycle transitions.
    ///
    /// Distinct from `emit_updated` — used when an entity's status changes
    /// (e.g., plan: draft→approved, task: pending→in_progress).
    /// Automatically builds a payload with `old_status` and `new_status`.
    fn emit_status_changed(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        old_status: &str,
        new_status: &str,
        project_id: Option<String>,
    ) {
        let payload = serde_json::json!({
            "old_status": old_status,
            "new_status": new_status,
        });
        let mut event =
            CrudEvent::new(entity_type, CrudAction::StatusChanged, entity_id).with_payload(payload);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit a Progress event for long-running operations.
    ///
    /// Used by protocol runs to report intermediate progress during
    /// long-running states (e.g., backfill, inference, health checks).
    fn emit_progress(
        &self,
        entity_type: EntityType,
        entity_id: &str,
        payload: serde_json::Value,
        project_id: Option<String>,
    ) {
        let mut event =
            CrudEvent::new(entity_type, CrudAction::Progress, entity_id).with_payload(payload);
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
            EntityType::ProtocolRun,
            EntityType::Runner,
            EntityType::Alert,
            EntityType::Persona,
            EntityType::Skill,
            EntityType::Protocol,
            EntityType::FeatureGraph,
            EntityType::Episode,
            EntityType::AnalysisProfile,
            EntityType::Trigger,
            EntityType::TopologyRule,
            EntityType::LifecycleHook,
            EntityType::Learning,
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
        assert_eq!(
            serde_json::to_string(&EntityType::FeatureGraph).unwrap(),
            "\"feature_graph\""
        );
        assert_eq!(
            serde_json::to_string(&EntityType::AnalysisProfile).unwrap(),
            "\"analysis_profile\""
        );
        assert_eq!(
            serde_json::to_string(&EntityType::TopologyRule).unwrap(),
            "\"topology_rule\""
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
            CrudAction::Progress,
            CrudAction::Synced,
            CrudAction::StatusChanged,
            CrudAction::Collected,
            CrudAction::PatternsDetected,
            CrudAction::FeedbackPatternsDetected,
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
    fn test_entity_type_has_26_variants() {
        // Ensure we don't accidentally add/remove variants
        let all = [
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
            EntityType::ProtocolRun,
            EntityType::Runner,
            EntityType::Alert,
            EntityType::Persona,
            EntityType::Skill,
            EntityType::Protocol,
            EntityType::FeatureGraph,
            EntityType::Episode,
            EntityType::AnalysisProfile,
            EntityType::Trigger,
            EntityType::TopologyRule,
            EntityType::LifecycleHook,
            EntityType::Learning,
        ];
        assert_eq!(all.len(), 28);
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
