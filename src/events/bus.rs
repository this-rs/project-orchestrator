//! Event bus for broadcasting CRUD events to WebSocket clients

use super::{CrudAction, CrudEvent, EntityType};
use tokio::sync::broadcast;
use tracing::debug;

/// Default broadcast channel capacity
const DEFAULT_CAPACITY: usize = 1024;

/// Event bus that distributes CrudEvents via `tokio::sync::broadcast`
///
/// Fire-and-forget: emitting never blocks, never panics.
/// If no subscribers are connected, events are silently dropped.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<CrudEvent>,
}

impl EventBus {
    /// Create a new EventBus with the given channel capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Subscribe to receive events (for WebSocket clients)
    pub fn subscribe(&self) -> broadcast::Receiver<CrudEvent> {
        self.sender.subscribe()
    }

    /// Emit a CrudEvent to all subscribers (fire-and-forget)
    ///
    /// If no subscribers are connected, the event is silently dropped.
    pub fn emit(&self, event: CrudEvent) {
        let entity = format!("{:?}", event.entity_type);
        let action = format!("{:?}", event.action);
        match self.sender.send(event) {
            Ok(n) => {
                debug!(
                    entity_type = %entity,
                    action = %action,
                    subscribers = n,
                    "CrudEvent emitted"
                );
            }
            Err(_) => {
                // No subscribers — this is expected and fine
            }
        }
    }

    /// Emit a Created event
    pub fn emit_created(
        &self,
        entity_type: EntityType,
        entity_id: impl Into<String>,
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
    pub fn emit_updated(
        &self,
        entity_type: EntityType,
        entity_id: impl Into<String>,
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
    pub fn emit_deleted(
        &self,
        entity_type: EntityType,
        entity_id: impl Into<String>,
        project_id: Option<String>,
    ) {
        let mut event = CrudEvent::new(entity_type, CrudAction::Deleted, entity_id);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit a Linked event (entity linked to a related entity)
    pub fn emit_linked(
        &self,
        entity_type: EntityType,
        entity_id: impl Into<String>,
        related_type: EntityType,
        related_id: impl Into<String>,
        project_id: Option<String>,
    ) {
        let mut event = CrudEvent::new(entity_type, CrudAction::Linked, entity_id)
            .with_related(related_type, related_id);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Emit an Unlinked event (entity unlinked from a related entity)
    pub fn emit_unlinked(
        &self,
        entity_type: EntityType,
        entity_id: impl Into<String>,
        related_type: EntityType,
        related_id: impl Into<String>,
        project_id: Option<String>,
    ) {
        let mut event = CrudEvent::new(entity_type, CrudAction::Unlinked, entity_id)
            .with_related(related_type, related_id);
        if let Some(pid) = project_id {
            event = event.with_project_id(pid);
        }
        self.emit(event);
    }

    /// Number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_without_subscriber_no_panic() {
        let bus = EventBus::default();
        bus.emit_created(
            EntityType::Plan,
            "plan-1",
            serde_json::json!({"title": "Test"}),
            None,
        );
        // Should not panic
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn test_emit_with_subscriber() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);

        bus.emit_created(
            EntityType::Task,
            "task-1",
            serde_json::json!({"title": "Task"}),
            Some("proj-1".into()),
        );

        let event = rx.try_recv().unwrap();
        assert_eq!(event.entity_type, EntityType::Task);
        assert_eq!(event.action, CrudAction::Created);
        assert_eq!(event.entity_id, "task-1");
        assert_eq!(event.project_id.as_deref(), Some("proj-1"));
    }

    #[test]
    fn test_multi_subscribers() {
        let bus = EventBus::default();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();
        let mut rx3 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 3);

        bus.emit_deleted(EntityType::Note, "note-1", None);

        // All 3 subscribers should receive the event
        let e1 = rx1.try_recv().unwrap();
        let e2 = rx2.try_recv().unwrap();
        let e3 = rx3.try_recv().unwrap();
        assert_eq!(e1.entity_id, "note-1");
        assert_eq!(e2.entity_id, "note-1");
        assert_eq!(e3.entity_id, "note-1");
    }

    #[test]
    fn test_emit_linked() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe();

        bus.emit_linked(
            EntityType::Task,
            "task-1",
            EntityType::Release,
            "release-1",
            None,
        );

        let event = rx.try_recv().unwrap();
        assert_eq!(event.action, CrudAction::Linked);
        let related = event.related.unwrap();
        assert_eq!(related.entity_type, EntityType::Release);
        assert_eq!(related.entity_id, "release-1");
    }

    #[test]
    fn test_emit_unlinked() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe();

        bus.emit_unlinked(
            EntityType::Note,
            "note-1",
            EntityType::Task,
            "task-2",
            Some("proj-1".into()),
        );

        let event = rx.try_recv().unwrap();
        assert_eq!(event.action, CrudAction::Unlinked);
        assert_eq!(event.project_id.as_deref(), Some("proj-1"));
    }

    #[test]
    fn test_emit_updated() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe();

        bus.emit_updated(
            EntityType::Plan,
            "plan-1",
            serde_json::json!({"status": "in_progress"}),
            None,
        );

        let event = rx.try_recv().unwrap();
        assert_eq!(event.action, CrudAction::Updated);
        assert_eq!(event.payload["status"], "in_progress");
    }

    #[test]
    fn test_dropped_subscriber_doesnt_affect_others() {
        let bus = EventBus::default();
        let rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);

        drop(rx1);
        assert_eq!(bus.subscriber_count(), 1);

        bus.emit_created(EntityType::Step, "step-1", serde_json::Value::Null, None);
        let event = rx2.try_recv().unwrap();
        assert_eq!(event.entity_id, "step-1");
    }

    #[test]
    fn test_clone_shares_channel() {
        let bus = EventBus::default();
        let bus2 = bus.clone();
        let mut rx = bus.subscribe();

        // Emit from the clone
        bus2.emit_created(EntityType::Workspace, "ws-1", serde_json::Value::Null, None);

        let event = rx.try_recv().unwrap();
        assert_eq!(event.entity_type, EntityType::Workspace);
    }
}
