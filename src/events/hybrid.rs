//! Hybrid event emitter: local broadcast + optional NATS inter-process sync
//!
//! Combines an in-process `EventBus` (tokio broadcast) with an optional
//! `NatsEmitter` for cross-instance event distribution.
//!
//! When NATS is not configured, the emitter works in local-only mode
//! with zero overhead — no connection attempts, no errors.

use super::bus::EventBus;
use super::nats::NatsEmitter;
use super::types::{CrudEvent, EventEmitter};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Hybrid emitter that fans out CrudEvents to both local broadcast and NATS.
///
/// - **Local bus**: always active, used by WebSocket handlers for intra-process delivery
/// - **NATS**: optional, used for inter-process sync (dev + desktop environments)
///
/// Implements `EventEmitter` so it can be injected as `Arc<dyn EventEmitter>`
/// into PlanManager, NoteManager, Orchestrator, etc. without any API change.
#[derive(Clone)]
pub struct HybridEmitter {
    local_bus: Arc<EventBus>,
    nats: Option<Arc<NatsEmitter>>,
}

impl HybridEmitter {
    /// Create a local-only HybridEmitter (no NATS).
    ///
    /// Events are broadcast to in-process subscribers only.
    pub fn new(local_bus: Arc<EventBus>) -> Self {
        Self {
            local_bus,
            nats: None,
        }
    }

    /// Create a HybridEmitter with both local broadcast and NATS.
    ///
    /// Events are emitted to both channels in parallel.
    pub fn with_nats(local_bus: Arc<EventBus>, nats_emitter: Arc<NatsEmitter>) -> Self {
        Self {
            local_bus,
            nats: Some(nats_emitter),
        }
    }

    /// Subscribe to the local broadcast channel.
    ///
    /// Used by WebSocket handlers to receive intra-process events.
    /// NATS events from other instances are bridged into this channel
    /// by the NATS subscriber task (see `start_nats_crud_subscriber`).
    pub fn subscribe(&self) -> broadcast::Receiver<CrudEvent> {
        self.local_bus.subscribe()
    }

    /// Number of active local subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.local_bus.subscriber_count()
    }

    /// Whether NATS is configured and connected.
    pub fn has_nats(&self) -> bool {
        self.nats.is_some()
    }

    /// Get a reference to the local EventBus.
    pub fn local_bus(&self) -> &Arc<EventBus> {
        &self.local_bus
    }

    /// Get a reference to the NatsEmitter, if configured.
    pub fn nats_emitter(&self) -> Option<&Arc<NatsEmitter>> {
        self.nats.as_ref()
    }

    /// Start the NATS→local bridge: subscribes to NATS CRUD events and
    /// re-injects them into the local broadcast bus.
    ///
    /// This makes the local broadcast channel the **single source of truth**
    /// for all CRUD events (both local and remote). WebSocket handlers only
    /// need to subscribe to the local bus — they no longer need their own
    /// NATS subscriptions for CRUD events.
    ///
    /// **Deduplication**: events that originated locally (and were already
    /// published to NATS by `HybridEmitter::emit`) are detected by fingerprint
    /// (`timestamp:entity_id`) and not re-injected, preventing duplicates.
    ///
    /// **Important**: uses `local_bus.emit()` (not `self.emit()`) to avoid
    /// re-publishing to NATS, which would create an infinite loop.
    ///
    /// No-op if NATS is not configured.
    pub fn start_nats_bridge(&self) {
        let Some(nats) = &self.nats else {
            return;
        };

        let nats = nats.clone();
        let local_bus = self.local_bus.clone();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_crud_events().await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!("Failed to start NATS→local bridge: {}", e);
                    return;
                }
            };

            info!("NATS→local bridge started: events.crud → local broadcast");

            // Bounded dedup window to avoid re-injecting events that originated locally
            const BRIDGE_DEDUP_WINDOW: usize = 256;
            let mut seen: VecDeque<String> = VecDeque::with_capacity(BRIDGE_DEDUP_WINDOW);
            let mut seen_set: HashSet<String> = HashSet::with_capacity(BRIDGE_DEDUP_WINDOW);

            use futures::StreamExt;
            while let Some(msg) = subscriber.next().await {
                match serde_json::from_slice::<CrudEvent>(&msg.payload) {
                    Ok(event) => {
                        let fp = format!("{}:{}", event.timestamp, event.entity_id);

                        // Skip if we already saw this event (local origin → NATS → back)
                        if seen_set.contains(&fp) {
                            continue;
                        }
                        seen_set.insert(fp.clone());
                        seen.push_back(fp);
                        if seen.len() > BRIDGE_DEDUP_WINDOW {
                            if let Some(old) = seen.pop_front() {
                                seen_set.remove(&old);
                            }
                        }

                        // Inject into local bus only (NOT HybridEmitter::emit to avoid NATS loop)
                        debug!(
                            entity_type = ?event.entity_type,
                            action = ?event.action,
                            entity_id = %event.entity_id,
                            "NATS→local bridge: injecting remote event"
                        );
                        local_bus.emit(event);
                    }
                    Err(e) => {
                        warn!("NATS→local bridge: failed to deserialize CrudEvent: {}", e);
                    }
                }
            }

            warn!("NATS→local bridge: subscriber closed");
        });
    }
}

impl EventEmitter for HybridEmitter {
    fn emit(&self, event: CrudEvent) {
        // 1. Always emit to local broadcast (intra-process)
        self.local_bus.emit(event.clone());

        // 2. Emit to NATS if configured (inter-process)
        if let Some(nats) = &self.nats {
            nats.emit(event);
        } else {
            debug!("HybridEmitter: NATS not configured, local-only mode");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, EntityType};

    #[test]
    fn test_local_only_emit() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus.clone());
        let mut rx = hybrid.subscribe();

        assert!(!hybrid.has_nats());
        assert_eq!(hybrid.subscriber_count(), 1);

        hybrid.emit_created(
            EntityType::Plan,
            "plan-1",
            serde_json::json!({"title": "Test"}),
            None,
        );

        let event = rx.try_recv().unwrap();
        assert_eq!(event.entity_type, EntityType::Plan);
        assert_eq!(event.action, CrudAction::Created);
        assert_eq!(event.entity_id, "plan-1");
    }

    #[test]
    fn test_local_only_no_nats_no_panic() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);

        // Emit without any subscriber and without NATS — should not panic
        hybrid.emit_created(EntityType::Task, "task-1", serde_json::Value::Null, None);
        hybrid.emit_updated(
            EntityType::Task,
            "task-1",
            serde_json::json!({"status": "completed"}),
            None,
        );
        hybrid.emit_deleted(EntityType::Task, "task-1", None);
    }

    #[test]
    fn test_local_only_multi_subscribers() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let mut rx1 = hybrid.subscribe();
        let mut rx2 = hybrid.subscribe();
        assert_eq!(hybrid.subscriber_count(), 2);

        hybrid.emit_created(
            EntityType::Note,
            "note-1",
            serde_json::Value::Null,
            Some("proj-1".into()),
        );

        let e1 = rx1.try_recv().unwrap();
        let e2 = rx2.try_recv().unwrap();
        assert_eq!(e1.entity_id, "note-1");
        assert_eq!(e2.entity_id, "note-1");
        assert_eq!(e1.project_id.as_deref(), Some("proj-1"));
    }

    #[test]
    fn test_local_only_linked_unlinked() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let mut rx = hybrid.subscribe();

        hybrid.emit_linked(
            EntityType::Task,
            "task-1",
            EntityType::Milestone,
            "ms-1",
            None,
        );

        let event = rx.try_recv().unwrap();
        assert_eq!(event.action, CrudAction::Linked);
        let related = event.related.unwrap();
        assert_eq!(related.entity_type, EntityType::Milestone);
        assert_eq!(related.entity_id, "ms-1");
    }

    #[test]
    fn test_clone_shares_channels() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let hybrid2 = hybrid.clone();
        let mut rx = hybrid.subscribe();

        // Emit from the clone
        hybrid2.emit_created(
            EntityType::Workspace,
            "ws-1",
            serde_json::Value::Null,
            None,
        );

        let event = rx.try_recv().unwrap();
        assert_eq!(event.entity_type, EntityType::Workspace);
    }

    #[test]
    fn test_accessors() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus.clone());

        assert!(!hybrid.has_nats());
        assert!(hybrid.nats_emitter().is_none());
        assert!(Arc::ptr_eq(hybrid.local_bus(), &bus));
    }
}
