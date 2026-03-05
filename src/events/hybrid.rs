//! Hybrid event emitter: local broadcast + optional NATS inter-process sync
//!
//! Combines an in-process `EventBus` (tokio broadcast) with an optional
//! `NatsEmitter` for cross-instance event distribution.
//!
//! When NATS is not configured, the emitter works in local-only mode
//! with zero overhead — no connection attempts, no errors.

use super::bus::EventBus;
use super::graph::GraphEvent;
use super::nats::NatsEmitter;
use super::types::{CrudEvent, EventEmitter};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Default broadcast channel capacity for graph events
const GRAPH_BUS_CAPACITY: usize = 512;

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
    /// Dedicated broadcast channel for graph visualization events.
    /// Separate from the CRUD bus to avoid coupling and allow independent
    /// filtering (by layer) in the WebSocket handler.
    graph_sender: broadcast::Sender<GraphEvent>,
    nats: Option<Arc<NatsEmitter>>,
}

impl HybridEmitter {
    /// Create a local-only HybridEmitter (no NATS).
    ///
    /// Events are broadcast to in-process subscribers only.
    pub fn new(local_bus: Arc<EventBus>) -> Self {
        let (graph_sender, _) = broadcast::channel(GRAPH_BUS_CAPACITY);
        Self {
            local_bus,
            graph_sender,
            nats: None,
        }
    }

    /// Create a HybridEmitter with both local broadcast and NATS.
    ///
    /// Events are emitted to both channels in parallel.
    pub fn with_nats(local_bus: Arc<EventBus>, nats_emitter: Arc<NatsEmitter>) -> Self {
        let (graph_sender, _) = broadcast::channel(GRAPH_BUS_CAPACITY);
        Self {
            local_bus,
            graph_sender,
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

    /// Subscribe to the graph event broadcast channel.
    ///
    /// Used by WebSocket handlers to receive graph mutation events
    /// (node_created, edge_created, reinforcement, etc.) for real-time
    /// visualization updates.
    pub fn subscribe_graph(&self) -> broadcast::Receiver<GraphEvent> {
        self.graph_sender.subscribe()
    }

    /// Emit a graph event to all subscribers.
    ///
    /// Fire-and-forget: if no subscribers are connected, the event is silently dropped.
    /// Graph events are local-only (not published to NATS) for now.
    pub fn emit_graph(&self, event: GraphEvent) {
        let event_type = format!("{:?}", event.event_type);
        let layer = format!("{:?}", event.layer);
        match self.graph_sender.send(event) {
            Ok(n) => {
                debug!(
                    event_type = %event_type,
                    layer = %layer,
                    subscribers = n,
                    "GraphEvent emitted"
                );
            }
            Err(_) => {
                // No subscribers — expected and fine
            }
        }
    }

    /// Number of active graph event subscribers.
    pub fn graph_subscriber_count(&self) -> usize {
        self.graph_sender.receiver_count()
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

    fn emit_graph(&self, event: GraphEvent) {
        self.emit_graph(event);
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
        hybrid2.emit_created(EntityType::Workspace, "ws-1", serde_json::Value::Null, None);

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

    // ================================================================
    // Graph event bus tests
    // ================================================================

    use crate::events::graph::{GraphEventType, GraphLayer};

    #[test]
    fn test_graph_emit_without_subscriber_no_panic() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);

        hybrid.emit_graph(GraphEvent::node(
            GraphEventType::NodeCreated,
            GraphLayer::Knowledge,
            "note-1",
            "proj-1",
        ));
        // Should not panic
        assert_eq!(hybrid.graph_subscriber_count(), 0);
    }

    #[test]
    fn test_graph_emit_with_subscriber() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let mut rx = hybrid.subscribe_graph();
        assert_eq!(hybrid.graph_subscriber_count(), 1);

        hybrid.emit_graph(GraphEvent::node(
            GraphEventType::NodeCreated,
            GraphLayer::Knowledge,
            "note-1",
            "proj-1",
        ));

        let event = rx.try_recv().unwrap();
        assert_eq!(event.event_type, GraphEventType::NodeCreated);
        assert_eq!(event.layer, GraphLayer::Knowledge);
        assert_eq!(event.node_id.as_deref(), Some("note-1"));
        assert_eq!(event.project_id, "proj-1");
    }

    #[test]
    fn test_graph_emit_multi_subscribers() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let mut rx1 = hybrid.subscribe_graph();
        let mut rx2 = hybrid.subscribe_graph();
        assert_eq!(hybrid.graph_subscriber_count(), 2);

        hybrid.emit_graph(GraphEvent::edge(
            GraphEventType::EdgeCreated,
            GraphLayer::Neural,
            "note-1",
            "note-2",
            "SYNAPSE",
            "proj-1",
        ));

        let e1 = rx1.try_recv().unwrap();
        let e2 = rx2.try_recv().unwrap();
        assert_eq!(e1.event_type, GraphEventType::EdgeCreated);
        assert_eq!(e2.event_type, GraphEventType::EdgeCreated);
    }

    #[test]
    fn test_graph_and_crud_buses_independent() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let mut crud_rx = hybrid.subscribe();
        let mut graph_rx = hybrid.subscribe_graph();

        // Emit a CRUD event — should NOT appear on graph bus
        hybrid.emit_created(EntityType::Note, "note-1", serde_json::Value::Null, None);

        // Emit a graph event — should NOT appear on CRUD bus
        hybrid.emit_graph(GraphEvent::reinforcement("note-1", 0.15, "proj-1"));

        // CRUD bus should have exactly 1 CRUD event
        let crud_event = crud_rx.try_recv().unwrap();
        assert_eq!(crud_event.entity_type, EntityType::Note);
        assert!(crud_rx.try_recv().is_err()); // no more

        // Graph bus should have exactly 1 graph event
        let graph_event = graph_rx.try_recv().unwrap();
        assert_eq!(graph_event.event_type, GraphEventType::Reinforcement);
        assert!(graph_rx.try_recv().is_err()); // no more
    }

    #[test]
    fn test_graph_clone_shares_channel() {
        let bus = Arc::new(EventBus::default());
        let hybrid = HybridEmitter::new(bus);
        let hybrid2 = hybrid.clone();
        let mut rx = hybrid.subscribe_graph();

        // Emit from the clone
        hybrid2.emit_graph(GraphEvent::community_changed("comm-1", 5, "proj-1"));

        let event = rx.try_recv().unwrap();
        assert_eq!(event.event_type, GraphEventType::CommunityChanged);
    }
}
