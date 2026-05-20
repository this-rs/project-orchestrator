//! Event bus for broadcasting CRUD events to WebSocket clients

use super::{CrudEvent, EventEmitter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::debug;

/// Default broadcast channel capacity
const DEFAULT_CAPACITY: usize = 1024;

/// Event bus that distributes CrudEvents via `tokio::sync::broadcast`
///
/// Fire-and-forget: emitting never blocks, never panics.
/// If no subscribers are connected, events are silently dropped.
///
/// Also exposes a **monotonic sequence counter** ([`Self::next_sequence`]) used
/// by the Live Activity Hub to tag every `ActivityEvent` with a `seq` number,
/// enabling replay and gap detection on the WebSocket client side. The counter
/// is shared across clones of the bus (backed by `Arc<AtomicU64>`), so all
/// emitters observe a single global sequence.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<CrudEvent>,
    /// Monotonic per-bus sequence counter, shared across clones.
    sequence: Arc<AtomicU64>,
}

impl EventBus {
    /// Create a new EventBus with the given channel capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            sequence: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Subscribe to receive events (for WebSocket clients)
    pub fn subscribe(&self) -> broadcast::Receiver<CrudEvent> {
        self.sender.subscribe()
    }

    /// Number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Return the next monotonic sequence number (post-increment).
    ///
    /// Each call returns a unique, strictly-increasing `u64`. Safe to call
    /// concurrently from any thread — backed by `AtomicU64` with `SeqCst`
    /// ordering. Shared across clones of the bus.
    ///
    /// Used by the Activity Hub to tag every emitted `ActivityEvent` with a
    /// monotonic `seq`, so a reconnecting client can detect gaps and request
    /// replay via `GET /api/activity/snapshot`.
    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    /// Peek the current sequence value without incrementing (for diagnostics).
    pub fn current_sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }
}

impl EventEmitter for EventBus {
    fn emit(&self, event: CrudEvent) {
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
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, EntityType};

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

    // ========================================================================
    // Sequence number tests (T1 / step 3e186596 — Activity Hub Wave A)
    // ========================================================================

    #[test]
    fn test_sequence_starts_at_zero() {
        let bus = EventBus::default();
        assert_eq!(bus.current_sequence(), 0);
    }

    #[test]
    fn test_sequence_is_monotonic_single_thread() {
        let bus = EventBus::default();
        let s0 = bus.next_sequence();
        let s1 = bus.next_sequence();
        let s2 = bus.next_sequence();
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        // Peek matches what would come next.
        assert_eq!(bus.current_sequence(), 3);
    }

    #[test]
    fn test_sequence_shared_across_clones() {
        let bus = EventBus::default();
        let bus2 = bus.clone();
        let s0 = bus.next_sequence();
        let s1 = bus2.next_sequence();
        let s2 = bus.next_sequence();
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(bus.current_sequence(), bus2.current_sequence());
    }

    #[test]
    fn test_sequence_monotonic_cross_thread() {
        use std::collections::HashSet;
        use std::sync::Arc;
        use std::thread;

        let bus = Arc::new(EventBus::default());
        const THREADS: usize = 8;
        const PER_THREAD: usize = 1000;

        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let bus = bus.clone();
            handles.push(thread::spawn(move || {
                let mut local = Vec::with_capacity(PER_THREAD);
                for _ in 0..PER_THREAD {
                    local.push(bus.next_sequence());
                }
                local
            }));
        }

        let mut all = Vec::with_capacity(THREADS * PER_THREAD);
        for h in handles {
            all.extend(h.join().unwrap());
        }

        // All sequence numbers must be unique (cross-thread atomicity).
        let unique: HashSet<u64> = all.iter().copied().collect();
        assert_eq!(
            unique.len(),
            THREADS * PER_THREAD,
            "every sequence number must be unique across threads"
        );

        // The set of sequence numbers must cover [0, THREADS*PER_THREAD).
        let max = *unique.iter().max().unwrap();
        let min = *unique.iter().min().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max as usize, THREADS * PER_THREAD - 1);

        // Counter should be parked at exactly the next value.
        assert_eq!(bus.current_sequence() as usize, THREADS * PER_THREAD);
    }
}
