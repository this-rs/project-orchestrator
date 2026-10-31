//! # Activity stream — one normalized feed for everything the orchestrator runs
//!
//! The CRUD bus ([`super::CrudEvent`]) is the single funnel every mutation goes
//! through, but it is a *mutation* feed: a client that wants to answer "what is
//! running right now, and what did it just do?" would have to know the payload
//! shape of every emitter (the runner nests a `RunnerEvent`, the chat manager
//! sends `{is_streaming}`, the protocol handlers send `{new_status}`…), and the
//! project scope is missing from the very events that matter most (runner and
//! chat events are emitted with `project_id: None`).
//!
//! [`ActivityEvent`] is that normalization: one flat, sequence-numbered record
//! per mutation, tagged with an [`ActivityCategory`] the UI can filter on, and
//! carrying the raw payload untouched for whoever needs the details.
//!
//! ## Sequence numbers and replay
//!
//! Every published event gets a strictly increasing `seq` from a process-wide
//! [`ActivityHub`] counter, and the last [`ACTIVITY_BUFFER_CAPACITY`] events are
//! kept in memory. A client that reconnects passes the last `seq` it saw and
//! receives the gap before the live stream resumes, so a page reload or a
//! dropped socket never leaves a hole in the log. The buffer is *best effort*:
//! past its capacity the oldest entries are gone and the client is told
//! (`lag_dropped` / [`ActivityHub::replay_since`] returning a gap), never
//! silently given a contiguous-looking history.
use super::types::{CrudEvent, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::broadcast;

/// How many events the in-memory replay buffer keeps.
///
/// Sized for "reconnect after a page reload": at the busiest observed rate
/// (a few hundred events per second during a sync) this is several seconds of
/// history, which is exactly the gap a reconnect has to cover.
pub const ACTIVITY_BUFFER_CAPACITY: usize = 1000;

/// Capacity of the live broadcast channel feeding `/ws/activity` subscribers.
pub const ACTIVITY_BROADCAST_CAPACITY: usize = 1024;

/// What kind of thing the event is about — the axis the Activity Hub filters on.
///
/// This is deliberately *not* [`EntityType`]: the UI groups by what the user is
/// watching (a plan run, a protocol run, a conversation), and several entity
/// types collapse into one category (the runner writes `Runner` events whose id
/// is the plan run id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityCategory {
    /// A plan execution (`PlanRun`), emitted by the runner.
    PlanRun,
    /// An FSM protocol execution (`ProtocolRun`).
    ProtocolRun,
    /// A conversation (`ChatSession`), including its background work.
    ChatSession,
    /// Anything else in the graph: plan, task, step, note, decision, …
    Entity,
    /// The orchestrator's own machinery: ingestion progress, attention changes.
    System,
}

impl ActivityCategory {
    /// The wire name (`snake_case`), used in `?entity_types=` filters.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PlanRun => "plan_run",
            Self::ProtocolRun => "protocol_run",
            Self::ChatSession => "chat_session",
            Self::Entity => "entity",
            Self::System => "system",
        }
    }

    /// Parse a wire name, case-insensitively. Unknown names are `None` so a
    /// caller can refuse them instead of silently subscribing to nothing.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "plan_run" | "planrun" => Some(Self::PlanRun),
            "protocol_run" | "protocolrun" => Some(Self::ProtocolRun),
            "chat_session" | "chatsession" => Some(Self::ChatSession),
            "entity" => Some(Self::Entity),
            "system" => Some(Self::System),
            _ => None,
        }
    }

    /// The category a CRUD event belongs to.
    pub fn of(entity_type: &EntityType) -> Self {
        match entity_type {
            EntityType::Runner => Self::PlanRun,
            EntityType::ProtocolRun => Self::ProtocolRun,
            EntityType::ChatSession => Self::ChatSession,
            EntityType::Ingestion | EntityType::AttentionChanged | EntityType::Learning => {
                Self::System
            }
            _ => Self::Entity,
        }
    }
}

/// One normalized record of something that happened.
///
/// Serialized as-is on `/ws/activity`; the frontend type is a hand mirror of
/// this struct (`src/types/activity.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityEvent {
    /// Strictly increasing within a process; the client's replay cursor.
    pub seq: u64,
    /// RFC 3339 timestamp of the mutation.
    pub ts: String,
    /// The axis the UI groups and filters on.
    pub category: ActivityCategory,
    /// The graph entity type, as emitted (`runner`, `task`, …).
    pub entity_type: String,
    /// The mutated entity's id. For a plan run this *is* the run id; for a
    /// protocol run and a chat session, likewise.
    pub entity_id: String,
    /// The mutation, as emitted (`created`, `updated`, `status_changed`, …).
    pub action: String,
    /// Owning project when the emitter knew it. `None` is common (the runner and
    /// the chat manager do not stamp it) — the WebSocket handler resolves it
    /// from the run/session id before filtering, so a client never has to guess.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Human label when the payload carries one (`title`, `name`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Short status hint when the payload carries one (`status`, `new_status`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// The emitter's own payload, untouched: a `RunnerEvent` for a plan run, the
    /// `ChatEvent`-derived fields for a session, the CRUD payload otherwise.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub detail: serde_json::Value,
}

impl ActivityEvent {
    /// Build an event from a CRUD event, attributing it to `seq`.
    ///
    /// `project_id` of the CRUD event wins; when it is absent the field stays
    /// `None` and the caller (the WebSocket handler) may fill it in — see the
    /// module docs on why runner and chat events arrive unscoped.
    pub fn from_crud(event: &CrudEvent, seq: u64) -> Self {
        let payload = &event.payload;
        Self {
            seq,
            ts: event.timestamp.clone(),
            category: ActivityCategory::of(&event.entity_type),
            entity_type: wire_name(&event.entity_type),
            entity_id: event.entity_id.clone(),
            action: wire_name(&event.action),
            project_id: event.project_id.clone(),
            title: payload
                .get("title")
                .or_else(|| payload.get("name"))
                .and_then(|v| v.as_str())
                .map(str::to_string),
            status: payload
                .get("new_status")
                .or_else(|| payload.get("status"))
                .and_then(|v| v.as_str())
                .map(str::to_string),
            detail: payload.clone(),
        }
    }
}

/// Serialize an enum to its `snake_case` wire name without allocating a `String`
/// through serde (falls back to the variant name lowercased, which cannot happen
/// in practice but must not panic if a variant is added without a serializer).
fn wire_name<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => s,
        _ => "unknown".to_string(),
    }
}

/// The activity feed: sequence counter + replay buffer + live broadcast.
///
/// One instance lives inside the [`super::EventBus`], so every mutation that is
/// already broadcast to WebSocket clients is also appended here — there is no
/// second source of truth and no emitter has to remember to publish twice.
#[derive(Debug)]
pub struct ActivityHub {
    seq: AtomicU64,
    ring: Mutex<VecDeque<ActivityEvent>>,
    tx: broadcast::Sender<ActivityEvent>,
    capacity: usize,
}

impl Default for ActivityHub {
    fn default() -> Self {
        Self::new(ACTIVITY_BUFFER_CAPACITY)
    }
}

impl ActivityHub {
    /// Create a hub keeping `capacity` events for replay.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(ACTIVITY_BROADCAST_CAPACITY);
        Self {
            seq: AtomicU64::new(0),
            ring: Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
            tx,
            capacity: capacity.max(1),
        }
    }

    /// The last sequence number handed out (`0` when nothing was published).
    pub fn latest_seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }

    /// Subscribe to the live feed.
    pub fn subscribe(&self) -> broadcast::Receiver<ActivityEvent> {
        self.tx.subscribe()
    }

    /// Number of live subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Stamp `event` with the next sequence number, buffer it, and broadcast it.
    ///
    /// Returns the stored event (with `seq` filled in). Never blocks on
    /// subscribers and never fails: with nobody listening the event is buffered
    /// for replay and dropped from the channel, exactly like the CRUD bus.
    pub fn publish(&self, mut event: ActivityEvent) -> ActivityEvent {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        event.seq = seq;
        {
            // `std::sync::Mutex` on purpose: the critical section is a push and
            // a truncate, with no `.await` inside, so an async mutex would only
            // add a yield point for nothing.
            let mut ring = match self.ring.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            ring.push_back(event.clone());
            while ring.len() > self.capacity {
                ring.pop_front();
            }
        }
        // Err means "no subscriber right now" — expected and fine.
        let _ = self.tx.send(event.clone());
        event
    }

    /// Every buffered event with `seq > last_seq`, oldest first.
    ///
    /// An empty vector means "nothing happened since", **except** when
    /// `last_seq` is older than the oldest buffered event: the caller learns
    /// that it lost events by comparing `latest_seq() - last_seq` with the
    /// number returned. [`Self::lost_before`] answers that question directly.
    pub fn replay_since(&self, last_seq: u64) -> Vec<ActivityEvent> {
        let ring = match self.ring.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        ring.iter().filter(|e| e.seq > last_seq).cloned().collect()
    }

    /// How many of the events the caller asked for have already been evicted.
    pub fn lost_before(&self, last_seq: u64) -> u64 {
        let ring = match self.ring.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let oldest_buffered = match ring.front() {
            Some(e) => e.seq,
            None => return 0,
        };
        // `last_seq` is the last event the client saw: everything after it up to
        // the oldest buffered event is gone.
        oldest_buffered.saturating_sub(last_seq).saturating_sub(1)
    }

    /// Current buffer length (tests, health endpoints).
    pub fn buffered(&self) -> usize {
        match self.ring.lock() {
            Ok(guard) => guard.len(),
            Err(poisoned) => poisoned.into_inner().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::CrudAction;
    use super::*;

    fn crud(entity_type: EntityType, action: CrudAction, id: &str) -> CrudEvent {
        CrudEvent {
            entity_type,
            action,
            entity_id: id.to_string(),
            related: None,
            payload: serde_json::json!({ "title": "T", "status": "in_progress" }),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            project_id: Some("proj-1".to_string()),
        }
    }

    #[test]
    fn sequence_numbers_are_monotonic() {
        let hub = ActivityHub::default();
        let a = hub.publish(ActivityEvent::from_crud(
            &crud(EntityType::Task, CrudAction::Updated, "t1"),
            0,
        ));
        let b = hub.publish(ActivityEvent::from_crud(
            &crud(EntityType::Task, CrudAction::Updated, "t2"),
            0,
        ));
        assert_eq!(a.seq, 1);
        assert_eq!(b.seq, 2);
        assert_eq!(hub.latest_seq(), 2);
    }

    #[test]
    fn replay_returns_only_newer_events() {
        let hub = ActivityHub::default();
        for i in 0..5 {
            hub.publish(ActivityEvent::from_crud(
                &crud(EntityType::Note, CrudAction::Created, &format!("n{i}")),
                0,
            ));
        }
        let replay = hub.replay_since(3);
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].seq, 4);
        assert_eq!(replay[1].seq, 5);
    }

    #[test]
    fn buffer_evicts_oldest_past_capacity() {
        let hub = ActivityHub::new(3);
        for i in 0..6 {
            hub.publish(ActivityEvent::from_crud(
                &crud(EntityType::Note, CrudAction::Created, &format!("n{i}")),
                0,
            ));
        }
        assert_eq!(hub.buffered(), 3);
        assert_eq!(hub.replay_since(0).len(), 3);
        // Events 1..=3 were evicted, so a client that saw seq 1 lost 2 events.
        assert_eq!(hub.lost_before(1), 2);
        assert_eq!(hub.lost_before(3), 0);
    }

    #[test]
    fn live_subscriber_receives_published_event() {
        let hub = ActivityHub::default();
        let mut rx = hub.subscribe();
        let published = hub.publish(ActivityEvent::from_crud(
            &crud(EntityType::Runner, CrudAction::Updated, "run-1"),
            0,
        ));
        let received = rx.try_recv().expect("subscriber should receive");
        assert_eq!(received.seq, published.seq);
        assert_eq!(received.category, ActivityCategory::PlanRun);
    }

    #[test]
    fn from_crud_maps_categories_and_labels() {
        let ev =
            ActivityEvent::from_crud(&crud(EntityType::Runner, CrudAction::Updated, "run-1"), 7);
        assert_eq!(ev.seq, 7);
        assert_eq!(ev.category, ActivityCategory::PlanRun);
        assert_eq!(ev.entity_type, "runner");
        assert_eq!(ev.action, "updated");
        assert_eq!(ev.entity_id, "run-1");
        assert_eq!(ev.title.as_deref(), Some("T"));
        assert_eq!(ev.status.as_deref(), Some("in_progress"));
        assert_eq!(ev.project_id.as_deref(), Some("proj-1"));

        let chat = ActivityEvent::from_crud(
            &crud(EntityType::ChatSession, CrudAction::Updated, "s-1"),
            8,
        );
        assert_eq!(chat.category, ActivityCategory::ChatSession);

        let ingestion = ActivityEvent::from_crud(
            &crud(EntityType::Ingestion, CrudAction::Progress, "proj-1"),
            9,
        );
        assert_eq!(ingestion.category, ActivityCategory::System);
    }

    #[test]
    fn unscoped_crud_event_keeps_project_none() {
        let mut event = crud(EntityType::Runner, CrudAction::Updated, "run-1");
        event.project_id = None;
        event.payload = serde_json::Value::Null;
        let ev = ActivityEvent::from_crud(&event, 1);
        assert!(ev.project_id.is_none());
        assert!(ev.detail.is_null());
        assert!(ev.title.is_none());
    }

    #[test]
    fn category_wire_names_round_trip() {
        for category in [
            ActivityCategory::PlanRun,
            ActivityCategory::ProtocolRun,
            ActivityCategory::ChatSession,
            ActivityCategory::Entity,
            ActivityCategory::System,
        ] {
            assert_eq!(
                ActivityCategory::parse(category.as_str()),
                Some(category),
                "round trip for {}",
                category.as_str()
            );
        }
        assert_eq!(
            ActivityCategory::parse(" PLan_Run "),
            Some(ActivityCategory::PlanRun)
        );
        assert_eq!(ActivityCategory::parse("nope"), None);
    }
}
