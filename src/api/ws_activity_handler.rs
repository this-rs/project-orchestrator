//! WebSocket handler for the **Live Activity Hub** (`/ws/activity`).
//!
//! Multiplexed real-time stream of all in-flight work for a project:
//! - **CRUD events** filtered to `{Plan, Task, ProtocolRun, ChatSession}`
//! - **RunnerEvents** (bridged through CRUD as `entity_type = Runner`,
//!   payload = serialized [`RunnerEvent`])
//! - **ProtocolRun status changes** (subset of CRUD)
//!
//! Each event is wrapped in an [`ActivityEvent`] envelope with a monotonic
//! `seq` produced by the shared [`EventBus`] counter.
//!
//! ## Query parameters
//!
//! | Name            | Required | Description                                                |
//! |-----------------|----------|------------------------------------------------------------|
//! | `project_id`    | yes      | Scope events to one project. Required to avoid noisy fanout |
//! | `entity_types`  | no       | CSV of `ActivityEvent.kind` or CRUD `entity_type` filters   |
//! | `statuses`      | no       | CSV of payload `status` / `new_status` strings to keep      |
//! | `lastEventSeq`  | no       | Replay events with `seq > lastEventSeq` from the ring buffer |
//! | `ticket`        | no       | One-time auth ticket (WS-upgrade cookie fallback)           |
//!
//! ## Backpressure
//!
//! When the broadcast receiver lags, the handler emits a `lag_dropped` control
//! frame to the client (with the count of skipped events) instead of dropping
//! the connection. The client is expected to refresh state via REST.
//!
//! ## Replay (ring buffer)
//!
//! A process-wide ring buffer (capacity 1000) keeps the most recent
//! [`ActivityEvent`]s. On (re)connect with `lastEventSeq=N`, the handler scans
//! the ring and forwards every cached event with `seq > N` **before** wiring
//! the live broadcast subscription. This is **best effort** — events older
//! than the buffer's tail are gone.

use super::handlers::OrchestratorState;
use super::ws_auth::CookieAuthResult;
use crate::auth::jwt::Claims;
use crate::events::{
    ActivityEvent, CrudEvent, EntityType, EventBus, is_crud_entity_relevant,
};
use crate::runner::models::RunnerEvent;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::{broadcast, OnceCell, RwLock};
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

// ============================================================================
// Constants
// ============================================================================

/// Capacity of the activity broadcast channel (matches T4 constraint: ≥ 1024).
const ACTIVITY_BUS_CAPACITY: usize = 2048;

/// Capacity of the replay ring buffer.
const RING_BUFFER_CAPACITY: usize = 1000;

// ============================================================================
// ActivityHub — shared broker
// ============================================================================

/// Process-wide broker that fans-out [`ActivityEvent`]s and keeps a ring
/// buffer for replay-on-reconnect.
///
/// Construction is intentionally **lazy** (see [`get_or_init_activity_hub`])
/// because it captures a clone of the application's [`EventBus`] for sequence
/// generation. The first `/ws/activity` connection initializes the singleton
/// and spawns the background ingestion task.
pub struct ActivityHub {
    /// Live broadcast channel for [`ActivityEvent`]s.
    sender: broadcast::Sender<ActivityEvent>,
    /// In-memory ring buffer of the most recent events (capacity = [`RING_BUFFER_CAPACITY`]).
    ring: RwLock<VecDeque<ActivityEvent>>,
}

impl ActivityHub {
    fn new() -> Self {
        let (sender, _) = broadcast::channel(ACTIVITY_BUS_CAPACITY);
        Self {
            sender,
            ring: RwLock::new(VecDeque::with_capacity(RING_BUFFER_CAPACITY)),
        }
    }

    /// Subscribe to the activity broadcast channel.
    pub fn subscribe(&self) -> broadcast::Receiver<ActivityEvent> {
        self.sender.subscribe()
    }

    /// Push a new event: store in ring buffer (bounded), then broadcast.
    pub async fn push(&self, event: ActivityEvent) {
        {
            let mut ring = self.ring.write().await;
            if ring.len() >= RING_BUFFER_CAPACITY {
                ring.pop_front();
            }
            ring.push_back(event.clone());
        }
        // Fire-and-forget: no subscriber = silent drop (matches EventBus contract).
        let _ = self.sender.send(event);
    }

    /// Return all cached events with `seq > last_seq`, in seq order.
    pub async fn replay_since(&self, last_seq: u64) -> Vec<ActivityEvent> {
        let ring = self.ring.read().await;
        ring.iter()
            .filter(|e| e.seq() > last_seq)
            .cloned()
            .collect()
    }

    /// Number of currently-subscribed receivers (diagnostics + tests).
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

/// Process-wide lazy-initialized [`ActivityHub`].
static ACTIVITY_HUB: OnceCell<Arc<ActivityHub>> = OnceCell::const_new();

/// Lazily build (or return) the singleton [`ActivityHub`] and spawn its
/// background ingestion task that consumes from the CRUD [`EventBus`] and
/// converts events into [`ActivityEvent`]s tagged with a fresh `seq`.
async fn get_or_init_activity_hub(state: &OrchestratorState) -> Arc<ActivityHub> {
    let bus: Arc<EventBus> = state.event_bus.local_bus().clone();
    let mut crud_rx = state.event_bus.subscribe();

    ACTIVITY_HUB
        .get_or_init(|| async move {
            let hub = Arc::new(ActivityHub::new());

            // Spawn the ingestion task — runs until the broadcast channel closes
            // (i.e. process shutdown).
            let hub_clone = hub.clone();
            tokio::spawn(async move {
                info!("ActivityHub ingestion task started");
                loop {
                    match crud_rx.recv().await {
                        Ok(crud) => {
                            if let Some(activity) = transform_crud(&bus, crud) {
                                hub_clone.push(activity).await;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!(
                                skipped = n,
                                "ActivityHub ingestion lagged — events skipped at source"
                            );
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            warn!("ActivityHub ingestion channel closed — task exiting");
                            break;
                        }
                    }
                }
            });

            hub
        })
        .await
        .clone()
}

/// Translate a [`CrudEvent`] into an [`ActivityEvent`], if relevant.
///
/// - `entity_type = Runner` → extract & wrap the inner [`RunnerEvent`].
/// - Other entity types → delegate to [`ActivityEvent::try_from_crud`],
///   which drops noise (Note, Decision, …) and keeps Plan/Task/ProtocolRun/ChatSession.
fn transform_crud(bus: &EventBus, crud: CrudEvent) -> Option<ActivityEvent> {
    if crud.entity_type == EntityType::Runner {
        // Try to deserialize the payload as a RunnerEvent.
        if let Ok(runner_event) =
            serde_json::from_value::<RunnerEvent>(crud.payload.clone())
        {
            let seq = bus.next_sequence();
            return Some(ActivityEvent::from_runner(seq, runner_event));
        }
        return None;
    }

    if !is_crud_entity_relevant(&crud.entity_type) {
        return None;
    }
    let seq = bus.next_sequence();
    ActivityEvent::try_from_crud(seq, crud)
}

// ============================================================================
// Query parameters
// ============================================================================

/// Query parameters for `/ws/activity`.
#[derive(Debug, Deserialize, Default)]
pub struct WsActivityQuery {
    /// Scope events to a project (REQUIRED).
    pub project_id: Option<String>,
    /// CSV of allowed kinds. Accepts both `ActivityEvent.kind`
    /// (`runner`, `chat`, `crud`, `protocol_progress`) and concrete CRUD
    /// entity types (`plan`, `task`, `protocol_run`, `chat_session`).
    pub entity_types: Option<String>,
    /// CSV of allowed payload statuses (matched against `status` or `new_status` fields).
    pub statuses: Option<String>,
    /// Replay events newer than this sequence number on connect.
    #[serde(rename = "lastEventSeq")]
    pub last_event_seq: Option<u64>,
    /// One-time WS ticket (cookie fallback).
    pub ticket: Option<String>,
}

/// Outgoing envelope sent to the client.
///
/// Three variants:
/// - `activity` — a regular [`ActivityEvent`]
/// - `lag_dropped` — backpressure notification (broadcast lagged)
/// - `connected` — handshake frame with the current hub buffer head
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutFrame<'a> {
    Activity { event: &'a ActivityEvent },
    LagDropped { skipped: u64 },
    Connected { last_seq: u64, replayed: usize },
}

// ============================================================================
// HTTP upgrade entrypoint
// ============================================================================

/// Upgrade `/ws/activity?project_id=…` to a WebSocket.
pub async fn ws_activity(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Query(query): Query<WsActivityQuery>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let Some(project_id) = query.project_id.clone() else {
        warn!("WS /ws/activity: missing project_id — rejecting");
        return Err(StatusCode::BAD_REQUEST);
    };

    let neo4j = state.orchestrator.neo4j_arc();
    let auth_result = super::ws_auth::ws_authenticate(
        &headers,
        &state.auth_config,
        &neo4j,
        query.ticket.as_deref(),
        &state.ws_ticket_store,
    )
    .await;

    match auth_result {
        CookieAuthResult::Authenticated(claims) => {
            info!(
                email = %claims.email,
                project_id = %project_id,
                last_event_seq = ?query.last_event_seq,
                "WS /ws/activity: authenticated, upgrading"
            );
            let entity_filter = parse_csv_set(query.entity_types.as_deref());
            let status_filter = parse_csv_set(query.statuses.as_deref());
            let last_event_seq = query.last_event_seq;

            Ok(ws.on_upgrade(move |socket| {
                handle_activity_ws(
                    socket,
                    state,
                    project_id,
                    entity_filter,
                    status_filter,
                    last_event_seq,
                    claims,
                )
            }))
        }
        CookieAuthResult::Invalid(reason) => {
            warn!(reason = %reason, "WS /ws/activity: auth REJECTED (401)");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

// ============================================================================
// Main loop
// ============================================================================

#[allow(clippy::too_many_arguments)]
async fn handle_activity_ws(
    socket: WebSocket,
    state: OrchestratorState,
    project_id: String,
    entity_filter: Option<HashSet<String>>,
    status_filter: Option<HashSet<String>>,
    last_event_seq: Option<u64>,
    claims: Claims,
) {
    // Wait for client "ready" then send auth_ok.
    let mut socket = socket;
    super::ws_auth::wait_ready_then_auth_ok(&mut socket, &claims).await;
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Lazily start the hub if needed and subscribe.
    let hub = get_or_init_activity_hub(&state).await;
    let mut rx = hub.subscribe();

    // Replay from the ring buffer if requested.
    let mut replayed_count = 0usize;
    if let Some(last_seq) = last_event_seq {
        let replay = hub.replay_since(last_seq).await;
        replayed_count = replay.len();
        for evt in replay {
            if !passes_filters(&evt, &project_id, &entity_filter, &status_filter) {
                continue;
            }
            if !send_json(&mut ws_sender, &OutFrame::Activity { event: &evt }).await {
                debug!("WS /ws/activity: client disconnected during replay");
                return;
            }
        }
    }

    // Handshake `connected` frame so the client knows how far we replayed.
    let head_seq = state.event_bus.local_bus().current_sequence();
    if !send_json(
        &mut ws_sender,
        &OutFrame::Connected {
            last_seq: head_seq,
            replayed: replayed_count,
        },
    )
    .await
    {
        return;
    }

    let mut ping_interval = interval(Duration::from_secs(30));
    ping_interval.tick().await; // skip first immediate tick

    loop {
        tokio::select! {
            // Live events from the hub
            result = rx.recv() => {
                match result {
                    Ok(event) => {
                        if !passes_filters(&event, &project_id, &entity_filter, &status_filter) {
                            continue;
                        }
                        if !send_json(&mut ws_sender, &OutFrame::Activity { event: &event }).await {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            skipped = n,
                            project_id = %project_id,
                            "WS /ws/activity: subscriber lagged — emitting lag_dropped"
                        );
                        // Tell the client so it can resync via REST.
                        if !send_json(
                            &mut ws_sender,
                            &OutFrame::LagDropped { skipped: n },
                        )
                        .await
                        {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("WS /ws/activity: hub channel closed");
                        break;
                    }
                }
            }

            // Inbound: handle close + pong
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => {
                        debug!("WS /ws/activity: client disconnected");
                        break;
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        debug!(error = %e, "WS /ws/activity: receive error");
                        break;
                    }
                }
            }

            // Keep-alive ping
            _ = ping_interval.tick() => {
                if ws_sender.send(Message::Ping(vec![].into())).await.is_err() {
                    break;
                }
            }
        }
    }

    debug!(email = %claims.email, project_id = %project_id, "WS /ws/activity: closed");
}

// ============================================================================
// Helpers
// ============================================================================

/// Parse a CSV list (case-insensitive, trimmed) into a [`HashSet`].
fn parse_csv_set(raw: Option<&str>) -> Option<HashSet<String>> {
    raw.map(|s| {
        s.split(',')
            .map(|p| p.trim().to_lowercase())
            .filter(|p| !p.is_empty())
            .collect::<HashSet<_>>()
    })
    .filter(|set| !set.is_empty())
}

/// Apply project_id + entity_types + statuses filters to an event.
fn passes_filters(
    event: &ActivityEvent,
    project_id: &str,
    entity_filter: &Option<HashSet<String>>,
    status_filter: &Option<HashSet<String>>,
) -> bool {
    // project_id filter — events without a project_id pass through (global),
    // events with a different project_id are dropped.
    if let Some(evt_pid) = event_project_id(event) {
        if evt_pid != project_id {
            return false;
        }
    }

    // entity_types filter — accepts both ActivityEvent.kind and concrete
    // entity_type strings (snake_case).
    if let Some(set) = entity_filter {
        let kind = event.kind();
        let entity = event_entity_type(event);
        let kind_match = set.contains(kind);
        let entity_match = entity.as_deref().map(|e| set.contains(e)).unwrap_or(false);
        if !kind_match && !entity_match {
            return false;
        }
    }

    // statuses filter — checks payload's `status` or `new_status` strings.
    if let Some(set) = status_filter {
        if let Some(status) = event_status(event) {
            if !set.contains(&status) {
                return false;
            }
        } else {
            // No status field on this event — drop when a status filter is active.
            return false;
        }
    }

    true
}

/// Extract the project_id from an [`ActivityEvent`], when present.
///
/// `Runner`/`ProtocolProgress`/`Chat` variants currently do NOT carry a
/// project_id at the envelope level — they rely on a downstream join. Only
/// `Crud` events carry an explicit `project_id` field. Returning `None`
/// allows those non-project events through (they're typically global runner
/// signals worth seeing in the hub).
fn event_project_id(event: &ActivityEvent) -> Option<&str> {
    match event {
        ActivityEvent::Crud { project_id, .. } => project_id.as_deref(),
        _ => None,
    }
}

/// Extract the concrete entity_type (snake_case) when the event has one.
fn event_entity_type(event: &ActivityEvent) -> Option<String> {
    match event {
        ActivityEvent::Crud { entity_type, .. } => serde_json::to_value(entity_type)
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string())),
        _ => None,
    }
}

/// Best-effort extraction of a `status`-like field from the event's payload.
fn event_status(event: &ActivityEvent) -> Option<String> {
    match event {
        ActivityEvent::Crud { payload, .. } => payload
            .get("new_status")
            .or_else(|| payload.get("status"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase()),
        ActivityEvent::ProtocolProgress { status, .. } => {
            // RunStatus is Serialize — convert via JSON for stable string form.
            serde_json::to_value(status)
                .ok()
                .and_then(|v| v.as_str().map(|s| s.to_lowercase()))
        }
        ActivityEvent::Runner { event, .. } => Some(runner_event_kind(event)),
        ActivityEvent::Chat { .. } => None,
    }
}

/// Map a [`RunnerEvent`] variant to a short status-like string used by the
/// `statuses` filter (kept short to align with the typical UI filter chips).
fn runner_event_kind(event: &RunnerEvent) -> String {
    match event {
        RunnerEvent::PlanStarted { .. } => "plan_started".into(),
        RunnerEvent::WaveStarted { .. } => "wave_started".into(),
        RunnerEvent::TaskStarted { .. } => "task_started".into(),
        RunnerEvent::TaskCompleted { .. } => "task_completed".into(),
        RunnerEvent::TaskFailed { .. } => "task_failed".into(),
        RunnerEvent::TaskTimeout { .. } => "task_timeout".into(),
        RunnerEvent::WaveCompleted { .. } => "wave_completed".into(),
        RunnerEvent::PlanCompleted { .. } => "plan_completed".into(),
        RunnerEvent::TaskCompletedWithoutSteps { .. } => "task_completed_without_steps".into(),
        RunnerEvent::CwdMismatch { .. } => "cwd_mismatch".into(),
        RunnerEvent::TaskSpawningTimeout { .. } => "task_spawning_timeout".into(),
        RunnerEvent::RunnerError { .. } => "runner_error".into(),
        RunnerEvent::BudgetExceeded { .. } => "budget_exceeded".into(),
        RunnerEvent::WorktreeRecovery { .. } => "worktree_recovery".into(),
        RunnerEvent::LifecycleTransition { .. } => "lifecycle_transition".into(),
    }
}

/// Send a JSON message over the WebSocket. Returns false if send failed.
async fn send_json<T: Serialize>(
    ws_sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    value: &T,
) -> bool {
    match serde_json::to_string(value) {
        Ok(json) => {
            if ws_sender.send(Message::Text(json.into())).await.is_err() {
                debug!("WebSocket send failed, client disconnected");
                return false;
            }
            true
        }
        Err(e) => {
            warn!("Failed to serialize WS activity frame: {}", e);
            true
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, CrudEvent, EntityType, EventBus};
    use crate::protocol::models::RunStatus as ProtocolRunStatus;
    use uuid::Uuid;

    fn run_id() -> Uuid {
        Uuid::nil()
    }

    fn sample_task_started(run: Uuid) -> RunnerEvent {
        RunnerEvent::TaskStarted {
            run_id: run,
            task_id: Uuid::nil(),
            task_title: "T".into(),
            wave_number: 1,
        }
    }

    // ------------------------------------------------------------------
    // transform_crud
    // ------------------------------------------------------------------

    #[test]
    fn transform_crud_drops_unrelated_entities() {
        let bus = EventBus::default();
        let crud = CrudEvent::new(EntityType::Note, CrudAction::Created, "n1");
        assert!(transform_crud(&bus, crud).is_none());
    }

    #[test]
    fn transform_crud_keeps_plan_task_protocolrun() {
        let bus = EventBus::default();
        for et in [EntityType::Plan, EntityType::Task, EntityType::ProtocolRun] {
            let crud = CrudEvent::new(et.clone(), CrudAction::Updated, "x");
            let out = transform_crud(&bus, crud);
            assert!(out.is_some(), "{:?} must be kept", et);
            assert_eq!(out.unwrap().kind(), "crud");
        }
    }

    #[test]
    fn transform_crud_extracts_runner_event_from_payload() {
        let bus = EventBus::default();
        let runner_event = sample_task_started(run_id());
        let payload = serde_json::to_value(&runner_event).unwrap();
        let crud = CrudEvent {
            entity_type: EntityType::Runner,
            action: CrudAction::Updated,
            entity_id: String::new(),
            related: None,
            payload,
            timestamp: chrono::Utc::now().to_rfc3339(),
            project_id: None,
        };

        let activity = transform_crud(&bus, crud).expect("expected ActivityEvent::Runner");
        assert_eq!(activity.kind(), "runner");
        // First call to next_sequence() returned 0.
        assert_eq!(activity.seq(), 0);
    }

    #[test]
    fn transform_crud_runner_with_garbage_payload_returns_none() {
        let bus = EventBus::default();
        let crud = CrudEvent {
            entity_type: EntityType::Runner,
            action: CrudAction::Updated,
            entity_id: String::new(),
            related: None,
            payload: serde_json::json!({"not_a_runner_event": true}),
            timestamp: chrono::Utc::now().to_rfc3339(),
            project_id: None,
        };
        assert!(transform_crud(&bus, crud).is_none());
    }

    // ------------------------------------------------------------------
    // filters
    // ------------------------------------------------------------------

    fn make_crud_activity(project_id: Option<&str>, status: Option<&str>) -> ActivityEvent {
        let mut crud = CrudEvent::new(EntityType::Task, CrudAction::StatusChanged, "t");
        if let Some(p) = project_id {
            crud = crud.with_project_id(p);
        }
        if let Some(s) = status {
            crud = crud.with_payload(serde_json::json!({"new_status": s}));
        }
        ActivityEvent::try_from_crud(1, crud).unwrap()
    }

    #[test]
    fn filter_keeps_matching_project_id() {
        let evt = make_crud_activity(Some("p1"), None);
        assert!(passes_filters(&evt, "p1", &None, &None));
    }

    #[test]
    fn filter_drops_other_project_id() {
        let evt = make_crud_activity(Some("p2"), None);
        assert!(!passes_filters(&evt, "p1", &None, &None));
    }

    #[test]
    fn filter_passes_event_without_project_id() {
        // Runner events have no project_id — they're forwarded as global signals.
        let evt = ActivityEvent::from_runner(0, sample_task_started(run_id()));
        assert!(passes_filters(&evt, "any-project", &None, &None));
    }

    #[test]
    fn filter_entity_types_kind_match() {
        let evt = ActivityEvent::from_runner(0, sample_task_started(run_id()));
        let mut set = HashSet::new();
        set.insert("runner".to_string());
        assert!(passes_filters(&evt, "p", &Some(set), &None));
    }

    #[test]
    fn filter_entity_types_concrete_entity_match() {
        let evt = make_crud_activity(None, None);
        let mut set = HashSet::new();
        set.insert("task".to_string());
        assert!(passes_filters(&evt, "p", &Some(set), &None));
    }

    #[test]
    fn filter_entity_types_no_match_drops() {
        let evt = make_crud_activity(None, None);
        let mut set = HashSet::new();
        set.insert("plan".to_string());
        assert!(!passes_filters(&evt, "p", &Some(set), &None));
    }

    #[test]
    fn filter_status_match() {
        let evt = make_crud_activity(None, Some("running"));
        let mut set = HashSet::new();
        set.insert("running".to_string());
        assert!(passes_filters(&evt, "p", &None, &Some(set)));
    }

    #[test]
    fn filter_status_no_match_drops() {
        let evt = make_crud_activity(None, Some("running"));
        let mut set = HashSet::new();
        set.insert("completed".to_string());
        assert!(!passes_filters(&evt, "p", &None, &Some(set)));
    }

    #[test]
    fn filter_status_present_filter_no_field_drops() {
        let evt = make_crud_activity(None, None);
        let mut set = HashSet::new();
        set.insert("running".to_string());
        // The event has no status field → drop.
        assert!(!passes_filters(&evt, "p", &None, &Some(set)));
    }

    #[test]
    fn filter_status_uses_runner_event_kind() {
        let evt = ActivityEvent::from_runner(0, sample_task_started(run_id()));
        let mut set = HashSet::new();
        set.insert("task_started".to_string());
        assert!(passes_filters(&evt, "p", &None, &Some(set)));
    }

    #[test]
    fn filter_status_uses_protocol_status() {
        let evt = ActivityEvent::from_protocol_progress(crate::events::ProtocolProgress {
            seq: 0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            run_id: Uuid::nil(),
            protocol_id: Uuid::nil(),
            current_state: Uuid::nil(),
            state_name: "Init".into(),
            status: ProtocolRunStatus::Running,
            progress: None,
        });
        let mut set = HashSet::new();
        set.insert("running".to_string());
        assert!(passes_filters(&evt, "p", &None, &Some(set)));
    }

    // ------------------------------------------------------------------
    // parse_csv_set
    // ------------------------------------------------------------------

    #[test]
    fn parse_csv_basic() {
        let s = parse_csv_set(Some("Plan, Task ,protocol_run")).unwrap();
        assert!(s.contains("plan"));
        assert!(s.contains("task"));
        assert!(s.contains("protocol_run"));
    }

    #[test]
    fn parse_csv_none() {
        assert!(parse_csv_set(None).is_none());
    }

    #[test]
    fn parse_csv_empty_string_returns_none() {
        assert!(parse_csv_set(Some("")).is_none());
        assert!(parse_csv_set(Some("  ,, ")).is_none());
    }

    // ------------------------------------------------------------------
    // ActivityHub — broadcast + ring buffer
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn hub_broadcasts_pushed_events_to_subscribers() {
        let hub = ActivityHub::new();
        let mut rx = hub.subscribe();
        let evt = ActivityEvent::from_runner(7, sample_task_started(run_id()));
        hub.push(evt.clone()).await;
        let received = rx.recv().await.unwrap();
        assert_eq!(received.seq(), 7);
        assert_eq!(received.kind(), "runner");
    }

    #[tokio::test]
    async fn hub_ring_buffer_replay_since_returns_newer_events() {
        let hub = ActivityHub::new();
        // Push 3 events with seq 0,1,2
        for seq in 0..3 {
            hub.push(ActivityEvent::from_runner(seq, sample_task_started(run_id())))
                .await;
        }
        let replay = hub.replay_since(0).await;
        // Should return events with seq > 0 → seq 1 and 2
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].seq(), 1);
        assert_eq!(replay[1].seq(), 2);
    }

    #[tokio::test]
    async fn hub_ring_buffer_caps_at_capacity() {
        let hub = ActivityHub::new();
        // Push more than capacity
        for seq in 0..(RING_BUFFER_CAPACITY as u64 + 50) {
            hub.push(ActivityEvent::from_runner(seq, sample_task_started(run_id())))
                .await;
        }
        let all = hub.replay_since(0).await;
        // Cap is RING_BUFFER_CAPACITY; oldest events were dropped.
        assert!(all.len() <= RING_BUFFER_CAPACITY);
        // The oldest retained event has seq ≥ 50 (since we pushed 1050 with cap 1000).
        let min_seq = all.iter().map(|e| e.seq()).min().unwrap();
        assert!(
            min_seq >= 50,
            "expected oldest seq ≥ 50 after eviction, got {}",
            min_seq
        );
    }

    #[tokio::test]
    async fn hub_replay_since_filters_strictly_greater() {
        let hub = ActivityHub::new();
        for seq in 0..5 {
            hub.push(ActivityEvent::from_runner(seq, sample_task_started(run_id())))
                .await;
        }
        // last_seq=3 → return seq 4 only
        let replay = hub.replay_since(3).await;
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].seq(), 4);
    }

    #[tokio::test]
    async fn hub_subscriber_count_tracks_receivers() {
        let hub = ActivityHub::new();
        assert_eq!(hub.subscriber_count(), 0);
        let _rx1 = hub.subscribe();
        let _rx2 = hub.subscribe();
        assert_eq!(hub.subscriber_count(), 2);
    }

    #[tokio::test]
    async fn hub_lag_does_not_lose_ring_entries() {
        // Even when broadcast receivers are absent (no subs), the ring buffer
        // still records events so future reconnects can replay.
        let hub = ActivityHub::new();
        for seq in 0..10 {
            hub.push(ActivityEvent::from_runner(seq, sample_task_started(run_id())))
                .await;
        }
        let replay = hub.replay_since(0).await;
        assert_eq!(replay.len(), 9);
    }

    // ------------------------------------------------------------------
    // Out-frame serialization
    // ------------------------------------------------------------------

    #[test]
    fn out_frame_activity_serializes_with_type_tag() {
        let evt = ActivityEvent::from_runner(1, sample_task_started(run_id()));
        let frame = OutFrame::Activity { event: &evt };
        let json = serde_json::to_string(&frame).unwrap();
        assert!(json.contains("\"type\":\"activity\""));
        assert!(json.contains("\"kind\":\"runner\""));
    }

    #[test]
    fn out_frame_lag_dropped_serializes() {
        let frame = OutFrame::LagDropped { skipped: 42 };
        let json = serde_json::to_string(&frame).unwrap();
        assert!(json.contains("\"type\":\"lag_dropped\""));
        assert!(json.contains("\"skipped\":42"));
    }

    #[test]
    fn out_frame_connected_serializes() {
        let frame = OutFrame::Connected {
            last_seq: 12,
            replayed: 3,
        };
        let json = serde_json::to_string(&frame).unwrap();
        assert!(json.contains("\"type\":\"connected\""));
        assert!(json.contains("\"last_seq\":12"));
        assert!(json.contains("\"replayed\":3"));
    }
}
