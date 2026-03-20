//! WebSocket handler for real-time plan run streaming
//!
//! Multiplexes two event streams into a single WebSocket connection:
//! - **RunnerEvents** (via CrudEvent bridge): TaskStarted, TaskCompleted, WaveStarted, etc.
//! - **ChatEvents** from ALL child sessions of the run: streaming text, tool_use, errors
//!
//! Sessions are discovered dynamically from `RUNNER_STATE.active_agents` and subscribed
//! to via `ChatManager::subscribe()`. New sessions (from subsequent waves) are picked up
//! automatically on each RunnerEvent::TaskStarted.

use super::handlers::OrchestratorState;
use super::ws_auth::CookieAuthResult;
use crate::auth::jwt::Claims;
use crate::chat::types::ChatEvent;
use crate::runner::{RUNNER_STATE, RunnerEvent};
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Path, Query, State, WebSocketUpgrade,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use tokio::sync::broadcast;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Query parameters for the run WebSocket
#[derive(Debug, Deserialize, Default)]
pub struct WsRunQuery {
    /// One-time ticket for auth (fallback when cookies aren't sent on WS upgrade)
    pub ticket: Option<String>,
}

/// WebSocket upgrade handler for `/ws/run/{run_id}`
///
/// Authentication: validates cookie or ticket BEFORE upgrade.
pub async fn ws_run(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Path(run_id): Path<Uuid>,
    Query(query): Query<WsRunQuery>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
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
            info!(email = %claims.email, run_id = %run_id, "WS /ws/run: authenticated, upgrading");
            Ok(ws.on_upgrade(move |socket| handle_run_ws(socket, state, run_id, claims)))
        }
        CookieAuthResult::Invalid(reason) => {
            warn!(reason = %reason, "WS /ws/run: auth REJECTED (401)");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// Envelope for events sent over the run WebSocket
#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RunWsEvent<'a> {
    /// A RunnerEvent (TaskStarted, TaskCompleted, WaveStarted, etc.)
    RunnerEvent {
        event: &'a RunnerEvent,
    },
    /// A ChatEvent from one of the run's child sessions
    ChatEvent {
        task_id: Uuid,
        session_id: String,
        event: &'a ChatEvent,
    },
    /// Run status snapshot (sent on connect and periodically)
    RunStatus {
        running: bool,
        tasks_completed: usize,
        tasks_total: usize,
        active_sessions: Vec<SessionInfo>,
    },
}

#[derive(serde::Serialize)]
struct SessionInfo {
    task_id: Uuid,
    task_title: String,
    session_id: String,
}

/// Tracked state for a subscribed chat session
struct TrackedSession {
    task_id: Uuid,
    session_id: String,
    rx: broadcast::Receiver<ChatEvent>,
}

/// Main WebSocket loop for a plan run
async fn handle_run_ws(
    socket: WebSocket,
    state: OrchestratorState,
    run_id: Uuid,
    claims: Claims,
) {
    // Wait for client "ready" signal (must be done before split)
    let mut socket = socket;
    super::ws_auth::wait_ready_then_auth_ok(&mut socket, &claims).await;
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Subscribe to CrudEvents (RunnerEvents are bridged here)
    let mut crud_rx = state.event_bus.subscribe();

    // Track chat sessions we've already subscribed to
    let mut tracked_sessions: HashMap<String, TrackedSession> = HashMap::new();
    let mut known_session_ids: HashSet<String> = HashSet::new();

    // Discover and subscribe to existing sessions
    discover_sessions(&state, run_id, &mut tracked_sessions, &mut known_session_ids).await;

    // Send initial status
    send_run_status(&mut ws_sender, run_id, &tracked_sessions).await;

    let mut ping_interval = interval(Duration::from_secs(30));
    ping_interval.tick().await; // skip first tick

    let mut discovery_interval = interval(Duration::from_secs(2));
    discovery_interval.tick().await; // skip first tick

    loop {
        tokio::select! {
            // Forward CrudEvents that are RunnerEvents for this run
            result = crud_rx.recv() => {
                match result {
                    Ok(event) => {
                        // Try to extract RunnerEvent from the CrudEvent
                        if let Some(runner_event) = extract_runner_event(&event, run_id) {
                            // On TaskStarted, discover new sessions
                            if matches!(runner_event, RunnerEvent::TaskStarted { .. }) {
                                discover_sessions(&state, run_id, &mut tracked_sessions, &mut known_session_ids).await;
                            }

                            let envelope = RunWsEvent::RunnerEvent { event: &runner_event };
                            if !send_json(&mut ws_sender, &envelope).await {
                                break;
                            }

                            // On PlanCompleted, send final status and close
                            if matches!(runner_event, RunnerEvent::PlanCompleted { .. }) {
                                send_run_status(&mut ws_sender, run_id, &tracked_sessions).await;
                                break;
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("WS run {}: CrudEvent receiver lagged by {} events", run_id, n);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        debug!("WS run {}: CrudEvent channel closed", run_id);
                        break;
                    }
                }
            }

            // Poll all tracked chat sessions for ChatEvents
            Some((session_key, chat_event)) = poll_chat_sessions(&mut tracked_sessions) => {
                if let Some(tracked) = tracked_sessions.get(&session_key) {
                    let envelope = RunWsEvent::ChatEvent {
                        task_id: tracked.task_id,
                        session_id: tracked.session_id.clone(),
                        event: &chat_event,
                    };
                    if !send_json(&mut ws_sender, &envelope).await {
                        break;
                    }
                }
            }

            // Periodic discovery of new sessions (wave N+1)
            _ = discovery_interval.tick() => {
                discover_sessions(&state, run_id, &mut tracked_sessions, &mut known_session_ids).await;
            }

            // Handle client messages (ping/pong, close)
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => {
                        debug!("WS run {}: client disconnected", run_id);
                        break;
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    Some(Ok(_)) => {} // Ignore other messages
                    Some(Err(e)) => {
                        debug!("WS run {}: error: {}", run_id, e);
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

    debug!("WS run {} disconnected ({})", run_id, claims.email);
}

/// Discover active sessions for the run and subscribe to new ones
async fn discover_sessions(
    state: &OrchestratorState,
    run_id: Uuid,
    tracked: &mut HashMap<String, TrackedSession>,
    known_ids: &mut HashSet<String>,
) {
    let global = RUNNER_STATE.read().await;
    let agents = match global.as_ref() {
        Some(s) if s.run_id == run_id => &s.active_agents,
        _ => return,
    };

    for agent in agents {
        if let Some(session_id) = &agent.session_id {
            let sid = session_id.to_string();
            if known_ids.contains(&sid) {
                continue;
            }
            known_ids.insert(sid.clone());

            // Try to subscribe to this session's ChatEvents
            if let Some(ref cm) = state.chat_manager {
                match cm.subscribe(&sid).await {
                    Ok(rx) => {
                        info!(
                            "WS run {}: subscribed to chat session {} (task: {})",
                            run_id, sid, agent.task_title
                        );
                        tracked.insert(
                            sid.clone(),
                            TrackedSession {
                                task_id: agent.task_id,
                                session_id: sid,
                                rx,
                            },
                        );
                    }
                    Err(e) => {
                        debug!(
                            "WS run {}: failed to subscribe to session {}: {}",
                            run_id, sid, e
                        );
                    }
                }
            }
        }
    }
}

/// Poll all tracked chat sessions, returning the first available event
async fn poll_chat_sessions(
    tracked: &mut HashMap<String, TrackedSession>,
) -> Option<(String, ChatEvent)> {
    if tracked.is_empty() {
        // Don't return immediately — yield to other select branches
        std::future::pending::<()>().await;
        return None;
    }

    // Use a FuturesUnordered-like approach: try_recv on all, then yield
    // For simplicity, we do a round-robin poll with try_recv
    for (key, session) in tracked.iter_mut() {
        match session.rx.try_recv() {
            Ok(event) => return Some((key.clone(), event)),
            Err(broadcast::error::TryRecvError::Empty) => continue,
            Err(broadcast::error::TryRecvError::Closed) => {
                debug!("Chat session {} broadcast closed", key);
                continue;
            }
            Err(broadcast::error::TryRecvError::Lagged(n)) => {
                warn!("Chat session {} lagged by {} events", key, n);
                continue;
            }
        }
    }

    // No events available — sleep briefly to avoid busy-loop
    tokio::time::sleep(Duration::from_millis(50)).await;
    None
}

/// Extract a RunnerEvent from a CrudEvent if it matches the run_id.
///
/// RunnerEvents are bridged as CrudEvents with `entity_type: EntityType::Runner`
/// and the RunnerEvent serialized in the `payload` field.
fn extract_runner_event(
    crud_event: &crate::events::CrudEvent,
    run_id: Uuid,
) -> Option<RunnerEvent> {
    use crate::events::EntityType;

    // Only Runner entity_type events contain RunnerEvents
    if crud_event.entity_type != EntityType::Runner {
        return None;
    }

    // payload is serde_json::Value — check it's not null
    if crud_event.payload.is_null() {
        return None;
    }

    // Try to deserialize as RunnerEvent
    let runner_event: RunnerEvent = serde_json::from_value(crud_event.payload.clone()).ok()?;

    // Filter by run_id — extract from each variant
    let event_run_id = match &runner_event {
        RunnerEvent::PlanStarted { run_id: rid, .. }
        | RunnerEvent::WaveStarted { run_id: rid, .. }
        | RunnerEvent::TaskStarted { run_id: rid, .. }
        | RunnerEvent::TaskCompleted { run_id: rid, .. }
        | RunnerEvent::TaskFailed { run_id: rid, .. }
        | RunnerEvent::PlanCompleted { run_id: rid, .. } => Some(*rid),
        _ => None,
    };

    if event_run_id == Some(run_id) {
        Some(runner_event)
    } else {
        None
    }
}

/// Send a run status snapshot
async fn send_run_status(
    ws_sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    run_id: Uuid,
    tracked: &HashMap<String, TrackedSession>,
) {
    let global = RUNNER_STATE.read().await;
    let (running, completed, total) = match global.as_ref() {
        Some(s) if s.run_id == run_id => (true, s.completed_tasks.len(), s.total_tasks),
        _ => (false, 0, 0),
    };

    let active_sessions: Vec<SessionInfo> = tracked
        .values()
        .map(|t| SessionInfo {
            task_id: t.task_id,
            task_title: String::new(), // Could be enriched from RUNNER_STATE
            session_id: t.session_id.clone(),
        })
        .collect();

    let envelope = RunWsEvent::RunStatus {
        running,
        tasks_completed: completed,
        tasks_total: total,
        active_sessions,
    };

    send_json(ws_sender, &envelope).await;
}

/// Send a JSON message over the WebSocket. Returns false if send failed.
async fn send_json<T: serde::Serialize>(
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
            warn!("Failed to serialize WS event: {}", e);
            true // Don't disconnect on serialization errors
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, CrudEvent, EntityType};
    use crate::runner::RunnerEvent;
    use uuid::Uuid;

    fn make_crud_runner_event(runner_event: &RunnerEvent) -> CrudEvent {
        CrudEvent {
            entity_type: EntityType::Runner,
            action: CrudAction::Updated,
            entity_id: String::new(),
            related: None,
            payload: serde_json::to_value(runner_event).unwrap(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            project_id: None,
        }
    }

    #[test]
    fn test_extract_runner_event_matches_run_id() {
        let run_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();
        let event = RunnerEvent::TaskStarted {
            run_id,
            task_id,
            task_title: "Test task".into(),
            wave_number: 1,
        };
        let crud = make_crud_runner_event(&event);

        let result = extract_runner_event(&crud, run_id);
        assert!(result.is_some(), "Should extract event matching run_id");

        if let Some(RunnerEvent::TaskStarted {
            run_id: rid,
            task_id: tid,
            ..
        }) = result
        {
            assert_eq!(rid, run_id);
            assert_eq!(tid, task_id);
        } else {
            panic!("Expected TaskStarted variant");
        }
    }

    #[test]
    fn test_extract_runner_event_rejects_wrong_run_id() {
        let run_id = Uuid::new_v4();
        let other_run_id = Uuid::new_v4();
        let event = RunnerEvent::WaveStarted {
            run_id,
            wave_number: 1,
            task_count: 3,
        };
        let crud = make_crud_runner_event(&event);

        let result = extract_runner_event(&crud, other_run_id);
        assert!(result.is_none(), "Should reject event with different run_id");
    }

    #[test]
    fn test_extract_runner_event_rejects_non_runner_entity() {
        let run_id = Uuid::new_v4();
        let event = RunnerEvent::PlanStarted {
            run_id,
            plan_id: Uuid::new_v4(),
            plan_title: "Test".into(),
            total_tasks: 5,
            total_waves: 2,
            prediction: None,
        };
        let mut crud = make_crud_runner_event(&event);
        crud.entity_type = EntityType::Plan; // Not Runner

        let result = extract_runner_event(&crud, run_id);
        assert!(result.is_none(), "Should reject non-Runner entity types");
    }

    #[test]
    fn test_extract_runner_event_handles_null_payload() {
        let crud = CrudEvent {
            entity_type: EntityType::Runner,
            action: CrudAction::Updated,
            entity_id: String::new(),
            related: None,
            payload: serde_json::Value::Null,
            timestamp: chrono::Utc::now().to_rfc3339(),
            project_id: None,
        };

        let result = extract_runner_event(&crud, Uuid::new_v4());
        assert!(result.is_none(), "Should handle null payload gracefully");
    }

    #[test]
    fn test_extract_runner_event_handles_invalid_payload() {
        let crud = CrudEvent {
            entity_type: EntityType::Runner,
            action: CrudAction::Updated,
            entity_id: String::new(),
            related: None,
            payload: serde_json::json!({"garbage": true}),
            timestamp: chrono::Utc::now().to_rfc3339(),
            project_id: None,
        };

        let result = extract_runner_event(&crud, Uuid::new_v4());
        assert!(
            result.is_none(),
            "Should handle invalid payload gracefully"
        );
    }

    #[tokio::test]
    async fn test_poll_chat_sessions_empty_pending() {
        let mut tracked: HashMap<String, TrackedSession> = HashMap::new();

        // poll_chat_sessions with empty map should pend forever — use timeout
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            poll_chat_sessions(&mut tracked),
        )
        .await;

        assert!(result.is_err(), "Should pend (not return) when no sessions");
    }

    #[tokio::test]
    async fn test_poll_chat_sessions_receives_event() {
        let (tx, rx) = broadcast::channel::<ChatEvent>(16);
        let task_id = Uuid::new_v4();
        let session_id = "test-session-1".to_string();

        let mut tracked: HashMap<String, TrackedSession> = HashMap::new();
        tracked.insert(
            session_id.clone(),
            TrackedSession {
                task_id,
                session_id: session_id.clone(),
                rx,
            },
        );

        // Send a ChatEvent
        let chat_event = ChatEvent::AssistantText {
            content: "hello world".into(),
            parent_tool_use_id: None,
        };
        tx.send(chat_event).unwrap();

        let result = tokio::time::timeout(
            Duration::from_millis(200),
            poll_chat_sessions(&mut tracked),
        )
        .await;

        assert!(result.is_ok(), "Should return within timeout");
        let result = result.unwrap();
        assert!(result.is_some(), "Should have received an event");
        let (key, _event) = result.unwrap();
        assert_eq!(key, session_id);
    }

    #[tokio::test]
    async fn test_poll_chat_sessions_no_event_returns_none() {
        let (_tx, rx) = broadcast::channel::<ChatEvent>(16);
        let task_id = Uuid::new_v4();

        let mut tracked: HashMap<String, TrackedSession> = HashMap::new();
        tracked.insert(
            "sid".to_string(),
            TrackedSession {
                task_id,
                session_id: "sid".to_string(),
                rx,
            },
        );

        // No event sent — should sleep 50ms then return None
        let result = tokio::time::timeout(
            Duration::from_millis(200),
            poll_chat_sessions(&mut tracked),
        )
        .await;

        assert!(result.is_ok(), "Should return within timeout");
        assert!(result.unwrap().is_none(), "Should return None when no events");
    }

    #[test]
    fn test_run_ws_event_serialization() {
        let run_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();

        // Test RunnerEvent variant
        let runner_event = RunnerEvent::TaskCompleted {
            run_id,
            task_id,
            task_title: "Do stuff".into(),
            cost_usd: 0.05,
            duration_secs: 120.0,
        };
        let envelope = RunWsEvent::RunnerEvent {
            event: &runner_event,
        };
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(json.contains("\"type\":\"runner_event\""));

        // Test RunStatus variant
        let status = RunWsEvent::RunStatus {
            running: true,
            tasks_completed: 2,
            tasks_total: 5,
            active_sessions: vec![SessionInfo {
                task_id,
                task_title: "Test".into(),
                session_id: "ses-123".into(),
            }],
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"type\":\"run_status\""));
        assert!(json.contains("\"tasks_completed\":2"));
        assert!(json.contains("\"tasks_total\":5"));
    }
}
