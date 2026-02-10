//! WebSocket handlers for real-time CRUD event notifications

use super::handlers::OrchestratorState;
use crate::events::CrudEvent;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use std::collections::{HashSet, VecDeque};
use tokio::time::{interval, Duration};
use tracing::{debug, warn};

/// Max number of recent event fingerprints kept for deduplication.
/// Events arriving via both local broadcast and NATS within the window
/// are sent to the client only once.
const DEDUP_WINDOW_SIZE: usize = 128;

/// Query parameters for filtering WebSocket events
#[derive(Debug, Deserialize, Default)]
pub struct WsQuery {
    /// Comma-separated entity types to subscribe to (e.g. "plan,task,note")
    pub entity_types: Option<String>,
    /// Filter by project ID
    pub project_id: Option<String>,
}

/// WebSocket upgrade handler for `/ws/events`
pub async fn ws_events(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Query(query): Query<WsQuery>,
) -> impl IntoResponse {
    let entity_filter: Option<HashSet<String>> = query.entity_types.map(|types| {
        types
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    });
    let project_filter = query.project_id;

    ws.on_upgrade(move |socket| handle_ws(socket, state, entity_filter, project_filter))
}

/// Build a deduplication fingerprint for a CrudEvent.
///
/// Uses timestamp (nanosecond precision) + entity_id to identify duplicates.
fn event_fingerprint(event: &CrudEvent) -> String {
    format!("{}:{}", event.timestamp, event.entity_id)
}

/// Check if the event passes entity_type and project_id filters.
fn passes_filters(
    event: &CrudEvent,
    entity_filter: &Option<HashSet<String>>,
    project_filter: &Option<String>,
) -> bool {
    // Apply entity_type filter
    if let Some(ref filter) = entity_filter {
        let entity_str = serde_json::to_value(&event.entity_type)
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();
        if !filter.contains(&entity_str) {
            return false;
        }
    }

    // Apply project_id filter
    if let Some(ref pid) = project_filter {
        match &event.project_id {
            Some(event_pid) if event_pid == pid => {}
            Some(_) => return false,
            // Events without project_id pass through (global events)
            None => {}
        }
    }

    true
}

/// Handle an individual WebSocket connection
async fn handle_ws(
    mut socket: WebSocket,
    state: OrchestratorState,
    entity_filter: Option<HashSet<String>>,
    project_filter: Option<String>,
) {
    // ========================================================================
    // Phase 0: Authenticate via first message
    // ========================================================================
    let claims = match super::ws_auth::ws_authenticate(&mut socket, &state.auth_config).await {
        Ok(claims) => claims,
        Err(reason) => {
            debug!(reason = %reason, "WS events: auth failed");
            return;
        }
    };

    let (mut ws_sender, mut ws_receiver) = socket.split();
    let mut event_rx = state.event_bus.subscribe();

    // Subscribe to NATS CRUD events if available
    let mut nats_sub = if let Some(ref nats) = state.nats_emitter {
        match nats.subscribe_crud_events().await {
            Ok(sub) => {
                debug!("WS handler: subscribed to NATS CRUD events");
                Some(sub)
            }
            Err(e) => {
                warn!("WS handler: failed to subscribe to NATS CRUD events: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Deduplication window: recent event fingerprints
    let mut seen: VecDeque<String> = VecDeque::with_capacity(DEDUP_WINDOW_SIZE);
    let mut seen_set: HashSet<String> = HashSet::with_capacity(DEDUP_WINDOW_SIZE);

    // Ping interval (30s)
    let mut ping_interval = interval(Duration::from_secs(30));
    // Skip the first immediate tick
    ping_interval.tick().await;

    debug!(
        email = %claims.email,
        entity_filter = ?entity_filter,
        project_filter = ?project_filter,
        nats = nats_sub.is_some(),
        "WebSocket events client authenticated"
    );

    loop {
        tokio::select! {
            // Forward local broadcast events to the WebSocket client
            result = event_rx.recv() => {
                match result {
                    Ok(event) => {
                        if !passes_filters(&event, &entity_filter, &project_filter) {
                            continue;
                        }

                        // Dedup: mark this event as seen (local events are authoritative)
                        let fp = event_fingerprint(&event);
                        if !seen_set.insert(fp.clone()) {
                            // Already seen (came via NATS first) — skip
                            continue;
                        }
                        seen.push_back(fp.clone());
                        if seen.len() > DEDUP_WINDOW_SIZE {
                            if let Some(old) = seen.pop_front() {
                                seen_set.remove(&old);
                            }
                        }

                        // Serialize and send
                        match serde_json::to_string(&event) {
                            Ok(json) => {
                                if ws_sender.send(Message::Text(json.into())).await.is_err() {
                                    debug!("WebSocket send failed, client disconnected");
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to serialize CrudEvent: {}", e);
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "WebSocket client lagged, skipping events");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        debug!("Event bus closed, shutting down WebSocket");
                        break;
                    }
                }
            }

            // Forward NATS CRUD events (from other instances / MCP servers)
            nats_msg = async {
                match nats_sub.as_mut() {
                    Some(sub) => sub.next().await,
                    None => std::future::pending().await,
                }
            } => {
                if let Some(msg) = nats_msg {
                    match serde_json::from_slice::<CrudEvent>(&msg.payload) {
                        Ok(event) => {
                            if !passes_filters(&event, &entity_filter, &project_filter) {
                                continue;
                            }

                            // Dedup: skip if already seen via local broadcast
                            let fp = event_fingerprint(&event);
                            if !seen_set.insert(fp.clone()) {
                                continue;
                            }
                            seen.push_back(fp.clone());
                            if seen.len() > DEDUP_WINDOW_SIZE {
                                if let Some(old) = seen.pop_front() {
                                    seen_set.remove(&old);
                                }
                            }

                            match serde_json::to_string(&event) {
                                Ok(json) => {
                                    if ws_sender.send(Message::Text(json.into())).await.is_err() {
                                        debug!("WebSocket send failed, client disconnected");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to serialize NATS CrudEvent: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to deserialize NATS CrudEvent: {}", e);
                        }
                    }
                } else {
                    // NATS subscriber closed — disable NATS branch
                    debug!("NATS subscriber closed");
                    nats_sub = None;
                }
            }

            // Send periodic pings to detect dead clients
            _ = ping_interval.tick() => {
                if ws_sender.send(Message::Ping(vec![].into())).await.is_err() {
                    debug!("Ping failed, client disconnected");
                    break;
                }
            }

            // Handle incoming messages from the client (Pong, Close)
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Pong(_))) => {
                        // Client is alive
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        debug!("WebSocket client disconnected");
                        break;
                    }
                    Some(Err(e)) => {
                        debug!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {
                        // Ignore text/binary messages from clients
                    }
                }
            }
        }
    }

    debug!("WebSocket connection closed");
}
