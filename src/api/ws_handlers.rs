//! WebSocket handlers for real-time CRUD event notifications
//!
//! Events from both local and remote sources (NATS) arrive via the local
//! broadcast bus thanks to the NATS→local bridge in `HybridEmitter`.
//! WS handlers only need to subscribe to the local bus.

use super::handlers::OrchestratorState;
use super::ws_auth::CookieAuthResult;
use crate::auth::jwt::Claims;
use crate::events::CrudEvent;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use std::collections::HashSet;
use tokio::time::{interval, Duration};
use tracing::{debug, warn};

/// Query parameters for filtering WebSocket events
#[derive(Debug, Deserialize, Default)]
pub struct WsQuery {
    /// Comma-separated entity types to subscribe to (e.g. "plan,task,note")
    pub entity_types: Option<String>,
    /// Filter by project ID
    pub project_id: Option<String>,
}

/// WebSocket upgrade handler for `/ws/events`
///
/// Authentication strategy (ordered by priority):
/// 1. **Cookie auth** (browser clients) — validate `refresh_token` cookie BEFORE upgrade
///    - Valid → upgrade + send `auth_ok` immediately
///    - Invalid → reject with 401 (no upgrade)
/// 2. **First-message handshake** (MCP/CLI clients) — no cookie present
///    - Upgrade first, then wait for `{ "type": "auth", "token": "<jwt>" }`
pub async fn ws_events(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Query(query): Query<WsQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let entity_filter: Option<HashSet<String>> = query.entity_types.map(|types| {
        types
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    });
    let project_filter = query.project_id;

    // Pre-upgrade cookie auth
    let neo4j = state.orchestrator.neo4j_arc();
    let cookie_result =
        super::ws_auth::ws_authenticate_from_cookie(&headers, &state.auth_config, &neo4j).await;

    match cookie_result {
        CookieAuthResult::Authenticated(claims) => {
            // Cookie valid → upgrade with pre-authenticated claims
            ws.on_upgrade(move |socket| {
                handle_ws_preauthed(socket, state, entity_filter, project_filter, claims)
            })
            .into_response()
        }
        CookieAuthResult::NoCookie => {
            // No cookie → upgrade and fall back to first-message handshake
            ws.on_upgrade(move |socket| {
                handle_ws_message_auth(socket, state, entity_filter, project_filter)
            })
            .into_response()
        }
        CookieAuthResult::Invalid(reason) => {
            // Cookie present but invalid → reject before upgrade
            debug!(reason = %reason, "WS events: cookie auth rejected");
            StatusCode::UNAUTHORIZED.into_response()
        }
    }
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

/// Handle a WebSocket connection that was pre-authenticated via cookie.
///
/// Sends `auth_ok` immediately, then enters the event loop.
async fn handle_ws_preauthed(
    mut socket: WebSocket,
    state: OrchestratorState,
    entity_filter: Option<HashSet<String>>,
    project_filter: Option<String>,
    claims: Claims,
) {
    // Send auth_ok immediately (client doesn't need to send auth message)
    super::ws_auth::send_auth_ok(&mut socket, &claims).await;
    handle_ws_loop(socket, state, entity_filter, project_filter, claims).await;
}

/// Handle a WebSocket connection that needs first-message authentication (fallback).
///
/// Waits for `{ "type": "auth", "token": "<jwt>" }`, validates, then enters the event loop.
async fn handle_ws_message_auth(
    mut socket: WebSocket,
    state: OrchestratorState,
    entity_filter: Option<HashSet<String>>,
    project_filter: Option<String>,
) {
    let claims = match super::ws_auth::ws_authenticate(&mut socket, &state.auth_config).await {
        Ok(claims) => claims,
        Err(reason) => {
            debug!(reason = %reason, "WS events: message auth failed");
            return;
        }
    };
    handle_ws_loop(socket, state, entity_filter, project_filter, claims).await;
}

/// Main event loop for an authenticated WebSocket connection.
///
/// All CRUD events (local + remote via NATS bridge) arrive through the
/// local broadcast bus — no per-connection NATS subscription needed.
async fn handle_ws_loop(
    socket: WebSocket,
    state: OrchestratorState,
    entity_filter: Option<HashSet<String>>,
    project_filter: Option<String>,
    claims: Claims,
) {
    let (mut ws_sender, mut ws_receiver) = socket.split();
    let mut event_rx = state.event_bus.subscribe();

    // Ping interval (30s)
    let mut ping_interval = interval(Duration::from_secs(30));
    // Skip the first immediate tick
    ping_interval.tick().await;

    debug!(
        email = %claims.email,
        entity_filter = ?entity_filter,
        project_filter = ?project_filter,
        "WebSocket events client authenticated"
    );

    loop {
        tokio::select! {
            // Forward broadcast events (local + NATS-bridged) to the WebSocket client
            result = event_rx.recv() => {
                match result {
                    Ok(event) => {
                        if !passes_filters(&event, &entity_filter, &project_filter) {
                            continue;
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
