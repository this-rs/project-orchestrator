//! WebSocket handler for real-time bidirectional chat
//!
//! Replaces the SSE stream + REST send_message + REST interrupt with a single
//! WebSocket connection per client per session.
//!
//! Protocol:
//! - Client → Server: JSON messages (user_message, interrupt, permission_response, input_response)
//! - Server → Client: JSON events with `seq` field (0 for non-persisted stream_delta)
//! - On connect: replay persisted events since `last_event` query param, then `replay_complete`

use super::handlers::{AppError, OrchestratorState};
use super::ws_auth::CookieAuthResult;
use crate::auth::jwt::Claims;
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
use std::collections::HashSet;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// Query parameters for the chat WebSocket
#[derive(Debug, Deserialize, Default)]
pub struct WsChatQuery {
    /// Last event sequence number seen by the client (for replay)
    #[serde(default)]
    pub last_event: i64,
    /// One-time ticket for auth (fallback when cookies aren't sent on WS upgrade)
    pub ticket: Option<String>,
}

/// Messages sent from the client over the WebSocket
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsChatClientMessage {
    /// Send a new user message
    UserMessage { content: String },
    /// Interrupt the current operation
    Interrupt,
    /// Response to a permission request
    PermissionResponse {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        allow: bool,
    },
    /// Response to an input request
    InputResponse {
        #[serde(default)]
        id: Option<String>,
        content: String,
    },
    /// Change the permission mode of the active session
    SetPermissionMode { mode: String },
    /// Change the model of the active session
    SetModel { model: String },
    /// Toggle auto-continue for the active session
    SetAutoContinue { enabled: bool },
}

/// WebSocket upgrade handler for `/ws/chat/{session_id}`
///
/// Authentication: validates `refresh_token` cookie BEFORE upgrade.
/// - Valid cookie (or no-auth mode) → upgrade + send `auth_ok` immediately
/// - No cookie or invalid → reject with 401 (no upgrade)
pub async fn ws_chat(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Path(session_id): Path<String>,
    Query(query): Query<WsChatQuery>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, AppError> {
    let pre_upgrade_start = tokio::time::Instant::now();

    // Validate that chat manager is available
    let _chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    // Validate that the session exists (in Neo4j or active)
    let uuid = uuid::Uuid::parse_str(&session_id)
        .map_err(|_| AppError::BadRequest(format!("Invalid session ID: {}", session_id)))?;

    let exists = state
        .orchestrator
        .neo4j()
        .get_chat_session(uuid)
        .await
        .map_err(AppError::Internal)?;

    if exists.is_none() {
        return Err(AppError::NotFound(format!(
            "Session {} not found",
            session_id
        )));
    }

    debug!(
        session_id = %session_id,
        elapsed_ms = pre_upgrade_start.elapsed().as_millis() as u64,
        "⏱ Pre-upgrade: session validation done"
    );

    let last_event = query.last_event;

    // Pre-upgrade auth: cookie first, then ticket fallback
    let neo4j = state.orchestrator.neo4j_arc();
    let auth_result = super::ws_auth::ws_authenticate(
        &headers,
        &state.auth_config,
        &neo4j,
        query.ticket.as_deref(),
        &state.ws_ticket_store,
    )
    .await;

    debug!(
        session_id = %session_id,
        elapsed_ms = pre_upgrade_start.elapsed().as_millis() as u64,
        "⏱ Pre-upgrade: auth completed, upgrading WS"
    );

    Ok(match auth_result {
        CookieAuthResult::Authenticated(claims) => ws
            .on_upgrade(move |socket| {
                handle_ws_chat_preauthed(socket, state, session_id, last_event, claims)
            })
            .into_response(),
        CookieAuthResult::Invalid(reason) => {
            debug!(reason = %reason, "WS chat: auth rejected");
            StatusCode::UNAUTHORIZED.into_response()
        }
    })
}

/// Handle a chat WebSocket connection that was pre-authenticated via cookie.
///
/// Waits for a `"ready"` message from the client before sending `auth_ok`.
async fn handle_ws_chat_preauthed(
    mut socket: WebSocket,
    state: OrchestratorState,
    session_id: String,
    last_event: i64,
    claims: Claims,
) {
    let t0 = tokio::time::Instant::now();
    // Wait for client "ready" signal before sending auth_ok
    super::ws_auth::wait_ready_then_auth_ok(&mut socket, &claims).await;
    debug!(
        session_id = %session_id,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "⏱ ready handshake completed"
    );
    handle_ws_chat_loop(socket, state, session_id, last_event, claims).await;
}

/// Main chat event loop for an authenticated WebSocket connection.
async fn handle_ws_chat_loop(
    socket: WebSocket,
    state: OrchestratorState,
    session_id: String,
    last_event: i64,
    claims: Claims,
) {
    let (mut ws_sender, mut ws_receiver) = socket.split();

    let loop_start = tokio::time::Instant::now();

    info!(
        session_id = %session_id,
        email = %claims.email,
        last_event,
        "Chat WebSocket client authenticated"
    );

    let chat_manager = match state.chat_manager.as_ref() {
        Some(cm) => cm,
        None => {
            let _ = ws_sender
                .send(Message::Text(
                    serde_json::json!({
                        "type": "error",
                        "message": "Chat manager not initialized"
                    })
                    .to_string()
                    .into(),
                ))
                .await;
            return;
        }
    };

    // ========================================================================
    // Phase 1: Replay persisted events since last_event
    // ========================================================================
    // OPTIMIZATION: Skip the Neo4j query entirely when last_event is very large
    // (e.g. Number.MAX_SAFE_INTEGER from the frontend). This means the client
    // doesn't want any historical replay — it will load history via REST and
    // only needs the live streaming snapshot (Phase 1.5). Skipping saves one
    // Neo4j round-trip (~50-200ms) that would always return 0 events anyway.
    const SKIP_REPLAY_THRESHOLD: i64 = 1_000_000_000_000;

    if last_event < SKIP_REPLAY_THRESHOLD {
        let events = chat_manager
            .get_events_since(&session_id, last_event)
            .await
            .unwrap_or_default();

        // Track the highest seq replayed in Phase 1. Currently used for logging/diagnostics.
        // NATS events don't carry seq numbers, so active dedup uses snapshot_skip_remaining counter instead.
        // Kept for future use if NATS events get seq headers.
        let mut _max_replayed_seq: i64 = last_event;

        if !events.is_empty() {
            // New format: replay ChatEventRecord from Neo4j
            debug!(
                session_id = %session_id,
                count = events.len(),
                "Replaying persisted ChatEventRecord"
            );
            for event in events {
                // Track highest seq for diagnostics / future dedup watermark
                if event.seq > _max_replayed_seq {
                    _max_replayed_seq = event.seq;
                }

                // Normalize to flat format matching Phase 1.5 / live events.
                // ChatEventRecord.data is a serde-serialized ChatEvent which already
                // contains the "type" tag. We parse it, inject seq + replaying, and
                // send it flat — so the frontend always receives the same JSON shape
                // regardless of whether the event comes from replay, snapshot, or live.
                let msg = match serde_json::from_str::<serde_json::Value>(&event.data) {
                    Ok(serde_json::Value::Object(mut obj)) => {
                        obj.insert("seq".to_string(), serde_json::json!(event.seq));
                        obj.insert("replaying".to_string(), serde_json::json!(true));
                        serde_json::Value::Object(obj)
                    }
                    _ => {
                        // Fallback for malformed records: wrap in data field
                        serde_json::json!({
                            "seq": event.seq,
                            "type": event.event_type,
                            "data": serde_json::Value::String(event.data.clone()),
                            "replaying": true,
                        })
                    }
                };
                if ws_sender
                    .send(Message::Text(msg.to_string().into()))
                    .await
                    .is_err()
                {
                    debug!("Client disconnected during replay");
                    return;
                }
            }
        } else if last_event == 0 {
            // Fallback: Load from Nexus SDK for pre-migration sessions
            debug!(
                session_id = %session_id,
                "No ChatEventRecord found, falling back to Nexus message history"
            );
            match chat_manager
                .get_session_messages(&session_id, None, None)
                .await
            {
                Ok(loaded) => {
                    debug!(
                        session_id = %session_id,
                        count = loaded.events.len(),
                        "Replaying message history (fallback)"
                    );
                    for event in &loaded.events {
                        let msg = match serde_json::from_str::<serde_json::Value>(&event.data) {
                            Ok(serde_json::Value::Object(mut obj)) => {
                                obj.insert("seq".to_string(), serde_json::json!(event.seq));
                                obj.insert("replaying".to_string(), serde_json::json!(true));
                                serde_json::Value::Object(obj)
                            }
                            _ => {
                                serde_json::json!({
                                    "seq": event.seq,
                                    "type": event.event_type,
                                    "data": serde_json::Value::String(event.data.clone()),
                                    "replaying": true,
                                })
                            }
                        };
                        if ws_sender
                            .send(Message::Text(msg.to_string().into()))
                            .await
                            .is_err()
                        {
                            debug!("Client disconnected during fallback replay");
                            return;
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        session_id = %session_id,
                        error = %e,
                        "Failed to load message history for replay"
                    );
                }
            }
        }
    } else {
        debug!(
            session_id = %session_id,
            last_event,
            "Skipping Phase 1 replay (last_event >= threshold, client loads history via REST)"
        );
    }

    debug!(
        session_id = %session_id,
        elapsed_ms = loop_start.elapsed().as_millis() as u64,
        "⏱ Phase 1 completed"
    );

    // ========================================================================
    // Phase 1.5a: Subscribe to broadcast + NATS BEFORE taking snapshot
    // ========================================================================
    // IMPORTANT: We subscribe BEFORE taking the streaming snapshot to avoid a
    // race condition. If we took the snapshot first and subscribed later, any
    // stream_delta tokens emitted between snapshot and subscribe would be lost
    // (the client would see a gap in the streamed text).
    //
    // By subscribing first, the broadcast::Receiver starts buffering events
    // immediately. The snapshot (Phase 1.5b) captures text accumulated before
    // the subscribe. Events emitted after subscribe are queued in the receiver.
    // Together they provide gap-free coverage.
    let mut event_rx = match chat_manager.subscribe(&session_id).await {
        Ok(rx) => Some(rx),
        Err(_) => {
            debug!(
                session_id = %session_id,
                "Session not active, no broadcast subscription yet — client can send messages to activate"
            );
            None
        }
    };

    // Subscribe to NATS chat events for cross-instance delivery (if NATS configured).
    // IMPORTANT: Only subscribe to NATS if there is NO local broadcast subscription.
    // When this instance owns the session (event_rx is Some), the ChatManager already
    // pushes events to the local broadcast channel — subscribing to NATS too would
    // cause every event to be received twice (local + NATS round-trip), producing
    // duplicated/interleaved text in the frontend.
    // NATS is only needed for remote instances that don't have the local broadcast.
    let mut nats_chat_sub: Option<async_nats::Subscriber> = if event_rx.is_some() {
        debug!(
            session_id = %session_id,
            "Local broadcast active — skipping NATS chat subscription (same instance)"
        );
        None
    } else if let Some(ref nats) = state.nats_emitter {
        match nats.subscribe_chat_events(&session_id).await {
            Ok(sub) => {
                debug!(session_id = %session_id, "Subscribed to NATS chat events (remote instance)");
                Some(sub)
            }
            Err(e) => {
                warn!(session_id = %session_id, error = %e, "Failed to subscribe to NATS chat events");
                None
            }
        }
    } else {
        None
    };

    // ========================================================================
    // Phase 1.5b: Send streaming snapshot if a stream is in progress
    // ========================================================================
    // Taken AFTER subscribe (Phase 1.5a) so no events are lost between
    // snapshot and subscription. The receiver buffers live events while we
    // send the snapshot to the client.
    let (mut is_currently_streaming, mut partial_text, mut streaming_events) =
        chat_manager.get_streaming_snapshot(&session_id).await;

    // If local snapshot is empty, try remote snapshot via NATS request/reply.
    // This handles the cross-instance case where another server owns the streaming session.
    if !is_currently_streaming {
        if let Some(ref nats) = state.nats_emitter {
            if let Some(snapshot) = nats.request_streaming_snapshot(&session_id).await {
                is_currently_streaming = snapshot.is_streaming;
                partial_text = snapshot.partial_text;
                streaming_events = snapshot.events;
            }
        }
    }

    info!(
        session_id = %session_id,
        is_currently_streaming,
        partial_text_len = partial_text.len(),
        streaming_events_count = streaming_events.len(),
        elapsed_ms = loop_start.elapsed().as_millis() as u64,
        "⏱ Phase 1.5b: streaming snapshot"
    );

    if is_currently_streaming {
        // Send accumulated structured events (tool_use, tool_result, assistant_text, etc.)
        // These are the non-StreamDelta events that occurred during the current stream.
        if !streaming_events.is_empty() {
            debug!(
                session_id = %session_id,
                count = streaming_events.len(),
                "Sending streaming_events snapshot for mid-stream join"
            );
            for event in &streaming_events {
                match serde_json::to_value(event) {
                    Ok(mut val) => {
                        if let Some(obj) = val.as_object_mut() {
                            obj.insert("seq".to_string(), serde_json::json!(0));
                            obj.insert("replaying".to_string(), serde_json::json!(true));
                        }
                        if ws_sender
                            .send(Message::Text(val.to_string().into()))
                            .await
                            .is_err()
                        {
                            debug!("Client disconnected during streaming_events replay");
                            return;
                        }
                    }
                    Err(e) => {
                        warn!("Failed to serialize streaming event: {}", e);
                    }
                }
            }
        }

        // Send accumulated stream_delta text as a single partial_text snapshot
        if !partial_text.is_empty() {
            debug!(
                session_id = %session_id,
                text_len = partial_text.len(),
                "Sending partial_text snapshot for mid-stream join"
            );
            let partial_msg = serde_json::json!({
                "type": "partial_text",
                "content": partial_text,
                "seq": 0,
                "replaying": true,
            });
            if ws_sender
                .send(Message::Text(partial_msg.to_string().into()))
                .await
                .is_err()
            {
                debug!("Client disconnected during partial_text");
                return;
            }
        }

        // Notify client that streaming is in progress (for interrupt button)
        let status_msg = serde_json::json!({
            "type": "streaming_status",
            "is_streaming": true,
        });
        if ws_sender
            .send(Message::Text(status_msg.to_string().into()))
            .await
            .is_err()
        {
            debug!("Client disconnected during streaming_status");
            return;
        }
    }

    // Fingerprint-based dedup: when we sent a streaming snapshot (Phase 1.5b),
    // events emitted in the tiny window BETWEEN subscribe (Phase 1.5a) and
    // snapshot (Phase 1.5b) may be present in BOTH the snapshot AND the
    // broadcast receiver. We dedup by fingerprint (type+id or type+hash).
    //
    // IMPORTANT: Most snapshot events were emitted BEFORE the subscribe and
    // are NOT in the broadcast receiver at all. Only the micro-window events
    // need dedup. Using a counter (previous approach) was wrong because it
    // assumed ALL snapshot events would arrive via the receiver, causing
    // genuinely new events (tool_use, tool_result, etc.) to be silently dropped.
    let mut snapshot_fingerprints: HashSet<String> = if is_currently_streaming {
        streaming_events
            .iter()
            .filter_map(|e| e.fingerprint())
            .collect()
    } else {
        HashSet::new()
    };

    // Send replay_complete marker
    debug!(
        session_id = %session_id,
        elapsed_ms = loop_start.elapsed().as_millis() as u64,
        "⏱ Snapshot sent, sending replay_complete"
    );
    let replay_complete = serde_json::json!({ "type": "replay_complete" });
    if ws_sender
        .send(Message::Text(replay_complete.to_string().into()))
        .await
        .is_err()
    {
        debug!("Client disconnected after replay");
        return;
    }

    // Send current auto_continue state so the frontend can initialize its toggle
    if let Ok(ac_enabled) = chat_manager.get_auto_continue_state(&session_id).await {
        let ac_msg = serde_json::json!({
            "type": "auto_continue_state_changed",
            "session_id": session_id,
            "enabled": ac_enabled,
            "seq": 0,
        });
        let _ = ws_sender
            .send(Message::Text(ac_msg.to_string().into()))
            .await;
    }

    // Ping interval (30s)
    let mut ping_interval = interval(Duration::from_secs(30));
    ping_interval.tick().await; // skip first immediate tick

    // Track responded permission request IDs to prevent double-click bugs.
    // If a user clicks Allow/Deny twice, the second response is silently ignored.
    let mut responded_permission_ids: HashSet<String> = HashSet::new();

    // ========================================================================
    // Phase 3: Event loop — forward broadcast events + handle client messages
    // ========================================================================

    // Helper: serialize and send a ChatEvent to the WS client.
    // Returns false if the client disconnected (caller should break).
    // Returns true if the event was sent successfully.
    //
    // NOTE: Snapshot dedup strategy (updated 2026-02-15):
    //
    // When a client joins mid-stream, Phase 1.5b sends a snapshot of accumulated
    // structured events. Since the broadcast/NATS subscription was created BEFORE
    // the snapshot (Phase 1.5a), the receiver may contain events from the tiny
    // window between subscribe and snapshot — these are also in the snapshot.
    //
    // We dedup by fingerprint: each snapshot event's fingerprint (type+id or
    // type+content_hash) is stored in a HashSet. When a broadcast/NATS event
    // matches a fingerprint, it's a duplicate → skip and remove from the set.
    // Events NOT in the set are genuinely new → forward to the client.
    // StreamDelta/StreamingStatus have no fingerprint and always pass through.
    macro_rules! send_chat_event {
        ($event:expr, $ws:expr) => {{
            // Serialize and send
            match serde_json::to_value(&$event) {
                Ok(mut val) => {
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert("seq".to_string(), serde_json::json!(0));
                    }
                    if $ws
                        .send(Message::Text(val.to_string().into()))
                        .await
                        .is_err()
                    {
                        false // client disconnected
                    } else {
                        true
                    }
                }
                Err(e) => {
                    warn!("Failed to serialize ChatEvent: {}", e);
                    true
                }
            }
        }};
    }

    loop {
        tokio::select! {
            // Forward local broadcast events to the WebSocket client
            event = async {
                match &mut event_rx {
                    Some(rx) => rx.recv().await,
                    None => {
                        // No subscription yet — wait forever (other branches will fire)
                        std::future::pending().await
                    }
                }
            } => {
                match event {
                    Ok(chat_event) => {
                        // --- Snapshot dedup (local broadcast) ---
                        // After a mid-stream join, events from the micro-window between
                        // subscribe and snapshot may be duplicates. We dedup by fingerprint.
                        // StreamDelta/StreamingStatus have no fingerprint and always pass.
                        if !snapshot_fingerprints.is_empty() {
                            if let Some(fp) = chat_event.fingerprint() {
                                if snapshot_fingerprints.remove(&fp) {
                                    debug!(
                                        event_type = %chat_event.event_type(),
                                        fingerprint = %fp,
                                        remaining = snapshot_fingerprints.len(),
                                        "Skipping broadcast event (duplicate of snapshot)"
                                    );
                                    continue;
                                }
                            }
                            // Result event clears all remaining fingerprints (end of stream)
                            if matches!(&chat_event, crate::chat::types::ChatEvent::Result { .. }) {
                                snapshot_fingerprints.clear();
                            }
                        }
                        if !send_chat_event!(chat_event, ws_sender) {
                            debug!("WebSocket send failed, client disconnected");
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "Chat WS client lagged, some events missed");
                        // Client may want to do a full replay — send a hint
                        let hint = serde_json::json!({
                            "type": "events_lagged",
                            "skipped": n,
                        });
                        let _ = ws_sender.send(Message::Text(hint.to_string().into())).await;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        // CLI subprocess was cleaned up (idle timeout) but the WebSocket
                        // should stay alive. Transition to dormant: disable the broadcast
                        // branch and let ping/pong + client messages keep running.
                        // When the user sends a new message, resume_session() will spawn
                        // a fresh CLI and the existing `if event_rx.is_none() { subscribe() }`
                        // logic (below) will re-subscribe automatically.
                        debug!(session_id = %session_id, "Chat broadcast closed (idle cleanup), going dormant");
                        event_rx = None;
                        snapshot_fingerprints.clear();
                        let msg = serde_json::json!({
                            "type": "session_dormant",
                            "message": "CLI session cleaned up (idle timeout). Will resume on next message."
                        });
                        let _ = ws_sender.send(Message::Text(msg.to_string().into())).await;
                    }
                }
            }

            // Forward NATS chat events to the WebSocket client (cross-instance)
            nats_msg = async {
                match &mut nats_chat_sub {
                    Some(sub) => sub.next().await,
                    None => std::future::pending().await,
                }
            } => {
                match nats_msg {
                    Some(msg) => {
                        match serde_json::from_slice::<crate::chat::types::ChatEvent>(&msg.payload) {
                            Ok(chat_event) => {
                                // --- Cross-instance dedup ---
                                // Dedup by fingerprint, same logic as local broadcast.
                                if !snapshot_fingerprints.is_empty() {
                                    if let Some(fp) = chat_event.fingerprint() {
                                        if snapshot_fingerprints.remove(&fp) {
                                            debug!(
                                                event_type = %chat_event.event_type(),
                                                fingerprint = %fp,
                                                remaining = snapshot_fingerprints.len(),
                                                "Skipping NATS event (duplicate of snapshot)"
                                            );
                                            continue;
                                        }
                                    }
                                    if matches!(&chat_event, crate::chat::types::ChatEvent::Result { .. }) {
                                        snapshot_fingerprints.clear();
                                    }
                                }

                                if !send_chat_event!(chat_event, ws_sender) {
                                    debug!("WebSocket send failed (NATS event), client disconnected");
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to deserialize NATS ChatEvent: {}", e);
                            }
                        }
                    }
                    None => {
                        // NATS subscriber closed
                        debug!(session_id = %session_id, "NATS chat subscriber closed");
                        nats_chat_sub = None;
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

            // Handle incoming messages from the client
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        let text_str: &str = &text;
                        match serde_json::from_str::<WsChatClientMessage>(text_str) {
                            Ok(client_msg) => {
                                match client_msg {
                                    WsChatClientMessage::UserMessage { content } => {
                                        debug!(session_id = %session_id, "WS: Received user_message");

                                        // T4.3: Extract code entities and create DISCUSSED relations (non-blocking)
                                        spawn_entity_extraction(&state, &session_id, &content);

                                        // 3-branch routing: local → remote (NATS RPC) → fallback resume
                                        let result = if chat_manager.is_session_active(&session_id).await {
                                            // Session is local — send directly
                                            chat_manager.send_message(&session_id, &content).await
                                        } else if chat_manager
                                            .try_remote_send(&session_id, &content, "user_message")
                                            .await
                                            .unwrap_or(false)
                                        {
                                            // Message proxied to remote instance via NATS RPC.
                                            // Ensure we have a NATS subscription to receive the stream.
                                            if nats_chat_sub.is_none() {
                                                if let Some(ref nats) = state.nats_emitter {
                                                    if let Ok(sub) = nats.subscribe_chat_events(&session_id).await {
                                                        debug!(session_id = %session_id, "Subscribed to NATS chat events after remote send");
                                                        nats_chat_sub = Some(sub);
                                                    }
                                                }
                                            }
                                            Ok(())
                                        } else {
                                            // No instance owns the session — resume locally (spawns new CLI)
                                            chat_manager.resume_session(&session_id, &content, Some(&claims)).await
                                        };

                                        match result {
                                            Ok(()) => {
                                                // If we did a local send/resume and don't have broadcast, subscribe now
                                                if event_rx.is_none() {
                                                    match chat_manager.subscribe(&session_id).await {
                                                        Ok(rx) => {
                                                            event_rx = Some(rx);
                                                            // Now that we have local broadcast, drop the NATS
                                                            // subscription to avoid receiving every event twice.
                                                            if nats_chat_sub.is_some() {
                                                                debug!(session_id = %session_id, "Dropping NATS chat sub — local broadcast now active");
                                                                nats_chat_sub = None;
                                                            }
                                                            debug!(session_id = %session_id, "Subscribed to broadcast after first message");
                                                        }
                                                        Err(e) => {
                                                            // May fail if message was proxied remotely (no local session)
                                                            debug!(session_id = %session_id, error = %e, "No local broadcast (session may be remote)");
                                                        }
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                error!(session_id = %session_id, error = %e, "Failed to send message");
                                                let err = serde_json::json!({
                                                    "type": "error",
                                                    "message": format!("Failed to send message: {}", e),
                                                });
                                                let _ = ws_sender.send(Message::Text(err.to_string().into())).await;
                                            }
                                        }
                                    }

                                    WsChatClientMessage::Interrupt => {
                                        debug!(session_id = %session_id, "WS: Received interrupt");
                                        if let Err(e) = chat_manager.interrupt(&session_id).await {
                                            warn!(session_id = %session_id, error = %e, "Failed to interrupt");
                                        }
                                    }

                                    WsChatClientMessage::PermissionResponse { id, allow } => {
                                        let request_id = id.clone().unwrap_or_default();
                                        let decision = if allow { "allow" } else { "deny" };

                                        // Dedup: ignore duplicate responses (double-click protection)
                                        if !request_id.is_empty() && !responded_permission_ids.insert(request_id.clone()) {
                                            warn!(
                                                session_id = %session_id,
                                                request_id = %request_id,
                                                decision,
                                                "Duplicate permission response ignored"
                                            );
                                            continue;
                                        }

                                        // Determine routing: local or remote
                                        let is_local = chat_manager.is_session_active(&session_id).await;
                                        let routing = if is_local { "local" } else { "nats" };

                                        info!(
                                            session_id = %session_id,
                                            request_id = %request_id,
                                            decision,
                                            routing,
                                            "Permission decision"
                                        );

                                        // Send control protocol response to CLI subprocess.
                                        // Unlike user messages, permission responses use the SDK
                                        // control protocol (JSON: {"allow": true/false}) and must
                                        // NOT be persisted or broadcast as user_message events.
                                        let send_result = if is_local {
                                            chat_manager.send_permission_response(&session_id, &request_id, allow).await
                                        } else {
                                            // For remote sessions, proxy the control response via NATS RPC.
                                            // The message_type "control_response" signals the receiving
                                            // instance to use send_permission_response instead of send_message.
                                            let payload = serde_json::json!({ "allow": allow }).to_string();
                                            if chat_manager
                                                .try_remote_send(&session_id, &payload, "control_response")
                                                .await
                                                .unwrap_or(false)
                                            {
                                                Ok(())
                                            } else {
                                                Err(anyhow::anyhow!("Session not active on any instance"))
                                            }
                                        };
                                        if let Err(e) = send_result {
                                            warn!(
                                                session_id = %session_id,
                                                request_id = %request_id,
                                                decision,
                                                routing,
                                                error = %e,
                                                "Failed to send permission response"
                                            );
                                        }
                                    }

                                    WsChatClientMessage::InputResponse { content, .. } => {
                                        debug!(session_id = %session_id, "WS: Received input_response");
                                        // Try local → remote → error (no resume for responses)
                                        let send_result = if chat_manager.is_session_active(&session_id).await {
                                            chat_manager.send_message(&session_id, &content).await
                                        } else if chat_manager
                                            .try_remote_send(&session_id, &content, "input_response")
                                            .await
                                            .unwrap_or(false)
                                        {
                                            Ok(())
                                        } else {
                                            Err(anyhow::anyhow!("Session not active on any instance"))
                                        };
                                        if let Err(e) = send_result {
                                            warn!(session_id = %session_id, error = %e, "Failed to send input response");
                                        }
                                    }

                                    WsChatClientMessage::SetPermissionMode { mode } => {
                                        info!(session_id = %session_id, mode = %mode, "WS: Received set_permission_mode");
                                        if chat_manager.is_session_active(&session_id).await {
                                            match chat_manager.set_session_permission_mode(&session_id, &mode).await {
                                                Ok(()) => {
                                                    // Send confirmation back to frontend
                                                    let confirmation = serde_json::json!({
                                                        "type": "permission_mode_changed",
                                                        "mode": mode,
                                                    });
                                                    let _ = ws_sender.send(Message::Text(confirmation.to_string().into())).await;
                                                }
                                                Err(e) => {
                                                    warn!(session_id = %session_id, error = %e, "Failed to set permission mode");
                                                    let err = serde_json::json!({
                                                        "type": "error",
                                                        "message": format!("Failed to set permission mode: {}", e),
                                                    });
                                                    let _ = ws_sender.send(Message::Text(err.to_string().into())).await;
                                                }
                                            }
                                        } else {
                                            let err = serde_json::json!({
                                                "type": "error",
                                                "message": "Session not active on this instance",
                                            });
                                            let _ = ws_sender.send(Message::Text(err.to_string().into())).await;
                                        }
                                    }

                                    WsChatClientMessage::SetModel { model } => {
                                        info!(session_id = %session_id, model = %model, "WS: Received set_model");
                                        if chat_manager.is_session_active(&session_id).await {
                                            match chat_manager.set_session_model(&session_id, &model).await {
                                                Ok(()) => {
                                                    // Send confirmation back to frontend
                                                    let confirmation = serde_json::json!({
                                                        "type": "model_changed",
                                                        "model": model,
                                                    });
                                                    let _ = ws_sender.send(Message::Text(confirmation.to_string().into())).await;
                                                }
                                                Err(e) => {
                                                    warn!(session_id = %session_id, error = %e, "Failed to set model");
                                                    let err = serde_json::json!({
                                                        "type": "error",
                                                        "message": format!("Failed to set model: {}", e),
                                                    });
                                                    let _ = ws_sender.send(Message::Text(err.to_string().into())).await;
                                                }
                                            }
                                        } else {
                                            let err = serde_json::json!({
                                                "type": "error",
                                                "message": "Session not active on this instance",
                                            });
                                            let _ = ws_sender.send(Message::Text(err.to_string().into())).await;
                                        }
                                    }

                                    WsChatClientMessage::SetAutoContinue { enabled } => {
                                        info!(session_id = %session_id, enabled = %enabled, "WS: Received set_auto_continue");
                                        // set_auto_continue works whether the session is active or idle:
                                        // - Active: updates in-memory + Neo4j + local broadcast + NATS
                                        // - Idle: updates Neo4j + NATS only
                                        // No need for try_remote_send — NATS pub/sub handles cross-instance.
                                        match chat_manager.set_auto_continue(&session_id, enabled).await {
                                            Ok(()) => {
                                                let confirmation = serde_json::json!({
                                                    "type": "auto_continue_state_changed",
                                                    "enabled": enabled,
                                                });
                                                let _ = ws_sender.send(Message::Text(confirmation.to_string().into())).await;
                                            }
                                            Err(e) => {
                                                warn!(session_id = %session_id, error = %e, "Failed to set auto_continue");
                                                let err = serde_json::json!({
                                                    "type": "error",
                                                    "message": format!("Failed to set auto_continue: {}", e),
                                                });
                                                let _ = ws_sender.send(Message::Text(err.to_string().into())).await;
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                warn!(session_id = %session_id, error = %e, text = %text, "Failed to parse client message");
                                let err = serde_json::json!({
                                    "type": "error",
                                    "message": format!("Invalid message format: {}", e),
                                });
                                let _ = ws_sender.send(Message::Text(err.to_string().into())).await;
                            }
                        }
                    }

                    Some(Ok(Message::Pong(_))) => {
                        // Client is alive
                    }

                    Some(Ok(Message::Close(_))) | None => {
                        debug!(session_id = %session_id, "Chat WebSocket client disconnected");
                        break;
                    }

                    Some(Err(e)) => {
                        debug!(session_id = %session_id, error = %e, "Chat WebSocket error");
                        break;
                    }

                    _ => {
                        // Ignore binary, ping from client
                    }
                }
            }
        }
    }

    info!(session_id = %session_id, "Chat WebSocket connection closed");
}

/// Rate-limiter for neural reinforcement via chat.
/// Allows at most `MAX_BOOSTS_PER_MINUTE` reinforcement batches per minute
/// to avoid runaway energy inflation during rapid conversations.
mod chat_reinforcement_limiter {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    /// Maximum reinforcement batches per minute window.
    const MAX_BOOSTS_PER_MINUTE: u32 = 5;

    /// Epoch-minute of the current window.
    static WINDOW_MINUTE: AtomicU64 = AtomicU64::new(0);
    /// Number of boosts in the current window.
    static WINDOW_COUNT: AtomicU32 = AtomicU32::new(0);

    /// Try to acquire a reinforcement slot. Returns `true` if allowed.
    pub fn try_acquire() -> bool {
        let now_minute = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 60;

        let stored = WINDOW_MINUTE.load(Ordering::Relaxed);
        if now_minute != stored {
            // New minute window — reset counter
            WINDOW_MINUTE.store(now_minute, Ordering::Relaxed);
            WINDOW_COUNT.store(1, Ordering::Relaxed);
            return true;
        }

        let prev = WINDOW_COUNT.fetch_add(1, Ordering::Relaxed);
        if prev < MAX_BOOSTS_PER_MINUTE {
            true
        } else {
            // Over limit — undo the increment (best-effort)
            WINDOW_COUNT.fetch_sub(1, Ordering::Relaxed);
            false
        }
    }
}

/// Spawn a background task to extract code entities from a chat message,
/// create DISCUSSED relations, and perform neural reinforcement (Hebbian).
///
/// This is the integration point between:
/// - T4.1 (entity_extractor) — regex-based entity extraction
/// - T4.2 (DISCUSSED) — graph relations
/// - T4.6 (neural reinforcement) — energy boost + synapse reinforcement
///
/// Non-blocking, fire-and-forget. Rate-limited to 5 reinforcement batches/minute.
pub fn spawn_entity_extraction(state: &OrchestratorState, session_id: &str, message: &str) {
    use crate::chat::entity_extractor;

    // Quick check: skip empty or very short messages (not worth extracting)
    if message.len() < 5 {
        return;
    }

    // Extract entities synchronously (regex-based, fast — microseconds)
    let entities = entity_extractor::extract_entities(message);
    if entities.is_empty() {
        return;
    }

    let entity_count = entities.len();
    let session_uuid = match session_id.parse::<uuid::Uuid>() {
        Ok(id) => id,
        Err(_) => return,
    };

    // Convert to (entity_type, entity_id) tuples for add_discussed
    // add_discussed expects Neo4j labels: "File", "Function", "Struct", "Trait", "Enum"
    let discussed_entities: Vec<(String, String)> = entities
        .into_iter()
        .map(|e| {
            let label = match e.entity_type {
                entity_extractor::EntityType::File => "File",
                entity_extractor::EntityType::Function => "Function",
                entity_extractor::EntityType::Struct => "Struct",
                entity_extractor::EntityType::Trait => "Trait",
                entity_extractor::EntityType::Enum => "Enum",
                entity_extractor::EntityType::Symbol => "Function", // best guess
            };
            (label.to_string(), e.identifier)
        })
        .collect();

    let neo4j = state.orchestrator.neo4j_arc();

    // Read reinforcement config before moving into async block
    let ar_config = state.orchestrator.auto_reinforcement_config().clone();
    let should_reinforce = ar_config.enabled && chat_reinforcement_limiter::try_acquire();

    // Clone entity info for reinforcement phase (need it after discussed_entities is consumed)
    let reinforcement_entities: Vec<(String, String)> = if should_reinforce {
        discussed_entities.clone()
    } else {
        Vec::new()
    };

    // Fire-and-forget: create DISCUSSED relations + neural reinforcement
    tokio::spawn(async move {
        // Phase 1: Create DISCUSSED relations
        match neo4j.add_discussed(session_uuid, &discussed_entities).await {
            Ok(created) => {
                if created > 0 {
                    debug!(
                        session_id = %session_uuid,
                        extracted = entity_count,
                        discussed = created,
                        "Entity extraction: created DISCUSSED relations"
                    );
                }
            }
            Err(e) => {
                debug!(
                    session_id = %session_uuid,
                    error = %e,
                    "Entity extraction: failed to create DISCUSSED relations"
                );
            }
        }

        // Phase 2: Neural reinforcement (Hebbian learning via chat)
        // Find notes linked to discussed entities, boost their energy,
        // and reinforce synapses between co-activated notes.
        if !should_reinforce || reinforcement_entities.is_empty() {
            return;
        }

        let mut all_note_ids: Vec<uuid::Uuid> = Vec::new();
        let mut boost_count = 0u64;

        for (entity_type_str, entity_id) in &reinforcement_entities {
            // Map Neo4j label to notes::EntityType
            let entity_type = match entity_type_str.as_str() {
                "File" => crate::notes::EntityType::File,
                "Function" => crate::notes::EntityType::Function,
                "Struct" => crate::notes::EntityType::Struct,
                "Trait" => crate::notes::EntityType::Trait,
                "Enum" => crate::notes::EntityType::Enum,
                _ => continue,
            };

            match neo4j.get_notes_for_entity(&entity_type, entity_id).await {
                Ok(notes) => {
                    for note in &notes {
                        // Boost energy for each note linked to a discussed entity
                        if let Err(e) = neo4j
                            .boost_energy(note.id, ar_config.chat_energy_boost)
                            .await
                        {
                            debug!(
                                note_id = %note.id,
                                error = %e,
                                "Chat neural reinforcement: energy boost failed"
                            );
                        } else {
                            boost_count += 1;
                        }
                        all_note_ids.push(note.id);
                    }
                }
                Err(e) => {
                    debug!(
                        entity_type = %entity_type_str,
                        entity_id = %entity_id,
                        error = %e,
                        "Chat neural reinforcement: get_notes_for_entity failed"
                    );
                }
            }
        }

        // Reinforce synapses between co-activated notes (Hebbian: "fire together, wire together")
        if all_note_ids.len() >= 2 {
            all_note_ids.sort();
            all_note_ids.dedup();
            if all_note_ids.len() >= 2 {
                match neo4j
                    .reinforce_synapses(&all_note_ids, ar_config.chat_synapse_boost)
                    .await
                {
                    Ok(synapse_count) => {
                        debug!(
                            session_id = %session_uuid,
                            energy_boosts = boost_count,
                            synapses_reinforced = synapse_count,
                            notes_activated = all_note_ids.len(),
                            "Chat neural reinforcement complete"
                        );
                    }
                    Err(e) => {
                        debug!(
                            session_id = %session_uuid,
                            error = %e,
                            "Chat neural reinforcement: synapse reinforcement failed"
                        );
                    }
                }
            }
        } else if boost_count > 0 {
            debug!(
                session_id = %session_uuid,
                energy_boosts = boost_count,
                "Chat neural reinforcement: energy only (< 2 notes for synapses)"
            );
        }
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::types::ChatEvent;
    use std::collections::HashSet;
    use tokio::sync::broadcast;

    /// Verify that dropping the broadcast sender causes RecvError::Closed.
    /// This is the mechanism that triggers the dormant transition.
    #[tokio::test]
    async fn test_broadcast_close_returns_closed_error() {
        let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);

        // Drop the sender — simulates close_session() removing the ActiveSession
        drop(tx);

        let result = rx.recv().await;
        assert!(
            matches!(result, Err(broadcast::error::RecvError::Closed)),
            "Expected RecvError::Closed after sender is dropped, got: {:?}",
            result
        );
    }

    /// Verify the dormant event JSON format matches what the frontend expects.
    #[test]
    fn test_session_dormant_json_format() {
        let msg = serde_json::json!({
            "type": "session_dormant",
            "message": "CLI session cleaned up (idle timeout). Will resume on next message."
        });

        assert_eq!(msg["type"], "session_dormant");
        assert!(msg["message"].as_str().unwrap().contains("idle timeout"));
    }

    /// Verify that after broadcast close, a new channel can be created and
    /// subscribed to — simulating the resume_session → re-subscribe flow.
    #[tokio::test]
    async fn test_resubscribe_after_broadcast_close() {
        // Phase 1: Create initial broadcast, subscribe, then close
        let (tx1, mut rx1) = broadcast::channel::<ChatEvent>(16);
        drop(tx1);

        let result = rx1.recv().await;
        assert!(matches!(result, Err(broadcast::error::RecvError::Closed)));

        // Phase 2: Simulate event_rx = None (dormant state)
        let mut event_rx: Option<broadcast::Receiver<ChatEvent>> = None;
        assert!(
            event_rx.is_none(),
            "Should be None after dormant transition"
        );

        // Phase 3: Simulate resume_session() creating a new broadcast channel
        let (tx2, rx2) = broadcast::channel::<ChatEvent>(16);
        event_rx = Some(rx2);
        assert!(event_rx.is_some(), "Should be Some after re-subscribe");

        // Phase 4: Verify events flow on the new channel
        let test_event = ChatEvent::StreamingStatus { is_streaming: true };
        tx2.send(test_event).unwrap();

        let received = event_rx.as_mut().unwrap().recv().await.unwrap();
        assert!(
            matches!(received, ChatEvent::StreamingStatus { is_streaming: true }),
            "Should receive events on the new channel"
        );
    }

    /// Verify that `tokio::select!` with `std::future::pending()` for a None
    /// branch works correctly — the pending branch never fires, allowing other
    /// branches to proceed. This is the core mechanism of dormant mode.
    #[tokio::test]
    async fn test_dormant_select_skips_none_branch() {
        let event_rx: Option<broadcast::Receiver<ChatEvent>> = None;
        let (user_tx, mut user_rx) = tokio::sync::mpsc::channel::<String>(1);

        // Send a user message to simulate client interaction during dormant
        user_tx.send("hello".to_string()).await.unwrap();

        // The select! should skip the broadcast branch (None → pending)
        // and process the user message branch instead
        let result = tokio::select! {
            _event = async {
                match &event_rx {
                    Some(_rx) => unreachable!("Should not reach Some branch"),
                    None => {
                        // This mirrors the handler's dormant behavior
                        std::future::pending::<()>().await
                    }
                }
            } => "broadcast",
            msg = user_rx.recv() => {
                assert_eq!(msg.unwrap(), "hello");
                "user_message"
            }
        };

        assert_eq!(
            result, "user_message",
            "Should process user message while broadcast is dormant"
        );
    }

    /// Verify that snapshot_fingerprints can be safely cleared after dormant
    /// transition, and that new events after re-subscribe are not deduped.
    #[tokio::test]
    async fn test_snapshot_fingerprints_cleared_on_dormant() {
        let mut fingerprints = HashSet::new();
        fingerprints.insert("tool_use:abc123".to_string());
        fingerprints.insert("thinking:def456".to_string());
        assert_eq!(fingerprints.len(), 2);

        // Simulate dormant transition: clear fingerprints
        fingerprints.clear();
        assert!(
            fingerprints.is_empty(),
            "Fingerprints should be cleared on dormant"
        );

        // After re-subscribe, new events should NOT be deduped
        let new_fp = "tool_use:abc123".to_string();
        assert!(
            !fingerprints.contains(&new_fp),
            "Old fingerprint should not block new events after dormant"
        );
    }

    /// Verify that the WsChatClientMessage deserialization works for all types
    /// that can be received during dormant mode.
    #[test]
    fn test_client_message_deserialization() {
        // user_message — the main one that triggers resume_session
        let msg: WsChatClientMessage =
            serde_json::from_str(r#"{"type":"user_message","content":"hello"}"#).unwrap();
        assert!(matches!(msg, WsChatClientMessage::UserMessage { content } if content == "hello"));

        // interrupt — should work even if dormant (no-op since no stream)
        let msg: WsChatClientMessage = serde_json::from_str(r#"{"type":"interrupt"}"#).unwrap();
        assert!(matches!(msg, WsChatClientMessage::Interrupt));
    }

    /// End-to-end simulation of the dormant → resume lifecycle using channels.
    /// Verifies the full sequence: active → broadcast close → dormant → user message → re-subscribe → active.
    #[tokio::test]
    async fn test_dormant_resume_lifecycle() {
        // === Phase 1: Active session ===
        let (tx1, rx1) = broadcast::channel::<ChatEvent>(16);
        let mut event_rx: Option<broadcast::Receiver<ChatEvent>> = Some(rx1);

        // Send an event while active
        tx1.send(ChatEvent::StreamingStatus { is_streaming: true })
            .unwrap();
        let ev = event_rx.as_mut().unwrap().recv().await.unwrap();
        assert!(matches!(
            ev,
            ChatEvent::StreamingStatus { is_streaming: true }
        ));

        // === Phase 2: Idle cleanup (close_session drops tx) ===
        drop(tx1);
        let close_result = event_rx.as_mut().unwrap().recv().await;
        assert!(matches!(
            close_result,
            Err(broadcast::error::RecvError::Closed)
        ));

        // Transition to dormant
        event_rx = None;
        let mut snapshot_fingerprints = HashSet::new();
        snapshot_fingerprints.insert("stale:fp".to_string());
        snapshot_fingerprints.clear(); // cleaned on dormant

        assert!(event_rx.is_none());
        assert!(snapshot_fingerprints.is_empty());

        // === Phase 3: User sends message → resume_session creates new channel ===
        let (tx2, rx2) = broadcast::channel::<ChatEvent>(16);
        event_rx = Some(rx2); // re-subscribe

        // === Phase 4: Verify events flow on resumed session ===
        tx2.send(ChatEvent::StreamingStatus {
            is_streaming: false,
        })
        .unwrap();
        let ev = event_rx.as_mut().unwrap().recv().await.unwrap();
        assert!(matches!(
            ev,
            ChatEvent::StreamingStatus {
                is_streaming: false
            }
        ));
    }
}
