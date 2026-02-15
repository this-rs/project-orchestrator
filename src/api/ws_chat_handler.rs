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
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Path, Query, State, WebSocketUpgrade,
    },
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
}

/// WebSocket upgrade handler for `/ws/chat/{session_id}`
pub async fn ws_chat(
    ws: WebSocketUpgrade,
    State(state): State<OrchestratorState>,
    Path(session_id): Path<String>,
    Query(query): Query<WsChatQuery>,
) -> Result<impl IntoResponse, AppError> {
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

    let last_event = query.last_event;

    Ok(ws.on_upgrade(move |socket| handle_ws_chat(socket, state, session_id, last_event)))
}

/// Handle an individual chat WebSocket connection
async fn handle_ws_chat(
    mut socket: WebSocket,
    state: OrchestratorState,
    session_id: String,
    last_event: i64,
) {
    // ========================================================================
    // Phase 0: Authenticate via first message
    // ========================================================================
    let claims = match super::ws_auth::ws_authenticate(&mut socket, &state.auth_config).await {
        Ok(claims) => claims,
        Err(reason) => {
            debug!(session_id = %session_id, reason = %reason, "WS chat: auth failed");
            return;
        }
    };

    let (mut ws_sender, mut ws_receiver) = socket.split();

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
    let events = chat_manager
        .get_events_since(&session_id, last_event)
        .await
        .unwrap_or_default();

    // Track the highest seq replayed in Phase 1. Currently used for logging/diagnostics.
    // NATS events don't carry seq numbers, so active dedup uses snapshot_dedup_active flag instead.
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
        "Phase 1.5b: streaming snapshot"
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

    // Dedup flag: if we sent a streaming snapshot, structured events that arrive
    // via the broadcast receiver or NATS are duplicates of what we just sent.
    // We skip them until the current stream ends (Result event clears this flag).
    // StreamDelta events are NEVER skipped — they're not in the snapshot and are
    // the real-time text tokens the client needs.
    let mut snapshot_dedup_active = is_currently_streaming && !streaming_events.is_empty();

    // Send replay_complete marker
    let replay_complete = serde_json::json!({ "type": "replay_complete" });
    if ws_sender
        .send(Message::Text(replay_complete.to_string().into()))
        .await
        .is_err()
    {
        debug!("Client disconnected after replay");
        return;
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
    // the snapshot (Phase 1.5a), the receiver may also contain those same structured
    // events. Without dedup, they would appear twice in the frontend.
    //
    // The `snapshot_dedup_active` flag filters structured events (tool_use,
    // tool_result, assistant_text, etc.) from BOTH the local broadcast and NATS
    // branches until a Result event arrives (stream ends).
    // StreamDelta tokens always pass through — they're never in the snapshot.
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
                        // After a mid-stream join, the broadcast receiver may deliver
                        // structured events that were already sent in the Phase 1.5b
                        // snapshot. Skip them to avoid duplicates in the frontend.
                        // StreamDelta and StreamingStatus always pass through (not in snapshot).
                        if snapshot_dedup_active {
                            match &chat_event {
                                crate::chat::types::ChatEvent::StreamDelta { .. }
                                | crate::chat::types::ChatEvent::StreamingStatus { .. } => {}
                                crate::chat::types::ChatEvent::Result { .. } => {
                                    snapshot_dedup_active = false;
                                }
                                _ => {
                                    debug!(
                                        event_type = %chat_event.event_type(),
                                        "Skipping broadcast event (already sent in snapshot)"
                                    );
                                    continue;
                                }
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
                        debug!("Chat broadcast closed for session {}", session_id);
                        // Channel closed — session was cleaned up. Notify client.
                        let msg = serde_json::json!({
                            "type": "session_closed",
                            "message": "Session has been closed"
                        });
                        let _ = ws_sender.send(Message::Text(msg.to_string().into())).await;
                        break;
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
                                // 1. Snapshot dedup: if we sent a Phase 1.5 snapshot, structured
                                //    events from the same stream arrive again via NATS — skip them.
                                //    StreamDelta and StreamingStatus are NEVER in the snapshot, so
                                //    they always pass through (no content-based dedup needed).
                                if snapshot_dedup_active {
                                    match &chat_event {
                                        // Always forward stream tokens — these are real-time
                                        crate::chat::types::ChatEvent::StreamDelta { .. }
                                        | crate::chat::types::ChatEvent::StreamingStatus { .. } => {}
                                        // Result marks end of stream — forward it and clear dedup
                                        crate::chat::types::ChatEvent::Result { .. } => {
                                            snapshot_dedup_active = false;
                                        }
                                        // All other structured events were already in the snapshot
                                        _ => {
                                            debug!(
                                                event_type = %chat_event.event_type(),
                                                "Skipping NATS event (already sent in snapshot)"
                                            );
                                            continue;
                                        }
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
                                            chat_manager.resume_session(&session_id, &content).await
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
