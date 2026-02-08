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
    socket: WebSocket,
    state: OrchestratorState,
    session_id: String,
    last_event: i64,
) {
    let (mut ws_sender, mut ws_receiver) = socket.split();

    info!(session_id = %session_id, last_event, "Chat WebSocket client connected");

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

    if !events.is_empty() {
        // New format: replay ChatEventRecord from Neo4j
        debug!(
            session_id = %session_id,
            count = events.len(),
            "Replaying persisted ChatEventRecord"
        );
        for event in events {
            let msg = serde_json::json!({
                "seq": event.seq,
                "type": event.event_type,
                "data": serde_json::from_str::<serde_json::Value>(&event.data)
                    .unwrap_or_else(|_| serde_json::Value::String(event.data.clone())),
                "replaying": true,
            });
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
        match chat_manager.get_session_messages(&session_id, None, None).await {
            Ok(loaded) => {
                let messages = loaded.messages_chronological();
                debug!(
                    session_id = %session_id,
                    count = messages.len(),
                    "Replaying Nexus message history"
                );
                let mut seq = 1i64;
                for msg in messages {
                    let event_type = if msg.role == "user" {
                        "user_message"
                    } else {
                        "assistant_text"
                    };
                    let data = serde_json::json!({ "content": msg.content });
                    let replay_msg = serde_json::json!({
                        "seq": seq,
                        "type": event_type,
                        "data": data,
                        "replaying": true,
                    });
                    seq += 1;
                    if ws_sender
                        .send(Message::Text(replay_msg.to_string().into()))
                        .await
                        .is_err()
                    {
                        debug!("Client disconnected during Nexus replay");
                        return;
                    }
                }
            }
            Err(e) => {
                warn!(
                    session_id = %session_id,
                    error = %e,
                    "Failed to load Nexus message history for replay"
                );
            }
        }
    }

    // ========================================================================
    // Phase 1.5: Send streaming snapshot if a stream is in progress
    // ========================================================================
    let (is_currently_streaming, partial_text) =
        chat_manager.get_streaming_snapshot(&session_id).await;

    if is_currently_streaming && !partial_text.is_empty() {
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

    if is_currently_streaming {
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

    // ========================================================================
    // Phase 2: Subscribe to broadcast for real-time events
    // ========================================================================
    // Try to get a broadcast receiver. If the session is not active (no CLI running),
    // that's OK — the client can still send messages to resume it.
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

    // Ping interval (30s)
    let mut ping_interval = interval(Duration::from_secs(30));
    ping_interval.tick().await; // skip first immediate tick

    // ========================================================================
    // Phase 3: Event loop — forward broadcast events + handle client messages
    // ========================================================================
    loop {
        tokio::select! {
            // Forward broadcast events to the WebSocket client
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
                        // Wrap the event with seq (0 for stream_delta)
                        let seq = match &chat_event {
                            crate::chat::types::ChatEvent::StreamDelta { .. } => 0i64,
                            _ => 0i64, // real seq comes from persistence; broadcast events don't carry seq
                        };

                        // Serialize the chat event and add seq wrapper
                        match serde_json::to_value(&chat_event) {
                            Ok(mut val) => {
                                if let Some(obj) = val.as_object_mut() {
                                    obj.insert("seq".to_string(), serde_json::json!(seq));
                                }
                                if ws_sender
                                    .send(Message::Text(val.to_string().into()))
                                    .await
                                    .is_err()
                                {
                                    debug!("WebSocket send failed, client disconnected");
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to serialize ChatEvent: {}", e);
                            }
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

                                        // Check if session is active; if not, auto-resume
                                        let result = if !chat_manager.is_session_active(&session_id).await {
                                            chat_manager.resume_session(&session_id, &content).await
                                        } else {
                                            chat_manager.send_message(&session_id, &content).await
                                        };

                                        match result {
                                            Ok(()) => {
                                                // If we didn't have a broadcast subscription, get one now
                                                if event_rx.is_none() {
                                                    match chat_manager.subscribe(&session_id).await {
                                                        Ok(rx) => {
                                                            event_rx = Some(rx);
                                                            debug!(session_id = %session_id, "Subscribed to broadcast after first message");
                                                        }
                                                        Err(e) => {
                                                            warn!(session_id = %session_id, error = %e, "Failed to subscribe after send");
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

                                    WsChatClientMessage::PermissionResponse { allow, .. } => {
                                        debug!(session_id = %session_id, allow, "WS: Received permission_response");
                                        let msg = format!(
                                            "Permission {}",
                                            if allow { "granted" } else { "denied" }
                                        );
                                        if let Err(e) = chat_manager.send_message(&session_id, &msg).await {
                                            warn!(session_id = %session_id, error = %e, "Failed to send permission response");
                                        }
                                    }

                                    WsChatClientMessage::InputResponse { content, .. } => {
                                        debug!(session_id = %session_id, "WS: Received input_response");
                                        if let Err(e) = chat_manager.send_message(&session_id, &content).await {
                                            warn!(session_id = %session_id, error = %e, "Failed to send input response");
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
