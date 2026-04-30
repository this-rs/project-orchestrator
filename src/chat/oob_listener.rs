//! Out-of-band SDK message listener.
//!
//! Background tokio task spawned per chat session that consumes the
//! long-lived `broadcast::Sender<Message>` exposed by `nexus_claude::
//! InteractiveClient::subscribe_messages()` and surfaces "out-of-band"
//! events into the chat event stream.
//!
//! ## What is an "out-of-band" event?
//!
//! Any SDK [`Message`] that arrives on the broadcast while no active
//! `stream_response` is running for the session — typically:
//!
//! - Background tool notifications (the `Monitor` tool firing on a
//!   matched log line, `BashOutput` from a `run_in_background` Bash
//!   completing, sub-Agent completion events from background `Task`s)
//! - System notifications fired by the CLI between turns
//!
//! ## Architecture
//!
//! Spawned once per session in [`super::manager::ChatManager::create_session`]
//! and again on [`super::manager::ChatManager::resume_session`]. The task
//! takes a brief lock on the [`InteractiveClient`] only to call
//! `subscribe_messages()`, which returns a `'static` stream that lives
//! independently of the lock. The task then loops on that stream and the
//! `cancel` token.
//!
//! ## Interaction with `stream_response`
//!
//! `stream_response` itself subscribes to the same broadcast (via
//! `send_and_receive_stream` internally). When a turn is active, BOTH
//! subscribers receive each `Message`. The `is_streaming` flag is the
//! coordination signal: this listener treats messages as OOB **only** when
//! `is_streaming == false`. In-stream messages are delegated to
//! `stream_response`'s normal handling and we silently skip them here to
//! avoid duplicate persistence/broadcast.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use futures::StreamExt;
use nexus_claude::{
    AssistantMessage, ContentBlock, ContentValue, InteractiveClient, Message, UserMessage,
};
use tokio::sync::{Mutex, broadcast};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::types::ChatEvent;

/// Spawn the OOB listener task for a session. Fire-and-forget — the task
/// is owned by the tokio runtime and stops when `cancel` is fired.
///
/// `session_id` is used only for tracing context.
///
/// `client` and `cancel` MUST outlive the task by the natural session
/// lifecycle: when the `ActiveSession` is dropped or replaced (resume),
/// the caller fires `cancel` and the task exits cleanly within ~100ms.
pub fn spawn_oob_listener(
    session_id: String,
    client: Arc<Mutex<InteractiveClient>>,
    events_tx: broadcast::Sender<ChatEvent>,
    is_streaming: Arc<AtomicBool>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        // Briefly take the client lock to call subscribe_messages(). The
        // returned stream is `'static` and independent of the lock — the
        // lock is released as soon as this block exits.
        let stream = {
            let client = client.lock().await;
            client.subscribe_messages().await
        };

        let mut stream = match stream {
            Some(s) => s,
            None => {
                warn!(
                    session_id = %session_id,
                    "OOB listener: subscribe_messages() returned None \
                     (mock transport or transport not connected). \
                     Spontaneous messages between turns will not be observed."
                );
                return;
            }
        };

        info!(session_id = %session_id, "OOB listener started");

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    info!(session_id = %session_id, "OOB listener cancelled, exiting");
                    return;
                }
                next = stream.next() => {
                    match next {
                        Some(Ok(message)) => {
                            // If a stream_response is currently running, that path
                            // already consumes the same broadcast and converts the
                            // message into proper ChatEvents (ToolUse, AssistantText, …).
                            // Emitting BackgroundOutput here would duplicate the event.
                            if is_streaming.load(Ordering::Relaxed) {
                                debug!(
                                    session_id = %session_id,
                                    "OOB listener: in-stream message — skipping (stream_response handles it)"
                                );
                                continue;
                            }
                            handle_oob_message(&session_id, &events_tx, &message);
                        }
                        Some(Err(e)) => {
                            warn!(
                                session_id = %session_id,
                                error = %e,
                                "OOB listener: error reading from broadcast stream"
                            );
                        }
                        None => {
                            // Stream closed: the broadcast::Sender was dropped, which
                            // means the subprocess transport is gone (CLI died or
                            // disconnect() was called). The session is effectively
                            // dead — let the caller know via SystemHint and exit.
                            warn!(
                                session_id = %session_id,
                                "OOB listener: SDK message broadcast closed (subprocess gone); exiting"
                            );
                            let _ = events_tx.send(ChatEvent::SystemHint {
                                content: "The CLI subprocess for this session has exited. \
                                          Send a message to start a new turn (will resume \
                                          the session if possible)."
                                    .into(),
                            });
                            return;
                        }
                    }
                }
            }
        }
    });
}

/// Convert an OOB `Message` to a `ChatEvent::BackgroundOutput` and emit it
/// on the session's broadcast. Persistence and queueing for trigger are
/// handled separately (T5).
fn handle_oob_message(
    session_id: &str,
    events_tx: &broadcast::Sender<ChatEvent>,
    message: &Message,
) {
    let payload = extract_payload(message);
    let Some((source, content, correlation_id)) = payload else {
        debug!(
            session_id = %session_id,
            "OOB listener: message variant produced no payload — skipping"
        );
        return;
    };

    if content.trim().is_empty() {
        debug!(
            session_id = %session_id,
            source = %source,
            "OOB listener: empty content payload — skipping"
        );
        return;
    }

    let event = ChatEvent::BackgroundOutput {
        source: source.clone(),
        content,
        received_at: Utc::now(),
        correlation_id,
    };

    info!(
        session_id = %session_id,
        source = %source,
        "OOB listener: emitting BackgroundOutput"
    );

    // Fire-and-forget: if there are no subscribers (no WebSocket clients,
    // no stream_response listening), the event is dropped. Persistence to
    // the graph is added in T5/T9 — for now the broadcast is the only sink.
    let _ = events_tx.send(event);
}

/// Map a `nexus_claude::Message` variant to `(source, content, correlation_id)`
/// suitable for `ChatEvent::BackgroundOutput`. Returns `None` for variants
/// that should be ignored at the OOB level (e.g. `Result` end-of-turn,
/// `StreamEvent` partial tokens — these are in-stream artefacts).
///
/// `source` is a coarse label intended for UI categorisation; `content`
/// is the human-readable payload; `correlation_id` is the parent tool_use
/// ID when the message came from a sidechain (e.g. background `Task`).
fn extract_payload(message: &Message) -> Option<(String, String, Option<String>)> {
    match message {
        Message::Assistant {
            message: m,
            parent_tool_use_id,
        } => Some((
            "assistant".to_string(),
            assistant_text(m),
            parent_tool_use_id.clone(),
        )),
        Message::User {
            message: m,
            parent_tool_use_id,
        } => Some((
            user_message_source(m),
            user_text(m),
            parent_tool_use_id.clone(),
        )),
        Message::System { subtype, data } => Some((
            format!("system:{subtype}"),
            // System data is opaque JSON — surface it pretty-printed so
            // humans can read it; the UI can collapse if too noisy.
            serde_json::to_string_pretty(data).unwrap_or_default(),
            None,
        )),
        // Result and StreamEvent are turn-internal and shouldn't normally
        // arrive OOB. If they do, we silently ignore them (a `Result`
        // arriving without a turn is meaningless; a `StreamEvent` is a
        // partial that has no value without its enclosing turn).
        Message::Result { .. } | Message::StreamEvent { .. } => None,
    }
}

fn assistant_text(m: &AssistantMessage) -> String {
    // AssistantMessage::content is `Vec<ContentBlock>`. Concatenate text
    // blocks; synthesise placeholders for non-text blocks so the UI sees
    // something meaningful.
    m.content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(t) => t.text.clone(),
            ContentBlock::Thinking(t) => format!("[thinking] {}", t.thinking),
            ContentBlock::ToolUse(t) => format!("[tool_use: {}]", t.name),
            ContentBlock::ToolResult(t) => format_tool_result(t),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_tool_result(t: &nexus_claude::ToolResultContent) -> String {
    match &t.content {
        Some(ContentValue::Text(s)) => format!("[tool_result] {s}"),
        Some(ContentValue::Structured(blocks)) => {
            let payload = serde_json::to_string(blocks).unwrap_or_default();
            format!("[tool_result] {payload}")
        }
        None => "[tool_result]".to_string(),
    }
}

fn user_text(m: &UserMessage) -> String {
    // UserMessage::content is a String (text content, possibly empty).
    // UserMessage::content_blocks holds structured blocks (tool_result, ...).
    // Concatenate both, preferring blocks when both are present.
    if let Some(blocks) = &m.content_blocks {
        let rendered: String = blocks
            .iter()
            .map(|block| match block {
                ContentBlock::Text(t) => t.text.clone(),
                ContentBlock::Thinking(t) => format!("[thinking] {}", t.thinking),
                ContentBlock::ToolUse(t) => format!("[tool_use: {}]", t.name),
                ContentBlock::ToolResult(t) => format_tool_result(t),
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !rendered.trim().is_empty() {
            return rendered;
        }
    }
    m.content.clone()
}

fn user_message_source(m: &UserMessage) -> String {
    // If the message carries a tool_result block, label accordingly so the
    // UI/dedup logic can distinguish from real user input.
    if let Some(blocks) = &m.content_blocks {
        for block in blocks {
            if matches!(block, ContentBlock::ToolResult(_)) {
                return "tool_result".to_string();
            }
        }
    }
    "user".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_claude::{TextContent, ToolResultContent};
    use serde_json::json;

    fn user_with_blocks(blocks: Vec<ContentBlock>) -> Message {
        Message::User {
            message: UserMessage {
                content: String::new(),
                content_blocks: Some(blocks),
            },
            parent_tool_use_id: Some("tool-bg-123".to_string()),
        }
    }

    fn user_plain(text: &str) -> Message {
        Message::User {
            message: UserMessage {
                content: text.to_string(),
                content_blocks: None,
            },
            parent_tool_use_id: None,
        }
    }

    #[test]
    fn test_extract_payload_ignores_result_and_stream_event() {
        let result_msg = Message::Result {
            subtype: "ok".into(),
            duration_ms: 0,
            duration_api_ms: 0,
            is_error: false,
            num_turns: 1,
            session_id: "s".into(),
            total_cost_usd: None,
            usage: None,
            result: None,
            structured_output: None,
        };
        assert!(extract_payload(&result_msg).is_none());
    }

    #[test]
    fn test_extract_payload_user_tool_result_labels_source() {
        let msg = user_with_blocks(vec![ContentBlock::ToolResult(ToolResultContent {
            tool_use_id: "t1".into(),
            content: Some(ContentValue::Text("epoch 50 cos=0.871".into())),
            is_error: Some(false),
        })]);
        let (source, content, corr) = extract_payload(&msg).expect("user message yields payload");
        assert_eq!(source, "tool_result");
        assert!(
            content.contains("epoch 50 cos=0.871"),
            "content should include tool_result text, got: {content}"
        );
        assert_eq!(corr.as_deref(), Some("tool-bg-123"));
    }

    #[test]
    fn test_extract_payload_user_plain_string() {
        let msg = user_plain("hello world");
        let (source, content, _) = extract_payload(&msg).expect("user message yields payload");
        assert_eq!(source, "user");
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_extract_payload_system_renders_data() {
        let msg = Message::System {
            subtype: "notification".into(),
            data: json!({"level": "info", "text": "training done"}),
        };
        let (source, content, _) = extract_payload(&msg).expect("system message yields payload");
        assert_eq!(source, "system:notification");
        assert!(content.contains("training done"));
    }

    #[test]
    fn test_assistant_text_concatenates_blocks() {
        let m = AssistantMessage {
            content: vec![
                ContentBlock::Text(TextContent {
                    text: "Hello".into(),
                }),
                ContentBlock::Text(TextContent {
                    text: "world".into(),
                }),
            ],
        };
        assert_eq!(assistant_text(&m), "Hello\nworld");
    }
}
