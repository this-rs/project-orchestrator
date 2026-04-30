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
//!
//! ## Triggering on idle
//!
//! When an OOB event arrives and the session is idle (no active turn),
//! the listener:
//!   1. Persists the event as a `ChatEventRecord` of type `background_output`
//!      (so it appears in `chat::list_messages` history)
//!   2. Broadcasts a `ChatEvent::BackgroundOutput` (so WebSocket clients
//!      see it in real time)
//!   3. Pushes a `PendingMessage::BackgroundOutput` carrying the content
//!   4. Tries to claim the streaming flag via `compare_exchange(false, true)`
//!      and, if it wins the race, spawns a new `stream_response` so the
//!      LLM reacts to the event without waiting for the next user turn
//!   5. If it loses the race (a concurrent user message claimed the
//!      flag first), the queued `PendingMessage` will be picked up by
//!      `drain_pending_messages` after that turn ends — no event is
//!      dropped.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::StreamExt;
use nexus_claude::{
    AssistantMessage, ContentBlock, ContentValue, InteractiveClient, Message, UserMessage,
};
use tokio::sync::{Mutex, RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::manager::{ActiveSession, ChatManager};
use super::types::{ChatEvent, PendingMessage};
use crate::meilisearch::SearchStore;
use crate::neo4j::GraphStore;
use crate::neo4j::models::ChatEventRecord;

/// Dependencies needed to trigger a fresh `stream_response` from an
/// idle session when an OOB event arrives. These are all
/// `ChatManager`-bound (not session-bound) and are cloned once at spawn
/// time. Session-bound state is looked up via `active_sessions` at
/// trigger time, mirroring the pattern used by `spawn_nats_rpc_listener`.
pub(crate) struct OobListenerDeps {
    pub graph: Arc<dyn GraphStore>,
    pub active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    pub context_injector: Option<Arc<nexus_claude::memory::ContextInjector>>,
    pub event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    pub retry_config: super::config::RetryConfig,
    pub enrichment_pipeline: Arc<super::enrichment::EnrichmentPipeline>,
    pub search: Arc<dyn SearchStore>,
    pub nats: Option<Arc<crate::events::NatsEmitter>>,
}

/// Spawn the OOB listener task for a session. Fire-and-forget — the task
/// is owned by the tokio runtime and stops when `cancel` is fired.
///
/// `session_id` is used only for tracing context.
///
/// ## Recovery semantics (T9 of plan 9a1684b2)
///
/// Recovery of interrupted runs at server boot — handled by
/// `crate::runner::runner::Runner::recover_interrupted_runs` — does **not**
/// need a dedicated `respawn_oob_listener` API. That recovery path
/// re-enters the normal session lifecycle:
///
///   `recover_interrupted_runs → execute_plan → ChatManager::create_session
///    → spawn_oob_listener (this function)`
///
/// Each restored runner session therefore gets its own fresh OOB
/// listener bound to a freshly spawned `InteractiveClient`, identical
/// to a never-interrupted session. The original cancellation token of
/// the pre-crash listener is gone with the process — there's nothing to
/// cancel — so we just spawn a new one.
///
/// `resume_session` (used when the user reconnects to an existing
/// `cli_session_id` mid-conversation) follows the same pattern: the old
/// `nats_cancel` token is fired before the new session is built, so any
/// stale listener exits cleanly before the new one starts.
pub(crate) fn spawn_oob_listener(
    session_id: String,
    client: Arc<Mutex<InteractiveClient>>,
    events_tx: broadcast::Sender<ChatEvent>,
    is_streaming: Arc<AtomicBool>,
    next_seq: Arc<AtomicI64>,
    cancel: CancellationToken,
    deps: OobListenerDeps,
) {
    let session_uuid = Uuid::parse_str(&session_id).ok();

    tokio::spawn(async move {
        // Briefly take the client lock to call subscribe_messages(). The
        // returned stream is `'static` and independent of the lock.
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
                            // already consumes the same broadcast. Skip silently.
                            if is_streaming.load(Ordering::Relaxed) {
                                debug!(
                                    session_id = %session_id,
                                    "OOB listener: in-stream message — skipping (stream_response handles it)"
                                );
                                continue;
                            }
                            handle_oob_message(
                                &session_id,
                                session_uuid,
                                &events_tx,
                                &is_streaming,
                                &next_seq,
                                &client,
                                &deps,
                                &message,
                            )
                            .await;
                        }
                        Some(Err(e)) => {
                            warn!(
                                session_id = %session_id,
                                error = %e,
                                "OOB listener: error reading from broadcast stream"
                            );
                        }
                        None => {
                            emit_subprocess_death(&session_id, &events_tx);
                            return;
                        }
                    }
                }
            }
        }
    });
}

/// Handle a single OOB Message: persist, broadcast, push to queue, and
/// try to claim the streaming flag to spawn a new turn.
#[allow(clippy::too_many_arguments)]
async fn handle_oob_message(
    session_id: &str,
    session_uuid: Option<Uuid>,
    events_tx: &broadcast::Sender<ChatEvent>,
    is_streaming: &Arc<AtomicBool>,
    next_seq: &Arc<AtomicI64>,
    _client: &Arc<Mutex<InteractiveClient>>,
    deps: &OobListenerDeps,
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
        content: content.clone(),
        received_at: Utc::now(),
        correlation_id,
    };

    info!(
        session_id = %session_id,
        source = %source,
        oob_event_type = "background_output",
        "OOB listener: emitting BackgroundOutput"
    );

    // 1. Persist as ChatEventRecord (so it appears in list_messages history).
    // Time the round-trip — `oob_persistence_latency_ms` is a structured
    // tracing field that log-based metrics pipelines (e.g. Vector/Loki/
    // Datadog) can extract as a histogram for production monitoring (T9).
    if let Some(uuid) = session_uuid {
        let record = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: event.event_type().to_string(),
            data: serde_json::to_string(&event).unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        let persist_start = Instant::now();
        match deps.graph.store_chat_events(uuid, vec![record]).await {
            Ok(_) => {
                let latency_ms = persist_start.elapsed().as_millis() as u64;
                debug!(
                    session_id = %session_id,
                    source = %source,
                    oob_persistence_latency_ms = latency_ms,
                    "OOB listener: persisted BackgroundOutput record"
                );
            }
            Err(e) => {
                let latency_ms = persist_start.elapsed().as_millis() as u64;
                warn!(
                    session_id = %session_id,
                    source = %source,
                    oob_persistence_latency_ms = latency_ms,
                    error = %e,
                    "OOB listener: failed to persist BackgroundOutput record (non-fatal)"
                );
            }
        }
    }

    // 2. Broadcast on local channel + NATS so all WebSocket clients see it
    let _ = events_tx.send(event.clone());
    if let Some(ref nats) = deps.nats {
        nats.publish_chat_event(session_id, event);
    }

    // 3 & 4. Push to pending_messages and try to claim streaming, then
    //         maybe trigger a new stream_response.
    maybe_trigger_stream(session_id, content, is_streaming, deps).await;
}

/// Push the OOB content as a `PendingMessage::BackgroundOutput` and, if
/// the session is idle, atomically claim the streaming flag and spawn
/// `stream_response` so the LLM reacts.
///
/// Race semantics: we use `compare_exchange(false, true)` on
/// `is_streaming`. Whichever caller first transitions false→true owns
/// the spawn; concurrent OOB events or user messages losing the race
/// just leave their entries in the queue and the active stream's
/// `drain_pending_messages` picks them up after Result.
async fn maybe_trigger_stream(
    session_id: &str,
    oob_content: String,
    is_streaming: &Arc<AtomicBool>,
    deps: &OobListenerDeps,
) {
    // Look up session-bound state. If the session was just removed,
    // there's nothing to do.
    let session_state = {
        let sessions = deps.active_sessions.read().await;
        sessions.get(session_id).map(|s| {
            (
                s.client.clone(),
                s.events_tx.clone(),
                s.interrupt_flag.clone(),
                s.memory_manager.clone(),
                s.next_seq.clone(),
                s.pending_messages.clone(),
                s.is_streaming.clone(),
                s.streaming_text.clone(),
                s.streaming_events.clone(),
                s.sdk_control_rx.clone(),
                s.auto_continue.clone(),
                s.oob_trigger_history.clone(),
                s.oob_trigger_cap,
                s.oob_trigger_window,
                s.oob_capped_warned.clone(),
            )
        })
    };

    let Some((
        client,
        events_tx,
        interrupt_flag,
        memory_manager,
        next_seq,
        pending_messages,
        sess_is_streaming,
        streaming_text,
        streaming_events,
        sdk_control_rx,
        auto_continue,
        oob_trigger_history,
        oob_trigger_cap,
        oob_trigger_window,
        oob_capped_warned,
    )) = session_state
    else {
        debug!(
            session_id = %session_id,
            "OOB listener: session no longer in active_sessions — dropping trigger"
        );
        return;
    };

    // T7 — Rate cap (sliding window) on OOB-driven `stream_response` spawns.
    // Drains entries older than the window, then either blocks (cap hit)
    // or records a fresh timestamp. Blocking ALSO skips queueing the
    // PendingMessage::BackgroundOutput so the queue cannot be flooded
    // and dumped into the next user-driven turn — the OOB event is
    // still persisted+broadcast (already done in `handle_oob_message`).
    if check_and_record_trigger_cap(
        session_id,
        &events_tx,
        &oob_trigger_history,
        oob_trigger_cap,
        oob_trigger_window,
        &oob_capped_warned,
    )
    .await
    {
        // Cap hit — refuse to queue + spawn. Warning already emitted
        // (at most once per window) by the helper.
        return;
    }

    // Push the OOB content as a queue entry so drain_pending_messages
    // (or the spawned stream_response below) can process it later.
    {
        let mut queue = pending_messages.lock().await;
        queue.push_back(PendingMessage::background_output(oob_content.clone()));
    }

    // Claim the streaming flag atomically. Loser leaves the message in
    // the queue — drain will pick it up after the active turn ends.
    let won_race = is_streaming
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok();

    if !won_race {
        debug!(
            session_id = %session_id,
            "OOB listener: lost is_streaming race — message queued for active stream's drain"
        );
        return;
    }

    // T9 — Structured trigger log. `oob_trigger_count` is a structured
    // tracing field (sliding-window depth at the moment of trigger),
    // suitable for a counter via log-based metrics. Pair with the cap
    // for a `oob_cap_utilization = count / cap` derived metric.
    let trigger_count = oob_trigger_history.lock().await.len();
    info!(
        session_id = %session_id,
        oob_trigger_count = trigger_count,
        oob_trigger_cap = oob_trigger_cap,
        "OOB listener: idle session, spawning stream_response for queued OOB event"
    );

    // Pop the entry we just pushed (drain will see it via the queue).
    // Actually we WANT drain to handle it — but stream_response needs an
    // initial prompt. Give it the OOB content directly; the queue
    // remains for any subsequent OOB events that may have stacked up
    // between the push and the spawn.
    let prompt = {
        let mut queue = pending_messages.lock().await;
        queue.pop_front().map(|m| m.content).unwrap_or(oob_content)
    };

    // Spawn stream_response. Same call shape as in `spawn_nats_rpc_listener`.
    let session_id_for_spawn = session_id.to_string();
    let graph = deps.graph.clone();
    let active_sessions = deps.active_sessions.clone();
    let context_injector = deps.context_injector.clone();
    let event_emitter = deps.event_emitter.clone();
    let retry_config = deps.retry_config.clone();
    let enrichment_pipeline = deps.enrichment_pipeline.clone();
    let search = deps.search.clone();
    let nats = deps.nats.clone();
    // `client` was cloned earlier from ActiveSession but only the spawned
    // stream_response below consumes it; touch it here to make ownership
    // explicit without moving `sess_is_streaming`, which we still need
    // below for the Arc::clone.
    let _ = client.clone();

    tokio::spawn(async move {
        ChatManager::stream_response(
            client,
            events_tx,
            prompt,
            session_id_for_spawn,
            graph,
            active_sessions,
            interrupt_flag,
            memory_manager,
            context_injector,
            next_seq,
            pending_messages,
            // Pass the OUTER is_streaming flag — it's the same Arc as
            // sess_is_streaming (cloned from the same source), but we
            // already set it to true above so stream_response won't
            // re-set it.
            // SAFETY: same Arc<AtomicBool> across the session.
            // NB: stream_response will set is_streaming=false at the
            // end via post_stream::finalize_streaming_status.
            std::sync::Arc::clone(&sess_is_streaming),
            streaming_text,
            streaming_events,
            event_emitter,
            nats,
            sdk_control_rx,
            auto_continue,
            retry_config,
            enrichment_pipeline,
            search,
        )
        .await;
    });
}

/// Emit a `ChatEvent::SessionError` describing subprocess death and log
/// a `warn!` (T9 of plan 9a1684b2). Extracted as a small helper so the
/// behaviour is unit-testable without spinning up a real
/// `InteractiveClient` and the full broadcast machinery.
///
/// Called from the listener loop's `None` branch — when
/// `stream.next()` yields `None` the SDK transport's broadcast sender
/// has been dropped, which means the CLI subprocess for this session
/// has exited (clean shutdown OR crash). Subsequent `send_message`
/// calls on this `session_id` will silently no-op until the session is
/// recreated, so we surface a typed event the frontend can render
/// prominently.
fn emit_subprocess_death(session_id: &str, events_tx: &broadcast::Sender<ChatEvent>) {
    warn!(
        session_id = %session_id,
        reason = "subprocess_exited",
        "OOB listener: SDK message broadcast closed (subprocess gone); exiting"
    );
    let _ = events_tx.send(ChatEvent::SessionError {
        reason: "subprocess_exited".to_string(),
        message: "The CLI subprocess for this session has exited. \
                  Send a message to start a new turn (will resume \
                  the session if possible)."
            .into(),
        received_at: Utc::now(),
    });
}

/// Sliding-window rate cap for OOB-driven `stream_response` spawns
/// (T7 of plan 9a1684b2). Returns `true` when the trigger should be
/// **blocked** (cap hit), `false` when it can proceed.
///
/// Behaviour:
/// 1. Lock the history deque, drain timestamps older than `window`.
/// 2. If `len >= cap`: emit a `ChatEvent::SystemHint` warning **at most
///    once per window** (gated by `capped_warned`), log a warn, return
///    `true`.
/// 3. Otherwise: clear the `capped_warned` flag (we're back below the
///    cap), push `Instant::now()`, return `false`.
async fn check_and_record_trigger_cap(
    session_id: &str,
    events_tx: &broadcast::Sender<ChatEvent>,
    history: &Arc<Mutex<VecDeque<Instant>>>,
    cap: u32,
    window: Duration,
    capped_warned: &Arc<AtomicBool>,
) -> bool {
    let mut hist = history.lock().await;
    let now = Instant::now();

    // Drain entries older than `window`. Since timestamps are pushed in
    // chronological order, popping from the front while the head is too
    // old is sufficient.
    while let Some(front) = hist.front() {
        if now.duration_since(*front) > window {
            hist.pop_front();
        } else {
            break;
        }
    }

    if (hist.len() as u32) >= cap {
        // Capped. Emit warning once per window via the AtomicBool gate.
        if capped_warned
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let warn_msg = format!(
                "⚠️ OOB auto-continue cap reached: {} background-triggered turns in the last \
                 {}s. Further background events will be persisted but will NOT auto-restart \
                 the agent until activity slows down. Send a message to continue manually \
                 if needed.",
                cap,
                window.as_secs()
            );
            warn!(
                session_id = %session_id,
                cap = cap,
                window_secs = window.as_secs(),
                "OOB listener: trigger cap reached, throttling further auto-continues"
            );
            let _ = events_tx.send(ChatEvent::SystemHint { content: warn_msg });
        } else {
            debug!(
                session_id = %session_id,
                "OOB listener: trigger still capped (warning already emitted)"
            );
        }
        return true;
    }

    // Below cap — clear the warned flag (so a new spike will re-warn)
    // and record this trigger.
    capped_warned.store(false, Ordering::SeqCst);
    hist.push_back(now);
    false
}

/// Map a `nexus_claude::Message` variant to `(source, content, correlation_id)`
/// suitable for `ChatEvent::BackgroundOutput`. Returns `None` for variants
/// that should be ignored at the OOB level (e.g. `Result` end-of-turn,
/// `StreamEvent` partial tokens — these are in-stream artefacts).
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
            serde_json::to_string_pretty(data).unwrap_or_default(),
            None,
        )),
        Message::Result { .. } | Message::StreamEvent { .. } => None,
    }
}

fn assistant_text(m: &AssistantMessage) -> String {
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

    // ── T7: trigger cap (sliding window) tests ──────────────────────────

    /// Subscribe to the broadcast and collect all SystemHint warnings
    /// received during the closure's execution.
    async fn collect_warnings(
        events_tx: &broadcast::Sender<ChatEvent>,
    ) -> tokio::sync::mpsc::UnboundedReceiver<String> {
        let mut rx = events_tx.subscribe();
        let (tx, out_rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Ok(ev) = rx.recv().await {
                if let ChatEvent::SystemHint { content } = ev {
                    let _ = tx.send(content);
                }
            }
        });
        out_rx
    }

    #[tokio::test]
    async fn test_trigger_cap_blocks_after_cap_and_warns_once() {
        let (events_tx, _rx_keep) = broadcast::channel(64);
        let mut warn_rx = collect_warnings(&events_tx).await;
        let history = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
        let cap: u32 = 3;
        let window = Duration::from_secs(60);
        let warned = Arc::new(AtomicBool::new(false));

        // First `cap` calls: all allowed, no warning.
        for _ in 0..cap {
            let blocked = check_and_record_trigger_cap(
                "test-session", &events_tx, &history, cap, window, &warned,
            )
            .await;
            assert!(!blocked, "first {cap} triggers must not be blocked");
        }

        // Next 50 calls: all blocked, warning emitted exactly once.
        for _ in 0..50 {
            let blocked = check_and_record_trigger_cap(
                "test-session", &events_tx, &history, cap, window, &warned,
            )
            .await;
            assert!(blocked, "calls beyond cap must be blocked");
        }

        // Drain warnings — give the spawn loop a tick to flush.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let mut warnings = Vec::new();
        while let Ok(w) = warn_rx.try_recv() {
            warnings.push(w);
        }
        assert_eq!(
            warnings.len(),
            1,
            "expected exactly one warning emitted per window, got: {warnings:?}"
        );
        assert!(
            warnings[0].contains("OOB auto-continue cap reached"),
            "warning content unexpected: {}",
            warnings[0]
        );
    }

    #[tokio::test]
    async fn test_trigger_cap_drops_old_entries() {
        let (events_tx, _rx_keep) = broadcast::channel(16);
        let history = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
        let cap: u32 = 2;
        // Tiny window so we can age entries out in test time.
        let window = Duration::from_millis(50);
        let warned = Arc::new(AtomicBool::new(false));

        // Fill the window.
        for _ in 0..cap {
            assert!(
                !check_and_record_trigger_cap(
                    "s", &events_tx, &history, cap, window, &warned
                )
                .await
            );
        }
        assert!(
            check_and_record_trigger_cap("s", &events_tx, &history, cap, window, &warned).await,
            "cap should be hit"
        );

        // Wait past the window.
        tokio::time::sleep(Duration::from_millis(80)).await;

        // Now the window is empty again — should accept fresh triggers
        // and re-arm the warning gate.
        assert!(
            !check_and_record_trigger_cap("s", &events_tx, &history, cap, window, &warned).await,
            "after window expiry, triggers should be accepted again"
        );
        assert!(
            !warned.load(Ordering::SeqCst),
            "warned flag must reset when below cap again"
        );
    }

    #[tokio::test]
    async fn test_trigger_cap_warning_rearms_after_recovery() {
        let (events_tx, _rx_keep) = broadcast::channel(16);
        let mut warn_rx = collect_warnings(&events_tx).await;
        let history = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
        let cap: u32 = 1;
        let window = Duration::from_millis(50);
        let warned = Arc::new(AtomicBool::new(false));

        // Hit the cap — first call OK, second blocked + warns.
        assert!(
            !check_and_record_trigger_cap("s", &events_tx, &history, cap, window, &warned).await
        );
        assert!(
            check_and_record_trigger_cap("s", &events_tx, &history, cap, window, &warned).await
        );
        // Wait past window so entries drop.
        tokio::time::sleep(Duration::from_millis(80)).await;
        // First post-recovery call accepted, clears warned.
        assert!(
            !check_and_record_trigger_cap("s", &events_tx, &history, cap, window, &warned).await
        );
        // Hit cap again — should re-warn (not silently swallow).
        assert!(
            check_and_record_trigger_cap("s", &events_tx, &history, cap, window, &warned).await
        );

        tokio::time::sleep(Duration::from_millis(20)).await;
        let mut warnings = Vec::new();
        while let Ok(w) = warn_rx.try_recv() {
            warnings.push(w);
        }
        assert_eq!(
            warnings.len(),
            2,
            "expected two warnings (one per cap-hit episode), got: {warnings:?}"
        );
    }

    // ── T9: subprocess death detection ──────────────────────────────────

    /// `emit_subprocess_death` must publish exactly one
    /// `ChatEvent::SessionError` with `reason = "subprocess_exited"` and
    /// a non-empty user-facing message. The frontend uses the `reason`
    /// field for routing (e.g., "transport_lost" vs "broadcast_dropped"
    /// in future variants), so the string is part of the public contract.
    #[tokio::test]
    async fn test_emit_subprocess_death_emits_session_error() {
        let (events_tx, mut rx) = broadcast::channel(8);
        emit_subprocess_death("session-123", &events_tx);

        // The send must have produced exactly one event.
        let received = rx.try_recv().expect("expected one event on broadcast");
        match received {
            ChatEvent::SessionError {
                reason,
                message,
                received_at,
            } => {
                assert_eq!(reason, "subprocess_exited");
                assert!(
                    message.contains("CLI subprocess"),
                    "message must mention the subprocess for the user, got: {message}"
                );
                // Sanity: timestamp is within the last second.
                let age = chrono::Utc::now()
                    .signed_duration_since(received_at)
                    .num_milliseconds()
                    .unsigned_abs();
                assert!(
                    age < 1000,
                    "received_at must be ~now, got age = {age}ms"
                );
            }
            other => panic!("expected SessionError, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "no further events expected");
    }
}
