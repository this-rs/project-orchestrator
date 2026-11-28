//! REST/SSE surface of the chat event stream: `GET /api/chat/sessions/{id}/events`.
//!
//! The same frames as `/ws/chat/{id}`, over a transport a plain HTTP client can
//! read: the catch-up core (`chat::event_stream`, RFC cdd8c6ee) replays the
//! persisted events since the client's cursor, subscribes to the live bus
//! BEFORE taking the streaming snapshot, and forwards live events with the same
//! fingerprint dedup as the socket. Nothing about the payload is re-invented
//! here — only the framing (`id:` / `data:`) and the cursor rules.
//!
//! ## The durable cursor vs. the transient deltas
//!
//! `seq` is the per-session, monotonically increasing sequence number the
//! persisted events carry. It is the ONLY thing an SSE `id:` may name:
//!
//! - a replayed frame carries its stored `seq` (`> 0`) → the frame emits
//!   `id: <seq>`, so a client's `Last-Event-ID` advances with the durable
//!   history;
//! - a live frame carries `seq: 0` (it is not persisted yet) → the frame emits
//!   `data:` only. Emitting `id: 0` would drag `Last-Event-ID` back to zero and
//!   replay the whole conversation on the next reconnect: the deltas are
//!   reconstructed from the `partial_text` snapshot instead, which is exactly
//!   what `catch_up` sends.
//!
//! A frame whose `seq` is not greater than the highest one already emitted is
//! DROPPED (the stream is append-only): replays are therefore idempotent, no
//! matter how many times the same range is delivered.
//!
//! ## Bounded memory
//!
//! Frames go through a bounded queue ([`STREAM_QUEUE_CAPACITY`]). A client that
//! stops reading fills it and the producer waits — it never buffers without
//! limit. While it waits, the broadcast bus drops the oldest events and the
//! producer then emits `events_lagged` with the number skipped, so the client
//! learns to reconnect from its cursor instead of silently missing events.
//!
//! ## Explicit terminal states
//!
//! `replay_complete` ends the replay phase, `events_lagged` reports a gap,
//! `session_dormant` reports that the runtime went away, and `stream_end`
//! closes the response on purpose (`reason`, `cursor`), so a client never has
//! to guess why the body ended.

use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::{interval, Instant};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, warn};
use uuid::Uuid;

use super::handlers::{AppError, OrchestratorState};
use super::ws_auth::CookieAuthResult;
use crate::chat::event_stream::{catch_up, live_frame, CatchUp, FrameSink};
use crate::chat::types::ChatEvent;

/// Frames a connection may hold before its producer waits. This is the whole
/// per-connection memory bound: a stalled client costs at most this many
/// frames, never an unbounded queue.
pub const STREAM_QUEUE_CAPACITY: usize = 256;

/// Period of the `: keep-alive` comment. A comment is not an event: clients
/// ignore it, but it keeps proxies from closing an idle stream and it is how a
/// dead peer is noticed (a failed write ends the loop).
#[cfg(not(test))]
pub const KEEPALIVE_EVERY: Duration = Duration::from_secs(15);
#[cfg(test)]
pub const KEEPALIVE_EVERY: Duration = Duration::from_millis(100);

/// How long a connection may stay silent before it is closed on purpose.
/// Only when nothing is working on the session: a turn that is running (a long
/// tool call produces no frame) is never cut. The last frame tells the client
/// the reason and the cursor to resume from.
#[cfg(not(test))]
pub const IDLE_CLOSE: Duration = Duration::from_secs(300);
#[cfg(test)]
pub const IDLE_CLOSE: Duration = Duration::from_millis(400);

/// Query parameters of the SSE route.
#[derive(Debug, Deserialize, Default)]
pub struct SseQuery {
    /// Durable cursor: deliver the persisted events whose `seq` is greater.
    /// The `Last-Event-ID` header wins over it (it is what an `EventSource`
    /// sends by itself); it exists for clients that keep their own cursor.
    pub cursor: Option<i64>,
    /// Alias of `cursor`, named like the socket's parameter so one client
    /// configuration can drive both transports.
    pub last_event: Option<i64>,
    /// One-time ticket, the same Tauri/WKWebView fallback the sockets accept.
    pub ticket: Option<String>,
    /// Refused on purpose: `seq` is a PER-SESSION counter, so one stream cannot
    /// carry one `Last-Event-ID` for several sessions. Named here so a client
    /// that tries it is told, rather than silently given one session.
    pub sessions: Option<String>,
}

/// Read the durable cursor a client resumes from.
///
/// Precedence: `Last-Event-ID` (sent by the browser on reconnect) → `?cursor` →
/// `?last_event` → 0. A negative value means "from the beginning": `seq`
/// starts at 1, so it is clamped rather than refused.
pub fn resolve_cursor(headers: &HeaderMap, query: &SseQuery) -> Result<i64, AppError> {
    let from_header = match headers.get("last-event-id") {
        Some(raw) => {
            let text = raw.to_str().map_err(|_| {
                AppError::BadRequest("Last-Event-ID is not a valid header value".to_string())
            })?;
            // A browser echoes the id verbatim; a value we never wrote is a
            // client bug we refuse to interpret as "from the beginning".
            Some(text.trim().parse::<i64>().map_err(|_| {
                AppError::BadRequest(format!(
                    "Last-Event-ID must be an event sequence number, got `{text}`"
                ))
            })?)
        }
        None => None,
    };

    let cursor = from_header
        .or(query.cursor)
        .or(query.last_event)
        .unwrap_or(0);
    Ok(cursor.max(0))
}

/// One SSE frame. The `id:` line names the durable sequence number and is
/// therefore emitted only for a replayed (persisted) frame: a transient frame
/// carries `seq: 0` and must not move the client's cursor.
pub fn encode_frame(seq: i64, frame: &Value) -> String {
    if seq > 0 {
        format!("id: {seq}\ndata: {frame}\n\n")
    } else {
        format!("data: {frame}\n\n")
    }
}

/// The sink `catch_up` and the live loop write to: it owns the durable cursor,
/// so the dedup and the `id:` line are decided in ONE place for both phases.
pub(crate) struct SseSink {
    tx: mpsc::Sender<String>,
    /// Highest durable `seq` this connection has already emitted.
    pub(crate) last_durable_seq: i64,
}

impl SseSink {
    pub(crate) fn new(tx: mpsc::Sender<String>, cursor: i64) -> Self {
        Self {
            tx,
            last_durable_seq: cursor,
        }
    }

    /// Send a raw chunk (a keep-alive comment). `false` once the client is gone.
    async fn send_raw(&mut self, chunk: String) -> bool {
        self.tx.send(chunk).await.is_ok()
    }
}

#[async_trait::async_trait]
impl FrameSink for SseSink {
    async fn send_frame(&mut self, frame: Value) -> bool {
        let seq = frame.get("seq").and_then(Value::as_i64).unwrap_or(0);
        if seq > 0 {
            if seq <= self.last_durable_seq {
                // Already delivered on this connection: the stream is
                // append-only, so a replayed range is ignored, not resent.
                debug!(
                    seq,
                    last = self.last_durable_seq,
                    "SSE: skipping an event already delivered"
                );
                return true;
            }
            self.last_durable_seq = seq;
        }
        self.send_raw(encode_frame(seq, &frame)).await
    }
}

/// `GET /api/chat/sessions/{id}/events`
///
/// Authentication is the chat socket's (cookie first, then `?ticket=`, then
/// anonymous when the server runs without auth): an `EventSource` cannot set an
/// `Authorization` header, and this surface carries exactly the frames the
/// socket already carries to the same audience.
pub async fn chat_events_sse(
    State(state): State<OrchestratorState>,
    Path(session_id): Path<String>,
    Query(query): Query<SseQuery>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    if query.sessions.is_some() {
        return Err(AppError::BadRequest(
            "one stream carries one session: `seq` is a per-session cursor, so a \
             multi-session stream has no single Last-Event-ID. Open one stream per \
             session, or use /ws/events?sessions=<id,...> for the multiplexed form."
                .to_string(),
        ));
    }

    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?
        .clone();

    let uuid = Uuid::parse_str(&session_id)
        .map_err(|_| AppError::BadRequest(format!("Invalid session ID: {}", session_id)))?;

    // Same existence check as the socket: an unknown session is a 404, never a
    // stream that stays open and silent.
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

    let cursor = resolve_cursor(&headers, &query)?;

    let auth_result = super::ws_auth::ws_authenticate(
        &headers,
        &state.auth_config,
        &state.orchestrator.neo4j_arc(),
        query.ticket.as_deref(),
        &state.ws_ticket_store,
    )
    .await;
    match auth_result {
        CookieAuthResult::Authenticated(claims) => {
            debug!(
                session_id = %session_id,
                email = %claims.email,
                cursor,
                "SSE chat stream authenticated"
            );
        }
        CookieAuthResult::Invalid(reason) => {
            debug!(reason = %reason, "SSE chat stream: auth rejected");
            return Err(AppError::Unauthorized(reason));
        }
    }

    // The producer owns the replay and the live loop; the response is the reader
    // end of the bounded queue.
    let (tx, rx) = mpsc::channel::<String>(STREAM_QUEUE_CAPACITY);
    let task_state = state.clone();
    let task_session = session_id.clone();
    tokio::spawn(async move {
        drive(task_state, chat_manager, task_session, cursor, tx).await;
    });

    let body = Body::from_stream(
        ReceiverStream::new(rx)
            .map(|chunk| Ok::<_, std::io::Error>(axum::body::Bytes::from(chunk))),
    );
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    let out = response.headers_mut();
    out.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    // No cache, and no proxy may buffer: an event stream that is held back is
    // indistinguishable from a silent agent.
    out.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-transform"),
    );
    out.insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(response.into_response())
}

/// Replay, then forward live events until the client leaves or the stream is
/// closed on purpose.
async fn drive(
    state: OrchestratorState,
    chat_manager: std::sync::Arc<crate::chat::manager::ChatManager>,
    session_id: String,
    cursor: i64,
    tx: mpsc::Sender<String>,
) {
    let mut sink = SseSink::new(tx, cursor);
    let caught = match catch_up(
        &chat_manager,
        state.nats_emitter.as_deref(),
        &session_id,
        cursor,
        &mut sink,
    )
    .await
    {
        Ok(caught) => caught,
        Err(_) => {
            debug!(session_id = %session_id, "SSE client gone during replay");
            return;
        }
    };
    live_loop(&chat_manager, &session_id, caught, sink).await;
}

/// Forward the conversation's live events, exactly as the socket does.
async fn live_loop(
    chat_manager: &crate::chat::manager::ChatManager,
    session_id: &str,
    caught: CatchUp,
    mut sink: SseSink,
) {
    let CatchUp {
        mut event_rx,
        mut nats_chat_sub,
        mut runtime_watch,
        mut snapshot_fingerprints,
    } = caught;

    let mut keepalive = interval(KEEPALIVE_EVERY);
    keepalive.tick().await; // skip the immediate tick
    let mut last_activity = Instant::now();

    loop {
        tokio::select! {
            biased;

            // Local broadcast of the session's runtime.
            event = async {
                match &mut event_rx {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match event {
                    Ok(chat_event) => {
                        if !deliver(&mut sink, chat_event, &mut snapshot_fingerprints).await {
                            break;
                        }
                        last_activity = Instant::now();
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        // Nothing is buffered without limit: the bus dropped the
                        // oldest events instead, and the client is told how many.
                        warn!(session_id = %session_id, skipped = n, "Chat SSE client lagged");
                        let hint = json!({ "type": "events_lagged", "skipped": n });
                        if !sink.send_frame(hint).await {
                            break;
                        }
                        last_activity = Instant::now();
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        // The runtime went away (idle cleanup). The conversation
                        // stays observable: a new runtime is picked up below.
                        debug!(session_id = %session_id, "Chat broadcast closed, going dormant");
                        event_rx = None;
                        snapshot_fingerprints.clear();
                        let msg = json!({
                            "type": "session_dormant",
                            "message": "CLI session cleaned up (idle timeout). Will resume on next message."
                        });
                        if !sink.send_frame(msg).await {
                            break;
                        }
                        last_activity = Instant::now();
                    }
                }
            }

            // A runtime opened after this connection attached: its receiver was
            // registered before it emitted anything.
            Some(next_rx) = runtime_watch.next() => {
                if event_rx.as_ref().is_none_or(|rx| !rx.same_channel(&next_rx)) {
                    event_rx = Some(next_rx);
                }
            }

            // Cross-instance events (another instance owns the session).
            nats_msg = async {
                match &mut nats_chat_sub {
                    Some(sub) => sub.next().await,
                    None => std::future::pending().await,
                }
            } => {
                match nats_msg {
                    Some(msg) => match serde_json::from_slice::<ChatEvent>(&msg.payload) {
                        Ok(chat_event) => {
                            if !deliver(&mut sink, chat_event, &mut snapshot_fingerprints).await {
                                break;
                            }
                            last_activity = Instant::now();
                        }
                        Err(e) => warn!("Failed to deserialize NATS ChatEvent: {}", e),
                    },
                    None => {
                        debug!(session_id = %session_id, "NATS chat subscriber closed");
                        nats_chat_sub = None;
                    }
                }
            }

            _ = keepalive.tick() => {
                let idle = last_activity.elapsed() >= IDLE_CLOSE;
                if idle && !chat_manager.is_session_working(session_id).await {
                    // A dormant conversation keeps its history in Neo4j: closing
                    // here costs the client one reconnect, and tells it exactly
                    // where to resume from.
                    let end = json!({
                        "type": "stream_end",
                        "reason": "dormant",
                        "cursor": sink.last_durable_seq,
                    });
                    sink.send_frame(end).await;
                    break;
                }
                if !sink.send_raw(": keep-alive\n\n".to_string()).await {
                    break;
                }
            }
        }
    }
    debug!(session_id = %session_id, "Chat SSE stream closed");
}

/// Deliver one live event, with the socket's snapshot dedup: an event already
/// sent inside the mid-stream snapshot is not sent twice.
async fn deliver(
    sink: &mut SseSink,
    chat_event: ChatEvent,
    snapshot_fingerprints: &mut std::collections::HashSet<String>,
) -> bool {
    if !snapshot_fingerprints.is_empty() {
        if let Some(fp) = chat_event.fingerprint() {
            if snapshot_fingerprints.remove(&fp) {
                debug!(
                    event_type = %chat_event.event_type(),
                    "SSE: skipping an event already in the snapshot"
                );
                return true;
            }
        }
        if matches!(&chat_event, ChatEvent::Result { .. }) {
            snapshot_fingerprints.clear();
        }
    }
    match live_frame(&chat_event, chrono::Utc::now()) {
        Ok(frame) => sink.send_frame(frame).await,
        Err(e) => {
            warn!("Failed to serialize ChatEvent: {}", e);
            true
        }
    }
}
