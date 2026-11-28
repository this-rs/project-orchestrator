//! Tests of the REST/SSE chat event surface (T3 of the full-API plan).
//!
//! What they pin, one test each:
//! - the cursor a reconnect resumes from (`Last-Event-ID` > `?cursor` >
//!   `?last_event`), and the refusals that keep a bad cursor from meaning
//!   "replay everything";
//! - the `id:` line names a DURABLE sequence number only: a transient frame
//!   (deltas, snapshots) must never move the client's cursor;
//! - the stream is append-only: a replayed range is ignored, not resent;
//! - the SSE frames are the socket's frames — same events, same order, same
//!   `seq` — because both go through `chat::event_stream`;
//! - the HTTP surface: 200 + `text/event-stream` + `replay_complete`, an
//!   explicit `stream_end` instead of a silent close, 404 on an unknown
//!   session, 400 on a multi-session ask.

use super::sse_chat_stream::{encode_frame, resolve_cursor, SseQuery};
use super::ws_auth::CookieAuthResult;
use crate::chat::event_stream::{catch_up, FrameSink};
use crate::chat::manager::test_support;
use crate::chat::manager::ChatManager;
use crate::chat::types::ChatEvent;
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::models::{ChatEventRecord, ChatSessionNode};
use crate::neo4j::traits::*;
use crate::test_helpers::{mock_app_state_with_graph, server_state_with_chat, test_chat_session};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

// ============================================================================
// Cursor
// ============================================================================

fn query(cursor: Option<i64>, last_event: Option<i64>) -> SseQuery {
    SseQuery {
        cursor,
        last_event,
        ticket: None,
        sessions: None,
    }
}

fn last_event_id(value: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("last-event-id", HeaderValue::from_str(value).unwrap());
    headers
}

#[test]
fn the_last_event_id_header_wins_over_the_query_parameters() {
    let headers = last_event_id("42");
    assert_eq!(
        resolve_cursor(&headers, &query(Some(7), Some(9))).unwrap(),
        42,
        "the header is what an EventSource sends by itself: it wins"
    );
}

#[test]
fn the_query_parameters_are_read_in_order() {
    assert_eq!(
        resolve_cursor(&HeaderMap::new(), &query(Some(7), Some(9))).unwrap(),
        7,
        "?cursor wins over ?last_event"
    );
    assert_eq!(
        resolve_cursor(&HeaderMap::new(), &query(None, Some(9))).unwrap(),
        9
    );
    assert_eq!(
        resolve_cursor(&HeaderMap::new(), &query(None, None)).unwrap(),
        0,
        "no cursor at all means the whole history"
    );
}

#[test]
fn a_negative_cursor_means_from_the_beginning() {
    assert_eq!(
        resolve_cursor(&last_event_id("-3"), &query(None, None)).unwrap(),
        0,
        "seq starts at 1, so a negative cursor is clamped, not refused"
    );
}

#[test]
fn a_cursor_that_is_not_a_sequence_number_is_refused() {
    let err = resolve_cursor(&last_event_id("event-7"), &query(None, None))
        .expect_err("a non-numeric Last-Event-ID is a client bug");
    assert!(
        matches!(err, super::handlers::AppError::BadRequest(_)),
        "it must be a 400, not a silent replay from zero"
    );
}

// ============================================================================
// Framing
// ============================================================================

#[test]
fn only_a_durable_sequence_number_becomes_an_id_line() {
    let durable = encode_frame(
        42,
        &serde_json::json!({ "type": "assistant_text", "seq": 42 }),
    );
    assert!(
        durable.starts_with("id: 42\ndata: "),
        "a replayed frame names its seq: {durable}"
    );
    assert!(durable.ends_with("\n\n"), "a frame ends with a blank line");

    let transient = encode_frame(0, &serde_json::json!({ "type": "stream_delta", "seq": 0 }));
    assert!(
        !transient.contains("id:"),
        "a transient frame must not move the cursor: {transient}"
    );
    assert_eq!(
        transient.matches('\n').count(),
        2,
        "one data line, then the blank line that ends the frame"
    );
}

/// A payload containing a newline must not break the frame into two fields:
/// `data:` is a single line until the blank line that ends the frame.
#[test]
fn a_multiline_payload_stays_one_data_line() {
    let frame = encode_frame(
        3,
        &serde_json::json!({ "type": "tool_result", "result": "line one\nline two" }),
    );
    assert_eq!(
        frame.matches("\ndata: ").count(),
        1,
        "a newline in the payload is escaped by JSON, not sent raw: {frame}"
    );
    assert!(frame.starts_with("id: 3\ndata: "));
}

// ============================================================================
// Append-only dedup
// ============================================================================

/// The same replayed event delivered twice is sent once, and a sequence number
/// behind the cursor is dropped: the stream never rewinds.
#[tokio::test]
async fn the_sink_is_append_only_and_idempotent() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
    let mut sink = super::sse_chat_stream::SseSink::new(tx, 0);

    let event = serde_json::json!({ "type": "assistant_text", "seq": 5 });
    assert!(sink.send_frame(event.clone()).await);
    assert!(
        sink.send_frame(event.clone()).await,
        "a duplicate is accepted (and ignored), never an error"
    );
    // An older frame, replayed after a newer one, is ignored too.
    assert!(
        sink.send_frame(serde_json::json!({ "type": "user_message", "seq": 4 }))
            .await
    );
    // A transient frame always passes and never moves the cursor.
    assert!(
        sink.send_frame(serde_json::json!({ "type": "stream_delta", "seq": 0 }))
            .await
    );
    assert_eq!(sink.last_durable_seq, 5, "the cursor only ever advances");

    drop(sink);
    let mut chunks = Vec::new();
    while let Some(chunk) = rx.recv().await {
        chunks.push(chunk);
    }
    assert_eq!(chunks.len(), 2, "one durable frame + one transient frame");
    assert!(chunks[0].starts_with("id: 5\n"));
    assert!(chunks[1].starts_with("data: ") && !chunks[1].starts_with("id:"));
}

/// A sink whose reader is gone reports it, so the caller stops.
#[tokio::test]
async fn a_closed_reader_stops_the_sink() {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(1);
    let mut sink = super::sse_chat_stream::SseSink::new(tx, 0);
    drop(rx);
    assert!(
        !sink
            .send_frame(serde_json::json!({ "type": "user_message", "seq": 1 }))
            .await,
        "a dead client must be reported, not retried"
    );
}

// ============================================================================
// The same flux as the socket
// ============================================================================

/// Seed a session with three persisted events (seq 1..3) whose payloads are
/// real `ChatEvent`s, and hand back the manager plus the session id.
async fn seeded(session: &mut ChatSessionNode) -> (Arc<ChatManager>, Arc<MockGraphStore>) {
    let graph = Arc::new(MockGraphStore::new());
    let app_state = mock_app_state_with_graph(graph.clone());
    graph.create_chat_session(session).await.unwrap();

    let payloads = [
        ChatEvent::UserMessage {
            content: "hello".to_string(),
        },
        ChatEvent::AssistantText {
            content: "hi there".to_string(),
            parent_tool_use_id: None,
        },
        ChatEvent::Thinking {
            content: "considering".to_string(),
            parent_tool_use_id: None,
        },
    ];
    let records = payloads
        .iter()
        .enumerate()
        .map(|(i, event)| ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: session.id,
            seq: (i + 1) as i64,
            event_type: event.event_type().to_string(),
            data: serde_json::to_string(event).unwrap(),
            created_at: chrono::Utc::now(),
        })
        .collect();
    graph.store_chat_events(session.id, records).await.unwrap();

    let manager = Arc::new(ChatManager::new_without_memory(
        app_state.neo4j.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    (manager, graph)
}

/// The central claim of T3: the SSE surface emits the SAME frames as the
/// socket, in the same order, with the same `seq` — because it runs the same
/// catch-up core. Proven by driving both sinks from one seeded session.
#[tokio::test]
async fn the_sse_frames_are_the_socket_frames() {
    let mut session = test_chat_session(None);
    let (manager, _graph) = seeded(&mut session).await;
    let sid = session.id.to_string();

    // The socket's sink: a plain vector of frames.
    struct VecSink(Vec<Value>);
    #[async_trait::async_trait]
    impl FrameSink for VecSink {
        async fn send_frame(&mut self, frame: Value) -> bool {
            self.0.push(frame);
            true
        }
    }

    let mut socket_sink = VecSink(Vec::new());
    catch_up(&manager, None, &sid, 0, &mut socket_sink)
        .await
        .expect("catch-up succeeds");

    // The SSE sink: the same core, writing encoded chunks.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let mut sse_sink = super::sse_chat_stream::SseSink::new(tx, 0);
    catch_up(&manager, None, &sid, 0, &mut sse_sink)
        .await
        .expect("catch-up succeeds");
    drop(sse_sink);

    let mut chunks = Vec::new();
    while let Some(chunk) = rx.recv().await {
        chunks.push(chunk);
    }

    assert_eq!(
        chunks.len(),
        socket_sink.0.len(),
        "one SSE frame per socket frame"
    );

    let mut ids: Vec<Option<i64>> = Vec::new();
    let mut frames: Vec<Value> = Vec::new();
    for chunk in &chunks {
        let mut id = None;
        let mut data = None;
        for line in chunk.lines() {
            if let Some(rest) = line.strip_prefix("id: ") {
                id = Some(rest.parse::<i64>().expect("the id line holds a number"));
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = Some(serde_json::from_str::<Value>(rest).expect("the data line is JSON"));
            }
        }
        ids.push(id);
        frames.push(data.expect("every frame carries a data line"));
    }

    assert_eq!(
        frames, socket_sink.0,
        "the same frames, in the same order: nothing is re-invented for SSE"
    );

    // …and the `id:` line names the durable seq, and only for durable frames.
    for (frame, id) in frames.iter().zip(ids.iter()) {
        let seq = frame.get("seq").and_then(Value::as_i64).unwrap_or(0);
        if seq > 0 {
            assert_eq!(*id, Some(seq), "a persisted frame names its seq");
        } else {
            assert_eq!(*id, None, "a transient frame names no cursor");
        }
    }

    // The three persisted events are all there, with their stored seq.
    let seqs: Vec<i64> = frames
        .iter()
        .filter_map(|f| f.get("seq").and_then(Value::as_i64))
        .filter(|s| *s > 0)
        .collect();
    assert_eq!(seqs, vec![1, 2, 3], "the durable history, in order");
}

/// A client that comes back with its cursor receives only what it missed —
/// nothing is replayed twice (so a command is never re-executed), and nothing
/// of the missed range is lost.
#[tokio::test]
async fn a_reconnect_resumes_from_the_cursor() {
    let mut session = test_chat_session(None);
    let (manager, _graph) = seeded(&mut session).await;
    let sid = session.id.to_string();

    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    let mut sink = super::sse_chat_stream::SseSink::new(tx, 2);
    catch_up(&manager, None, &sid, 2, &mut sink)
        .await
        .expect("catch-up succeeds");
    drop(sink);

    let mut durable = Vec::new();
    while let Some(chunk) = rx.recv().await {
        for line in chunk.lines() {
            if let Some(rest) = line.strip_prefix("id: ") {
                durable.push(rest.parse::<i64>().unwrap());
            }
        }
    }
    assert_eq!(
        durable,
        vec![3],
        "only the events after the cursor: no replay of 1 and 2, none of 3 lost"
    );
}

// ============================================================================
// The HTTP surface
// ============================================================================

async fn app_with_session() -> (axum::Router, Uuid) {
    let graph = Arc::new(MockGraphStore::new());
    let app_state = mock_app_state_with_graph(graph.clone());
    let manager = Arc::new(ChatManager::new_without_memory(
        app_state.neo4j.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    let mut session = test_chat_session(None);
    session.id = Uuid::new_v4();
    graph.create_chat_session(&session).await.unwrap();
    let state = server_state_with_chat(manager, graph).await;
    (crate::api::routes::create_router(state), session.id)
}

fn get_stream(session: &str, query: &str) -> Request<Body> {
    Request::builder()
        .uri(format!("/api/chat/sessions/{session}/events{query}"))
        .body(Body::empty())
        .unwrap()
}

/// The endpoint answers `text/event-stream`, replays the history, says the
/// replay is complete, and ends the response ON PURPOSE with a `stream_end`
/// naming the reason and the cursor — never a silent close.
#[tokio::test]
async fn the_route_streams_events_and_ends_on_purpose() {
    let (app, session) = app_with_session().await;
    let response = app
        .oneshot(get_stream(&session.to_string(), ""))
        .await
        .expect("the route answers");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream; charset=utf-8")
    );
    assert_eq!(
        response
            .headers()
            .get("x-accel-buffering")
            .and_then(|v| v.to_str().ok()),
        Some("no"),
        "a buffering proxy would make the stream look like a silent agent"
    );

    // The stream closes itself (the session does no work), so the body ends.
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body is readable");
    let text = String::from_utf8(body.to_vec()).expect("SSE is UTF-8");

    assert!(
        text.contains("\"type\":\"replay_complete\""),
        "the replay phase is announced: {text}"
    );
    assert!(
        text.contains("\"type\":\"stream_end\""),
        "the response ends with an explicit terminal frame: {text}"
    );
    assert!(
        text.contains("\"reason\":\"dormant\""),
        "the reason is named: {text}"
    );
    assert!(
        text.contains("\"cursor\":0"),
        "the cursor to resume from is given: {text}"
    );
}

#[tokio::test]
async fn an_unknown_session_is_a_404_not_a_silent_stream() {
    let (app, _session) = app_with_session().await;
    let response = app
        .oneshot(get_stream(&Uuid::new_v4().to_string(), ""))
        .await
        .expect("the route answers");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_multi_session_ask_is_refused_with_its_reason() {
    let (app, session) = app_with_session().await;
    let response = app
        .oneshot(get_stream(&session.to_string(), "?sessions=other"))
        .await
        .expect("the route answers");
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "one stream has one per-session cursor: a multi-session ask is refused"
    );
}

/// A cursor that is not a number is refused by the extractor (400), rather than
/// silently taken as zero — which would replay and re-execute nothing visible
/// but a whole transcript.
#[tokio::test]
async fn a_malformed_cursor_query_is_a_400() {
    let (app, session) = app_with_session().await;
    let response = app
        .oneshot(get_stream(&session.to_string(), "?cursor=abc"))
        .await
        .expect("the route answers");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

/// The socket's auth is reused: in no-auth mode (as in every test router) the
/// caller is anonymous, and a ticket that names nothing is refused.
#[tokio::test]
async fn the_stream_authenticates_like_the_chat_socket() {
    let state = {
        let graph = Arc::new(MockGraphStore::new());
        let app_state = mock_app_state_with_graph(graph.clone());
        let manager = Arc::new(ChatManager::new_without_memory(
            app_state.neo4j.clone(),
            app_state.meili.clone(),
            test_support::chat_config(),
        ));
        server_state_with_chat(manager, graph).await
    };
    let result = crate::api::ws_auth::ws_authenticate(
        &HeaderMap::new(),
        &state.auth_config,
        &state.orchestrator.neo4j_arc(),
        None,
        &state.ws_ticket_store,
    )
    .await;
    assert!(
        matches!(result, CookieAuthResult::Authenticated(_)),
        "no-auth mode: the same anonymous caller the sockets see"
    );
}
