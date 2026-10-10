//! Server times on the chat wire, over a real server on a loopback port: the
//! REST history and the WebSocket frames (replayed and live) carry
//! `created_at` in seconds since the epoch with the milliseconds as the
//! fraction. Each test fails if a path drops the field or rounds it to the
//! second.

use std::sync::Arc;
use std::time::Duration;

use chrono::TimeZone;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use uuid::Uuid;

use crate::chat::manager::{test_support, ChatManager};
use crate::chat::types::ChatEvent;
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::GraphStore;
use crate::test_helpers::{mock_app_state_with_graph, test_chat_session};

const WAIT: Duration = Duration::from_secs(10);
/// 2025-10-10T12:39:59.123Z: a stored time whose milliseconds are not zero.
const STORED_MS: i64 = 1_760_099_999_123;

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct Rig {
    manager: Arc<ChatManager>,
    sid: String,
    addr: std::net::SocketAddr,
}

/// A server (no-auth mode) over a store holding one session with one event
/// stored at `STORED_MS`.
async fn rig() -> Rig {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let session = test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();
    graph
        .store_chat_events(
            session.id,
            vec![ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: session.id,
                seq: 1,
                event_type: "assistant_text".into(),
                data: json!({"type": "assistant_text", "content": "stocké"}).to_string(),
                created_at: chrono::Utc.timestamp_millis_opt(STORED_MS).unwrap(),
            }],
        )
        .await
        .unwrap();
    let app_state = mock_app_state_with_graph(graph.clone());
    let manager = Arc::new(ChatManager::new_without_memory(
        graph.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    let addr = crate::test_helpers::serve_chat(manager.clone(), graph).await;
    Rig {
        manager,
        sid: session.id.to_string(),
        addr,
    }
}

async fn connect(rig: &Rig, last_event: i64) -> Client {
    let url = format!(
        "ws://{}/ws/chat/{}?last_event={last_event}",
        rig.addr, rig.sid
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    ws.send(WsMessage::text("ready")).await.unwrap();
    ws
}

/// The next JSON frame satisfying `pred`.
async fn frame_where(ws: &mut Client, pred: impl Fn(&Value) -> bool) -> Value {
    loop {
        let msg = tokio::time::timeout(WAIT, ws.next())
            .await
            .expect("a frame within 10 s")
            .expect("the socket is open")
            .expect("a frame");
        if let WsMessage::Text(t) = msg {
            if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                if pred(&v) {
                    return v;
                }
            }
        }
    }
}

fn seconds(ms: i64) -> f64 {
    ms as f64 / 1000.0
}

#[tokio::test]
async fn history_created_at_keeps_the_milliseconds() {
    let rig = rig().await;
    let body: Value = reqwest::get(format!(
        "http://{}/api/chat/sessions/{}/messages",
        rig.addr, rig.sid
    ))
    .await
    .unwrap()
    .json()
    .await
    .unwrap();
    let event = &body["messages"][0];
    assert_eq!(event["content"], "stocké", "{body}");
    assert_eq!(
        event["created_at"].as_f64(),
        Some(seconds(STORED_MS)),
        "{body}"
    );
}

#[tokio::test]
async fn replayed_frames_carry_the_stored_time_with_the_milliseconds() {
    let rig = rig().await;
    let mut ws = connect(&rig, 0).await;
    let frame = frame_where(&mut ws, |v| v["type"] == "assistant_text").await;
    assert_eq!(frame["replaying"], true, "{frame}");
    assert_eq!(frame["seq"], 1, "{frame}");
    assert_eq!(
        frame["created_at"].as_f64(),
        Some(seconds(STORED_MS)),
        "{frame}"
    );
}

#[tokio::test]
async fn live_frames_carry_the_forwarding_time_in_seconds() {
    let rig = rig().await;
    test_support::insert_live_session_without_cli(&rig.manager, &rig.sid).await;
    let mut ws = connect(&rig, 999_999_999_999_999).await;
    frame_where(&mut ws, |v| v["type"] == "replay_complete").await;

    let before = chrono::Utc::now().timestamp_millis();
    rig.manager
        .get_events_tx(&rig.sid)
        .await
        .unwrap()
        .send(ChatEvent::AssistantText {
            content: "en direct".into(),
            parent_tool_use_id: None,
        })
        .unwrap();
    let frame = frame_where(&mut ws, |v| v["content"] == "en direct").await;
    let after = chrono::Utc::now().timestamp_millis();

    let at = frame["created_at"]
        .as_f64()
        .unwrap_or_else(|| panic!("a live frame carries created_at: {frame}"));
    assert!(
        (seconds(before)..=seconds(after)).contains(&at),
        "{at} not within [{before}, {after}] ms: {frame}"
    );
    assert_eq!(frame["seq"], 0, "{frame}");
}
