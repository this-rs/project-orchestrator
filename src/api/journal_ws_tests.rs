//! Journal reconnect over the real chat WebSocket: the handler replays a
//! session's persisted events from the `JournalTransport`, not from Neo4j, and
//! a cursor below the journal's retained window is a typed error frame — never
//! a silently amputated stream.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::chat::manager::{test_support, ChatManager};
use crate::chat::types::ChatEvent;
use crate::events::chat_transport::ChatTransport;
use crate::events::JournalTransport;
#[allow(unused_imports)]
use crate::test_helpers::{
    mock_app_state_with_graph, serve_chat_with_transport, test_chat_session,
};

const WAIT: Duration = Duration::from_secs(10);

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

struct Rig {
    journal: JournalTransport,
    sid: String,
    addr: std::net::SocketAddr,
}

/// A server (no-auth mode) over a mock graph holding one chat session, whose
/// chat transport is a journal with `max_events` retention.
async fn rig(max_events: usize) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let journal = JournalTransport::open(dir.path().join("chat-journal.log"), max_events).unwrap();

    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let session = test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();

    let app_state = mock_app_state_with_graph(graph.clone());
    let manager = Arc::new(ChatManager::new_without_memory(
        graph.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    let transport: Arc<dyn crate::events::ChatTransport> = Arc::new(journal.clone());
    let addr = serve_chat_with_transport(manager, graph, Some(transport)).await;

    Rig {
        journal,
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

/// The next JSON text frame, in order (no filtering).
async fn next_json(ws: &mut Client) -> Value {
    loop {
        let msg = tokio::time::timeout(WAIT, ws.next())
            .await
            .expect("a frame within 10 s")
            .expect("the socket is open")
            .expect("a frame");
        if let WsMessage::Text(t) = msg {
            if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                return v;
            }
        }
    }
}

async fn frame_where(ws: &mut Client, pred: impl Fn(&Value) -> bool) -> Value {
    loop {
        let v = next_json(ws).await;
        if pred(&v) {
            return v;
        }
    }
}

fn of_type(t: &'static str) -> impl Fn(&Value) -> bool {
    move |v| v["type"] == t
}

/// Every replayed (`replaying: true`) frame up to `replay_complete`, in order.
async fn replayed_frames(ws: &mut Client) -> Vec<Value> {
    let mut all = Vec::new();
    loop {
        let v = next_json(ws).await;
        if v["type"] == "replay_complete" {
            break;
        }
        all.push(v);
    }
    all.into_iter().filter(|v| v["replaying"] == true).collect()
}

#[tokio::test]
async fn journal_reconnects_replay_the_sessions_persisted_events_in_order() {
    let rig = rig(1_000).await;

    rig.journal.publish_chat_event(
        &rig.sid,
        ChatEvent::UserMessage {
            content: "one".into(),
        },
    );
    rig.journal.publish_chat_event(
        &rig.sid,
        ChatEvent::AssistantText {
            content: "two".into(),
            parent_tool_use_id: None,
        },
    );
    // An event of another session shares the journal's cursor space but must
    // never leak into this session's replay.
    rig.journal.publish_chat_event(
        "another-session",
        ChatEvent::UserMessage {
            content: "not-mine".into(),
        },
    );

    let mut ws = connect(&rig, 0).await;
    let auth = frame_where(&mut ws, of_type("auth_ok")).await;
    assert!(auth["user"]["email"].is_string());

    let replayed = replayed_frames(&mut ws).await;
    assert_eq!(replayed.len(), 2, "{replayed:#?}");
    assert_eq!(replayed[0]["type"], "user_message");
    assert_eq!(replayed[0]["content"], "one");
    assert_eq!(replayed[0]["seq"], json!(1));
    assert_eq!(replayed[1]["type"], "assistant_text");
    assert_eq!(replayed[1]["content"], "two");
    assert_eq!(replayed[1]["seq"], json!(2));
}

#[tokio::test]
async fn journal_reconnects_replay_only_events_after_the_cursor() {
    let rig = rig(1_000).await;

    rig.journal.publish_chat_event(
        &rig.sid,
        ChatEvent::UserMessage {
            content: "one".into(),
        },
    );
    rig.journal.publish_chat_event(
        &rig.sid,
        ChatEvent::AssistantText {
            content: "two".into(),
            parent_tool_use_id: None,
        },
    );

    // The client already holds cursor 1: only the event after it is replayed.
    let mut ws = connect(&rig, 1).await;
    frame_where(&mut ws, of_type("auth_ok")).await;

    let replayed = replayed_frames(&mut ws).await;
    assert_eq!(replayed.len(), 1, "{replayed:#?}");
    assert_eq!(replayed[0]["type"], "assistant_text");
    assert_eq!(replayed[0]["content"], "two");
    assert_eq!(replayed[0]["seq"], json!(2));
}

#[tokio::test]
async fn journal_cursor_below_the_window_is_a_typed_error_frame() {
    let rig = rig(5).await;

    for i in 0..10u32 {
        rig.journal.publish_chat_event(
            &rig.sid,
            ChatEvent::UserMessage {
                content: format!("event-{i}"),
            },
        );
    }
    // Retention 5 keeps cursors 6..10: a client asking for 0 is below the window.
    assert_eq!(rig.journal.oldest_cursor(), Some(6));

    let mut ws = connect(&rig, 0).await;
    frame_where(&mut ws, of_type("auth_ok")).await;

    let err = frame_where(&mut ws, of_type("error")).await;
    assert_eq!(err["code"], "cursor_too_old", "{err}");
    assert_eq!(err["oldest"], json!(6));
    assert_eq!(err["requested"], json!(0));
    assert!(err["message"].as_str().unwrap().contains("oldest"), "{err}");
}

#[tokio::test]
async fn journal_reconnects_after_a_restart_replays_events_read_back_from_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chat-journal.log");

    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    let session = test_chat_session(None);
    graph.create_chat_session(&session).await.unwrap();
    let sid = session.id.to_string();

    // First "run": the journal persists events, then everything is dropped —
    // only the file on disk survives (a server restart leaves no in-memory
    // state behind).
    {
        let journal = JournalTransport::open(&path, 1_000).unwrap();
        journal.publish_chat_event(
            &sid,
            ChatEvent::UserMessage {
                content: "one".into(),
            },
        );
        journal.publish_chat_event(
            &sid,
            ChatEvent::AssistantText {
                content: "two".into(),
                parent_tool_use_id: None,
            },
        );
    }

    // Second "run": a fresh journal reopens the same file (cursors are read
    // back from disk, never reused) and a fresh server serves the reconnect
    // over the real WebSocket route.
    let journal = JournalTransport::open(&path, 1_000).unwrap();
    assert_eq!(
        journal.next_cursor(),
        3,
        "cursors continue across a restart"
    );

    let app_state = mock_app_state_with_graph(graph.clone());
    let manager = Arc::new(ChatManager::new_without_memory(
        graph.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    let transport: Arc<dyn crate::events::ChatTransport> = Arc::new(journal);
    let addr = serve_chat_with_transport(manager, graph, Some(transport)).await;

    let url = format!("ws://{addr}/ws/chat/{sid}?last_event=0");
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    ws.send(WsMessage::text("ready")).await.unwrap();

    frame_where(&mut ws, of_type("auth_ok")).await;
    let replayed = replayed_frames(&mut ws).await;
    assert_eq!(replayed.len(), 2, "{replayed:#?}");
    assert_eq!(replayed[0]["type"], "user_message");
    assert_eq!(replayed[0]["content"], "one");
    assert_eq!(replayed[0]["seq"], json!(1));
    assert_eq!(replayed[1]["type"], "assistant_text");
    assert_eq!(replayed[1]["content"], "two");
    assert_eq!(replayed[1]["seq"], json!(2));
}
