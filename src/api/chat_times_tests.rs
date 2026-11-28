//! Server times on the chat wire, over a real server on a loopback port: the
//! REST history and the WebSocket frames (replayed and live) carry
//! `created_at` in seconds since the epoch with the milliseconds as the
//! fraction. Each test fails if a path drops the field or rounds it to the
//! second.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

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
#[allow(unused_imports)]
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

// MCP mutations must appear on the same sockets used by the browser.
mod mcp_realtime {
    use std::sync::Arc;
    use std::time::Duration;

    use futures::{SinkExt, StreamExt};
    use nexus_claude::agent::{AgentEvent, StopReason};
    use serde_json::{json, Value};
    use tokio_tungstenite::tungstenite::Message;

    use crate::chat::agent_runtime::fake::FakeProvider;
    use crate::chat::config::{ChatConfig, ProviderPath};
    use crate::chat::manager::ChatManager;
    use crate::events::{nats_broker_test::TestBroker, EventBus, HybridEmitter, NatsEmitter};
    use crate::mcp::{handlers::ToolHandler, McpHttpClient};
    use crate::neo4j::mock::MockGraphStore;
    use crate::test_helpers::{mock_app_state_with_graph, server_state_with_chat};

    use super::{Client, WAIT};

    async fn through_frame(
        ws: &mut Client,
        description: &str,
        predicate: impl Fn(&Value) -> bool,
    ) -> Vec<Value> {
        let mut frames = Vec::new();
        tokio::time::timeout(WAIT, async {
            while let Some(frame) = ws.next().await {
                if let Message::Text(text) = frame.unwrap() {
                    let value: Value = serde_json::from_str(text.as_str()).unwrap();
                    let matched = predicate(&value);
                    frames.push(value);
                    if matched {
                        return;
                    }
                }
            }
            panic!("socket closed waiting for {description}: {frames:?}");
        })
        .await
        .unwrap_or_else(|_| panic!("no {description} on the same socket: {frames:?}"));
        frames
    }

    async fn connect(addr: std::net::SocketAddr, sid: &str) -> Client {
        let url = format!("ws://{addr}/ws/chat/{sid}?last_event=999999999999999");
        let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        ws.send(Message::text("ready")).await.unwrap();
        through_frame(&mut ws, "replay_complete", |v| {
            v["type"] == "replay_complete"
        })
        .await;
        ws
    }

    fn finish_turn(provider: &FakeProvider) {
        provider.state.push(AgentEvent::Done {
            stop_reason: StopReason::Completed,
            subtype: Some("success".into()),
            is_error: false,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 5,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: Some("test-mcp-chat".into()),
            structured_output: None,
            error: None,
        });
    }

    fn turn_finished(frame: &Value) -> bool {
        frame["type"] == "streaming_status" && frame["is_streaming"] == false
    }

    async fn mcp_sends_to_dormant_conversation(nats: Option<Arc<NatsEmitter>>) {
        let graph = Arc::new(MockGraphStore::new());
        let provider = FakeProvider::new();
        let event_bus = Arc::new(HybridEmitter::new(Arc::new(EventBus::default())));
        let config = ChatConfig {
            provider_path: ProviderPath::Agent,
            nexus_tools_path: None,
            nexus_browser_path: None,
            obscura_path: None,
            max_sessions: 10,
            ..Default::default()
        };
        let manager = ChatManager::new_without_memory(
            graph.clone(),
            mock_app_state_with_graph(graph.clone()).meili,
            config,
        )
        .with_provider_source(Arc::new(provider.clone()))
        .with_claude_cli_present(true)
        .with_event_emitter(event_bus.clone());
        let manager = Arc::new(match &nats {
            Some(nats) => manager.with_nats(Arc::clone(nats)),
            None => manager,
        });
        let mut state = server_state_with_chat(manager.clone(), graph).await;
        let state_mut = Arc::get_mut(&mut state).unwrap();
        state_mut.event_bus = event_bus.clone();
        state_mut.nats_emitter = nats;
        let router = crate::api::create_router(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let (mut events, _) = tokio_tungstenite::connect_async(format!(
            "ws://{addr}/ws/events?entity_types=chat_session"
        ))
        .await
        .unwrap();
        events.send(Message::text("ready")).await.unwrap();
        through_frame(&mut events, "event socket authentication", |v| {
            v["type"] == "auth_ok"
        })
        .await;
        // The handler finishes subscribing just after sending the handshake frame.
        tokio::time::timeout(WAIT, async {
            while event_bus.subscriber_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the event socket is subscribed before the MCP call");

        let mcp = ToolHandler::new(McpHttpClient::new(format!("http://{addr}"), None));
        let cwd = tempfile::tempdir().unwrap();
        let created = mcp
            .handle(
                "chat",
                Some(json!({"action": "send_message", "message": "First MCP turn", "cwd": cwd.path()})),
            )
            .await
            .unwrap();
        let sid = created["session_id"].as_str().expect("MCP session id");
        let created_frames = through_frame(&mut events, "MCP conversation creation", |v| {
            v["entity_type"] == "chat_session" && v["action"] == "created"
        })
        .await;
        assert_eq!(created_frames.last().unwrap()["entity_id"], sid);
        let mut open = connect(addr, sid).await;
        finish_turn(&provider);
        through_frame(&mut open, "first turn's completion", turn_finished).await;

        assert_eq!(manager.idle_cleanup_pass(Duration::ZERO).await, [sid]);
        assert!(!manager.is_session_active(sid).await);
        let mut dormant = connect(addr, sid).await;

        let sent = mcp
            .handle(
                "chat",
                Some(json!({"action": "send_message", "session_id": sid, "message": "MCP follow-up"})),
            )
            .await
            .unwrap();
        assert_eq!(sent["session_id"], sid);
        provider.state.push(AgentEvent::Text {
            text: "MCP answer".into(),
            seq: None,
            parent: None,
        });
        finish_turn(&provider);

        let (open_frames, dormant_frames) = tokio::join!(
            through_frame(&mut open, "follow-up on the original socket", turn_finished),
            through_frame(
                &mut dormant,
                "follow-up on the dormant socket",
                turn_finished
            ),
        );
        for frames in [open_frames, dormant_frames] {
            let turn: Vec<_> = frames
                .iter()
                .filter(|frame| {
                    frame["content"] == "MCP follow-up"
                        || frame["content"] == "MCP answer"
                        || frame["type"] == "result"
                })
                .map(|frame| frame["type"].as_str().unwrap())
                .collect();
            assert_eq!(
                turn,
                ["user_message", "assistant_text", "result"],
                "{frames:?}"
            );
        }
    }

    #[tokio::test]
    async fn mcp_follow_up_reaches_open_and_dormant_chat_sockets_without_nats() {
        mcp_sends_to_dormant_conversation(None).await;
    }

    #[tokio::test]
    async fn mcp_follow_up_reaches_open_and_dormant_chat_sockets_with_nats() {
        let broker = TestBroker::start().await;
        let nats = Arc::new(NatsEmitter::new(broker.client().await, "events"));
        mcp_sends_to_dormant_conversation(Some(nats)).await;
    }
}
