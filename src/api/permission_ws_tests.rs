//! Permission answers over the real chat WebSocket (P11b, review of #699): a server on a
//! loopback port (authentication off: the socket carries the anonymous user), a real
//! client, the real handler. Each test fails if the loop stops checking who answers
//! before anything is delivered, or lets an answer through when the owner cannot be read.

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::chat::manager::{test_support, ChatManager};
use crate::neo4j::models::ChatSessionNode;
use crate::neo4j::GraphStore;
use crate::test_helpers::{mock_app_state_with_graph, test_chat_session};

const WAIT: Duration = Duration::from_secs(10);

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// A server over a store holding `session`, and a client connected to it.
async fn connected(session: &ChatSessionNode) -> (Arc<crate::neo4j::mock::MockGraphStore>, Client) {
    let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
    graph.create_chat_session(session).await.unwrap();
    let app_state = mock_app_state_with_graph(graph.clone());
    let manager = Arc::new(ChatManager::new_without_memory(
        graph.clone(),
        app_state.meili.clone(),
        test_support::chat_config(),
    ));
    let addr = crate::test_helpers::serve_chat(manager, graph.clone()).await;
    let url = format!(
        "ws://{addr}/ws/chat/{}?last_event=999999999999999",
        session.id
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
    ws.send(WsMessage::text("ready")).await.unwrap();
    frame_where(&mut ws, |v| v["type"] == "auth_ok").await;
    (graph, ws)
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

async fn answer(ws: &mut Client, request_id: &str) -> Value {
    let frame =
        json!({"type": "permission_response", "id": request_id, "allow": true, "scope": "session"});
    ws.send(WsMessage::text(frame.to_string())).await.unwrap();
    frame_where(ws, |v| {
        v["type"] == "error" && v["code"] == "permission_forbidden"
    })
    .await
}

/// Another person's answer (here the anonymous user of a server without authentication,
/// on a session someone owns) is refused by the loop itself, naming its request.
#[tokio::test]
async fn another_persons_answer_is_refused_by_the_websocket_loop() {
    let session = ChatSessionNode {
        owner: Some(uuid::Uuid::new_v4().to_string()),
        ..test_chat_session(None)
    };
    let (_graph, mut ws) = connected(&session).await;
    let refused = answer(&mut ws, "pr_owned").await;
    assert_eq!(refused["reason"], "not_owner", "{refused}");
    assert_eq!(refused["request_id"], "pr_owned", "{refused}");
}

/// The owner cannot be read (the graph fails): the loop refuses, it never lets the
/// answer reach the routing (which does not check the owner again).
#[tokio::test]
async fn an_answer_is_refused_by_the_websocket_loop_when_the_owner_cannot_be_read() {
    let session = test_chat_session(None);
    let (graph, mut ws) = connected(&session).await;
    graph.fail_reads.lock().unwrap().insert("get_chat_session");
    let refused = answer(&mut ws, "pr_unread").await;
    assert_eq!(refused["reason"], "owner_unreadable", "{refused}");
    assert_eq!(refused["request_id"], "pr_unread", "{refused}");
}
