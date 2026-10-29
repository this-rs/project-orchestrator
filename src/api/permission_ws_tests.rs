//! Permission answers over the real chat WebSocket (P11b, review of #699): a server on a
//! loopback port (authentication off: the socket carries the anonymous user), a real
//! client, the real handler. Each test fails if the loop stops checking who answers
//! before anything is delivered, or lets an answer through when the owner cannot be read.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::chat::manager::{test_support, ChatManager};
use crate::neo4j::models::ChatSessionNode;
#[allow(unused_imports)]
use crate::neo4j::traits::*;
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
        file_scope_lifted: false,
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

/// Review of #714, S1: an `always` answer from credentials that may not keep a lasting
/// approval (here the anonymous user of a server without authentication, on a session
/// nobody owns, which it may otherwise answer) is refused by the loop, naming its request;
/// nothing reaches the routing.
#[tokio::test]
async fn an_always_answer_from_a_non_person_is_refused_by_the_websocket_loop() {
    let session = test_chat_session(None);
    let (_graph, mut ws) = connected(&session).await;
    let frame =
        json!({"type": "permission_response", "id": "pr_always", "allow": true, "scope": "always"});
    ws.send(WsMessage::text(frame.to_string())).await.unwrap();
    let refused = frame_where(&mut ws, |v| {
        v["type"] == "error" && v["code"] == "permission_forbidden"
    })
    .await;
    assert_eq!(refused["reason"], "not_a_person", "{refused}");
    assert_eq!(refused["request_id"], "pr_always", "{refused}");
}

/// The predicate the loop applies: an MCP token, an agent's, the server's account or the
/// anonymous user never answer `always`; a person's session token does; `once` / `session`
/// and a denial are not concerned.
#[test]
fn only_a_person_signed_in_answers_always_on_the_websocket() {
    use crate::api::ws_chat_handler::{lasting_answer_refused, AnswerRefusal};
    use crate::auth::jwt::Claims;
    use crate::chat::types::PermissionAnswerScope::{Always, Once, Session};
    let person = Claims {
        sub: uuid::Uuid::new_v4().to_string(),
        email: "a@ffs.holdings".into(),
        name: "A".into(),
        iat: 0,
        exp: 0,
        token_type: None,
        scope: None,
        jti: None,
    };
    assert_eq!(lasting_answer_refused(&person, true, Always), None);
    for kind in ["mcp", "agent_session", "vault"] {
        let c = Claims {
            token_type: Some(kind.into()),
            ..person.clone()
        };
        assert_eq!(
            lasting_answer_refused(&c, true, Always),
            Some(AnswerRefusal::NotAPerson),
            "{kind}"
        );
        assert_eq!(lasting_answer_refused(&c, true, Session), None, "{kind}");
        assert_eq!(lasting_answer_refused(&c, true, Once), None, "{kind}");
        assert_eq!(lasting_answer_refused(&c, false, Always), None, "{kind}");
    }
    for c in [Claims::service_account("po"), Claims::anonymous()] {
        assert_eq!(
            lasting_answer_refused(&c, true, Always),
            Some(AnswerRefusal::NotAPerson)
        );
    }
}

/// Decision fce98b8a, review of #729 (point 6): `set_permission_mode` is refused to anyone but
/// the session's owner (here the anonymous user of a server without authentication, on a session
/// someone owns), before the mode changes; nothing is persisted, the scope is not lifted.
#[tokio::test]
async fn another_persons_set_permission_mode_is_refused_by_the_websocket_loop() {
    let session = ChatSessionNode {
        file_scope_lifted: false,
        owner: Some(uuid::Uuid::new_v4().to_string()),
        permission_mode: Some("default".into()),
        ..test_chat_session(None)
    };
    let (graph, mut ws) = connected(&session).await;
    let frame = json!({"type": "set_permission_mode", "mode": "bypassPermissions"});
    ws.send(WsMessage::text(frame.to_string())).await.unwrap();
    let refused = frame_where(&mut ws, |v| v["type"] == "error").await;
    assert!(
        refused["message"]
            .as_str()
            .is_some_and(|m| m.contains("not_owner")),
        "{refused}"
    );
    let stored = graph.get_chat_session(session.id).await.unwrap().unwrap();
    assert_eq!(stored.permission_mode.as_deref(), Some("default"));
    assert!(!stored.file_scope_lifted);
}

// ── P18c: a read-only mirror of a Claude Code sub-agent refuses every write ─────────

/// A mirror of a Claude Code sub-agent (`chat::mirror`) of some parent.
fn mirror_session() -> ChatSessionNode {
    ChatSessionNode {
        mirror_of: Some(crate::chat::mirror::MIRROR_OF_CLAUDE_CODE.to_string()),
        spawned_by: Some(
            crate::chat::types::SpawnedBy::Conversation {
                parent_session_id: uuid::Uuid::new_v4(),
                tool_use_id: Some("toolu_task".into()),
            }
            .to_json_string(),
        ),
        ..test_chat_session(None)
    }
}

/// Sends `frame` to a mirror and returns the refusal: the FLAT frame
/// `{type: "error", code: "mirror_read_only", message}`, its message never behind the
/// generic "Failed to …" prefix (the client recognises the code). Nothing was stored.
async fn refused_by_a_mirror(frame: Value) -> Value {
    let session = mirror_session();
    let (graph, mut ws) = connected(&session).await;
    ws.send(WsMessage::text(frame.to_string())).await.unwrap();
    let refused = frame_where(&mut ws, |v| v["type"] == "error").await;
    assert_eq!(refused["code"], "mirror_read_only", "{refused}");
    let message = refused["message"].as_str().unwrap();
    assert!(!message.starts_with("Failed"), "{refused}");
    assert!(message.contains("read-only"), "{refused}");
    assert!(
        graph
            .get_chat_events(session.id, 0, 100)
            .await
            .unwrap()
            .is_empty(),
        "nothing was written to the mirror"
    );
    refused
}

#[tokio::test]
async fn a_message_to_a_mirror_is_refused_by_the_websocket_loop() {
    refused_by_a_mirror(json!({"type": "user_message", "content": "hello"})).await;
}

#[tokio::test]
async fn a_queued_message_to_a_mirror_is_refused_by_the_websocket_loop() {
    refused_by_a_mirror(json!({"type": "user_message", "content": "hello", "queue": true})).await;
}

/// A permission answer (an `AskUserQuestion` of Claude Code is answered the same way).
#[tokio::test]
async fn a_permission_answer_to_a_mirror_is_refused_by_the_websocket_loop() {
    refused_by_a_mirror(
        json!({"type": "permission_response", "id": "pr_m", "allow": true, "scope": "once"}),
    )
    .await;
}

/// The answer to a question (`input_response`).
#[tokio::test]
async fn a_question_answer_to_a_mirror_is_refused_by_the_websocket_loop() {
    refused_by_a_mirror(json!({"type": "input_response", "id": "q_m", "content": "Postgres"}))
        .await;
}

/// A change of model or of permission mode (M7).
#[tokio::test]
async fn a_model_or_mode_change_on_a_mirror_is_refused_by_the_websocket_loop() {
    refused_by_a_mirror(json!({"type": "set_model", "model": "other"})).await;
    refused_by_a_mirror(json!({"type": "set_permission_mode", "mode": "plan"})).await;
}

/// The routing's own refusals reach the socket as the same flat frame, whichever way
/// they come: its `MirrorReadOnly`, or carried by a failed resume (`Resume`) or by a send
/// then a resume that both failed (`SendAndResume`). Any other failure keeps its text.
#[test]
fn a_mirror_refusal_from_the_routing_is_the_flat_frame_on_every_path() {
    use crate::api::ws_chat_handler::{delivery_error_frame, mirror_read_only_frame};
    use crate::chat::manager::MessageDeliveryError;
    let flat = mirror_read_only_frame();
    assert_eq!(flat["type"], "error");
    assert_eq!(flat["code"], "mirror_read_only");
    let mirror = || anyhow::Error::new(crate::chat::mirror::MirrorReadOnly);
    assert_eq!(
        delivery_error_frame(&MessageDeliveryError::MirrorReadOnly),
        flat
    );
    assert_eq!(
        delivery_error_frame(&MessageDeliveryError::Resume(mirror())),
        flat
    );
    assert_eq!(
        delivery_error_frame(&MessageDeliveryError::SendAndResume {
            send: anyhow::anyhow!("no stdin"),
            resume: mirror(),
        }),
        flat
    );
    let other = delivery_error_frame(&MessageDeliveryError::Resume(anyhow::anyhow!("boom")));
    assert!(other.get("code").is_none());
    assert!(other["message"]
        .as_str()
        .unwrap()
        .starts_with("Failed to resume session"));
}
