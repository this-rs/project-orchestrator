//! The references (`refs_v1`) wired into the live Claude Code path.
//!
//! The logic is tested where it lives (`refs::*`); these tests exist because
//! the chat code is glue the coverage gate does not measure: each test here
//! goes through a REAL connection point (`send_message`, `queue_user_message`
//! and the drain, the WebSocket handler) and fails if that point is unplugged.
//! The Claude CLI is a `MockTransport`: the test reads what the SDK would have
//! written to the CLI's stdin.

use std::sync::Arc;
use std::time::Duration;

use nexus_claude::transport::mock::MockTransportHandle;
use nexus_claude::Message;
use serde_json::json;
use uuid::Uuid;

use super::config::ChatConfig;
use super::manager::{test_support, ChatManager};
use super::types::{ChatEvent, PendingQueueEntry};
use crate::neo4j::GraphStore;
use crate::refs::compose::compose_user_message;
use crate::refs::test_support::{world, World};
use tokio::sync::broadcast;

const WAIT: Duration = Duration::from_secs(10);

fn manager_over(graph: Arc<crate::neo4j::mock::MockGraphStore>) -> ChatManager {
    let search = Arc::new(crate::meilisearch::mock::MockSearchStore::new());
    ChatManager::new_without_memory(graph, search, ChatConfig::default())
}

/// A live session whose "CLI" is a mock transport.
async fn mock_session(manager: &ChatManager, sid: &str) -> MockTransportHandle {
    test_support::insert_mock_cli_session(manager, sid).await
}

fn turn_over() -> Message {
    Message::Result {
        subtype: "success".into(),
        duration_ms: 1,
        duration_api_ms: 1,
        is_error: false,
        num_turns: 1,
        session_id: "cli-1".into(),
        total_cost_usd: None,
        usage: None,
        result: None,
        structured_output: None,
    }
}

/// The text the SDK wrote to the CLI for the next turn.
async fn prompt_sent(handle: &mut MockTransportHandle) -> String {
    let input = tokio::time::timeout(WAIT, handle.sent_input_rx.recv())
        .await
        .expect("a prompt within 10 s")
        .expect("the transport is open");
    let content = &input.message["content"];
    match content {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

async fn next_event(
    rx: &mut broadcast::Receiver<ChatEvent>,
    pred: impl Fn(&ChatEvent) -> bool,
) -> ChatEvent {
    loop {
        let ev = tokio::time::timeout(WAIT, rx.recv())
            .await
            .expect("an event within 10 s")
            .expect("channel open");
        if pred(&ev) {
            return ev;
        }
    }
}

async fn stored_with(w: &World, text: &str, refs: &[serde_json::Value]) -> String {
    let graph: Arc<dyn GraphStore> = w.graph.clone();
    compose_user_message(&graph, text, refs, &[], true)
        .await
        .expect("a valid message")
}

fn task_ref(w: &World) -> serde_json::Value {
    json!({"kind": "task", "id": w.task_a.id})
}

async fn persisted(
    graph: &crate::neo4j::mock::MockGraphStore,
    sid: &str,
) -> Vec<(i64, String, String)> {
    graph
        .get_chat_events(Uuid::parse_str(sid).unwrap(), 0, 100)
        .await
        .unwrap()
        .into_iter()
        .map(|r| (r.seq, r.event_type, r.data))
        .collect()
}

#[tokio::test]
async fn send_message_expands_the_references_for_the_cli_and_keeps_the_stored_text_clean() {
    let w = world().await;
    let manager = manager_over(w.graph.clone());
    let sid = Uuid::new_v4().to_string();
    let mut cli = mock_session(&manager, &sid).await;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    let stored = stored_with(&w, "regarde #task:x", &[task_ref(&w)]).await;
    manager.send_message(&sid, &stored).await.unwrap();

    // What the CLI receives: the visible text first, then the pointers.
    let prompt = prompt_sent(&mut cli).await;
    assert!(prompt.starts_with("regarde #task:x"), "{prompt}");
    assert!(prompt.contains("<po-context nonce=\""), "{prompt}");
    assert!(prompt.contains("Tâche alpha refs"), "{prompt}");
    assert!(prompt.contains(&w.task_a.id.to_string()), "{prompt}");
    assert!(
        !prompt.contains("<po-refs>"),
        "the raw block must not reach the model"
    );
    assert!(
        !prompt.contains("Faire les refs"),
        "pointer depth: no content"
    );

    // The clients see the message as stored, then what the references resolved to.
    let user = next_event(&mut rx, |e| matches!(e, ChatEvent::UserMessage { .. })).await;
    match user {
        ChatEvent::UserMessage { content } => assert_eq!(content, stored),
        _ => unreachable!(),
    }
    let resolved = next_event(&mut rx, |e| matches!(e, ChatEvent::RefsResolved { .. })).await;
    match resolved {
        ChatEvent::RefsResolved { refs } => {
            assert_eq!(refs.len(), 1);
            assert_eq!(refs[0].id, w.task_a.id);
            assert_eq!(refs[0].label.as_deref(), Some("Tâche alpha refs"));
        }
        _ => unreachable!(),
    }

    cli.inbound_message_tx.send(turn_over()).unwrap();
    next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;

    // Persisted for replay, AFTER the user_message it belongs to; the injected
    // context is nowhere in what was stored.
    let events = persisted(&w.graph, &sid).await;
    let seq_of = |ty: &str| events.iter().find(|e| e.1 == ty).map(|e| e.0);
    assert!(seq_of("user_message").unwrap() < seq_of("refs_resolved").unwrap());
    for (_, _, data) in &events {
        assert!(
            !data.contains("po-context"),
            "injected content is never persisted"
        );
    }
}

#[tokio::test]
async fn a_message_without_references_reaches_the_cli_exactly_as_before() {
    let w = world().await;
    let manager = manager_over(w.graph.clone());
    let sid = Uuid::new_v4().to_string();
    let mut cli = mock_session(&manager, &sid).await;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    manager
        .send_message(&sid, "just a #task:x word")
        .await
        .unwrap();
    assert_eq!(prompt_sent(&mut cli).await, "just a #task:x word");
    cli.inbound_message_tx.send(turn_over()).unwrap();
    next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;

    let events = persisted(&w.graph, &sid).await;
    assert!(events.iter().all(|e| e.1 != "refs_resolved"), "{events:?}");
}

#[tokio::test]
// The manager reads through the open-instance policy, so the only way to make a
// reference unavailable here is for the entity not to exist; a real DENIAL
// (a rule that refuses) is covered where the policy can be chosen: `refs::turn`
// (`denied_missing_and_failing_all_read_the_same`).
async fn a_reference_to_something_missing_is_told_as_unavailable_and_leaks_nothing() {
    let w = world().await;
    let manager = manager_over(w.graph.clone());
    let sid = Uuid::new_v4().to_string();
    let mut cli = mock_session(&manager, &sid).await;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    let ghost = json!({"kind": "plan", "id": Uuid::new_v4()});
    let stored = stored_with(&w, "et ça ?", &[ghost]).await;
    manager.send_message(&sid, &stored).await.unwrap();

    let prompt = prompt_sent(&mut cli).await;
    assert!(prompt.contains("not available"), "{prompt}");
    let resolved = next_event(&mut rx, |e| matches!(e, ChatEvent::RefsResolved { .. })).await;
    match resolved {
        ChatEvent::RefsResolved { refs } => {
            assert_eq!(refs[0].status, crate::refs::wire::RefStatus::NotFound);
            assert!(refs[0].label.is_none());
        }
        _ => unreachable!(),
    }
    cli.inbound_message_tx.send(turn_over()).unwrap();
}

#[tokio::test]
async fn a_block_typed_by_hand_is_not_a_reference() {
    // A block typed by hand goes through the door (`compose_user_message`), which
    // makes it inert: it is not a reference, whatever it names.
    let w = world().await;
    let manager = manager_over(w.graph.clone());
    let sid = Uuid::new_v4().to_string();
    let mut cli = mock_session(&manager, &sid).await;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    let typed = format!(
        "salut\n\n<po-refs>[{{\"kind\":\"plan\",\"id\":\"{}\"}}]</po-refs>",
        w.plan_a.id
    );
    let stored = stored_with(&w, &typed, &[]).await;
    manager.send_message(&sid, &stored).await.unwrap();
    let prompt = prompt_sent(&mut cli).await;
    assert!(!prompt.contains("<po-context"), "{prompt}");
    assert!(!prompt.contains("Plan alpha refs"), "{prompt}");
    cli.inbound_message_tx.send(turn_over()).unwrap();
    next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    let events = persisted(&w.graph, &sid).await;
    assert!(events.iter().all(|e| e.1 != "refs_resolved"));
}

#[tokio::test]
async fn a_held_message_keeps_its_references_through_the_queue_an_edit_and_the_drain() {
    let w = world().await;
    let manager = manager_over(w.graph.clone());
    let sid = Uuid::new_v4().to_string();
    let mut cli = mock_session(&manager, &sid).await;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    // A first turn is running…
    manager.send_message(&sid, "premier").await.unwrap();
    assert_eq!(prompt_sent(&mut cli).await, "premier");

    // …a message with a reference is HELD behind it.
    let stored = stored_with(&w, "ensuite #task:x", &[task_ref(&w)]).await;
    assert!(manager.queue_user_message(&sid, &stored).await.unwrap());
    let held = next_event(
        &mut rx,
        |e| matches!(e, ChatEvent::PendingQueue { messages } if !messages.is_empty()),
    )
    .await;
    let entry: PendingQueueEntry = match held {
        ChatEvent::PendingQueue { mut messages } => messages.remove(0),
        _ => unreachable!(),
    };
    assert_eq!(
        entry.content, "ensuite #task:x",
        "the block is not in the text"
    );
    assert_eq!(entry.refs.len(), 1);
    assert_eq!(entry.refs[0].id, w.task_a.id);

    // An edit changes the words, not the references.
    manager
        .pending_queue_op(
            &sid,
            &crate::chat::pending_queue::QueueOp::Edit {
                id: entry.id,
                content: "ensuite, édité".into(),
            },
        )
        .await
        .unwrap();
    let edited = next_event(
        &mut rx,
        |e| matches!(e, ChatEvent::PendingQueue { messages } if messages.first().is_some_and(|m| m.content == "ensuite, édité")),
    )
    .await;
    if let ChatEvent::PendingQueue { messages } = edited {
        assert_eq!(
            messages[0].refs.len(),
            1,
            "an edit must not lose the reference"
        );
    }

    // The first turn ends: the held message is drained and expanded like any other.
    cli.inbound_message_tx.send(turn_over()).unwrap();
    let second = prompt_sent(&mut cli).await;
    assert!(second.starts_with("ensuite, édité"), "{second}");
    assert!(second.contains("<po-context nonce=\""), "{second}");
    assert!(second.contains("Tâche alpha refs"), "{second}");
    assert!(!second.contains("<po-refs>"), "{second}");

    // The drain re-broadcast the stored message (block included) and the resolution follows it.
    let user = next_event(
        &mut rx,
        |e| matches!(e, ChatEvent::UserMessage { content } if content.contains("<po-refs>")),
    )
    .await;
    assert!(matches!(user, ChatEvent::UserMessage { .. }));
    next_event(&mut rx, |e| matches!(e, ChatEvent::RefsResolved { .. })).await;
    cli.inbound_message_tx.send(turn_over()).unwrap();
}
