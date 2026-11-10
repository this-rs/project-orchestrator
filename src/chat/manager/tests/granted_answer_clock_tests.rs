
use super::super::session_grants::{AskedCall, Asker, SessionGrants};
use super::super::tool_clock::ToolClock;
use super::*;
#[allow(unused_imports)]
use serde_json::json;

fn granted() -> std::sync::Mutex<SessionGrants> {
    let mut grants = SessionGrants::default();
    let grant = grants
        .grant_for(&AskedCall {
            asker: Asker::ClaudeCode,
            tool: "Read".into(),
            canonical: None,
            input: json!({"file_path": "/w/a.rs"}),
        })
        .expect("a Read can be granted for the session");
    grants.add(grant);
    std::sync::Mutex::new(grants)
}

fn asked(clock: &ToolClock, pending: &mut std::collections::HashMap<String, PendingPermission>) {
    clock.observe(
        &ChatEvent::ToolUse {
            id: "t1".into(),
            tool: "Read".into(),
            input: json!({"file_path": "/w/a.rs"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
        },
        chrono::Utc::now(),
    );
    clock.observe(
        &ChatEvent::PermissionRequest {
            id: "req-1".into(),
            tool: "Read".into(),
            input: json!({"file_path": "/w/a.rs"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
            tool_use_id: Some("t1".into()),
            always: None,
            consent: None,
        },
        chrono::Utc::now(),
    );
    pending.insert(
        "req-1".into(),
        PendingPermission {
            input: json!({"file_path": "/w/a.rs"}),
            tool: "Read".into(),
        },
    );
}

fn timing(clock: &ToolClock) -> serde_json::Value {
    let event = clock
        .observe(
            &ChatEvent::ToolResult {
                id: "t1".into(),
                result: json!("fn main() {}"),
                is_error: false,
                parent_tool_use_id: None,
            },
            chrono::Utc::now(),
        )
        .expect("a timing");
    serde_json::to_value(event).unwrap()
}

#[tokio::test]
async fn an_answer_of_the_backend_is_on_the_clock_before_the_cli_has_it() {
    let clock = ToolClock::default();
    let mut pending = std::collections::HashMap::new();
    asked(&clock, &mut pending);
    let pending = tokio::sync::Mutex::new(pending);
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let decision = answer_granted_request(
        &granted(),
        &pending,
        &clock,
        Some(&tx),
        "req-1",
        "Read",
        &json!({"file_path": "/w/a.rs"}),
    )
    .await;
    assert!(decision.is_some(), "the grant answers it");
    assert!(rx.try_recv().unwrap().contains("req-1"), "the CLI got it");
    let t = timing(&clock);
    assert_eq!(t["permission_outcome"], "allowed", "{t}");
    assert!(t["permission_resolved_at"].is_f64(), "{t}");
    assert_eq!(t["run_started_at"], t["permission_resolved_at"], "{t}");
}

#[tokio::test]
async fn an_answer_of_the_backend_the_cli_never_got_is_taken_back() {
    let clock = ToolClock::default();
    let mut pending = std::collections::HashMap::new();
    asked(&clock, &mut pending);
    let pending = tokio::sync::Mutex::new(pending);
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(4);
    drop(rx);
    let decision = answer_granted_request(
        &granted(),
        &pending,
        &clock,
        Some(&tx),
        "req-1",
        "Read",
        &json!({"file_path": "/w/a.rs"}),
    )
    .await;
    assert!(decision.is_none(), "the write failed");
    assert!(
        pending.lock().await.contains_key("req-1"),
        "still waits for the user"
    );
    // The user answers it afterwards: that answer is the one on the clock.
    let at = chrono::Utc::now();
    assert!(clock.decided("req-1", false, at).is_some(), "pending again");
    let t = timing(&clock);
    assert_eq!(t["permission_outcome"], "denied", "{t}");
}

/// The CLI is slow to take a first answer (its stdin is full): a second strict
/// answer to the same request waits for it instead of being told `NotPending`, and
/// claims the request given back when the first write fails.
#[tokio::test]
async fn a_second_answer_waits_for_the_first_write_and_claims_a_failed_one() {
    let state = crate::test_helpers::mock_app_state();
    let manager = Arc::new(ChatManager::new_without_memory(
        state.neo4j,
        state.meili,
        test_support::chat_config(),
    ));
    let sid = uuid::Uuid::new_v4().to_string();
    let client = nexus_claude::InteractiveClient::new(nexus_claude::ClaudeCodeOptions {
        model: Some("test".into()),
        cli_path: Some("/nonexistent/claude".into()),
        ..Default::default()
    })
    .unwrap();
    let (stdin_rx, _queue) =
        test_support::insert_session_with_client(&manager, &sid, client, false, &["req-1"])
            .await
            .unwrap();
    let stdin_tx = manager.active_sessions.read().await[&sid]
        .stdin_tx
        .clone()
        .unwrap();
    while stdin_tx.try_send("filler".into()).is_ok() {}
    let answer = |allow: bool| {
        let manager = Arc::clone(&manager);
        let sid = sid.clone();
        tokio::spawn(async move {
            manager
                .send_permission_response_inner(
                    &sid,
                    "req-1",
                    allow,
                    crate::chat::types::PermissionAnswerScope::Once,
                    true,
                )
                .await
        })
    };
    let first = answer(true);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let second = answer(false);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!first.is_finished(), "the first write is blocked");
    assert!(
        !second.is_finished(),
        "the second answer waits for the first write, not NotPending"
    );
    // The CLI is gone: the first write fails and gives the request back.
    drop(stdin_rx);
    let first = first.await.unwrap();
    assert!(
        matches!(first, Err(PermissionDeliveryError::Failed(_))),
        "{first:?}"
    );
    let second = second.await.unwrap();
    assert!(
        matches!(second, Err(PermissionDeliveryError::Failed(_))),
        "the second claimed the request given back (its write fails too): {second:?}"
    );
}
