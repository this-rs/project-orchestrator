//! End to end: a REGISTERED OpenAI-compatible instance opens a session through
//! the native harness of nexus (`CHAT_PROVIDER_PATH=agent`), against the
//! `fake_openai` and `fake_mcp` binaries of the nexus repository (loopback only,
//! no network, no key).
//!
//! The two binaries are built from the nexus checkout the build is patched to:
//! `cargo build -p nexus-claude --features "provider-native testkit" --bin
//! fake_openai --bin fake_mcp`. `NEXUS_FAKES_DIR` names the directory holding
//! them (default: `<repo>/../.target-nexus-fakes/debug`). Missing binaries FAIL
//! the tests: a test that cannot run must not pass.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use chrono::{Duration as ChronoDuration, Utc};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use uuid::Uuid;

use super::config::ProviderPath;
use super::manager::ChatManager;
use super::provider::settings::{
    ConsentRecord, InstanceRecord, CONSENT_PREFIX, GLOBAL, INSTANCE_PREFIX,
};
use super::types::{ChatEvent, ChatRequest};
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::GraphStore;
use crate::test_helpers::mock_app_state;
use crate::vault::grants::{GrantScope, SecretSelector};
use crate::vault::VaultService;

fn fakes_dir() -> PathBuf {
    std::env::var_os("NEXUS_FAKES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.target-nexus-fakes/debug")
        })
}

fn fake_bin(name: &str) -> PathBuf {
    let path = fakes_dir().join(name);
    assert!(
        path.exists(),
        "{} is missing: build the nexus fakes (see the module doc; CI does it in the step 'Build the nexus fake servers')",
        path.display()
    );
    path
}

/// A running `fake_openai`, killed on drop.
struct FakeOpenAi {
    child: Child,
    port: u16,
    dir: tempfile::TempDir,
}

impl FakeOpenAi {
    fn start(routes: Value) -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("script.json");
        std::fs::write(&script, routes.to_string()).unwrap();
        let mut child = Command::new(fake_bin("fake_openai"))
            .env("FAKE_OPENAI_SCRIPT", &script)
            .env(
                "FAKE_OPENAI_REQUESTS_OUT",
                dir.path().join("requests.jsonl"),
            )
            .env("FAKE_OPENAI_MAX_RUNTIME_MS", "120000")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn fake_openai");
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let port = line
            .trim()
            .strip_prefix("LISTENING ")
            .and_then(|p| p.parse().ok())
            .unwrap_or_else(|| panic!("unexpected first line: {line:?}"));
        Self { child, port, dir }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    fn origin(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn requests(&self) -> Vec<Value> {
        std::fs::read_to_string(self.dir.path().join("requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    fn chat_requests(&self) -> Vec<Value> {
        self.requests()
            .into_iter()
            .filter(|r| r["path"] == "/v1/chat/completions")
            .collect()
    }
}

impl Drop for FakeOpenAi {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn delta(d: Value) -> Value {
    json!({"choices": [{"index": 0, "delta": d}]})
}

fn sse_route(body_contains: &str, events: Vec<Value>) -> Value {
    json!({"method": "POST", "path": "/v1/chat/completions", "status": 200,
           "body_contains": body_contains, "sse": events})
}

fn script() -> Value {
    json!([
        // The probe: the model must be able to call a tool.
        sse_route("Call the ping tool now", vec![
            delta(json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]})),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            json!("[DONE]"),
        ]),
        {"method": "GET", "path": "/v1/models", "status": 200,
         "body": {"object": "list", "data": [{"id": "m", "context_length": 32000}]}},
        // The session turn.
        sse_route("hi there", vec![
            delta(json!({"content": "hello from "})),
            delta(json!({"content": "the fake model"})),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}),
            json!("[DONE]"),
        ]),
    ])
}

fn instance(fake: &FakeOpenAi, credential_ref: &str) -> InstanceRecord {
    InstanceRecord {
        id: "local".into(),
        kind: "openai_compatible".into(),
        preset: Some("llama_server".into()),
        label: "Local".into(),
        base_url: fake.base_url(),
        origin: fake.origin(),
        default_model: Some("m".into()),
        cost_source: "free".into(),
        credential_ref: credential_ref.into(),
        ..Default::default()
    }
}

async fn store_instance(graph: &MockGraphStore, record: &InstanceRecord) {
    graph
        .put_llm_setting(
            GLOBAL,
            &format!("{INSTANCE_PREFIX}{}", record.id),
            &serde_json::to_string(record).unwrap(),
        )
        .await
        .unwrap();
}

async fn consent(graph: &MockGraphStore, slug: &str, id: &str, origin: &str) {
    // The consent names the credential reference the instance has NOW.
    let credential_ref = graph
        .get_llm_setting(GLOBAL, &format!("{INSTANCE_PREFIX}{id}"))
        .await
        .unwrap()
        .and_then(|raw| serde_json::from_str::<InstanceRecord>(&raw).ok())
        .map(|i| i.credential_ref)
        .unwrap_or_else(|| "none".to_string());
    let record = ConsentRecord {
        provider_id: id.into(),
        origin: origin.into(),
        consented_by: "me@example.com".into(),
        consented_at: Utc::now().to_rfc3339(),
        credential_ref: Some(credential_ref.to_string()),
        ..Default::default()
    };
    graph
        .put_llm_setting(
            &format!("project:{slug}"),
            &format!("{CONSENT_PREFIX}{id}"),
            &serde_json::to_string(&record).unwrap(),
        )
        .await
        .unwrap();
}

fn manager(graph: Arc<MockGraphStore>, secure: bool) -> ChatManager {
    let state = mock_app_state();
    let dyn_graph: Arc<dyn GraphStore> = graph;
    let config = super::config::ChatConfig {
        provider_path: ProviderPath::Agent,
        mcp_server_path: fake_bin("fake_mcp"),
        jwt_secret: secure.then(|| "test-secret-test-secret-test-secret".to_string()),
        max_sessions: 10,
        ..Default::default()
    };
    ChatManager::new_without_memory(dyn_graph, state.meili, config)
}

fn request(provider: Option<&str>, project: Option<&str>, mode: &str) -> ChatRequest {
    ChatRequest {
        routing_pool: None,
        routing_mode: None,
        attachments: Vec::new(),
        message: "hi there".into(),
        session_id: None,
        cwd: std::env::temp_dir().display().to_string(),
        project_slug: project.map(str::to_string),
        model: None,
        provider: provider.map(str::to_string),
        task_alias: None,
        run_provider: None,
        run_model: None,
        max_tokens: None,
        task_class: None,
        permission_mode: Some(mode.into()),
        add_dirs: None,
        workspace_slug: None,
        user_claims: Some(crate::auth::jwt::Claims::service_account("e2e")),
        spawned_by: None,
        task_context: None,
        scaffolding_override: None,
        runner_context: None,
        routing_decision_id: None,
    }
}

async fn next_event(
    rx: &mut broadcast::Receiver<ChatEvent>,
    pred: impl Fn(&ChatEvent) -> bool,
) -> ChatEvent {
    loop {
        let ev = tokio::time::timeout(Duration::from_secs(20), rx.recv())
            .await
            .expect("an event within 20 s")
            .expect("channel open");
        if pred(&ev) {
            return ev;
        }
    }
}

fn failure(err: &anyhow::Error) -> (u16, &'static str) {
    let f = super::provider::errors::classify_open_error(err, Some("local"))
        .unwrap_or_else(|| panic!("not a typed failure: {err:#}"));
    (f.status, f.code)
}

#[tokio::test]
async fn a_registered_instance_opens_a_native_session_and_streams_a_turn() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let manager = manager(graph.clone(), true);

    let created = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_or_else(|e| panic!("open failed: {e:#}"));
    let sid = created.session_id;
    let mut rx = manager.subscribe(&sid).await.unwrap();

    let text = next_event(
        &mut rx,
        |e| matches!(e, ChatEvent::AssistantText { content, .. } if content.contains("fake model")),
    )
    .await;
    assert!(matches!(text, ChatEvent::AssistantText { .. }));
    let result = next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    match result {
        ChatEvent::Result {
            cost,
            stop_reason,
            model,
            ..
        } => {
            assert_eq!(stop_reason.as_deref(), Some("completed"));
            assert_eq!(model.as_deref(), Some("m"));
            // The instance is declared free: the basis says so.
            assert_eq!(cost.unwrap()["basis"], "free");
        }
        other => panic!("{other:?}"),
    }

    // The persisted session names the instance and carries the snapshot.
    let node = graph
        .get_chat_session(Uuid::parse_str(&sid).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.provider_id.as_deref(), Some("local"));
    assert_eq!(node.routed_by.as_deref(), Some("request"));
    let caps: Value = serde_json::from_str(node.capabilities.as_deref().unwrap()).unwrap();
    assert_eq!(caps["tools"], true, "the probe called a tool");
    assert_eq!(caps["sandbox"], "none");

    // The model saw the project-orchestrator tools, and only through MCP.
    let chats = fake.chat_requests();
    let turn = chats
        .iter()
        .find(|r| r["body"].to_string().contains("hi there"))
        .expect("the turn");
    assert!(
        turn["body"]
            .to_string()
            .contains("mcp__project-orchestrator__"),
        "{turn}"
    );

    // The sending was journaled: who, which project, which origin.
    let journal = graph.list_llm_settings("journal", "send:").await.unwrap();
    assert_eq!(journal.len(), 1);
    let entry: Value = serde_json::from_str(&journal[0].1).unwrap();
    assert_eq!(entry["project"], "proj");
    assert_eq!(entry["origin"], fake.origin());
    assert_eq!(entry["provider"], "local");
    assert!(
        !journal[0].1.contains("hi there"),
        "the journal never holds content"
    );

    manager.close_session(&sid).await.unwrap();
}

#[tokio::test]
async fn without_consent_nothing_reaches_the_endpoint() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    let manager = manager(graph.clone(), true);

    // A project that never consented.
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (403, "endpoint_not_allowed"));
    // No project at all: claude-code only.
    let err = manager
        .create_session(&request(Some("local"), None, "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (403, "endpoint_not_allowed"));
    assert!(fake.requests().is_empty(), "no connection was made");
    assert!(
        graph.chat_sessions.read().await.is_empty(),
        "nothing persisted"
    );
    assert!(graph
        .list_llm_settings("journal", "")
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn a_consent_for_another_origin_does_not_hold() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", "https://old.example.com").await;
    let manager = manager(graph.clone(), true);
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (403, "endpoint_not_allowed"));
    assert!(fake.requests().is_empty());
}

#[tokio::test]
async fn the_endpoint_guard_runs_before_any_connection() {
    let graph = Arc::new(MockGraphStore::new());
    let record = InstanceRecord {
        id: "internal".into(),
        kind: "openai_compatible".into(),
        preset: None,
        label: "Internal".into(),
        base_url: "https://10.0.0.5/v1".into(),
        origin: "https://10.0.0.5".into(),
        default_model: Some("m".into()),
        cost_source: "unknown".into(),
        credential_ref: "none".into(),
        ..Default::default()
    };
    // Stored by an older version, or by hand: the guard must still refuse it.
    store_instance(&graph, &record).await;
    consent(&graph, "proj", "internal", "https://10.0.0.5").await;
    let manager = manager(graph.clone(), true);
    let err = manager
        .create_session(&request(Some("internal"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (403, "endpoint_not_allowed"));
}

#[tokio::test]
async fn trust_opens_on_a_provider_without_a_sandbox_like_it_does_on_claude_code() {
    // Decision of 2026-10-07 (replaces A35): a third-party provider behaves like Claude Code.
    // The sandbox level informs the user, it gates nothing: Rock'n roll opens, and the sending
    // is journaled like any other.
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let manager = manager(graph.clone(), true);
    manager
        .create_session(&request(Some("local"), Some("proj"), "bypassPermissions"))
        .await
        .expect("trust opens on a provider with no sandbox");
    assert_eq!(
        graph.list_llm_settings("journal", "").await.unwrap().len(),
        1,
        "the sending is journaled"
    );
}

#[tokio::test]
async fn a_third_party_is_refused_while_authentication_is_off() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let manager = manager(graph.clone(), false);
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (422, "unsupported"));
    assert!(fake.requests().is_empty());
}

#[tokio::test]
async fn a_locked_vault_is_credentials_locked_and_never_falls_back() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "vault:endpoint-key")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    // A vault that was never unlocked.
    let manager = manager(graph.clone(), true).with_vault(VaultService::ephemeral());
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (423, "credentials_locked"));
    assert!(
        fake.chat_requests().is_empty(),
        "no request was sent unauthenticated"
    );
    assert!(graph.chat_sessions.read().await.len() <= 1);
}

#[tokio::test]
async fn the_key_is_read_under_a_grant_for_that_instance_and_sent_as_a_header() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "vault:endpoint-key")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let vault = VaultService::ephemeral();
    vault
        .init(
            "correct horse battery staple".into(),
            ChronoDuration::hours(1),
        )
        .await
        .unwrap();
    let now = Utc::now();
    vault
        .put("endpoint-key", "sk-e2e-value-4321", None, now)
        .unwrap();
    vault
        .grant(
            SecretSelector::Names(["endpoint-key".to_string()].into()),
            GrantScope::Provider("local".to_string()),
            ChronoDuration::hours(1),
            None,
            now,
        )
        .unwrap();
    let manager = manager(graph.clone(), true).with_vault(vault);
    let created = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_or_else(|e| panic!("open failed: {e:#}"));
    let mut rx = manager.subscribe(&created.session_id).await.unwrap();
    next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
    let with_auth = fake
        .requests()
        .iter()
        .filter(|r| r["path"] == "/v1/chat/completions")
        .all(|r| r["authorization_present"] == true);
    assert!(
        with_auth,
        "requests carry an Authorization header: {:?}",
        fake.requests()
    );
    // The value itself is nowhere: not in the journal, not in a persisted event.
    let journal = graph.list_llm_settings("journal", "").await.unwrap();
    assert!(!journal[0].1.contains("sk-e2e"));
    let events = graph
        .get_chat_events(Uuid::parse_str(&created.session_id).unwrap(), 0, 100)
        .await
        .unwrap();
    assert!(events.iter().all(|e| !e.data.contains("sk-e2e")));
}

#[tokio::test]
async fn a_pilot_role_routes_a_project_to_the_instance_and_consent_still_applies() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    graph
        .put_llm_setting(
            "project:proj",
            "roles",
            &json!({"pilot": {"provider": "local"}}).to_string(),
        )
        .await
        .unwrap();
    let manager = manager(graph.clone(), true);
    // The rule names the instance, the project has not consented: refused (a pilot never falls back).
    let err = manager
        .create_session(&request(None, Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (403, "endpoint_not_allowed"));
    consent(&graph, "proj", "local", &fake.origin()).await;
    let created = manager
        .create_session(&request(None, Some("proj"), "default"))
        .await
        .unwrap_or_else(|e| panic!("open failed: {e:#}"));
    let node = graph
        .get_chat_session(Uuid::parse_str(&created.session_id).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.provider_id.as_deref(), Some("local"));
    assert_eq!(node.routed_by.as_deref(), Some("project_rule"));
}

#[tokio::test]
async fn a_third_party_runs_on_the_agent_engine_without_any_flag() {
    // CHAT_PROVIDER_PATH is at its default (`legacy`): a registered instance still opens,
    // because the agent engine serves third parties by design; only claude-code stays on legacy.
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let state = mock_app_state();
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let mut config = super::config::ChatConfig::default();
    assert_eq!(config.provider_path, ProviderPath::Legacy, "the default");
    config.mcp_server_path = fake_bin("fake_mcp");
    config.jwt_secret = Some("test-secret-test-secret-test-secret".to_string());
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config);
    let created = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_or_else(|e| panic!("a third party must open on the agent engine: {e:#}"));
    assert!(manager.agent_runtime.owns(&created.session_id).await);
    let mut rx = manager.subscribe(&created.session_id).await.unwrap();
    next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
}

async fn put_policy(graph: &MockGraphStore, mode: &str) {
    graph
        .put_llm_setting(
            GLOBAL,
            "model_aliases",
            &json!([{"alias": "fast", "provider": "local", "model": "m"}]).to_string(),
        )
        .await
        .unwrap();
    graph
        .put_llm_setting(
            GLOBAL,
            "model_policy",
            &json!({"mode": mode, "rules": {"chat": "fast"}, "fallback": [], "caps": {}})
                .to_string(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn an_enforced_policy_routes_the_chat_to_the_instance_and_records_the_rule() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    put_policy(&graph, "enforce").await;
    let manager = manager(graph.clone(), true);
    let created = manager
        .create_session(&request(None, Some("proj"), "default"))
        .await
        .unwrap_or_else(|e| panic!("open failed: {e:#}"));
    let node = graph
        .get_chat_session(Uuid::parse_str(&created.session_id).unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(node.provider_id.as_deref(), Some("local"));
    assert_eq!(node.routed_by.as_deref(), Some("global_rule"));
    let note: Value = serde_json::from_str(
        &graph
            .get_llm_setting(&format!("routing:{}", created.session_id), "note")
            .await
            .unwrap()
            .expect("the rule is recorded"),
    )
    .unwrap();
    assert_eq!(note["route_rule"], "chat");
    assert!(note["shadow_provider"].is_null());
}

#[tokio::test]
async fn a_shadow_policy_changes_nothing_and_only_records_what_it_would_have_done() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    put_policy(&graph, "shadow").await;
    // Claude Code is not installable here: use the legacy engine's refusal as the
    // witness that the policy did NOT route to the instance, and read the note
    // through the resolver directly.
    let manager = manager(graph.clone(), true);
    let choice = manager
        .resolve_provider_choice(&request(None, Some("proj"), "default"), Some("proj"))
        .await
        .unwrap();
    assert_eq!(choice.provider_id, "claude-code", "shadow never applies");
    assert_eq!(choice.shadow.as_ref().map(|s| s.0.as_str()), Some("local"));
    assert_eq!(choice.route_rule.as_deref(), Some("chat"));
    assert!(fake.requests().is_empty());
}

#[tokio::test]
async fn an_off_policy_is_invisible() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    put_policy(&graph, "off").await;
    let manager = manager(graph.clone(), true);
    let choice = manager
        .resolve_provider_choice(&request(None, Some("proj"), "default"), Some("proj"))
        .await
        .unwrap();
    assert_eq!(
        (
            choice.provider_id.as_str(),
            choice.route_rule,
            choice.shadow
        ),
        ("claude-code", None, None)
    );
}

fn draft_record(fake: &FakeOpenAi, credential_ref: &str) -> InstanceRecord {
    instance(fake, credential_ref)
}

#[tokio::test]
async fn the_connection_test_really_probes_a_tool_call_and_reads_the_window() {
    let fake = FakeOpenAi::start(script());
    let report = super::provider::native_factory::probe_instance(
        &draft_record(&fake, "none"),
        None,
        Some("m"),
    )
    .await
    .unwrap();
    assert_eq!(report.tools, Some(true), "{report:?}");
    assert_eq!(report.context_window, Some(32_000));
    assert!(report.models.contains(&"m".to_string()));
    // The probe really called the endpoint with the ping tool.
    assert!(fake
        .chat_requests()
        .iter()
        .any(|r| r["body"].to_string().contains("Call the ping tool now")));
}

#[tokio::test]
async fn a_model_that_cannot_call_tools_is_reported_not_swallowed() {
    // The probe answers with plain text: no tool call.
    let fake = FakeOpenAi::start(json!([
        sse_route("Call the ping tool now", vec![
            delta(json!({"content": "I cannot"})),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            json!("[DONE]"),
        ]),
        {"method": "GET", "path": "/v1/models", "status": 200,
         "body": {"object": "list", "data": [{"id": "m"}]}},
    ]));
    let report = super::provider::native_factory::probe_instance(
        &draft_record(&fake, "none"),
        None,
        Some("m"),
    )
    .await
    .unwrap();
    assert_eq!(report.tools, Some(false), "{report:?}");
}

#[tokio::test]
async fn a_probe_with_a_locked_vault_is_credentials_locked() {
    let fake = FakeOpenAi::start(script());
    let err = super::provider::native_factory::probe_instance(
        &draft_record(&fake, "vault:endpoint-key"),
        Some(VaultService::ephemeral()),
        Some("m"),
    )
    .await;
    // Either the health check or the probe surfaces it; nothing is sent unauthenticated.
    match err {
        Err(e) => assert_eq!(e.kind(), "credentials_locked"),
        Ok(report) => {
            let locked = report
                .probe_error
                .as_ref()
                .is_some_and(|e| e.kind() == "credentials_locked")
                || matches!(report.health.error, Some(ref e) if e.kind() == "credentials_locked");
            assert!(locked, "{report:?}");
        }
    }
    assert!(fake.chat_requests().is_empty());
}

#[tokio::test]
async fn changing_the_credential_reference_after_consent_stops_the_sending() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    // The same instance is now pointed at a vault key the project never agreed to.
    store_instance(&graph, &instance(&fake, "vault:another-key")).await;
    let manager = manager(graph.clone(), true);
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (403, "endpoint_not_allowed"));
    assert!(fake.requests().is_empty());
}

#[tokio::test]
async fn a_window_too_small_for_the_tool_schemas_refuses_the_session_after_the_probe() {
    // The endpoint says the model's window is 1 500 tokens: the tool schemas alone are larger.
    let mut routes = script().as_array().unwrap().clone();
    routes[1] = json!({"method": "GET", "path": "/v1/models", "status": 200,
        "body": {"object": "list", "data": [{"id": "m", "context_length": 1500}]}});
    let fake = FakeOpenAi::start(Value::Array(routes));
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let manager = manager(graph.clone(), true);
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (422, "context_too_small"));
    assert!(
        manager.agent_runtime.len().await == 0,
        "the session was closed"
    );
    assert!(
        !fake
            .chat_requests()
            .iter()
            .any(|r| r["body"].to_string().contains("hi there")),
        "the user's message never reached the model"
    );
}

#[test]
fn the_tool_schemas_have_a_size_and_an_unknown_window_is_not_a_refusal() {
    use nexus_claude::agent::{Capabilities, ContextWindow, ContextWindowSource};
    let tokens = super::manager::restricted_tool_schema_tokens();
    assert!(
        tokens > 500,
        "the restricted profile still has tools: {tokens}"
    );
    let mut caps = Capabilities::none();
    assert!(
        super::manager::window_holds_the_tools(&caps).is_ok(),
        "unknown window"
    );
    caps.context_window = Some(ContextWindow {
        value: tokens * 2 + 1,
        source: ContextWindowSource::Probed,
    });
    assert!(super::manager::window_holds_the_tools(&caps).is_ok());
    caps.context_window = Some(ContextWindow {
        value: tokens,
        source: ContextWindowSource::Probed,
    });
    assert!(super::manager::window_holds_the_tools(&caps).is_err());
}

#[tokio::test]
async fn verify4_authorize_refuses_a_project_without_consent_even_when_called_directly() {
    // A resume re-checks consent (it may have been revoked meanwhile): the check
    // of authorize_provider_use itself, not only the resolver's.
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    let manager = manager(graph.clone(), true);
    let claims = crate::auth::jwt::Claims::service_account("e2e");
    let r = manager
        .authorize_provider_use(super::manager::ProviderUse {
            provider_id: "local",
            model: "m",
            mode: nexus_claude::agent::PolicyMode::Ask,
            project_slug: Some("proj"),
            claims: Some(&claims),
            session_id: "verify4-sid",
        })
        .await;
    assert!(r.is_err(), "no consent: must be refused");
}

#[tokio::test]
async fn verify4_a_refused_opening_revokes_the_session_token() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    // The refusal comes from the same gate (authorize_provider_use), after the token was
    // bound to the session: no trace of the sending, no sending (A37).
    graph
        .fail_journal_writes
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let manager = manager(graph.clone(), true);
    let _ = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    let ids: Vec<_> = graph.chat_sessions.read().await.keys().cloned().collect();
    assert_eq!(
        ids.len(),
        1,
        "the persisted session node is still there: {ids:?}"
    );
    assert_eq!(
        crate::auth::agent_tokens::revoke_session(&ids[0].to_string()),
        0,
        "the token of a refused opening was left live"
    );
}

#[tokio::test]
async fn an_opening_is_refused_when_the_send_journal_cannot_be_written() {
    // A37: no trace of the sending, no sending. The refusal leaves nothing live.
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    graph
        .fail_journal_writes
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let manager = manager(graph.clone(), true);
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (503, "provider_unavailable"));
    assert!(manager.agent_runtime.is_empty().await, "no live session");
    assert!(
        fake.chat_requests().is_empty(),
        "the model was never called: the content did not leave"
    );
    assert!(graph
        .list_llm_settings("journal", "")
        .await
        .unwrap()
        .is_empty());
    // The token minted for the refused opening is dead.
    let ids: Vec<_> = graph.chat_sessions.read().await.keys().cloned().collect();
    assert_eq!(ids.len(), 1);
    assert_eq!(
        crate::auth::agent_tokens::revoke_session(&ids[0].to_string()),
        0,
        "the token of a refused opening was left live"
    );
    // Once the journal is writable again, the same request opens.
    graph
        .fail_journal_writes
        .store(false, std::sync::atomic::Ordering::SeqCst);
    manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_or_else(|e| panic!("opens once the journal works: {e:#}"));
}

#[test]
fn only_a_remote_machine_that_does_not_allow_it_holds_trust_back() {
    use super::manager::ChatManager;
    use super::provider::settings::{InstanceRecord, KIND_CLAUDE_CODE_REMOTE};
    let record = |kind: &str, allow_trust: bool| InstanceRecord {
        kind: kind.into(),
        allow_trust,
        ..Default::default()
    };
    // A machine nobody watches keeps its own explicit switch ...
    assert!(ChatManager::trust_needs_opt_in(&record(
        KIND_CLAUDE_CODE_REMOTE,
        false
    )));
    assert!(!ChatManager::trust_needs_opt_in(&record(
        KIND_CLAUDE_CODE_REMOTE,
        true
    )));
    // ... every other provider behaves like Claude Code (decision of 2026-10-07).
    for kind in ["openai_compatible", "codex", "acp", "claude_code"] {
        assert!(
            !ChatManager::trust_needs_opt_in(&record(kind, false)),
            "{kind}"
        );
    }
}

// ── Cognitive routing (R2, B-R4) ────────────────────────────────────────────
//
// `resolve_provider_choice` with the cognitive router wired, against a stored
// instance served by `fake_openai`. The built-in Claude Code is hidden from the
// provider source so the pool holds exactly one pair (`local`/`m`) and the
// choice does not depend on what is installed on the machine.

mod cognitive_routing {
    use super::*;
    use crate::chat::agent_runtime::ProviderSource;
    use crate::chat::provider::cognitive::decider::CognitiveRouting;
    use crate::chat::provider::cognitive::store::{DecisionFilter, RoutingArmStore};
    use crate::chat::provider::resolver::{ProviderChoice, RoutedBy};
    use crate::neo4j::routing::Neo4jRoutingStore;

    struct NoBuiltin;

    impl ProviderSource for NoBuiltin {
        fn get(&self, _provider_id: &str) -> Option<Arc<dyn nexus_claude::agent::AgentProvider>> {
            None
        }
    }

    struct Setup {
        _fake: FakeOpenAi,
        graph: Arc<MockGraphStore>,
        manager: ChatManager,
    }

    async fn setup(mode: &str, stage: &str, wired: bool) -> Setup {
        setup_with(mode, stage, wired, true).await
    }

    /// `warm` opens one session first, which probes the instance as a side effect.
    async fn setup_with(mode: &str, stage: &str, wired: bool, warm: bool) -> Setup {
        let fake = FakeOpenAi::start(script());
        let graph = Arc::new(MockGraphStore::new());
        store_instance(&graph, &instance(&fake, "none")).await;
        consent(&graph, "proj", "local", &fake.origin()).await;
        graph
            .put_llm_setting(
                GLOBAL,
                "routing",
                &json!({"mode": mode, "stage": stage, "exploration_epsilon": 0.0}).to_string(),
            )
            .await
            .unwrap();
        let mut manager = manager(graph.clone(), true).with_provider_source(Arc::new(NoBuiltin));
        // The router only trusts what an instance is KNOWN to do: a provider that
        // was never probed reports no tools and an unknown window. Opening one
        // session on the instance probes it and keeps the probe in the cache.
        if warm {
            let opened = manager
                .create_session(&request(Some("local"), Some("proj"), "default"))
                .await
                .unwrap_or_else(|e| panic!("warm-up failed: {e:#}"));
            let mut rx = manager.subscribe(&opened.session_id).await.unwrap();
            next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        }
        if wired {
            let store = Arc::new(Neo4jRoutingStore::new(graph.clone()));
            manager = manager.with_cognitive_routing(CognitiveRouting::new(store));
        }
        Setup {
            _fake: fake,
            graph,
            manager,
        }
    }

    fn executor_request() -> ChatRequest {
        let mut r = request(None, Some("proj"), "default");
        r.spawned_by = Some("{}".into());
        r.task_class = Some("simple".into());
        r
    }

    fn pilot_request() -> ChatRequest {
        request(None, Some("proj"), "default")
    }

    async fn choose(s: &Setup, r: &ChatRequest) -> ProviderChoice {
        s.manager
            .resolve_provider_choice(r, Some("proj"))
            .await
            .unwrap_or_else(|e| panic!("resolve failed: {e:#}"))
    }

    async fn decisions(
        s: &Setup,
    ) -> Vec<crate::chat::provider::cognitive::decision::CognitiveDecision> {
        Neo4jRoutingStore::new(s.graph.clone())
            .decisions(&DecisionFilter::default())
            .await
            .unwrap()
    }

    /// What the choice is, without the recorded shadow.
    fn essence(c: &ProviderChoice) -> (String, Option<String>, RoutedBy, Option<String>) {
        (
            c.provider_id.clone(),
            c.model.clone(),
            c.routed_by,
            c.route_rule.clone(),
        )
    }

    #[tokio::test]
    async fn primary_changes_nothing_at_any_stage_for_anyone() {
        for stage in ["shadow", "advisory", "auto"] {
            let with = setup("primary", stage, true).await;
            let without = setup("primary", stage, false).await;
            let sent_after_warm_up = with._fake.chat_requests().len();
            for r in [pilot_request(), executor_request()] {
                let a = choose(&with, &r).await;
                let b = choose(&without, &r).await;
                assert_eq!(essence(&a), essence(&b), "primary/{stage}");
                assert_ne!(a.routed_by, RoutedBy::Auto);
                assert!(a.reason.is_none());
            }
            // Nothing was sent to the instance beyond the warm-up session.
            assert_eq!(with._fake.chat_requests().len(), sent_after_warm_up);
            // Every decision was stored, none applied, and the pick is only recorded.
            let all = decisions(&with).await;
            assert_eq!(all.len(), 2, "primary/{stage}");
            assert!(all.iter().all(|d| !d.applied && d.chosen.is_some()));
        }
    }

    #[tokio::test]
    async fn an_unprobed_instance_is_probed_before_the_first_decision_and_the_decision_names_its_session(
    ) {
        let s = setup_with("mixed", "auto", true, false).await;
        let session = Uuid::new_v4();
        let choice = s
            .manager
            .resolve_provider_choice_for(&executor_request(), Some("proj"), Some(session))
            .await
            .unwrap_or_else(|e| panic!("resolve failed: {e:#}"));
        assert_eq!(
            choice.routed_by,
            RoutedBy::Auto,
            "the probe made local/m eligible"
        );
        assert_eq!(choice.provider_id, "local");
        let all = decisions(&s).await;
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].session_id, Some(session));
    }

    #[tokio::test]
    async fn a_manual_switch_to_another_model_counts_as_an_override_when_the_session_closes() {
        let s = setup("full", "auto", true).await;
        let session_id = open_routed_session(&s).await;

        s.manager
            .set_session_model(&session_id, "some-other-model")
            .await
            .unwrap_or_else(|e| panic!("switch failed: {e:#}"));
        s.manager.close_session(&session_id).await.unwrap();

        let closed = decisions(&s).await.remove(0);
        let outcome = closed.outcome.expect("the close recorded an outcome");
        assert!(outcome.overridden, "the switch by hand is counted");
        assert!(outcome.reward.is_some(), "an override closes with a reward");
    }

    #[tokio::test]
    async fn an_override_that_closes_a_losing_class_demotes_it_to_shadow() {
        use crate::chat::provider::cognitive::{
            decision::{CognitiveDecision, DecisionOutcome},
            load_routing, LearningStage,
        };
        let s = setup("full", "auto", true).await;
        // The smallest window the settings accept.
        s.graph
            .put_llm_setting(
                GLOBAL,
                "routing",
                &json!({
                    "mode": "full",
                    "stage": "auto",
                    "exploration_epsilon": 0.0,
                    "demote_after": 5
                })
                .to_string(),
            )
            .await
            .unwrap();
        let session_id = open_routed_session(&s).await;
        let routed = decisions(&s).await.remove(0);

        // Same class as the session's decision: four learnt choices that did badly
        // (the session's own, closed below, is the fifth) and five declarative ones
        // that did well.
        let store = Neo4jRoutingStore::new(s.graph.clone());
        let seed = |applied: bool, reward: f64, age: i64| {
            let mut d: CognitiveDecision = routed.clone();
            d.id = Uuid::new_v4();
            d.at = routed.at - ChronoDuration::seconds(age);
            d.applied = applied;
            d.stage = if applied {
                LearningStage::Auto
            } else {
                LearningStage::Shadow
            };
            d.session_id = None;
            d.outcome = Some(DecisionOutcome {
                reward: Some(reward),
                ..DecisionOutcome::default()
            });
            d
        };
        for age in 1..=4 {
            store.put_decision(&seed(true, 0.2, age)).await.unwrap();
        }
        for age in 1..=5 {
            store.put_decision(&seed(false, 0.9, age)).await.unwrap();
        }

        let before = load_routing(s.graph.as_ref(), Some("proj"))
            .await
            .unwrap()
            .0;
        assert!(before.stage == LearningStage::Auto);

        s.manager
            .set_session_model(&session_id, "some-other-model")
            .await
            .unwrap_or_else(|e| panic!("switch failed: {e:#}"));
        s.manager.close_session(&session_id).await.unwrap();

        let after = load_routing(s.graph.as_ref(), Some("proj"))
            .await
            .unwrap()
            .0;
        assert!(
            after.stage == LearningStage::Shadow,
            "the override closed a class the router does worse on: demoted"
        );
    }

    #[tokio::test]
    async fn mixed_routes_an_executor_by_auto_and_leaves_the_pilot_on_the_primary() {
        let s = setup("mixed", "auto", true).await;
        let executor = choose(&s, &executor_request()).await;
        assert_eq!(executor.provider_id, "local");
        assert_eq!(executor.model.as_deref(), Some("m"));
        assert_eq!(executor.routed_by, RoutedBy::Auto);
        assert_eq!(executor.route_rule.as_deref(), Some("auto:simple"));
        assert!(executor
            .reason
            .as_deref()
            .is_some_and(|r| r.contains("local/m")));

        let pilot = choose(&s, &pilot_request()).await;
        assert_eq!(
            pilot.provider_id, "claude-code",
            "the pilot keeps the primary"
        );
        assert_ne!(pilot.routed_by, RoutedBy::Auto);
        assert_eq!(
            pilot.shadow.as_ref().map(|p| p.0.as_str()),
            Some("local"),
            "what the router would have chosen is recorded"
        );
        let all = decisions(&s).await;
        assert_eq!(all.len(), 2);
        assert_eq!(all.iter().filter(|d| d.applied).count(), 1);
    }

    #[tokio::test]
    async fn full_routes_the_pilot_too() {
        let s = setup("full", "auto", true).await;
        let pilot = choose(&s, &pilot_request()).await;
        assert_eq!(pilot.provider_id, "local");
        assert_eq!(pilot.routed_by, RoutedBy::Auto);
        assert!(pilot
            .route_rule
            .as_deref()
            .is_some_and(|r| r.starts_with("auto:chat")));
        let executor = choose(&s, &executor_request()).await;
        assert_eq!(executor.routed_by, RoutedBy::Auto);
        assert!(decisions(&s).await.iter().all(|d| d.applied));
    }

    /// Opens a session on the automatic choice and waits for its first result.
    async fn open_routed_session(s: &Setup) -> String {
        let created = s
            .manager
            .create_session(&pilot_request())
            .await
            .unwrap_or_else(|e| panic!("create_session failed: {e:#}"));
        let mut rx = s.manager.subscribe(&created.session_id).await.unwrap();
        next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        created.session_id
    }

    #[tokio::test]
    async fn a_chat_decision_is_linked_to_its_session_and_a_plain_close_learns_nothing() {
        let s = setup("full", "auto", true).await;
        let session_id = open_routed_session(&s).await;
        let all = decisions(&s).await;
        assert_eq!(all.len(), 1, "one decision for the one routed session");
        let decision = &all[0];
        assert_eq!(
            decision.session_id.map(|id| id.to_string()).as_deref(),
            Some(session_id.as_str()),
            "the decision names the session it routed"
        );
        let arm = decision.arm().expect("the decision chose an arm");

        s.manager.close_session(&session_id).await.unwrap();

        let closed = decisions(&s).await.remove(0);
        let outcome = closed.outcome.expect("the close recorded what it saw");
        assert!(outcome.duration_ms.is_some(), "{outcome:?}");
        assert_eq!(
            outcome.reward, None,
            "a chat cannot tell success: no reward"
        );
        // Nothing was fed to the arm: an unknown outcome reads as a failure.
        let store = Neo4jRoutingStore::new(s.graph.clone());
        assert!(
            store.arm(&arm).await.unwrap().is_none(),
            "a plain close must not create or move the arm"
        );
        assert!(s.manager.open_decisions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_model_switched_by_hand_closes_the_decision_with_the_override_and_teaches_the_arm() {
        let s = setup("full", "auto", true).await;
        let session_id = open_routed_session(&s).await;
        let arm = decisions(&s).await[0].arm().unwrap();

        // The switch itself may be refused by the fake; the user's intent counts.
        let _ = s.manager.set_session_model(&session_id, "m2").await;
        s.manager.close_session(&session_id).await.unwrap();

        let closed = decisions(&s).await.remove(0);
        let outcome = closed.outcome.expect("closed");
        assert!(outcome.overridden, "{outcome:?}");
        assert!(outcome.reward.is_some_and(|r| r < 0.5), "{outcome:?}");
        let stats = Neo4jRoutingStore::new(s.graph.clone())
            .arm(&arm)
            .await
            .unwrap()
            .expect("the override taught the arm");
        assert!(
            stats.mean() < 0.5,
            "one failure on a Beta(1,1) prior: {stats:?}"
        );
    }

    #[tokio::test]
    async fn without_the_router_a_session_keeps_no_open_decision() {
        let s = setup("full", "auto", false).await;
        let created = s
            .manager
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap();
        let mut rx = s.manager.subscribe(&created.session_id).await.unwrap();
        next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        let session_id = created.session_id;
        assert!(s.manager.open_decisions.lock().unwrap().is_empty());
        s.manager.close_session(&session_id).await.unwrap();
        assert!(decisions(&s).await.is_empty());
    }

    #[tokio::test]
    async fn wire_learning_gives_the_per_turn_router_and_the_runner_a_handle() {
        let s = setup("full", "auto", true).await;
        let manager = Arc::new(s.manager);
        crate::runner::routing::install(None);
        assert!(crate::runner::routing::installed().is_none());
        assert!(manager.turn_routing.configured().is_none());

        crate::chat::provider::cognitive::wiring::wire_learning(&manager);
        assert!(manager.turn_routing.configured().is_some());
        let handle = crate::runner::routing::installed().expect("installed");
        let pool = handle.pool.facts(Some("proj")).await;
        assert!(
            pool.iter().any(|f| f.provider_id == "local"),
            "the runner pool is the manager's pool: {pool:?}"
        );
        crate::runner::routing::install(None);
    }

    #[tokio::test]
    async fn the_shadow_stage_applies_nothing_in_any_mode() {
        for mode in ["primary", "mixed", "full"] {
            let s = setup(mode, "shadow", true).await;
            for r in [pilot_request(), executor_request()] {
                let c = choose(&s, &r).await;
                assert_eq!(c.provider_id, "claude-code", "{mode}/shadow");
                assert_ne!(c.routed_by, RoutedBy::Auto);
            }
            let all = decisions(&s).await;
            assert_eq!(all.len(), 2);
            assert!(all.iter().all(|d| !d.applied), "{mode}/shadow");
        }
    }

    #[tokio::test]
    async fn the_advisory_stage_applies_to_executors_only() {
        let s = setup("full", "advisory", true).await;
        assert_eq!(
            choose(&s, &executor_request()).await.routed_by,
            RoutedBy::Auto
        );
        assert_ne!(choose(&s, &pilot_request()).await.routed_by, RoutedBy::Auto);
    }

    #[tokio::test]
    async fn a_choice_the_caller_named_is_never_substituted() {
        let s = setup("full", "auto", true).await;
        let mut explicit_provider = pilot_request();
        explicit_provider.provider = Some("claude-code".into());
        let mut explicit_model = executor_request();
        explicit_model.model = Some("some-model".into());
        let mut alias = executor_request();
        alias.task_alias = Some("fast".into());
        let mut run = executor_request();
        run.run_provider = Some("claude-code".into());
        for r in [explicit_provider, explicit_model, run] {
            let c = choose(&s, &r).await;
            assert_eq!(c.provider_id, "claude-code");
            assert_ne!(c.routed_by, RoutedBy::Auto);
        }
        // An alias that is not defined is the caller's problem, not a reason to route.
        assert!(s
            .manager
            .resolve_provider_choice(&alias, Some("proj"))
            .await
            .map(|c| c.routed_by != RoutedBy::Auto)
            .unwrap_or(true));
        let all = decisions(&s).await;
        assert_eq!(all.len(), 4, "persisted in every case");
        assert!(all.iter().all(|d| !d.applied && d.chosen.is_none()));
        assert!(all.iter().all(|d| d.reason.contains("explicit")));
    }

    #[tokio::test]
    async fn without_a_candidate_the_declared_rules_apply_and_the_decision_says_so() {
        let s = setup("full", "auto", true).await;
        // The project withdraws its consent: nothing is eligible.
        s.graph
            .delete_llm_setting("project:proj", "consent:local")
            .await
            .ok();
        for (key, _) in s.graph.list_llm_settings("project:proj", "").await.unwrap() {
            s.graph
                .delete_llm_setting("project:proj", &key)
                .await
                .unwrap();
        }
        let c = choose(&s, &pilot_request()).await;
        assert_eq!(c.provider_id, "claude-code");
        assert_ne!(c.routed_by, RoutedBy::Auto);
        let all = decisions(&s).await;
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].reason, "no_candidate");
        assert!(!all[0].applied && all[0].chosen.is_none());
    }
}

// ============================================================================
// Per-turn model routing (`full` mode): `before_turn` on a scripted provider.
// ============================================================================

/// `before_turn` chooses the model of a turn. A scripted provider plays the
/// harness of nexus (it calls the hooks and honours `set_model_live`); the
/// decider is a fake that answers by turn, so the tests pin WHAT the hook does
/// with an answer, not how a scorer would choose.
mod turn_routing {
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use nexus_claude::agent::{
        AgentProvider, AgentSession, Capabilities, HookSupport, ModelInfo, ProviderError,
        ProviderHealth, ProviderKind, ResumeToken, SessionSpec,
    };
    use nexus_claude::testkit::{RecordedCall, Script, ScriptedProvider};

    use super::*;
    use crate::chat::agent_hooks::PoolSource;
    use crate::chat::provider::cognitive::candidates::ModelFacts;
    use crate::chat::provider::cognitive::decision::{
        CognitiveDecision, DecideRequest, Decider, Pick,
    };
    use crate::chat::provider::cognitive::{LearningStage, ProviderRoutingMode, ROUTING_KEY};

    /// The scripted provider under the id (and kind) of Claude Code, so the
    /// engine hands it the graph hooks.
    struct AsClaudeCode(Arc<ScriptedProvider>);

    #[async_trait]
    impl AgentProvider for AsClaudeCode {
        fn id(&self) -> &str {
            "claude-code"
        }
        fn kind(&self) -> ProviderKind {
            ProviderKind::ClaudeCode
        }
        async fn health(&self) -> ProviderHealth {
            self.0.health().await
        }
        async fn catalog(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            self.0.catalog().await
        }
        fn capabilities(&self, model: Option<&str>) -> Capabilities {
            self.0.capabilities(model)
        }
        async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn AgentSession>, ProviderError> {
            self.0.open(spec).await
        }
        async fn resume(
            &self,
            spec: SessionSpec,
            token: ResumeToken,
        ) -> Result<Arc<dyn AgentSession>, ProviderError> {
            self.0.resume(spec, token).await
        }
    }

    impl super::super::agent_runtime::ProviderSource for AsClaudeCode {
        fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
            (provider_id == "claude-code")
                .then(|| Arc::new(AsClaudeCode(Arc::clone(&self.0))) as Arc<dyn AgentProvider>)
        }
    }

    struct Pool;

    #[async_trait]
    impl PoolSource for Pool {
        async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
            ["small", "big"]
                .iter()
                .map(|m| ModelFacts {
                    provider_id: provider_id.to_owned(),
                    model: (*m).to_owned(),
                    supports_tools: true,
                    supports_images: true,
                    context_window: Some(200_000),
                    price: None,
                    cost_basis: nexus_claude::agent::CostBasis::Unknown,
                    healthy: Some(true),
                    allowed_for_project: true,
                    sandboxed: false,
                })
                .collect()
        }
    }

    /// What the fake decider does on one call.
    #[derive(Clone)]
    enum Answer {
        /// Pick this model, applied when the request's mode and stage say so.
        Pick(&'static str),
        /// Pick the model the session runs on now.
        Stay,
        Fail,
        Hang,
    }

    /// Answers from a queue (the last answer repeats) and remembers every request.
    struct FakeDecider {
        answers: StdMutex<VecDeque<Answer>>,
        requests: StdMutex<Vec<DecideRequest>>,
        decisions: StdMutex<Vec<CognitiveDecision>>,
    }

    impl FakeDecider {
        fn new(answers: Vec<Answer>) -> Arc<Self> {
            Arc::new(Self {
                answers: StdMutex::new(answers.into()),
                requests: StdMutex::new(Vec::new()),
                decisions: StdMutex::new(Vec::new()),
            })
        }
        fn calls(&self) -> usize {
            self.requests.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl Decider for FakeDecider {
        async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
            self.requests.lock().unwrap().push(request.clone());
            let answer = {
                let mut queue = self.answers.lock().unwrap();
                if queue.len() > 1 {
                    queue.pop_front().unwrap()
                } else {
                    queue.front().cloned().unwrap_or(Answer::Fail)
                }
            };
            let model: String = match answer {
                Answer::Pick(model) => model.to_owned(),
                Answer::Stay => request
                    .current
                    .as_ref()
                    .map(|pick| pick.model.clone())
                    .unwrap_or_default(),
                Answer::Fail => anyhow::bail!("the scorer is down"),
                Answer::Hang => {
                    std::future::pending::<()>().await;
                    unreachable!()
                }
            };
            let applied = request.settings.mode == ProviderRoutingMode::Full
                && request.settings.stage == LearningStage::Auto;
            let decision = CognitiveDecision {
                id: Uuid::new_v4(),
                at: Utc::now(),
                signature: request.signature.clone(),
                chosen: Some(Pick::new(
                    request.restrict_provider.clone().unwrap_or_default(),
                    model.clone(),
                )),
                score: Some(0.9),
                explored: false,
                reason: format!("fake: {model}"),
                alternatives: Vec::new(),
                applied,
                mode: request.settings.mode,
                stage: request.settings.stage,
                session_id: request.session_id,
                task_id: None,
                run_id: None,
                turn_index: request.turn_index,
                outcome: None,
                used: None,
            };
            self.decisions.lock().unwrap().push(decision.clone());
            Ok(decision)
        }
    }

    fn caps(set_model_live: bool) -> Capabilities {
        let mut caps = Capabilities::none();
        caps.set_model_live = set_model_live;
        caps.hooks = HookSupport::InProtocol;
        caps.per_session_mcp = true;
        caps.tools = true;
        caps
    }

    struct Rig {
        manager: ChatManager,
        decider: Arc<FakeDecider>,
        provider: Arc<ScriptedProvider>,
        sid: String,
        rx: broadcast::Receiver<ChatEvent>,
    }

    async fn rig(
        mode: &str,
        stage: &str,
        set_model_live: bool,
        explicit_model: Option<&str>,
        answers: Vec<Answer>,
    ) -> Rig {
        let graph = Arc::new(MockGraphStore::new());
        graph
            .put_llm_setting(
                GLOBAL,
                ROUTING_KEY,
                &json!({ "mode": mode, "stage": stage }).to_string(),
            )
            .await
            .unwrap();
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: PathBuf::from("/nonexistent/mcp"),
            max_sessions: 10,
            ..Default::default()
        };
        let provider = Arc::new(ScriptedProvider::new(
            "claude-code",
            Script::builder().capabilities(caps(set_model_live)).build(),
        ));
        let decider = FakeDecider::new(answers);
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_provider_source(Arc::new(AsClaudeCode(Arc::clone(&provider))))
            .with_turn_decider(decider.clone(), Arc::new(Pool));
        let mut req = request(None, None, "default");
        // No opening message: every turn below is sent by `turn`, from index 0.
        req.message = String::new();
        req.model = explicit_model.map(str::to_owned);
        let created = manager.create_session(&req).await.unwrap();
        let rx = manager.subscribe(&created.session_id).await.unwrap();
        Rig {
            manager,
            decider,
            provider,
            sid: created.session_id,
            rx,
        }
    }

    impl Rig {
        /// Sends one message and waits for its turn to end; returns the models
        /// announced by `model_changed` during it.
        async fn turn(&mut self, text: &str) -> Vec<String> {
            self.manager.send_message(&self.sid, text).await.unwrap();
            let mut changed = Vec::new();
            loop {
                match next_event(&mut self.rx, |e| {
                    matches!(
                        e,
                        ChatEvent::ModelChanged { .. }
                            | ChatEvent::StreamingStatus {
                                is_streaming: false
                            }
                    )
                })
                .await
                {
                    ChatEvent::ModelChanged { model } => changed.push(model),
                    _ => return changed,
                }
            }
        }
    }

    const SIMPLE: &str = "rename this variable";
    const DEBUG: &str = "why does this crash with a stack trace error";

    #[tokio::test]
    async fn full_auto_a_simple_turn_then_a_debug_turn_changes_the_model_of_the_second() {
        let mut r = rig(
            "full",
            "auto",
            true,
            None,
            vec![Answer::Stay, Answer::Pick("big")],
        )
        .await;
        let first = r.turn(SIMPLE).await;
        let second = r.turn(DEBUG).await;
        assert_eq!(r.decider.calls(), 2);
        assert!(
            first.is_empty(),
            "the simple turn keeps its model: {first:?}"
        );
        assert_eq!(second, ["big"], "the debug turn runs on the other model");
        let requests = r.decider.requests.lock().unwrap();
        assert_eq!(requests[0].turn_index, Some(0));
        assert_eq!(requests[1].turn_index, Some(1));
        assert_eq!(
            requests[1].restrict_provider.as_deref(),
            Some("claude-code")
        );
        assert!(requests[0].session_id.is_some());
        assert_eq!(requests[0].pool.len(), 2, "the pool of that provider only");
        assert!(r
            .provider
            .calls()
            .iter()
            .any(|c| matches!(c, RecordedCall::SendTurn(_))));
    }

    #[tokio::test]
    async fn without_set_model_live_there_is_no_directive_and_no_error() {
        let mut r = rig("full", "auto", false, None, vec![Answer::Pick("big")]).await;
        assert!(r.turn(SIMPLE).await.is_empty());
        assert!(r.turn(DEBUG).await.is_empty());
        assert_eq!(r.decider.calls(), 0, "nothing to apply: nothing is asked");
    }

    #[tokio::test]
    async fn the_shadow_stage_records_the_decision_and_changes_nothing() {
        let mut r = rig("full", "shadow", true, None, vec![Answer::Pick("big")]).await;
        assert!(r.turn(SIMPLE).await.is_empty());
        assert!(r.turn(DEBUG).await.is_empty());
        assert_eq!(r.decider.calls(), 2, "asked, so the decision is recorded");
        let decisions = r.decider.decisions.lock().unwrap();
        assert!(decisions
            .iter()
            .all(|d| !d.applied && d.stage == LearningStage::Shadow));
    }

    #[tokio::test]
    async fn mixed_and_primary_modes_leave_the_pilot_alone() {
        for mode in ["mixed", "primary"] {
            let mut r = rig(mode, "auto", true, None, vec![Answer::Pick("big")]).await;
            assert!(r.turn(DEBUG).await.is_empty(), "{mode}");
            assert_eq!(r.decider.calls(), 0, "{mode}");
        }
    }

    #[tokio::test]
    async fn a_model_named_by_the_request_is_never_replaced() {
        let mut r = rig(
            "full",
            "auto",
            true,
            Some("explicit-model"),
            vec![Answer::Pick("big")],
        )
        .await;
        assert!(r.turn(DEBUG).await.is_empty());
        assert_eq!(r.decider.calls(), 0);
    }

    #[tokio::test]
    async fn never_two_changes_in_consecutive_turns() {
        let mut r = rig(
            "full",
            "auto",
            true,
            None,
            vec![
                Answer::Pick("small"),
                Answer::Pick("big"),
                Answer::Pick("small"),
                Answer::Pick("big"),
                Answer::Pick("big"),
            ],
        )
        .await;
        let mut flips = Vec::new();
        for text in [SIMPLE, DEBUG, SIMPLE, DEBUG, DEBUG] {
            flips.push(!r.turn(text).await.is_empty());
        }
        for pair in flips.windows(2) {
            assert!(!(pair[0] && pair[1]), "two changes in a row: {flips:?}");
        }
        assert!(
            flips.iter().any(|f| *f),
            "the guard does not freeze the router: {flips:?}"
        );
        // The turn that follows a change is still asked, but as a shadow decision.
        let decisions = r.decider.decisions.lock().unwrap();
        assert!(
            decisions.iter().any(|d| !d.applied),
            "recorded unapplied: {decisions:?}"
        );
    }

    #[tokio::test]
    async fn a_failing_or_hanging_decider_never_fails_the_turn() {
        let mut r = rig("full", "auto", true, None, vec![Answer::Fail, Answer::Hang]).await;
        assert!(
            r.turn(SIMPLE).await.is_empty(),
            "an error: the turn completes"
        );
        assert!(
            r.turn(DEBUG).await.is_empty(),
            "a timeout: the turn completes"
        );
        assert_eq!(r.decider.calls(), 2);
    }

    #[tokio::test]
    async fn a_manual_model_change_ends_the_automatic_ones_for_the_session() {
        let mut r = rig("full", "auto", true, None, vec![Answer::Pick("big")]).await;
        assert!(r
            .manager
            .set_session_model(&r.sid, "manual-model")
            .await
            .unwrap());
        // The confirmation of the user's own change is the first model_changed.
        let changed = r.turn(DEBUG).await;
        assert_eq!(
            changed,
            ["manual-model"],
            "no automatic change after it: {changed:?}"
        );
        assert_eq!(r.decider.calls(), 0);
    }

    #[tokio::test]
    async fn a_pinned_model_stays_pinned_when_the_session_is_resumed() {
        use crate::chat::manager::OpeningTurn;
        let r = rig("full", "auto", true, None, vec![Answer::Pick("big")]).await;
        r.manager
            .set_session_model(&r.sid, "manual-model")
            .await
            .unwrap();
        // A resume builds a new router that knows nothing of the hand change: the
        // stored pin must carry it.
        let router = r
            .manager
            .register_turn_router(
                &r.sid,
                "claude-code",
                "manual-model",
                None,
                OpeningTurn {
                    routing_pool: None,
                    routing_mode: None,
                    explicit_model: false,
                    permission_mode: None,
                    message: DEBUG,
                    next_turn: 0,
                },
            )
            .await
            .expect("a router is registered");
        router.set_model_live(true);
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        assert_eq!(r.decider.calls(), 0, "a pinned model is never routed again");
    }

    /// Opens a router the way a conversation does, with the mode its request named.
    async fn reopen_with_mode(
        r: &Rig,
        mode: Option<ProviderRoutingMode>,
    ) -> Arc<crate::chat::agent_hooks::TurnRouter> {
        reopen_with_pool(r, mode, None).await
    }

    async fn reopen_with_pool(
        r: &Rig,
        mode: Option<ProviderRoutingMode>,
        pool: Option<&[&str]>,
    ) -> Arc<crate::chat::agent_hooks::TurnRouter> {
        use crate::chat::manager::OpeningTurn;
        let router = r
            .manager
            .register_turn_router(
                &r.sid,
                "claude-code",
                "small",
                None,
                OpeningTurn {
                    routing_pool: pool.map(|models| {
                        models
                            .iter()
                            .map(|m| crate::chat::types::RoutingPoolEntry {
                                provider: "claude-code".to_owned(),
                                model: (*m).to_owned(),
                            })
                            .collect()
                    }),
                    routing_mode: mode,
                    explicit_model: false,
                    permission_mode: None,
                    message: DEBUG,
                    next_turn: 0,
                },
            )
            .await
            .expect("a router is registered");
        router.set_model_live(true);
        router
    }

    #[tokio::test]
    async fn a_conversation_opened_in_strict_is_never_routed_even_when_the_settings_say_full() {
        let r = rig("full", "auto", true, None, vec![Answer::Pick("big")]).await;
        reopen_with_mode(&r, Some(ProviderRoutingMode::Primary)).await;
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        assert_eq!(r.decider.calls(), 0, "strict: PO is not even asked");
    }

    #[tokio::test]
    async fn a_conversation_opened_in_auto_is_routed_even_when_the_settings_say_primary() {
        let r = rig("primary", "auto", true, None, vec![Answer::Pick("big")]).await;
        reopen_with_mode(&r, Some(ProviderRoutingMode::Full)).await;
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        assert_eq!(
            r.decider.calls(),
            1,
            "auto: PO decides for this conversation"
        );
    }

    #[tokio::test]
    async fn a_mixed_conversation_is_routed_among_its_pool_only() {
        let mut r = rig("primary", "auto", true, None, vec![Answer::Pick("big")]).await;
        reopen_with_pool(&r, Some(ProviderRoutingMode::Mixed), Some(&["big"])).await;
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        let requests = r.decider.requests.lock().unwrap().clone();
        assert_eq!(
            requests.len(),
            1,
            "a pool is routed whatever the settings say"
        );
        let offered: Vec<&str> = requests[0].pool.iter().map(|f| f.model.as_str()).collect();
        assert_eq!(offered, ["big"], "only the ticked models are candidates");
        drop(requests);
        let ev = next_event(&mut r.rx, |e| matches!(e, ChatEvent::ModelChanged { .. })).await;
        assert!(matches!(ev, ChatEvent::ModelChanged { ref model } if model == "big"));
    }

    #[tokio::test]
    async fn a_pilot_named_with_a_pool_is_not_a_pin() {
        use crate::chat::manager::OpeningTurn;
        let r = rig("primary", "auto", true, None, vec![Answer::Pick("big")]).await;
        // The request names its pilot (explicit_model) AND a pool: PO still routes.
        let router = r
            .manager
            .register_turn_router(
                &r.sid,
                "claude-code",
                "small",
                None,
                OpeningTurn {
                    routing_mode: Some(ProviderRoutingMode::Mixed),
                    routing_pool: Some(vec![
                        crate::chat::types::RoutingPoolEntry {
                            provider: "claude-code".to_owned(),
                            model: "small".to_owned(),
                        },
                        crate::chat::types::RoutingPoolEntry {
                            provider: "claude-code".to_owned(),
                            model: "big".to_owned(),
                        },
                    ]),
                    explicit_model: true,
                    permission_mode: None,
                    message: DEBUG,
                    next_turn: 0,
                },
            )
            .await
            .expect("a router is registered");
        router.set_model_live(true);
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        assert_eq!(r.decider.calls(), 1);
    }

    #[tokio::test]
    async fn without_a_mode_in_the_request_the_settings_decide() {
        let r = rig("primary", "auto", true, None, vec![Answer::Pick("big")]).await;
        reopen_with_mode(&r, None).await;
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        assert_eq!(r.decider.calls(), 0);
    }

    #[tokio::test]
    async fn the_legacy_engines_entry_point_applies_the_same_decision_as_a_set_model() {
        // `apply_turn_directive` is what the legacy engine (Claude Code CLI) calls from
        // `send_message` before writing the message; here it drives a session whose
        // `set_model` the scripted provider records.
        let mut r = rig("full", "auto", true, None, vec![Answer::Pick("big")]).await;
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        let ev = next_event(&mut r.rx, |e| matches!(e, ChatEvent::ModelChanged { .. })).await;
        assert!(matches!(ev, ChatEvent::ModelChanged { ref model } if model == "big"));
        assert!(r
            .provider
            .calls()
            .iter()
            .any(|c| matches!(c, RecordedCall::SetModel(m) if m == "big")));
        // Same guard as on the agent engine: no second change on the next turn.
        r.manager.apply_turn_directive(&r.sid, DEBUG).await;
        assert_eq!(r.decider.calls(), 2);
        let set_models = r
            .provider
            .calls()
            .iter()
            .filter(|c| matches!(c, RecordedCall::SetModel(_)))
            .count();
        assert_eq!(set_models, 1);
    }
}

// ── Provider switch (B-SW) ──────────────────────────────────────────────────
//
// A conversation moves to another provider: a new session on the target gets the
// earlier conversation as a relay in front of the next message, the old session
// closes, and a refusal leaves the old one untouched.

mod provider_switch {
    use super::*;
    use crate::chat::types::SwitchProviderError;

    fn two_turn_script() -> Value {
        json!([
            sse_route("Call the ping tool now", vec![
                delta(json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ]),
            {"method": "GET", "path": "/v1/models", "status": 200,
             "body": {"object": "list", "data": [{"id": "m", "context_length": 32000}]}},
            // Listed first: the relayed request also contains "hi there".
            sse_route("second question", vec![
                delta(json!({"content": "the answer"})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}),
                json!("[DONE]"),
            ]),
            sse_route("hi there", vec![
                delta(json!({"content": "hello from "})),
                delta(json!({"content": "the fake model"})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}),
                json!("[DONE]"),
            ]),
        ])
    }

    struct World {
        fake: FakeOpenAi,
        graph: Arc<MockGraphStore>,
        manager: ChatManager,
    }

    async fn world() -> World {
        let fake = FakeOpenAi::start(two_turn_script());
        let graph = Arc::new(MockGraphStore::new());
        for id in ["local", "local2"] {
            let mut record = instance(&fake, "none");
            record.id = id.into();
            store_instance(&graph, &record).await;
            consent(&graph, "proj", id, &fake.origin()).await;
        }
        // A third instance the project never consented to.
        let mut unconsented = instance(&fake, "none");
        unconsented.id = "local3".into();
        store_instance(&graph, &unconsented).await;
        let manager = manager(graph.clone(), true);
        World {
            fake,
            graph,
            manager,
        }
    }

    /// Opens a session on `local`, runs its first turn, and waits until the turn is stored.
    async fn first_turn(w: &World) -> String {
        let created = w
            .manager
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap_or_else(|e| panic!("open failed: {e:#}"));
        let mut rx = w.manager.subscribe(&created.session_id).await.unwrap();
        next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        stored_until(w, &created.session_id, |events| {
            events.iter().any(|e| e.event_type == "assistant_text")
        })
        .await;
        created.session_id
    }

    async fn stored_until(
        w: &World,
        session_id: &str,
        done: impl Fn(&[crate::neo4j::models::ChatEventRecord]) -> bool,
    ) -> Vec<crate::neo4j::models::ChatEventRecord> {
        let id = Uuid::parse_str(session_id).unwrap();
        for _ in 0..100 {
            let events = w.graph.get_chat_events(id, 0, 500).await.unwrap();
            if done(&events) {
                return events;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("the expected events were never stored for {session_id}");
    }

    #[tokio::test]
    async fn a_conversation_moves_to_another_provider_with_its_history_relayed() {
        let w = world().await;
        let old = first_turn(&w).await;

        let moved = w
            .manager
            .switch_session_provider(&old, "local2", None, "second question", None)
            .await
            .unwrap_or_else(|e| panic!("switch failed: {e:#}"));
        assert_ne!(moved.session_id, old);
        assert_eq!(moved.previous_session_id, old);
        assert!(
            moved.relayed_entries >= 2,
            "the user and the assistant turn"
        );
        assert_eq!(moved.omitted_entries, 0);

        // The new session runs on the target and answers.
        // Read what was stored, not the live stream: the answer may be out before we subscribe.
        stored_until(&w, &moved.session_id, |events| {
            events
                .iter()
                .any(|e| e.event_type == "assistant_text" && e.data.contains("the answer"))
        })
        .await;
        let node = w
            .graph
            .get_chat_session(Uuid::parse_str(&moved.session_id).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(node.provider_id.as_deref(), Some("local2"));
        assert_eq!(node.project_slug.as_deref(), Some("proj"));

        // The model of the target saw the earlier conversation, then the new question.
        let chats = w.fake.chat_requests();
        let turn = chats
            .iter()
            .find(|r| r["body"].to_string().contains("second question"))
            .expect("the relayed turn");
        let body = turn["body"].to_string();
        assert!(body.contains("<conversation_relay"), "{body}");
        assert!(body.contains("hi there"), "the earlier user message");
        assert!(
            body.contains("hello from the fake model"),
            "the earlier answer"
        );
        assert!(
            body.find("<conversation_relay").unwrap() < body.find("second question").unwrap(),
            "the relay comes first"
        );

        // The conversation shows the user's words only: the relay is not a message.
        let stored = stored_until(&w, &moved.session_id, |events| {
            events.iter().any(|e| e.event_type == "user_message")
        })
        .await;
        let first_user = stored
            .iter()
            .find(|e| e.event_type == "user_message")
            .unwrap();
        assert!(first_user.data.contains("second question"));
        assert!(
            !first_user.data.contains("conversation_relay"),
            "the relay must not be stored as the user's message: {}",
            first_user.data
        );

        // The old session is closed and the handoff is traceable.
        assert!(!w.manager.is_session_active(&old).await);
        let note = w
            .graph
            .get_llm_setting(&format!("handoff:{}", moved.session_id), "note")
            .await
            .unwrap()
            .expect("the handoff note");
        let note: Value = serde_json::from_str(&note).unwrap();
        assert_eq!(note["from_session"], old.as_str());
        assert_eq!(note["from_provider"], "local");
    }

    #[tokio::test]
    async fn a_refused_switch_leaves_the_old_session_untouched() {
        let w = world().await;
        let old = first_turn(&w).await;
        let sessions_before = w.fake.chat_requests().len();

        // An unknown provider, and one the project never consented to.
        for target in ["ghost", "local3"] {
            let err = w
                .manager
                .switch_session_provider(&old, target, None, "second question", None)
                .await
                .expect_err(target);
            assert!(
                err.downcast_ref::<SwitchProviderError>().is_none(),
                "{target}: a refusal from the resolver, not a switch error: {err:#}"
            );
            assert!(
                w.manager.is_session_active(&old).await,
                "{target}: the old session must stay open"
            );
        }
        assert_eq!(
            w.fake.chat_requests().len(),
            sessions_before,
            "nothing was sent anywhere"
        );
    }

    #[tokio::test]
    async fn the_same_provider_and_an_empty_message_are_refused_before_anything_opens() {
        let w = world().await;
        let old = first_turn(&w).await;

        let same = w
            .manager
            .switch_session_provider(&old, "local", None, "second question", None)
            .await
            .expect_err("same provider");
        assert_eq!(
            same.downcast_ref::<SwitchProviderError>(),
            Some(&SwitchProviderError::SameProvider("local".into()))
        );
        let empty = w
            .manager
            .switch_session_provider(&old, "local2", None, "   ", None)
            .await
            .expect_err("empty message");
        assert_eq!(
            empty.downcast_ref::<SwitchProviderError>(),
            Some(&SwitchProviderError::EmptyMessage)
        );
        let missing = w
            .manager
            .switch_session_provider(
                &Uuid::new_v4().to_string(),
                "local2",
                None,
                "second question",
                None,
            )
            .await
            .expect_err("unknown session");
        assert_eq!(
            missing.downcast_ref::<SwitchProviderError>(),
            Some(&SwitchProviderError::NotFound)
        );
        assert!(w.manager.is_session_active(&old).await);
    }
}
