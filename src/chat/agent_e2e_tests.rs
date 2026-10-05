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
        "{} is missing: build the nexus fakes (see the module doc)",
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
    let mut config = super::config::ChatConfig::default();
    config.provider_path = ProviderPath::Agent;
    config.mcp_server_path = fake_bin("fake_mcp");
    config.jwt_secret = secure.then(|| "test-secret-test-secret-test-secret".to_string());
    config.max_sessions = 10;
    ChatManager::new_without_memory(dyn_graph, state.meili, config)
}

fn request(provider: Option<&str>, project: Option<&str>, mode: &str) -> ChatRequest {
    ChatRequest {
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
async fn trust_is_refused_for_a_provider_without_a_sandbox() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let manager = manager(graph.clone(), true);
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "bypassPermissions"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (422, "unsupported"));
    assert!(
        fake.chat_requests().is_empty(),
        "refused before the model was called"
    );
    assert!(graph
        .list_llm_settings("journal", "")
        .await
        .unwrap()
        .is_empty());
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
async fn the_legacy_engine_cannot_open_a_registered_instance() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let state = mock_app_state();
    let dyn_graph: Arc<dyn GraphStore> = graph.clone();
    let mut config = super::config::ChatConfig::default();
    config.provider_path = ProviderPath::Legacy;
    let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config);
    let err = manager
        .create_session(&request(Some("local"), Some("proj"), "default"))
        .await
        .unwrap_err();
    assert_eq!(failure(&err), (503, "provider_unavailable"));
    assert!(fake.requests().is_empty());
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
