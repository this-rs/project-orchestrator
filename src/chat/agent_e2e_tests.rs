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

pub(super) fn fakes_dir() -> PathBuf {
    std::env::var_os("NEXUS_FAKES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.target-nexus-fakes/debug")
        })
}

pub(super) fn fake_bin(name: &str) -> PathBuf {
    let path = fakes_dir().join(name);
    assert!(
        path.exists(),
        "{} is missing: build the nexus fakes (see the module doc; CI does it in the step 'Build the nexus fake servers')",
        path.display()
    );
    path
}

/// A running `fake_openai`, killed on drop.
pub(super) struct FakeOpenAi {
    child: Child,
    port: u16,
    dir: tempfile::TempDir,
}

impl FakeOpenAi {
    pub(super) fn start(routes: Value) -> Self {
        Self::start_for(routes, 120_000)
    }

    /// [`Self::start`] with the watchdog of the fake set to `max_runtime_ms`.
    pub(super) fn start_for(routes: Value, max_runtime_ms: u64) -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        let script = dir.path().join("script.json");
        std::fs::write(&script, routes.to_string()).unwrap();
        let mut child = Command::new(fake_bin("fake_openai"))
            .env("FAKE_OPENAI_SCRIPT", &script)
            .env(
                "FAKE_OPENAI_REQUESTS_OUT",
                dir.path().join("requests.jsonl"),
            )
            .env("FAKE_OPENAI_MAX_RUNTIME_MS", max_runtime_ms.to_string())
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

    pub(super) fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    pub(super) fn origin(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub(super) fn requests(&self) -> Vec<Value> {
        std::fs::read_to_string(self.dir.path().join("requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    pub(super) fn chat_requests(&self) -> Vec<Value> {
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

pub(super) fn delta(d: Value) -> Value {
    json!({"choices": [{"index": 0, "delta": d}]})
}

pub(super) fn sse_route(body_contains: &str, events: Vec<Value>) -> Value {
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

pub(super) fn instance(fake: &FakeOpenAi, credential_ref: &str) -> InstanceRecord {
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

pub(super) async fn store_instance(graph: &MockGraphStore, record: &InstanceRecord) {
    graph
        .put_llm_setting(
            GLOBAL,
            &format!("{INSTANCE_PREFIX}{}", record.id),
            &serde_json::to_string(record).unwrap(),
        )
        .await
        .unwrap();
}

pub(super) async fn consent(graph: &MockGraphStore, slug: &str, id: &str, origin: &str) {
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
        nexus_tools_path: None,
        nexus_browser_path: None,
        jwt_secret: secure.then(|| "test-secret-test-secret-test-secret".to_string()),
        max_sessions: 10,
        ..Default::default()
    };
    ChatManager::new_without_memory(dyn_graph, state.meili, config)
}

pub(super) fn request(provider: Option<&str>, project: Option<&str>, mode: &str) -> ChatRequest {
    ChatRequest {
        access: None,
        routing_pool: None,
        routing_mode: None,
        attachments: Vec::new(),
        refs: Vec::new(),
        message: "hi there".into(),
        session_id: None,
        cwd: std::env::temp_dir().display().to_string(),
        project_slug: project.map(str::to_string),
        model: None,
        provider: provider.map(str::to_string),
        task_alias: None,
        persona_alias: None,
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

/// H6: a third party a person opens in trust is given the FULL profile, whose
/// schemas are larger than the restricted ones. The window check measures what
/// the session is actually given: a window that holds the restricted schemas but
/// not the full ones refuses a session in trust and still opens one in ask.
#[tokio::test]
async fn the_window_check_measures_the_profile_the_session_is_given() {
    let restricted = crate::auth::tool_profile::ToolProfile::Restricted
        .filter_tools(crate::mcp::tools::all_tools());
    let full =
        crate::auth::tool_profile::ToolProfile::Full.filter_tools(crate::mcp::tools::all_tools());
    let tokens = |tools: &Vec<crate::mcp::protocol::ToolDefinition>| {
        (serde_json::to_string(tools).unwrap().len() / 4) as u64
    };
    let window = tokens(&restricted) * 2 + 1;
    assert!(
        window < tokens(&full) * 2,
        "the full profile must be larger for this test"
    );
    let mut routes = script().as_array().unwrap().clone();
    routes[1] = json!({"method": "GET", "path": "/v1/models", "status": 200,
        "body": {"object": "list", "data": [{"id": "m", "context_length": window}]}});
    let fake = FakeOpenAi::start(Value::Array(routes));
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let manager = manager(graph.clone(), true);
    let person = crate::auth::jwt::Claims {
        sub: Uuid::new_v4().to_string(),
        email: "alice@example.com".into(),
        name: "Alice".into(),
        iat: 0,
        exp: 0,
        token_type: None,
        scope: None,
        jti: None,
    };
    let mut ask = request(Some("local"), Some("proj"), "default");
    ask.user_claims = Some(person.clone());
    ask.message = String::new();
    manager
        .create_session(&ask)
        .await
        .unwrap_or_else(|e| panic!("the restricted schemas fit: {e:#}"));
    let mut trust = request(Some("local"), Some("proj"), "bypassPermissions");
    trust.user_claims = Some(person);
    trust.message = String::new();
    let err = manager.create_session(&trust).await.unwrap_err();
    assert_eq!(failure(&err), (422, "context_too_small"));
}

/// B40: a `nexus-tools` that cannot be run (NEXUS_TOOLS_PATH naming a file that is
/// gone, or not executable) leaves the native session with the project-orchestrator
/// tools only. It never refuses the session.
#[tokio::test]
async fn a_nexus_tools_that_cannot_run_does_not_refuse_the_native_session() {
    let fake = FakeOpenAi::start(script());
    let graph = Arc::new(MockGraphStore::new());
    store_instance(&graph, &instance(&fake, "none")).await;
    consent(&graph, "proj", "local", &fake.origin()).await;
    let dir = tempfile::TempDir::new().unwrap();
    let not_executable = dir.path().join("nexus-tools");
    std::fs::write(&not_executable, "#!/bin/sh\nexit 1\n").unwrap();
    for program in [dir.path().join("missing/nexus-tools"), not_executable] {
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let config = super::config::ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: Some(program.clone()),
            nexus_browser_path: None,
            jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
            max_sessions: 10,
            ..Default::default()
        };
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config);
        let req = request(Some("local"), Some("proj"), "default");
        let created = manager
            .create_session(&req)
            .await
            .unwrap_or_else(|e| panic!("{}: {e:#}", program.display()));
        let mut rx = manager.subscribe(&created.session_id).await.unwrap();
        next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        // And it says so: `nexus_tools` is among what the session does not have, on
        // the `system_init` the interface reads (not only a line in the server log).
        // The system_init is persisted by the out-of-turn pump, possibly after the Result.
        let uuid = Uuid::parse_str(&created.session_id).unwrap();
        let mut init = None;
        let mut seen = Vec::new();
        for _ in 0..100 {
            let events = graph.get_chat_events(uuid, 0, 100).await.unwrap();
            seen = events.iter().map(|r| r.event_type.clone()).collect();
            init = events
                .iter()
                .filter_map(|r| serde_json::from_str::<ChatEvent>(&r.data).ok())
                .find(|e| matches!(e, ChatEvent::SystemInit { .. }));
            if init.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let Some(ChatEvent::SystemInit {
            degraded_features, ..
        }) = init
        else {
            panic!("{}: no system_init persisted: {seen:?}", program.display());
        };
        let degraded = degraded_features.unwrap_or_default();
        assert!(
            degraded
                .iter()
                .any(|f| f == super::agent_runtime::NEXUS_TOOLS_FEATURE),
            "{}: {degraded:?}",
            program.display()
        );
    }
}

#[test]
fn the_tool_schemas_have_a_size_and_an_unknown_window_is_not_a_refusal() {
    use nexus_claude::agent::{Capabilities, ContextWindow, ContextWindowSource};
    let tokens =
        super::manager::tool_schema_tokens(crate::auth::tool_profile::ToolProfile::Restricted);
    assert!(
        tokens > 500,
        "the restricted profile still has tools: {tokens}"
    );
    let mut caps = Capabilities::none();
    assert!(
        super::manager::window_holds_the_tools(
            &caps,
            crate::auth::tool_profile::ToolProfile::Restricted
        )
        .is_ok(),
        "unknown window"
    );
    caps.context_window = Some(ContextWindow {
        value: tokens * 2 + 1,
        source: ContextWindowSource::Probed,
    });
    assert!(super::manager::window_holds_the_tools(
        &caps,
        crate::auth::tool_profile::ToolProfile::Restricted
    )
    .is_ok());
    caps.context_window = Some(ContextWindow {
        value: tokens,
        source: ContextWindowSource::Probed,
    });
    assert!(super::manager::window_holds_the_tools(
        &caps,
        crate::auth::tool_profile::ToolProfile::Restricted
    )
    .is_err());
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

    fn pool_of(models: &[(&str, &str)]) -> Option<Vec<crate::chat::types::RoutingPoolEntry>> {
        Some(
            models
                .iter()
                .map(|(provider, model)| crate::chat::types::RoutingPoolEntry {
                    provider: (*provider).into(),
                    model: (*model).into(),
                })
                .collect(),
        )
    }

    /// The pool restricts the decision taken when the conversation OPENS, not only the
    /// per-turn ones: a model nobody ticked never opens it.
    #[tokio::test]
    async fn the_models_ticked_are_the_only_candidates_when_a_conversation_opens() {
        let s = setup("full", "auto", true).await;
        let mut r = pilot_request();
        r.routing_mode = Some(crate::chat::provider::cognitive::ProviderRoutingMode::Mixed);
        r.routing_pool = pool_of(&[("local", "not-m"), ("local", "other")]);
        let c = choose(&s, &r).await;
        assert_eq!(
            (c.provider_id.as_str(), c.model.as_deref(), c.routed_by),
            ("local", Some("not-m"), RoutedBy::Request),
            "local/m was not ticked: the first ticked model opens it"
        );
    }

    /// A pool is routed at open inside it, whatever the settings' mode (primary here).
    #[tokio::test]
    async fn a_pool_is_routed_at_open_even_when_the_settings_say_primary() {
        let s = setup("primary", "auto", true).await;
        let mut r = pilot_request();
        r.routing_mode = Some(crate::chat::provider::cognitive::ProviderRoutingMode::Mixed);
        r.routing_pool = pool_of(&[("local", "m"), ("claude-code", "x")]);
        let c = choose(&s, &r).await;
        assert_eq!(c.routed_by, RoutedBy::Auto);
        assert_eq!(
            (c.provider_id.as_str(), c.model.as_deref()),
            ("local", Some("m"))
        );
    }

    /// Auto in the chat menu (`routing_mode: full`) routes the opening too, whatever the settings.
    #[tokio::test]
    async fn auto_in_the_request_routes_the_opening_even_when_the_settings_say_primary() {
        let s = setup("primary", "auto", true).await;
        let mut r = pilot_request();
        r.routing_mode = Some(crate::chat::provider::cognitive::ProviderRoutingMode::Full);
        assert_eq!(choose(&s, &r).await.routed_by, RoutedBy::Auto);
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

    async fn put_aliases(s: &Setup, aliases: serde_json::Value) {
        s.graph
            .put_llm_setting(GLOBAL, "model_aliases", &aliases.to_string())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn the_persona_level_sits_between_the_task_alias_and_the_run() {
        let s = setup("full", "auto", true).await;
        put_aliases(
            &s,
            json!([
                {"alias": "fast", "provider": "local", "model": "m"},
                {"alias": "deep", "provider": "local", "model": "m-deep"}
            ]),
        )
        .await;
        // Persona alone: it is the persona level and names its model.
        let mut persona = executor_request();
        persona.persona_alias = Some("fast".into());
        let c = choose(&s, &persona).await;
        assert_eq!(
            (c.routed_by, c.provider_id.as_str(), c.model.as_deref()),
            (RoutedBy::Persona, "local", Some("m"))
        );
        // Persona beats the run.
        let mut over_run = persona.clone();
        over_run.run_provider = Some("claude-code".into());
        assert_eq!(choose(&s, &over_run).await.routed_by, RoutedBy::Persona);
        // The task alias beats the persona.
        let mut over_persona = persona.clone();
        over_persona.task_alias = Some("deep".into());
        let c = choose(&s, &over_persona).await;
        assert_eq!(
            (c.routed_by, c.model.as_deref()),
            (RoutedBy::Task, Some("m-deep"))
        );
        // The user's request beats everything.
        let mut requested = persona.clone();
        requested.provider = Some("claude-code".into());
        assert_eq!(choose(&s, &requested).await.routed_by, RoutedBy::Request);
        // Nothing the router chose replaced a named model.
        assert!(decisions(&s).await.iter().all(|d| !d.applied));
    }

    #[tokio::test]
    async fn a_preference_no_alias_maps_falls_through_and_never_reaches_a_decision() {
        let s = setup("full", "auto", true).await;
        let sentinel = "SENTINEL-persona-free-text-7f3a";
        let mut r = executor_request();
        r.persona_alias = Some(sentinel.into());
        let c = choose(&s, &r).await;
        // The choice stays the persona's to make: the router does not take over.
        assert_ne!(c.routed_by, RoutedBy::Auto);
        assert_ne!(c.routed_by, RoutedBy::Persona);
        assert!(!format!("{c:?}").contains(sentinel));
        let all = decisions(&s).await;
        assert!(!all.is_empty());
        assert!(all.iter().all(|d| !d.applied && d.chosen.is_none()));
        assert!(!serde_json::to_string(&all).unwrap().contains(sentinel));
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
                    window_unknown: None,
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
        graph: Arc<MockGraphStore>,
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
            nexus_tools_path: None,
            nexus_browser_path: None,
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
            graph,
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

    /// A turn the queue starts (a held message sent to an idle session here) is
    /// routed on ITS text, not on the message sent before it.
    #[tokio::test]
    async fn a_turn_started_by_a_held_message_is_routed_on_its_own_text() {
        use crate::chat::provider::cognitive::signature::{ContextHints, TaskSignature};
        let mut r = rig("full", "shadow", true, None, vec![Answer::Stay]).await;
        r.turn(SIMPLE).await;
        let held = r.manager.queue_user_message(&r.sid, DEBUG).await.unwrap();
        assert!(!held, "an idle session sends it at once");
        next_event(&mut r.rx, |e| {
            matches!(
                e,
                ChatEvent::StreamingStatus {
                    is_streaming: false
                }
            )
        })
        .await;
        let requests = r.decider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let expected =
            TaskSignature::from_chat_request(DEBUG, false, None, ContextHints::default());
        assert_eq!(
            format!("{:?}", requests[1].signature),
            format!("{expected:?}"),
            "the second turn is routed on its own message"
        );
    }

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
                    provider_imposed: false,
                    moved_in: false,
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
                    provider_imposed: false,
                    moved_in: false,
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
        // Two ticks ("medium" is not offered by this provider): one alone would be strict.
        reopen_with_pool(
            &r,
            Some(ProviderRoutingMode::Mixed),
            Some(&["big", "medium"]),
        )
        .await;
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
                    provider_imposed: false,
                    moved_in: false,
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

    // ── PUT /api/chat/sessions/{id}/routing: the menu of an open conversation ──

    fn auto() -> crate::chat::types::SessionRoutingRequest {
        crate::chat::types::SessionRoutingRequest {
            auto: true,
            routing_pool: Vec::new(),
        }
    }

    fn ticked(models: &[(&str, &str)]) -> crate::chat::types::SessionRoutingRequest {
        crate::chat::types::SessionRoutingRequest {
            auto: false,
            routing_pool: models
                .iter()
                .map(|(provider, model)| crate::chat::types::RoutingPoolEntry {
                    provider: (*provider).to_owned(),
                    model: (*model).to_owned(),
                })
                .collect(),
        }
    }

    async fn pinned(r: &Rig) -> bool {
        r.graph
            .get_llm_setting(&format!("session:{}", r.sid), "model_pinned")
            .await
            .unwrap()
            .is_some_and(|v| v == "true")
    }

    /// "Rendre la main a PO": Auto on a conversation whose model was imposed releases it.
    #[tokio::test]
    async fn auto_on_an_open_conversation_hands_an_imposed_model_back_to_po() {
        let mut r = rig(
            "primary",
            "auto",
            true,
            Some("small"),
            vec![Answer::Pick("big")],
        )
        .await;
        assert!(r.turn(DEBUG).await.is_empty());
        assert_eq!(r.decider.calls(), 0, "imposed: PO is not even asked");
        assert!(pinned(&r).await);

        let node = r
            .manager
            .set_session_routing(&r.sid, &auto())
            .await
            .unwrap();
        assert_eq!(node.routing_mode.as_deref(), Some("full"));
        assert_eq!(node.routed_by.as_deref(), Some("auto"));
        assert_eq!(node.routing_pool, None);
        assert!(
            !pinned(&r).await,
            "the pin is released, a resume routes too"
        );

        assert_eq!(r.turn(DEBUG).await, ["big"], "the next turn is routed");
    }

    /// A model changed by hand ends the automatic changes; Auto starts them again.
    #[tokio::test]
    async fn auto_after_a_model_changed_by_hand_routes_the_next_turn_again() {
        let mut r = rig("full", "auto", true, None, vec![Answer::Pick("big")]).await;
        r.manager.set_session_model(&r.sid, "small").await.unwrap();
        let by_hand = r.turn(DEBUG).await;
        assert!(!by_hand.iter().any(|m| m == "big"), "{by_hand:?}");
        assert_eq!(r.decider.calls(), 0, "set by hand: PO is not asked");

        r.manager
            .set_session_routing(&r.sid, &auto())
            .await
            .unwrap();
        assert!(!pinned(&r).await);
        assert_eq!(r.turn(DEBUG).await, ["big"]);
    }

    /// A16: one model ticked is imposed, and never substituted afterwards.
    #[tokio::test]
    async fn one_model_ticked_on_an_open_conversation_is_imposed_and_never_substituted() {
        let mut r = rig("full", "auto", true, None, vec![Answer::Pick("big")]).await;
        let node = r
            .manager
            .set_session_routing(&r.sid, &ticked(&[("claude-code", "small")]))
            .await
            .unwrap();
        assert_eq!(node.routing_mode.as_deref(), Some("primary"));
        assert_eq!(node.routed_by.as_deref(), Some("request"));
        assert_eq!(node.model, "small", "set now, not at some later turn");
        assert!(pinned(&r).await);

        let mut seen = r.turn(DEBUG).await;
        seen.extend(r.turn(DEBUG).await);
        assert!(
            !seen.iter().any(|m| m == "big"),
            "the router never replaces an imposed model: {seen:?}"
        );
        assert_eq!(
            r.decider.calls(),
            0,
            "full settings, but this conversation is strict"
        );
    }

    #[tokio::test]
    async fn models_ticked_on_an_open_conversation_restrict_its_next_turns() {
        let mut r = rig("primary", "auto", true, None, vec![Answer::Pick("big")]).await;
        assert!(r.turn(DEBUG).await.is_empty());
        assert_eq!(
            r.decider.calls(),
            0,
            "primary: not routed before the change"
        );

        let node = r
            .manager
            .set_session_routing(
                &r.sid,
                &ticked(&[("claude-code", "big"), ("local-llama", "qwen")]),
            )
            .await
            .unwrap();
        assert_eq!(node.routing_mode.as_deref(), Some("mixed"));
        let stored: Vec<crate::chat::types::RoutingPoolEntry> =
            serde_json::from_str(node.routing_pool.as_deref().unwrap()).unwrap();
        assert_eq!(
            stored.len(),
            2,
            "every tick is kept, the other provider's too"
        );

        assert_eq!(r.turn(DEBUG).await, ["big"]);
        let requests = r.decider.requests.lock().unwrap();
        let offered: Vec<&str> = requests[0].pool.iter().map(|f| f.model.as_str()).collect();
        assert_eq!(
            offered,
            ["big"],
            "only this provider's ticked models are candidates"
        );
    }

    #[tokio::test]
    async fn a_conversations_routing_never_writes_the_global_or_project_settings() {
        let r = rig("primary", "shadow", true, None, vec![Answer::Stay]).await;
        let settings = || async {
            let all = r.graph.llm_settings.read().await;
            let mut kept: Vec<((String, String), String)> = all
                .iter()
                .filter(|((scope, _), _)| !scope.starts_with("session:"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            kept.sort();
            kept
        };
        let before = settings().await;
        for body in [
            auto(),
            ticked(&[("claude-code", "small")]),
            ticked(&[("claude-code", "small"), ("claude-code", "big")]),
            auto(),
        ] {
            r.manager.set_session_routing(&r.sid, &body).await.unwrap();
        }
        assert_eq!(
            settings().await,
            before,
            "only the session's own scope changes"
        );
    }

    #[tokio::test]
    async fn a_routing_that_needs_another_provider_or_no_model_is_refused_and_changes_nothing() {
        use crate::chat::types::SessionRoutingError as Refused;
        let r = rig("primary", "auto", true, None, vec![Answer::Stay]).await;
        let refusal = |body| {
            let manager = &r.manager;
            let sid = r.sid.clone();
            async move {
                manager
                    .set_session_routing(&sid, &body)
                    .await
                    .unwrap_err()
                    .downcast::<Refused>()
                    .unwrap()
            }
        };
        assert_eq!(
            refusal(ticked(&[("local-llama", "qwen")])).await,
            Refused::OtherProvider("claude-code".into())
        );
        assert_eq!(
            refusal(ticked(&[("local-llama", "qwen"), ("local-llama", "mini")])).await,
            Refused::OtherProvider("claude-code".into())
        );
        assert_eq!(refusal(ticked(&[])).await, Refused::EmptyPool);
        assert_eq!(
            refusal(ticked(&[("claude-code", " ")])).await,
            Refused::BlankEntry(0)
        );
        let node = r
            .graph
            .get_chat_session(Uuid::parse_str(&r.sid).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(node.routing_mode, None);
        assert_eq!(node.routing_pool, None);
        let missing = r
            .manager
            .set_session_routing(&Uuid::new_v4().to_string(), &auto())
            .await
            .unwrap_err();
        assert_eq!(missing.downcast_ref::<Refused>(), Some(&Refused::NotFound));
    }

    /// One model ticked when the conversation opens is strict: imposed, not a pool of one.
    #[tokio::test]
    async fn one_model_ticked_at_open_is_strict() {
        let mut req = request(None, None, "default");
        req.routing_mode = Some(ProviderRoutingMode::Mixed);
        req.routing_pool = Some(vec![crate::chat::types::RoutingPoolEntry {
            provider: "claude-code".into(),
            model: "small".into(),
        }]);
        let settled = req.settled_routing().expect("a pool of one is settled");
        assert_eq!(settled.provider.as_deref(), Some("claude-code"));
        assert_eq!(settled.model.as_deref(), Some("small"));
        assert_eq!(settled.routing_mode, Some(ProviderRoutingMode::Primary));
        assert_eq!(settled.routing_pool, None);
        req.routing_pool = Some(Vec::new());
        assert_eq!(req.settled_routing().unwrap().routing_pool, None);
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

    // ── Both directions, across engines: Claude Code → native → Claude Code ──
    //
    // Claude Code stays on its historical engine (the Claude CLI): here the
    // `fake_claude` of nexus, behind a wrapper that hands it a transcript and records
    // what it reads. The native target is `fake_openai`. Decision 896b7d7c: the two
    // are interchangeable at any point of a conversation.

    use crate::chat::provider::cognitive::decider::CognitiveRouting;
    use crate::chat::provider::cognitive::decision::CognitiveDecision;
    use crate::chat::provider::cognitive::store::{DecisionFilter, RoutingArmStore};
    use crate::neo4j::models::ChatEventRecord;
    use crate::neo4j::routing::Neo4jRoutingStore;

    const CLAUDE_ANSWER: &str = "first answer, from claude";

    /// A Claude CLI: `fake_claude` behind a shell wrapper. Every CLI it starts plays
    /// one turn (waits for the user's message, answers [`CLAUDE_ANSWER`]) and appends
    /// every line it reads to `stdin.jsonl`.
    struct FakeClaude {
        dir: tempfile::TempDir,
    }

    impl FakeClaude {
        fn new() -> Self {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let transcript = dir.path().join("transcript.jsonl");
            let lines = [
                json!({"op": "capture_hooks", "optional": true, "timeout_ms": 3000}),
                json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 30000}),
                json!({"op": "emit_json", "json": {"type": "system", "subtype": "init",
                    "session_id": "fake-cli-session", "model": "fake-claude", "tools": [],
                    "permissionMode": "default", "apiKeySource": "none"}}),
                json!({"op": "emit_json", "json": {"type": "assistant", "message": {
                    "id": "msg_fake_1", "type": "message", "role": "assistant",
                    "model": "fake-claude", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": CLAUDE_ANSWER}]}}}),
                json!({"op": "emit_json", "json": {"type": "result", "subtype": "success",
                    "duration_ms": 1, "duration_api_ms": 1, "is_error": false, "num_turns": 1,
                    "session_id": "fake-cli-session", "total_cost_usd": 0.0,
                    "result": CLAUDE_ANSWER}}),
                json!({"op": "wait_eof", "optional": true, "timeout_ms": 110000}),
            ];
            let text: Vec<String> = lines.iter().map(Value::to_string).collect();
            std::fs::write(&transcript, text.join("\n")).unwrap();
            let wrapper = dir.path().join("claude");
            std::fs::write(
                &wrapper,
                format!(
                    "#!/bin/sh\nFAKE_CLAUDE_TRANSCRIPT='{}' FAKE_CLAUDE_STDIN_OUT='{}' \
                     FAKE_CLAUDE_MAX_RUNTIME_MS=120000 exec '{}' \"$@\"\n",
                    transcript.display(),
                    dir.path().join("stdin.jsonl").display(),
                    fake_bin("fake_claude").display()
                ),
            )
            .unwrap();
            std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
            Self { dir }
        }

        fn path(&self) -> String {
            self.dir.path().join("claude").display().to_string()
        }

        /// Every line the CLIs read, in order.
        fn stdin(&self) -> String {
            std::fs::read_to_string(self.dir.path().join("stdin.jsonl")).unwrap_or_default()
        }
    }

    /// `world()` as deployed: Claude Code on the Claude CLI (the legacy engine), every
    /// other provider on the agent engine, the cognitive router wired.
    async fn hybrid_world(cli: &FakeClaude) -> World {
        let mut w = world().await;
        let dyn_graph: Arc<dyn GraphStore> = w.graph.clone();
        let config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Legacy,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: None,
            nexus_browser_path: None,
            jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
            max_sessions: 10,
            ..Default::default()
        };
        let store = Arc::new(Neo4jRoutingStore::new(w.graph.clone()));
        w.manager = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config)
            .with_cognitive_routing(CognitiveRouting::new(store));
        w.manager.update_claude_cli_path(Some(cli.path())).await;
        w
    }

    async fn stored_events(w: &World, session_id: &str) -> Vec<ChatEvent> {
        w.graph
            .get_chat_events(Uuid::parse_str(session_id).unwrap(), 0, 500)
            .await
            .unwrap()
            .iter()
            .filter_map(|r| serde_json::from_str(&r.data).ok())
            .collect()
    }

    fn relayed_of(events: &[ChatEvent]) -> Vec<&ChatEvent> {
        events
            .iter()
            .filter(|e| matches!(e, ChatEvent::ConversationRelayed { .. }))
            .collect()
    }

    async fn decision_of(w: &World, session_id: &str) -> Option<CognitiveDecision> {
        let id = Uuid::parse_str(session_id).unwrap();
        Neo4jRoutingStore::new(w.graph.clone())
            .decisions(&DecisionFilter::default())
            .await
            .unwrap()
            .into_iter()
            .find(|d| d.session_id == Some(id))
    }

    /// The line the CLIs read that carries `needle`.
    fn cli_line_with(cli: &FakeClaude, needle: &str) -> String {
        cli.stdin()
            .lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no line with {needle:?} reached the CLI: {}", cli.stdin()))
            .to_string()
    }

    #[tokio::test]
    async fn claude_code_to_native_and_back_the_target_knows_the_history_and_the_thread_says_so() {
        let cli = FakeClaude::new();
        let w = hybrid_world(&cli).await;

        // 1. A conversation on Claude Code, served by the Claude CLI.
        let cc1 = w
            .manager
            .create_session(&request(None, Some("proj"), "default"))
            .await
            .unwrap_or_else(|err| panic!("open on claude-code failed: {err:#}"))
            .session_id;
        assert!(w.manager.is_session_active(&cc1).await);
        assert!(
            !w.manager.agent_runtime.owns(&cc1).await,
            "Claude Code runs on the legacy engine"
        );
        stored_until(&w, &cc1, |events| {
            events
                .iter()
                .any(|ev| ev.event_type == "assistant_text" && ev.data.contains(CLAUDE_ANSWER))
        })
        .await;
        cli_line_with(&cli, "hi there");
        // The memory conversation the move must keep.
        w.graph
            .update_chat_session(
                Uuid::parse_str(&cc1).unwrap(),
                None,
                None,
                None,
                None,
                Some("conv-kept".into()),
                None,
            )
            .await
            .unwrap();
        let first_decision = decision_of(&w, &cc1)
            .await
            .expect("the routed opening left a decision on the session");
        assert!(
            first_decision.outcome.is_none(),
            "open while the session runs"
        );
        let mut old_rx = w.manager.subscribe(&cc1).await.unwrap();

        // 2. Claude Code → native: the target answers knowing the history.
        let moved = w
            .manager
            .switch_session_provider(&cc1, "local", None, "second question", None)
            .await
            .unwrap_or_else(|err| panic!("switch to local failed: {err:#}"));
        let native = moved.session_id.clone();
        assert_eq!((moved.relayed_entries, moved.omitted_entries), (2, 0));
        assert_eq!(moved.conversation_id.as_deref(), Some("conv-kept"));
        assert!(
            w.manager.agent_runtime.owns(&native).await,
            "native: agent engine"
        );
        stored_until(&w, &native, |events| {
            events
                .iter()
                .any(|ev| ev.event_type == "assistant_text" && ev.data.contains("the answer"))
        })
        .await;
        let body = w
            .fake
            .chat_requests()
            .iter()
            .map(|r| r["body"].to_string())
            .find(|b| b.contains("second question"))
            .expect("the relayed turn reached the native model");
        assert!(
            body.contains("<conversation_relay from=\\\"claude-code\\\" to=\\\"local\\\">"),
            "{body}"
        );
        assert!(
            body.contains("hi there") && body.contains(CLAUDE_ANSWER),
            "{body}"
        );

        // The thread it left says where it went, then closes; the thread it reached
        // says what was relayed, before the user's message.
        let on_old = next_event(&mut old_rx, |ev| {
            matches!(ev, ChatEvent::ConversationRelayed { .. })
        })
        .await;
        let as_json = |ev: &ChatEvent| serde_json::to_value(ev).unwrap();
        assert_eq!(
            as_json(&on_old),
            as_json(&ChatEvent::ConversationRelayed {
                from_session_id: cc1.clone(),
                to_session_id: native.clone(),
                from_provider: "claude-code".into(),
                to_provider: "local".into(),
                relayed_entries: 2,
                omitted_entries: 0,
                moved_by: "user".into(),
                conversation_id: Some("conv-kept".into()),
            })
        );
        next_event(&mut old_rx, |ev| {
            matches!(ev, ChatEvent::SessionClosed { .. })
        })
        .await;
        assert!(
            !w.manager.is_session_active(&cc1).await,
            "the old session is closed"
        );
        assert_eq!(
            relayed_of(&stored_events(&w, &cc1).await).len(),
            1,
            "stored on the old thread"
        );
        let new_thread = stored_events(&w, &native).await;
        let relay_at = new_thread
            .iter()
            .position(|ev| matches!(ev, ChatEvent::ConversationRelayed { .. }))
            .expect("stated on the new thread");
        let message_at = new_thread
            .iter()
            .position(|ev| matches!(ev, ChatEvent::UserMessage { content } if content == "second question"))
            .expect("the user's message");
        assert!(
            relay_at < message_at,
            "the relay is stated before the message"
        );
        assert_eq!(
            as_json(&new_thread[relay_at]),
            as_json(&on_old),
            "both threads carry the same statement"
        );

        // The routing decision of the session that closed is closed too.
        assert!(w.manager.open_decisions.lock().unwrap().get(&cc1).is_none());
        let closed = decision_of(&w, &cc1).await.unwrap();
        assert!(
            closed.outcome.is_some(),
            "closed with what the session saw: {closed:?}"
        );

        // 3. Native → Claude Code: back, with everything so far.
        let back = w
            .manager
            .switch_session_provider(&native, "claude-code", None, "third question", None)
            .await
            .unwrap_or_else(|err| panic!("switch back to claude-code failed: {err:#}"));
        let cc2 = back.session_id.clone();
        assert_eq!(back.previous_session_id, native);
        assert_eq!(
            (back.relayed_entries, back.omitted_entries),
            (2, 0),
            "the native session's own two entries (the relay it was given is not a message)"
        );
        assert_eq!(back.conversation_id.as_deref(), Some("conv-kept"));
        stored_until(&w, &cc2, |events| {
            events.iter().any(|ev| ev.event_type == "result")
        })
        .await;
        let third = cli_line_with(&cli, "third question");
        assert!(
            third.contains(r#"<conversation_relay from=\"local\" to=\"claude-code\">"#),
            "{third}"
        );
        assert!(
            third.contains("second question") && third.contains("the answer"),
            "the native turn is in the relay: {third}"
        );
        assert!(
            !w.manager.agent_runtime.owns(&native).await,
            "the native session is closed"
        );
        assert!(w.manager.is_session_active(&cc2).await);
        let cc2_thread = stored_events(&w, &cc2).await;
        assert!(
            matches!(relayed_of(&cc2_thread).as_slice(),
                [ChatEvent::ConversationRelayed { from_provider, to_provider, .. }]
                    if from_provider == "local" && to_provider == "claude-code"),
            "{cc2_thread:?}"
        );
        for id in [&native, &cc2] {
            let node = w
                .graph
                .get_chat_session(Uuid::parse_str(id).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(node.conversation_id.as_deref(), Some("conv-kept"), "{id}");
            assert_eq!(
                node.routed_by.as_deref(),
                Some("request"),
                "the user chose: explicit"
            );
        }
        w.manager.close_session(&cc2).await.unwrap();
    }

    #[tokio::test]
    async fn a_switch_from_claude_code_refused_by_consent_states_nothing_and_closes_nothing() {
        let cli = FakeClaude::new();
        let w = hybrid_world(&cli).await;
        let cc1 = w
            .manager
            .create_session(&request(None, Some("proj"), "default"))
            .await
            .unwrap()
            .session_id;
        stored_until(&w, &cc1, |events| {
            events.iter().any(|ev| ev.event_type == "result")
        })
        .await;
        let sent_before = w.fake.chat_requests().len();

        // No consent of the project for `local3`: a typed refusal.
        let err = w
            .manager
            .switch_session_provider(&cc1, "local3", None, "second question", None)
            .await
            .expect_err("no consent");
        assert_eq!(failure(&err), (403, "endpoint_not_allowed"));

        assert!(
            w.manager.is_session_active(&cc1).await,
            "the conversation stays"
        );
        assert!(
            relayed_of(&stored_events(&w, &cc1).await).is_empty(),
            "nothing stated"
        );
        assert!(
            decision_of(&w, &cc1).await.unwrap().outcome.is_none(),
            "its decision stays open"
        );
        assert_eq!(
            w.fake.chat_requests().len(),
            sent_before,
            "nothing was sent"
        );
        w.manager.close_session(&cc1).await.unwrap();
    }

    #[tokio::test]
    async fn a_history_longer_than_the_targets_window_is_bounded_and_the_omission_is_stated() {
        let w = world().await;
        // A stored Claude Code conversation far larger than 40 % of a 32k window.
        let mut node = crate::test_helpers::test_chat_session(Some("proj"));
        node.provider_id = Some("claude-code".into());
        node.cwd = std::env::temp_dir().display().to_string();
        let old = node.id;
        w.graph.create_chat_session(&node).await.unwrap();
        let filler = "w".repeat(2_000);
        let records: Vec<ChatEventRecord> = (0..80)
            .map(|i| ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: old,
                seq: i + 1,
                event_type: "user_message".into(),
                data: serde_json::to_string(&ChatEvent::UserMessage {
                    content: format!("turn-{i:02} {filler}"),
                })
                .unwrap(),
                created_at: Utc::now(),
            })
            .collect();
        w.graph.store_chat_events(old, records).await.unwrap();

        let moved = w
            .manager
            .switch_session_provider(&old.to_string(), "local", None, "second question", None)
            .await
            .unwrap_or_else(|e| panic!("switch failed: {e:#}"));
        assert!(moved.omitted_entries > 0, "{moved:?}");
        assert!(moved.relayed_entries > 0, "{moved:?}");
        assert_eq!(moved.relayed_entries + moved.omitted_entries, 80);

        stored_until(&w, &moved.session_id, |events| {
            events.iter().any(|e| e.event_type == "assistant_text")
        })
        .await;
        let body = w
            .fake
            .chat_requests()
            .iter()
            .map(|r| r["body"].to_string())
            .find(|b| b.contains("second question"))
            .expect("the relayed turn");
        assert!(
            body.contains("left out to fit your context window"),
            "the omission is stated to the model"
        );
        assert!(body.contains("turn-79"), "the newest turn is kept");
        assert!(!body.contains("turn-00"), "the oldest turn is dropped");
        // The relay alone (the request also carries the system prompt and the tool
        // schemas) stays within its budget: 40 % of a 32k window = 51 200 chars, or the
        // 48 000 of an unknown window, whichever the target reported at the switch.
        let start = body.find("<conversation_relay").unwrap();
        let end = body.find("</conversation_relay>").unwrap();
        let relay_chars = body[start..end].chars().count();
        assert!(
            relay_chars <= 51_200 + 1_000,
            "the relay is bounded by the window: {relay_chars} chars"
        );
        // The thread says how much was left out: never a silent truncation.
        let stated = stored_events(&w, &moved.session_id).await;
        match relayed_of(&stated).as_slice() {
            [ChatEvent::ConversationRelayed {
                relayed_entries,
                omitted_entries,
                ..
            }] => {
                assert_eq!(*relayed_entries, moved.relayed_entries);
                assert_eq!(*omitted_entries, moved.omitted_entries);
            }
            other => panic!("one statement expected: {other:?}"),
        }
        // Stated on the thread it left as well, even though it was not live.
        assert_eq!(
            relayed_of(&stored_events(&w, &old.to_string()).await).len(),
            1
        );
    }

    // ── The router moves a conversation (P1c): mode `full`, stage `auto` ──
    //
    // Before a turn starts, the decision of a conversation in `full` may name another
    // provider: the conversation then moves through the same relay as the switch route,
    // `moved_by: auto`, and the turn's message is answered by the new provider only.

    mod auto_move {
        use super::*;
        use crate::chat::agent_hooks::PoolSource;
        use crate::chat::provider::cognitive::candidates::ModelFacts;
        use crate::chat::provider::cognitive::decision::{
            DecideRequest, Decider, DecisionAlternative, Pick,
        };
        use crate::chat::provider::cognitive::{LearningStage, ProviderRoutingMode, ROUTING_KEY};
        use crate::chat::types::{RoutingPoolEntry, SessionRoutingRequest};
        use std::collections::VecDeque;
        use std::sync::Mutex as StdMutex;

        /// The instances of `world()` as the routing pool sees them: `local3` without the
        /// project's consent.
        struct Instances;

        #[async_trait::async_trait]
        impl PoolSource for Instances {
            async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
                self.project_pool(None)
                    .await
                    .into_iter()
                    .filter(|f| f.provider_id == provider_id)
                    .collect()
            }
            async fn project_pool(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
                [
                    ("claude-code", true),
                    ("local", true),
                    ("local2", true),
                    ("local3", false),
                ]
                .iter()
                .map(|(provider, allowed)| ModelFacts {
                    provider_id: (*provider).into(),
                    model: if *provider == "claude-code" {
                        "fake-claude"
                    } else {
                        "m"
                    }
                    .into(),
                    supports_tools: true,
                    supports_images: false,
                    context_window: Some(32_000),
                    window_unknown: None,
                    price: None,
                    cost_basis: nexus_claude::agent::CostBasis::Free,
                    healthy: Some(true),
                    allowed_for_project: *allowed,
                    sandboxed: false,
                })
                .collect()
            }
        }

        /// Answers each provider check (a request across providers) with the next
        /// provider of its queue (the last one repeats), clearly ahead of the current pair;
        /// a model decision of the session's own provider keeps the current model.
        struct Mover {
            answers: StdMutex<VecDeque<&'static str>>,
            checks: StdMutex<Vec<DecideRequest>>,
        }

        impl Mover {
            fn new(answers: Vec<&'static str>) -> Arc<Self> {
                Arc::new(Self {
                    answers: StdMutex::new(answers.into()),
                    checks: StdMutex::new(Vec::new()),
                })
            }
            fn checks(&self) -> Vec<DecideRequest> {
                self.checks.lock().unwrap().clone()
            }
        }

        #[async_trait::async_trait]
        impl Decider for Mover {
            async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
                let current = request.current.clone().unwrap_or_else(|| Pick::new("", ""));
                let chosen = match &request.restrict_provider {
                    Some(_) => current.clone(),
                    None => {
                        self.checks.lock().unwrap().push(request.clone());
                        let mut queue = self.answers.lock().unwrap();
                        let provider = if queue.len() > 1 {
                            queue.pop_front().unwrap()
                        } else {
                            *queue.front().unwrap()
                        };
                        let model = if provider == "claude-code" {
                            "fake-claude"
                        } else {
                            "m"
                        };
                        Pick::new(provider, model)
                    }
                };
                let mut alternatives = vec![DecisionAlternative {
                    pick: chosen.clone(),
                    score: Some(0.9),
                    rejected: None,
                }];
                if chosen != current {
                    alternatives.push(DecisionAlternative {
                        pick: current,
                        score: Some(0.3),
                        rejected: None,
                    });
                }
                Ok(CognitiveDecision {
                    id: Uuid::new_v4(),
                    at: Utc::now(),
                    signature: request.signature.clone(),
                    chosen: Some(chosen.clone()),
                    score: Some(0.9),
                    explored: false,
                    reason: format!("fake: {}/{}", chosen.provider_id, chosen.model),
                    alternatives,
                    applied: request.settings.mode == ProviderRoutingMode::Full
                        && request.settings.stage == LearningStage::Auto,
                    mode: request.settings.mode,
                    stage: request.settings.stage,
                    session_id: request.session_id,
                    task_id: None,
                    run_id: None,
                    turn_index: request.turn_index,
                    outcome: None,
                    used: None,
                })
            }
        }

        async fn routing(w: &World, mode: &str, stage: &str) {
            w.graph
                .put_llm_setting(
                    GLOBAL,
                    ROUTING_KEY,
                    &json!({ "mode": mode, "stage": stage }).to_string(),
                )
                .await
                .unwrap();
        }

        /// `world()` with the router wired (and its decision store when `store`).
        async fn auto_world(
            stage: &str,
            answers: Vec<&'static str>,
            store: bool,
        ) -> (World, Arc<Mover>) {
            let mut w = world().await;
            routing(&w, "full", stage).await;
            let mover = Mover::new(answers);
            let mut manager = manager(w.graph.clone(), true);
            if store {
                manager = manager.with_cognitive_routing(CognitiveRouting::new(Arc::new(
                    Neo4jRoutingStore::new(w.graph.clone()),
                )));
            }
            w.manager = manager.with_turn_decider(mover.clone(), Arc::new(Instances));
            (w, mover)
        }

        /// "Auto" in the menu: the provider named at the opening is no longer imposed.
        async fn hand_back_to_po(w: &World, session_id: &str) {
            w.manager
                .set_session_routing(
                    session_id,
                    &SessionRoutingRequest {
                        auto: true,
                        routing_pool: Vec::new(),
                    },
                )
                .await
                .unwrap();
        }

        async fn idle(w: &World, session_id: &str) {
            for _ in 0..200 {
                if !w.manager.is_session_streaming(session_id).await {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            panic!("{session_id} never went idle");
        }

        fn requests_with(w: &World, needle: &str) -> Vec<String> {
            w.fake
                .chat_requests()
                .iter()
                .map(|r| r["body"].to_string())
                .filter(|b| b.contains(needle))
                .collect()
        }

        fn answers_on(events: &[ChatEvent]) -> usize {
            events
                .iter()
                .filter(|e| matches!(e, ChatEvent::AssistantText { .. }))
                .count()
        }

        /// Sends `text` to a session that stays, and waits for its answer there.
        async fn answered_in_place(w: &World, session_id: &str, text: &str) {
            let answers_before = answers_on(&stored_events(w, session_id).await);
            let relayed_before = relayed_of(&stored_events(w, session_id).await).len();
            w.manager.send_message(session_id, text).await.unwrap();
            stored_until(w, session_id, |records| {
                records
                    .iter()
                    .filter(|r| r.event_type == "assistant_text")
                    .count()
                    > answers_before
            })
            .await;
            assert!(
                w.manager.is_session_active(session_id).await,
                "the conversation stays"
            );
            assert_eq!(
                relayed_of(&stored_events(w, session_id).await).len(),
                relayed_before,
                "no move stated"
            );
        }

        #[tokio::test]
        async fn full_auto_a_turn_for_another_provider_moves_through_the_relay_and_is_answered_there(
        ) {
            let (w, mover) = auto_world("auto", vec!["local2"], false).await;
            let old = first_turn(&w).await;
            hand_back_to_po(&w, &old).await;
            idle(&w, &old).await;
            let mut old_rx = w.manager.subscribe(&old).await.unwrap();

            w.manager
                .send_message(&old, "second question")
                .await
                .unwrap();

            let on_old = next_event(&mut old_rx, |e| {
                matches!(e, ChatEvent::ConversationRelayed { .. })
            })
            .await;
            let ChatEvent::ConversationRelayed {
                from_session_id,
                to_session_id,
                from_provider,
                to_provider,
                moved_by,
                ..
            } = on_old.clone()
            else {
                unreachable!()
            };
            assert_eq!(moved_by, "auto");
            assert_eq!(
                (from_provider.as_str(), to_provider.as_str()),
                ("local", "local2")
            );
            assert_eq!(from_session_id, old);
            let new = to_session_id;

            // Answered by the new provider, sent ONCE, with the history relayed in front.
            stored_until(&w, &new, |records| {
                records
                    .iter()
                    .any(|r| r.event_type == "assistant_text" && r.data.contains("the answer"))
            })
            .await;
            let sent = requests_with(&w, "second question");
            assert_eq!(sent.len(), 1, "the turn is sent once: {sent:?}");
            assert!(
                sent[0].contains("<conversation_relay from=\\\"local\\\" to=\\\"local2\\\">"),
                "{}",
                sent[0]
            );
            // The old session started no turn of its own and is closed.
            assert!(!w.manager.is_session_active(&old).await);
            assert!(
                !stored_events(&w, &old).await.iter().any(
                    |e| matches!(e, ChatEvent::UserMessage { content } if content == "second question")
                ),
                "the old thread never got the message"
            );
            // Both threads carry the same statement, `moved_by: auto`.
            let as_json = |ev: &ChatEvent| serde_json::to_value(ev).unwrap();
            for thread in [&old, &new] {
                let stated = stored_events(&w, thread).await;
                let relayed = relayed_of(&stated);
                assert_eq!(relayed.len(), 1, "{thread}");
                assert_eq!(as_json(relayed[0]), as_json(&on_old), "{thread}");
            }
            // Nobody imposed the target: the router goes on choosing there.
            let node = w
                .graph
                .get_chat_session(Uuid::parse_str(&new).unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(node.provider_id.as_deref(), Some("local2"));
            assert_eq!(node.model, "m");
            assert_eq!(node.routed_by.as_deref(), Some("auto"));
            assert_eq!(node.routing_mode.as_deref(), Some("full"));
            let checks = mover.checks();
            assert_eq!(checks.len(), 1);
            assert!(checks[0].pool.iter().any(|f| f.provider_id == "local2"));
        }

        #[tokio::test]
        async fn two_consecutive_turns_do_not_flip_flop_between_providers() {
            let (w, mover) = auto_world("auto", vec!["local2", "local"], false).await;
            let old = first_turn(&w).await;
            hand_back_to_po(&w, &old).await;
            idle(&w, &old).await;
            let mut old_rx = w.manager.subscribe(&old).await.unwrap();
            w.manager
                .send_message(&old, "second question")
                .await
                .unwrap();
            let ChatEvent::ConversationRelayed {
                to_session_id: new, ..
            } = next_event(&mut old_rx, |e| {
                matches!(e, ChatEvent::ConversationRelayed { .. })
            })
            .await
            else {
                unreachable!()
            };
            stored_until(&w, &new, |records| {
                records
                    .iter()
                    .any(|r| r.event_type == "assistant_text" && r.data.contains("the answer"))
            })
            .await;
            idle(&w, &new).await;

            // The router now prefers the way back: right after a move, it is only recorded.
            answered_in_place(&w, &new, "second question, again").await;
            assert_eq!(
                relayed_of(&stored_events(&w, &new).await).len(),
                1,
                "the move in only"
            );
            let checks = mover.checks();
            assert_eq!(checks.len(), 2);
            assert_eq!(
                checks[1].settings.stage,
                LearningStage::Shadow,
                "recorded, not applied"
            );
            assert_eq!(checks[1].current, Some(Pick::new("local2", "m")));
        }

        #[tokio::test]
        async fn the_shadow_stage_records_the_provider_decision_and_moves_nothing() {
            let (w, mover) = auto_world("shadow", vec!["local2"], false).await;
            let old = first_turn(&w).await;
            hand_back_to_po(&w, &old).await;
            idle(&w, &old).await;
            answered_in_place(&w, &old, "second question").await;
            let checks = mover.checks();
            assert_eq!(checks.len(), 1, "the decision is taken, so recorded");
            assert_eq!(checks[0].settings.stage, LearningStage::Shadow);
            assert_eq!(requests_with(&w, "conversation_relay").len(), 0);
        }

        #[tokio::test]
        async fn a_model_imposed_by_the_request_is_never_moved() {
            let (w, mover) = auto_world("auto", vec!["local2"], false).await;
            let mut req = request(Some("local"), Some("proj"), "default");
            req.model = Some("m".into());
            let old = w.manager.create_session(&req).await.unwrap().session_id;
            stored_until(&w, &old, |records| {
                records.iter().any(|r| r.event_type == "assistant_text")
            })
            .await;
            idle(&w, &old).await;
            answered_in_place(&w, &old, "second question").await;
            assert!(
                mover.checks().is_empty(),
                "an imposed pair is not even asked about"
            );
        }

        #[tokio::test]
        async fn mixed_never_leaves_the_sessions_provider() {
            let (w, mover) = auto_world("auto", vec!["local2"], false).await;
            let old = first_turn(&w).await;
            let pair = |provider: &str| RoutingPoolEntry {
                provider: provider.into(),
                model: "m".into(),
            };
            w.manager
                .set_session_routing(
                    &old,
                    &SessionRoutingRequest {
                        auto: false,
                        routing_pool: vec![pair("local"), pair("local2")],
                    },
                )
                .await
                .unwrap();
            idle(&w, &old).await;
            answered_in_place(&w, &old, "second question").await;
            assert!(mover.checks().is_empty(), "mixed asks no provider question");
        }

        #[tokio::test]
        async fn a_move_the_projects_consent_refuses_stays_and_its_decision_says_why() {
            // The fake decider ignores the consent the filter would apply: the relay's
            // opening refuses on its own (endpoint_not_allowed), whoever asked.
            let (w, _mover) = auto_world("auto", vec!["local3"], true).await;
            let old = first_turn(&w).await;
            hand_back_to_po(&w, &old).await;
            idle(&w, &old).await;
            answered_in_place(&w, &old, "second question").await;
            let sent = requests_with(&w, "second question");
            assert_eq!(sent.len(), 1);
            assert!(
                !sent[0].contains("conversation_relay"),
                "sent to the old session"
            );
            let decisions = Neo4jRoutingStore::new(w.graph.clone())
                .decisions(&DecisionFilter::default())
                .await
                .unwrap();
            let refused = decisions
                .iter()
                .find(|d| d.chosen.as_ref().is_some_and(|p| p.provider_id == "local3"))
                .expect("the decision of the refused move is stored");
            assert!(!refused.applied);
            assert!(
                refused.reason.ends_with("not_moved: endpoint_not_allowed"),
                "{}",
                refused.reason
            );
        }

        /// The legacy engine (the Claude CLI) moves the same way: the check is on the
        /// manager's send path, before either engine starts the turn.
        #[tokio::test]
        async fn a_claude_code_cli_conversation_moves_too() {
            let cli = FakeClaude::new();
            let mut w = hybrid_world(&cli).await;
            // Opened under `primary` (so the opening itself stays on Claude Code), stage auto.
            routing(&w, "primary", "auto").await;
            let mover = Mover::new(vec!["local"]);
            w.manager = std::mem::replace(&mut w.manager, manager(w.graph.clone(), true))
                .with_turn_decider(mover.clone(), Arc::new(Instances));
            let cc1 = w
                .manager
                .create_session(&request(None, Some("proj"), "default"))
                .await
                .unwrap()
                .session_id;
            assert!(!w.manager.agent_runtime.owns(&cc1).await, "legacy engine");
            stored_until(&w, &cc1, |records| {
                records.iter().any(|r| r.event_type == "result")
            })
            .await;
            idle(&w, &cc1).await;
            hand_back_to_po(&w, &cc1).await;
            let mut old_rx = w.manager.subscribe(&cc1).await.unwrap();

            w.manager
                .send_message(&cc1, "second question")
                .await
                .unwrap();

            let ChatEvent::ConversationRelayed {
                to_session_id: native,
                moved_by,
                from_provider,
                to_provider,
                ..
            } = next_event(&mut old_rx, |e| {
                matches!(e, ChatEvent::ConversationRelayed { .. })
            })
            .await
            else {
                unreachable!()
            };
            assert_eq!(moved_by, "auto");
            assert_eq!(
                (from_provider.as_str(), to_provider.as_str()),
                ("claude-code", "local")
            );
            stored_until(&w, &native, |records| {
                records
                    .iter()
                    .any(|r| r.event_type == "assistant_text" && r.data.contains("the answer"))
            })
            .await;
            assert!(!w.manager.is_session_active(&cc1).await);
            assert!(
                !cli.stdin().contains("second question"),
                "the CLI never got the turn: {}",
                cli.stdin()
            );
            assert_eq!(mover.checks().len(), 1);
            w.manager.close_session(&native).await.ok();
        }
    }
}

// ============================================================================
// References (`refs_v1`) on the native engine
// ============================================================================
//
// A native session never goes through `stream_response`: it sends the text to
// the model itself (`agent_runtime`). These tests prove the references reach
// the model there too, as pointers, against the real wire of the fake server.

mod refs_native {
    use super::*;
    use crate::refs::compose::compose_user_message;
    use crate::refs::test_support::{world, World};

    fn script_for(keys: &[&str]) -> Value {
        let mut routes = script().as_array().cloned().unwrap();
        for key in keys {
            routes.push(sse_route(
                key,
                vec![
                    delta(json!({"content": format!("answer to {key}")})),
                    json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                    json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
                    json!("[DONE]"),
                ],
            ));
        }
        Value::Array(routes)
    }

    async fn setup(keys: &[&str]) -> (FakeOpenAi, World, ChatManager) {
        let fake = FakeOpenAi::start(script_for(keys));
        let w = world().await;
        store_instance(&w.graph, &instance(&fake, "none")).await;
        consent(&w.graph, "alpha", "local", &fake.origin()).await;
        let manager = manager(w.graph.clone(), true);
        (fake, w, manager)
    }

    async fn stored(w: &World, text: &str, refs: &[Value]) -> String {
        let graph: Arc<dyn GraphStore> = w.graph.clone();
        compose_user_message(&graph, text, refs, &[], true)
            .await
            .unwrap()
    }

    fn task_ref(w: &World) -> Value {
        json!({"kind": "task", "id": w.task_a.id})
    }

    /// The body the model received for the turn named by `key`.
    fn body_for(fake: &FakeOpenAi, key: &str) -> String {
        fake.chat_requests()
            .iter()
            .map(|r| r["body"].to_string())
            .find(|b| b.contains(key))
            .unwrap_or_else(|| panic!("no request to the model mentions {key}"))
    }

    async fn persisted_types(w: &World, sid: &str) -> Vec<(i64, String)> {
        w.graph
            .get_chat_events(Uuid::parse_str(sid).unwrap(), 0, 200)
            .await
            .unwrap()
            .into_iter()
            .map(|r| (r.seq, r.event_type))
            .collect()
    }

    async fn wait_for_persisted(w: &World, sid: &str, event_type: &str, count: usize) {
        for _ in 0..200 {
            let n = persisted_types(w, sid)
                .await
                .iter()
                .filter(|e| e.1 == event_type)
                .count();
            if n >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("{count} {event_type} event(s) never persisted");
    }

    #[tokio::test]
    async fn a_native_session_opened_with_references_gives_the_model_the_pointers() {
        let (fake, w, manager) = setup(&["PINNED-ONE"]).await;
        let mut req = request(Some("local"), Some("alpha"), "default");
        req.message = stored(&w, "PINNED-ONE #task:x", &[task_ref(&w)]).await;
        let created = manager.create_session(&req).await.unwrap();
        let sid = created.session_id;
        wait_for_persisted(&w, &sid, "result", 1).await;

        let body = body_for(&fake, "PINNED-ONE");
        assert!(body.contains("po-context"), "{body}");
        assert!(body.contains("Tâche alpha refs"), "{body}");
        assert!(
            !body.contains("po-refs"),
            "the model must never see the raw block: {body}"
        );
        assert!(
            !body.contains("Faire les refs"),
            "pointer depth: no content"
        );

        // Stored and replayed: the message as written, then what it resolved to.
        let events = persisted_types(&w, &sid).await;
        let user = events.iter().find(|e| e.1 == "user_message").unwrap().0;
        let resolved = events.iter().find(|e| e.1 == "refs_resolved").unwrap().0;
        assert!(user < resolved);
        manager.close_session(&sid).await.unwrap();
    }

    #[tokio::test]
    async fn a_native_session_that_is_sent_a_message_with_references_gives_the_model_the_pointers()
    {
        let (fake, w, manager) = setup(&["hi there", "PINNED-TWO"]).await;
        let created = manager
            .create_session(&request(Some("local"), Some("alpha"), "default"))
            .await
            .unwrap();
        let sid = created.session_id;
        wait_for_persisted(&w, &sid, "result", 1).await;

        manager
            .send_message(&sid, &stored(&w, "PINNED-TWO", &[task_ref(&w)]).await)
            .await
            .unwrap();
        wait_for_persisted(&w, &sid, "result", 2).await;
        let body = body_for(&fake, "PINNED-TWO");
        assert!(
            body.contains("po-context") && body.contains("Tâche alpha refs"),
            "{body}"
        );
        assert!(!body.contains("po-refs"), "{body}");
        manager.close_session(&sid).await.unwrap();
    }

    #[tokio::test]
    async fn a_native_message_without_references_is_sent_untouched() {
        let (fake, w, manager) = setup(&["hi there"]).await;
        let created = manager
            .create_session(&request(Some("local"), Some("alpha"), "default"))
            .await
            .unwrap();
        let sid = created.session_id;
        wait_for_persisted(&w, &sid, "result", 1).await;
        let body = body_for(&fake, "hi there");
        // The system prompt may NAME the block (citation section); no block was injected.
        assert!(!body.contains("<po-context nonce="), "{body}");
        assert!(persisted_types(&w, &sid)
            .await
            .iter()
            .all(|e| e.1 != "refs_resolved"));
        manager.close_session(&sid).await.unwrap();
    }

    #[tokio::test]
    async fn a_resumed_native_session_expands_the_references_of_the_message_that_resumes_it() {
        let (fake, w, manager) = setup(&["hi there", "PINNED-THREE"]).await;
        let created = manager
            .create_session(&request(Some("local"), Some("alpha"), "default"))
            .await
            .unwrap();
        let sid = created.session_id;
        wait_for_persisted(&w, &sid, "result", 1).await;
        manager.close_session(&sid).await.unwrap();

        let claims = crate::auth::jwt::Claims::service_account("e2e");
        manager
            .resume_session(
                &sid,
                &stored(&w, "PINNED-THREE", &[task_ref(&w)]).await,
                Some(&claims),
            )
            .await
            .unwrap_or_else(|e| panic!("resume failed: {e:#}"));
        wait_for_persisted(&w, &sid, "result", 2).await;
        let body = body_for(&fake, "PINNED-THREE");
        assert!(
            body.contains("po-context") && body.contains("Tâche alpha refs"),
            "{body}"
        );
        assert!(!body.contains("po-refs"), "{body}");
        manager.close_session(&sid).await.unwrap();
    }
}

// ============================================================================
// Engine parity: what the Claude Code engine does around a turn, the agent
// engine does too, on a scripted provider (no process, no network).
// ============================================================================

mod parity {
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use nexus_claude::agent::{
        AgentEvent, AgentProvider, AgentSession, Capabilities, CompactionInfo, CompactionPhase,
        CompactionTrigger, HookSupport, HookVerdict, ModelInfo, ProviderError, ProviderHealth,
        ProviderKind, ResumeToken, SessionHooks, SessionSpec, ToolCallInfo, ToolResultInfo,
        TurnContext, TurnDirective,
    };
    use nexus_claude::testkit::scripted::steps;
    use nexus_claude::testkit::{RecordedCall, Script, ScriptedProvider, Step};

    use super::*;

    /// What the graph hooks answered while the scripted provider played its turns.
    #[derive(Default)]
    pub(super) struct Answers {
        pub after_tool: StdMutex<Vec<Option<String>>>,
        pub before_compaction: StdMutex<Vec<Option<String>>>,
    }

    /// The hooks the engine gave the session, with their answers recorded (the
    /// scripted provider observes a hook's answer, it does not apply it).
    struct Recording {
        inner: Arc<dyn SessionHooks>,
        answers: Arc<Answers>,
    }

    #[async_trait]
    impl SessionHooks for Recording {
        async fn before_tool(&self, call: &ToolCallInfo) -> HookVerdict {
            self.inner.before_tool(call).await
        }
        async fn after_tool(&self, result: &ToolResultInfo) -> Option<String> {
            let said = self.inner.after_tool(result).await;
            self.answers.after_tool.lock().unwrap().push(said.clone());
            said
        }
        async fn before_compaction(&self, info: &CompactionInfo) -> Option<String> {
            let said = self.inner.before_compaction(info).await;
            self.answers
                .before_compaction
                .lock()
                .unwrap()
                .push(said.clone());
            said
        }
        async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
            self.inner.before_turn(ctx).await
        }
    }

    /// The scripted provider under the id of Claude Code and the kind asked.
    #[derive(Clone)]
    pub(super) struct Tapped {
        pub inner: Arc<ScriptedProvider>,
        pub kind: ProviderKind,
        pub answers: Arc<Answers>,
    }

    impl Tapped {
        fn tap(&self, mut spec: SessionSpec) -> SessionSpec {
            if let Some(inner) = spec.hooks.take() {
                spec.hooks = Some(Arc::new(Recording {
                    inner,
                    answers: Arc::clone(&self.answers),
                }));
            }
            spec
        }
    }

    #[async_trait]
    impl AgentProvider for Tapped {
        fn id(&self) -> &str {
            "claude-code"
        }
        fn kind(&self) -> ProviderKind {
            self.kind
        }
        async fn health(&self) -> ProviderHealth {
            self.inner.health().await
        }
        async fn catalog(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            self.inner.catalog().await
        }
        fn capabilities(&self, model: Option<&str>) -> Capabilities {
            self.inner.capabilities(model)
        }
        async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn AgentSession>, ProviderError> {
            self.inner.open(self.tap(spec)).await
        }
        async fn resume(
            &self,
            spec: SessionSpec,
            token: ResumeToken,
        ) -> Result<Arc<dyn AgentSession>, ProviderError> {
            self.inner.resume(self.tap(spec), token).await
        }
    }

    impl super::super::agent_runtime::ProviderSource for Tapped {
        fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
            (provider_id == "claude-code").then(|| Arc::new(self.clone()) as Arc<dyn AgentProvider>)
        }
    }

    pub(super) fn caps() -> Capabilities {
        let mut caps = Capabilities::none();
        caps.hooks = HookSupport::InProtocol;
        caps.per_session_mcp = true;
        caps.tools = true;
        caps.tool_cancel = true;
        caps
    }

    pub(super) struct Rig {
        pub manager: ChatManager,
        pub graph: Arc<MockGraphStore>,
        pub provider: Tapped,
        pub sid: String,
        pub rx: broadcast::Receiver<ChatEvent>,
        pub project: crate::neo4j::models::ProjectNode,
        _dir: tempfile::TempDir,
    }

    /// A session of `kind` opened on a project, no opening message: the turns
    /// below are played in order by the scripted provider.
    pub(super) async fn rig(kind: ProviderKind, turns: Vec<Vec<Step>>) -> Rig {
        rig_with(kind, turns, None).await
    }

    /// [`rig`], the manager connected to NATS when `nats` is given.
    pub(super) async fn rig_with(
        kind: ProviderKind,
        turns: Vec<Vec<Step>>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
    ) -> Rig {
        rig_full(kind, turns, nats, None, false).await
    }

    /// [`rig_with`], the manager in the given anchor mode (`None`: the default) and
    /// the session opened without project nor cwd (a neutral session) when `neutral`.
    pub(super) async fn rig_full(
        kind: ProviderKind,
        turns: Vec<Vec<Step>>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
        anchor_mode: Option<crate::chat::anchor_resolver::AnchorContextMode>,
        neutral: bool,
    ) -> Rig {
        let dir = tempfile::tempdir().unwrap();
        let graph = Arc::new(MockGraphStore::new());
        let mut project = crate::test_helpers::test_project();
        project.root_path = dir.path().display().to_string();
        graph.create_project(&project).await.unwrap();
        crate::skills::project_resolver::seed_resolve_cache_for_tests(std::slice::from_ref(
            &project,
        ));
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: PathBuf::from("/nonexistent/mcp"),
            max_sessions: 10,
            ..Default::default()
        };
        let mut builder = Script::builder().capabilities(caps());
        for turn in turns {
            builder = builder.turn(turn);
        }
        let provider = Tapped {
            inner: Arc::new(ScriptedProvider::new("claude-code", builder.build())),
            kind,
            answers: Arc::default(),
        };
        let mut manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_provider_source(Arc::new(provider.clone()));
        if let Some(nats) = nats {
            manager = manager.with_nats(nats);
        }
        if let Some(mode) = anchor_mode {
            manager = manager.with_anchor_context_mode(mode);
        }
        let mut req = request(None, Some(&project.slug), "default");
        req.message = String::new();
        req.cwd = project.root_path.clone();
        if neutral {
            req.project_slug = None;
            req.cwd = String::new();
        }
        let created = manager.create_session(&req).await.unwrap();
        let rx = manager.subscribe(&created.session_id).await.unwrap();
        Rig {
            manager,
            graph,
            provider,
            sid: created.session_id,
            rx,
            project,
            _dir: dir,
        }
    }

    impl Rig {
        /// Waits for the end of the running turn (`streaming_status: false`).
        pub(super) async fn turn_end(&mut self) {
            next_event(&mut self.rx, |e| {
                matches!(
                    e,
                    ChatEvent::StreamingStatus {
                        is_streaming: false
                    }
                )
            })
            .await;
        }

        /// The texts of the turns the provider was sent, in order.
        pub(super) fn sent(&self) -> Vec<String> {
            self.provider
                .inner
                .calls()
                .into_iter()
                .filter_map(|c| match c {
                    RecordedCall::SendTurn(input) => Some(
                        input
                            .blocks
                            .iter()
                            .filter_map(|b| match b {
                                nexus_claude::agent::InputBlock::Text { text } => {
                                    Some(text.clone())
                                }
                                _ => None,
                            })
                            .collect::<String>(),
                    ),
                    _ => None,
                })
                .collect()
        }
    }

    fn compaction_started() -> Step {
        Step::Emit(AgentEvent::Compaction {
            phase: CompactionPhase::Started,
            trigger: Some(CompactionTrigger::Auto),
            pre_tokens: None,
        })
    }

    /// H2: a session of the agent engine receives the graph's context after a
    /// tool and before a compaction — through the very hooks the Claude Code
    /// engine registers (one table, `graph_hook_table`), whatever the kind.
    #[tokio::test]
    async fn the_graph_hooks_answer_after_a_tool_and_before_a_compaction() {
        for kind in [ProviderKind::Native, ProviderKind::ClaudeCode] {
            let noisy: String = (0..30)
                .map(|i| format!("src/chat/manager.rs:{i}: build_agent_spec(input)\n"))
                .collect();
            let mut r = rig(
                kind,
                vec![vec![
                    steps::tool_call(
                        "t1",
                        "Grep",
                        json!({ "pattern": "build_agent_spec", "path": "." }),
                    ),
                    steps::tool_result("t1", noisy),
                    compaction_started(),
                    steps::done(&caps()),
                ]],
            )
            .await;
            r.manager
                .send_message(&r.sid, "where is it built?")
                .await
                .unwrap();
            r.turn_end().await;
            assert_eq!(r.sent().len(), 1, "one turn played");

            let after = r.provider.answers.after_tool.lock().unwrap().clone();
            let advice = after
                .first()
                .cloned()
                .flatten()
                .unwrap_or_else(|| panic!("{kind:?}: no context after the tool: {after:?}"));
            assert!(
                advice.contains("find_references") && advice.contains("build_agent_spec"),
                "{kind:?}: the redirect advice of the graph: {advice}"
            );
            let compaction = r.provider.answers.before_compaction.lock().unwrap().clone();
            let guidance = compaction
                .first()
                .cloned()
                .flatten()
                .unwrap_or_else(|| panic!("{kind:?}: no compaction guidance: {compaction:?}"));
            assert!(
                guidance.contains(&r.project.name),
                "{kind:?}: the compaction is told what the session works on: {guidance}"
            );
        }
    }

    /// H3 enrichment: a turn of the agent engine receives, in front of the message,
    /// the knowledge graph context the Claude Code engine puts in front of the same
    /// message (`enrichment_for_turn`, called by `stream_response`).
    #[tokio::test]
    async fn a_turn_of_the_agent_engine_gets_the_graph_context_of_a_claude_code_turn() {
        let mut r = rig(ProviderKind::Native, vec![]).await;
        let mut plan = crate::test_helpers::test_plan();
        plan.title = "Port the enrichment".into();
        plan.status = crate::neo4j::models::PlanStatus::InProgress;
        plan.project_id = Some(r.project.id);
        r.graph.create_plan(&plan).await.unwrap();
        r.graph
            .link_plan_to_project(plan.id, r.project.id)
            .await
            .unwrap();
        let message = "what is left on the port?";
        let graph: Arc<dyn GraphStore> = r.graph.clone();
        let expected = super::super::manager::enrichment_for_turn(
            &graph,
            &r.manager.enrichment_pipeline,
            &r.sid,
            message,
            Default::default(),
            Default::default(),
            &r.manager.anchor_session(),
        )
        .await
        .expect("the graph has context for this message");
        assert!(expected.contains("Port the enrichment"), "{expected}");

        r.manager.send_message(&r.sid, message).await.unwrap();
        r.turn_end().await;
        assert_eq!(
            r.sent()
                .iter()
                .map(|s| crate::chat::untrusted::mask_nonces(s))
                .collect::<Vec<_>>(),
            vec![crate::chat::untrusted::mask_nonces(
                &super::super::manager::prepend_enrichment(&expected, message)
            )]
        );
    }

    impl Rig {
        /// Waits until the provider was sent `n` turns.
        async fn turns_sent(&self, n: usize) {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
            while self.sent().len() < n {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "{n} turns within 20 s: {:?}",
                    self.sent()
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        pub(super) fn interrupts(&self) -> usize {
            self.provider
                .inner
                .calls()
                .iter()
                .filter(|c| matches!(c, RecordedCall::Interrupt(_)))
                .count()
        }
    }

    /// H3 message queue: a message sent while a turn runs is queued, and the turn
    /// interrupted so it is read sooner — as on the Claude Code engine. It used to
    /// be refused (`turn_in_progress`).
    #[tokio::test]
    async fn a_message_sent_during_a_turn_is_queued_and_read_next_not_refused() {
        let mut r = rig(ProviderKind::Native, vec![vec![Step::AwaitInterrupt]]).await;
        r.manager.send_message(&r.sid, "first").await.unwrap();
        r.turns_sent(1).await;
        r.manager
            .send_message(&r.sid, "second")
            .await
            .expect("queued, not refused");
        r.turn_end().await;
        let sent = r.sent();
        assert_eq!(sent.len(), 2, "{sent:?}");
        assert!(sent[1].ends_with("second"), "{sent:?}");
        assert_eq!(r.interrupts(), 1, "the running turn is cut short");
    }

    /// H3 message queue: a HELD message (`queue_user_message`) waits for the turn
    /// to end, is listed meanwhile, and interrupts nothing.
    #[tokio::test]
    async fn a_held_message_is_listed_waits_for_the_turn_and_interrupts_nothing() {
        let mut r = rig(
            ProviderKind::Native,
            vec![vec![
                Step::Sleep { ms: 300 },
                steps::text("done"),
                steps::done(&caps()),
            ]],
        )
        .await;
        r.manager.send_message(&r.sid, "first").await.unwrap();
        r.turns_sent(1).await;
        assert!(r
            .manager
            .queue_user_message(&r.sid, "after you")
            .await
            .unwrap());
        let listed = next_event(&mut r.rx, |e| matches!(e, ChatEvent::PendingQueue { .. })).await;
        assert!(
            matches!(&listed, ChatEvent::PendingQueue { messages } if messages.len() == 1 && messages[0].content == "after you"),
            "{listed:?}"
        );
        assert_eq!(r.sent().len(), 1, "it waits for the turn");
        r.turn_end().await;
        let sent = r.sent();
        assert_eq!(sent.len(), 2, "{sent:?}");
        assert!(sent[1].ends_with("after you"), "{sent:?}");
        assert_eq!(r.interrupts(), 0);
        assert_eq!(r.manager.pending_queue_snapshot(&r.sid).await, Some(vec![]));
    }

    /// H3 message queue: "send now" on a held message interrupts the turn and
    /// sends it next.
    #[tokio::test]
    async fn send_now_cuts_the_turn_short_for_the_held_message() {
        let mut r = rig(ProviderKind::Native, vec![vec![Step::AwaitInterrupt]]).await;
        r.manager.send_message(&r.sid, "first").await.unwrap();
        r.turns_sent(1).await;
        assert!(r.manager.queue_user_message(&r.sid, "later").await.unwrap());
        let held = r.manager.pending_queue_snapshot(&r.sid).await.unwrap();
        assert_eq!(held.len(), 1);
        assert!(r
            .manager
            .pending_queue_op(
                &r.sid,
                &crate::chat::pending_queue::QueueOp::SendNow { id: held[0].id }
            )
            .await
            .unwrap());
        r.turn_end().await;
        let sent = r.sent();
        assert!(sent.len() == 2 && sent[1].ends_with("later"), "{sent:?}");
        assert_eq!(r.interrupts(), 1);
    }

    fn stopped_on_its_turn_limit() -> Vec<Step> {
        vec![
            steps::text("partial"),
            Step::Emit(nexus_claude::testkit::done_event(
                &caps(),
                nexus_claude::agent::StopReason::MaxTurns,
                None,
            )),
        ]
    }

    /// Every event of the run, up to the end of streaming.
    async fn run_events(r: &mut Rig) -> Vec<ChatEvent> {
        let mut seen = Vec::new();
        loop {
            let e = next_event(&mut r.rx, |_| true).await;
            let end = matches!(
                e,
                ChatEvent::StreamingStatus {
                    is_streaming: false
                }
            );
            seen.push(e);
            if end {
                return seen;
            }
        }
    }

    /// H3 auto-continue: a turn that stops on its turn limit is announced
    /// (`auto_continue`) and continued with the "continue" hint of the Claude Code
    /// engine when the session's toggle is on.
    #[tokio::test]
    async fn a_turn_stopped_on_its_limit_is_continued_when_auto_continue_is_on() {
        let mut r = rig(ProviderKind::Native, vec![stopped_on_its_turn_limit()]).await;
        assert!(!r.manager.get_auto_continue_state(&r.sid).await.unwrap());
        r.manager.set_auto_continue(&r.sid, true).await.unwrap();
        assert!(r.manager.get_auto_continue_state(&r.sid).await.unwrap());
        r.manager.send_message(&r.sid, "go").await.unwrap();
        let events = run_events(&mut r).await;
        assert!(
            events.iter().any(
                |e| matches!(e, ChatEvent::AutoContinue { session_id, .. } if *session_id == r.sid)
            ),
            "{events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e, ChatEvent::SystemHint { content } if content.starts_with("Continue where you left off"))),
            "{events:?}"
        );
        let sent = r.sent();
        assert_eq!(sent.len(), 2, "{sent:?}");
        assert!(sent[1].contains("Continue where you left off"), "{sent:?}");
    }

    /// H3 auto-continue: toggle off (the default of an interactive session), the
    /// turn stays stopped.
    #[tokio::test]
    async fn a_turn_stopped_on_its_limit_stays_stopped_when_auto_continue_is_off() {
        let mut r = rig(ProviderKind::Native, vec![stopped_on_its_turn_limit()]).await;
        r.manager.send_message(&r.sid, "go").await.unwrap();
        let events = run_events(&mut r).await;
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ChatEvent::AutoContinue { .. })),
            "{events:?}"
        );
        assert_eq!(r.sent().len(), 1);
    }

    /// A Stop of the user during the auto-continue pause cancels the continuation,
    /// as on Claude Code (the Stop goes through the session handle, not around it).
    #[tokio::test]
    async fn a_stop_cancels_a_pending_auto_continue() {
        let mut r = rig(ProviderKind::Native, vec![stopped_on_its_turn_limit()]).await;
        r.manager.set_auto_continue(&r.sid, true).await.unwrap();
        r.manager.send_message(&r.sid, "go").await.unwrap();
        next_event(&mut r.rx, |e| matches!(e, ChatEvent::AutoContinue { .. })).await;
        r.manager.interrupt(&r.sid).await.unwrap();
        r.turn_end().await;
        assert_eq!(r.sent().len(), 1, "{:?}", r.sent());
    }

    /// H3 NATS: another instance sees the session's events and reaches the session
    /// (a message, a held message, a snapshot, an interrupt), as for a Claude
    /// Code session. A real `async_nats` client against an in-process broker.
    #[tokio::test]
    async fn another_instance_sees_the_events_and_reaches_the_session_through_nats() {
        use crate::events::nats_broker_test::TestBroker;
        use crate::events::NatsEmitter;
        use futures::StreamExt;

        let broker = TestBroker::start().await;
        let owner = Arc::new(NatsEmitter::new(broker.client().await, "events"));
        let mut r = rig_with(
            ProviderKind::Native,
            vec![vec![Step::AwaitInterrupt]],
            Some(owner),
        )
        .await;
        let other = NatsEmitter::new(broker.client().await, "events");
        let mut seen = other.subscribe_chat_events(&r.sid).await.unwrap();
        other.client().flush().await.unwrap();

        // A message sent on the other instance runs here (the listener may still be
        // subscribing: ask again until it answers).
        let mut answered = None;
        for _ in 0..5 {
            answered = other
                .request_send_message(&r.sid, "from afar", "user_message")
                .await;
            if answered.is_some() {
                break;
            }
        }
        assert!(answered.expect("the owner answers").success);
        r.turns_sent(1).await;
        assert!(r.sent()[0].ends_with("from afar"));

        // Its events reach the other instance.
        let shown = tokio::time::timeout(Duration::from_secs(10), async {
            while let Some(msg) = seen.next().await {
                let event: ChatEvent = serde_json::from_slice(&msg.payload).unwrap();
                if matches!(&event, ChatEvent::UserMessage { content } if content == "from afar") {
                    return true;
                }
            }
            false
        })
        .await;
        assert_eq!(shown, Ok(true), "the user message is published");

        // A client joining there mid-turn gets the snapshot.
        let snapshot = other
            .request_streaming_snapshot(&r.sid)
            .await
            .expect("a snapshot");
        assert!(snapshot.is_streaming);

        // A message held from there waits here.
        let held = other
            .request_send_message(&r.sid, "after", "queued_user_message")
            .await
            .unwrap();
        assert!(held.success, "{:?}", held.error);
        assert_eq!(
            r.manager
                .pending_queue_snapshot(&r.sid)
                .await
                .unwrap()
                .len(),
            1
        );

        // A Stop from there stops the turn here; the held message then runs.
        other.publish_interrupt(&r.sid);
        r.turn_end().await;
        let sent = r.sent();
        assert!(sent.len() == 2 && sent[1].ends_with("after"), "{sent:?}");
        assert_eq!(r.interrupts(), 1);
    }

    // ----- references (`refs_v1`) × the H3 queue of the agent engine -----

    /// A decider that is never right: the router of the test only records the text.
    struct NoDecider;

    #[async_trait]
    impl crate::chat::provider::cognitive::decision::Decider for NoDecider {
        async fn decide(
            &self,
            _request: &crate::chat::provider::cognitive::decision::DecideRequest,
        ) -> anyhow::Result<crate::chat::provider::cognitive::decision::CognitiveDecision> {
            anyhow::bail!("not asked in this test")
        }
    }

    struct NoPool;

    #[async_trait]
    impl crate::chat::agent_hooks::PoolSource for NoPool {
        async fn pool(
            &self,
            _provider_id: &str,
        ) -> Vec<crate::chat::provider::cognitive::candidates::ModelFacts> {
            Vec::new()
        }
    }

    /// refs × H3 message queue: a message HELD with `#` references while a turn
    /// runs is listed with its references (not the raw block); when the queue plays
    /// it, the model gets the typed text and the `<po-context>` pointers — never the
    /// `<po-refs>` block — behind ONE enrichment computed on the typed text; the
    /// `refs_resolved` event is stored; the turn router reads the typed text.
    #[tokio::test]
    async fn a_held_message_with_references_keeps_them_and_reaches_the_model_expanded() {
        use crate::refs::types::{EntityRef, RefKind};

        let mut r = rig(
            ProviderKind::Native,
            vec![vec![
                Step::Sleep { ms: 300 },
                steps::text("done"),
                steps::done(&caps()),
            ]],
        )
        .await;
        let mut plan = crate::test_helpers::test_plan();
        plan.title = "Wire the references".into();
        plan.status = crate::neo4j::models::PlanStatus::InProgress;
        plan.project_id = Some(r.project.id);
        r.graph.create_plan(&plan).await.unwrap();
        r.graph
            .link_plan_to_project(plan.id, r.project.id)
            .await
            .unwrap();
        let router = Arc::new(crate::chat::agent_hooks::TurnRouter::new(
            crate::chat::agent_hooks::TurnRouterSpec {
                decider: Arc::new(NoDecider),
                pool: Arc::new(NoPool),
                routing: Default::default(),
                provider_id: "claude-code".into(),
                session_id: Uuid::parse_str(&r.sid).ok(),
                project_slug: None,
                trust: false,
                explicit_model: false,
                provider_imposed: false,
                allowed_models: None,
                current_model: "m".into(),
                next_turn: 0,
                routing_pool: None,
                moved_in: false,
            },
        ));
        r.manager.turn_routing.insert(&r.sid, Arc::clone(&router));

        r.manager.send_message(&r.sid, "first").await.unwrap();
        r.turns_sent(1).await;
        let typed = "after you, see #plan:wire";
        let pointed = EntityRef::new(RefKind::Plan, plan.id);
        let stored = crate::refs::block::encode(typed, std::slice::from_ref(&pointed));
        assert!(r.manager.queue_user_message(&r.sid, &stored).await.unwrap());

        // Listed with its references, the block kept out of the text.
        let listed = next_event(&mut r.rx, |e| matches!(e, ChatEvent::PendingQueue { .. })).await;
        let ChatEvent::PendingQueue { messages } = listed else {
            unreachable!()
        };
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, typed);
        assert_eq!(messages[0].refs, vec![pointed]);

        r.turn_end().await;
        let sent = r.sent();
        assert_eq!(sent.len(), 2, "{sent:?}");
        let second = &sent[1];
        assert!(!second.contains("<po-refs>"), "the raw block: {second}");
        assert!(second.contains("<po-context nonce=\""), "{second}");
        assert!(
            second.contains(&format!(
                "- plan \"Wire the references\" [in_progress] id={}",
                plan.id
            )),
            "the pointer: {second}"
        );
        assert_eq!(second.matches(typed).count(), 1, "{second}");

        // One enrichment, on what the user typed, in front of the expanded text.
        let graph: Arc<dyn GraphStore> = r.graph.clone();
        let enrichment = super::super::manager::enrichment_for_turn(
            &graph,
            &r.manager.enrichment_pipeline,
            &r.sid,
            typed,
            Default::default(),
            Default::default(),
            &r.manager.anchor_session(),
        )
        .await
        .expect("the graph has context for this message");
        assert!(
            crate::chat::untrusted::mask_nonces(second).starts_with(
                &crate::chat::untrusted::mask_nonces(&super::super::manager::prepend_enrichment(
                    &enrichment,
                    typed
                ))
            ),
            "{second}"
        );

        // What the references resolved to is stored for replay.
        let stored_events = r
            .graph
            .get_chat_events(Uuid::parse_str(&r.sid).unwrap(), 0, 500)
            .await
            .unwrap();
        assert!(
            stored_events
                .iter()
                .any(|e| e.event_type == "refs_resolved" && e.data.contains(&plan.id.to_string())),
            "{:?}",
            stored_events
                .iter()
                .map(|e| &e.event_type)
                .collect::<Vec<_>>()
        );

        // The router of the session reads the typed text of the queued turn.
        assert_eq!(router.last_message().as_deref(), Some(typed));
    }

    // ── Anchor context (PO_ANCHOR_CONTEXT) through a whole turn ────────────

    use crate::chat::anchor::{AnchorActor, AnchorOp, AnchorRole, AnchorTargetType, NewAnchor};
    use crate::chat::anchor_resolver::AnchorContextMode;

    /// A plan of the project with a task beside it.
    async fn anchor_plan(r: &Rig, by: AnchorActor) {
        let plan = crate::test_helpers::seed_plan_with_task(&r.graph, r.project.id).await;
        r.graph
            .apply_anchor_op(
                Uuid::parse_str(&r.sid).unwrap(),
                AnchorOp::Add(NewAnchor::new(
                    AnchorTargetType::Plan,
                    plan,
                    [role_of(by)],
                    by,
                    "e2e",
                )),
            )
            .await
            .unwrap();
    }

    /// An agent can only mention; a human focuses.
    fn role_of(by: AnchorActor) -> AnchorRole {
        if by == AnchorActor::Agent {
            AnchorRole::Mention
        } else {
            AnchorRole::Focus
        }
    }

    async fn anchor_project(r: &Rig, by: AnchorActor) {
        r.graph
            .apply_anchor_op(
                Uuid::parse_str(&r.sid).unwrap(),
                AnchorOp::Add(NewAnchor::new(
                    AnchorTargetType::Project,
                    r.project.id.to_string(),
                    [role_of(by)],
                    by,
                    "e2e",
                )),
            )
            .await
            .unwrap();
    }

    /// The system prompt the manager builds for the session of `r` now.
    async fn system_prompt_of(r: &Rig) -> String {
        let node = r
            .graph
            .get_chat_session(Uuid::parse_str(&r.sid).unwrap())
            .await
            .unwrap()
            .unwrap();
        r.manager
            .build_system_prompt_anchored(
                node.execution_place,
                &node.cwd,
                node.project_slug.as_deref(),
                "hello",
                None,
                &r.sid,
                None,
            )
            .await
            .0
    }

    /// Opens a neutral session in `mode`, optionally anchors the project, plays one
    /// turn; returns (the rig, the prompt of the turn, the system prompt built after).
    async fn anchored_turn(
        mode: AnchorContextMode,
        anchored_by: Option<AnchorActor>,
    ) -> (Rig, String, String) {
        let mut r = rig_full(ProviderKind::Native, vec![], None, Some(mode), true).await;
        if let Some(by) = anchored_by {
            anchor_project(&r, by).await;
            anchor_plan(&r, by).await;
        }
        r.manager.send_message(&r.sid, "hello").await.unwrap();
        r.turn_end().await;
        let sent = r.sent();
        assert_eq!(sent.len(), 1, "{sent:?}");
        let prompt = sent[0].clone();
        let system = system_prompt_of(&r).await;
        (r, prompt, system)
    }

    /// A neutral session whose project is anchored by a human, in mode `on`: the
    /// turn carries the live block at its head, the system prompt the map in the
    /// untrusted container; the same anchor put by an agent decides nothing.
    #[tokio::test]
    async fn on_mode_a_human_project_anchor_reaches_the_turn_and_an_agent_one_does_not() {
        let (r, prompt, system) =
            anchored_turn(AnchorContextMode::On, Some(AnchorActor::User)).await;
        assert!(
            prompt.starts_with("<untrusted_data"),
            "the live block opens the turn: {prompt}"
        );
        assert!(prompt.contains(&r.project.name), "{prompt}");
        assert!(
            !prompt.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT),
            "{prompt}"
        );
        let map_at = system
            .find("anchor_map")
            .expect("the map is in the system prompt");
        assert!(
            system[..map_at].contains("<untrusted_data"),
            "the map sits in the untrusted container: {system}"
        );

        let (r2, prompt2, system2) =
            anchored_turn(AnchorContextMode::On, Some(AnchorActor::Agent)).await;
        // an agent's mention decides no project: nothing of the project reaches the turn
        assert!(
            !prompt2.contains("Task mid") && !prompt2.contains("anchor_context"),
            "an agent's project anchor decides no project: {prompt2}"
        );
        assert!(!prompt2.contains(&r2.project.name), "{prompt2}");
        assert!(!system2.contains(&r2.project.name), "{system2}");
    }

    /// Mode `shadow` is invisible: the turn and the system prompt are the very bytes
    /// of mode `off` (the resolver only journals).
    #[tokio::test]
    async fn shadow_mode_turn_and_system_prompt_are_byte_identical_to_off() {
        let mut seen = Vec::new();
        for mode in [AnchorContextMode::Off, AnchorContextMode::Shadow] {
            let (r, prompt, system) = anchored_turn(mode, Some(AnchorActor::User)).await;
            let norm = |s: &str| {
                crate::chat::untrusted::mask_nonces(s)
                    .replace(&r.sid, "<sid>")
                    .replace(&r.project.id.to_string(), "<pid>")
            };
            seen.push((norm(&prompt), norm(&system)));
        }
        assert_eq!(seen[0].0.as_bytes(), seen[1].0.as_bytes(), "the turn");
        assert_eq!(
            seen[0].1.as_bytes(),
            seen[1].1.as_bytes(),
            "the system prompt"
        );
        assert!(!seen[1].0.contains("anchor_context"), "{}", seen[1].0);
    }
}

// ============================================================================
// The resume token of a Claude Code session on the agent engine
// ============================================================================

/// Claude Code on the agent engine, the REAL nexus façade against `fake_claude`: the CLI
/// names its session only with its first `system/init` (and each `result`), never at
/// open. The token must still reach the graph, and a session resumed after a restart
/// must give the CLI `--resume <that id>` — not open a blank conversation.
mod claude_code_resume_token {
    use async_trait::async_trait;
    use nexus_claude::agent::{
        AgentProvider, AgentSession, Capabilities, ModelInfo, ProviderError, ProviderHealth,
        ProviderKind, ResumeToken, SessionSpec,
    };
    use nexus_claude::providers::claude_code::{ClaudeCodeConfig, ClaudeCodeProvider};

    use super::*;

    const CLI_SESSION: &str = "cli-session-7f3a";

    /// The Claude Code provider of nexus, its CLI the fake, which gets its transcript and
    /// its recording file through the session's environment.
    #[derive(Clone)]
    pub(super) struct FakeCli {
        pub(super) inner: Arc<ClaudeCodeProvider>,
        pub(super) env: Vec<(String, String)>,
    }

    impl FakeCli {
        fn tap(&self, mut spec: SessionSpec) -> SessionSpec {
            spec.env.set.extend(self.env.iter().cloned());
            spec
        }
    }

    #[async_trait]
    impl AgentProvider for FakeCli {
        fn id(&self) -> &str {
            "claude-code"
        }
        fn kind(&self) -> ProviderKind {
            ProviderKind::ClaudeCode
        }
        async fn health(&self) -> ProviderHealth {
            ProviderHealth::ok(None)
        }
        async fn catalog(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            self.inner.catalog().await
        }
        fn capabilities(&self, model: Option<&str>) -> Capabilities {
            self.inner.capabilities(model)
        }
        async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn AgentSession>, ProviderError> {
            self.inner.open(self.tap(spec)).await
        }
        async fn resume(
            &self,
            spec: SessionSpec,
            token: ResumeToken,
        ) -> Result<Arc<dyn AgentSession>, ProviderError> {
            self.inner.resume(self.tap(spec), token).await
        }
    }

    impl super::super::agent_runtime::ProviderSource for FakeCli {
        fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
            (provider_id == "claude-code").then(|| Arc::new(self.clone()) as Arc<dyn AgentProvider>)
        }
    }

    /// One turn as the CLI plays it: it reads the user's message, opens with
    /// `system/init` (its session id, the first time anyone hears of it), answers, ends
    /// with a `result`, and stays up until stdin closes. Both processes (open, resume)
    /// replay it.
    fn cli_turn() -> String {
        let lines = [
            json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 15000}),
            json!({"op": "emit_json", "json": {
                "type": "system", "subtype": "init", "session_id": CLI_SESSION,
                "model": "fake-claude", "cwd": ".", "tools": ["Read"],
                "permissionMode": "default", "apiKeySource": "none"}}),
            json!({"op": "emit_json", "json": {
                "type": "assistant", "message": {"id": "msg_fake", "type": "message",
                "role": "assistant", "model": "fake-claude",
                "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn"}}}),
            json!({"op": "emit_json", "json": {
                "type": "result", "subtype": "success", "duration_ms": 12,
                "duration_api_ms": 7, "is_error": false, "num_turns": 1,
                "session_id": CLI_SESSION, "total_cost_usd": 0.0001,
                "usage": {"input_tokens": 3, "output_tokens": 5}, "result": "hello"}}),
            json!({"op": "wait_eof", "timeout_ms": 15000, "optional": true}),
        ];
        lines.iter().map(|l| format!("{l}\n")).collect()
    }

    async fn stored_token(graph: &MockGraphStore, sid: &str) -> Option<String> {
        graph
            .get_chat_session(Uuid::parse_str(sid).unwrap())
            .await
            .unwrap()
            .unwrap()
            .resume_token
    }

    #[tokio::test]
    async fn a_claude_code_session_resumes_the_cli_session_it_learned_after_open() {
        let cli = fake_bin("fake_claude");
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join("transcript.jsonl");
        let argv_out = dir.path().join("argv.json");
        std::fs::write(&transcript, cli_turn()).unwrap();
        let mut config = ClaudeCodeConfig::default();
        config.cli_path = Some(cli);
        let provider = FakeCli {
            inner: Arc::new(ClaudeCodeProvider::new(config)),
            env: vec![
                (
                    "FAKE_CLAUDE_TRANSCRIPT".into(),
                    transcript.display().to_string(),
                ),
                (
                    "FAKE_CLAUDE_ARGS_OUT".into(),
                    argv_out.display().to_string(),
                ),
            ],
        };
        let graph = Arc::new(MockGraphStore::new());
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: PathBuf::from("/nonexistent/mcp"),
            max_sessions: 10,
            ..Default::default()
        };
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_provider_source(Arc::new(provider));

        let mut req = request(None, None, "default");
        req.message = "first".into();
        req.cwd = dir.path().display().to_string();
        let created = manager.create_session(&req).await.unwrap();
        let sid = created.session_id;
        assert!(manager.agent_runtime.owns(&sid).await, "the agent engine");
        let mut rx = manager.subscribe(&sid).await.unwrap();
        next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;

        // The CLI named its session during the turn: the graph knows it now.
        let token = stored_token(&graph, &sid)
            .await
            .expect("the resume token is persisted once the CLI named its session");
        let token = ResumeToken::from_wire(&token).unwrap();
        assert_eq!(token.data()["session_id"], CLI_SESSION);

        // A restart: the live session is gone, the next message resumes it.
        manager.close_session(&sid).await.unwrap();
        assert!(!manager.agent_runtime.owns(&sid).await);
        std::fs::remove_file(&argv_out).unwrap();
        manager.resume_session(&sid, "second", None).await.unwrap();
        for _ in 0..200 {
            if argv_out.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let recorded: Value =
            serde_json::from_str(&std::fs::read_to_string(&argv_out).expect("argv recorded"))
                .unwrap();
        let argv: Vec<String> = recorded["argv"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|a| a.as_str().map(str::to_string))
            .collect();
        let resume_at = argv
            .iter()
            .position(|a| a == "--resume")
            .unwrap_or_else(|| panic!("the resumed CLI gets --resume: {argv:?}"));
        assert_eq!(
            argv.get(resume_at + 1).map(String::as_str),
            Some(CLI_SESSION)
        );
        // The resumed CLI plays the second turn (it may end before anyone subscribes).
        let uuid = Uuid::parse_str(&sid).unwrap();
        let mut results = 0;
        for _ in 0..400 {
            results = graph
                .get_chat_events(uuid, 0, 500)
                .await
                .unwrap()
                .iter()
                .filter(|e| e.event_type == "result")
                .count();
            if results >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(results, 2, "the resumed session answered its turn");
        manager.close_session(&sid).await.unwrap();
    }
}

/// P13: a permission of a legacy Claude Code session granted from ANOTHER instance
/// (NATS RPC) reaches the CLI as the same `control_response` as a local answer —
/// subtype, request_id, behavior —, exactly once, while the turn waits for it.
mod legacy_nats_permission {
    use super::*;
    use crate::chat::manager::{DeliveryRoute, PermissionDeliveryError};
    use crate::events::nats_broker_test::TestBroker;
    use crate::events::NatsEmitter;

    fn emit(v: Value) -> Value {
        json!({"op": "emit_json", "json": v})
    }

    /// One turn that asks `req-nats` for `Bash pwd` and waits, up to 15 s, for an
    /// answer naming it before it goes on.
    fn transcript() -> Vec<Value> {
        vec![
            json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 30000}),
            emit(json!({"type": "system", "subtype": "init",
                "session_id": "fake-cli-session", "model": "fake-claude", "tools": [],
                "permissionMode": "default", "apiKeySource": "none"})),
            emit(json!({"type": "assistant", "message": {
                "id": "msg_fake_1", "type": "message", "role": "assistant",
                "model": "fake-claude", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t8", "name": "Bash",
                             "input": {"command": "pwd"}}]}})),
            emit(json!({"type": "control_request", "request_id": "req-nats",
                "request": {"subtype": "can_use_tool", "tool_name": "Bash",
                            "input": {"command": "pwd"}, "tool_use_id": "t8"}})),
            json!({"op": "await_stdin", "contains": "req-nats", "timeout_ms": 15000}),
            emit(json!({"type": "assistant", "message": {
                "id": "msg_fake_2", "type": "message", "role": "assistant",
                "model": "fake-claude", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "allowed and done"}]}})),
            emit(json!({"type": "result", "subtype": "success",
                "duration_ms": 1, "duration_api_ms": 1, "is_error": false, "num_turns": 1,
                "session_id": "fake-cli-session", "total_cost_usd": 0.0,
                "result": "allowed and done"})),
            json!({"op": "wait_eof", "optional": true, "timeout_ms": 110000}),
        ]
    }

    /// `fake_claude` behind a wrapper that plays the transcript and records stdin.
    fn wrapper(dir: &std::path::Path) -> String {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join("transcript.jsonl");
        let lines: String = transcript().iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(&script, lines).unwrap();
        let wrapper = dir.join("claude");
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\nFAKE_CLAUDE_TRANSCRIPT='{}' FAKE_CLAUDE_STDIN_OUT='{}' \
                 FAKE_CLAUDE_MAX_RUNTIME_MS=120000 exec '{}' \"$@\"\n",
                script.display(),
                dir.join("stdin.jsonl").display(),
                fake_bin("fake_claude").display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        wrapper.display().to_string()
    }

    /// The `control_response` lines the CLI read.
    fn control_responses(dir: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(dir.join("stdin.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v["type"] == "control_response")
            .collect()
    }

    async fn stored_until(graph: &MockGraphStore, sid: &str, needle: &str) {
        let id = Uuid::parse_str(sid).unwrap();
        for _ in 0..400 {
            let events = graph.get_chat_events(id, 0, 500).await.unwrap();
            if events.iter().any(|e| e.data.contains(needle)) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("{needle:?} was never stored for {sid}");
    }

    #[tokio::test]
    async fn a_permission_granted_from_another_instance_reaches_the_cli_once_with_its_request_id() {
        let broker = TestBroker::start().await;
        let dir = tempfile::tempdir().unwrap();
        let cli = wrapper(dir.path());

        // The owner: Claude Code on the legacy engine, on NATS.
        let graph = Arc::new(MockGraphStore::new());
        let owner = {
            let dyn_graph: Arc<dyn GraphStore> = graph.clone();
            let config = super::super::config::ChatConfig {
                provider_path: ProviderPath::Legacy,
                mcp_server_path: fake_bin("fake_mcp"),
                nexus_tools_path: None,
                nexus_browser_path: None,
                jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
                max_sessions: 10,
                ..Default::default()
            };
            let nats = Arc::new(NatsEmitter::new(broker.client().await, "events"));
            ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config)
                .with_nats(nats)
        };
        owner.update_claude_cli_path(Some(cli)).await;
        // The other instance: no session of its own.
        let other = Arc::new(NatsEmitter::new(broker.client().await, "events"));
        let far = {
            let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
            let config = super::super::config::ChatConfig {
                provider_path: ProviderPath::Legacy,
                mcp_server_path: PathBuf::from("/nonexistent/mcp"),
                max_sessions: 10,
                ..Default::default()
            };
            ChatManager::new_without_memory(graph, mock_app_state().meili, config)
                .with_nats(Arc::clone(&other))
        };

        let mut req = request(None, None, "default");
        req.message = "run pwd".into();
        req.cwd = dir.path().display().to_string();
        let sid = owner
            .create_session(&req)
            .await
            .unwrap_or_else(|e| panic!("open failed: {e:#}"))
            .session_id;
        assert!(!owner.agent_runtime.owns(&sid).await, "legacy engine");
        // The CLI asked: the request waits on the owner.
        let id = Uuid::parse_str(&sid).unwrap();
        let mut asked = false;
        for _ in 0..400 {
            let snap = owner.live_session_snapshot().await;
            if snap
                .pending_permissions
                .get(&id)
                .is_some_and(|ids| ids.contains("req-nats"))
            {
                asked = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(asked, "the CLI's can_use_tool never waited on the owner");

        // The owner's RPC listener is up (a harmless RPC answers) before the ONE
        // permission answer: a resend would write it twice.
        let mut listening = false;
        for _ in 0..20 {
            if other
                .request_send_message(&sid, r#"{"enabled":false}"#, "set_auto_continue")
                .await
                .is_some()
            {
                listening = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(listening, "the owner's RPC listener never answered");

        let routed = far
            .route_permission_response(&sid, "req-nats", true, true)
            .await;
        assert!(
            matches!(routed, Ok(DeliveryRoute::Remote)),
            "delivered by the owner: {routed:?}"
        );
        // The CLI matched it to its can_use_tool: the tool ran, the turn ended.
        stored_until(&graph, &sid, "allowed and done").await;

        let answers = control_responses(dir.path());
        assert_eq!(answers.len(), 1, "one RPC, one write: {answers:?}");
        let answer = &answers[0]["response"];
        assert_eq!(answer["subtype"], "success", "{answer}");
        assert_eq!(answer["request_id"], "req-nats", "{answer}");
        assert_eq!(answer["response"]["behavior"], "allow", "{answer}");
        assert_eq!(
            answer["response"]["updatedInput"],
            json!({"command": "pwd"}),
            "the original input goes back: {answer}"
        );

        // The same request answered again: no longer waiting, a typed refusal, and
        // nothing more reaches the CLI.
        let again = far
            .route_permission_response(&sid, "req-nats", true, true)
            .await;
        assert!(
            matches!(again, Err(PermissionDeliveryError::NotPending)),
            "{again:?}"
        );
        assert_eq!(control_responses(dir.path()).len(), 1);
        owner.close_session(&sid).await.unwrap();
    }
}

/// P5 (parity): `cancel_tools` on the agent engine stops the running tools and
/// keeps the turn — from this instance (the WS handler calls
/// `cancel_running_tools`), from another instance over NATS, and under the cap
/// of the Claude Code engine.
mod cancel_tools {
    use nexus_claude::agent::{CancelScope, ProviderKind};
    use nexus_claude::testkit::scripted::steps;
    use nexus_claude::testkit::{RecordedCall, Step};

    use super::parity::{caps, rig, rig_with, Rig};
    use super::*;

    /// A turn calling one tool that runs until it is cancelled, then goes on.
    fn slow_tool_turn() -> Vec<Step> {
        vec![
            steps::tool_call("s1", "slow", json!({})),
            Step::AwaitCancel,
            steps::text("I carry on"),
            steps::done(&caps()),
        ]
    }

    impl Rig {
        /// How many times the provider was asked to cancel all its tools.
        fn cancels(&self) -> usize {
            self.provider
                .inner
                .calls()
                .iter()
                .filter(|c| matches!(c, RecordedCall::CancelTools(CancelScope::All)))
                .count()
        }
    }

    /// The events received up to and including the first one matching `pred`,
    /// and whether it came within `within`.
    async fn collect_until(
        rx: &mut broadcast::Receiver<ChatEvent>,
        within: Duration,
        pred: impl Fn(&ChatEvent) -> bool,
    ) -> (Vec<ChatEvent>, bool) {
        let mut events = Vec::new();
        let deadline = tokio::time::Instant::now() + within;
        loop {
            let Ok(Ok(event)) = tokio::time::timeout_at(deadline, rx.recv()).await else {
                return (events, false);
            };
            let found = pred(&event);
            events.push(event);
            if found {
                return (events, true);
            }
        }
    }

    fn turn_ended(e: &ChatEvent) -> bool {
        matches!(
            e,
            ChatEvent::StreamingStatus {
                is_streaming: false
            }
        )
    }

    /// The turn went on after the cancel: the cut tool ended in a cancelled
    /// `tool_result`, the model spoke again, and the turn completed.
    fn assert_the_turn_went_on(events: &[ChatEvent], tool_id: &str) {
        let types: Vec<&str> = events.iter().map(|e| e.event_type()).collect();
        assert!(
            events.iter().any(|e| matches!(
                e,
                ChatEvent::ToolResult { id, is_error: true, result, .. }
                    if id == tool_id && result.to_string().contains("cancelled")
            )),
            "a cancelled tool_result: {types:?}"
        );
        assert!(
            events.iter().any(
                |e| matches!(e, ChatEvent::AssistantText { content, .. } if content.contains("I carry on"))
            ),
            "the model went on: {types:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                ChatEvent::Result { stop_reason, .. } if stop_reason.as_deref() == Some("completed")
            )),
            "the turn completed: {types:?}"
        );
    }

    /// (a) From this instance: the tool is stopped, `tools_cancelled` is on the
    /// wire (and stored for replay), the turn goes on — nothing interrupted it.
    #[tokio::test]
    async fn a_cancel_tools_from_this_instance_stops_the_tool_and_the_turn_goes_on() {
        let mut r = rig(ProviderKind::Native, vec![slow_tool_turn()]).await;
        r.manager.send_message(&r.sid, "slow").await.unwrap();
        next_event(
            &mut r.rx,
            |e| matches!(e, ChatEvent::ToolUse { id, .. } if id == "s1"),
        )
        .await;

        let result = r.manager.cancel_running_tools(&r.sid).await.unwrap();
        assert!(!result.capped);

        let (events, ended) = collect_until(&mut r.rx, Duration::from_secs(20), turn_ended).await;
        assert!(ended, "the turn ended");
        assert!(
            events.iter().any(|e| matches!(
                e,
                ChatEvent::ToolsCancelled { killed_count: 1, requested_by, .. } if requested_by == "user"
            )),
            "{:?}",
            events.iter().map(|e| e.event_type()).collect::<Vec<_>>()
        );
        assert_the_turn_went_on(&events, "s1");
        assert_eq!(r.cancels(), 1);
        assert_eq!(r.interrupts(), 0, "the turn was not interrupted");

        let stored = r
            .graph
            .get_chat_events(Uuid::parse_str(&r.sid).unwrap(), 0, 500)
            .await
            .unwrap();
        assert!(
            stored.iter().any(|e| e.event_type == "tools_cancelled"),
            "stored for replay: {:?}",
            stored.iter().map(|e| &e.event_type).collect::<Vec<_>>()
        );
    }

    /// (b) From another instance: its `cancel_running_tools` holds no session and
    /// publishes over NATS; the owner stops the tool, keeps the turn, and the
    /// other instance sees `tools_cancelled` in the session's feed.
    #[tokio::test]
    async fn a_cancel_tools_from_another_instance_stops_the_tool_here_and_keeps_the_turn() {
        use crate::events::nats_broker_test::TestBroker;
        use crate::events::NatsEmitter;
        use futures::StreamExt;

        let broker = TestBroker::start().await;
        let owner = Arc::new(NatsEmitter::new(broker.client().await, "events"));
        let mut r = rig_with(ProviderKind::Native, vec![slow_tool_turn()], Some(owner)).await;

        let other = Arc::new(NatsEmitter::new(broker.client().await, "events"));
        let mut seen = other.subscribe_chat_events(&r.sid).await.unwrap();
        other.client().flush().await.unwrap();
        // The other instance: no session of its own, the WS cancel goes through
        // its manager and out over NATS.
        let far = {
            let state = mock_app_state();
            let graph: Arc<dyn GraphStore> = Arc::new(MockGraphStore::new());
            let config = super::super::config::ChatConfig {
                provider_path: ProviderPath::Agent,
                mcp_server_path: PathBuf::from("/nonexistent/mcp"),
                max_sessions: 10,
                ..Default::default()
            };
            ChatManager::new_without_memory(graph, state.meili, config)
                .with_nats(Arc::clone(&other))
        };

        r.manager.send_message(&r.sid, "slow").await.unwrap();
        next_event(
            &mut r.rx,
            |e| matches!(e, ChatEvent::ToolUse { id, .. } if id == "s1"),
        )
        .await;

        // The owner's listener may still be subscribing: ask again until it answers.
        let mut events = Vec::new();
        let mut cancelled = false;
        for _ in 0..5 {
            let routed = far.cancel_running_tools(&r.sid).await.unwrap();
            assert!(
                !routed.capped && routed.killed_pids.is_empty(),
                "{routed:?}"
            );
            let (got, found) = collect_until(&mut r.rx, Duration::from_secs(1), |e| {
                matches!(e, ChatEvent::ToolsCancelled { .. })
            })
            .await;
            events.extend(got);
            if found {
                cancelled = true;
                break;
            }
        }
        assert!(cancelled, "the owner announced the cancel");
        let (rest, ended) = collect_until(&mut r.rx, Duration::from_secs(20), turn_ended).await;
        assert!(ended, "the turn ended");
        events.extend(rest);
        assert!(
            events.iter().any(|e| matches!(
                e,
                ChatEvent::ToolsCancelled {
                    killed_count: 1,
                    ..
                }
            )),
            "{:?}",
            events.iter().map(|e| e.event_type()).collect::<Vec<_>>()
        );
        assert_the_turn_went_on(&events, "s1");
        assert!(r.cancels() >= 1);
        assert_eq!(r.interrupts(), 0, "the turn was not interrupted");

        // The other instance sees the cancel in the session's feed.
        let shown = tokio::time::timeout(Duration::from_secs(10), async {
            while let Some(msg) = seen.next().await {
                let event: ChatEvent = serde_json::from_slice(&msg.payload).unwrap();
                if matches!(event, ChatEvent::ToolsCancelled { .. }) {
                    return true;
                }
            }
            false
        })
        .await;
        assert_eq!(shown, Ok(true), "tools_cancelled is published");
    }

    /// (c) The cap of the Claude Code engine (`CANCEL_TOOLS_CAP` per window)
    /// applies: the call past it says `capped` and never reaches the provider.
    #[tokio::test]
    async fn the_cancel_tools_cap_applies_on_the_agent_engine_as_on_claude_code() {
        use super::super::manager::CANCEL_TOOLS_CAP;

        let r = rig(ProviderKind::Native, vec![]).await;
        for i in 0..CANCEL_TOOLS_CAP {
            let result = r.manager.cancel_running_tools(&r.sid).await.unwrap();
            assert!(!result.capped, "call {i} is within the cap");
        }
        let past = r.manager.cancel_running_tools(&r.sid).await.unwrap();
        assert!(past.capped, "the call past the cap is refused");
        assert_eq!(
            r.cancels(),
            CANCEL_TOOLS_CAP as usize,
            "the refused call never reaches the provider"
        );
    }

    /// (a) over the real chat WebSocket, on a real native session (the nexus
    /// harness against `fake_openai` and `fake_mcp`): the model calls the `slow`
    /// tool of the project-orchestrator server, which runs until it is told to
    /// stop; the client's `cancel_tools` frame stops it, `tools_cancelled` and
    /// the cancelled `tool_result` come back on the socket, the model is told
    /// and answers, and the turn completes.
    #[tokio::test]
    async fn a_ws_cancel_tools_on_a_native_session_stops_the_running_mcp_tool_and_the_turn_goes_on()
    {
        use futures::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as WsMessage;

        let script = json!([
            // The probe: the model must be able to call a tool.
            sse_route("Call the ping tool now", vec![
                delta(json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ]),
            {"method": "GET", "path": "/v1/models", "status": 200,
             "body": {"object": "list", "data": [{"id": "m", "context_length": 32000}]}},
            // The turn: one call of the slow tool.
            sse_route("slow please", vec![
                delta(json!({"tool_calls": [{"index": 0, "id": "s1", "type": "function",
                    "function": {"name": "mcp__project-orchestrator__slow", "arguments": "{}"}}]})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ]),
            // Once the model hears of the cancelled tool, it answers (the first
            // unused matching route answers: the slow call above is used by then).
            sse_route("\"role\":\"tool\"", vec![
                delta(json!({"content": "I carry on"})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!("[DONE]"),
            ]),
        ]);
        let fake = FakeOpenAi::start(script);
        let graph = Arc::new(MockGraphStore::new());
        store_instance(&graph, &instance(&fake, "none")).await;
        consent(&graph, "proj", "local", &fake.origin()).await;
        let manager = Arc::new(manager(graph.clone(), true));

        let mut req = request(Some("local"), Some("proj"), "default");
        req.message = "slow please".into();
        let sid = manager
            .create_session(&req)
            .await
            .unwrap_or_else(|e| panic!("open failed: {e:#}"))
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();
        next_event(&mut rx, |e| {
            matches!(e, ChatEvent::ToolUse { id, tool, .. } if id == "s1" && tool.ends_with("slow"))
        })
        .await;

        // The client: the real handler, nothing replayed (the tool already runs).
        let addr = crate::test_helpers::serve_chat(Arc::clone(&manager), graph.clone()).await;
        let url = format!("ws://{addr}/ws/chat/{sid}?last_event=999999999999999");
        let (mut ws, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        ws.send(WsMessage::text("ready")).await.unwrap();
        ws.send(WsMessage::text(json!({"type": "cancel_tools"}).to_string()))
            .await
            .unwrap();

        // The frames up to the end of the turn.
        let mut frames: Vec<Value> = Vec::new();
        let ended = tokio::time::timeout(Duration::from_secs(20), async {
            while let Some(Ok(msg)) = ws.next().await {
                let WsMessage::Text(t) = msg else { continue };
                let Ok(v) = serde_json::from_str::<Value>(t.as_str()) else {
                    continue;
                };
                let done = v["type"] == "streaming_status" && v["is_streaming"] == false;
                frames.push(v);
                if done {
                    return true;
                }
            }
            false
        })
        .await
        .unwrap_or(false);
        let types: Vec<String> = frames
            .iter()
            .map(|f| f["type"].as_str().unwrap_or("?").to_string())
            .collect();
        assert!(ended, "the turn ended on the socket: {types:?}");
        let cancelled = frames
            .iter()
            .find(|f| f["type"] == "tools_cancelled")
            .unwrap_or_else(|| panic!("tools_cancelled on the socket: {types:?}"));
        assert_eq!(cancelled["killed_count"], 1, "{cancelled}");
        assert_eq!(cancelled["requested_by"], "user", "{cancelled}");
        assert!(
            cancelled.get("cli_pid").is_none(),
            "no PID on the agent engine: {cancelled}"
        );
        assert!(
            frames.iter().any(|f| f["type"] == "tool_result"
                && f["id"] == "s1"
                && f["is_error"] == true
                && f["result"].to_string().contains("cancelled")),
            "a cancelled tool_result on the socket: {frames:?}"
        );
        assert!(
            frames.iter().any(|f| f["type"] == "assistant_text"
                && f["content"].as_str().unwrap_or("").contains("I carry on")),
            "the model went on: {types:?}"
        );
        assert!(
            frames
                .iter()
                .any(|f| f["type"] == "result" && f["stop_reason"] == "completed"),
            "the turn completed: {frames:?}"
        );

        // The model was told: the request that followed carries the cancelled result.
        let chats = fake.chat_requests();
        let last = chats.last().expect("a request after the cancel")["body"].to_string();
        assert!(last.contains("cancelled"), "{last}");
        manager.close_session(&sid).await.unwrap();
    }
}

/// P3c (parity): the images a user attaches reach the model on the agent engine,
/// inline with the turn — the native harness on a model with vision (an
/// `image_url` part), Claude Code (an `image` block on the CLI's stdin) — and a
/// model without vision refuses them on the wire, nothing sent.
mod attached_images {
    use nexus_claude::providers::claude_code::{ClaudeCodeConfig, ClaudeCodeProvider};

    use super::claude_code_resume_token::FakeCli;
    use super::*;
    use crate::documents::store::DocumentStore;
    use crate::documents::DocumentFormat;
    use crate::neo4j::document::Document;

    /// A 1×1 PNG.
    const PIXEL: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x64,
        0x60, 0xf8, 0x5f, 0x0f, 0x00, 0x02, 0x87, 0x01, 0x80, 0xeb, 0x47, 0xba, 0x92, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];

    fn pixel_base64() -> String {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(PIXEL)
    }

    /// The pixel uploaded as a document (blob in `store`, node in `graph`), and
    /// the stored form of a message `text` that attaches it — what the API layer
    /// hands the manager (`message_attachments::compose`).
    async fn attach_pixel(
        graph: &Arc<MockGraphStore>,
        store: &DocumentStore,
        text: &str,
    ) -> String {
        attach_image(graph, store, text, "pixel.png", "image/png").await
    }

    /// [`attach_pixel`] under another name and recorded media type.
    async fn attach_image(
        graph: &Arc<MockGraphStore>,
        store: &DocumentStore,
        text: &str,
        filename: &str,
        media_type: &str,
    ) -> String {
        let id = Uuid::new_v4();
        let sha256 = store.put(PIXEL).unwrap();
        graph.documents.write().await.insert(
            id,
            Document {
                id,
                filename: filename.to_string(),
                format: DocumentFormat::Binary,
                sha256,
                size_bytes: PIXEL.len() as u64,
                page_count: 0,
                chunk_count: 0,
                warnings: vec![],
                created_at: Utc::now(),
                project_id: None,
                session_id: None,
                extracted: false,
                mime_type: Some(media_type.to_string()),
            },
        );
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        super::super::message_attachments::compose(&dyn_graph, text, &[id])
            .await
            .unwrap()
    }

    /// `fake_openai` serving model `m` with `catalogue` as its `/models` entry: the
    /// tool probe, the opening turn (`hi there`), and the turn of the image.
    fn script_with(catalogue: Value) -> Value {
        json!([
            sse_route("Call the ping tool now", vec![
                delta(json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                json!("[DONE]"),
            ]),
            {"method": "GET", "path": "/v1/models", "status": 200,
             "body": {"object": "list", "data": [catalogue]}},
            sse_route("look at this", vec![
                delta(json!({"content": "a single pixel"})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!("[DONE]"),
            ]),
            sse_route("hi there", vec![
                delta(json!({"content": "hello"})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!("[DONE]"),
            ]),
        ])
    }

    /// A native session on `fake_openai`, its opening turn played; the receiver
    /// subscribed before the next message.
    async fn native_session(
        catalogue: Value,
    ) -> (
        FakeOpenAi,
        Arc<MockGraphStore>,
        DocumentStore,
        tempfile::TempDir,
        ChatManager,
        String,
        broadcast::Receiver<ChatEvent>,
    ) {
        let fake = FakeOpenAi::start(script_with(catalogue));
        let graph = Arc::new(MockGraphStore::new());
        store_instance(&graph, &instance(&fake, "none")).await;
        consent(&graph, "proj", "local", &fake.origin()).await;
        let blobs = tempfile::tempdir().unwrap();
        let store = DocumentStore::new(blobs.path());
        let manager = manager(graph.clone(), true).with_document_store(store.clone());
        let created = manager
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap_or_else(|e| panic!("open failed: {e:#}"));
        let sid = created.session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();
        next_event(&mut rx, |e| {
            matches!(
                e,
                ChatEvent::StreamingStatus {
                    is_streaming: false
                }
            )
        })
        .await;
        (fake, graph, store, blobs, manager, sid, rx)
    }

    /// (a) A model whose catalogue says vision: the image is an `image_url` part
    /// of the user message the endpoint receives, and the session does not list
    /// `images` among what it cannot do.
    #[tokio::test]
    async fn a_native_model_with_vision_receives_the_attached_image_as_an_image_url_part() {
        let (fake, graph, store, _blobs, manager, sid, mut rx) =
            native_session(json!({"id": "m", "context_length": 32000, "capabilities": ["vision"]}))
                .await;
        let caps = manager
            .agent_runtime
            .get(&sid)
            .await
            .unwrap()
            .capabilities
            .clone();
        assert!(caps.images, "the catalogue says vision");
        assert!(
            !super::super::agent_runtime::degraded_features(&caps)
                .iter()
                .any(|d| d == "images"),
            "a vision model does not lose images"
        );

        let message = attach_pixel(&graph, &store, "look at this").await;
        manager.send_message(&sid, &message).await.unwrap();
        next_event(
            &mut rx,
            |e| matches!(e, ChatEvent::AssistantText { content, .. } if content.contains("single pixel")),
        )
        .await;

        let chats = fake.chat_requests();
        let turn = chats
            .iter()
            .find(|r| r["body"].to_string().contains("look at this"))
            .expect("the turn of the image reached the endpoint");
        let user = turn["body"]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .rev()
            .find(|m| m["role"] == "user")
            .unwrap();
        let text = user["content"].to_string();
        assert!(
            !text.contains("no text could be extracted"),
            "the image is sent, its /raw line is not: {text}"
        );
        assert!(
            text.contains("pixel.png"),
            "the document heading stays: {text}"
        );
        let parts = user["content"]
            .as_array()
            .unwrap_or_else(|| panic!("a list of parts: {user}"));
        assert_eq!(parts[0]["type"], "text", "{user}");
        let image = parts
            .iter()
            .find(|p| p["type"] == "image_url")
            .unwrap_or_else(|| panic!("an image_url part: {user}"));
        assert_eq!(
            image["image_url"]["url"],
            format!("data:image/png;base64,{}", pixel_base64())
        );
        manager.close_session(&sid).await.unwrap();
    }

    /// (c) A model whose catalogue says text only: the turn is refused with a
    /// typed error on the wire (`images_refused` / `unsupported`), nothing reaches
    /// the endpoint, and the session is usable again.
    #[tokio::test]
    async fn a_native_model_without_vision_refuses_the_image_on_the_wire_and_sends_nothing() {
        let (fake, graph, store, _blobs, manager, sid, mut rx) = native_session(
            json!({"id": "m", "context_length": 32000, "input_modalities": ["text"]}),
        )
        .await;
        let caps = manager
            .agent_runtime
            .get(&sid)
            .await
            .unwrap()
            .capabilities
            .clone();
        assert!(!caps.images);

        let message = attach_pixel(&graph, &store, "look at this").await;
        manager.send_message(&sid, &message).await.unwrap();
        let error = next_event(&mut rx, |e| matches!(e, ChatEvent::Error { .. })).await;
        match &error {
            ChatEvent::Error {
                message,
                code,
                reason,
                ..
            } => {
                assert_eq!(code.as_deref(), Some("images_refused"), "{error:?}");
                assert_eq!(reason.as_deref(), Some("unsupported"), "{error:?}");
                assert!(message.contains("not sent"), "{message}");
            }
            other => panic!("{other:?}"),
        }
        next_event(&mut rx, |e| {
            matches!(
                e,
                ChatEvent::StreamingStatus {
                    is_streaming: false
                }
            )
        })
        .await;
        assert!(
            !fake
                .chat_requests()
                .iter()
                .any(|r| r["body"].to_string().contains("look at this")),
            "nothing was sent"
        );
        // Stored for replay: a reload shows the refusal.
        let stored = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 500)
            .await
            .unwrap();
        assert!(
            stored
                .iter()
                .any(|e| e.event_type == "error" && e.data.contains("images_refused")),
            "{:?}",
            stored.iter().map(|e| &e.event_type).collect::<Vec<_>>()
        );
        manager.close_session(&sid).await.unwrap();
    }

    /// (b) Claude Code on the agent engine (the real nexus façade, `fake_claude`
    /// as its CLI): the user line written on stdin carries the text and the image
    /// block, base64 inline.
    #[tokio::test]
    async fn claude_code_on_the_agent_engine_writes_the_attached_image_on_stdin() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join("transcript.jsonl");
        let stdin_out = dir.path().join("stdin.jsonl");
        let lines = [
            json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 15000}),
            json!({"op": "emit_json", "json": {
                "type": "system", "subtype": "init", "session_id": "cli-images",
                "model": "fake-claude", "cwd": ".", "tools": ["Read"],
                "permissionMode": "default", "apiKeySource": "none"}}),
            json!({"op": "emit_json", "json": {
                "type": "assistant", "message": {"id": "msg_fake", "type": "message",
                "role": "assistant", "model": "fake-claude",
                "content": [{"type": "text", "text": "a pixel"}], "stop_reason": "end_turn"}}}),
            json!({"op": "emit_json", "json": {
                "type": "result", "subtype": "success", "duration_ms": 12,
                "duration_api_ms": 7, "is_error": false, "num_turns": 1,
                "session_id": "cli-images", "total_cost_usd": 0.0001,
                "usage": {"input_tokens": 3, "output_tokens": 5}, "result": "a pixel"}}),
            json!({"op": "wait_eof", "timeout_ms": 15000, "optional": true}),
        ];
        std::fs::write(
            &transcript,
            lines.iter().map(|l| format!("{l}\n")).collect::<String>(),
        )
        .unwrap();
        let mut config = ClaudeCodeConfig::default();
        config.cli_path = Some(fake_bin("fake_claude"));
        let provider = FakeCli {
            inner: Arc::new(ClaudeCodeProvider::new(config)),
            env: vec![
                (
                    "FAKE_CLAUDE_TRANSCRIPT".into(),
                    transcript.display().to_string(),
                ),
                (
                    "FAKE_CLAUDE_STDIN_OUT".into(),
                    stdin_out.display().to_string(),
                ),
            ],
        };
        let graph = Arc::new(MockGraphStore::new());
        let store = DocumentStore::new(dir.path().join("blobs"));
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let chat_config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: PathBuf::from("/nonexistent/mcp"),
            max_sessions: 10,
            ..Default::default()
        };
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, chat_config)
            .with_provider_source(Arc::new(provider))
            .with_document_store(store.clone());

        let mut req = request(None, None, "default");
        req.message = attach_pixel(&graph, &store, "look at this").await;
        req.cwd = dir.path().display().to_string();
        let created = manager.create_session(&req).await.unwrap();
        let sid = created.session_id;
        assert!(manager.agent_runtime.owns(&sid).await, "the agent engine");
        let uuid = Uuid::parse_str(&sid).unwrap();
        let mut answered = false;
        for _ in 0..400 {
            answered = graph
                .get_chat_events(uuid, 0, 500)
                .await
                .unwrap()
                .iter()
                .any(|e| e.event_type == "result");
            if answered {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(answered, "the CLI answered the turn");

        let written = std::fs::read_to_string(&stdin_out).expect("stdin recorded");
        assert!(
            !written.contains("no text could be extracted"),
            "the image is sent, its /raw line is not: {written}"
        );
        assert!(
            written.contains("pixel.png"),
            "the document heading stays: {written}"
        );
        let user: Value = written
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .find(|l| l["type"] == "user")
            .unwrap_or_else(|| panic!("a user line: {written}"));
        let blocks = user["message"]["content"]
            .as_array()
            .unwrap_or_else(|| panic!("a list of blocks: {user}"));
        assert!(
            blocks.iter().any(|b| b["type"] == "text"
                && b["text"].as_str().unwrap_or("").contains("look at this")),
            "{user}"
        );
        let image = blocks
            .iter()
            .find(|b| b["type"] == "image")
            .unwrap_or_else(|| panic!("an image block: {user}"));
        assert_eq!(image["source"]["type"], "base64");
        assert_eq!(image["source"]["media_type"], "image/png");
        assert_eq!(image["source"]["data"], pixel_base64());
        manager.close_session(&sid).await.unwrap();
    }

    /// Claude Code on its historical engine (the Claude CLI driven by the manager,
    /// `ProviderPath::Legacy`, the default for a local Claude Code): the CLI is
    /// nexus' `fake_claude` behind a wrapper that plays `transcript` and records
    /// every line it reads on stdin.
    mod legacy_engine {
        use super::*;

        struct LegacyCli {
            dir: tempfile::TempDir,
            graph: Arc<MockGraphStore>,
            store: DocumentStore,
            manager: ChatManager,
        }

        impl LegacyCli {
            async fn start(transcript: &[Value]) -> Self {
                use std::os::unix::fs::PermissionsExt;
                let dir = tempfile::tempdir().unwrap();
                let script = dir.path().join("transcript.jsonl");
                std::fs::write(
                    &script,
                    transcript
                        .iter()
                        .map(|l| format!("{l}\n"))
                        .collect::<String>(),
                )
                .unwrap();
                let wrapper = dir.path().join("claude");
                std::fs::write(
                    &wrapper,
                    format!(
                        "#!/bin/sh\nFAKE_CLAUDE_TRANSCRIPT='{}' FAKE_CLAUDE_STDIN_OUT='{}' \
                         FAKE_CLAUDE_MAX_RUNTIME_MS=120000 exec '{}' \"$@\"\n",
                        script.display(),
                        dir.path().join("stdin.jsonl").display(),
                        fake_bin("fake_claude").display()
                    ),
                )
                .unwrap();
                std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
                let graph = Arc::new(MockGraphStore::new());
                let store = DocumentStore::new(dir.path().join("blobs"));
                let dyn_graph: Arc<dyn GraphStore> = graph.clone();
                let config = super::super::super::config::ChatConfig {
                    provider_path: ProviderPath::Legacy,
                    mcp_server_path: fake_bin("fake_mcp"),
                    nexus_tools_path: None,
                    nexus_browser_path: None,
                    jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
                    max_sessions: 10,
                    retry: super::super::super::config::RetryConfig {
                        max_attempts: 3,
                        initial_delay_ms: 10,
                        backoff_multiplier: 1.0,
                    },
                    ..Default::default()
                };
                let manager =
                    ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config)
                        .with_document_store(store.clone());
                manager
                    .update_claude_cli_path(Some(wrapper.display().to_string()))
                    .await;
                Self {
                    dir,
                    graph,
                    store,
                    manager,
                }
            }

            /// Opens a conversation whose first message is `message`, on the
            /// legacy engine.
            async fn open(&self, message: String) -> String {
                let mut req = request(None, None, "default");
                req.message = message;
                req.cwd = self.dir.path().display().to_string();
                let sid = self
                    .manager
                    .create_session(&req)
                    .await
                    .unwrap_or_else(|e| panic!("open failed: {e:#}"))
                    .session_id;
                assert!(
                    !self.manager.agent_runtime.owns(&sid).await,
                    "Claude Code runs on the legacy engine"
                );
                sid
            }

            /// The stored events of `sid` once `done` holds.
            async fn stored_until(
                &self,
                sid: &str,
                done: impl Fn(&[crate::neo4j::models::ChatEventRecord]) -> bool,
            ) -> Vec<crate::neo4j::models::ChatEventRecord> {
                let id = Uuid::parse_str(sid).unwrap();
                for _ in 0..400 {
                    let events = self.graph.get_chat_events(id, 0, 500).await.unwrap();
                    if done(&events) {
                        return events;
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                panic!("the expected events were never stored for {sid}");
            }

            /// The `user` lines the CLI read on stdin, as written.
            fn user_lines(&self) -> Vec<String> {
                std::fs::read_to_string(self.dir.path().join("stdin.jsonl"))
                    .unwrap_or_default()
                    .lines()
                    .filter(|l| {
                        serde_json::from_str::<Value>(l)
                            .map(|v| v["type"] == "user")
                            .unwrap_or(false)
                    })
                    .map(str::to_string)
                    .collect()
            }
        }

        fn init() -> Value {
            json!({"op": "emit_json", "json": {"type": "system", "subtype": "init",
                "session_id": "fake-cli-session", "model": "fake-claude", "tools": [],
                "permissionMode": "default", "apiKeySource": "none"}})
        }

        fn await_user() -> Value {
            json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 30000})
        }

        /// One answered turn: `text`, then a successful result.
        fn answer(text: &str) -> [Value; 2] {
            [
                json!({"op": "emit_json", "json": {"type": "assistant", "message": {
                    "id": "msg_fake_1", "type": "message", "role": "assistant",
                    "model": "fake-claude", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": text}]}}}),
                json!({"op": "emit_json", "json": {"type": "result", "subtype": "success",
                    "duration_ms": 1, "duration_api_ms": 1, "is_error": false, "num_turns": 1,
                    "session_id": "fake-cli-session", "total_cost_usd": 0.0, "result": text}}),
            ]
        }

        fn eof() -> Value {
            json!({"op": "wait_eof", "optional": true, "timeout_ms": 110000})
        }

        fn answered(events: &[crate::neo4j::models::ChatEventRecord], text: &str) -> bool {
            events
                .iter()
                .any(|e| e.event_type == "assistant_text" && e.data.contains(text))
        }

        /// The blocks of a user line written as content blocks, else a panic.
        fn blocks_of(line: &str) -> Vec<Value> {
            let user: Value = serde_json::from_str(line).unwrap();
            user["message"]["content"]
                .as_array()
                .unwrap_or_else(|| panic!("a list of blocks: {user}"))
                .clone()
        }

        fn assert_text_then_pixel(line: &str) {
            assert!(
                !line.contains("no text could be extracted"),
                "the image is sent, its /raw line is not: {line}"
            );
            assert!(
                line.contains("pixel.png"),
                "the document heading stays: {line}"
            );
            let blocks = blocks_of(line);
            assert_eq!(blocks.len(), 2, "text, then the image: {line}");
            assert_eq!(blocks[0]["type"], "text", "{line}");
            assert!(
                blocks[0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .contains("look at this"),
                "{line}"
            );
            assert_eq!(blocks[1]["type"], "image", "{line}");
            assert_eq!(blocks[1]["source"]["type"], "base64");
            assert_eq!(blocks[1]["source"]["media_type"], "image/png");
            assert_eq!(blocks[1]["source"]["data"], pixel_base64());
        }

        /// An image attached to the message reaches the CLI: the user line on
        /// stdin is a list of blocks, the text then the image (base64 inline),
        /// and the text no longer carries the image's "/raw" line.
        #[tokio::test]
        async fn the_legacy_engine_writes_the_attached_image_on_the_cli_stdin() {
            let mut transcript = vec![await_user(), init()];
            transcript.extend(answer("a pixel"));
            transcript.push(eof());
            let cli = LegacyCli::start(&transcript).await;
            let message = attach_pixel(&cli.graph, &cli.store, "look at this").await;
            let sid = cli.open(message).await;
            cli.stored_until(&sid, |e| answered(e, "a pixel")).await;

            let lines = cli.user_lines();
            assert_eq!(lines.len(), 1, "one user line: {lines:?}");
            assert_text_then_pixel(&lines[0]);
            cli.manager.close_session(&sid).await.unwrap();
        }

        /// A text-only turn is written as it always was: `content` is one
        /// string, the line is byte for byte `InputMessage::user` of it.
        #[tokio::test]
        async fn a_text_only_turn_on_the_legacy_engine_writes_the_same_stdin_line_as_before() {
            let mut transcript = vec![await_user(), init()];
            transcript.extend(answer("hello"));
            transcript.push(eof());
            let cli = LegacyCli::start(&transcript).await;
            let sid = cli.open("hi there".to_string()).await;
            cli.stored_until(&sid, |e| answered(e, "hello")).await;

            let lines = cli.user_lines();
            assert_eq!(lines.len(), 1, "one user line: {lines:?}");
            let user: Value = serde_json::from_str(&lines[0]).unwrap();
            let content = user["message"]["content"]
                .as_str()
                .unwrap_or_else(|| panic!("a string content: {user}"))
                .to_string();
            assert!(content.contains("hi there"), "{content}");
            let historical = serde_json::to_string(&nexus_claude::transport::InputMessage::user(
                content,
                "default".to_string(),
            ))
            .unwrap();
            assert_eq!(lines[0], historical);
            cli.manager.close_session(&sid).await.unwrap();
        }

        /// An image nexus will not hand the CLI (a media type outside png, jpeg,
        /// gif, webp): `images_refused` / `invalid` on the wire and stored,
        /// nothing written on stdin, no retry; the next message is served.
        #[tokio::test]
        async fn an_invalid_image_on_the_legacy_engine_is_refused_and_nothing_reaches_the_cli() {
            let mut transcript = vec![await_user(), init()];
            transcript.extend(answer("hello again"));
            transcript.push(eof());
            let cli = LegacyCli::start(&transcript).await;
            let message = attach_image(
                &cli.graph,
                &cli.store,
                "look at this",
                "pixel.svg",
                "image/svg+xml",
            )
            .await;
            let sid = cli.open(message).await;
            let stored = cli
                .stored_until(&sid, |e| {
                    e.iter()
                        .any(|e| e.event_type == "error" && e.data.contains("images_refused"))
                })
                .await;
            let refusal: ChatEvent = stored
                .iter()
                .find(|e| e.event_type == "error")
                .map(|e| serde_json::from_str(&e.data).unwrap())
                .unwrap();
            match &refusal {
                ChatEvent::Error {
                    message,
                    code,
                    reason,
                    ..
                } => {
                    assert_eq!(code.as_deref(), Some("images_refused"), "{refusal:?}");
                    assert_eq!(reason.as_deref(), Some("invalid"), "{refusal:?}");
                    assert!(message.contains("image/svg+xml"), "{message}");
                    assert!(message.contains("not sent"), "{message}");
                }
                other => panic!("{other:?}"),
            }
            assert!(
                !stored.iter().any(|e| e.event_type == "retrying"),
                "a refusal is not retried"
            );
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert!(
                cli.user_lines().is_empty(),
                "nothing was written on stdin: {:?}",
                cli.user_lines()
            );

            // The session is usable: the next message is sent (as a string).
            cli.manager.send_message(&sid, "hi again").await.unwrap();
            cli.stored_until(&sid, |e| answered(e, "hello again")).await;
            let lines = cli.user_lines();
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert!(lines[0].contains("hi again"), "{}", lines[0]);
            assert!(!lines[0].contains("\"type\":\"image\""), "{}", lines[0]);
            cli.manager.close_session(&sid).await.unwrap();
        }

        /// A retryable failure (529 overloaded, nothing streamed) makes the
        /// engine send the turn again: the second user line carries the image
        /// too, never the text alone.
        #[tokio::test]
        async fn a_retried_turn_on_the_legacy_engine_sends_the_image_again() {
            let mut transcript = vec![
                await_user(),
                init(),
                json!({"op": "emit_json", "json": {"type": "result",
                    "subtype": "error_during_execution", "duration_ms": 1,
                    "duration_api_ms": 1, "is_error": true, "num_turns": 1,
                    "session_id": "fake-cli-session", "total_cost_usd": 0.0,
                    "result": "API Error: 529 {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}"}}),
                await_user(),
            ];
            transcript.extend(answer("a pixel"));
            transcript.push(eof());
            let cli = LegacyCli::start(&transcript).await;
            let message = attach_pixel(&cli.graph, &cli.store, "look at this").await;
            let sid = cli.open(message).await;
            let stored = cli.stored_until(&sid, |e| answered(e, "a pixel")).await;
            assert!(
                stored.iter().any(|e| e.event_type == "retrying"),
                "the turn was retried: {:?}",
                stored.iter().map(|e| &e.event_type).collect::<Vec<_>>()
            );

            let lines = cli.user_lines();
            assert_eq!(lines.len(), 2, "the turn, then its retry: {lines:?}");
            assert_text_then_pixel(&lines[0]);
            assert_text_then_pixel(&lines[1]);
            assert_eq!(lines[0], lines[1], "the retry sends the same blocks");
            cli.manager.close_session(&sid).await.unwrap();
        }
    }
}

/// The session record of a conversation on the legacy Claude Code engine (P15): the
/// counter of user messages survives every turn of the CLI.
mod legacy_session_record {
    use super::*;

    fn turn(text: &str, cost: f64) -> [Value; 3] {
        [
            json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 30000}),
            json!({"op": "emit_json", "json": {"type": "assistant", "message": {
                "id": "msg_fake_1", "type": "message", "role": "assistant",
                "model": "fake-claude", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": text}]}}}),
            json!({"op": "emit_json", "json": {"type": "result", "subtype": "success",
                "duration_ms": 1, "duration_api_ms": 1, "is_error": false, "num_turns": 1,
                "session_id": "fake-cli-session", "total_cost_usd": cost, "result": text}}),
        ]
    }

    /// `fake_claude` behind a wrapper that plays `transcript`, and a manager on the
    /// legacy engine that starts it.
    async fn legacy(transcript: &[Value]) -> (tempfile::TempDir, Arc<MockGraphStore>, ChatManager) {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("transcript.jsonl");
        std::fs::write(
            &script,
            transcript
                .iter()
                .map(|l| format!("{l}\n"))
                .collect::<String>(),
        )
        .unwrap();
        let wrapper = dir.path().join("claude");
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\nFAKE_CLAUDE_TRANSCRIPT='{}' FAKE_CLAUDE_MAX_RUNTIME_MS=120000 \
                 exec '{}' \"$@\"\n",
                script.display(),
                fake_bin("fake_claude").display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let graph = Arc::new(MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Legacy,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: None,
            nexus_browser_path: None,
            jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
            max_sessions: 10,
            ..Default::default()
        };
        let manager = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config);
        manager
            .update_claude_cli_path(Some(wrapper.display().to_string()))
            .await;
        (dir, graph, manager)
    }

    /// The record of `sid` once `done` holds.
    async fn record_until(
        graph: &MockGraphStore,
        sid: &str,
        done: impl Fn(&crate::neo4j::models::ChatSessionNode) -> bool,
    ) -> crate::neo4j::models::ChatSessionNode {
        let id = Uuid::parse_str(sid).unwrap();
        let mut last = None;
        for _ in 0..400 {
            let node = graph.get_chat_session(id).await.unwrap();
            if let Some(node) = node {
                if done(&node) {
                    return node;
                }
                last = Some(node);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the record of {sid} never got there: {last:?}");
    }

    /// Three user messages, three turns of the CLI: the record counts three, each
    /// `result` of the CLI still leaves its cost and its session id (it used to write
    /// `message_count = 1` on every `result`: 1 measured for 13 messages).
    #[tokio::test]
    async fn the_message_count_of_a_legacy_session_counts_every_user_message() {
        let mut transcript = turn("answer one", 0.01).to_vec();
        transcript.insert(
            1,
            json!({"op": "emit_json", "json": {"type": "system", "subtype": "init",
                "session_id": "fake-cli-session", "model": "fake-claude", "tools": [],
                "permissionMode": "default", "apiKeySource": "none"}}),
        );
        transcript.extend(turn("answer two", 0.02));
        transcript.extend(turn("answer three", 0.03));
        transcript.push(json!({"op": "wait_eof", "optional": true, "timeout_ms": 110000}));
        let (dir, graph, manager) = legacy(&transcript).await;

        let mut req = request(None, None, "default");
        req.message = "message one".into();
        req.cwd = dir.path().display().to_string();
        let sid = manager
            .create_session(&req)
            .await
            .unwrap_or_else(|e| panic!("open failed: {e:#}"))
            .session_id;
        assert!(
            !manager.agent_runtime.owns(&sid).await,
            "Claude Code runs on the legacy engine"
        );
        let node = record_until(&graph, &sid, |n| n.total_cost_usd == Some(0.01)).await;
        assert_eq!(node.message_count, 1, "after the first turn: {node:?}");

        manager.send_message(&sid, "message two").await.unwrap();
        let node = record_until(&graph, &sid, |n| n.total_cost_usd == Some(0.02)).await;
        assert_eq!(node.message_count, 2, "after the second turn: {node:?}");

        manager.send_message(&sid, "message three").await.unwrap();
        let node = record_until(&graph, &sid, |n| n.total_cost_usd == Some(0.03)).await;
        assert_eq!(node.message_count, 3, "after the third turn: {node:?}");
        assert_eq!(node.cli_session_id.as_deref(), Some("fake-cli-session"));
        manager.close_session(&sid).await.unwrap();
    }

    /// A turn whose wrap-up never returns, on the legacy engine with a real
    /// (fake) CLI: the CLI has answered (`result`), then the store that should
    /// save the turn never answers. Measured on 2026-10-05 (note 1a31aadb) and
    /// 2026-10-09 (session b4c28b9f): the turn stayed "streaming" for hours and
    /// every Stop was delivered to nothing.
    mod stuck_after_result {
        use super::*;
        use std::sync::atomic::Ordering;

        async fn is_streaming(manager: &ChatManager, sid: &str) -> bool {
            manager
                .active_sessions
                .read()
                .await
                .get(sid)
                .is_some_and(|s| s.is_streaming.load(Ordering::SeqCst))
        }

        /// Waits up to `secs` for the session to be idle; says whether it got there.
        async fn idle_within(manager: &ChatManager, sid: &str, secs: u64) -> bool {
            for _ in 0..secs * 20 {
                if !is_streaming(manager, sid).await {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            false
        }

        /// The persisted events of `sid` whose type is `event_type`, once there
        /// are `count` of them.
        async fn persisted_until(
            graph: &MockGraphStore,
            sid: &str,
            event_type: &str,
            count: usize,
        ) {
            let id = Uuid::parse_str(sid).unwrap();
            let mut seen = 0;
            for _ in 0..400 {
                seen = graph
                    .chat_events
                    .read()
                    .await
                    .get(&id)
                    .map(|events| events.iter().filter(|e| e.event_type == event_type).count())
                    .unwrap_or(0);
                if seen >= count {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            panic!("{count} `{event_type}` event(s) never persisted for {sid}: {seen}");
        }

        fn received(rx: &mut broadcast::Receiver<ChatEvent>) -> Vec<ChatEvent> {
            let mut events = Vec::new();
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
            events
        }

        fn has_error(events: &[ChatEvent], code: &str, reason: Option<&str>) -> bool {
            events.iter().any(|e| {
                matches!(e, ChatEvent::Error { code: Some(c), reason: r, .. }
                    if c == code && (reason.is_none() || r.as_deref() == reason))
            })
        }

        /// `turns` turns of the CLI (costs 0.01, 0.02...); the second one takes
        /// its time, room to hold a message during it.
        fn transcript(turns: usize) -> Vec<Value> {
            let mut transcript = vec![];
            for n in 1..=turns {
                let mut t = turn(&format!("answer {n}"), n as f64 / 100.0).to_vec();
                if n == 1 {
                    t.insert(
                        1,
                        json!({"op": "emit_json", "json": {"type": "system", "subtype": "init",
                            "session_id": "fake-cli-session", "model": "fake-claude", "tools": [],
                            "permissionMode": "default", "apiKeySource": "none"}}),
                    );
                }
                if n == 2 {
                    t.insert(1, json!({"op": "sleep", "ms": 1500}));
                }
                transcript.extend(t);
            }
            transcript.push(json!({"op": "wait_eof", "optional": true, "timeout_ms": 110000}));
            transcript
        }

        /// A session whose first turn is over and saved, then a store that
        /// never saves a `result` again, then a second turn with a message held
        /// during it (`queue_user_message`: it interrupts nothing). Returns once
        /// the CLI has answered the second turn (its cost is written).
        ///
        /// The held message is what keeps the session "streaming" past the CLI's
        /// answer: `finalize_streaming_status` leaves it for the drain to send —
        /// the drain that comes after the write that never answers.
        async fn stuck_after_second_result(
            turns: usize,
            budget: Option<Duration>,
        ) -> (
            tempfile::TempDir,
            Arc<MockGraphStore>,
            ChatManager,
            String,
            broadcast::Receiver<ChatEvent>,
        ) {
            let (dir, graph, manager) = legacy(&transcript(turns)).await;
            let mut req = request(None, None, "default");
            req.message = "message one".into();
            req.cwd = dir.path().display().to_string();
            let sid = manager
                .create_session(&req)
                .await
                .unwrap_or_else(|e| panic!("open failed: {e:#}"))
                .session_id;
            record_until(&graph, &sid, |n| n.total_cost_usd == Some(0.01)).await;
            assert!(idle_within(&manager, &sid, 15).await, "the first turn ends");
            persisted_until(&graph, &sid, "result", 1).await;

            if let Some(budget) = budget {
                manager
                    .active_sessions
                    .write()
                    .await
                    .get_mut(&sid)
                    .unwrap()
                    .post_stream_budget = budget;
            }
            graph.stall_chat_events("result");
            let rx = manager.subscribe(&sid).await.unwrap();
            manager.send_message(&sid, "message two").await.unwrap();
            for _ in 0..100 {
                if is_streaming(&manager, &sid).await {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(is_streaming(&manager, &sid).await, "the second turn runs");
            assert!(
                manager
                    .queue_user_message(&sid, "message three")
                    .await
                    .unwrap(),
                "held during the second turn"
            );
            record_until(&graph, &sid, |n| n.total_cost_usd == Some(0.02)).await;
            (dir, graph, manager, sid, rx)
        }

        /// The Stop on a turn stuck after `result`: the interrupt watchdog ends
        /// it, the clients learn why, the turn's events still land once the store
        /// answers, the session takes the next message and the held one goes
        /// out after it. Red without the watchdog: the session stays "streaming"
        /// whatever the Stop (until the 30 s step budget, without it for good).
        #[tokio::test]
        async fn a_stop_frees_a_turn_whose_wrap_up_never_returns() {
            let (_dir, graph, manager, sid, mut rx) = stuck_after_second_result(4, None).await;
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert!(
                is_streaming(&manager, &sid).await,
                "the reproduction: the turn is stuck after its result"
            );

            let outcome = manager.interrupt_scoped(&sid, true).await.unwrap();
            assert!(outcome.delivered, "the session is local: {outcome:?}");
            assert!(
                idle_within(&manager, &sid, 15).await,
                "a delivered Stop frees the session even when its turn ignores the token"
            );
            let events = received(&mut rx);
            assert!(
                has_error(&events, "turn_abandoned", None),
                "the clients learn why: {events:?}"
            );
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    ChatEvent::StreamingStatus {
                        is_streaming: false
                    }
                )),
                "the composer is released: {events:?}"
            );

            // The turn's events were not dropped with it: they land when the
            // store answers.
            graph.release_chat_events("result");
            persisted_until(&graph, &sid, "result", 2).await;

            // The session is free: the next message gets its turn, and the
            // message held during the abandoned one goes out after it.
            manager.send_message(&sid, "message four").await.unwrap();
            record_until(&graph, &sid, |n| n.total_cost_usd == Some(0.04)).await;
            assert!(idle_within(&manager, &sid, 15).await, "the last turn ends");
            persisted_until(&graph, &sid, "result", 4).await;
            manager.close_session(&sid).await.unwrap();
        }

        /// Without any Stop, the case of note 1a31aadb: the write before the
        /// drain never answers. Past the step budget it no longer holds the turn:
        /// the held message goes out, the clients learn that the save is late
        /// (`persistence_delayed`, the step named), every write lands when the
        /// store answers. Red without the budget: the held message is never
        /// sent, the session "streaming" for good.
        #[tokio::test]
        async fn a_wrap_up_that_never_returns_no_longer_holds_the_queue() {
            let (_dir, graph, manager, sid, mut rx) =
                stuck_after_second_result(3, Some(Duration::from_millis(300))).await;

            record_until(&graph, &sid, |n| n.total_cost_usd == Some(0.03)).await;
            assert!(
                idle_within(&manager, &sid, 10).await,
                "a store that never answers no longer pins the turn"
            );
            let events = received(&mut rx);
            assert!(
                has_error(&events, "persistence_delayed", Some("persist_events")),
                "the clients learn that the save is late, and which one: {events:?}"
            );

            graph.release_chat_events("result");
            persisted_until(&graph, &sid, "result", 3).await;
            manager.close_session(&sid).await.unwrap();
        }
    }
}

/// P14: a native session's conversation lives in a transcript the resume token only
/// names. Kept on disk (owner-only), the session resumes after a restart with its
/// whole history; a deletion of the session removes it.
mod native_transcripts {
    use super::*;

    fn answer(key: &str) -> Value {
        sse_route(
            key,
            vec![
                delta(json!({"content": format!("answer to {key}")})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
                json!("[DONE]"),
            ],
        )
    }

    /// The most specific routes first: the fake answers with the first unused match,
    /// and a later turn's body holds the earlier turns' text.
    fn script_with_turns() -> Value {
        let mut routes = vec![answer("AFTER-RESTART"), answer("SECOND-TURN")];
        routes.extend(script().as_array().cloned().unwrap());
        Value::Array(routes)
    }

    async fn wait_results(graph: &MockGraphStore, sid: &str, count: usize) {
        let uuid = Uuid::parse_str(sid).unwrap();
        for _ in 0..400 {
            let n = graph
                .get_chat_events(uuid, 0, 500)
                .await
                .unwrap()
                .iter()
                .filter(|e| e.event_type == "result")
                .count();
            if n >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("{count} result event(s) never persisted");
    }

    #[tokio::test]
    async fn a_native_session_resumes_its_history_after_a_restart_and_its_deletion_removes_the_transcript(
    ) {
        let fake = FakeOpenAi::start(script_with_turns());
        let graph = Arc::new(MockGraphStore::new());
        store_instance(&graph, &instance(&fake, "none")).await;
        consent(&graph, "proj", "local", &fake.origin()).await;
        let data = tempfile::tempdir().unwrap();
        let root = data.path().join("native-transcripts");

        // Before the restart: two turns.
        let before = manager(graph.clone(), true).with_native_transcripts(&root);
        let created = before
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap();
        let sid = created.session_id;
        assert!(before.agent_runtime.owns(&sid).await, "the agent engine");
        wait_results(&graph, &sid, 1).await;
        before.send_message(&sid, "SECOND-TURN").await.unwrap();
        wait_results(&graph, &sid, 2).await;
        before.close_session(&sid).await.unwrap();
        drop(before);

        // The transcript the token names is on disk, owner-only.
        let uuid = Uuid::parse_str(&sid).unwrap();
        let token = graph
            .get_chat_session(uuid)
            .await
            .unwrap()
            .unwrap()
            .resume_token
            .expect("a native session persists its resume token");
        let id = super::super::provider::transcripts::transcript_id_of(&token)
            .expect("a native token names a transcript");
        let file = root.join("local").join(format!("{id}.json"));
        assert!(
            file.exists(),
            "the transcript is on disk: {}",
            file.display()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode =
                |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&file), 0o600, "the transcript file");
            assert_eq!(mode(&root.join("local")), 0o700, "the instance directory");
            assert_eq!(mode(&root), 0o700, "the root");
        }

        // A restart: a new manager on the same graph and the same data directory.
        let after = manager(graph.clone(), true).with_native_transcripts(&root);
        let claims = crate::auth::jwt::Claims::service_account("e2e");
        after
            .resume_session(&sid, "AFTER-RESTART", Some(&claims))
            .await
            .unwrap_or_else(|e| panic!("resume after a restart failed: {e:#}"));
        wait_results(&graph, &sid, 3).await;
        let body = fake
            .chat_requests()
            .iter()
            .map(|r| r["body"].to_string())
            .find(|b| b.contains("AFTER-RESTART"))
            .expect("the resumed turn reached the model");
        for earlier in [
            "hi there",
            "hello from the fake model",
            "SECOND-TURN",
            "answer to SECOND-TURN",
        ] {
            assert!(
                body.contains(earlier),
                "the resumed turn carries the history ({earlier}): {body}"
            );
        }

        // Deleting the session removes its transcript, then the node.
        assert!(after.delete_session(uuid).await.unwrap());
        assert!(!file.exists(), "the transcript goes with the session");
        assert!(graph.get_chat_session(uuid).await.unwrap().is_none());
        assert!(!after.delete_session(uuid).await.unwrap(), "already gone");
    }
}

/// A message sent during a turn of the legacy Claude Code engine interrupts it, then
/// runs. The CLI ends the interrupted turn with its own `result`, which may come
/// LATE: after the queued turn has already subscribed to the CLI's output. That
/// stale `result` must not end the queued turn (it used to: the queued turn ended at
/// once, its real answer arrived with no turn to read it, and every later turn was
/// one `result` behind — the parity matrix saw it under CI load).
mod legacy_interrupt_stale_result {
    use super::*;

    fn emit(v: Value) -> Value {
        json!({"op": "emit_json", "json": v})
    }

    fn text(t: &str, id: &str) -> Value {
        emit(json!({"type": "assistant", "message": {
            "id": id, "type": "message", "role": "assistant",
            "model": "fake-claude", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": t}]}}))
    }

    fn result(subtype: &str, t: &str, is_error: bool) -> Value {
        emit(json!({"type": "result", "subtype": subtype,
            "duration_ms": 1, "duration_api_ms": 1, "is_error": is_error, "num_turns": 1,
            "session_id": "fake-cli-session", "total_cost_usd": 0.0, "result": t}))
    }

    fn await_in(needle: &str) -> Value {
        json!({"op": "await_stdin", "contains": needle, "timeout_ms": 15000, "optional": true})
    }

    /// The first turn stalls until it is interrupted; the CLI answers the interrupt
    /// with its `result` only after `late_ms`; then the queued message is answered,
    /// then a third one.
    fn transcript(late_ms: u64) -> Vec<Value> {
        vec![
            json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 30000}),
            emit(json!({"type": "system", "subtype": "init",
                "session_id": "fake-cli-session", "model": "fake-claude", "tools": [],
                "permissionMode": "default", "apiKeySource": "none"})),
            text("working on it", "msg_1"),
            await_in("\"interrupt\""),
            json!({"op": "sleep", "ms": late_ms}),
            result("error_during_execution", "", true),
            await_in("SECOND-MESSAGE"),
            text("answered second", "msg_2"),
            result("success", "answered second", false),
            await_in("THIRD-MESSAGE"),
            text("answered third", "msg_3"),
            result("success", "answered third", false),
            json!({"op": "wait_eof", "optional": true, "timeout_ms": 110000}),
        ]
    }

    struct Cli {
        dir: tempfile::TempDir,
        graph: Arc<MockGraphStore>,
        manager: ChatManager,
    }

    impl Cli {
        async fn start(late_ms: u64) -> Self {
            use std::os::unix::fs::PermissionsExt;
            let dir = tempfile::tempdir().unwrap();
            let script = dir.path().join("transcript.jsonl");
            let lines: String = transcript(late_ms)
                .iter()
                .map(|l| format!("{l}\n"))
                .collect();
            std::fs::write(&script, lines).unwrap();
            let wrapper = dir.path().join("claude");
            std::fs::write(
                &wrapper,
                format!(
                    "#!/bin/sh\nFAKE_CLAUDE_TRANSCRIPT='{}' FAKE_CLAUDE_STDIN_OUT='{}' \
                     FAKE_CLAUDE_MAX_RUNTIME_MS=120000 exec '{}' \"$@\"\n",
                    script.display(),
                    dir.path().join("stdin.jsonl").display(),
                    fake_bin("fake_claude").display()
                ),
            )
            .unwrap();
            std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
            let graph = Arc::new(MockGraphStore::new());
            let dyn_graph: Arc<dyn GraphStore> = graph.clone();
            let config = super::super::config::ChatConfig {
                provider_path: ProviderPath::Legacy,
                mcp_server_path: fake_bin("fake_mcp"),
                nexus_tools_path: None,
                nexus_browser_path: None,
                jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
                max_sessions: 10,
                ..Default::default()
            };
            let manager =
                ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config);
            manager
                .update_claude_cli_path(Some(wrapper.display().to_string()))
                .await;
            Self {
                dir,
                graph,
                manager,
            }
        }

        fn stdin(&self) -> String {
            std::fs::read_to_string(self.dir.path().join("stdin.jsonl")).unwrap_or_default()
        }

        async fn read_by_cli(&self, needle: &str) {
            for _ in 0..400 {
                if self.stdin().contains(needle) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            panic!("the CLI never read {needle:?}: {}", self.stdin());
        }

        /// Whether `text` was stored as an answer of the assistant within `secs`.
        async fn answered(&self, sid: &str, text: &str, secs: u64) -> bool {
            let id = Uuid::parse_str(sid).unwrap();
            let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
            while tokio::time::Instant::now() < deadline {
                let events = self.graph.get_chat_events(id, 0, 500).await.unwrap();
                if events
                    .iter()
                    .any(|e| e.event_type == "assistant_text" && e.data.contains(text))
                {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            false
        }
    }

    async fn queued_message_after(late_ms: u64) {
        let cli = Cli::start(late_ms).await;
        let mut req = request(None, None, "default");
        req.message = "FIRST-MESSAGE".into();
        req.cwd = cli.dir.path().display().to_string();
        let sid = cli
            .manager
            .create_session(&req)
            .await
            .unwrap_or_else(|e| panic!("open failed: {e:#}"))
            .session_id;
        assert!(!cli.manager.agent_runtime.owns(&sid).await, "legacy engine");
        cli.read_by_cli("FIRST-MESSAGE").await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            cli.manager.is_session_streaming(&sid).await,
            "the first turn runs"
        );

        // Sent while the first turn runs: queued, the turn is interrupted.
        cli.manager
            .send_message(&sid, "SECOND-MESSAGE")
            .await
            .unwrap();
        cli.read_by_cli("SECOND-MESSAGE").await;
        assert!(
            cli.answered(&sid, "answered second", 10).await,
            "the queued turn read its own answer (CLI result {late_ms} ms after the interrupt): {:?}\nstdin: {}",
            cli.graph.get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 500).await.unwrap().iter().map(|e| (e.event_type.clone(), e.data.chars().take(90).collect::<String>())).collect::<Vec<_>>(),
            cli.stdin()
        );

        // And the turns stay in step.
        for _ in 0..200 {
            if !cli.manager.is_session_streaming(&sid).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        cli.manager
            .send_message(&sid, "THIRD-MESSAGE")
            .await
            .unwrap();
        assert!(
            cli.answered(&sid, "answered third", 10).await,
            "the next turn reads its own answer"
        );
        cli.manager.close_session(&sid).await.unwrap();
    }

    /// The CLI answers the interrupt at once: the queued message runs.
    #[tokio::test]
    async fn a_queued_message_runs_after_the_interrupt() {
        queued_message_after(0).await;
    }

    /// The CLI answers the interrupt late (a loaded machine): its stale `result`
    /// does not end the queued turn.
    #[tokio::test]
    async fn a_late_result_of_the_interrupted_turn_does_not_end_the_queued_turn() {
        queued_message_after(1500).await;
    }
}

/// The listener of out-of-turn output (`oob_listener`) reads the CLI's output on a
/// subscription of its own, and decides "in a turn or not" from `is_streaming` when
/// it gets to a message. Lagging behind the turn (a loaded machine), it used to get
/// to the turn's last `tool_result` after the turn had ended, took it for background
/// output and started a turn of its own with it ("[tool_result] …"): every later
/// turn was then one `result` behind. Seen by the parity matrix on main (cancel_tools
/// and nats.cancel_tools, "tour poursuivi=false").
mod legacy_oob_lag {
    use super::*;

    fn emit(v: Value) -> Value {
        json!({"op": "emit_json", "json": v})
    }

    fn text(t: &str, id: &str) -> Value {
        emit(json!({"type": "assistant", "message": {
            "id": id, "type": "message", "role": "assistant",
            "model": "fake-claude", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": t}]}}))
    }

    fn result_ok(t: &str) -> Value {
        emit(json!({"type": "result", "subtype": "success",
            "duration_ms": 1, "duration_api_ms": 1, "is_error": false, "num_turns": 1,
            "session_id": "fake-cli-session", "total_cost_usd": 0.0, "result": t}))
    }

    fn transcript() -> Vec<Value> {
        vec![
            json!({"op": "await_stdin", "contains": "\"type\":\"user\"", "timeout_ms": 30000}),
            emit(json!({"type": "system", "subtype": "init",
                "session_id": "fake-cli-session", "model": "fake-claude", "tools": [],
                "permissionMode": "default", "apiKeySource": "none"})),
            text("ready", "msg_1"),
            result_ok("ready"),
            // A turn whose last tool_result is followed at once by its answer.
            json!({"op": "await_stdin", "contains": "SECOND-MESSAGE", "timeout_ms": 30000}),
            emit(json!({"type": "assistant", "message": {
                "id": "msg_2", "type": "message", "role": "assistant",
                "model": "fake-claude", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                             "input": {"command": "ls"}}]}})),
            emit(
                json!({"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a.rs",
                 "is_error": false}]}}),
            ),
            text("answered second", "msg_3"),
            result_ok("answered second"),
            json!({"op": "wait_eof", "optional": true, "timeout_ms": 110000}),
        ]
    }

    /// A listener a little behind (150 ms per message).
    #[tokio::test]
    async fn a_lagging_oob_listener_does_not_replay_the_turns_tool_result_as_a_new_turn() {
        lagging_listener(Duration::from_millis(150)).await;
    }

    /// A listener far behind: more than 3 s for the turn's four messages, longer than
    /// any bounded wait of the turn — only a rule on the listener's side holds.
    #[tokio::test]
    async fn a_listener_lagging_longer_than_any_bound_still_does_not_replay_the_tool_result() {
        lagging_listener(Duration::from_millis(800)).await;
    }

    async fn lagging_listener(lag: Duration) {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("transcript.jsonl");
        let lines: String = transcript().iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(&script, lines).unwrap();
        let wrapper = dir.path().join("claude");
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\nFAKE_CLAUDE_TRANSCRIPT='{}' FAKE_CLAUDE_STDIN_OUT='{}' \
                 FAKE_CLAUDE_MAX_RUNTIME_MS=120000 exec '{}' \"$@\"\n",
                script.display(),
                dir.path().join("stdin.jsonl").display(),
                fake_bin("fake_claude").display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let graph = Arc::new(MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Legacy,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: None,
            nexus_browser_path: None,
            jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
            max_sessions: 10,
            ..Default::default()
        };
        let manager = ChatManager::new_without_memory(dyn_graph, mock_app_state().meili, config);
        manager
            .update_claude_cli_path(Some(wrapper.display().to_string()))
            .await;

        let mut req = request(None, None, "default");
        req.message = "FIRST-MESSAGE".into();
        req.cwd = dir.path().display().to_string();
        let sid = manager
            .create_session(&req)
            .await
            .unwrap_or_else(|e| panic!("open failed: {e:#}"))
            .session_id;
        assert!(!manager.agent_runtime.owns(&sid).await, "legacy engine");
        let id = Uuid::parse_str(&sid).unwrap();
        let stored = |needle: &'static str| {
            let graph = graph.clone();
            async move {
                graph
                    .get_chat_events(id, 0, 500)
                    .await
                    .unwrap()
                    .iter()
                    .any(|e| e.data.contains(needle))
            }
        };
        let mut ready = false;
        for _ in 0..400 {
            if stored("\"ready\"").await && !manager.is_session_streaming(&sid).await {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(ready, "the first turn ended");

        // From now on the OOB listener of this session lags behind the turn.
        crate::chat::oob_listener::TEST_LAG
            .lock()
            .unwrap()
            .push((sid.clone(), lag));
        manager.send_message(&sid, "SECOND-MESSAGE").await.unwrap();
        let mut answered = false;
        for _ in 0..400 {
            if stored("answered second").await {
                answered = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        // Long enough for the lagging listener to get to every message of the turn.
        tokio::time::sleep(lag * 6 + Duration::from_secs(1)).await;
        crate::chat::oob_listener::TEST_LAG
            .lock()
            .unwrap()
            .retain(|(s, _)| s != &sid);

        let stdin = std::fs::read_to_string(dir.path().join("stdin.jsonl")).unwrap_or_default();
        let events: Vec<(String, String)> = graph
            .get_chat_events(id, 0, 500)
            .await
            .unwrap()
            .iter()
            .map(|e| (e.event_type.clone(), e.data.chars().take(80).collect()))
            .collect();
        assert!(answered, "the turn answered: {events:?}");
        assert!(
            !stdin.contains("[tool_result]"),
            "the turn's own tool_result started a turn of its own: {stdin}"
        );
        assert!(
            !events.iter().any(|(t, _)| t == "background_output"),
            "the turn's own output was taken for background output: {events:?}"
        );
        manager.close_session(&sid).await.unwrap();
    }
}

/// P8 — the record of a native session (the agent engine) is kept as the legacy
/// engine keeps its own (`chat::session_record`): before, it stayed at what the
/// creation wrote (`message_count: 1`, no cost, no title) whatever the conversation did.
mod native_session_record {
    use super::*;

    fn answer(key: &str) -> Value {
        sse_route(
            key,
            vec![
                delta(json!({"content": format!("answer to {key}")})),
                json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}),
                json!("[DONE]"),
            ],
        )
    }

    async fn record_until(
        graph: &MockGraphStore,
        sid: &str,
        done: impl Fn(&crate::neo4j::models::ChatSessionNode) -> bool,
    ) -> crate::neo4j::models::ChatSessionNode {
        let id = Uuid::parse_str(sid).unwrap();
        let mut last = None;
        for _ in 0..400 {
            if let Some(node) = graph.get_chat_session(id).await.unwrap() {
                if done(&node) {
                    return node;
                }
                last = Some(node);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the record of {sid} never got there: {last:?}");
    }

    async fn results(graph: &MockGraphStore, sid: &str) -> usize {
        graph
            .get_chat_events(Uuid::parse_str(sid).unwrap(), 0, 500)
            .await
            .unwrap()
            .iter()
            .filter(|e| e.event_type == "result")
            .count()
    }

    async fn wait_results(graph: &MockGraphStore, sid: &str, count: usize) {
        for _ in 0..400 {
            if results(graph, sid).await >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("{count} result event(s) never persisted");
    }

    #[tokio::test]
    async fn a_native_session_keeps_its_title_its_message_count_and_its_cost_across_a_resume() {
        let mut routes = vec![answer("AFTER-RESTART"), answer("SECOND-TURN")];
        routes.extend(script().as_array().cloned().unwrap());
        let fake = FakeOpenAi::start(Value::Array(routes));
        let graph = Arc::new(MockGraphStore::new());
        store_instance(&graph, &instance(&fake, "none")).await;
        consent(&graph, "proj", "local", &fake.origin()).await;
        let data = tempfile::tempdir().unwrap();
        let root = data.path().join("native-transcripts");

        let before = manager(graph.clone(), true).with_native_transcripts(&root);
        let sid = before
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap()
            .session_id;
        assert!(before.agent_runtime.owns(&sid).await, "the agent engine");
        wait_results(&graph, &sid, 1).await;
        // A free endpoint: its turns cost a real 0, recorded.
        let node = record_until(&graph, &sid, |n| n.total_cost_usd.is_some()).await;
        assert_eq!(node.message_count, 1, "the opening message, once: {node:?}");
        assert_eq!(node.total_cost_usd, Some(0.0), "{node:?}");
        assert_eq!(node.title.as_deref(), Some("hi there"), "{node:?}");
        assert_eq!(node.preview.as_deref(), Some("hi there"), "{node:?}");

        before.send_message(&sid, "SECOND-TURN").await.unwrap();
        wait_results(&graph, &sid, 2).await;
        let node = record_until(&graph, &sid, |n| n.message_count >= 2).await;
        assert_eq!(node.message_count, 2, "{node:?}");
        before.close_session(&sid).await.unwrap();
        drop(before);

        // The message that resumes the session after a restart counts too.
        let after = manager(graph.clone(), true).with_native_transcripts(&root);
        let claims = crate::auth::jwt::Claims::service_account("e2e");
        after
            .resume_session(&sid, "AFTER-RESTART", Some(&claims))
            .await
            .unwrap_or_else(|e| panic!("resume failed: {e:#}"));
        wait_results(&graph, &sid, 3).await;
        let node = record_until(&graph, &sid, |n| n.message_count >= 3).await;
        assert_eq!(node.message_count, 3, "{node:?}");
        assert_eq!(node.title.as_deref(), Some("hi there"), "the title stays");
        assert_eq!(node.total_cost_usd, Some(0.0));
        after.close_session(&sid).await.unwrap();
    }
}

/// P8 — the lifecycle of the agent engine's sessions follows the chat's
/// configuration as the legacy engine's do: the idle cleanup closes them, they
/// count in `max_sessions` / `active_session_count`, and their retries follow
/// `RetryConfig`. Before, they were invisible to all three.
mod agent_lifecycle {
    use super::*;

    fn configured(
        graph: Arc<MockGraphStore>,
        edit: impl FnOnce(&mut super::super::config::ChatConfig),
    ) -> Arc<ChatManager> {
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph;
        let mut config = super::super::config::ChatConfig {
            provider_path: ProviderPath::Agent,
            mcp_server_path: fake_bin("fake_mcp"),
            nexus_tools_path: None,
            nexus_browser_path: None,
            jwt_secret: Some("test-secret-test-secret-test-secret".to_string()),
            max_sessions: 10,
            ..Default::default()
        };
        edit(&mut config);
        Arc::new(ChatManager::new_without_memory(
            dyn_graph,
            state.meili,
            config,
        ))
    }

    async fn native_world() -> (FakeOpenAi, Arc<MockGraphStore>) {
        let fake = FakeOpenAi::start(script());
        let graph = Arc::new(MockGraphStore::new());
        store_instance(&graph, &instance(&fake, "none")).await;
        consent(&graph, "proj", "local", &fake.origin()).await;
        (fake, graph)
    }

    async fn idle(manager: &ChatManager, sid: &str) {
        for _ in 0..400 {
            if !manager.is_session_streaming(sid).await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the opening turn never ended");
    }

    #[tokio::test]
    async fn an_idle_native_session_is_closed_by_the_cleanup() {
        let (_fake, graph) = native_world().await;
        let manager = configured(graph, |c| c.session_timeout = Duration::from_millis(300));
        let sid = manager
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap()
            .session_id;
        assert!(manager.agent_runtime.owns(&sid).await, "the agent engine");
        idle(&manager, &sid).await;
        manager.start_cleanup_task();
        for _ in 0..200 {
            if !manager.agent_runtime.owns(&sid).await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("an idle native session outlived the session timeout");
    }

    #[tokio::test]
    async fn native_sessions_count_toward_max_sessions_and_get_the_chat_retry_config() {
        let (_fake, graph) = native_world().await;
        let manager = configured(graph, |c| {
            c.max_sessions = 1;
            c.retry = super::super::config::RetryConfig {
                max_attempts: 7,
                initial_delay_ms: 5,
                backoff_multiplier: 1.5,
            };
        });
        let sid = manager
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap()
            .session_id;
        assert_eq!(manager.active_session_count().await, 1);
        let live = manager.live_session_snapshot().await;
        assert!(
            live.live.contains(&Uuid::parse_str(&sid).unwrap()),
            "the cockpit sees the native session live"
        );
        let second = manager
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await;
        let err = second.expect_err("the limit refuses a second session");
        assert!(
            err.to_string()
                .contains("Maximum number of active sessions"),
            "{err:#}"
        );
        let retry = manager
            .agent_runtime
            .get(&sid)
            .await
            .unwrap()
            .retry_config();
        assert_eq!(
            (retry.max_attempts, retry.initial_delay_ms),
            (7, 5),
            "the chat's RetryConfig, not a constant"
        );
        manager.close_session(&sid).await.unwrap();
        assert_eq!(manager.active_session_count().await, 0);
    }
}

/// P8 (c) — the post-stream steps of the Claude Code engine run at the end of each
/// turn of the agent engine, with the same functions (`post_stream`): the context
/// re-injected after a compaction, the objective reminder, and `cancel_task`
/// refused, typed, instead of a silent success.
mod post_turn {
    use super::parity::{caps, rig};
    use super::*;
    use nexus_claude::agent::{AgentEvent, CompactionPhase, ProviderKind};
    use nexus_claude::testkit::scripted::steps;
    use nexus_claude::testkit::Step;

    async fn seed_pending_tasks(graph: &MockGraphStore, project: Uuid) {
        use crate::neo4j::models::{PlanNode, PlanStatus, TaskStatus};
        let plan = PlanNode::new_for_project(
            "Active Plan".into(),
            "Plan with pending tasks".into(),
            "test".into(),
            50,
            project,
        );
        graph.create_plan(&plan).await.unwrap();
        graph
            .update_plan_status(plan.id, PlanStatus::InProgress)
            .await
            .unwrap();
        let mut task = crate::test_helpers::test_task_titled("Fix the parser bug");
        task.status = TaskStatus::Pending;
        graph.create_task(plan.id, &task).await.unwrap();
    }

    #[tokio::test]
    async fn after_a_compaction_the_context_is_reinjected_and_the_objectives_recalled() {
        let compacted = Step::Emit(AgentEvent::Compaction {
            phase: CompactionPhase::Completed,
            trigger: None,
            pre_tokens: Some(1000),
        });
        let quiet = || vec![steps::done(&caps())];
        let mut r = rig(
            ProviderKind::Native,
            vec![
                vec![compacted, steps::done(&caps())],
                quiet(),
                quiet(),
                quiet(),
                quiet(),
            ],
        )
        .await;
        seed_pending_tasks(&r.graph, r.project.id).await;
        r.manager.send_message(&r.sid, "go on").await.unwrap();
        let recovery = next_event(&mut r.rx, |e| {
            matches!(e, ChatEvent::CompactionRecovery { .. })
        })
        .await;
        assert!(
            matches!(
                recovery,
                ChatEvent::CompactionRecovery {
                    recovery_success: true,
                    ..
                }
            ),
            "{recovery:?}"
        );
        for _ in 0..400 {
            if !r.manager.is_session_streaming(&r.sid).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        // No turn of its own for the context, nor any automated turn on the history
        // just compacted: the run is the user's turn alone.
        assert_eq!(r.sent().len(), 1, "{:#?}", r.sent());
        // The next turn carries in front, once, the context and the reminder of the
        // turn that compacted (it did no work: the objective is pending).
        r.manager.send_message(&r.sid, "next").await.unwrap();
        r.turn_end().await;
        for _ in 0..400 {
            if !r.manager.is_session_streaming(&r.sid).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let sent = r.sent();
        assert!(
            sent[1].contains("Post-Compaction Context")
                && sent[1].contains(&r.project.name)
                && sent[1].contains("next"),
            "the context of the project, in front of the next turn: {}",
            sent[1]
        );
        assert!(
            sent[2..]
                .iter()
                .all(|s| !s.contains("Post-Compaction Context")),
            "re-injected once: {sent:#?}"
        );
        assert!(
            sent[1].contains(super::super::post_stream::OBJECTIVE_REMINDER_MARKER)
                && sent[1].contains("Fix the parser bug"),
            "the pending objective recalled with the context: {}",
            sent[1]
        );
        assert!(
            sent.len() <= 4,
            "the reminders stop at their cap: {sent:#?}"
        );
    }

    /// After a compaction, the context comes back BEFORE the model continues: the
    /// continuation turn carries it in front of "Continue…" (the legacy engine queues
    /// the re-injection before the auto-continue hint).
    #[tokio::test]
    async fn after_a_compaction_the_continuation_carries_the_context_first() {
        let compacted = Step::Emit(AgentEvent::Compaction {
            phase: CompactionPhase::Completed,
            trigger: None,
            pre_tokens: Some(1000),
        });
        let r = rig(
            ProviderKind::Native,
            vec![
                vec![
                    compacted,
                    Step::Emit(nexus_claude::testkit::done_event(
                        &caps(),
                        nexus_claude::agent::StopReason::MaxTurns,
                        None,
                    )),
                ],
                vec![steps::done(&caps())],
                vec![steps::done(&caps())],
            ],
        )
        .await;
        r.manager.set_auto_continue(&r.sid, true).await.unwrap();
        r.manager.send_message(&r.sid, "go").await.unwrap();
        for _ in 0..400 {
            if r.sent().len() >= 2 && !r.manager.is_session_streaming(&r.sid).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let sent = r.sent();
        assert_eq!(sent.len(), 2, "the turn, then its continuation: {sent:#?}");
        let context = sent[1].find("Post-Compaction Context");
        let cont = sent[1].find("Continue where you left off");
        assert!(
            matches!((context, cont), (Some(a), Some(b)) if a < b),
            "the context first, then the continuation: {}",
            sent[1]
        );
    }

    /// A tool call announced before its input was complete (Claude Code, ACP) is
    /// judged on its RESOLVED input: a `git commit` is conclusive, so a turn that
    /// only committed still gets the objective reminder.
    #[tokio::test]
    async fn a_tool_is_judged_on_its_resolved_input() {
        let r = rig(
            ProviderKind::ClaudeCode,
            vec![
                vec![
                    steps::tool_call_start("c1", "Bash"),
                    steps::tool_call("c1", "Bash", json!({ "command": "git commit -m done" })),
                    steps::tool_result("c1", "committed"),
                    steps::done(&caps()),
                ],
                vec![steps::done(&caps())],
                vec![steps::done(&caps())],
            ],
        )
        .await;
        seed_pending_tasks(&r.graph, r.project.id).await;
        r.manager.send_message(&r.sid, "commit it").await.unwrap();
        for _ in 0..400 {
            if r.sent().len() >= 2 && !r.manager.is_session_streaming(&r.sid).await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let sent = r.sent();
        assert!(
            sent.iter()
                .skip(1)
                .any(|s| s.contains(super::super::post_stream::OBJECTIVE_REMINDER_MARKER)),
            "a commit alone is wrapping up, not working: the objectives are recalled: {sent:#?}"
        );
    }

    #[tokio::test]
    async fn cancel_task_on_the_agent_engine_is_refused_typed() {
        let r = rig(ProviderKind::Native, vec![vec![steps::done(&caps())]]).await;
        let err = r
            .manager
            .cancel_task(&r.sid, "task-1")
            .await
            .expect_err("no silent success");
        assert!(
            err.downcast_ref::<super::super::manager::CancelTaskUnsupported>()
                .is_some(),
            "{err:#}"
        );
    }
}

/// P8c — the context re-injected after a compaction never starts a turn of its
/// own: such a turn is measured against the window right after the compaction and
/// compacted again (measured on integ/p8: a second summarisation call whose history
/// held the re-injected context). The context rides in front of the session's next
/// turn instead. What is guaranteed: the re-injection itself never starts a turn,
/// hence never a compaction; in this scenario (one model call per message) a user
/// turn compacts at most once. Nexus may still compact between the tool steps of
/// one turn: that is its own policy, not bounded here.
mod compaction_loop {
    use super::*;

    const SUMMARY_PROMPT: &str = "You are compacting the history";
    const TURNS: [&str; 5] = [
        "turn two",
        "turn three",
        "turn four",
        "turn five",
        "turn six",
    ];

    fn answer(text: &str, prompt_tokens: u64) -> Vec<Value> {
        vec![
            delta(json!({ "content": text })),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 4, "total_tokens": prompt_tokens + 4}}),
            json!("[DONE]"),
        ]
    }

    /// A 32k window; every long turn reports a prompt near it (31k), so each turn
    /// after enough history calls for a compaction before it.
    fn script() -> Value {
        let mut routes = vec![
            sse_route(
                "Call the ping tool now",
                vec![
                    delta(
                        json!({"tool_calls": [{"index": 0, "id": "p1", "function": {"name": "ping", "arguments": "{}"}}]}),
                    ),
                    json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
                    json!("[DONE]"),
                ],
            ),
            json!({"method": "GET", "path": "/v1/models", "status": 200,
                   "body": {"object": "list", "data": [{"id": "m", "context_length": 32000}]}}),
            sse_route(SUMMARY_PROMPT, answer("a dense summary", 50)),
        ];
        for text in TURNS.iter().rev() {
            routes.push(sse_route(text, answer("noted", 31_000)));
        }
        routes.push(sse_route("hi there", answer("hello", 10)));
        Value::Array(routes)
    }

    fn bodies(fake: &FakeOpenAi) -> Vec<String> {
        fake.chat_requests()
            .iter()
            .map(|r| r["body"].to_string())
            .filter(|b| !b.contains("Call the ping tool now"))
            .collect()
    }

    async fn idle(manager: &ChatManager, sid: &str) {
        for _ in 0..600 {
            if !manager.is_session_streaming(sid).await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the session never went idle");
    }

    #[tokio::test]
    async fn a_reinjected_context_never_triggers_a_compaction_and_a_user_turn_compacts_at_most_once(
    ) {
        let fake = FakeOpenAi::start(script());
        let graph = Arc::new(MockGraphStore::new());
        store_instance(&graph, &instance(&fake, "none")).await;
        consent(&graph, "proj", "local", &fake.origin()).await;
        let manager = manager(graph.clone(), true);
        let sid = manager
            .create_session(&request(Some("local"), Some("proj"), "default"))
            .await
            .unwrap()
            .session_id;
        idle(&manager, &sid).await;
        let mut rx = manager.subscribe(&sid).await.unwrap();

        let mut recovered = 0;
        for text in TURNS {
            let before = bodies(&fake).len();
            manager.send_message(&sid, text).await.unwrap();
            // Its result, then the end of its run (hints included).
            next_event(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
            idle(&manager, &sid).await;
            let run: Vec<String> = bodies(&fake)[before..].to_vec();
            let compactions = run.iter().filter(|b| b.contains(SUMMARY_PROMPT)).count();
            assert!(
                compactions <= 1,
                "{text}: {compactions} compactions in one user turn: {run:#?}"
            );
            for body in run.iter().filter(|b| b.contains(SUMMARY_PROMPT)) {
                assert!(
                    !body.contains("Post-Compaction Context"),
                    "{text}: a compaction summarised the re-injected context: {body}"
                );
            }
            let turns = run.iter().filter(|b| !b.contains(SUMMARY_PROMPT)).count();
            assert_eq!(turns, 1, "{text}: one model turn per user turn: {run:#?}");
            while let Ok(event) = rx.try_recv() {
                recovered += usize::from(matches!(event, ChatEvent::CompactionRecovery { .. }));
            }
        }
        assert!(recovered >= 1, "the scenario compacted at least once");
        // The context of the last compaction reaches the model, in front of a turn.
        let last = bodies(&fake)
            .into_iter()
            .filter(|b| !b.contains(SUMMARY_PROMPT))
            .rfind(|b| b.contains("Post-Compaction Context"));
        assert!(
            last.is_some(),
            "the re-injected context rode on a user turn"
        );
        manager.close_session(&sid).await.unwrap();
    }
}
