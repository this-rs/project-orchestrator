//! Routes of the network tools (P6): auth, validation, persistence, and the end-to-end effect on
//! a native session's tool list.

use crate::api::handlers::{OrchestratorState, ServerState};
use crate::api::routes::create_router;
use crate::chat::provider::nexus_tools::{GraphConsents, ToolConsents};
use crate::neo4j::mock::MockGraphStore;
use crate::neo4j::GraphStore;
use crate::orchestrator::{FileWatcher, Orchestrator};
use crate::test_helpers::{mock_app_state_with_graph, test_bearer_token, test_project_named};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

const PROJECT: &str = "proj";

async fn state_on(graph: Arc<MockGraphStore>) -> OrchestratorState {
    let app_state = mock_app_state_with_graph(graph);
    let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
    let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
        orchestrator.clone(),
    )));
    Arc::new(ServerState {
        orchestrator,
        watcher,
        chat_manager: None,
        event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
            crate::events::EventBus::default(),
        ))),
        nats_emitter: None,
        auth_config: Some(crate::test_helpers::test_auth_config()),
        serve_frontend: false,
        frontend_path: "./dist".to_string(),
        setup_completed: true,
        server_port: 6600,
        public_url: None,
        remote_mcp: crate::RemoteMcpConfig::default(),
        ws_ticket_store: Arc::new(crate::api::ws_auth::WsTicketStore::new()),
        registry_remote_url: None,
        oidc_client: None,
        neural_router: crate::test_helpers::mock_neural_router(),
        trajectory_collector: std::sync::RwLock::new(None),
        trajectory_store_neo4j: None,
        trajectory_store: None,
        identity: None,
        reactor_counters: std::sync::OnceLock::new(),
        confidence_tracker: Arc::new(crate::graph::confidence::ConfidenceTracker::default()),
        mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
        model_catalog: crate::chat::model_catalog::ModelCatalogCache::new(None),
        vault: crate::vault::VaultService::ephemeral(),
    })
}

/// A graph holding the project `proj`, and a router over it.
async fn app() -> (axum::Router, Arc<MockGraphStore>) {
    let graph = Arc::new(MockGraphStore::new());
    graph
        .create_project(&test_project_named(PROJECT))
        .await
        .unwrap();
    (create_router(state_on(graph.clone()).await), graph)
}

fn req(token: &str, method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
    let b = Request::builder()
        .method(method)
        .uri(uri)
        .header("authorization", token);
    match body {
        Some(v) => b
            .header("content-type", "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

fn human(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
    req(&test_bearer_token(), method, uri, body)
}

/// A bound, live agent token.
fn agent_bearer() -> String {
    let session = Uuid::new_v4();
    let claims = crate::auth::jwt::Claims::service_account("agent");
    let binding = crate::auth::jwt::AgentSessionBinding {
        session_id: session.to_string(),
        ceiling: Some("default".into()),
        tool_profile: Some("restricted".into()),
        third_party: false,
    };
    let (token, jti) = crate::auth::jwt::generate_session_token(
        &claims,
        Some(&binding),
        &crate::test_helpers::test_auth_config().jwt_secret,
        3600,
    )
    .unwrap();
    crate::auth::agent_tokens::register(&jti, Some(&session.to_string()));
    format!("Bearer {token}")
}

async fn call(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(request).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, body)
}

const ORIGINS: &str = "/api/projects/proj/network-tools/origins";
const BROWSER: &str = "/api/projects/proj/network-tools/browser";
const SUMMARY: &str = "/api/projects/proj/network-tools";
const ENGINES: &str = "/api/chat/search-engines";

// ---------------------------------------------------------------------------
// Origins
// ---------------------------------------------------------------------------

#[tokio::test]
async fn network_origin_consent_is_stored_where_the_gate_reads_it_and_revoked() {
    let (app, graph) = app().await;
    let consents = GraphConsents(graph.clone());
    assert!(!consents.any_origin_consented(PROJECT).await.unwrap());
    // A URL is reduced to its origin, lowercased, default port dropped.
    let (status, body) = call(
        &app,
        human(
            "PUT",
            ORIGINS,
            Some(json!({"origin": "HTTPS://Docs.RS:443/serde/latest?x=1"})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["origin"], "https://docs.rs");
    assert_eq!(body["consented_by"], "test@ffs.holdings");
    assert!(body["consented_at"].is_string());
    // The predicate of B40 sees it, under the key it reads.
    assert!(consents
        .origin_consented(PROJECT, "https://docs.rs")
        .await
        .unwrap());
    assert!(consents.any_origin_consented(PROJECT).await.unwrap());
    assert!(!consents
        .origin_consented(PROJECT, "http://docs.rs")
        .await
        .unwrap());
    // A non-default port is part of the origin.
    let (status, _) = call(
        &app,
        human(
            "PUT",
            ORIGINS,
            Some(json!({"origin": "http://localhost:8888"})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, summary) = call(&app, human("GET", SUMMARY, None)).await;
    assert_eq!(status, StatusCode::OK, "{summary}");
    let listed: Vec<&str> = summary["origins"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["origin"].as_str().unwrap())
        .collect();
    assert_eq!(listed, ["http://localhost:8888", "https://docs.rs"]);
    assert_eq!(summary["browser"]["allowed"], false);
    // Revoke: by origin (a URL is reduced the same way), then 404 the second time.
    let uri = format!("{ORIGINS}?origin=https%3A%2F%2Fdocs.rs%2Fanything");
    let (status, _) = call(&app, human("DELETE", &uri, None)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(!consents
        .origin_consented(PROJECT, "https://docs.rs")
        .await
        .unwrap());
    let (status, body) = call(&app, human("DELETE", &uri, None)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "tool_origin_not_found");
}

#[tokio::test]
async fn network_origin_refuses_what_is_not_an_http_origin_with_a_typed_400_and_stores_nothing() {
    let (app, graph) = app().await;
    for body in [
        json!({"origin": "ftp://files.example.com"}),
        json!({"origin": "file:///etc/passwd"}),
        json!({"origin": "not a url"}),
        json!({"origin": ""}),
        json!({"origin": "*"}),
        json!({"origin": "https://user:hunter2@example.com"}),
        json!({"origin": format!("https://example.com/{}", "a".repeat(3000))}),
        json!({"origin": "https://example.com", "extra": true}),
        json!({"url": "https://example.com"}),
        json!([]),
    ] {
        let (status, got) = call(&app, human("PUT", ORIGINS, Some(body.clone()))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} -> {got}");
        assert_eq!(got["code"], "invalid_tool_origin", "{body} -> {got}");
        assert!(
            !got.to_string().contains("hunter2"),
            "a refused value is never echoed: {got}"
        );
    }
    let (status, got) = call(&app, human("DELETE", ORIGINS, None)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    assert!(!GraphConsents(graph)
        .any_origin_consented(PROJECT)
        .await
        .unwrap());
}

#[tokio::test]
async fn network_tools_of_an_unknown_project_are_a_404() {
    let (app, _) = app().await;
    for (method, uri, body) in [
        ("GET", "/api/projects/nope/network-tools", None),
        (
            "PUT",
            "/api/projects/nope/network-tools/origins",
            Some(json!({"origin": "https://docs.rs"})),
        ),
        (
            "PUT",
            "/api/projects/nope/network-tools/browser",
            Some(json!({"allowed": true})),
        ),
    ] {
        let (status, got) = call(&app, human(method, uri, body)).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{method} {uri} -> {got}");
        assert_eq!(got["code"], "project_not_found");
    }
}

#[tokio::test]
async fn network_tools_mutations_are_refused_to_an_agent_token_and_reads_are_not() {
    let (app, graph) = app().await;
    let agent = agent_bearer();
    for (method, uri, body) in [
        (
            "PUT",
            ORIGINS.to_string(),
            Some(json!({"origin": "https://docs.rs"})),
        ),
        ("DELETE", format!("{ORIGINS}?origin=https://docs.rs"), None),
        ("PUT", BROWSER.to_string(), Some(json!({"allowed": true}))),
        (
            "POST",
            ENGINES.to_string(),
            Some(json!({"id": "sx", "engine": "searxng", "base_url": "https://sx.example"})),
        ),
        ("DELETE", format!("{ENGINES}/sx"), None),
    ] {
        let (status, got) = call(&app, req(&agent, method, &uri, body)).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{method} {uri} -> {got}");
    }
    let consents = GraphConsents(graph);
    assert!(!consents.any_origin_consented(PROJECT).await.unwrap());
    assert!(!consents.browser_authorized(PROJECT).await.unwrap());
    assert!(consents.search_providers().await.is_empty());
    // Reads stay open.
    let (status, _) = call(&app, req(&agent, "GET", SUMMARY, None)).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = call(&app, req(&agent, "GET", ENGINES, None)).await;
    assert_eq!(status, StatusCode::OK);
    // Without a token at all: 401.
    let anonymous = Request::builder()
        .method("PUT")
        .uri(ORIGINS)
        .header("content-type", "application/json")
        .body(Body::from(json!({"origin": "https://docs.rs"}).to_string()))
        .unwrap();
    let (status, _) = call(&app, anonymous).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------

#[tokio::test]
async fn network_browser_authorisation_is_stored_read_by_the_gate_and_withdrawn() {
    let (app, graph) = app().await;
    let consents = GraphConsents(graph);
    let (status, body) = call(&app, human("PUT", BROWSER, Some(json!({"allowed": true})))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["allowed"], true);
    assert_eq!(body["authorized_by"], "test@ffs.holdings");
    assert!(consents.browser_authorized(PROJECT).await.unwrap());
    let (_, summary) = call(&app, human("GET", SUMMARY, None)).await;
    assert_eq!(summary["browser"]["allowed"], true);
    let (status, body) = call(&app, human("PUT", BROWSER, Some(json!({"allowed": false})))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, json!({"allowed": false}));
    assert!(!consents.browser_authorized(PROJECT).await.unwrap());
    // Denying twice is still a deny, not an error.
    let (status, _) = call(&app, human("PUT", BROWSER, Some(json!({"allowed": false})))).await;
    assert_eq!(status, StatusCode::OK);
    for bad in [
        json!({"allowed": "yes"}),
        json!({}),
        json!({"allowed": true, "stealth": true}),
    ] {
        let (status, got) = call(&app, human("PUT", BROWSER, Some(bad.clone()))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad} -> {got}");
        assert_eq!(got["code"], "invalid_browser_setting");
    }
    assert!(!consents.browser_authorized(PROJECT).await.unwrap());
}

// ---------------------------------------------------------------------------
// Search engines
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_engine_is_declared_by_vault_reference_listed_and_removed() {
    let (app, graph) = app().await;
    let brave = json!({"id": "brave", "engine": "brave", "credential_ref": "vault:brave_key"});
    let (status, body) = call(&app, human("POST", ENGINES, Some(brave.clone()))).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["id"], "brave");
    assert_eq!(body["credential_ref"], "vault:brave_key");
    assert_eq!(body["origin"], "https://api.search.brave.com");
    assert_eq!(body["grant_id"], "tool:brave");
    assert_eq!(body["key_granted"], false);
    let (status, body) = call(&app, human("POST", ENGINES, Some(brave))).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["code"], "search_engine_exists");
    let (status, _) = call(
        &app,
        human(
            "POST",
            ENGINES,
            Some(json!({"id": "home", "engine": "searxng", "base_url": "http://localhost:8888/search"})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    // The B40 reader sees both, as stored.
    let stored = GraphConsents(graph.clone()).search_providers().await;
    let mut ids: Vec<&str> = stored.iter().map(|p| p.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["brave", "home"]);
    // Consent the SearXNG origin for the project: the listing says which engine it can use.
    let (status, _) = call(
        &app,
        human(
            "PUT",
            ORIGINS,
            Some(json!({"origin": "http://localhost:8888"})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (_, listed) = call(&app, human("GET", &format!("{ENGINES}?project=proj"), None)).await;
    let flags: Vec<(&str, bool)> = listed
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["id"].as_str().unwrap(),
                e["origin_consented"].as_bool().unwrap(),
            )
        })
        .collect();
    assert_eq!(flags, [("brave", false), ("home", true)]);
    let (_, summary) = call(&app, human("GET", SUMMARY, None)).await;
    assert_eq!(summary["search_engines"].as_array().unwrap().len(), 2);
    // Without a project, no consent flag.
    let (_, listed) = call(&app, human("GET", ENGINES, None)).await;
    assert!(listed[0].get("origin_consented").is_none(), "{listed}");
    // Remove.
    let (status, _) = call(&app, human("DELETE", &format!("{ENGINES}/home"), None)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (status, body) = call(&app, human("DELETE", &format!("{ENGINES}/home"), None)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "search_engine_not_found");
    assert_eq!(GraphConsents(graph).search_providers().await.len(), 1);
}

#[tokio::test]
async fn search_engine_refuses_a_raw_key_and_invalid_drafts_with_typed_codes() {
    let (app, graph) = app().await;
    let secret = "sk-live-0123456789abcdef";
    for (body, code) in [
        (
            json!({"id": "brave", "engine": "brave", "api_key": secret}),
            "secret_value_refused",
        ),
        (
            json!({"id": "brave", "engine": "brave", "credential_ref": "vault:k", "Key": secret}),
            "secret_value_refused",
        ),
        (
            json!({"id": "brave", "engine": "brave", "credential_ref": format!("env:{secret}")}),
            "invalid_credential_ref",
        ),
        (
            json!({"id": "brave", "engine": "brave", "credential_ref": secret}),
            "invalid_credential_ref",
        ),
        (
            json!({"id": "brave", "engine": "brave"}),
            "credential_ref_required",
        ),
        (
            json!({"id": "brave", "engine": "brave", "credential_ref": "vault:k", "base_url": "https://x.example"}),
            "invalid_search_engine",
        ),
        (
            json!({"id": "Brave!", "engine": "brave", "credential_ref": "vault:k"}),
            "invalid_search_engine_id",
        ),
        (
            json!({"id": "g", "engine": "google", "credential_ref": "vault:k"}),
            "unknown_search_engine",
        ),
        (
            json!({"id": "sx", "engine": "searxng"}),
            "invalid_search_engine_url",
        ),
        (
            json!({"id": "sx", "engine": "searxng", "base_url": "ftp://sx.example"}),
            "invalid_search_engine_url",
        ),
        (
            json!({"id": "sx", "engine": "searxng", "base_url": "https://sx.example", "credential_ref": "vault:k"}),
            "invalid_credential_ref",
        ),
        (
            json!({"id": "sx", "engine": "searxng", "base_url": "https://sx.example", "label": "x"}),
            "invalid_search_engine",
        ),
        (json!("brave"), "invalid_search_engine"),
    ] {
        let (status, got) = call(&app, human("POST", ENGINES, Some(body.clone()))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} -> {got}");
        assert_eq!(got["code"], code, "{body} -> {got}");
        assert!(
            !got.to_string().contains(secret),
            "a refused value is never echoed: {got}"
        );
    }
    assert!(GraphConsents(graph).search_providers().await.is_empty());
}

#[tokio::test]
async fn search_engine_key_is_granted_under_tool_id_only_by_its_reference_and_revoked_with_it() {
    let (app, _) = app().await;
    let (status, _) = call(
        &app,
        human(
            "POST",
            ENGINES,
            Some(json!({"id": "brave", "engine": "brave", "credential_ref": "vault:brave_key"})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, init) = call(
        &app,
        human(
            "POST",
            "/api/vault/init",
            Some(json!({"passphrase": "correct horse battery staple 2026"})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{init}");
    let proof = init["unlock_proof"].as_str().unwrap().to_owned();
    let grant = |names: Value, value: &str| {
        let mut request = human(
            "POST",
            "/api/vault/grants",
            Some(json!({
                "secrets": {"kind": "names", "names": names},
                "scope": {"kind": "provider", "value": value},
            })),
        );
        request
            .headers_mut()
            .insert("x-vault-proof", proof.parse().unwrap());
        request
    };
    // Another secret than the engine's reference, `all`, or an unknown engine: refused.
    let (status, got) = call(&app, grant(json!(["other"]), "tool:brave")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    let mut all = grant(json!([]), "tool:brave");
    *all.body_mut() = Body::from(
        json!({"secrets": {"kind": "all"}, "scope": {"kind": "provider", "value": "tool:brave"}})
            .to_string(),
    );
    let (status, got) = call(&app, all).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{got}");
    let (status, got) = call(&app, grant(json!(["brave_key"]), "tool:nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{got}");
    // Its own reference: granted, and the listing says so.
    let (status, got) = call(&app, grant(json!(["brave_key"]), "tool:brave")).await;
    assert_eq!(status, StatusCode::CREATED, "{got}");
    let (_, listed) = call(&app, human("GET", ENGINES, None)).await;
    assert_eq!(listed[0]["key_granted"], true, "{listed}");
    // Removing the engine revokes its grant: one declared again under the id starts without.
    let (status, _) = call(&app, human("DELETE", &format!("{ENGINES}/brave"), None)).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (_, vault) = call(&app, human("GET", "/api/vault", None)).await;
    assert!(
        vault["grants"]
            .as_array()
            .unwrap()
            .iter()
            .all(|g| g["scope"]["value"] != "tool:brave"),
        "{vault}"
    );
}

// ---------------------------------------------------------------------------
// End to end: the consent opens WebFetch to a native session
// ---------------------------------------------------------------------------

#[cfg(unix)]
/// The `nexus-tools` tools a native session of `proj` is opened with (`--tools` of its `nexus`
/// server; empty when nothing is attached).
async fn native_session_tools(graph: Arc<MockGraphStore>, sid: &str) -> Vec<String> {
    use std::os::unix::fs::PermissionsExt;
    let bin = tempfile::TempDir::new().unwrap();
    let program = bin.path().join("nexus-tools");
    std::fs::write(&program, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    let state = mock_app_state_with_graph(graph);
    let mut config = crate::chat::config::ChatConfig::from_env();
    config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
    config.nexus_tools_path = Some(program);
    config.nexus_browser_path = None;
    let manager =
        crate::chat::manager::ChatManager::new_without_memory(state.neo4j, state.meili, config);
    let claims = crate::auth::jwt::Claims::service_account("p6");
    let spec = manager
        .build_agent_spec(crate::chat::manager::AgentSpecInput {
            cwd: "/work/app",
            model: "m",
            system_prompt: "p",
            permission_mode: None,
            add_dirs: &[],
            user_claims: Some(&claims),
            session_id: sid,
            third_party: true,
            max_tokens: None,
            kind: nexus_claude::agent::ProviderKind::Native,
            remote_cwd: None,
            per_session_mcp: true,
            hooks: Some(crate::chat::manager::AgentHookScope {
                project_slug: Some(PROJECT.into()),
                task_id: None,
                runner: false,
            }),
        })
        .await
        .unwrap();
    crate::auth::agent_tokens::revoke_session(sid);
    match spec
        .mcp_servers
        .get(nexus_claude::providers::native::NEXUS_TOOLS_SERVER)
    {
        Some(nexus_claude::agent::McpServerSpec::Stdio { args, .. }) => args
            .windows(2)
            .find(|w| w[0] == "--tools")
            .map(|w| w[1].split(',').map(str::to_owned).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn a_native_session_is_offered_webfetch_once_the_project_consents_to_an_origin_by_the_route()
{
    let (app, graph) = app().await;
    let before = native_session_tools(graph.clone(), "p6-before").await;
    assert!(
        before.iter().any(|t| t == "Read"),
        "the file tools are attached: {before:?}"
    );
    assert!(
        !before.iter().any(|t| t == "WebFetch"),
        "no consent, no WebFetch: {before:?}"
    );
    let (status, _) = call(
        &app,
        human("PUT", ORIGINS, Some(json!({"origin": "https://docs.rs"}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let after = native_session_tools(graph.clone(), "p6-after").await;
    assert!(
        after.iter().any(|t| t == "WebFetch"),
        "a consented origin offers WebFetch: {after:?}"
    );
    // No engine declared: still no WebSearch.
    assert!(!after.iter().any(|t| t == "WebSearch"), "{after:?}");
    // Revoked: the next session is not offered it any more.
    let (status, _) = call(
        &app,
        human("DELETE", &format!("{ORIGINS}?origin=https://docs.rs"), None),
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let revoked = native_session_tools(graph, "p6-revoked").await;
    assert!(!revoked.iter().any(|t| t == "WebFetch"), "{revoked:?}");
}
