//! Auth middleware for Axum routes.
//!
//! Validates JWT Bearer tokens and injects Claims into request extensions.
//! In no-auth mode (auth_config is None), anonymous Claims are injected
//! and requests pass through freely (open access).

use crate::api::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::{decode_jwt, Claims, ANONYMOUS_USER_ID};
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};

/// The only routes a vault token may call.
pub const VAULT_AGENT_PATH_PREFIX: &str = "/api/vault/agent/";

/// Middleware that handles authentication adaptively.
///
/// # Behavior
/// 1. If `auth_config` is `None` → **open access** (no-auth mode):
///    inject anonymous Claims and pass through.
/// 2. If `auth_config` is `Some(...)` → **JWT required**:
///    a. Extract `Authorization: Bearer <token>` header → 401 if missing
///    b. Validate JWT with the configured secret → 401 if invalid/expired
///    c. Check `allowed_email_domain` if configured → 403 if domain mismatch
///    d. Inject `Claims` into request extensions for downstream handlers
///    e. A vault token is refused outside [`VAULT_AGENT_PATH_PREFIX`] → 403
pub async fn require_auth(
    State(state): State<OrchestratorState>,
    mut req: Request,
    next: Next,
) -> Result<Response, AppError> {
    // 1. No-auth mode: inject anonymous claims and pass through
    let auth_config = match state.auth_config.as_ref() {
        Some(config) => config,
        None => {
            req.extensions_mut().insert(Claims::anonymous());
            return Ok(next.run(req).await);
        }
    };

    // 2. Extract Bearer token from Authorization header
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AppError::Unauthorized("Missing Authorization header".to_string()))?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| AppError::Unauthorized("Invalid Authorization header format".to_string()))?;

    // 3. Decode and validate JWT
    let claims = decode_jwt(token, &auth_config.jwt_secret).map_err(|e| {
        // The jsonwebtoken error says which check failed; keep it server-side.
        tracing::debug!(error = %e, "JWT rejected");
        AppError::Unauthorized("Invalid or expired token".to_string())
    })?;

    // 4. Email allowlist, MCP-token revocation, agent-token liveness,
    //    vault-token path scope.
    enforce_token_policy(&state, auth_config, &claims, req.uri().path()).await?;

    // 4b. An agent's session token never reaches the routes that decide what
    //     agents may do or where a project's content may be sent.
    if claims.is_agent_session() && is_human_only_mutation(req.method(), req.uri().path()) {
        return Err(AppError::Forbidden(
            "this route requires a human session; an agent session token cannot change \
             providers, consent, roles, model policy or chat permissions"
                .to_string(),
        ));
    }

    // 5. Inject claims into request extensions
    req.extensions_mut().insert(claims);

    Ok(next.run(req).await)
}

/// Where a Bearer token is exchanged for a WebSocket ticket.
pub const WS_TICKET_PATH: &str = "/auth/ws-ticket";

/// Path prefixes whose MUTATION is reserved to a human session (decision A25):
/// provider instances, per-project consent, pilot/executor roles, model policy,
/// the chat permission config, and long-lived MCP tokens. Reads stay open.
const HUMAN_ONLY_MUTATION_PREFIXES: &[&str] = &[
    "/auth/mcp-tokens",
    "/api/chat/config",
    "/api/chat/providers",
    "/api/chat/roles",
    "/api/chat/model-policy",
    "/api/chat/model-aliases",
];

/// Per-project settings that decide where a project's content may go
/// (`/api/projects/{slug}/llm-consent`, `/llm-roles`, …).
const HUMAN_ONLY_PROJECT_SEGMENT: &str = "/llm-";

/// Whether `method path` is a mutation only a human session may perform.
///
/// An `agent_session` token sits in the agent's environment: if it could reach
/// these routes, a prompt injection could authorise an endpoint, widen the
/// permission config or re-route sessions — the agent would grant itself.
pub fn is_human_only_mutation(method: &axum::http::Method, path: &str) -> bool {
    use axum::http::Method;
    if matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS) {
        return false;
    }
    let under = |prefix: &str| {
        path == prefix
            || path
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with('/'))
    };
    HUMAN_ONLY_MUTATION_PREFIXES.iter().any(|p| under(p))
        || (path.starts_with("/api/projects/") && path.contains(HUMAN_ONLY_PROJECT_SEGMENT))
        // Answering a permission prompt IS the human's decision: an agent that
        // could post it would approve its own tool calls.
        || (path.starts_with("/api/chat/sessions/") && path.contains("/permissions/"))
}

/// Policy every decoded token must satisfy before it is trusted: the email
/// allowlist, MCP-token revocation and the vault-token path scope. Shared by
/// [`require_auth`] and every other entry point that turns a Bearer token into
/// a credential (e.g. WebSocket tickets), so none of them can drift.
pub async fn enforce_token_policy(
    state: &OrchestratorState,
    auth_config: &crate::AuthConfig,
    claims: &Claims,
    path: &str,
) -> Result<(), AppError> {
    // Email restrictions (domain + individual whitelist).
    // Bypass for the anonymous/MCP system user (ANONYMOUS_USER_ID = UUID nil)
    // since it's a machine identity used by `mcp_server` auto-auth, not a human
    // subject to email policies.
    let is_system_user = claims.sub == ANONYMOUS_USER_ID.to_string();
    if !is_system_user && !auth_config.is_email_allowed(&claims.email) {
        return Err(AppError::Forbidden(
            "Email not allowed by server policy".to_string(),
        ));
    }

    // MCP tokens are long-lived but revocable: signature + expiry alone
    // are not enough — the jti must still be active in the McpToken
    // store. Fail closed on a missing jti or a store error.
    if claims.is_mcp_token() {
        let jti = claims
            .jti
            .as_deref()
            .ok_or_else(|| AppError::Unauthorized("MCP token missing jti".to_string()))?;
        let active = state
            .orchestrator
            .neo4j()
            .is_mcp_token_active(jti)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "MCP token revocation check failed");
                AppError::Unauthorized("MCP token check failed".to_string())
            })?;
        if !active {
            return Err(AppError::Unauthorized(
                "MCP token revoked, expired or unknown".to_string(),
            ));
        }
    }

    // An agent session token is bound to its session and dies with it: the
    // `jti` must still be registered. Fail closed on a token without one
    // (minted before tokens were bound — its holder is gone anyway).
    if claims.is_agent_session() {
        let live = claims
            .jti
            .as_deref()
            .is_some_and(crate::auth::agent_tokens::is_live);
        if !live {
            return Err(AppError::Unauthorized(
                "Agent session token revoked: its session is closed".to_string(),
            ));
        }
        // The chat WebSocket answers permission prompts and changes a session's
        // permission mode: it is a human surface. No ticket for an agent.
        if path == WS_TICKET_PATH {
            return Err(AppError::Forbidden(
                "an agent session token cannot open a WebSocket".to_string(),
            ));
        }
    }

    // A vault token lives in an agent's shell environment. It opens the
    // agent read path and nothing else — otherwise that shell would hold a
    // key to the whole API.
    if crate::auth::jwt::vault_token_session(claims).is_some()
        && !path.starts_with(VAULT_AGENT_PATH_PREFIX)
    {
        return Err(AppError::Forbidden(
            "vault tokens are only valid for reading granted secrets".to_string(),
        ));
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::jwt::{encode_jwt, Claims};
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::mock_app_state;
    use crate::AuthConfig;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::Router;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt; // for `oneshot`

    const TEST_SECRET: &str = "test-secret-key-minimum-32-chars!!";

    fn test_auth_config() -> AuthConfig {
        AuthConfig {
            jwt_secret: TEST_SECRET.to_string(),
            access_token_expiry_secs: 3600,
            refresh_token_expiry_secs: 604800,
            allowed_email_domain: None,
            allowed_emails: None,
            frontend_url: None,
            additional_origins: vec![],
            allow_registration: false,
            root_account: None,
            oidc: None,
            google_client_id: Some("test".to_string()),
            google_client_secret: Some("test".to_string()),
            google_redirect_uri: Some("http://localhost/callback".to_string()),
        }
    }

    async fn make_server_state(auth_config: Option<AuthConfig>) -> OrchestratorState {
        let state = mock_app_state();
        let event_bus = Arc::new(crate::events::HybridEmitter::new(Arc::new(
            EventBus::default(),
        )));
        let orchestrator = Arc::new(
            Orchestrator::with_event_bus(state, event_bus.clone())
                .await
                .unwrap(),
        );
        let watcher = FileWatcher::new(orchestrator.clone());

        Arc::new(crate::api::handlers::ServerState {
            orchestrator,
            watcher: Arc::new(RwLock::new(watcher)),
            chat_manager: None,
            event_bus,
            nats_emitter: None,
            auth_config,
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

    /// Build a test router with the auth middleware applied
    async fn test_app(auth_config: Option<AuthConfig>) -> Router {
        let state = make_server_state(auth_config).await;

        // Simple handler that returns 200 OK
        async fn ok_handler() -> &'static str {
            "ok"
        }

        Router::new()
            .route("/test", get(ok_handler))
            .layer(from_fn_with_state(state.clone(), require_auth))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_no_auth_config_allows_access() {
        // No-auth mode: requests pass through freely (open access)
        let app = test_app(None).await;

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_no_auth_config_injects_anonymous_claims() {
        // Verify that anonymous Claims are injected in no-auth mode
        use crate::auth::jwt::ANONYMOUS_USER_ID;

        let state = make_server_state(None).await;

        // Handler that checks the injected claims
        async fn check_claims(axum::Extension(claims): axum::Extension<Claims>) -> String {
            format!("{}|{}|{}", claims.sub, claims.email, claims.name)
        }

        let app = Router::new()
            .route("/test", get(check_claims))
            .layer(from_fn_with_state(state.clone(), require_auth))
            .with_state(state);

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains(&ANONYMOUS_USER_ID.to_string()));
        assert!(body_str.contains("anonymous@local"));
        assert!(body_str.contains("Anonymous"));
    }

    #[tokio::test]
    async fn test_auth_config_still_requires_jwt() {
        // With auth_config present, requests without a token are rejected
        let app = test_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_no_header_returns_401() {
        let app = test_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_invalid_token_returns_401() {
        let app = test_app(Some(test_auth_config())).await;

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", "Bearer invalid.token.here")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        // The jsonwebtoken error (which check failed) must not reach the client.
        assert!(body.contains("Invalid or expired token"), "body: {body}");
        assert!(!body.to_lowercase().contains("base64") && !body.contains("InvalidToken"));
    }

    #[tokio::test]
    async fn test_expired_token_returns_401() {
        let app = test_app(Some(test_auth_config())).await;

        // Craft an expired token
        let now = chrono::Utc::now().timestamp();
        let claims = Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "test@ffs.holdings".to_string(),
            name: "Test".to_string(),
            iat: now - 7200,
            exp: now - 3600,
            token_type: None,
            scope: None,
            jti: None,
        };
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::default(),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(TEST_SECRET.as_bytes()),
        )
        .unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_valid_token_passes() {
        let app = test_app(Some(test_auth_config())).await;

        let user_id = uuid::Uuid::new_v4();
        let token = encode_jwt(user_id, "alice@ffs.holdings", "Alice", TEST_SECRET, 3600).unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_wrong_domain_returns_403() {
        let mut config = test_auth_config();
        config.allowed_email_domain = Some("ffs.holdings".to_string());

        let app = test_app(Some(config)).await;

        let user_id = uuid::Uuid::new_v4();
        let token = encode_jwt(user_id, "alice@gmail.com", "Alice", TEST_SECRET, 3600).unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_anonymous_user_bypasses_email_policy() {
        // MCP system user (ANONYMOUS_USER_ID) should bypass email restrictions
        // even when allowed_email_domain is set.
        let mut config = test_auth_config();
        config.allowed_email_domain = Some("ffs.holdings".to_string());

        let app = test_app(Some(config)).await;

        let token = encode_jwt(
            crate::auth::jwt::ANONYMOUS_USER_ID,
            "mcp-server@local",
            "MCP Server",
            TEST_SECRET,
            3600,
        )
        .unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_correct_domain_passes() {
        let mut config = test_auth_config();
        config.allowed_email_domain = Some("ffs.holdings".to_string());

        let app = test_app(Some(config)).await;

        let user_id = uuid::Uuid::new_v4();
        let token = encode_jwt(user_id, "alice@ffs.holdings", "Alice", TEST_SECRET, 3600).unwrap();

        let req = HttpRequest::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── agent_session tokens: bound, revocable, kept off human-only routes ──

    fn agent_token(session_id: &str) -> (String, String) {
        let human = Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "alice@ffs.holdings".to_string(),
            name: "Alice".to_string(),
            iat: 0,
            exp: 0,
            token_type: None,
            scope: None,
            jti: None,
        };
        let binding = crate::auth::jwt::AgentSessionBinding {
            session_id: session_id.to_string(),
            ceiling: Some("default".to_string()),
            tool_profile: None,
        };
        let (token, jti) =
            crate::auth::jwt::generate_session_token(&human, Some(&binding), TEST_SECRET, 3600)
                .unwrap();
        crate::auth::agent_tokens::register(&jti, Some(session_id));
        (token, jti)
    }

    async fn status_of(app: Router, method: &str, uri: &str, token: &str) -> StatusCode {
        let req = HttpRequest::builder()
            .method(method)
            .uri(uri)
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        app.oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn agent_token_of_a_closed_session_is_refused() {
        let sid = uuid::Uuid::new_v4().to_string();
        let (token, _) = agent_token(&sid);
        let app = test_app(Some(test_auth_config())).await;
        assert_eq!(
            status_of(app.clone(), "GET", "/test", &token).await,
            StatusCode::OK,
            "a live session's token authenticates"
        );

        crate::auth::agent_tokens::revoke_session(&sid);
        assert_eq!(
            status_of(app, "GET", "/test", &token).await,
            StatusCode::UNAUTHORIZED,
            "closing the session must kill its token"
        );
    }

    #[tokio::test]
    async fn agent_token_never_registered_is_refused() {
        // Signed correctly, but minted outside the registry (or before a
        // restart): fail closed.
        let human = Claims::service_account("runner:test");
        let (token, _) =
            crate::auth::jwt::generate_session_token(&human, None, TEST_SECRET, 3600).unwrap();
        let app = test_app(Some(test_auth_config())).await;
        assert_eq!(
            status_of(app, "GET", "/test", &token).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn agent_token_cannot_mutate_human_only_routes() {
        let state = make_server_state(Some(test_auth_config())).await;
        async fn ok_handler() -> &'static str {
            "ok"
        }
        use axum::routing::{post, put};
        let app = Router::new()
            .route(
                "/api/chat/config/permissions",
                get(ok_handler).put(ok_handler),
            )
            .route("/api/chat/providers", get(ok_handler).post(ok_handler))
            .route("/api/chat/providers/{id}", put(ok_handler))
            .route("/api/projects/{slug}/llm-consent", put(ok_handler))
            .route("/api/notes", post(ok_handler))
            .layer(from_fn_with_state(state.clone(), require_auth))
            .with_state(state);

        let sid = uuid::Uuid::new_v4().to_string();
        let (agent, _) = agent_token(&sid);
        let human = encode_jwt(
            uuid::Uuid::new_v4(),
            "alice@ffs.holdings",
            "Alice",
            TEST_SECRET,
            3600,
        )
        .unwrap();

        for (method, uri) in [
            ("PUT", "/api/chat/config/permissions"),
            ("POST", "/api/chat/providers"),
            ("PUT", "/api/chat/providers/deepseek"),
            ("PUT", "/api/projects/demo/llm-consent"),
        ] {
            assert_eq!(
                status_of(app.clone(), method, uri, &agent).await,
                StatusCode::FORBIDDEN,
                "agent token must get 403 on {method} {uri}"
            );
            assert_eq!(
                status_of(app.clone(), method, uri, &human).await,
                StatusCode::OK,
                "a human keeps {method} {uri}"
            );
        }
        // Reads and ordinary work stay open to the agent.
        for (method, uri) in [
            ("GET", "/api/chat/config/permissions"),
            ("GET", "/api/chat/providers"),
            ("POST", "/api/notes"),
        ] {
            assert_eq!(
                status_of(app.clone(), method, uri, &agent).await,
                StatusCode::OK,
                "agent token keeps {method} {uri}"
            );
        }
        crate::auth::agent_tokens::revoke_session(&sid);
    }

    #[test]
    fn human_only_mutation_matches_whole_segments_only() {
        use axum::http::Method;
        assert!(is_human_only_mutation(&Method::POST, "/api/chat/providers"));
        assert!(is_human_only_mutation(
            &Method::DELETE,
            "/api/chat/providers/x"
        ));
        assert!(is_human_only_mutation(&Method::PATCH, "/api/chat/config"));
        assert!(!is_human_only_mutation(&Method::GET, "/api/chat/config"));
        assert!(
            !is_human_only_mutation(&Method::POST, "/api/chat/providers-export"),
            "a sibling route sharing the prefix text is not covered"
        );
        assert!(!is_human_only_mutation(&Method::POST, "/api/chat/sessions"));
        assert!(is_human_only_mutation(
            &Method::PUT,
            "/api/projects/p/llm-roles"
        ));
        assert!(
            is_human_only_mutation(&Method::POST, "/api/chat/sessions/abc/permissions/req-1"),
            "an agent must not answer permission prompts"
        );
        assert!(!is_human_only_mutation(
            &Method::POST,
            "/api/chat/sessions/abc/interrupt"
        ));
    }

    #[tokio::test]
    async fn agent_token_gets_no_websocket_ticket() {
        let state = make_server_state(Some(test_auth_config())).await;
        let sid = uuid::Uuid::new_v4().to_string();
        let (token, _) = agent_token(&sid);
        let claims = decode_jwt(&token, TEST_SECRET).unwrap();
        let cfg = test_auth_config();
        let refused = enforce_token_policy(&state, &cfg, &claims, WS_TICKET_PATH).await;
        assert!(matches!(refused, Err(AppError::Forbidden(_))));
        assert!(enforce_token_policy(&state, &cfg, &claims, "/api/notes")
            .await
            .is_ok());
        crate::auth::agent_tokens::revoke_session(&sid);
    }
}
