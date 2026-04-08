//! HTTP client for MCP → REST API proxy
//!
//! Used in HTTP mode where the MCP server delegates tool calls to the REST API
//! instead of directly using the Orchestrator. The auth token (JWT session token)
//! is read from the `PO_AUTH_TOKEN` env var and injected as `Authorization: Bearer`.

use anyhow::{anyhow, Context, Result};
use reqwest::{Client, Response, StatusCode};
use serde_json::Value;
use tracing::{debug, warn};

/// HTTP client that proxies MCP tool calls to the REST API.
///
/// Created when `PO_SERVER_URL` is set, indicating the MCP server should
/// operate in HTTP wrapper mode instead of direct Orchestrator access.
#[derive(Clone)]
pub struct McpHttpClient {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
    /// Session ID injected by ChatManager via `PO_SESSION_ID` env var.
    /// Sent as `X-Session-Id` header on all requests to enable server-side
    /// auto-linking of chat sessions to tasks/plans.
    session_id: Option<String>,
}

impl McpHttpClient {
    /// Create a new HTTP client for the REST API.
    ///
    /// - `base_url`: REST server URL (e.g. `http://127.0.0.1:8080`)
    /// - `auth_token`: JWT session token from `PO_AUTH_TOKEN` env var (optional for no-auth mode)
    pub fn new(base_url: String, auth_token: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");

        // Strip trailing slash for consistent URL joining
        let base_url = base_url.trim_end_matches('/').to_string();

        Self {
            client,
            base_url,
            auth_token,
            session_id: None,
        }
    }

    /// Create from environment variables.
    ///
    /// Returns `Some` if `PO_SERVER_URL` is set, `None` otherwise.
    /// This is the primary constructor used by `mcp_server` binary.
    ///
    /// Token resolution order:
    /// 1. `PO_AUTH_TOKEN` (explicit JWT, injected by ChatManager)
    /// 2. `PO_JWT_SECRET` (auto-generate JWT from shared secret)
    /// 3. No auth (open access / no-auth mode)
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("PO_SERVER_URL").ok()?;
        let auth_token = Self::resolve_auth_token();
        let session_id = std::env::var("PO_SESSION_ID")
            .ok()
            .filter(|s| !s.is_empty());

        if auth_token.is_none() {
            warn!(
                "No auth token available — HTTP client will operate without authentication. \
                 Set PO_AUTH_TOKEN or PO_JWT_SECRET."
            );
        }

        debug!(
            "McpHttpClient initialized: base_url={}, auth={}, session_id={:?}",
            base_url,
            auth_token.is_some(),
            session_id.as_deref()
        );

        let mut client = Self::new(base_url, auth_token);
        client.session_id = session_id;
        Some(client)
    }

    /// Resolve auth token from environment.
    ///
    /// Priority: `PO_AUTH_TOKEN` (explicit) > `PO_JWT_SECRET` (auto-generate).
    fn resolve_auth_token() -> Option<String> {
        // 1. Explicit token (from ChatManager injection or manual config)
        if let Ok(token) = std::env::var("PO_AUTH_TOKEN") {
            if !token.is_empty() {
                debug!("Using explicit PO_AUTH_TOKEN");
                return Some(token);
            }
        }

        // 2. Auto-generate from jwt_secret (for standalone MCP usage via setup_claude)
        if let Ok(secret) = std::env::var("PO_JWT_SECRET") {
            if !secret.is_empty() {
                match crate::auth::jwt::encode_jwt(
                    crate::auth::jwt::ANONYMOUS_USER_ID,
                    "mcp-server@local",
                    "MCP Server",
                    &secret,
                    7 * 86400, // 7 days
                ) {
                    Ok(token) => {
                        debug!("Auto-generated auth token from PO_JWT_SECRET (7-day expiry)");
                        return Some(token);
                    }
                    Err(e) => {
                        warn!("Failed to generate token from PO_JWT_SECRET: {}", e);
                    }
                }
            }
        }

        None
    }

    /// GET request returning JSON.
    pub async fn get(&self, path: &str) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        debug!("GET {}", url);

        let mut req = self.client.get(&url);
        req = self.inject_auth(req);

        let resp = req.send().await.context("HTTP GET failed")?;
        self.handle_response(resp, "GET", &url).await
    }

    /// GET request with query parameters.
    pub async fn get_with_query(&self, path: &str, query: &[(String, String)]) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        debug!("GET {} ?{:?}", url, query);

        let mut req = self.client.get(&url).query(query);
        req = self.inject_auth(req);

        let resp = req.send().await.context("HTTP GET failed")?;
        self.handle_response(resp, "GET", &url).await
    }

    /// POST request with JSON body.
    pub async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        debug!("POST {}", url);

        let mut req = self.client.post(&url).json(body);
        req = self.inject_auth(req);

        let resp = req.send().await.context("HTTP POST failed")?;
        self.handle_response(resp, "POST", &url).await
    }

    /// PUT request with JSON body.
    pub async fn put(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        debug!("PUT {}", url);

        let mut req = self.client.put(&url).json(body);
        req = self.inject_auth(req);

        let resp = req.send().await.context("HTTP PUT failed")?;
        self.handle_response(resp, "PUT", &url).await
    }

    /// PATCH request with JSON body.
    pub async fn patch(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        debug!("PATCH {}", url);

        let mut req = self.client.patch(&url).json(body);
        req = self.inject_auth(req);

        let resp = req.send().await.context("HTTP PATCH failed")?;
        self.handle_response(resp, "PATCH", &url).await
    }

    /// DELETE request.
    pub async fn delete(&self, path: &str) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        debug!("DELETE {}", url);

        let mut req = self.client.delete(&url);
        req = self.inject_auth(req);

        let resp = req.send().await.context("HTTP DELETE failed")?;
        self.handle_response(resp, "DELETE", &url).await
    }

    /// DELETE request with JSON body.
    pub async fn delete_with_body(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}", self.base_url, path);
        debug!("DELETE {} (with body)", url);

        let mut req = self.client.delete(&url).json(body);
        req = self.inject_auth(req);

        let resp = req.send().await.context("HTTP DELETE failed")?;
        self.handle_response(resp, "DELETE", &url).await
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Inject Authorization: Bearer header and X-Session-Id header if available.
    fn inject_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let req = match &self.auth_token {
            Some(token) => req.bearer_auth(token),
            None => req,
        };
        match &self.session_id {
            Some(sid) => req.header("X-Session-Id", sid),
            None => req,
        }
    }

    /// Process HTTP response: check status, parse JSON, provide context on errors.
    async fn handle_response(&self, resp: Response, method: &str, url: &str) -> Result<Value> {
        let status = resp.status();

        if status.is_success() {
            // 204 No Content → return null
            if status == StatusCode::NO_CONTENT {
                return Ok(Value::Null);
            }

            let body = resp.text().await.context("Failed to read response body")?;

            // Empty body → return a generic success
            if body.is_empty() {
                return Ok(serde_json::json!({ "success": true }));
            }

            serde_json::from_str(&body).context("Failed to parse JSON response")
        } else {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());

            // Truncate body for error messages (avoid flooding logs with huge HTML errors)
            let body_excerpt = if body.chars().count() > 500 {
                let truncated: String = body.chars().take(500).collect();
                format!("{truncated}...")
            } else {
                body
            };

            Err(anyhow!(
                "{} {} returned {} — {}",
                method,
                url,
                status.as_u16(),
                body_excerpt
            ))
        }
    }
}

// ── Arg extraction helpers ──────────────────────────────────────────────────

/// Extract a required string field from tool arguments.
pub fn extract_string(args: &Value, field: &str) -> Result<String> {
    args.get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("{} is required", field))
}

/// Extract an optional string field from tool arguments.
pub fn extract_optional_string(args: &Value, field: &str) -> Option<String> {
    args.get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract a required UUID string field from tool arguments.
pub fn extract_id(args: &Value, field: &str) -> Result<String> {
    let value = extract_string(args, field)?;
    // Validate UUID format
    uuid::Uuid::parse_str(&value).context(format!("{} must be a valid UUID", field))?;
    Ok(value)
}

/// Extract an optional integer field from tool arguments.
pub fn extract_optional_i64(args: &Value, field: &str) -> Option<i64> {
    args.get(field).and_then(|v| v.as_i64())
}

/// Extract an optional boolean field from tool arguments.
pub fn extract_optional_bool(args: &Value, field: &str) -> Option<bool> {
    args.get(field).and_then(|v| v.as_bool())
}

/// Extract an optional array of strings from tool arguments.
pub fn extract_string_array(args: &Value, field: &str) -> Option<Vec<String>> {
    args.get(field).and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_string() {
        let args = json!({"name": "test", "id": "123"});
        assert_eq!(extract_string(&args, "name").unwrap(), "test");
        assert!(extract_string(&args, "missing").is_err());
    }

    #[test]
    fn test_extract_optional_string() {
        let args = json!({"name": "test"});
        assert_eq!(
            extract_optional_string(&args, "name"),
            Some("test".to_string())
        );
        assert_eq!(extract_optional_string(&args, "missing"), None);
    }

    #[test]
    fn test_extract_id() {
        let args = json!({"task_id": "4ee35887-fe28-4536-9c55-411c3559dbb6"});
        assert!(extract_id(&args, "task_id").is_ok());

        let bad = json!({"task_id": "not-a-uuid"});
        assert!(extract_id(&bad, "task_id").is_err());
    }

    #[test]
    fn test_extract_optional_i64() {
        let args = json!({"limit": 50, "name": "test"});
        assert_eq!(extract_optional_i64(&args, "limit"), Some(50));
        assert_eq!(extract_optional_i64(&args, "name"), None);
        assert_eq!(extract_optional_i64(&args, "missing"), None);
    }

    #[test]
    fn test_extract_string_array() {
        let args = json!({"tags": ["a", "b", "c"]});
        assert_eq!(
            extract_string_array(&args, "tags"),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
        assert_eq!(extract_string_array(&args, "missing"), None);
    }

    #[test]
    fn test_new_strips_trailing_slash() {
        let client = McpHttpClient::new("http://localhost:8080/".to_string(), None);
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_new_no_trailing_slash() {
        let client = McpHttpClient::new("http://localhost:8080".to_string(), None);
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    // ── extract_optional_bool ────────────────────────────────────────────

    #[test]
    fn test_extract_optional_bool_true() {
        let args = json!({"flag": true});
        assert_eq!(extract_optional_bool(&args, "flag"), Some(true));
    }

    #[test]
    fn test_extract_optional_bool_false() {
        let args = json!({"flag": false});
        assert_eq!(extract_optional_bool(&args, "flag"), Some(false));
    }

    #[test]
    fn test_extract_optional_bool_missing() {
        let args = json!({"other": 42});
        assert_eq!(extract_optional_bool(&args, "flag"), None);
    }

    #[test]
    fn test_extract_optional_bool_wrong_type() {
        let args = json!({"flag": "yes"});
        assert_eq!(extract_optional_bool(&args, "flag"), None);
    }

    // ── handle_response (mock axum server) ───────────────────────────────

    use axum::{
        body::Body, extract::Request, http::StatusCode as AxumStatusCode, routing::get, Router,
    };
    use tokio::net::TcpListener;

    /// Helper: spin up a one-route axum server and return an McpHttpClient pointed at it.
    async fn make_test_server(handler: axum::routing::MethodRouter) -> McpHttpClient {
        let app = Router::new().route("/test", handler);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        McpHttpClient::new(base_url, Some("test-token".to_string()))
    }

    #[tokio::test]
    async fn test_handle_response_200_valid_json() {
        let client = make_test_server(get(|| async {
            (
                AxumStatusCode::OK,
                axum::Json(json!({"id": 1, "name": "ok"})),
            )
        }))
        .await;

        let result = client.get("/test").await.unwrap();
        assert_eq!(result["id"], 1);
        assert_eq!(result["name"], "ok");
    }

    #[tokio::test]
    async fn test_handle_response_204_no_content() {
        let client = make_test_server(get(|| async { (AxumStatusCode::NO_CONTENT, "") })).await;

        let result = client.get("/test").await.unwrap();
        assert_eq!(result, Value::Null);
    }

    #[tokio::test]
    async fn test_handle_response_200_empty_body() {
        let client = make_test_server(get(|| async { (AxumStatusCode::OK, "") })).await;

        let result = client.get("/test").await.unwrap();
        assert_eq!(result, json!({"success": true}));
    }

    #[tokio::test]
    async fn test_handle_response_400_error() {
        let client = make_test_server(get(|| async {
            (AxumStatusCode::BAD_REQUEST, "missing required field")
        }))
        .await;

        let err = client.get("/test").await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("400"), "expected status 400 in: {msg}");
        assert!(
            msg.contains("missing required field"),
            "expected body excerpt in: {msg}"
        );
    }

    #[tokio::test]
    async fn test_handle_response_200_invalid_json() {
        let client = make_test_server(get(|| async { (AxumStatusCode::OK, "not json {{{") })).await;

        let err = client.get("/test").await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Failed to parse JSON response"),
            "expected JSON parse error in: {msg}"
        );
    }

    // ── inject_auth ──────────────────────────────────────────────────────

    /// Verify Bearer token is sent when auth_token is Some.
    #[tokio::test]
    async fn test_inject_auth_sends_bearer() {
        let app = Router::new().route(
            "/auth-check",
            get(|req: Request<Body>| async move {
                let auth = req
                    .headers()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                (AxumStatusCode::OK, axum::Json(json!({"auth": auth})))
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let client = McpHttpClient::new(base_url, Some("my-secret-jwt".to_string()));
        let result = client.get("/auth-check").await.unwrap();
        assert_eq!(result["auth"], "Bearer my-secret-jwt");
    }

    /// Verify no Authorization header is sent when auth_token is None.
    #[tokio::test]
    async fn test_inject_auth_none_sends_no_header() {
        let app = Router::new().route(
            "/auth-check",
            get(|req: Request<Body>| async move {
                let has_auth = req.headers().contains_key("authorization");
                (
                    AxumStatusCode::OK,
                    axum::Json(json!({"has_auth": has_auth})),
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let client = McpHttpClient::new(base_url, None);
        let result = client.get("/auth-check").await.unwrap();
        assert_eq!(result["has_auth"], false);
    }

    // ── resolve_auth_token ─────────────────────────────────────────────
    // Combined into a single test to avoid env var race conditions
    // (Rust runs tests in parallel by default, and env vars are global).

    #[test]
    fn test_resolve_auth_token_lifecycle() {
        // Phase 1: explicit PO_AUTH_TOKEN takes priority over PO_JWT_SECRET
        std::env::set_var("PO_AUTH_TOKEN", "explicit-token-123");
        std::env::set_var("PO_JWT_SECRET", "some-secret-key-minimum-32-chars!!");
        let token = McpHttpClient::resolve_auth_token();
        assert_eq!(token, Some("explicit-token-123".to_string()));

        // Phase 2: when PO_AUTH_TOKEN is absent, auto-generate from PO_JWT_SECRET
        std::env::remove_var("PO_AUTH_TOKEN");
        let token = McpHttpClient::resolve_auth_token();
        assert!(token.is_some(), "should auto-generate token from secret");
        let t = token.unwrap();
        assert_eq!(t.split('.').count(), 3, "should be a valid JWT format");

        // Phase 3: no env vars → None
        std::env::remove_var("PO_JWT_SECRET");
        let token = McpHttpClient::resolve_auth_token();
        assert!(token.is_none(), "no env vars should return None");

        // Phase 4: empty values should be treated as absent
        std::env::set_var("PO_AUTH_TOKEN", "");
        std::env::set_var("PO_JWT_SECRET", "");
        let token = McpHttpClient::resolve_auth_token();
        assert!(
            token.is_none(),
            "empty env vars should be treated as absent"
        );

        // Cleanup
        std::env::remove_var("PO_AUTH_TOKEN");
        std::env::remove_var("PO_JWT_SECRET");
    }

    // ── session_id / X-Session-Id ───────────────────────────────────────

    #[test]
    fn test_new_has_no_session_id() {
        let client = McpHttpClient::new("http://localhost:8080".to_string(), None);
        assert!(client.session_id.is_none());
    }

    #[test]
    fn test_session_id_can_be_set() {
        let mut client = McpHttpClient::new("http://localhost:8080".to_string(), None);
        client.session_id = Some("test-session-123".to_string());
        assert_eq!(client.session_id.as_deref(), Some("test-session-123"));
    }

    /// Verify X-Session-Id header is sent when session_id is set.
    #[tokio::test]
    async fn test_inject_session_id_sends_header() {
        let app = Router::new().route(
            "/session-check",
            get(|req: Request<Body>| async move {
                let session_id = req
                    .headers()
                    .get("x-session-id")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                (
                    AxumStatusCode::OK,
                    axum::Json(json!({"session_id": session_id})),
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let mut client = McpHttpClient::new(base_url, Some("jwt-token".to_string()));
        client.session_id = Some("abc-def-123".to_string());

        let result = client.get("/session-check").await.unwrap();
        assert_eq!(result["session_id"], "abc-def-123");
    }

    /// Verify no X-Session-Id header is sent when session_id is None.
    #[tokio::test]
    async fn test_no_session_id_sends_no_header() {
        let app = Router::new().route(
            "/session-check",
            get(|req: Request<Body>| async move {
                let has_header = req.headers().contains_key("x-session-id");
                (
                    AxumStatusCode::OK,
                    axum::Json(json!({"has_session_id": has_header})),
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let client = McpHttpClient::new(base_url, None);
        let result = client.get("/session-check").await.unwrap();
        assert_eq!(result["has_session_id"], false);
    }

    /// Verify X-Session-Id is sent on POST requests too (not just GET).
    #[tokio::test]
    async fn test_session_id_on_post_request() {
        let app = Router::new().route(
            "/session-check",
            axum::routing::post(|req: Request<Body>| async move {
                let session_id = req
                    .headers()
                    .get("x-session-id")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                (
                    AxumStatusCode::OK,
                    axum::Json(json!({"session_id": session_id})),
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let mut client = McpHttpClient::new(base_url, Some("token".to_string()));
        client.session_id = Some("post-session-456".to_string());

        let result = client.post("/session-check", &json!({})).await.unwrap();
        assert_eq!(result["session_id"], "post-session-456");
    }

    /// Verify X-Session-Id is sent on PATCH requests (used by task updates).
    #[tokio::test]
    async fn test_session_id_on_patch_request() {
        let app = Router::new().route(
            "/session-check",
            axum::routing::patch(|req: Request<Body>| async move {
                let session_id = req
                    .headers()
                    .get("x-session-id")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                let auth = req
                    .headers()
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                (
                    AxumStatusCode::OK,
                    axum::Json(json!({"session_id": session_id, "auth": auth})),
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let mut client = McpHttpClient::new(base_url, Some("jwt-abc".to_string()));
        client.session_id = Some("patch-session-789".to_string());

        let result = client.patch("/session-check", &json!({})).await.unwrap();
        assert_eq!(result["session_id"], "patch-session-789");
        assert_eq!(result["auth"], "Bearer jwt-abc");
    }
}
