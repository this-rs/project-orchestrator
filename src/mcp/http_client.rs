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
        }
    }

    /// Create from environment variables.
    ///
    /// Returns `Some` if `PO_SERVER_URL` is set, `None` otherwise.
    /// This is the primary constructor used by `mcp_server` binary.
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("PO_SERVER_URL").ok()?;
        let auth_token = std::env::var("PO_AUTH_TOKEN").ok();

        if auth_token.is_none() {
            warn!("PO_AUTH_TOKEN not set — HTTP client will operate without authentication");
        }

        debug!(
            "McpHttpClient initialized: base_url={}, auth={}",
            base_url,
            auth_token.is_some()
        );

        Some(Self::new(base_url, auth_token))
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

    /// Inject Authorization: Bearer header if token is available.
    fn inject_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth_token {
            Some(token) => req.bearer_auth(token),
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
            let body_excerpt = if body.len() > 500 {
                format!("{}...", &body[..500])
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
}
