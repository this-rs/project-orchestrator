//! MCP Streamable HTTP server transport.
//!
//! Exposes the Project Orchestrator MCP server over HTTP so remote Claude
//! clients (Claude Code CLI, Claude.ai connectors) can call the 28 mega-tools
//! without spawning the stdio binary. Mounted on the existing Axum server
//! under `require_auth` (see `src/api/routes.rs`), so every request carries a
//! validated bearer token.
//!
//! ## Transport shape (MCP Streamable HTTP)
//! - `POST /mcp` — client→server JSON-RPC 2.0 messages. Responses are plain
//!   `application/json` (the spec allows servers to answer JSON directly
//!   instead of opening an SSE stream per request).
//! - `GET /mcp` — would open a server→client SSE stream; we do not push
//!   server-initiated messages yet, so we return `405 Method Not Allowed`
//!   (explicitly permitted by the spec).
//! - `DELETE /mcp` — terminates the session identified by `Mcp-Session-Id`.
//!
//! ## Sessions
//! `initialize` creates a session and returns its id in the `Mcp-Session-Id`
//! response header. Every subsequent request must echo that header; unknown
//! or expired ids get `404` so the client knows to re-initialize.
//!
//! ## Dispatch
//! Tool calls go through the same transport-agnostic [`ToolHandler`] as the
//! stdio binary. The handler is constructed per request with a loopback
//! [`McpHttpClient`] carrying the *caller's* bearer token, so REST-side
//! authentication, authorization and audit apply unchanged per user.

use super::formatter::json_to_compact;
use super::handlers::ToolHandler;
use super::http_client::McpHttpClient;
use super::protocol::*;
use super::tools::all_tools;
use crate::api::handlers::OrchestratorState;
use axum::{
    extract::State,
    http::{header::AUTHORIZATION, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Protocol versions this server can speak, newest first. The negotiated
/// version is the client's requested version when we support it, otherwise
/// our newest — per the MCP version-negotiation rules.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

const SERVER_NAME: &str = "project-orchestrator";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Header carrying the MCP session id (spec-defined name).
pub const MCP_SESSION_HEADER: &str = "Mcp-Session-Id";

/// Idle sessions older than this are expired lazily on access.
const SESSION_IDLE_TTL: Duration = Duration::from_secs(2 * 60 * 60); // 2h

// ============================================================================
// Session registry
// ============================================================================

struct McpSession {
    /// Email of the authenticated user who opened the session — a session
    /// must only be usable by the identity that created it.
    user_email: String,
    /// Negotiated protocol version (kept for observability/debugging).
    #[allow(dead_code)]
    protocol_version: String,
    last_seen: Instant,
}

/// Process-wide session registry keyed by `Mcp-Session-Id`.
///
/// Deliberately a static (not a `ServerState` field): `ServerState` is
/// constructed at ~70 call sites and this registry has no per-instance
/// configuration. Tests exercise it through the public handlers.
fn sessions() -> &'static RwLock<HashMap<String, McpSession>> {
    static SESSIONS: OnceLock<RwLock<HashMap<String, McpSession>>> = OnceLock::new();
    SESSIONS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Insert a new session and return its id.
fn create_session(user_email: &str, protocol_version: &str) -> String {
    let id = Uuid::new_v4().to_string();
    let mut map = sessions().write().expect("mcp session lock poisoned");
    // Opportunistic sweep: drop idle sessions so the map cannot grow unbounded.
    map.retain(|_, s| s.last_seen.elapsed() < SESSION_IDLE_TTL);
    map.insert(
        id.clone(),
        McpSession {
            user_email: user_email.to_string(),
            protocol_version: protocol_version.to_string(),
            last_seen: Instant::now(),
        },
    );
    id
}

/// Validate a session id for the given user; refreshes `last_seen` on hit.
fn touch_session(id: &str, user_email: &str) -> SessionCheck {
    let mut map = sessions().write().expect("mcp session lock poisoned");
    match map.get_mut(id) {
        Some(s) if s.last_seen.elapsed() >= SESSION_IDLE_TTL => {
            map.remove(id);
            SessionCheck::Expired
        }
        Some(s) if s.user_email != user_email => SessionCheck::WrongUser,
        Some(s) => {
            s.last_seen = Instant::now();
            SessionCheck::Valid
        }
        None => SessionCheck::Unknown,
    }
}

fn remove_session(id: &str) -> bool {
    sessions()
        .write()
        .expect("mcp session lock poisoned")
        .remove(id)
        .is_some()
}

enum SessionCheck {
    Valid,
    Unknown,
    Expired,
    WrongUser,
}

// ============================================================================
// Origin validation (DNS-rebinding protection — MCP transport security spec)
// ============================================================================

/// Browser-originated requests must come from an allowlisted origin
/// (`allowed_origins()`: localhost:{port}, public_url, frontend_url,
/// additional_origins). Non-browser clients (Claude Code CLI, claude.ai's
/// server-side fetchers) send no `Origin` header and pass through — the
/// bearer token remains their auth gate. This closes the DNS-rebinding
/// vector where a malicious page scripts requests at a local MCP server.
fn origin_allowed(state: &OrchestratorState, headers: &HeaderMap) -> bool {
    match headers
        .get(axum::http::header::ORIGIN)
        .and_then(|v| v.to_str().ok())
    {
        None => true, // non-browser client
        Some(origin) => {
            let origin = origin.trim_end_matches('/');
            state
                .allowed_origins()
                .iter()
                .any(|allowed| allowed.trim_end_matches('/') == origin)
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// `POST /mcp` — handle one client→server JSON-RPC message.
pub async fn mcp_post(
    State(state): State<OrchestratorState>,
    auth_user: crate::auth::extractor::AuthUser,
    headers: HeaderMap,
    body: String,
) -> Response {
    if !origin_allowed(&state, &headers) {
        warn!("MCP HTTP request rejected: non-allowlisted Origin");
        return StatusCode::FORBIDDEN.into_response();
    }

    // Parse the JSON-RPC envelope. Batches are not supported (Claude clients
    // send single messages); reject arrays explicitly rather than silently
    // processing only the first entry.
    let parsed: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return jsonrpc_http_response(
                StatusCode::OK,
                None,
                JsonRpcResponse::error(Value::Null, JsonRpcError::parse_error(e.to_string())),
            );
        }
    };
    if parsed.is_array() {
        return jsonrpc_http_response(
            StatusCode::OK,
            None,
            JsonRpcResponse::error(
                Value::Null,
                JsonRpcError::invalid_request("batch requests are not supported"),
            ),
        );
    }
    let request: JsonRpcRequest = match serde_json::from_value(parsed) {
        Ok(r) => r,
        Err(e) => {
            return jsonrpc_http_response(
                StatusCode::OK,
                None,
                JsonRpcResponse::error(Value::Null, JsonRpcError::parse_error(e.to_string())),
            );
        }
    };

    // Notifications (no id) are acknowledged with 202 and no body.
    let Some(id) = request.id.clone() else {
        debug!(method = %request.method, "MCP HTTP notification");
        return StatusCode::ACCEPTED.into_response();
    };

    // `initialize` is the only method allowed without a session header.
    if request.method == "initialize" {
        return handle_initialize(&auth_user, id, &request);
    }

    // Everything else requires a valid session bound to this user.
    let session_id = match headers
        .get(MCP_SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
    {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => {
            return jsonrpc_http_response(
                StatusCode::BAD_REQUEST,
                None,
                JsonRpcResponse::error(
                    id,
                    JsonRpcError::invalid_request("missing Mcp-Session-Id header"),
                ),
            );
        }
    };
    match touch_session(&session_id, &auth_user.email) {
        SessionCheck::Valid => {}
        SessionCheck::WrongUser => {
            warn!(session = %session_id, user = %auth_user.email, "MCP session user mismatch");
            return StatusCode::FORBIDDEN.into_response();
        }
        SessionCheck::Unknown | SessionCheck::Expired => {
            // 404 tells a spec-compliant client to re-initialize.
            return StatusCode::NOT_FOUND.into_response();
        }
    }

    let result = match request.method.as_str() {
        "ping" => Ok(json!({})),
        "tools/list" => serde_json::to_value(ToolsListResult { tools: all_tools() })
            .map_err(|e| JsonRpcError::internal_error(e.to_string())),
        "tools/call" => handle_tools_call(&state, &headers, &request).await,
        other => Err(JsonRpcError::method_not_found(other)),
    };

    let response = match result {
        Ok(value) => JsonRpcResponse::success(id, value),
        Err(error) => JsonRpcResponse::error(id, error),
    };
    jsonrpc_http_response(StatusCode::OK, Some(&session_id), response)
}

/// `GET /mcp` — server→client SSE stream. Not offered yet: the spec allows
/// returning 405 when the server has no server-initiated messages to push.
pub async fn mcp_get() -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        [(axum::http::header::ALLOW, "POST, DELETE")],
    )
        .into_response()
}

/// `DELETE /mcp` — explicit session termination.
pub async fn mcp_delete(
    State(state): State<OrchestratorState>,
    auth_user: crate::auth::extractor::AuthUser,
    headers: HeaderMap,
) -> Response {
    if !origin_allowed(&state, &headers) {
        warn!("MCP HTTP DELETE rejected: non-allowlisted Origin");
        return StatusCode::FORBIDDEN.into_response();
    }
    let Some(session_id) = headers
        .get(MCP_SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
    else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    // Only the owning user may close a session.
    match touch_session(session_id, &auth_user.email) {
        SessionCheck::WrongUser => return StatusCode::FORBIDDEN.into_response(),
        SessionCheck::Unknown | SessionCheck::Expired => {
            return StatusCode::NOT_FOUND.into_response()
        }
        SessionCheck::Valid => {}
    }
    remove_session(session_id);
    info!(session = %session_id, "MCP HTTP session terminated");
    StatusCode::NO_CONTENT.into_response()
}

// ============================================================================
// Method implementations
// ============================================================================

fn handle_initialize(
    auth_user: &crate::auth::extractor::AuthUser,
    id: Value,
    request: &JsonRpcRequest,
) -> Response {
    let params: InitializeParams = match request
        .params
        .as_ref()
        .map(|p| serde_json::from_value(p.clone()))
        .transpose()
    {
        Ok(p) => p.unwrap_or(InitializeParams {
            protocol_version: SUPPORTED_PROTOCOL_VERSIONS[0].to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: None,
        }),
        Err(e) => {
            return jsonrpc_http_response(
                StatusCode::OK,
                None,
                JsonRpcResponse::error(id, JsonRpcError::invalid_params(e.to_string())),
            );
        }
    };

    // Version negotiation: echo the client's version when supported,
    // otherwise offer our newest and let the client decide.
    let negotiated = if SUPPORTED_PROTOCOL_VERSIONS.contains(&params.protocol_version.as_str()) {
        params.protocol_version.clone()
    } else {
        SUPPORTED_PROTOCOL_VERSIONS[0].to_string()
    };

    let session_id = create_session(&auth_user.email, &negotiated);
    info!(
        session = %session_id,
        user = %auth_user.email,
        client_version = %params.protocol_version,
        negotiated = %negotiated,
        client = %params
            .client_info
            .as_ref()
            .map(|c| c.name.as_str())
            .unwrap_or("unknown"),
        "MCP HTTP session initialized"
    );

    let result = InitializeResult {
        protocol_version: negotiated,
        capabilities: ServerCapabilities {
            tools: ToolsCapability {
                list_changed: false,
            },
        },
        server_info: ServerInfo {
            name: SERVER_NAME.to_string(),
            version: SERVER_VERSION.to_string(),
        },
    };
    match serde_json::to_value(result) {
        Ok(v) => jsonrpc_http_response(
            StatusCode::OK,
            Some(&session_id),
            JsonRpcResponse::success(id, v),
        ),
        Err(e) => jsonrpc_http_response(
            StatusCode::OK,
            Some(&session_id),
            JsonRpcResponse::error(id, JsonRpcError::internal_error(e.to_string())),
        ),
    }
}

async fn handle_tools_call(
    state: &OrchestratorState,
    headers: &HeaderMap,
    request: &JsonRpcRequest,
) -> Result<Value, JsonRpcError> {
    let params: ToolCallParams = request
        .params
        .as_ref()
        .ok_or_else(|| JsonRpcError::invalid_params("params required"))?
        .clone()
        .pipe_into()
        .map_err(|e: serde_json::Error| JsonRpcError::invalid_params(e.to_string()))?;

    info!(tool = %params.name, "MCP HTTP tool call");

    // Loopback dispatch carrying the CALLER'S bearer token so REST-side
    // auth/authorization/audit apply per user. This is the interim
    // architecture recorded in the plan decision (loopback → later direct
    // dispatch); the hop is localhost-only.
    let token = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string());
    let base_url = format!("http://127.0.0.1:{}", state.server_port);
    let handler = ToolHandler::new(McpHttpClient::new(base_url, token));

    let result = handler.handle(&params.name, params.arguments).await;
    let tool_result = match result {
        Ok(value) => ToolCallResult::success(json_to_compact(&value)),
        Err(e) => {
            warn!(tool = %params.name, error = %e, "MCP HTTP tool error");
            ToolCallResult::error(e.to_string())
        }
    };
    serde_json::to_value(tool_result).map_err(|e| JsonRpcError::internal_error(e.to_string()))
}

// ============================================================================
// Helpers
// ============================================================================

/// Build the HTTP response for a JSON-RPC message, attaching the session
/// header when present.
fn jsonrpc_http_response(
    status: StatusCode,
    session_id: Option<&str>,
    body: JsonRpcResponse,
) -> Response {
    let mut response = (status, Json(body)).into_response();
    if let Some(sid) = session_id {
        if let Ok(value) = sid.parse() {
            response.headers_mut().insert(MCP_SESSION_HEADER, value);
        }
    }
    response
}

/// Tiny helper so `serde_json::from_value` reads left-to-right above.
trait PipeInto: Sized {
    fn pipe_into<T: serde::de::DeserializeOwned>(self) -> Result<T, serde_json::Error>;
}
impl PipeInto for Value {
    fn pipe_into<T: serde::de::DeserializeOwned>(self) -> Result<T, serde_json::Error> {
        serde_json::from_value(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_email() -> String {
        format!("{}@test.local", Uuid::new_v4())
    }

    #[test]
    fn test_create_and_touch_session() {
        let email = unique_email();
        let id = create_session(&email, "2025-06-18");
        assert!(matches!(touch_session(&id, &email), SessionCheck::Valid));
    }

    #[test]
    fn test_touch_unknown_session() {
        let email = unique_email();
        assert!(matches!(
            touch_session("nonexistent-session", &email),
            SessionCheck::Unknown
        ));
    }

    #[test]
    fn test_session_user_isolation() {
        let owner = unique_email();
        let attacker = unique_email();
        let id = create_session(&owner, "2025-06-18");
        assert!(matches!(
            touch_session(&id, &attacker),
            SessionCheck::WrongUser
        ));
        // Owner still valid — a mismatch must not destroy the session.
        assert!(matches!(touch_session(&id, &owner), SessionCheck::Valid));
    }

    #[test]
    fn test_remove_session() {
        let email = unique_email();
        let id = create_session(&email, "2025-03-26");
        assert!(remove_session(&id));
        assert!(!remove_session(&id));
        assert!(matches!(touch_session(&id, &email), SessionCheck::Unknown));
    }

    #[test]
    fn test_version_negotiation_supported() {
        // All supported versions are echoed back as-is.
        for v in SUPPORTED_PROTOCOL_VERSIONS {
            assert!(SUPPORTED_PROTOCOL_VERSIONS.contains(v));
        }
    }

    #[test]
    fn test_supported_versions_newest_first() {
        assert_eq!(SUPPORTED_PROTOCOL_VERSIONS[0], "2025-06-18");
        assert!(SUPPORTED_PROTOCOL_VERSIONS.len() >= 2);
    }
}
