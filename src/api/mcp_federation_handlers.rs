//! REST API handlers for MCP Federation.
//!
//! Endpoints:
//! - GET    /api/mcp-federation/servers              — list connected servers
//! - POST   /api/mcp-federation/servers              — connect to a new server
//! - GET    /api/mcp-federation/servers/:id           — get server status
//! - DELETE /api/mcp-federation/servers/:id           — disconnect a server
//! - GET    /api/mcp-federation/servers/:id/tools     — list tools for a server
//! - POST   /api/mcp-federation/servers/:id/probe     — probe a server's tools
//! - POST   /api/mcp-federation/servers/:id/reconnect — reconnect a server

use super::handlers::{AppError, OrchestratorState};
use crate::mcp_federation::client::McpTransportConfig;
use crate::mcp_federation::McpTransport;
use crate::neo4j::models::McpServerNode;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ============================================================================
// Request / Response types
// ============================================================================

/// Body for POST /api/mcp-federation/servers (connect).
#[derive(Debug, Deserialize)]
pub struct ConnectServerBody {
    /// Unique identifier for this server connection (e.g. "grafeo-shared").
    pub server_id: String,
    /// Optional human-readable display name.
    pub display_name: Option<String>,
    /// Transport type: "stdio", "sse", or "streamable_http".
    pub transport: String,
    // -- Stdio fields --
    /// Command to spawn (required for stdio transport).
    pub command: Option<String>,
    /// Arguments for the command.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables for the spawned process.
    #[serde(default)]
    pub env: HashMap<String, String>,
    // -- HTTP fields (SSE / StreamableHTTP) --
    /// URL for SSE or StreamableHTTP transport.
    pub url: Option<String>,
    /// Additional HTTP headers.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// Lightweight response for disconnect / reconnect.
#[derive(Debug, Serialize)]
pub struct ActionResponse {
    pub success: bool,
    pub server_id: String,
    pub message: String,
}

// ============================================================================
// Helper
// ============================================================================

/// Extract the shared registry from state.
fn get_registry(state: &OrchestratorState) -> &crate::mcp_federation::registry::SharedRegistry {
    &state.mcp_registry
}

/// Build an `McpTransport` from the connect body.
pub(crate) fn build_transport(body: &ConnectServerBody) -> Result<McpTransport, AppError> {
    match body.transport.as_str() {
        "stdio" => {
            let command = body.command.clone().ok_or_else(|| {
                AppError::BadRequest("'command' is required for stdio transport".to_string())
            })?;
            Ok(McpTransport::Stdio {
                command,
                args: body.args.clone(),
                env: body.env.clone(),
            })
        }
        "sse" => {
            let url = body.url.clone().ok_or_else(|| {
                AppError::BadRequest("'url' is required for sse transport".to_string())
            })?;
            Ok(McpTransport::Sse {
                url,
                headers: body.headers.clone(),
            })
        }
        "streamable_http" => {
            let url = body.url.clone().ok_or_else(|| {
                AppError::BadRequest("'url' is required for streamable_http transport".to_string())
            })?;
            Ok(McpTransport::StreamableHttp {
                url,
                headers: body.headers.clone(),
            })
        }
        other => Err(AppError::BadRequest(format!(
            "Unknown transport '{}'. Expected: stdio, sse, streamable_http",
            other
        ))),
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// List all connected MCP servers.
///
/// GET /api/mcp-federation/servers
pub async fn list_servers(
    State(state): State<OrchestratorState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let registry = get_registry(&state);
    let reg = registry.read().await;
    let servers = reg.list();
    Ok(Json(serde_json::to_value(servers).map_err(|e| {
        AppError::Internal(anyhow::anyhow!("Serialization error: {}", e))
    })?))
}

/// Extract persistence fields from an `McpTransport`.
///
/// Returns `(transport_type, transport_url, transport_command, transport_args)`.
pub(crate) fn extract_transport_fields(
    transport: &McpTransport,
) -> (String, Option<String>, Option<String>, Option<String>) {
    match transport {
        McpTransport::Stdio { command, args, .. } => (
            "stdio".to_string(),
            None,
            Some(command.clone()),
            Some(serde_json::to_string(args).unwrap_or_default()),
        ),
        McpTransport::Sse { url, .. } => ("sse".to_string(), Some(url.clone()), None, None),
        McpTransport::StreamableHttp { url, .. } => {
            ("streamable_http".to_string(), Some(url.clone()), None, None)
        }
    }
}

/// Build an `McpServerNode` from a connect request + server summary.
pub(crate) fn build_server_node(
    server_id: &str,
    display_name: Option<&str>,
    transport: &McpTransport,
    server_name: Option<&str>,
    tool_count: usize,
) -> McpServerNode {
    let (transport_type, transport_url, transport_command, transport_args) =
        extract_transport_fields(transport);
    let now = Utc::now();
    McpServerNode {
        id: Uuid::new_v4(),
        project_id: Uuid::nil(), // cross-project; not tied to a specific project
        server_id: server_id.to_string(),
        display_name: display_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| server_id.to_string()),
        transport_type,
        transport_url,
        transport_command,
        transport_args,
        status: "connected".to_string(),
        protocol_version: server_name.map(|s| s.to_string()),
        server_name: server_name.map(|s| s.to_string()),
        tool_count,
        created_at: now,
        updated_at: Some(now),
        last_connected_at: Some(now),
    }
}

/// Connect to a new external MCP server.
///
/// POST /api/mcp-federation/servers
pub async fn connect_server(
    State(state): State<OrchestratorState>,
    Json(body): Json<ConnectServerBody>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    let registry = get_registry(&state);
    let transport = build_transport(&body)?;

    let config = McpTransportConfig {
        server_id: body.server_id.clone(),
        display_name: body.display_name.clone(),
        transport: transport.clone(),
    };

    let mut reg = registry.write().await;
    let summary = reg.connect(config).await.map_err(AppError::Internal)?;

    // Build node for Neo4j persistence
    let server_node = build_server_node(
        &body.server_id,
        body.display_name.as_deref(),
        &transport,
        summary.server_name.as_deref(),
        summary.tool_count,
    );

    // Fire-and-forget persistence — don't fail the connect if Neo4j write fails
    let neo4j = state.orchestrator.neo4j_arc();
    tokio::spawn(async move {
        if let Err(e) = neo4j.create_mcp_server(&server_node).await {
            tracing::warn!(
                server_id = %server_node.server_id,
                "Failed to persist MCP server to Neo4j: {}",
                e
            );
        } else {
            tracing::info!(
                server_id = %server_node.server_id,
                "MCP server persisted to Neo4j for restart recovery"
            );
        }
    });

    Ok((
        StatusCode::CREATED,
        Json(
            serde_json::to_value(summary)
                .map_err(|e| AppError::Internal(anyhow::anyhow!("Serialization error: {}", e)))?,
        ),
    ))
}

/// Get status of a specific connected server.
///
/// GET /api/mcp-federation/servers/:id
pub async fn get_server_status(
    State(state): State<OrchestratorState>,
    Path(server_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let registry = get_registry(&state);
    let reg = registry.read().await;

    // Find this server in the list
    let servers = reg.list();
    let server = servers
        .into_iter()
        .find(|s| s.id == server_id)
        .ok_or_else(|| AppError::NotFound(format!("MCP server '{}' not found", server_id)))?;

    Ok(Json(serde_json::to_value(server).map_err(|e| {
        AppError::Internal(anyhow::anyhow!("Serialization error: {}", e))
    })?))
}

/// Disconnect from an external MCP server.
///
/// DELETE /api/mcp-federation/servers/:id
pub async fn disconnect_server(
    State(state): State<OrchestratorState>,
    Path(server_id): Path<String>,
) -> Result<Json<ActionResponse>, AppError> {
    let registry = get_registry(&state);
    let mut reg = registry.write().await;
    reg.disconnect(&server_id)
        .await
        .map_err(AppError::Internal)?;

    // Remove from Neo4j (fire-and-forget)
    let neo4j = state.orchestrator.neo4j_arc();
    let sid = server_id.clone();
    tokio::spawn(async move {
        if let Err(e) = neo4j.delete_mcp_server_by_server_id(&sid).await {
            tracing::warn!(
                server_id = %sid,
                "Failed to remove MCP server from Neo4j: {}",
                e
            );
        }
    });

    Ok(Json(ActionResponse {
        success: true,
        server_id,
        message: "Server disconnected successfully".to_string(),
    }))
}

/// List tools discovered on a specific server.
///
/// GET /api/mcp-federation/servers/:id/tools
pub async fn list_server_tools(
    State(state): State<OrchestratorState>,
    Path(server_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let registry = get_registry(&state);
    let reg = registry.read().await;

    let tools = reg.tools_for_server(&server_id);
    if tools.is_empty() {
        // Check if server exists at all
        if reg.get(&server_id).is_none() {
            return Err(AppError::NotFound(format!(
                "MCP server '{}' not found",
                server_id
            )));
        }
    }

    Ok(Json(serde_json::to_value(tools).map_err(|e| {
        AppError::Internal(anyhow::anyhow!("Serialization error: {}", e))
    })?))
}

/// Probe a server's tools (run read-only smoke tests).
///
/// POST /api/mcp-federation/servers/:id/probe
pub async fn probe_server(
    State(state): State<OrchestratorState>,
    Path(server_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let registry = get_registry(&state);
    let mut reg = registry.write().await;

    let conn = reg
        .get_mut(&server_id)
        .ok_or_else(|| AppError::NotFound(format!("MCP server '{}' not found", server_id)))?;

    // Probe all safe (read-only) tools on this server
    let prober = crate::mcp_federation::prober::ToolProber::new(
        crate::mcp_federation::prober::ProberConfig::default(),
    );
    prober
        .probe_batch(conn.client.as_ref(), &mut conn.discovered_tools)
        .await;

    let probed_count = conn
        .discovered_tools
        .iter()
        .filter(|t| t.profile.is_some())
        .count();

    Ok(Json(serde_json::json!({
        "server_id": server_id,
        "probed": probed_count,
        "total": conn.discovered_tools.len(),
    })))
}

/// Reconnect to an external MCP server (disconnect + connect with same config).
///
/// POST /api/mcp-federation/servers/:id/reconnect
pub async fn reconnect_server(
    State(state): State<OrchestratorState>,
    Path(server_id): Path<String>,
) -> Result<Json<ActionResponse>, AppError> {
    let registry = get_registry(&state);

    // Get the existing transport config before disconnecting
    let transport_config = {
        let reg = registry.read().await;
        let conn = reg
            .get(&server_id)
            .ok_or_else(|| AppError::NotFound(format!("MCP server '{}' not found", server_id)))?;
        McpTransportConfig {
            server_id: conn.id.clone(),
            display_name: Some(conn.display_name.clone()),
            transport: conn.transport.clone(),
        }
    };

    // Disconnect
    {
        let mut reg = registry.write().await;
        reg.disconnect(&server_id)
            .await
            .map_err(AppError::Internal)?;
    }

    // Reconnect
    {
        let mut reg = registry.write().await;
        reg.connect(transport_config)
            .await
            .map_err(AppError::Internal)?;
    }

    Ok(Json(ActionResponse {
        success: true,
        server_id,
        message: "Server reconnected successfully".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::ServerState;
    use crate::api::routes::create_router;
    use crate::events::EventBus;
    use crate::orchestrator::{FileWatcher, Orchestrator};
    use crate::test_helpers::{mock_app_state, test_auth_config, test_bearer_token};
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use std::collections::HashMap;
    use std::sync::Arc;
    use tower::ServiceExt;

    /// App with empty mcp_registry (no servers connected)
    async fn test_app() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
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
        });
        create_router(state)
    }

    /// App WITH mcp_registry enabled (empty registry)
    async fn test_app_with_registry() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));
        let registry = crate::mcp_federation::registry::new_shared_registry();
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
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
            mcp_registry: registry,
        });
        create_router(state)
    }

    fn auth_get(uri: &str) -> Request<Body> {
        Request::builder()
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn auth_delete(uri: &str) -> Request<Body> {
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .body(Body::empty())
            .unwrap()
    }

    fn auth_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("authorization", test_bearer_token())
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    }

    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ========================================================================
    // 1. list_servers with default (empty) registry => 200 + empty array
    // ========================================================================

    #[tokio::test]
    async fn test_list_servers_no_registry() {
        let app = test_app().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.as_array().unwrap().is_empty());
    }

    // ========================================================================
    // 2. list_servers with empty registry => 200 + empty array
    // ========================================================================

    #[tokio::test]
    async fn test_list_servers_empty() {
        let app = test_app_with_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(json.is_array(), "Expected JSON array, got: {:?}", json);
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    // ========================================================================
    // 3. get_server_status for nonexistent server => 404
    // ========================================================================

    #[tokio::test]
    async fn test_get_server_not_found() {
        let app = test_app_with_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers/nonexistent"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ========================================================================
    // 4. disconnect nonexistent server => 500 (registry.disconnect returns Err)
    // ========================================================================

    #[tokio::test]
    async fn test_disconnect_server_not_found() {
        let app = test_app_with_registry().await;
        let resp = app
            .oneshot(auth_delete("/api/mcp-federation/servers/nonexistent"))
            .await
            .unwrap();
        // disconnect returns anyhow::Error mapped to AppError::Internal (500)
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ========================================================================
    // 5. list_tools for nonexistent server => 404
    // ========================================================================

    #[tokio::test]
    async fn test_list_tools_not_found() {
        let app = test_app_with_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers/nonexistent/tools"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ========================================================================
    // 6. connect with invalid transport type => 400
    // ========================================================================

    #[tokio::test]
    async fn test_connect_server_invalid_transport() {
        let app = test_app_with_registry().await;
        let body = serde_json::json!({
            "server_id": "test-srv",
            "transport": "carrier_pigeon"
        });
        let resp = app
            .oneshot(auth_post("/api/mcp-federation/servers", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        let err = json["error"].as_str().unwrap();
        assert!(err.contains("Unknown transport"), "Error was: {}", err);
    }

    // ========================================================================
    // 7. connect stdio without command => 400
    // ========================================================================

    #[tokio::test]
    async fn test_connect_server_missing_command() {
        let app = test_app_with_registry().await;
        let body = serde_json::json!({
            "server_id": "test-srv",
            "transport": "stdio"
        });
        let resp = app
            .oneshot(auth_post("/api/mcp-federation/servers", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        let err = json["error"].as_str().unwrap();
        assert!(err.contains("command"), "Error was: {}", err);
    }

    // ========================================================================
    // 8. connect sse without url => 400
    // ========================================================================

    #[tokio::test]
    async fn test_connect_server_missing_url() {
        let app = test_app_with_registry().await;
        let body = serde_json::json!({
            "server_id": "test-srv",
            "transport": "sse"
        });
        let resp = app
            .oneshot(auth_post("/api/mcp-federation/servers", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        let err = json["error"].as_str().unwrap();
        assert!(err.contains("url"), "Error was: {}", err);
    }

    // ========================================================================
    // 9. build_transport: valid stdio
    // ========================================================================

    #[test]
    fn test_build_transport_stdio() {
        let body = ConnectServerBody {
            server_id: "s1".into(),
            display_name: None,
            transport: "stdio".into(),
            command: Some("node".into()),
            args: vec!["server.js".into()],
            env: HashMap::from([("FOO".into(), "bar".into())]),
            url: None,
            headers: HashMap::new(),
        };
        let transport = build_transport(&body).unwrap();
        match transport {
            McpTransport::Stdio { command, args, env } => {
                assert_eq!(command, "node");
                assert_eq!(args, vec!["server.js"]);
                assert_eq!(env.get("FOO").unwrap(), "bar");
            }
            other => panic!("Expected Stdio, got {:?}", other),
        }
    }

    // ========================================================================
    // 10. build_transport: valid SSE
    // ========================================================================

    #[test]
    fn test_build_transport_sse() {
        let body = ConnectServerBody {
            server_id: "s2".into(),
            display_name: Some("My SSE".into()),
            transport: "sse".into(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: Some("http://localhost:8080/sse".into()),
            headers: HashMap::from([("Authorization".into(), "Bearer tok".into())]),
        };
        let transport = build_transport(&body).unwrap();
        match transport {
            McpTransport::Sse { url, headers } => {
                assert_eq!(url, "http://localhost:8080/sse");
                assert_eq!(headers.get("Authorization").unwrap(), "Bearer tok");
            }
            other => panic!("Expected Sse, got {:?}", other),
        }
    }

    // ========================================================================
    // 11. build_transport: valid streamable_http
    // ========================================================================

    #[test]
    fn test_build_transport_streamable_http() {
        let body = ConnectServerBody {
            server_id: "s3".into(),
            display_name: None,
            transport: "streamable_http".into(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: Some("http://localhost:9090/mcp".into()),
            headers: HashMap::new(),
        };
        let transport = build_transport(&body).unwrap();
        match transport {
            McpTransport::StreamableHttp { url, headers } => {
                assert_eq!(url, "http://localhost:9090/mcp");
                assert!(headers.is_empty());
            }
            other => panic!("Expected StreamableHttp, got {:?}", other),
        }
    }

    // ========================================================================
    // 12. build_transport: unknown transport => error
    // ========================================================================

    #[test]
    fn test_build_transport_unknown() {
        let body = ConnectServerBody {
            server_id: "s4".into(),
            display_name: None,
            transport: "websocket".into(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
        };
        let result = build_transport(&body);
        assert!(result.is_err());
        // Verify it's a BadRequest
        match result.unwrap_err() {
            AppError::BadRequest(msg) => {
                assert!(msg.contains("Unknown transport"), "Message was: {}", msg);
                assert!(msg.contains("websocket"), "Message was: {}", msg);
            }
            other => panic!("Expected BadRequest, got {:?}", other),
        }
    }

    // ========================================================================
    // Helper: app with a pre-populated registry (one server with 2 tools)
    // ========================================================================

    async fn test_app_with_populated_registry() -> axum::Router {
        use crate::mcp_federation::circuit_breaker::CircuitBreaker;
        use crate::mcp_federation::discovery::{DiscoveredTool, InferredCategory};
        use crate::mcp_federation::registry::{
            ConnectionStatus, McpServerConnection, McpServerRegistry, ServerStats,
        };

        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(
            orchestrator.clone(),
        )));

        // Build a registry with a mock server
        let mut registry = McpServerRegistry::new();

        // Mock client (minimal)
        #[derive(Debug)]
        struct MinimalMockClient;
        #[async_trait::async_trait]
        impl crate::mcp_federation::client::McpClient for MinimalMockClient {
            async fn initialize(
                &self,
            ) -> anyhow::Result<crate::mcp_federation::client::InitializeResult> {
                unimplemented!()
            }
            async fn initialized_notification(&self) -> anyhow::Result<()> {
                Ok(())
            }
            async fn tools_list(
                &self,
            ) -> anyhow::Result<Vec<crate::mcp_federation::client::McpToolDef>> {
                Ok(vec![])
            }
            async fn call_tool(
                &self,
                _: &str,
                _: Option<serde_json::Value>,
            ) -> anyhow::Result<serde_json::Value> {
                Ok(serde_json::json!({}))
            }
            async fn ping(&self) -> anyhow::Result<()> {
                Ok(())
            }
            async fn shutdown(&self) -> anyhow::Result<()> {
                Ok(())
            }
            fn transport_name(&self) -> &'static str {
                "mock"
            }
        }

        let tools = vec![
            DiscoveredTool {
                name: "list_items".to_string(),
                fqn: "test-srv::list_items".to_string(),
                description: "List all items".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
                category: InferredCategory::Query,
                embedding: None,
                similar_internal: vec![],
                profile: None,
            },
            DiscoveredTool {
                name: "get_item".to_string(),
                fqn: "test-srv::get_item".to_string(),
                description: "Get a single item".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {"id": {"type": "string"}}}),
                category: InferredCategory::Query,
                embedding: None,
                similar_internal: vec![],
                profile: None,
            },
        ];

        let conn = McpServerConnection {
            id: "test-srv".to_string(),
            display_name: "Test Server".to_string(),
            transport: crate::mcp_federation::McpTransport::Stdio {
                command: "echo".to_string(),
                args: vec![],
                env: HashMap::new(),
            },
            status: ConnectionStatus::Connected,
            client: Box::new(MinimalMockClient),
            discovered_tools: tools,
            circuit_breaker: CircuitBreaker::default(),
            stats: ServerStats::new(),
            connected_at: chrono::Utc::now(),
            server_protocol_version: Some("2024-11-05".to_string()),
            server_name: Some("MockMCP".to_string()),
        };

        registry.insert_connection_for_test(conn);

        let shared_registry = std::sync::Arc::new(tokio::sync::RwLock::new(registry));

        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(
                EventBus::default(),
            ))),
            nats_emitter: None,
            auth_config: Some(test_auth_config()),
            serve_frontend: false,
            frontend_path: "./dist".to_string(),
            setup_completed: true,
            server_port: 6600,
            public_url: None,
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
            mcp_registry: shared_registry,
        });
        create_router(state)
    }

    // ========================================================================
    // 13. list_servers with populated registry => server details
    // ========================================================================

    #[tokio::test]
    async fn test_list_servers_with_data() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "test-srv");
        assert_eq!(arr[0]["display_name"], "Test Server");
        assert_eq!(arr[0]["tool_count"], 2);
        assert_eq!(arr[0]["status"], "connected");
    }

    // ========================================================================
    // 14. get_server_status with populated registry => server found
    // ========================================================================

    #[tokio::test]
    async fn test_get_server_status_found() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers/test-srv"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["id"], "test-srv");
        assert_eq!(json["display_name"], "Test Server");
        assert_eq!(json["tool_count"], 2);
        assert!(!json["server_name"].is_null());
    }

    // ========================================================================
    // 15. list_tools with populated registry => tools returned
    // ========================================================================

    #[tokio::test]
    async fn test_list_server_tools_with_data() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers/test-srv/tools"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 2);
    }

    // ========================================================================
    // 16. disconnect with populated registry => success
    // ========================================================================

    #[tokio::test]
    async fn test_disconnect_server_success() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_delete("/api/mcp-federation/servers/test-srv"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["server_id"], "test-srv");
    }

    // ========================================================================
    // 17. connect streamable_http missing url => 400
    // ========================================================================

    #[tokio::test]
    async fn test_connect_server_streamable_http_missing_url() {
        let app = test_app_with_registry().await;
        let body = serde_json::json!({
            "server_id": "test-srv",
            "transport": "streamable_http"
        });
        let resp = app
            .oneshot(auth_post("/api/mcp-federation/servers", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ========================================================================
    // 18. probe nonexistent server => 404
    // ========================================================================

    #[tokio::test]
    async fn test_probe_server_not_found() {
        let app = test_app_with_registry().await;
        let resp = app
            .oneshot(auth_post(
                "/api/mcp-federation/servers/nonexistent/probe",
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ========================================================================
    // 19. reconnect nonexistent server => 404
    // ========================================================================

    #[tokio::test]
    async fn test_reconnect_server_not_found() {
        let app = test_app_with_registry().await;
        let resp = app
            .oneshot(auth_post(
                "/api/mcp-federation/servers/nonexistent/reconnect",
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ========================================================================
    // 20. get server status returns all expected fields
    // ========================================================================

    #[tokio::test]
    async fn test_get_server_status_fields() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers/test-srv"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;

        // Check all expected fields are present
        assert_eq!(json["id"], "test-srv");
        assert_eq!(json["display_name"], "Test Server");
        assert_eq!(json["status"], "connected");
        assert_eq!(json["tool_count"], 2);
        assert!(json["connected_at"].is_string());
        assert_eq!(json["server_name"], "MockMCP");
        // stats should be present
        assert!(json["stats"].is_object());
        assert_eq!(json["stats"]["call_count"], 0);
    }

    // ========================================================================
    // 21. list server tools returns tool details
    // ========================================================================

    #[tokio::test]
    async fn test_list_server_tools_returns_details() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers/test-srv/tools"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let tools = json.as_array().unwrap();
        assert_eq!(tools.len(), 2);

        // Check tool names
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"list_items"));
        assert!(names.contains(&"get_item"));

        // Check FQN format
        let fqns: Vec<&str> = tools.iter().map(|t| t["fqn"].as_str().unwrap()).collect();
        assert!(fqns.contains(&"test-srv::list_items"));
        assert!(fqns.contains(&"test-srv::get_item"));
    }

    // ========================================================================
    // 22. build_transport preserves display_name
    // ========================================================================

    #[test]
    fn test_build_transport_display_name_not_used() {
        // build_transport only creates the transport, not the full config
        let body = ConnectServerBody {
            server_id: "s1".into(),
            display_name: Some("Pretty Name".into()),
            transport: "sse".into(),
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: Some("http://test/sse".into()),
            headers: HashMap::new(),
        };
        let transport = build_transport(&body).unwrap();
        // display_name is not part of McpTransport — it's in McpTransportConfig
        match transport {
            McpTransport::Sse { url, .. } => assert_eq!(url, "http://test/sse"),
            _ => panic!("Expected Sse"),
        }
    }

    // ========================================================================
    // 23. disconnect already-disconnected server from populated registry
    // ========================================================================

    #[tokio::test]
    async fn test_list_servers_returns_correct_format() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_get("/api/mcp-federation/servers"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let arr = json.as_array().unwrap();

        // Each server should have consistent structure
        for server in arr {
            assert!(server["id"].is_string());
            assert!(server["display_name"].is_string());
            assert!(server["status"].is_string());
            assert!(server["tool_count"].is_number());
            assert!(server["connected_at"].is_string());
        }
    }

    // ========================================================================
    // 24. connect body deserialization tests
    // ========================================================================

    #[test]
    fn test_connect_body_minimal_deserialization() {
        let json = serde_json::json!({
            "server_id": "test",
            "transport": "stdio"
        });
        let body: ConnectServerBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.server_id, "test");
        assert!(body.command.is_none());
        assert!(body.args.is_empty());
        assert!(body.env.is_empty());
        assert!(body.url.is_none());
        assert!(body.headers.is_empty());
    }

    #[test]
    fn test_connect_body_full_deserialization() {
        let json = serde_json::json!({
            "server_id": "full-srv",
            "display_name": "Full Server",
            "transport": "stdio",
            "command": "node",
            "args": ["index.js", "--port", "3000"],
            "env": {"KEY": "val"},
            "url": null,
            "headers": {}
        });
        let body: ConnectServerBody = serde_json::from_value(json).unwrap();
        assert_eq!(body.server_id, "full-srv");
        assert_eq!(body.display_name, Some("Full Server".to_string()));
        assert_eq!(body.command, Some("node".to_string()));
        assert_eq!(body.args.len(), 3);
        assert_eq!(body.env.get("KEY").unwrap(), "val");
    }

    #[test]
    fn test_action_response_serialization() {
        let resp = ActionResponse {
            success: true,
            server_id: "srv1".to_string(),
            message: "Done".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["server_id"], "srv1");
        assert_eq!(json["message"], "Done");
    }

    // ========================================================================
    // 25. probe_server success path
    // ========================================================================

    #[tokio::test]
    async fn test_probe_server_success() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_post(
                "/api/mcp-federation/servers/test-srv/probe",
                serde_json::json!({}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["server_id"], "test-srv");
        assert!(json["total"].as_u64().unwrap() >= 2);
        assert!(json["probed"].is_number());
    }

    // ========================================================================
    // 26. disconnect cleans up and returns correct response
    // ========================================================================

    #[tokio::test]
    async fn test_disconnect_server_response_format() {
        let app = test_app_with_populated_registry().await;
        let resp = app
            .oneshot(auth_delete("/api/mcp-federation/servers/test-srv"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["server_id"], "test-srv");
        assert!(json["message"].as_str().unwrap().contains("disconnected"));
    }

    // ========================================================================
    // 27. extract_transport_fields (tests the extracted helper)
    // ========================================================================

    #[test]
    fn test_extract_transport_fields_stdio() {
        let transport = McpTransport::Stdio {
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "@pasympa/discord-mcp".to_string()],
            env: HashMap::new(),
        };
        let (tt, url, cmd, args_json) = extract_transport_fields(&transport);
        assert_eq!(tt, "stdio");
        assert!(url.is_none());
        assert_eq!(cmd.unwrap(), "npx");
        let args: Vec<String> = serde_json::from_str(&args_json.unwrap()).unwrap();
        assert_eq!(args, vec!["-y", "@pasympa/discord-mcp"]);
    }

    #[test]
    fn test_extract_transport_fields_sse() {
        let transport = McpTransport::Sse {
            url: "http://localhost:8080/sse".to_string(),
            headers: HashMap::new(),
        };
        let (tt, url, cmd, args) = extract_transport_fields(&transport);
        assert_eq!(tt, "sse");
        assert_eq!(url.unwrap(), "http://localhost:8080/sse");
        assert!(cmd.is_none());
        assert!(args.is_none());
    }

    #[test]
    fn test_extract_transport_fields_streamable_http() {
        let transport = McpTransport::StreamableHttp {
            url: "http://localhost:9090/mcp".to_string(),
            headers: HashMap::new(),
        };
        let (tt, url, cmd, args) = extract_transport_fields(&transport);
        assert_eq!(tt, "streamable_http");
        assert_eq!(url.unwrap(), "http://localhost:9090/mcp");
        assert!(cmd.is_none());
        assert!(args.is_none());
    }

    // ========================================================================
    // 28. build_server_node (tests the extracted helper)
    // ========================================================================

    #[test]
    fn test_build_server_node_stdio_with_display_name() {
        let transport = McpTransport::Stdio {
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "@pasympa/discord-mcp".to_string()],
            env: HashMap::new(),
        };
        let node = build_server_node("discord", Some("Discord Bot"), &transport, Some("discord-mcp"), 90);
        assert_eq!(node.server_id, "discord");
        assert_eq!(node.display_name, "Discord Bot");
        assert_eq!(node.transport_type, "stdio");
        assert!(node.transport_url.is_none());
        assert_eq!(node.transport_command.as_deref(), Some("npx"));
        assert_eq!(node.server_name.as_deref(), Some("discord-mcp"));
        assert_eq!(node.tool_count, 90);
        assert_eq!(node.status, "connected");
        assert!(node.updated_at.is_some());
        assert!(node.last_connected_at.is_some());
        assert_eq!(node.project_id, Uuid::nil());
    }

    #[test]
    fn test_build_server_node_display_name_fallback() {
        let transport = McpTransport::Sse {
            url: "http://host/sse".to_string(),
            headers: HashMap::new(),
        };
        // When display_name is None, server_id should be used
        let node = build_server_node("my-server", None, &transport, None, 5);
        assert_eq!(node.display_name, "my-server");
        assert_eq!(node.transport_type, "sse");
        assert_eq!(node.transport_url.as_deref(), Some("http://host/sse"));
        assert!(node.server_name.is_none());
    }

    #[test]
    fn test_build_server_node_streamable_http() {
        let transport = McpTransport::StreamableHttp {
            url: "http://localhost:9090/mcp".to_string(),
            headers: HashMap::new(),
        };
        let node = build_server_node("grafeo", Some("Grafeo"), &transport, Some("grafeo-server"), 12);
        assert_eq!(node.transport_type, "streamable_http");
        assert_eq!(node.transport_url.as_deref(), Some("http://localhost:9090/mcp"));
        assert!(node.transport_command.is_none());
        assert!(node.transport_args.is_none());
    }

    // ========================================================================
    // 29. Round-trip: build_server_node → reconstruct transport from stored fields
    // ========================================================================

    #[test]
    fn test_server_node_roundtrip_stdio() {
        let original = McpTransport::Stdio {
            command: "node".to_string(),
            args: vec!["server.js".to_string(), "--port".to_string(), "3000".to_string()],
            env: HashMap::new(),
        };
        let node = build_server_node("srv", None, &original, None, 0);

        // Reconstruct transport from stored fields (simulates bootstrap)
        assert_eq!(node.transport_type, "stdio");
        let command = node.transport_command.unwrap();
        let args: Vec<String> = serde_json::from_str(&node.transport_args.unwrap()).unwrap();
        assert_eq!(command, "node");
        assert_eq!(args, vec!["server.js", "--port", "3000"]);
    }

    #[test]
    fn test_server_node_roundtrip_sse() {
        let original = McpTransport::Sse {
            url: "http://localhost:8080/sse".to_string(),
            headers: HashMap::new(),
        };
        let node = build_server_node("srv", None, &original, None, 0);
        assert_eq!(node.transport_type, "sse");
        assert_eq!(node.transport_url.unwrap(), "http://localhost:8080/sse");
    }

    #[test]
    fn test_server_node_roundtrip_streamable_http() {
        let original = McpTransport::StreamableHttp {
            url: "http://localhost:9090/mcp".to_string(),
            headers: HashMap::new(),
        };
        let node = build_server_node("srv", None, &original, None, 0);
        assert_eq!(node.transport_type, "streamable_http");
        assert_eq!(node.transport_url.unwrap(), "http://localhost:9090/mcp");
    }
}
