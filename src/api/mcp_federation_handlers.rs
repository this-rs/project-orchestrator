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
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// Extract the shared registry from state, returning 501 if federation is not configured.
fn get_registry(
    state: &OrchestratorState,
) -> Result<&crate::mcp_federation::registry::SharedRegistry, AppError> {
    state.mcp_registry.as_ref().ok_or_else(|| {
        AppError::Internal(anyhow::anyhow!(
            "MCP Federation is not configured. Enable it in the server configuration."
        ))
    })
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
    let registry = get_registry(&state)?;
    let reg = registry.read().await;
    let servers = reg.list();
    Ok(Json(serde_json::to_value(servers).map_err(|e| {
        AppError::Internal(anyhow::anyhow!("Serialization error: {}", e))
    })?))
}

/// Connect to a new external MCP server.
///
/// POST /api/mcp-federation/servers
pub async fn connect_server(
    State(state): State<OrchestratorState>,
    Json(body): Json<ConnectServerBody>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    let registry = get_registry(&state)?;
    let transport = build_transport(&body)?;

    let config = McpTransportConfig {
        server_id: body.server_id.clone(),
        display_name: body.display_name,
        transport,
    };

    let mut reg = registry.write().await;
    let summary = reg.connect(config).await.map_err(AppError::Internal)?;

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
    let registry = get_registry(&state)?;
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
    let registry = get_registry(&state)?;
    let mut reg = registry.write().await;
    reg.disconnect(&server_id)
        .await
        .map_err(AppError::Internal)?;

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
    let registry = get_registry(&state)?;
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
    let registry = get_registry(&state)?;
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
    let registry = get_registry(&state)?;

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
    use axum::{body::Body, http::{Request, StatusCode}};
    use tower::ServiceExt;
    use crate::api::routes::create_router;
    use crate::test_helpers::{test_auth_config, test_bearer_token, mock_app_state};
    use crate::events::EventBus;
    use crate::orchestrator::{Orchestrator, FileWatcher};
    use crate::api::handlers::ServerState;
    use std::sync::Arc;
    use std::collections::HashMap;

    /// App WITHOUT mcp_registry (None)
    async fn test_app() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(orchestrator.clone())));
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(EventBus::default()))),
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
            mcp_registry: None,
        });
        create_router(state)
    }

    /// App WITH mcp_registry enabled (empty registry)
    async fn test_app_with_registry() -> axum::Router {
        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(orchestrator.clone())));
        let registry = crate::mcp_federation::registry::new_shared_registry();
        let state = Arc::new(ServerState {
            orchestrator,
            watcher,
            chat_manager: None,
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(EventBus::default()))),
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
            mcp_registry: Some(registry),
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
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ========================================================================
    // 1. list_servers when mcp_registry is None => 500
    // ========================================================================

    #[tokio::test]
    async fn test_list_servers_no_registry() {
        let app = test_app().await;
        let resp = app.oneshot(auth_get("/api/mcp-federation/servers")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ========================================================================
    // 2. list_servers with empty registry => 200 + empty array
    // ========================================================================

    #[tokio::test]
    async fn test_list_servers_empty() {
        let app = test_app_with_registry().await;
        let resp = app.oneshot(auth_get("/api/mcp-federation/servers")).await.unwrap();
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
        use crate::mcp_federation::registry::{
            McpServerConnection, McpServerRegistry, ConnectionStatus, ServerStats,
        };
        use crate::mcp_federation::circuit_breaker::CircuitBreaker;
        use crate::mcp_federation::discovery::{DiscoveredTool, InferredCategory};

        let app_state = mock_app_state();
        let orchestrator = Arc::new(Orchestrator::new(app_state).await.unwrap());
        let watcher = Arc::new(tokio::sync::RwLock::new(FileWatcher::new(orchestrator.clone())));

        // Build a registry with a mock server
        let mut registry = McpServerRegistry::new();

        // Mock client (minimal)
        #[derive(Debug)]
        struct MinimalMockClient;
        #[async_trait::async_trait]
        impl crate::mcp_federation::client::McpClient for MinimalMockClient {
            async fn initialize(&self) -> anyhow::Result<crate::mcp_federation::client::InitializeResult> {
                unimplemented!()
            }
            async fn initialized_notification(&self) -> anyhow::Result<()> { Ok(()) }
            async fn tools_list(&self) -> anyhow::Result<Vec<crate::mcp_federation::client::McpToolDef>> {
                Ok(vec![])
            }
            async fn call_tool(&self, _: &str, _: Option<serde_json::Value>) -> anyhow::Result<serde_json::Value> {
                Ok(serde_json::json!({}))
            }
            async fn ping(&self) -> anyhow::Result<()> { Ok(()) }
            async fn shutdown(&self) -> anyhow::Result<()> { Ok(()) }
            fn transport_name(&self) -> &'static str { "mock" }
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
            event_bus: Arc::new(crate::events::HybridEmitter::new(Arc::new(EventBus::default()))),
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
            mcp_registry: Some(shared_registry),
        });
        create_router(state)
    }

    // ========================================================================
    // 13. list_servers with populated registry => server details
    // ========================================================================

    #[tokio::test]
    async fn test_list_servers_with_data() {
        let app = test_app_with_populated_registry().await;
        let resp = app.oneshot(auth_get("/api/mcp-federation/servers")).await.unwrap();
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
}
