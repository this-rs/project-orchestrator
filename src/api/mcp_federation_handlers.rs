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
fn build_transport(body: &ConnectServerBody) -> Result<McpTransport, AppError> {
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
                AppError::BadRequest(
                    "'url' is required for streamable_http transport".to_string(),
                )
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
        Json(serde_json::to_value(summary).map_err(|e| {
            AppError::Internal(anyhow::anyhow!("Serialization error: {}", e))
        })?),
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
        .ok_or_else(|| {
            AppError::NotFound(format!("MCP server '{}' not found", server_id))
        })?;

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

    let conn = reg.get_mut(&server_id).ok_or_else(|| {
        AppError::NotFound(format!("MCP server '{}' not found", server_id))
    })?;

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
        let conn = reg.get(&server_id).ok_or_else(|| {
            AppError::NotFound(format!("MCP server '{}' not found", server_id))
        })?;
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
