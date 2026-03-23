//! MCP Federation — connect to external MCP servers as a consumer.
//!
//! PO is normally an MCP **server**. This module makes it also an MCP **client**,
//! enabling dynamic connection to external MCP servers via the three standard
//! transports (Stdio, SSE, Streamable HTTP).
//!
//! ## Architecture
//!
//! ```text
//! McpServerRegistry (Arc<RwLock<>>)
//!   ├── server "grafeo-shared"
//!   │     ├── transport: StreamableHttp { url }
//!   │     ├── client: Box<dyn McpClient>
//!   │     ├── circuit_breaker: CircuitBreaker
//!   │     ├── tools: Vec<DiscoveredTool>
//!   │     └── stats: ServerStats
//!   └── server "github"
//!         ├── transport: Stdio { command, args }
//!         └── ...
//! ```

pub mod client;
pub mod circuit_breaker;
pub mod discovery;
pub mod prober;
pub mod registry;

// Re-exports
pub use client::{McpClient, McpTransportConfig};
pub use circuit_breaker::CircuitBreaker;
pub use discovery::{DiscoveredTool, InferredCategory, ToolIntrospector, ToolProfile};
pub use prober::ToolProber;
pub use registry::{
    ConnectionStatus, McpServerConnection, McpServerRegistry, ServerStats,
};

use serde::{Deserialize, Serialize};

/// Unique identifier for a connected MCP server (e.g. "grafeo-shared", "github").
pub type ServerId = String;

/// Fully-qualified tool name: "server_id::tool_name".
pub type ToolFqn = String;

/// Transport configuration for connecting to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpTransport {
    /// Spawn a child process, communicate via stdin/stdout.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: std::collections::HashMap<String, String>,
    },
    /// Legacy MCP transport: HTTP GET (SSE) for server→client, HTTP POST for client→server.
    Sse {
        url: String,
        #[serde(default)]
        headers: std::collections::HashMap<String, String>,
    },
    /// MCP 2025-03 standard: POST with optional SSE response stream.
    StreamableHttp {
        url: String,
        #[serde(default)]
        headers: std::collections::HashMap<String, String>,
    },
}

/// Basic information about a discovered tool from an external MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalToolInfo {
    /// Tool name as reported by the server (e.g. "run_cypher").
    pub name: String,
    /// Fully-qualified name: "server_id::tool_name".
    pub fqn: ToolFqn,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// MCP protocol version we speak as a client.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Our client info sent during initialize handshake.
pub fn client_info() -> serde_json::Value {
    serde_json::json!({
        "name": "project-orchestrator",
        "version": env!("CARGO_PKG_VERSION"),
    })
}

/// Build an initialize request payload.
pub fn initialize_params() -> serde_json::Value {
    serde_json::json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "roots": { "listChanged": false },
        },
        "clientInfo": client_info(),
    })
}
