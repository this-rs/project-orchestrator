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

pub mod circuit_breaker;
pub mod client;
pub mod discovery;
pub mod prober;
pub mod registry;
pub mod security;

// Re-exports
pub use circuit_breaker::CircuitBreaker;
pub use client::{McpClient, McpTransportConfig};
pub use discovery::{DiscoveredTool, InferredCategory, ToolIntrospector, ToolProfile};
pub use prober::ToolProber;
pub use registry::{ConnectionStatus, McpServerConnection, McpServerRegistry, ServerStats};
pub use security::{McpSecurityPolicy, PolicyViolation, RateLimiter, SecurityEnforcer};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_info() {
        let info = client_info();
        assert_eq!(info["name"], "project-orchestrator");
        assert!(info["version"].is_string());
        // Version should match CARGO_PKG_VERSION
        assert_eq!(info["version"], env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_initialize_params() {
        let params = initialize_params();
        assert_eq!(params["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert!(params["capabilities"].is_object());
        assert!(params["clientInfo"].is_object());
        assert_eq!(params["clientInfo"]["name"], "project-orchestrator");
    }

    #[test]
    fn test_external_tool_info_serde_roundtrip() {
        let info = ExternalToolInfo {
            name: "run_cypher".to_string(),
            fqn: "grafeo::run_cypher".to_string(),
            description: "Execute a Cypher query".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        };

        let json = serde_json::to_string(&info).unwrap();
        let roundtrip: ExternalToolInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(roundtrip.name, "run_cypher");
        assert_eq!(roundtrip.fqn, "grafeo::run_cypher");
        assert_eq!(roundtrip.description, "Execute a Cypher query");
        assert_eq!(roundtrip.input_schema["type"], "object");
    }

    #[test]
    fn test_mcp_transport_serde_stdio() {
        let transport = McpTransport::Stdio {
            command: "node".to_string(),
            args: vec!["server.js".to_string()],
            env: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&transport).unwrap();
        assert!(json.contains("\"type\":\"stdio\""));
        let back: McpTransport = serde_json::from_str(&json).unwrap();
        match back {
            McpTransport::Stdio { command, args, .. } => {
                assert_eq!(command, "node");
                assert_eq!(args, vec!["server.js"]);
            }
            _ => panic!("Expected Stdio variant"),
        }
    }

    #[test]
    fn test_mcp_transport_serde_sse() {
        let transport = McpTransport::Sse {
            url: "https://example.com/sse".to_string(),
            headers: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&transport).unwrap();
        assert!(json.contains("\"type\":\"sse\""));
        let back: McpTransport = serde_json::from_str(&json).unwrap();
        match back {
            McpTransport::Sse { url, .. } => assert_eq!(url, "https://example.com/sse"),
            _ => panic!("Expected Sse variant"),
        }
    }

    #[test]
    fn test_mcp_transport_serde_streamable_http() {
        let transport = McpTransport::StreamableHttp {
            url: "https://example.com/mcp".to_string(),
            headers: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&transport).unwrap();
        assert!(json.contains("\"type\":\"streamable_http\""));
        let back: McpTransport = serde_json::from_str(&json).unwrap();
        match back {
            McpTransport::StreamableHttp { url, .. } => {
                assert_eq!(url, "https://example.com/mcp")
            }
            _ => panic!("Expected StreamableHttp variant"),
        }
    }

    #[test]
    fn test_mcp_protocol_version_constant() {
        assert_eq!(MCP_PROTOCOL_VERSION, "2024-11-05");
    }
}
