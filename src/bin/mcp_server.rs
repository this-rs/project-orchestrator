//! MCP Server Binary — HTTP Proxy Mode
//!
//! This binary runs a lightweight MCP server that proxies tool calls to the
//! project-orchestrator REST API. It communicates over stdio for integration
//! with Claude Code and other MCP clients.
//!
//! # Environment Variables
//!
//! - `PO_SERVER_URL` (required): REST API base URL (e.g. `http://127.0.0.1:8080`)
//! - `PO_AUTH_TOKEN` (optional): JWT session token for authenticated requests
//!
//! # Architecture
//!
//! The MCP server is a thin proxy: it receives MCP tool calls over stdio,
//! translates them to HTTP REST requests, and returns the results. It does NOT
//! connect to Neo4j, Meilisearch, or any other backend directly — all state
//! lives in the REST API server.
//!
//! # Claude Code Integration
//!
//! Add to your Claude Code MCP settings (e.g., `~/.claude/mcp.json`):
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "project-orchestrator": {
//!       "command": "/path/to/mcp_server",
//!       "env": {
//!         "PO_SERVER_URL": "http://127.0.0.1:8080",
//!         "PO_AUTH_TOKEN": "<jwt-session-token>"
//!       }
//!     }
//!   }
//! }
//! ```

use anyhow::{anyhow, Result};
use project_orchestrator::mcp::{McpHttpClient, McpServer};
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env if present
    let _ = dotenvy::dotenv();

    // Initialize logging (to stderr to keep stdout clean for MCP)
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive("project_orchestrator=info".parse()?))
        .init();

    // Create HTTP client from environment
    let http_client = McpHttpClient::from_env().ok_or_else(|| {
        anyhow!(
            "PO_SERVER_URL environment variable is required.\n\
             The MCP server operates as an HTTP proxy to the REST API.\n\
             Set PO_SERVER_URL to the orchestrator server URL (e.g. http://127.0.0.1:8080)."
        )
    })?;

    info!("MCP server starting in HTTP proxy mode");
    info!(
        "Proxying tool calls to REST API (auth={})",
        std::env::var("PO_AUTH_TOKEN").is_ok()
    );

    let mut server = McpServer::new(http_client);

    if let Err(e) = server.run().await {
        error!("MCP server error: {}", e);
        return Err(e);
    }

    Ok(())
}
