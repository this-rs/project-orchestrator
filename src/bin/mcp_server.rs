//! MCP Server Binary
//!
//! This binary runs the project-orchestrator as an MCP server,
//! communicating over stdio for integration with Claude Code.
//!
//! # Usage
//!
//! ```bash
//! # Run directly
//! ./mcp_server
//!
//! # With environment variables
//! NEO4J_URI=bolt://localhost:7687 ./mcp_server
//!
//! # With debug logging
//! RUST_LOG=debug ./mcp_server
//! ```
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
//!         "NEO4J_URI": "bolt://localhost:7687",
//!         "NEO4J_USER": "neo4j",
//!         "NEO4J_PASSWORD": "your-password",
//!         "MEILISEARCH_URL": "http://localhost:7700",
//!         "MEILISEARCH_KEY": "your-key"
//!       }
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use clap::Parser;
use project_orchestrator::chat::{ChatConfig, ChatManager};
use project_orchestrator::events::EventNotifier;
use project_orchestrator::mcp::McpServer;
use project_orchestrator::orchestrator::Orchestrator;
use project_orchestrator::{AppState, Config};
use std::sync::Arc;
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// MCP Server for project-orchestrator
#[derive(Parser, Debug)]
#[command(name = "mcp_server")]
#[command(about = "MCP server exposing project-orchestrator tools for Claude Code")]
#[command(version)]
struct Args {
    /// Neo4j connection URI
    #[arg(long, env = "NEO4J_URI", default_value = "bolt://localhost:7687")]
    neo4j_uri: String,

    /// Neo4j username
    #[arg(long, env = "NEO4J_USER", default_value = "neo4j")]
    neo4j_user: String,

    /// Neo4j password
    #[arg(long, env = "NEO4J_PASSWORD", default_value = "orchestrator123")]
    neo4j_password: String,

    /// Meilisearch URL
    #[arg(long, env = "MEILISEARCH_URL", default_value = "http://localhost:7700")]
    meilisearch_url: String,

    /// Meilisearch API key
    #[arg(
        long,
        env = "MEILISEARCH_KEY",
        default_value = "orchestrator-meili-key-change-me"
    )]
    meilisearch_key: String,

    /// HTTP server URL for event forwarding (MCP → HTTP bridge)
    #[arg(long, env = "MCP_HTTP_URL", default_value = "http://localhost:8080")]
    http_url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env if present
    let _ = dotenvy::dotenv();

    // Initialize logging (to stderr to keep stdout clean for MCP)
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env().add_directive("project_orchestrator=info".parse()?))
        .init();

    let args = Args::parse();

    info!("Starting MCP server for project-orchestrator");
    info!("Neo4j: {}", args.neo4j_uri);
    info!("Meilisearch: {}", args.meilisearch_url);

    // Create config and app state
    let config = Config {
        neo4j_uri: args.neo4j_uri,
        neo4j_user: args.neo4j_user,
        neo4j_password: args.neo4j_password,
        meilisearch_url: args.meilisearch_url,
        meilisearch_key: args.meilisearch_key,
        workspace_path: ".".to_string(),
        server_port: 8080,          // Not used in MCP mode
        auth_config: None,           // MCP server doesn't need auth (stdio-based)
        serve_frontend: false,       // MCP server doesn't serve frontend
        frontend_path: "./dist".to_string(),
    };

    let state = match AppState::new(config).await {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to create app state: {}", e);
            return Err(e);
        }
    };

    // Create event notifier for MCP → HTTP bridge
    let notifier = Arc::new(EventNotifier::new(&args.http_url));
    info!("Event notifier targeting: {}", args.http_url);

    // Create orchestrator with event notifier
    let orchestrator = match Orchestrator::with_event_emitter(state, notifier).await {
        Ok(o) => Arc::new(o),
        Err(e) => {
            error!("Failed to create orchestrator: {}", e);
            return Err(e);
        }
    };

    // Create ChatManager
    let chat_config = ChatConfig::from_env();
    let chat_manager = Arc::new(
        ChatManager::new(
            orchestrator.neo4j_arc(),
            orchestrator.meili_arc(),
            chat_config,
        )
        .await,
    );
    chat_manager.start_cleanup_task();
    info!("Chat manager initialized");

    // Create and run MCP server
    let mut server = McpServer::with_chat_manager(orchestrator, chat_manager);

    if let Err(e) = server.run().await {
        error!("MCP server error: {}", e);
        return Err(e);
    }

    Ok(())
}
