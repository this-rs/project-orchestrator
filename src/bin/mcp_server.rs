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
//!         "MEILISEARCH_KEY": "your-key",
//!         "NATS_URL": "nats://localhost:4222"
//!       }
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use clap::Parser;
use project_orchestrator::chat::{ChatConfig, ChatManager};
use project_orchestrator::events::{connect_nats, EventNotifier, NatsEmitter};
use project_orchestrator::mcp::McpServer;
use project_orchestrator::orchestrator::Orchestrator;
use project_orchestrator::{AppState, Config};
use std::sync::Arc;
use tracing::{error, info, warn};
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

    /// NATS URL for inter-process event sync
    #[arg(long, env = "NATS_URL")]
    nats_url: Option<String>,

    /// [DEPRECATED] HTTP server URL for event forwarding — use NATS_URL instead
    #[arg(long, env = "MCP_HTTP_URL")]
    http_url: Option<String>,
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

    // Deprecation warning for MCP_HTTP_URL
    if args.http_url.is_some() {
        warn!(
            "MCP_HTTP_URL / --http-url is deprecated and will be removed in a future version. \
             Use NATS_URL instead for inter-process event sync."
        );
    }

    // Create config and app state
    let config = Config {
        setup_completed: true,
        neo4j_uri: args.neo4j_uri,
        neo4j_user: args.neo4j_user,
        neo4j_password: args.neo4j_password,
        meilisearch_url: args.meilisearch_url,
        meilisearch_key: args.meilisearch_key,
        nats_url: args.nats_url.clone(),
        workspace_path: ".".to_string(),
        server_port: 8080, // Not used in MCP mode
        auth_config: None, // MCP server doesn't need auth (stdio-based)
        serve_frontend: false, // MCP server doesn't serve frontend
        frontend_path: "./dist".to_string(),
    };

    let state = match AppState::new(config).await {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to create app state: {}", e);
            return Err(e);
        }
    };

    // Create event emitter: prefer NATS, fallback to HTTP (deprecated), or none
    let nats_emitter: Option<Arc<NatsEmitter>> = if let Some(ref nats_url) = args.nats_url {
        match connect_nats(nats_url).await {
            Ok(client) => {
                let emitter = Arc::new(NatsEmitter::new(client, "events"));
                info!("MCP events will be published via NATS");
                Some(emitter)
            }
            Err(e) => {
                warn!("Failed to connect to NATS: {} — events will not be forwarded", e);
                None
            }
        }
    } else {
        None
    };

    let event_emitter: Option<Arc<dyn project_orchestrator::events::EventEmitter>> =
        if let Some(ref nats) = nats_emitter {
            Some(nats.clone() as Arc<dyn project_orchestrator::events::EventEmitter>)
        } else if let Some(ref http_url) = args.http_url {
            // Legacy fallback: HTTP POST bridge (deprecated)
            let notifier = Arc::new(EventNotifier::new(http_url));
            warn!(
                "Using deprecated HTTP event bridge targeting: {} — migrate to NATS_URL",
                http_url
            );
            Some(notifier)
        } else {
            info!("No NATS_URL configured — MCP events will not be forwarded to HTTP instances");
            None
        };

    // Create orchestrator (with or without event emitter)
    let orchestrator = match event_emitter {
        Some(emitter) => Orchestrator::with_event_emitter(state, emitter).await,
        None => Orchestrator::new(state).await,
    };

    let orchestrator = match orchestrator {
        Ok(o) => Arc::new(o),
        Err(e) => {
            error!("Failed to create orchestrator: {}", e);
            return Err(e);
        }
    };

    // Create ChatManager
    let chat_config = ChatConfig::from_env();
    let mut cm = ChatManager::new(
        orchestrator.neo4j_arc(),
        orchestrator.meili_arc(),
        chat_config,
    )
    .await;
    if let Some(ref nats) = nats_emitter {
        cm = cm.with_nats(nats.clone());
    }
    let chat_manager = Arc::new(cm);
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
