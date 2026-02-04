//! Project Orchestrator - Main Server
//!
//! An AI agent orchestrator with Neo4j, Meilisearch, and Tree-sitter.

use anyhow::Result;
use clap::{Parser, Subcommand};
use project_orchestrator::{
    api::{self, handlers::ServerState},
    orchestrator::{FileWatcher, Orchestrator},
    AppState, Config,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "orchestrator")]
#[command(about = "AI Agent Orchestrator Server")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the orchestrator server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// Sync a directory to the knowledge base
    Sync {
        /// Directory path to sync
        #[arg(short, long, default_value = ".")]
        path: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,project_orchestrator=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    // Load configuration
    let mut config = Config::from_env()?;

    match cli.command {
        Commands::Serve { port } => {
            config.server_port = port;
            run_server(config).await
        }
        Commands::Sync { path } => run_sync(config, &path).await,
    }
}

async fn run_server(config: Config) -> Result<()> {
    tracing::info!("Starting Project Orchestrator server...");

    // Initialize application state
    tracing::info!("Connecting to Neo4j at {}...", config.neo4j_uri);
    tracing::info!("Connecting to Meilisearch at {}...", config.meilisearch_url);

    let state = AppState::new(config.clone()).await?;
    tracing::info!("Connected to databases");

    // Create orchestrator
    let orchestrator = Arc::new(Orchestrator::new(state).await?);

    // Create file watcher
    let watcher = FileWatcher::new(orchestrator.clone());

    // Create server state
    let server_state = Arc::new(ServerState {
        orchestrator,
        watcher: Arc::new(RwLock::new(watcher)),
    });

    // Create router
    let app = api::create_router(server_state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server_port));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn run_sync(config: Config, path: &str) -> Result<()> {
    tracing::info!("Syncing directory: {}", path);

    // Initialize application state
    let state = AppState::new(config).await?;
    tracing::info!("Connected to databases");

    // Create orchestrator
    let orchestrator = Orchestrator::new(state).await?;

    // Run sync
    let result = orchestrator
        .sync_directory(std::path::Path::new(path))
        .await?;

    tracing::info!(
        "Sync complete: {} files synced, {} skipped, {} errors",
        result.files_synced,
        result.files_skipped,
        result.errors
    );

    Ok(())
}
