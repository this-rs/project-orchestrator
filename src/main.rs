//! Project Orchestrator - Main Server
//!
//! An AI agent orchestrator with Neo4j, Meilisearch, and Tree-sitter.

use anyhow::Result;
use clap::{Parser, Subcommand};
use project_orchestrator::{orchestrator::Orchestrator, AppState, Config};
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

        /// Disable serving the frontend static files (API-only mode)
        #[arg(long)]
        no_frontend: bool,

        /// Path to the frontend dist/ directory (overrides config.yaml)
        #[arg(long)]
        frontend_path: Option<String>,
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
        Commands::Serve {
            port,
            no_frontend,
            frontend_path,
        } => {
            config.server_port = port;
            if no_frontend {
                config.serve_frontend = false;
            }
            if let Some(path) = frontend_path {
                config.frontend_path = path;
            }
            project_orchestrator::start_server(config).await
        }
        Commands::Sync { path } => run_sync(config, &path).await,
    }
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
