//! Project Orchestrator - Main Server
//!
//! An AI agent orchestrator with Neo4j, Meilisearch, and Tree-sitter.

use anyhow::Result;
use clap::{Parser, Subcommand};
use project_orchestrator::{orchestrator::Orchestrator, setup_claude, update, AppState, Config};
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

    /// Check for updates and optionally install them
    Update {
        /// Only check for updates, don't install
        #[arg(long)]
        check: bool,
    },

    /// Configure Claude Code to use this server as MCP provider
    SetupClaude {
        /// MCP server URL (default: http://localhost:{port}/mcp/sse)
        #[arg(long)]
        url: Option<String>,

        /// Port used to build the default MCP server URL (default: from config or 8080)
        #[arg(long)]
        port: Option<u16>,
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
        Commands::Update { check } => run_update(check).await,
        Commands::SetupClaude { url, port } => {
            let effective_port = port.unwrap_or(config.server_port);
            run_setup_claude(url.as_deref(), effective_port);
            Ok(())
        }
    }
}

fn run_setup_claude(url: Option<&str>, port: u16) {
    println!("Configuring Claude Code MCP server...");
    println!();

    let default_url = format!("http://localhost:{}/mcp/sse", port);

    match setup_claude::setup_claude_code(url, Some(port)) {
        Ok(setup_claude::SetupResult::ConfiguredViaCli) => {
            println!("  Claude Code configured via CLI.");
            println!();
            println!("  The MCP server has been added. You can verify with:");
            println!("    claude mcp list");
        }
        Ok(setup_claude::SetupResult::ConfiguredViaFile { path }) => {
            println!("  Claude Code configured via {}.", path.display());
            println!();
            println!("  The MCP server entry has been added to your mcp.json.");
            println!("  Restart Claude Code to pick up the changes.");
        }
        Ok(setup_claude::SetupResult::AlreadyConfigured) => {
            println!("  Project Orchestrator is already configured in Claude Code.");
            println!("  No changes made.");
        }
        Err(e) => {
            let fallback_url = url.unwrap_or(&default_url);
            eprintln!("  Failed to configure Claude Code: {}", e);
            eprintln!();
            eprintln!("  You can configure it manually:");
            eprintln!(
                "    claude mcp add project-orchestrator --transport sse --url {}",
                fallback_url
            );
            eprintln!();
            eprintln!("  Or add to ~/.claude/mcp.json:");
            eprintln!("    {{");
            eprintln!("      \"mcpServers\": {{");
            eprintln!("        \"project-orchestrator\": {{");
            eprintln!("          \"type\": \"sse\",");
            eprintln!("          \"url\": \"{}\"", fallback_url);
            eprintln!("        }}");
            eprintln!("      }}");
            eprintln!("    }}");
        }
    }
}

async fn run_update(check_only: bool) -> Result<()> {
    println!("Checking for updates...");

    let info = match update::check_for_update().await? {
        Some(info) => info,
        None => {
            println!(
                "You're already on the latest version (v{}).",
                env!("CARGO_PKG_VERSION")
            );
            return Ok(());
        }
    };

    println!();
    println!(
        "  New version available: v{} (current: v{})",
        info.latest_version, info.current_version
    );
    println!("  Release: {}", info.html_url);

    if let Some(notes) = &info.release_notes {
        let preview: Vec<&str> = notes.lines().take(10).collect();
        println!();
        println!("  Release notes:");
        for line in &preview {
            println!("    {}", line);
        }
        if notes.lines().count() > 10 {
            println!("    ...");
        }
    }

    if check_only {
        println!();
        println!("Run `orchestrator update` to install this update.");
        return Ok(());
    }

    // Ask for confirmation
    println!();
    print!("  Install update? [Y/n] ");
    std::io::Write::flush(&mut std::io::stdout())?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    if !input.is_empty() && input != "y" && input != "yes" {
        println!("Update cancelled.");
        return Ok(());
    }

    println!();
    match update::perform_update(&info).await? {
        update::UpdateStatus::Updated { from, to } => {
            println!("  Successfully updated from v{} to v{}!", from, to);
            println!("  Please restart orchestrator to use the new version.");
        }
        update::UpdateStatus::AlreadyUpToDate => {
            println!("  Already up to date.");
        }
    }

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
