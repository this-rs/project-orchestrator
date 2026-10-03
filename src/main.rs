//! Project Orchestrator - Main Server
//!
//! An AI agent orchestrator with Neo4j, Meilisearch, and Tree-sitter.

use anyhow::Result;
use clap::{Parser, Subcommand};
use project_orchestrator::{orchestrator::Orchestrator, setup_claude, update, AppState, Config};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "orchestrator")]
#[command(about = "AI Agent Orchestrator Server")]
struct Cli {
    /// Path to config.yaml (default: auto-detect)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the orchestrator server
    Serve {
        /// Port to listen on (overrides config.yaml server.port, default: 8080 if no config)
        #[arg(short, long)]
        port: Option<u16>,

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
        /// Override the server port (default: from config or 8080)
        #[arg(long)]
        port: Option<u16>,
    },

    /// Use a secret granted to this chat session (agents only; needs PO_VAULT_TOKEN)
    #[command(subcommand)]
    Secret(SecretCommand),
}

#[derive(Subcommand)]
enum SecretCommand {
    /// Run a command with secrets in its environment — nothing is printed.
    /// Example: orchestrator secret exec -e PGPASSWORD=db-prod -- psql -h db
    Exec {
        /// VAR=SECRET_NAME (repeatable)
        #[arg(short = 'e', long = "env", required = true, value_parser = project_orchestrator::vault::agent_cli::parse_mapping)]
        env: Vec<(String, String)>,
        /// The command and its arguments, after `--`
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    /// Write a secret to stdout, to pipe it: orchestrator secret get NAME | cmd --password-stdin
    Get { name: String },
}

/// Stack size of every runtime thread (workers and `spawn_blocking`), in bytes.
///
/// The default is 2 MiB, and the chat turn does not fit in it: `stream_response`
/// is one very large `async fn`, polled through the enrichment pipeline, the
/// status stage, a Neo4j query, `neo4rs`, `deadpool` and the Bolt encoder, each
/// layer's future living on the worker's stack while it is polled. On
/// 2026-10-03 that chain overflowed a worker at the leaf (`BytesMut::reserve`),
/// the process aborted with SIGABRT, and every live Claude CLI died with it
/// ("The CLI subprocess for this session has exited"). Stack pages are mapped
/// lazily, so a larger size costs address space, not memory.
pub const RUNTIME_STACK_BYTES: usize = 16 * 1024 * 1024;

// Checked when the crate compiles, not when tests run: a size that is not a whole
// number of 16 KiB pages (Apple silicon) or that falls under 8 MiB is a build error.
const _: () = assert!(
    RUNTIME_STACK_BYTES.is_multiple_of(16 * 1024) && RUNTIME_STACK_BYTES >= 8 * 1024 * 1024
);

/// The multi-thread runtime of the server: `#[tokio::main]` with the stack above.
fn build_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(RUNTIME_STACK_BYTES)
        .build()
}

fn main() -> Result<()> {
    build_runtime()?.block_on(run())
}

async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Before anything else: no tracing, no config, no .env — stdout must carry
    // the value and nothing else, and none of it is needed.
    if let Commands::Secret(cmd) = &cli.command {
        use project_orchestrator::vault::agent_cli;
        let code = match cmd {
            SecretCommand::Get { name } => agent_cli::get(name).await,
            SecretCommand::Exec { env, command } => agent_cli::exec(env, command).await,
        };
        // Exit here: falling through would start logging.
        std::process::exit(code);
    }

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

    // Load configuration — explicit --config path wins, otherwise auto-detect
    let mut config = Config::from_yaml_and_env(cli.config.as_deref())?;

    match cli.command {
        Commands::Serve {
            port,
            no_frontend,
            frontend_path,
        } => {
            // --port flag overrides config.yaml; if neither is set, default to 8080
            if let Some(p) = port {
                config.server_port = p;
            } else if config.server_port == 0 {
                config.server_port = 8080;
            }
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
        Commands::SetupClaude { port } => {
            let effective_port = port.unwrap_or(config.server_port);
            run_setup_claude(&config, effective_port);
            Ok(())
        }
        Commands::Secret(_) => unreachable!("handled before startup"),
    }
}

fn run_setup_claude(config: &Config, port: u16) {
    use project_orchestrator::chat::ChatConfig;

    println!("Configuring Claude Code MCP server (stdio mode)...");
    println!();

    // Auto-detect the mcp_server binary path using the same logic as ChatManager
    let mcp_server_path = ChatConfig::detect_mcp_server_path_public();

    let setup_config = setup_claude::SetupConfig {
        mcp_server_path,
        server_port: port,
        jwt_secret: config.auth_config.as_ref().map(|a| a.jwt_secret.clone()),
    };

    match setup_claude::setup_claude_code(&setup_config) {
        Ok(setup_claude::SetupResult::ConfiguredViaCli {
            allowed_tools_configured,
        }) => {
            println!("  Claude Code configured via CLI (stdio mode).");
            if allowed_tools_configured {
                println!("  MCP tools pre-approved in settings.json.");
            }
            println!();
            println!("  The MCP server has been added. You can verify with:");
            println!("    claude mcp list");
        }
        Ok(setup_claude::SetupResult::ConfiguredViaFile {
            path,
            allowed_tools_configured,
        }) => {
            println!("  Claude Code configured via {}.", path.display());
            if allowed_tools_configured {
                println!("  MCP tools pre-approved in settings.json.");
            }
            println!();
            println!("  The MCP server entry has been added to your mcp.json.");
            println!("  Restart Claude Code to pick up the changes.");
        }
        Ok(setup_claude::SetupResult::Updated {
            path,
            allowed_tools_configured,
        }) => {
            println!("  Claude Code config updated in {}.", path.display());
            if allowed_tools_configured {
                println!("  MCP tools pre-approved in settings.json.");
            }
            println!();
            println!("  Stale SSE config replaced with stdio mode.");
            println!("  Restart Claude Code to pick up the changes.");
        }
        Ok(setup_claude::SetupResult::AlreadyConfigured {
            allowed_tools_configured,
        }) => {
            println!("  Project Orchestrator is already configured in Claude Code.");
            if allowed_tools_configured {
                println!("  MCP tools pre-approved in settings.json.");
            } else {
                println!("  No changes made.");
            }
        }
        Err(e) => {
            eprintln!("  Failed to configure Claude Code: {}", e);
            eprintln!();
            eprintln!("  You can configure it manually:");
            eprintln!(
                "    claude mcp add -e PO_SERVER_URL=http://127.0.0.1:{} -e PO_JWT_SECRET=<secret> project-orchestrator -- {}",
                port,
                setup_config.mcp_server_path.display()
            );
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
        update::UpdateOutcome::Updated { from, to } => {
            println!("  Successfully updated from v{} to v{}!", from, to);
            println!("  Please restart orchestrator to use the new version.");
        }
        update::UpdateOutcome::AlreadyUpToDate => {
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

#[cfg(test)]
mod runtime_tests {
    use super::*;

    /// Touches `bytes` of stack. Not inlined, and `black_box` keeps the array.
    #[inline(never)]
    fn burn_stack<const N: usize>() -> usize {
        let buf = [7u8; N];
        std::hint::black_box(&buf)
            .iter()
            .step_by(4096)
            .map(|b| *b as usize)
            .sum()
    }

    #[test]
    fn a_worker_has_room_for_a_stack_far_beyond_the_two_mib_default() {
        // 8 MiB of live stack: more than 4 times the default, below our size.
        // On a 2 MiB worker this aborts the whole test process, as it did the server.
        let rt = build_runtime().expect("runtime");
        let touched = rt
            .block_on(async { tokio::spawn(async { burn_stack::<{ 8 * 1024 * 1024 }>() }).await })
            .expect("task");
        assert!(touched > 0);
    }
}
