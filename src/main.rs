//! Project Orchestrator - Main Server
//!
//! An AI agent orchestrator with Neo4j, Meilisearch, and Tree-sitter.
#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use anyhow::Result;
use clap::{Parser, Subcommand};
use project_orchestrator::{orchestrator::Orchestrator, setup_claude, update, AppState, Config};
use std::path::PathBuf;

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

    /// Data migrations that are triggered, never run at boot
    #[command(subcommand)]
    Migrate(MigrateCommand),
}

#[derive(Subcommand)]
enum MigrateCommand {
    /// Rename a project's code graph from the canonical (alias) path to its declared root (T7).
    /// Without --apply: a dry-run that prints the control measures and what would be done.
    /// Example: orchestrator migrate file-paths --project project-orchestrator-backend
    FilePaths {
        /// Slug of the project to migrate
        #[arg(long)]
        project: String,
        /// Write: rename File.path, the symbols' file_path/id/stable_key and the anchors, by batches of 2 000
        #[arg(long)]
        apply: bool,
        /// After the rename: re-anchor the knowledge of the `.wl` worktree copies on their twin, then delete the copies (needs --apply; refused when the measures are wrong)
        #[arg(long, requires = "apply")]
        purge_wl: bool,
    },
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

use project_orchestrator::runtime::build_runtime;

/// The highest soft limit of open files this process ever asks for. A server
/// serving HTTP, websockets, database connections and child CLIs needs
/// thousands of descriptors — never hundreds of thousands, so the cap keeps a
/// fat-fingered value harmless.
const FD_LIMIT_CAP: u64 = 65_536;

/// The soft limit of open files to request, given the current soft and hard
/// limits, or `None` when there is nothing to raise.
///
/// The hard limit always wins: a soft limit above it is refused by the kernel.
/// A soft limit already at or above what we would ask for is left alone, so an
/// environment that is already generous costs one `getrlimit` and no syscall
/// that changes anything.
fn fd_limit_target(soft: u64, hard: u64) -> Option<u64> {
    let wanted = FD_LIMIT_CAP.min(hard);
    (wanted > soft).then_some(wanted)
}

/// Raise this process's soft limit of open files, once, before anything opens a
/// socket, a pipe or a file.
///
/// Why this exists: a server inherits the soft limit of whoever started it, and
/// that is 256 in a macOS shell. Past that wall the *next* `open`, `socket` or
/// `fork` fails with `EMFILE` — `Too many open files` — and the failure surfaces
/// on whatever tried: a sub-agent spawn, a log rotation, a database connection.
/// It never says that the descriptor limit is the cause. Measured on
/// 22/11/2026: a sub-agent that could not start, twice, with the backend holding
/// 209 established sockets after a day and a half of uptime.
///
/// Failing to raise it is not a reason to refuse to serve: the inherited limit
/// is kept and the reason is stated once, on stderr.
#[cfg(unix)]
fn raise_fd_limit() {
    // SAFETY: both calls only write into a `rlimit` we own, and `RLIMIT_NOFILE`
    // set to a value read from the kernel on the line before is a valid request.
    unsafe {
        let mut current = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut current) != 0 {
            eprintln!("open files: could not read RLIMIT_NOFILE, keeping the inherited limit");
            return;
        }
        // `rlim_t` is `u64` on every target this server is built for (aarch64 and
        // x86_64, darwin and linux): no cast, and none that could truncate.
        let (soft, hard) = (current.rlim_cur, current.rlim_max);
        let Some(target) = fd_limit_target(soft, hard) else {
            return;
        };
        let wanted = libc::rlimit {
            rlim_cur: target,
            rlim_max: current.rlim_max,
        };
        if libc::setrlimit(libc::RLIMIT_NOFILE, &wanted) != 0 {
            eprintln!(
                "open files: could not raise RLIMIT_NOFILE from {soft} to {target}, keeping {soft}"
            );
            return;
        }
        eprintln!("open files: RLIMIT_NOFILE raised from {soft} to {target}");
    }
}

/// Nothing to raise outside unix: the limit there is not a per-process soft one.
#[cfg(not(unix))]
fn raise_fd_limit() {}

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

    // The server serves HTTP, websockets, database connections and child CLIs:
    // raise the descriptor limit before anything opens one. Deliberately after
    // the `secret` exit above — that path must print nothing, and it opens no
    // socket.
    raise_fd_limit();

    // Load .env file
    dotenvy::dotenv().ok();

    // Configuration first: the log file's location lives in it. Anything it traces
    // while loading goes to stdout through a throwaway subscriber.
    let stdout_only = tracing_subscriber::fmt().with_env_filter(
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "info,project_orchestrator=debug,tower_http=debug".into()),
    );
    let mut config = tracing::subscriber::with_default(stdout_only.finish(), || {
        Config::from_yaml_and_env(cli.config.as_deref())
    })?;

    // Initialize tracing: stdout + the persistent file (held until the process ends).
    let log_guard = project_orchestrator::logging::init(&config.log);

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
            // Dates every (re)start in the log file.
            project_orchestrator::logging::log_startup(&config.log, &log_guard);
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
        Commands::Migrate(MigrateCommand::FilePaths {
            project,
            apply,
            purge_wl,
        }) => run_migrate_file_paths(&config, &project, apply, purge_wl).await,
    }
}

/// `orchestrator migrate file-paths` — the graph only, no Meilisearch, no
/// orchestrator: a migration that is triggered must not start anything else.
async fn run_migrate_file_paths(
    config: &Config,
    project: &str,
    apply: bool,
    purge_wl: bool,
) -> Result<()> {
    use project_orchestrator::neo4j::client::Neo4jClient;
    use project_orchestrator::neo4j::path_migration;

    let client = Neo4jClient::new(
        &config.neo4j_uri,
        &config.neo4j_user,
        &config.neo4j_password,
    )
    .await?;
    let code = path_migration::run(&client, project, apply, purge_wl).await?;
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
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
        // Only the boot skips (this command always configures, as asked).
        Ok(setup_claude::SetupResult::SkippedCliAbsent) => {
            println!("  Claude Code is not installed: nothing was configured.");
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

/// The descriptor limit is raised before anything else runs, so a bad value here
/// costs every later `open` — hence pure, exhaustive tests.
#[cfg(test)]
mod fd_limit_tests {
    use super::*;

    #[test]
    fn a_256_soft_limit_is_raised_to_the_cap() {
        assert_eq!(fd_limit_target(256, u64::MAX), Some(FD_LIMIT_CAP));
    }

    #[test]
    fn the_hard_limit_wins_when_it_is_lower_than_the_cap() {
        assert_eq!(fd_limit_target(256, 1024), Some(1024));
    }

    #[cfg(unix)]
    #[test]
    fn an_unlimited_hard_limit_still_caps_the_request() {
        assert_eq!(
            fd_limit_target(1024, libc::RLIM_INFINITY),
            Some(FD_LIMIT_CAP)
        );
    }

    #[test]
    fn nothing_is_requested_when_the_soft_limit_is_already_enough() {
        assert_eq!(fd_limit_target(FD_LIMIT_CAP, u64::MAX), None);
        assert_eq!(fd_limit_target(FD_LIMIT_CAP + 1, u64::MAX), None);
    }

    #[test]
    fn nothing_is_requested_when_the_hard_limit_is_already_the_soft_one() {
        assert_eq!(fd_limit_target(1024, 1024), None);
    }

    /// The behaviour, not the arithmetic: start from the 256 a macOS shell gives,
    /// call the real function, read the limit back from the kernel.
    #[cfg(unix)]
    #[test]
    fn the_raise_takes_effect_on_the_kernel_limit() {
        unsafe {
            let mut original = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &mut original), 0);
            if original.rlim_max < 4096 {
                // A locked-down box: there is nothing to prove, and lowering the
                // soft limit first would only risk `EMFILE` for the other tests.
                return;
            }
            let low = libc::rlimit {
                rlim_cur: 256,
                rlim_max: original.rlim_max,
            };
            assert_eq!(
                libc::setrlimit(libc::RLIMIT_NOFILE, &low),
                0,
                "could not lower the soft limit to 256 for the test"
            );

            raise_fd_limit();

            let mut after = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            assert_eq!(libc::getrlimit(libc::RLIMIT_NOFILE, &mut after), 0);
            assert!(
                after.rlim_cur > 256,
                "the soft limit stayed at {}: the raise did nothing",
                after.rlim_cur
            );

            // Leave the process as we found it for whatever runs next.
            assert_eq!(libc::setrlimit(libc::RLIMIT_NOFILE, &original), 0);
        }
    }

    #[test]
    fn the_requested_value_never_exceeds_the_hard_limit() {
        for hard in [64u64, 1024, 65_535, 65_536, 65_537, u64::MAX] {
            if let Some(target) = fd_limit_target(0, hard) {
                assert!(target <= hard, "asked {target} with a hard limit of {hard}");
            }
        }
    }
}
