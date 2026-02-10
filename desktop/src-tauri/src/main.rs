// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod setup;

use project_orchestrator::Config;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Tauri command: get the server port for the frontend to connect to
#[tauri::command]
fn get_server_port() -> u16 {
    // TODO: read from config
    8080
}

/// Tauri command: check if the backend is healthy
#[tauri::command]
async fn check_health(port: u16) -> Result<bool, String> {
    let url = format!("http://localhost:{}/health", port);
    match reqwest::get(&url).await {
        Ok(resp) => Ok(resp.status().is_success()),
        Err(_) => Ok(false),
    }
}

fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,project_orchestrator=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Project Orchestrator Desktop...");

    // Check if this is first launch (no config.yaml)
    let has_config = setup::check_config_exists();

    if has_config {
        // Config exists — load and start the backend server immediately
        let config = match Config::from_env() {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    "Failed to load config: {}. Please check your config.yaml or env vars.",
                    e
                );
                std::process::exit(1);
            }
        };

        let port = config.server_port;

        // Start the backend server in a background thread
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
            rt.block_on(async {
                if let Err(e) = project_orchestrator::start_server(config).await {
                    tracing::error!("Server error: {}", e);
                }
            });
        });

        // Wait for the server to be ready
        tracing::info!("Waiting for backend server on port {}...", port);
        let health_url = format!("http://localhost:{}/health", port);
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > std::time::Duration::from_secs(30) {
                tracing::error!("Backend server did not start within 30 seconds");
                break;
            }
            if let Ok(resp) = reqwest::blocking::get(&health_url) {
                if resp.status().is_success() {
                    tracing::info!("Backend server is ready!");
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    } else {
        tracing::info!("No config.yaml found — launching setup wizard");
        // The frontend will detect the absence of config and show /setup
    }

    // Launch Tauri application
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            get_server_port,
            check_health,
            setup::check_config_exists,
            setup::get_config_path,
            setup::generate_config,
            setup::detect_claude_code,
        ])
        .run(tauri::generate_context!())
        .expect("Error while running Tauri application");
}
