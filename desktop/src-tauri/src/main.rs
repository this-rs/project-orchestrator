// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod docker;
mod setup;
mod tray;
mod updater;

use project_orchestrator::Config;
use tauri::Manager;
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

    // Initialize Docker manager (shared state for Tauri commands)
    let docker_manager = docker::create_docker_manager();

    // Clone docker manager for the exit handler
    let docker_for_exit = docker_manager.clone();

    // Launch Tauri application (splash screen shows immediately)
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(docker_manager)
        .invoke_handler(tauri::generate_handler![
            get_server_port,
            check_health,
            // Setup wizard commands
            setup::check_config_exists,
            setup::get_config_path,
            setup::generate_config,
            setup::detect_claude_code,
            setup::setup_claude_code,
            // Docker management commands
            docker::check_docker,
            docker::start_docker_services,
            docker::check_services_health,
            docker::stop_docker_services,
            docker::get_service_logs,
            // Auto-update commands
            updater::check_update,
            updater::install_update,
        ])
        .setup(|app| {
            // Create system tray
            if let Err(e) = tray::create_tray(app.handle()) {
                tracing::warn!("Failed to create system tray: {}", e);
            }
            // Set up minimize-to-tray behavior
            tray::setup_minimize_to_tray(app.handle());
            // Check for updates in background
            updater::check_for_updates(app.handle().clone());

            // Start backend server and manage splash → main window transition
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let has_config = setup::check_config_exists();

                if has_config {
                    // Load config and start server
                    let config = match Config::from_env() {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!("Failed to load config: {}", e);
                            // Close splash, show main (which will display /setup or error)
                            show_main_window(&handle);
                            return;
                        }
                    };

                    let port = config.server_port;

                    // Start the backend server
                    std::thread::spawn(move || {
                        let rt = tokio::runtime::Runtime::new()
                            .expect("Failed to create tokio runtime");
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
                            tracing::error!(
                                "Backend server did not start within 30 seconds"
                            );
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
                    // Small delay so splash is visible briefly
                    std::thread::sleep(std::time::Duration::from_millis(800));
                }

                // Transition: close splash → show main window
                show_main_window(&handle);
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Error while building Tauri application");

    app.run(move |_app, event| {
        if let tauri::RunEvent::ExitRequested { .. } = event {
            tracing::info!("Application exiting — stopping Docker services...");
            let dm = docker_for_exit.clone();
            // Spawn a blocking task to stop containers
            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let mgr = dm.read().await;
                    if let Err(e) = mgr.stop_services().await {
                        tracing::warn!("Failed to stop Docker services on exit: {}", e);
                    }
                });
            })
            .join()
            .ok();
        }
    });
}

/// Close the splash screen and show the main application window.
fn show_main_window(handle: &tauri::AppHandle) {
    // Close splash screen
    if let Some(splash) = handle.get_webview_window("splashscreen") {
        if let Err(e) = splash.close() {
            tracing::warn!("Failed to close splash screen: {}", e);
        }
    }

    // Show main window
    if let Some(main_window) = handle.get_webview_window("main") {
        if let Err(e) = main_window.show() {
            tracing::warn!("Failed to show main window: {}", e);
        }
        if let Err(e) = main_window.set_focus() {
            tracing::warn!("Failed to focus main window: {}", e);
        }
    }
}
