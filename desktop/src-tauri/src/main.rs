// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod docker;
mod plugins;
mod setup;
mod tray;
mod updater;

use project_orchestrator::Config;
use tauri::Manager;
use tauri_plugin_opener::OpenerExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Tauri command: get the server port for the frontend to connect to
#[tauri::command]
fn get_server_port() -> u16 {
    setup::DEFAULT_DESKTOP_PORT
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

/// Tauri command: open a URL in the system's default browser.
///
/// This is the most reliable way to open external links from the webview.
/// Called by ExternalLink onClick in Tauri mode.
#[tauri::command]
fn open_url(app: tauri::AppHandle, url: String) -> Result<(), String> {
    app.opener()
        .open_url(&url, None::<&str>)
        .map_err(|e| e.to_string())
}

/// Tauri command: restart the entire application.
///
/// Called by the setup wizard after generating config.yaml so the backend
/// restarts with the new configuration. This is the simplest and safest
/// approach — the backend thread has no graceful shutdown handle, so we
/// relaunch the whole process.
#[tauri::command]
fn restart_app(app: tauri::AppHandle) {
    tracing::info!("Restarting application (requested by setup wizard)...");
    app.restart();
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
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(docker_manager)
        .invoke_handler(tauri::generate_handler![
            get_server_port,
            check_health,
            open_url,
            restart_app,
            // Splash screen dependency checks
            setup::check_dependencies,
            // Setup wizard commands
            setup::check_config_exists,
            setup::get_config_path,
            setup::generate_config,
            setup::read_config,
            setup::detect_claude_code,
            setup::setup_claude_code,
            setup::verify_oidc_discovery,
            // Docker management commands
            docker::check_docker,
            docker::start_docker_services,
            docker::check_services_health,
            docker::stop_docker_services,
            docker::get_service_logs,
            docker::test_connection,
            // Auto-update commands
            updater::check_update,
            updater::install_update,
            // macOS native rounded corners
            plugins::mac_rounded_corners::enable_rounded_corners,
            plugins::mac_rounded_corners::enable_modern_window_style,
            plugins::mac_rounded_corners::reposition_traffic_lights,
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

            // Resolve absolute paths to bundled resources so the backend can find them.
            if let Ok(resource_dir) = app.path().resource_dir() {
                // Frontend dist/ for HTTP serving.
                // Force SERVE_FRONTEND=true so the backend also serves the SPA via HTTP.
                // This is required for OIDC callback: after OAuth redirect to
                // http://localhost:6600/auth/callback, the backend must serve
                // index.html as SPA fallback (not just the Tauri protocol).
                let dist_path = resource_dir.join("dist");
                if dist_path.exists() {
                    tracing::info!("Frontend dist path: {}", dist_path.display());
                    std::env::set_var("FRONTEND_PATH", dist_path.to_str().unwrap_or("./dist"));
                    std::env::set_var("SERVE_FRONTEND", "true");
                } else {
                    tracing::warn!("Bundled dist/ not found at: {}", dist_path.display());
                }

                // MCP server binary for chat sessions (InteractiveClient / Claude Code CLI)
                let mcp_path = resource_dir.join("mcp_server");
                if mcp_path.exists() {
                    tracing::info!("MCP server path: {}", mcp_path.display());
                    std::env::set_var("MCP_SERVER_PATH", mcp_path.to_str().unwrap_or("mcp_server"));
                } else {
                    tracing::warn!("Bundled mcp_server not found at: {}", mcp_path.display());
                }
            }

            // Create the main window PROGRAMMATICALLY (not from tauri.conf.json)
            // so we can attach an on_new_window handler that intercepts
            // target="_blank" links and opens them in the system browser.
            //
            // NOTE: We intentionally do NOT use on_navigation here because
            // the OIDC/SSO flow navigates the webview to the OAuth provider
            // (e.g. accounts.google.com) and back to localhost:6600/auth/callback.
            // Blocking external navigations would break SSO.
            let new_win_handle = app.handle().clone();
            let _main_window = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            .title("Project Orchestrator")
            .inner_size(1280.0, 800.0)
            .min_inner_size(900.0, 600.0)
            .resizable(true)
            .center()
            .visible(false)
            .decorations(false)
            // Intercept target="_blank" / window.open() — open in system browser
            // instead of creating a new Tauri window. This handles all <a target="_blank">
            // links in the chat UI (PR links, web search results, etc.).
            .on_new_window(move |url, _features| {
                tracing::info!("on_new_window: opening external URL in browser: {}", url);
                let _ = new_win_handle.opener().open_url(url.as_str(), None::<&str>);
                tauri::webview::NewWindowResponse::Deny
            })
            .build()
            .expect("Failed to create main window");

            // Start backend server and manage splash → main window transition
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                // Ensure a config.yaml exists — generate a default one if missing
                let config_path = setup::config_path();
                if !config_path.exists() {
                    tracing::info!("No config.yaml found — generating default (unconfigured) config");
                    if let Err(e) = setup::generate_default_config() {
                        tracing::error!("Failed to generate default config: {}", e);
                        show_main_window(&handle);
                        return;
                    }
                }

                // Load config from the desktop config path
                let config = match Config::from_yaml_and_env(Some(&config_path)) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Failed to load config: {}", e);
                        show_main_window(&handle);
                        return;
                    }
                };

                let port = config.server_port;
                let is_configured = config.setup_completed;

                // Always start the backend server (even in unconfigured mode)
                // In unconfigured mode it runs no-auth so the setup wizard can load
                use std::sync::Arc;
                use std::sync::atomic::{AtomicBool, Ordering};

                let server_started = Arc::new(AtomicBool::new(false));
                let server_started_clone = server_started.clone();

                std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new()
                        .expect("Failed to create tokio runtime");
                    rt.block_on(async {
                        // Signal that the runtime is up and server is about to start
                        server_started_clone.store(true, Ordering::SeqCst);
                        if let Err(e) = project_orchestrator::start_server(config).await {
                            tracing::error!("Server error: {}", e);
                        }
                    });
                });

                // Wait for the tokio runtime to start
                while !server_started.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }

                // Poll /health until the server is listening.
                // The frontend has its own retry logic for DB-dependent endpoints,
                // so we only need to wait for the HTTP listener to be up.
                tracing::info!(
                    "Waiting for backend on port {} (configured: {})...",
                    port,
                    is_configured
                );
                let health_url = format!("http://localhost:{}/health", port);
                let start = std::time::Instant::now();
                loop {
                    if start.elapsed() > std::time::Duration::from_secs(30) {
                        tracing::error!("Backend did not start within 30 seconds");
                        break;
                    }
                    if let Ok(resp) = reqwest::blocking::get(&health_url) {
                        if resp.status().is_success() {
                            tracing::info!(
                                "Backend ready in {:?}",
                                start.elapsed()
                            );
                            break;
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }

                // Transition: close splash → show main window
                // The frontend will check /api/setup-status and redirect to /setup if needed
                show_main_window(&handle);
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Error while building Tauri application");

    app.run(move |_app, event| {
        // macOS: clicking the dock icon when the window is hidden should reshow it
        if let tauri::RunEvent::Reopen {
            has_visible_windows, ..
        } = &event
        {
            if !has_visible_windows {
                tray::show_main_window(&_app);
            }
        }

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
        // On macOS, apply native rounded corners + dark background BEFORE showing
        // the window to avoid any flash of white/square corners.
        // This does the same work as the JS `enableModernWindowStyle()` plugin,
        // but earlier — before the webview JS has a chance to run.
        #[cfg(target_os = "macos")]
        {
            let _ = main_window.with_webview(|webview| {
                unsafe {
                    use cocoa::appkit::{
                        NSColor, NSView, NSWindow, NSWindowStyleMask, NSWindowTitleVisibility,
                    };
                    use cocoa::base::id;
                    use objc::{msg_send, sel, sel_impl};

                    let ns_window = webview.ns_window() as id;

                    // 1. Set dark background color matching --surface-base (#0f1117)
                    let bg_color: id = NSColor::colorWithSRGBRed_green_blue_alpha_(
                        cocoa::base::nil,
                        15.0 / 255.0, // 0x0f
                        17.0 / 255.0, // 0x11
                        23.0 / 255.0, // 0x17
                        1.0,
                    );
                    ns_window.setBackgroundColor_(bg_color);

                    // 2. Apply modern window style with rounded corners
                    let mut style_mask = ns_window.styleMask();
                    style_mask |= NSWindowStyleMask::NSFullSizeContentViewWindowMask;
                    style_mask |= NSWindowStyleMask::NSTitledWindowMask;
                    style_mask |= NSWindowStyleMask::NSClosableWindowMask;
                    style_mask |= NSWindowStyleMask::NSMiniaturizableWindowMask;
                    style_mask |= NSWindowStyleMask::NSResizableWindowMask;
                    ns_window.setStyleMask_(style_mask);

                    // 3. Transparent titlebar that blends with content
                    ns_window.setTitlebarAppearsTransparent_(cocoa::base::YES);
                    ns_window.setTitleVisibility_(
                        NSWindowTitleVisibility::NSWindowTitleHidden,
                    );
                    ns_window.setHasShadow_(cocoa::base::YES);

                    // 4. Set corner radius on the content view layer
                    let content_view = ns_window.contentView();
                    content_view.setWantsLayer(cocoa::base::YES);
                    let layer: id = msg_send![content_view, layer];
                    if !layer.is_null() {
                        let _: () = msg_send![layer, setCornerRadius: 10.0_f64];
                        let _: () =
                            msg_send![layer, setMasksToBounds: cocoa::base::YES];
                    }
                }
            });
        }

        // Inject desktop-specific JS before showing the window:
        // Block Cmd/Ctrl +/-/0 zoom shortcuts to prevent accidental zoom.
        // (Overscroll bounce and external link handling are managed in the
        // frontend CSS and React components via @tauri-apps/plugin-shell.)
        let _ = main_window.eval(r#"
            (function() {
                document.addEventListener('keydown', function(e) {
                    if ((e.metaKey || e.ctrlKey) && (e.key === '+' || e.key === '-' || e.key === '=' || e.key === '0')) {
                        e.preventDefault();
                    }
                }, true);
            })();
        "#);

        if let Err(e) = main_window.show() {
            tracing::warn!("Failed to show main window: {}", e);
        }
        if let Err(e) = main_window.set_focus() {
            tracing::warn!("Failed to focus main window: {}", e);
        }
    }
}
