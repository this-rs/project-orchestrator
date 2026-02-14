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

/// Tauri command: check backend health and return the full `/health` response.
///
/// Returns the JSON body from the `/health` endpoint, which includes:
/// - `status`: "ok" | "degraded" | "unhealthy"
/// - `version`: semver string
/// - `services.neo4j`: "connected" | "disconnected"
/// - `services.meilisearch`: "connected" | "disconnected"
///
/// Returns `null` if the server is not reachable yet.
#[tauri::command]
async fn check_health(port: u16) -> Result<Option<serde_json::Value>, String> {
    let url = format!("http://localhost:{}/health", port);
    match reqwest::get(&url).await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(body) => Ok(Some(body)),
            Err(_) => Ok(Some(serde_json::json!({"status": "ok"}))),
        },
        Err(_) => Ok(None),
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

/// Tauri command: open a native directory picker dialog.
///
/// Returns the selected directory path, or `None` if the user cancelled.
/// This uses `tauri_plugin_dialog` under the hood but is exposed as a
/// custom command so the frontend can call it via `invoke()` — which is
/// more reliable than the JS plugin bindings (no build-time IPC glue needed).
#[tauri::command]
async fn pick_directory(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog()
        .file()
        .set_title("Select project folder")
        .pick_folder(move |path| {
            let _ = tx.send(path.map(|p| p.to_string()));
        });
    rx.recv()
        .map_err(|e| format!("Dialog channel error: {}", e))
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

    // Clone docker manager for the exit handler and for the splash→main transition
    let docker_for_exit = docker_manager.clone();
    let docker_for_setup = docker_manager.clone();

    // Launch Tauri application (splash screen shows immediately)
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(docker_manager)
        .invoke_handler(tauri::generate_handler![
            get_server_port,
            check_health,
            open_url,
            pick_directory,
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
            docker::open_docker_desktop,
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
        .setup(move |app| {
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
            let docker_for_transition = docker_for_setup.clone();
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

                // Read infra_mode from the YAML config (defaults to "docker")
                let infra_mode = std::fs::read_to_string(&config_path)
                    .ok()
                    .and_then(|contents| serde_yaml::from_str::<serde_yaml::Value>(&contents).ok())
                    .and_then(|v| v.get("infra_mode").and_then(|m| m.as_str().map(String::from)))
                    .unwrap_or_else(|| "docker".to_string());

                // Read nats_enabled from the YAML config
                let nats_enabled = std::fs::read_to_string(&config_path)
                    .ok()
                    .and_then(|contents| serde_yaml::from_str::<serde_yaml::Value>(&contents).ok())
                    .map(|v| v.get("nats").and_then(|n| n.get("url")).is_some())
                    .unwrap_or(false);

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
                tracing::info!(
                    "Waiting for backend on port {} (configured: {}, infra: {})...",
                    port,
                    is_configured,
                    infra_mode
                );
                let health_url = format!("http://localhost:{}/health", port);
                let start = std::time::Instant::now();
                let global_timeout = std::time::Duration::from_secs(120);

                loop {
                    if start.elapsed() > std::time::Duration::from_secs(30) {
                        tracing::error!("Backend HTTP listener did not start within 30 seconds");
                        break;
                    }
                    // We only check that the HTTP listener is up (any response),
                    // NOT that services are connected. /health may return 503 when
                    // Neo4j is still starting — that's fine, the Docker health loop
                    // below handles service readiness.
                    if reqwest::blocking::get(&health_url).is_ok() {
                        tracing::info!(
                            "Backend HTTP ready in {:?}",
                            start.elapsed()
                        );
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }

                // After backend HTTP is up, wait for Docker services if applicable.
                // In setup mode or external mode, skip this — go straight to main window.
                if is_configured && infra_mode != "external" {
                    tracing::info!("Waiting for Docker services to become healthy...");
                    let rt = tokio::runtime::Runtime::new()
                        .expect("Failed to create tokio runtime for Docker health check");
                    rt.block_on(async {
                        let services_start = std::time::Instant::now();
                        loop {
                            if services_start.elapsed() > global_timeout {
                                tracing::warn!(
                                    "Docker services did not become healthy within {:?} — proceeding anyway",
                                    global_timeout
                                );
                                break;
                            }

                            let mgr = docker_for_transition.read().await;
                            let health = mgr.check_health(nats_enabled).await;
                            drop(mgr); // release lock quickly

                            let neo4j_ok = health.neo4j == docker::ServiceStatus::Healthy;
                            let meili_ok = health.meilisearch == docker::ServiceStatus::Healthy;
                            let nats_ok = !nats_enabled
                                || health.nats == docker::ServiceStatus::Healthy
                                || health.nats == docker::ServiceStatus::Disabled;

                            if neo4j_ok && meili_ok && nats_ok {
                                tracing::info!(
                                    "All Docker services healthy in {:?}",
                                    services_start.elapsed()
                                );
                                break;
                            }

                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        }
                    });
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
        #[cfg(target_os = "macos")]
        if let tauri::RunEvent::Reopen {
            has_visible_windows, ..
        } = &event
        {
            if !has_visible_windows {
                tray::show_main_window(&_app);
            }
        }

        if let tauri::RunEvent::ExitRequested { .. } = event {
            // Only stop Docker containers if infra_mode is "docker" (managed mode).
            // In "external" mode the user manages their own services — we must NOT
            // stop containers that the app didn't start.
            let infra_mode = std::fs::read_to_string(setup::config_path())
                .ok()
                .and_then(|c| serde_yaml::from_str::<serde_yaml::Value>(&c).ok())
                .and_then(|v| v.get("infra_mode").and_then(|m| m.as_str().map(String::from)))
                .unwrap_or_else(|| "docker".to_string());

            if infra_mode == "external" {
                tracing::info!("Application exiting — infra_mode is external, skipping Docker stop");
            } else {
                tracing::info!("Application exiting — stopping Docker services...");
                let dm = docker_for_exit.clone();
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
                    use cocoa::foundation::NSString;
                    use objc::{class, msg_send, sel, sel_impl};

                    let ns_window = webview.ns_window() as id;

                    // 1. Force dark appearance on the window.
                    //    This makes the webview report prefers-color-scheme: dark,
                    //    so external pages (Google SSO, etc.) render in dark mode.
                    //    It also prevents white flashes during page transitions.
                    let dark_appearance: id = msg_send![
                        class!(NSAppearance),
                        appearanceNamed: cocoa::foundation::NSString::alloc(cocoa::base::nil)
                            .init_str("NSAppearanceNameDarkAqua")
                    ];
                    if !dark_appearance.is_null() {
                        let _: () = msg_send![ns_window, setAppearance: dark_appearance];
                    }

                    // 2. Set dark background color matching --surface-base (#0f1117)
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

                    // 4. Set dark underPageBackgroundColor on the WKWebView.
                    //    This is the color shown behind page content — visible
                    //    during overscroll, navigation transitions, and before a
                    //    page paints. Setting it to dark prevents white flashes
                    //    when navigating to external SSO pages.
                    let wk_webview = webview.inner() as id;
                    let _: () = msg_send![wk_webview, setUnderPageBackgroundColor: bg_color];

                    // 5. Set corner radius on the content view layer
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
