//! System tray (menu bar on macOS) with service status and controls.

use crate::docker::SharedDockerManager;
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager};

/// Create and configure the system tray icon with menu.
pub fn create_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    // Build menu items (we keep references for the status items)
    let title_item = MenuItemBuilder::with_id("title", "Project Orchestrator")
        .enabled(false)
        .build(app)?;
    let neo4j_item = MenuItemBuilder::with_id("neo4j-status", "Neo4j: ... Checking")
        .enabled(false)
        .build(app)?;
    let meili_item = MenuItemBuilder::with_id("meili-status", "MeiliSearch: ... Checking")
        .enabled(false)
        .build(app)?;
    let api_item = MenuItemBuilder::with_id("api-status", "API Server: ... Checking")
        .enabled(false)
        .build(app)?;
    let open_item = MenuItemBuilder::with_id("open", "Open").build(app)?;
    let reconfig_item = MenuItemBuilder::with_id("reconfigure", "Reconfigure...").build(app)?;
    let quit_item = MenuItemBuilder::with_id("quit", "Quit").build(app)?;

    // Build the tray menu
    let menu = MenuBuilder::new(app)
        .item(&title_item)
        .separator()
        .item(&neo4j_item)
        .item(&meili_item)
        .item(&api_item)
        .separator()
        .item(&open_item)
        .item(&reconfig_item)
        .separator()
        .item(&quit_item)
        .build()?;

    // Load the tray icon from the embedded PNG
    let icon = app.default_window_icon().cloned().unwrap_or_else(|| {
        tauri::image::Image::from_bytes(include_bytes!("../icons/32x32.png"))
            .expect("Failed to load tray icon")
    });

    // Build the tray
    let _tray = TrayIconBuilder::new()
        .icon(icon)
        .menu(&menu)
        .tooltip("Project Orchestrator")
        .on_menu_event(move |app, event| {
            handle_menu_event(app, event.id().as_ref());
        })
        .on_tray_icon_event(|tray, event| {
            if let tauri::tray::TrayIconEvent::DoubleClick { .. } = event {
                show_main_window(tray.app_handle());
            }
        })
        .build(app)?;

    // Start background status polling with the menu item references
    start_status_polling(app.clone(), neo4j_item, meili_item, api_item);

    Ok(())
}

/// Handle tray menu item clicks.
///
/// Any menu item that navigates the frontend must include `?from=tray` in the
/// URL so that the frontend route guards (SetupGuard, etc.) respect the user's
/// intended destination instead of auto-redirecting.
fn handle_menu_event(app: &AppHandle, item_id: &str) {
    match item_id {
        "open" => {
            show_main_window(app);
        }
        "reconfigure" => {
            if let Some(window) = app.get_webview_window("main") {
                window.show().ok();
                window.set_focus().ok();
                let _ = window.eval("window.location.href = '/setup?from=tray'");
            }
        }
        "quit" => {
            tracing::info!("Quit requested from tray");
            let app_clone = app.clone();
            std::thread::spawn(move || {
                // Only stop Docker containers in managed "docker" mode.
                // In "external" mode, the user's services must not be touched.
                let infra_mode = std::fs::read_to_string(crate::setup::config_path())
                    .ok()
                    .and_then(|c| serde_yaml::from_str::<serde_yaml::Value>(&c).ok())
                    .and_then(|v| v.get("infra_mode").and_then(|m| m.as_str().map(String::from)))
                    .unwrap_or_else(|| "docker".to_string());

                if infra_mode != "external" {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        if let Some(dm) = app_clone.try_state::<SharedDockerManager>() {
                            let mgr = dm.read().await;
                            if let Err(e) = mgr.stop_services().await {
                                tracing::warn!("Failed to stop Docker services: {}", e);
                            }
                        }
                    });
                } else {
                    tracing::info!("Skipping Docker stop — infra_mode is external");
                }
                app_clone.exit(0);
            });
        }
        _ => {}
    }
}

/// Show and focus the main window.
pub fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        window.show().ok();
        window.set_focus().ok();
        window.unminimize().ok();
    }
}

/// Start a background task that polls service health and updates tray menu items.
fn start_status_polling(
    app: AppHandle,
    neo4j_item: tauri::menu::MenuItem<tauri::Wry>,
    meili_item: tauri::menu::MenuItem<tauri::Wry>,
    api_item: tauri::menu::MenuItem<tauri::Wry>,
) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;

                // Check Docker services
                let (neo4j_text, meili_text) =
                    if let Some(dm) = app.try_state::<SharedDockerManager>() {
                        let mgr = dm.read().await;
                        let health = mgr.check_health(true).await;
                        (
                            format_status("Neo4j", &health.neo4j),
                            format_status("MeiliSearch", &health.meilisearch),
                        )
                    } else {
                        ("Neo4j: ? Unknown".into(), "MeiliSearch: ? Unknown".into())
                    };

                // Check API server
                let api_text = match reqwest::get("http://127.0.0.1:8080/health").await {
                    Ok(resp) if resp.status().is_success() => {
                        "API Server: OK Healthy".to_string()
                    }
                    _ => "API Server: X Down".to_string(),
                };

                // Update menu item texts
                neo4j_item.set_text(&neo4j_text).ok();
                meili_item.set_text(&meili_text).ok();
                api_item.set_text(&api_text).ok();
            }
        });
    });
}

fn format_status(name: &str, status: &crate::docker::ServiceStatus) -> String {
    match status {
        crate::docker::ServiceStatus::Healthy => format!("{}: OK Healthy", name),
        crate::docker::ServiceStatus::Starting => format!("{}: ... Starting", name),
        crate::docker::ServiceStatus::Unhealthy => format!("{}: X Unhealthy", name),
        crate::docker::ServiceStatus::NotRunning => format!("{}: X Not Running", name),
        crate::docker::ServiceStatus::Disabled => format!("{}: - Disabled", name),
        crate::docker::ServiceStatus::Unknown => format!("{}: ? Unknown", name),
    }
}

/// Set up the "minimize to tray" behavior: closing the window hides it instead of quitting.
pub fn setup_minimize_to_tray(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let window_clone = window.clone();
        window.on_window_event(move |event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Prevent the window from actually closing — hide it instead
                api.prevent_close();
                window_clone.hide().ok();
            }
        });
    }
}
