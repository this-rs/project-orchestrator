//! Auto-update logic using tauri-plugin-updater.
//!
//! - Checks for updates on startup (configurable)
//! - Emits Tauri events for the frontend to display update notifications
//! - Handles download + install with progress reporting

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::UpdaterExt;

// ============================================================================
// Event payloads
// ============================================================================

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateAvailablePayload {
    pub version: String,
    pub body: Option<String>,
    pub date: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateProgressPayload {
    /// Bytes downloaded so far
    pub downloaded: u64,
    /// Total bytes (if known)
    pub total: Option<u64>,
    /// Progress percentage (0-100), None if total is unknown
    pub percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateErrorPayload {
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInstallingPayload {
    pub version: String,
}

// ============================================================================
// Update check (called on startup)
// ============================================================================

/// Check for updates in the background and emit events to the frontend.
pub fn check_for_updates(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        tracing::info!("Checking for updates...");

        let updater = match app.updater() {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!("Failed to initialize updater: {}", e);
                return;
            }
        };

        let update = match updater.check().await {
            Ok(Some(update)) => update,
            Ok(None) => {
                tracing::info!("No update available — already on latest version");
                return;
            }
            Err(e) => {
                tracing::warn!("Update check failed: {}", e);
                let _ = app.emit(
                    "update-error",
                    UpdateErrorPayload {
                        message: format!("Update check failed: {}", e),
                    },
                );
                return;
            }
        };

        let version = update.version.clone();
        let body = update.body.clone();
        let date = update.date.map(|d| d.to_string());

        tracing::info!("Update available: v{}", version);

        // Emit update-available event for the frontend
        let _ = app.emit(
            "update-available",
            UpdateAvailablePayload {
                version,
                body,
                date,
            },
        );

        // Store the update in app state so the install command can access it
        app.manage(AvailableUpdate(std::sync::Mutex::new(Some(update))));
    });
}

// ============================================================================
// State: holds the pending update
// ============================================================================

struct AvailableUpdate(std::sync::Mutex<Option<tauri_plugin_updater::Update>>);

// ============================================================================
// Tauri commands
// ============================================================================

/// Manually trigger an update check from the frontend.
#[tauri::command]
pub async fn check_update(app: AppHandle) -> Result<Option<UpdateAvailablePayload>, String> {
    let updater = app.updater().map_err(|e| format!("Updater init failed: {}", e))?;

    match updater.check().await {
        Ok(Some(update)) => {
            let payload = UpdateAvailablePayload {
                version: update.version.clone(),
                body: update.body.clone(),
                date: update.date.map(|d| d.to_string()),
            };

            // Store for later install
            if let Some(state) = app.try_state::<AvailableUpdate>() {
                if let Ok(mut guard) = state.0.lock() {
                    *guard = Some(update);
                }
            } else {
                app.manage(AvailableUpdate(std::sync::Mutex::new(Some(update))));
            }

            Ok(Some(payload))
        }
        Ok(None) => Ok(None),
        Err(e) => Err(format!("Update check failed: {}", e)),
    }
}

/// Download and install the pending update.
/// Emits "update-progress" events during download and "update-installing" before restart.
#[tauri::command]
pub async fn install_update(app: AppHandle) -> Result<(), String> {
    let update = {
        let state = app
            .try_state::<AvailableUpdate>()
            .ok_or("No update available")?;
        let mut guard = state.0.lock().map_err(|e| format!("Lock error: {}", e))?;
        guard.take().ok_or("No pending update to install")?
    };

    let version = update.version.clone();
    let app_clone = app.clone();

    tracing::info!("Downloading update v{}...", version);

    // Download with progress tracking
    let mut downloaded: u64 = 0;

    update
        .download_and_install(
            |chunk_len, content_length| {
                downloaded += chunk_len as u64;
                let percent = content_length.map(|t| {
                    if t > 0 {
                        (downloaded as f64 / t as f64 * 100.0).min(100.0)
                    } else {
                        0.0
                    }
                });

                let _ = app_clone.emit(
                    "update-progress",
                    UpdateProgressPayload {
                        downloaded,
                        total: content_length,
                        percent,
                    },
                );
            },
            || {
                tracing::info!("Update v{} downloaded — installing...", version);
                let _ = app_clone.emit(
                    "update-installing",
                    UpdateInstallingPayload {
                        version: version.clone(),
                    },
                );
            },
        )
        .await
        .map_err(|e| format!("Download/install failed: {}", e))?;

    tracing::info!("Update installed — restarting application");

    // Restart the app to apply the update
    app.restart();
}
