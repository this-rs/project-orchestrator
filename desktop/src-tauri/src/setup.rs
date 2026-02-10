//! Setup wizard Tauri commands.
//!
//! These commands are called by the frontend setup wizard to:
//! - Check if a config.yaml already exists
//! - Generate a config.yaml from wizard input
//! - Detect if Claude Code CLI is installed

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration received from the frontend setup wizard.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupConfig {
    // Infrastructure
    pub infra_mode: String,
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub meilisearch_url: String,
    pub meilisearch_key: String,
    pub server_port: u16,

    // Auth
    pub auth_mode: String,
    pub root_email: String,
    pub root_password: String,
    pub oidc_discovery_url: String,
    pub oidc_client_id: String,
    pub oidc_client_secret: String,

    // Chat
    pub chat_model: String,
    pub chat_max_sessions: u32,
}

/// YAML-serializable config structure (matches backend config.yaml format).
#[derive(Debug, Serialize)]
struct YamlOutput {
    server: ServerSection,
    neo4j: Neo4jSection,
    meilisearch: MeilisearchSection,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat: Option<ChatSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    auth: Option<AuthSection>,
}

#[derive(Debug, Serialize)]
struct ServerSection {
    port: u16,
    workspace_path: String,
    serve_frontend: bool,
    frontend_path: String,
}

#[derive(Debug, Serialize)]
struct Neo4jSection {
    uri: String,
    user: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct MeilisearchSection {
    url: String,
    key: String,
}

#[derive(Debug, Serialize)]
struct ChatSection {
    default_model: String,
    max_sessions: u32,
}

#[derive(Debug, Serialize)]
struct AuthSection {
    jwt_secret: String,
    jwt_expiry_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_account: Option<RootAccountSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oidc: Option<OidcSection>,
}

#[derive(Debug, Serialize)]
struct RootAccountSection {
    email: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct OidcSection {
    provider_name: String,
    discovery_url: String,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
}

// ============================================================================
// Helpers
// ============================================================================

/// Get the path to config.yaml (next to the executable or in CWD).
fn config_path() -> PathBuf {
    // Prefer a well-known location in the user's app data
    if let Some(dir) = dirs_next_config_dir() {
        return dir.join("config.yaml");
    }
    // Fallback: current working directory
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("config.yaml")
}

/// Platform-specific config directory.
fn dirs_next_config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        dirs::config_dir().map(|d| d.join("project-orchestrator"))
    }
    #[cfg(target_os = "linux")]
    {
        dirs::config_dir().map(|d| d.join("project-orchestrator"))
    }
    #[cfg(target_os = "windows")]
    {
        dirs::config_dir().map(|d| d.join("ProjectOrchestrator"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Generate a random alphanumeric string of the given length.
fn random_secret(len: usize) -> String {
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| {
            let idx = rng.gen_range(0..62);
            let c = match idx {
                0..=9 => (b'0' + idx) as char,
                10..=35 => (b'a' + idx - 10) as char,
                _ => (b'A' + idx - 36) as char,
            };
            c
        })
        .collect()
}

// ============================================================================
// Tauri commands
// ============================================================================

/// Check if a config.yaml already exists.
#[tauri::command]
pub fn check_config_exists() -> bool {
    let path = config_path();
    tracing::info!("Checking config at: {}", path.display());
    path.exists()
}

/// Get the config file path (for display in the UI).
#[tauri::command]
pub fn get_config_path() -> String {
    config_path().display().to_string()
}

/// Generate a config.yaml from the wizard configuration.
///
/// - Generates random secrets (JWT, MeiliSearch key) if not provided
/// - Creates parent directories if needed
/// - Writes the YAML file
#[tauri::command]
pub fn generate_config(config: SetupConfig) -> Result<String, String> {
    let path = config_path();
    tracing::info!("Generating config at: {}", path.display());

    // Build YAML structure
    let neo4j_password = if config.neo4j_password.is_empty() {
        random_secret(24)
    } else {
        config.neo4j_password.clone()
    };

    let meilisearch_key = if config.meilisearch_key.is_empty() {
        random_secret(32)
    } else {
        config.meilisearch_key.clone()
    };

    let auth = match config.auth_mode.as_str() {
        "password" => Some(AuthSection {
            jwt_secret: random_secret(48),
            jwt_expiry_secs: 28800, // 8h
            root_account: Some(RootAccountSection {
                email: config.root_email.clone(),
                password: config.root_password.clone(),
            }),
            oidc: None,
        }),
        "oidc" => Some(AuthSection {
            jwt_secret: random_secret(48),
            jwt_expiry_secs: 28800,
            root_account: None,
            oidc: Some(OidcSection {
                provider_name: "OIDC".into(),
                discovery_url: config.oidc_discovery_url.clone(),
                client_id: config.oidc_client_id.clone(),
                client_secret: config.oidc_client_secret.clone(),
                redirect_uri: format!(
                    "http://localhost:{}/auth/callback",
                    config.server_port
                ),
            }),
        }),
        _ => None, // "none" — no auth section
    };

    let yaml = YamlOutput {
        server: ServerSection {
            port: config.server_port,
            workspace_path: ".".into(),
            serve_frontend: true,
            frontend_path: "./dist".into(),
        },
        neo4j: Neo4jSection {
            uri: config.neo4j_uri.clone(),
            user: config.neo4j_user.clone(),
            password: neo4j_password,
        },
        meilisearch: MeilisearchSection {
            url: config.meilisearch_url.clone(),
            key: meilisearch_key,
        },
        chat: Some(ChatSection {
            default_model: config.chat_model.clone(),
            max_sessions: config.chat_max_sessions,
        }),
        auth,
    };

    // Serialize
    let yaml_str = serde_yaml::to_string(&yaml).map_err(|e| format!("YAML error: {}", e))?;

    // Add header comment
    let output = format!(
        "# =============================================================================\n\
         # Project Orchestrator — Configuration\n\
         # Generated by Setup Wizard on {}\n\
         # =============================================================================\n\n\
         {}",
        chrono_now(),
        yaml_str
    );

    // Create parent directory
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    // Write file
    std::fs::write(&path, &output)
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;

    tracing::info!("Config written to: {}", path.display());
    Ok(path.display().to_string())
}

/// Detect if Claude Code CLI is available on this machine.
#[tauri::command]
pub fn detect_claude_code() -> Result<bool, String> {
    // Try `which claude` on Unix or `where claude` on Windows
    #[cfg(unix)]
    let result = std::process::Command::new("which")
        .arg("claude")
        .output();

    #[cfg(windows)]
    let result = std::process::Command::new("where")
        .arg("claude")
        .output();

    match result {
        Ok(output) => {
            let found = output.status.success();
            if found {
                let path = String::from_utf8_lossy(&output.stdout);
                tracing::info!("Claude Code CLI found at: {}", path.trim());
            } else {
                tracing::info!("Claude Code CLI not found");
            }
            Ok(found)
        }
        Err(e) => {
            tracing::warn!("Failed to detect Claude Code: {}", e);
            Ok(false)
        }
    }
}

/// Simple timestamp without pulling in chrono.
fn chrono_now() -> String {
    // Use std::time for a basic ISO-ish timestamp
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Basic formatting: days since epoch → approximate date
    // For a proper date we'd need chrono, but this is good enough for a comment
    format!("unix:{}", secs)
}
