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
    #[serde(default)]
    pub serve_frontend: bool,
    /// Public URL for reverse-proxy setups (e.g. https://po.ffs.dev).
    /// Used for frontend_url, redirect_uri, and CORS when non-empty.
    #[serde(default)]
    pub public_url: String,

    // Auth
    pub auth_mode: String,
    pub root_email: String,
    pub root_password: String,
    /// Selected OIDC provider key (google, microsoft, okta, auth0, keycloak, custom).
    #[serde(default)]
    pub oidc_provider: String,
    pub oidc_discovery_url: String,
    pub oidc_client_id: String,
    pub oidc_client_secret: String,
    #[serde(default)]
    pub oidc_provider_name: String,
    #[serde(default)]
    pub oidc_scopes: String,
    /// Resolved OIDC endpoints (from frontend Verify button or previous config).
    /// When non-empty, these are used directly — skipping the network fetch in resolve_oidc_endpoints.
    #[serde(default)]
    pub oidc_auth_endpoint: String,
    #[serde(default)]
    pub oidc_token_endpoint: String,
    #[serde(default)]
    pub oidc_userinfo_endpoint: String,

    // Access restrictions
    #[serde(default)]
    pub allowed_email_domain: String,
    #[serde(default)]
    pub allowed_emails: String, // newline-separated list of emails

    // NATS
    #[serde(default)]
    pub nats_url: String,
    /// Whether NATS is enabled (controls Docker container and YAML section).
    #[serde(default = "default_true")]
    pub nats_enabled: bool,

    // Chat
    pub chat_model: String,
    pub chat_max_sessions: u32,
    #[serde(default = "default_max_turns")]
    pub chat_max_turns: u32,
    /// Permission mode: "default", "acceptEdits", "plan", "bypassPermissions"
    #[serde(default = "default_permission_mode")]
    pub chat_permission_mode: String,
}

fn default_max_turns() -> u32 {
    50
}

fn default_permission_mode() -> String {
    "bypassPermissions".into()
}

fn default_true() -> bool {
    true
}

/// YAML-serializable config structure (matches backend config.yaml format).
#[derive(Debug, Serialize)]
struct YamlOutput {
    /// When false, the frontend shows the setup wizard instead of the main app.
    setup_completed: bool,
    /// Persisted so that reconfigure mode can restore the exact wizard state.
    infra_mode: String,
    server: ServerSection,
    neo4j: Neo4jSection,
    meilisearch: MeilisearchSection,
    #[serde(skip_serializing_if = "Option::is_none")]
    nats: Option<NatsSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat: Option<ChatSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    auth: Option<AuthSection>,
}

#[derive(Debug, Serialize)]
struct NatsSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
}

#[derive(Debug, Serialize)]
struct ServerSection {
    port: u16,
    workspace_path: String,
    serve_frontend: bool,
    frontend_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    public_url: Option<String>,
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
    max_turns: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    permissions: Option<ChatPermissionsSection>,
}

#[derive(Debug, Serialize)]
struct ChatPermissionsSection {
    mode: String,
}

#[derive(Debug, Serialize)]
struct AuthSection {
    jwt_secret: String,
    jwt_expiry_secs: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    frontend_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_email_domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_emails: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_account: Option<RootAccountSection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    oidc: Option<OidcSection>,
}

#[derive(Debug, Serialize)]
struct RootAccountSection {
    email: String,
    name: String,
    password_hash: String,
}

#[derive(Debug, Serialize)]
struct OidcSection {
    provider_name: String,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    /// Provider key for the frontend wizard (google, microsoft, okta, auth0, keycloak, custom).
    /// Persisted so reconfigure mode can restore the exact provider selection.
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    discovery_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    auth_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    userinfo_endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scopes: Option<String>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Default port for the desktop app (avoids conflicts with dev 8080/3002).
pub const DEFAULT_DESKTOP_PORT: u16 = 6600;

/// Get the path to config.yaml (next to the executable or in CWD).
/// Public so main.rs can use it to load config from the right location.
pub fn config_path() -> PathBuf {
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
            match idx {
                0..=9 => (b'0' + idx) as char,
                10..=35 => (b'a' + idx - 10) as char,
                _ => (b'A' + idx - 36) as char,
            }
        })
        .collect()
}

// ============================================================================
// Dependency checks (splash screen)
// ============================================================================

/// Result of checking all desktop dependencies at once.
/// Used by the splash screen to show a checklist.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DependencyStatus {
    /// Backward compat: true when Docker daemon is reachable.
    pub docker_available: bool,
    /// Fine-grained Docker status: "running", "installed", or "not_installed".
    pub docker_status: String,
    pub claude_code_available: bool,
    pub config_exists: bool,
    /// Whether setup has been completed (from config.yaml).
    pub setup_completed: bool,
    /// Infrastructure mode: "docker" or "external" (from config.yaml, defaults to "docker").
    pub infra_mode: String,
    /// Whether NATS is enabled (from config.yaml, defaults to true).
    pub nats_enabled: bool,
    /// Neo4j password from config.yaml (needed by start_docker_services).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neo4j_password: Option<String>,
    /// Meilisearch key from config.yaml (needed by start_docker_services).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meilisearch_key: Option<String>,
    pub os: String,
    pub arch: String,
}

/// Check all dependencies in one call (for the splash screen).
///
/// This avoids multiple round-trips between JS and Rust.
/// Returns fine-grained Docker status, infra_mode, credentials (for start_docker_services),
/// and other dependency info.
#[tauri::command]
pub async fn check_dependencies(
    docker: tauri::State<'_, crate::docker::SharedDockerManager>,
) -> Result<DependencyStatus, String> {
    // Docker check — fine-grained status (async via bollard)
    let docker_status = {
        let mgr = docker.read().await;
        mgr.status().await
    };
    let docker_available = docker_status == crate::docker::DockerStatus::Running;

    // Claude Code CLI check
    let claude_code_available =
        project_orchestrator::setup_claude::detect_claude_cli().is_some();

    // Config file check + read config for infra_mode, credentials, etc.
    let config_path = config_path();
    let config_exists = config_path.exists();

    let (setup_completed, infra_mode, nats_enabled, neo4j_password, meilisearch_key) =
        if config_exists {
            match std::fs::read_to_string(&config_path) {
                Ok(contents) => {
                    match serde_yaml::from_str::<project_orchestrator::YamlConfig>(&contents) {
                        Ok(yaml) => {
                            let setup = yaml.setup_completed;
                            let mode = yaml
                                .infra_mode
                                .clone()
                                .unwrap_or_else(|| "docker".to_string());
                            let nats = yaml.nats.url.is_some();
                            let neo4j_pw = if yaml.neo4j.password.is_empty() {
                                None
                            } else {
                                Some(yaml.neo4j.password.clone())
                            };
                            let meili_key = if yaml.meilisearch.key.is_empty() {
                                None
                            } else {
                                Some(yaml.meilisearch.key.clone())
                            };
                            (setup, mode, nats, neo4j_pw, meili_key)
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse config.yaml: {}", e);
                            (false, "docker".to_string(), true, None, None)
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read config.yaml: {}", e);
                    (false, "docker".to_string(), true, None, None)
                }
            }
        } else {
            (false, "docker".to_string(), true, None, None)
        };

    // Platform info
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    Ok(DependencyStatus {
        docker_available,
        docker_status: docker_status.to_string(),
        claude_code_available,
        config_exists,
        setup_completed,
        infra_mode,
        nats_enabled,
        neo4j_password,
        meilisearch_key,
        os,
        arch,
    })
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

    // ── Read existing config for secret preservation (reconfigure mode) ──
    let existing: Option<project_orchestrator::YamlConfig> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|contents| serde_yaml::from_str(&contents).ok());

    // Build YAML structure — preserve secrets from existing config when fields are empty
    let neo4j_password = if !config.neo4j_password.is_empty() {
        config.neo4j_password.clone()
    } else if let Some(ref old) = existing {
        if old.neo4j.password.is_empty() {
            random_secret(24)
        } else {
            old.neo4j.password.clone()
        }
    } else {
        random_secret(24)
    };

    let meilisearch_key = if !config.meilisearch_key.is_empty() {
        config.meilisearch_key.clone()
    } else if let Some(ref old) = existing {
        if old.meilisearch.key.is_empty() {
            random_secret(32)
        } else {
            old.meilisearch.key.clone()
        }
    } else {
        random_secret(32)
    };

    // Preserve JWT secret across reconfigures — never regenerate if one exists
    let existing_jwt_secret = existing
        .as_ref()
        .and_then(|old| old.auth.as_ref())
        .map(|auth| auth.jwt_secret.clone())
        .filter(|s| !s.is_empty());

    // Preserve OIDC client secret when the frontend sends it empty (redacted)
    let existing_oidc_secret = existing
        .as_ref()
        .and_then(|old| old.auth.as_ref())
        .and_then(|auth| auth.effective_oidc())
        .map(|oidc| oidc.client_secret.clone())
        .filter(|s| !s.is_empty());

    // Use public_url for frontend_url/redirect_uri when configured (reverse proxy),
    // otherwise default to localhost.
    let frontend_url = if config.public_url.trim().is_empty() {
        format!("http://localhost:{}", config.server_port)
    } else {
        config.public_url.trim().trim_end_matches('/').to_string()
    };

    // Parse access restrictions from frontend
    let allowed_email_domain = if config.allowed_email_domain.trim().is_empty() {
        None
    } else {
        Some(config.allowed_email_domain.trim().to_string())
    };

    let allowed_emails: Option<Vec<String>> = {
        let emails: Vec<String> = config
            .allowed_emails
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if emails.is_empty() {
            None
        } else {
            Some(emails)
        }
    };

    // JWT secret: reuse existing or generate new
    let jwt_secret = existing_jwt_secret.unwrap_or_else(|| random_secret(48));

    let auth = match config.auth_mode.as_str() {
        "password" => Some(AuthSection {
            jwt_secret: jwt_secret.clone(),
            jwt_expiry_secs: 28800, // 8h
            frontend_url: Some(frontend_url),
            allowed_email_domain: allowed_email_domain.clone(),
            allowed_emails: allowed_emails.clone(),
            root_account: Some(RootAccountSection {
                email: config.root_email.clone(),
                name: "Admin".to_string(),
                password_hash: config.root_password.clone(),
            }),
            oidc: None,
        }),
        "oidc" => {
            // Use frontend-provided endpoints when available (resolved via Verify button).
            // Only fall back to resolve_oidc_endpoints() network fetch when endpoints are missing.
            let (auth_ep, token_ep, userinfo_ep, resolved_provider_name) =
                if !config.oidc_auth_endpoint.trim().is_empty()
                    && !config.oidc_token_endpoint.trim().is_empty()
                {
                    tracing::info!("Using frontend-provided OIDC endpoints (skipping network fetch)");
                    (
                        config.oidc_auth_endpoint.trim().to_string(),
                        config.oidc_token_endpoint.trim().to_string(),
                        config.oidc_userinfo_endpoint.trim().to_string(),
                        // Detect provider name from discovery URL for fallback
                        detect_provider_name(&config.oidc_discovery_url),
                    )
                } else {
                    tracing::info!("No frontend endpoints — resolving via discovery URL fetch");
                    resolve_oidc_endpoints(&config.oidc_discovery_url)
                };

            // Use frontend-provided provider_name/scopes, fallback to resolved values
            let provider_name = if config.oidc_provider_name.trim().is_empty() {
                resolved_provider_name
            } else {
                config.oidc_provider_name.trim().to_string()
            };
            let scopes = if config.oidc_scopes.trim().is_empty() {
                "openid email profile".to_string()
            } else {
                config.oidc_scopes.trim().to_string()
            };

            // OIDC client secret: preserve existing when frontend sends empty (redacted)
            let oidc_secret = if !config.oidc_client_secret.is_empty() {
                config.oidc_client_secret.clone()
            } else {
                existing_oidc_secret.unwrap_or_default()
            };

            Some(AuthSection {
                jwt_secret,
                jwt_expiry_secs: 28800,
                frontend_url: Some(frontend_url.clone()),
                allowed_email_domain,
                allowed_emails,
                root_account: None,
                oidc: Some(OidcSection {
                    provider_name,
                    client_id: config.oidc_client_id.clone(),
                    client_secret: oidc_secret,
                    redirect_uri: format!("{}/auth/callback", frontend_url),
                    provider_key: if config.oidc_provider.trim().is_empty() {
                        None
                    } else {
                        Some(config.oidc_provider.trim().to_string())
                    },
                    discovery_url: if config.oidc_discovery_url.trim().is_empty() {
                        None
                    } else {
                        Some(config.oidc_discovery_url.trim().to_string())
                    },
                    auth_endpoint: Some(auth_ep),
                    token_endpoint: Some(token_ep),
                    userinfo_endpoint: Some(userinfo_ep),
                    scopes: Some(scopes),
                }),
            })
        }
        _ => None, // "none" — no auth section
    };

    // Build NATS section — only when enabled
    let nats = if !config.nats_enabled {
        None
    } else if config.infra_mode == "docker" {
        // Docker mode: auto URL, no need for user input
        Some(NatsSection {
            url: Some("nats://localhost:4222".to_string()),
        })
    } else if config.nats_url.trim().is_empty() {
        // External mode but no URL provided — still create section with default
        Some(NatsSection {
            url: Some("nats://localhost:4222".to_string()),
        })
    } else {
        Some(NatsSection {
            url: Some(config.nats_url.trim().to_string()),
        })
    };

    // Build public_url for YAML (None if empty)
    let public_url = if config.public_url.trim().is_empty() {
        None
    } else {
        Some(config.public_url.trim().trim_end_matches('/').to_string())
    };

    let yaml = YamlOutput {
        setup_completed: true,
        infra_mode: config.infra_mode.clone(),
        server: ServerSection {
            port: config.server_port,
            workspace_path: ".".into(),
            serve_frontend: config.serve_frontend,
            frontend_path: "./dist".into(),
            public_url,
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
        nats,
        chat: Some(ChatSection {
            default_model: config.chat_model.clone(),
            max_sessions: config.chat_max_sessions,
            max_turns: config.chat_max_turns,
            permissions: if config.chat_permission_mode != "bypassPermissions" {
                // Only write permissions section if non-default (keeps config.yaml clean)
                Some(ChatPermissionsSection {
                    mode: config.chat_permission_mode.clone(),
                })
            } else {
                None
            },
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
    Ok(project_orchestrator::setup_claude::detect_claude_cli().is_some())
}

/// Result of the Claude Code MCP setup, sent to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaudeSetupResult {
    pub success: bool,
    pub method: String, // "cli", "file", "already_configured", "error"
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
}

/// Configure Claude Code to use the Project Orchestrator MCP server.
///
/// Reuses the shared logic from `project_orchestrator::setup_claude`:
/// 1. If already configured → returns immediately
/// 2. Tries `claude mcp add` if CLI is available
/// 3. Falls back to editing `~/.claude/mcp.json` directly
#[tauri::command]
pub fn setup_claude_code(server_url: Option<String>, port: Option<u16>) -> ClaudeSetupResult {
    match project_orchestrator::setup_claude::setup_claude_code(server_url.as_deref(), port) {
        Ok(result) => match result {
            project_orchestrator::setup_claude::SetupResult::ConfiguredViaCli => ClaudeSetupResult {
                success: true,
                method: "cli".into(),
                message: "Claude Code configured via CLI (claude mcp add)".into(),
                file_path: None,
            },
            project_orchestrator::setup_claude::SetupResult::ConfiguredViaFile { path } => {
                ClaudeSetupResult {
                    success: true,
                    method: "file".into(),
                    message: format!("Claude Code configured by editing {}", path.display()),
                    file_path: Some(path.display().to_string()),
                }
            }
            project_orchestrator::setup_claude::SetupResult::AlreadyConfigured => {
                ClaudeSetupResult {
                    success: true,
                    method: "already_configured".into(),
                    message: "Project Orchestrator is already configured in Claude Code".into(),
                    file_path: None,
                }
            }
        },
        Err(e) => ClaudeSetupResult {
            success: false,
            method: "error".into(),
            message: format!("Failed to configure: {}", e),
            file_path: None,
        },
    }
}

/// Read the existing config.yaml and return its values for the frontend wizard
/// to pre-fill fields in reconfigure mode.
///
/// Sensitive fields (passwords, JWT secret, MeiliSearch key) are redacted —
/// the frontend will show empty password fields that the user can optionally
/// re-fill.
#[tauri::command]
pub fn read_config() -> Result<ReadConfigResponse, String> {
    let path = config_path();
    tracing::info!("Reading config for reconfigure from: {}", path.display());

    let contents = std::fs::read_to_string(&path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;

    let yaml: project_orchestrator::YamlConfig = serde_yaml::from_str(&contents)
        .map_err(|e| format!("Cannot parse {}: {}", path.display(), e))?;

    // Determine infra mode: read from YAML (new configs), fallback to heuristic (old configs)
    let infra_mode = yaml.infra_mode.clone().unwrap_or_else(|| {
        // Heuristic fallback for config.yaml files generated before infra_mode was persisted
        if yaml.neo4j.uri.contains("localhost") || yaml.neo4j.uri.contains("127.0.0.1") {
            "docker".into()
        } else {
            "external".into()
        }
    });

    // Read public_url from server section (new field, empty for old configs)
    let public_url = yaml.server.public_url.clone().unwrap_or_default();

    // Determine NATS enabled status: section present with a URL → enabled
    let nats_enabled = yaml.nats.url.is_some();

    // Determine auth mode and extract OIDC + restriction fields
    let mut auth_mode = "none".to_string();
    let mut root_email = String::new();
    let mut oidc_provider = String::new();
    let mut oidc_discovery_url = String::new();
    let mut oidc_client_id = String::new();
    let mut oidc_provider_name = String::new();
    let mut oidc_scopes = String::new();
    let mut oidc_auth_endpoint = String::new();
    let mut oidc_token_endpoint = String::new();
    let mut oidc_userinfo_endpoint = String::new();
    let mut allowed_email_domain = String::new();
    let mut allowed_emails = String::new();
    let mut has_oidc_secret = false;
    let has_neo4j_password = !yaml.neo4j.password.is_empty();
    let has_meilisearch_key = !yaml.meilisearch.key.is_empty();

    if let Some(ref auth) = yaml.auth {
        // Access restrictions (common to all auth modes)
        if let Some(ref domain) = auth.allowed_email_domain {
            allowed_email_domain = domain.clone();
        }
        if let Some(ref emails) = auth.allowed_emails {
            allowed_emails = emails.join("\n");
        }

        if auth.root_account.is_some() {
            auth_mode = "password".to_string();
            root_email = auth
                .root_account
                .as_ref()
                .map(|r| r.email.clone())
                .unwrap_or_default();
        }

        if auth.oidc.is_some() || auth.google_client_id.is_some() {
            if auth_mode == "none" {
                auth_mode = "oidc".to_string();
            }
            let oidc = auth.effective_oidc();
            if let Some(ref o) = oidc {
                oidc_discovery_url = o.discovery_url.clone().unwrap_or_default();
                oidc_client_id = o.client_id.clone();
                has_oidc_secret = !o.client_secret.is_empty();
                oidc_provider_name = o.provider_name.clone();
                oidc_scopes = o.scopes.clone();
                oidc_auth_endpoint = o.auth_endpoint.clone().unwrap_or_default();
                oidc_token_endpoint = o.token_endpoint.clone().unwrap_or_default();
                oidc_userinfo_endpoint = o.userinfo_endpoint.clone().unwrap_or_default();

                // Use persisted provider_key when available (new configs),
                // fallback to heuristic from discovery URL (old configs).
                oidc_provider = o.provider_key.clone().filter(|k| !k.is_empty()).unwrap_or_else(|| {
                    let disc = oidc_discovery_url.to_lowercase();
                    if disc.contains("accounts.google.com") || disc.contains("googleapis") {
                        "google".into()
                    } else if disc.contains("login.microsoftonline.com") {
                        "microsoft".into()
                    } else if disc.contains(".okta.com") {
                        "okta".into()
                    } else if disc.contains(".auth0.com") {
                        "auth0".into()
                    } else if disc.contains("/realms/") {
                        "keycloak".into()
                    } else {
                        "custom".into()
                    }
                });
            }
        }
    }

    // NATS URL
    let nats_url = yaml.nats.url.unwrap_or_default();

    Ok(ReadConfigResponse {
        infra_mode,
        neo4j_uri: yaml.neo4j.uri,
        neo4j_user: yaml.neo4j.user,
        neo4j_password: String::new(), // redacted
        meilisearch_url: yaml.meilisearch.url,
        meilisearch_key: String::new(), // redacted
        server_port: yaml.server.port,
        serve_frontend: yaml.server.serve_frontend,
        public_url,
        auth_mode,
        root_email,
        root_password: String::new(), // redacted
        oidc_provider,
        oidc_discovery_url,
        oidc_client_id,
        oidc_client_secret: String::new(), // redacted
        oidc_provider_name,
        oidc_scopes,
        oidc_auth_endpoint,
        oidc_token_endpoint,
        oidc_userinfo_endpoint,
        allowed_email_domain,
        allowed_emails,
        nats_url,
        nats_enabled,
        chat_model: yaml.chat.default_model.unwrap_or_else(|| "sonnet-4-5".into()),
        chat_max_sessions: yaml.chat.max_sessions.unwrap_or(3) as u32,
        chat_max_turns: yaml.chat.max_turns.unwrap_or(50) as u32,
        chat_permission_mode: yaml.chat.permissions
            .map(|p| p.mode)
            .unwrap_or_else(|| "bypassPermissions".into()),
        has_oidc_secret,
        has_neo4j_password,
        has_meilisearch_key,
    })
}

/// Response from `read_config` — mirrors the frontend `SetupConfig` type.
///
/// Sensitive fields are empty strings (passwords, secrets, keys).
/// `has_*` booleans indicate whether a secret exists in the config
/// so the frontend can show "Secret exists — leave blank to keep".
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadConfigResponse {
    pub infra_mode: String,
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub meilisearch_url: String,
    pub meilisearch_key: String,
    pub server_port: u16,
    pub serve_frontend: bool,
    /// Public URL for reverse-proxy setups (empty if not configured).
    pub public_url: String,
    pub auth_mode: String,
    pub root_email: String,
    pub root_password: String,
    /// Selected OIDC provider key (google, microsoft, etc.) or empty.
    pub oidc_provider: String,
    pub oidc_discovery_url: String,
    pub oidc_client_id: String,
    pub oidc_client_secret: String,
    pub oidc_provider_name: String,
    pub oidc_scopes: String,
    pub oidc_auth_endpoint: String,
    pub oidc_token_endpoint: String,
    pub oidc_userinfo_endpoint: String,
    pub allowed_email_domain: String,
    pub allowed_emails: String,
    pub nats_url: String,
    /// Whether NATS was enabled (deduced from presence of nats section in YAML).
    pub nats_enabled: bool,
    pub chat_model: String,
    pub chat_max_sessions: u32,
    pub chat_max_turns: u32,
    pub chat_permission_mode: String,
    // Indicators for existing secrets (reconfigure mode)
    pub has_oidc_secret: bool,
    pub has_neo4j_password: bool,
    pub has_meilisearch_key: bool,
}

/// Generate a minimal config.yaml for first-launch (setup_completed = false).
///
/// The backend starts in no-auth mode so the frontend can show the setup wizard.
/// Returns the path where the config was written.
pub fn generate_default_config() -> Result<PathBuf, String> {
    let path = config_path();
    tracing::info!("Generating default (unconfigured) config at: {}", path.display());

    let yaml = YamlOutput {
        setup_completed: false,
        infra_mode: "docker".into(),
        server: ServerSection {
            port: DEFAULT_DESKTOP_PORT,
            workspace_path: ".".into(),
            serve_frontend: false,
            frontend_path: "./dist".into(),
            public_url: None,
        },
        neo4j: Neo4jSection {
            uri: "bolt://localhost:7687".into(),
            user: "neo4j".into(),
            password: random_secret(24),
        },
        meilisearch: MeilisearchSection {
            url: "http://localhost:7700".into(),
            key: random_secret(32),
        },
        nats: None,
        chat: None,
        auth: None, // no-auth mode — wizard can load freely
    };

    let yaml_str = serde_yaml::to_string(&yaml).map_err(|e| format!("YAML error: {}", e))?;
    let output = format!(
        "# =============================================================================\n\
         # Project Orchestrator — Default Configuration (not yet configured)\n\
         # Run the setup wizard to complete configuration.\n\
         # =============================================================================\n\n\
         {}",
        yaml_str
    );

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
    }

    std::fs::write(&path, &output)
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;

    tracing::info!("Default config written to: {}", path.display());
    Ok(path)
}

// ============================================================================
// OIDC Discovery verification (Tauri command — bypasses browser CSP)
// ============================================================================

/// Response from `verify_oidc_discovery` — sent back to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OidcDiscoveryResponse {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub userinfo_endpoint: String,
}

/// Fetch and parse an OIDC discovery document from the given URL.
///
/// This Tauri command exists because the browser CSP in the desktop app blocks
/// fetch() to external HTTPS domains. By doing the fetch in Rust, we bypass
/// the CSP entirely and also avoid CORS issues with self-hosted providers.
#[tauri::command]
pub async fn verify_oidc_discovery(url: String) -> Result<OidcDiscoveryResponse, String> {
    let url = url.trim().to_string();
    if url.is_empty() {
        return Err("Discovery URL is empty".into());
    }

    // Ensure URL ends with .well-known/openid-configuration
    let well_known = if url.contains(".well-known/openid-configuration") {
        url.clone()
    } else {
        let base = url.trim_end_matches('/');
        format!("{}/.well-known/openid-configuration", base)
    };

    tracing::info!("Fetching OIDC discovery document: {}", well_known);

    let resp = reqwest::get(&well_known)
        .await
        .map_err(|e| format!("Failed to fetch discovery document: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}: {}", resp.status().as_u16(), resp.status().canonical_reason().unwrap_or("error")));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Invalid JSON in discovery document: {}", e))?;

    let auth = json["authorization_endpoint"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let token = json["token_endpoint"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if auth.is_empty() || token.is_empty() {
        return Err("Invalid discovery document: missing authorization_endpoint or token_endpoint".into());
    }

    Ok(OidcDiscoveryResponse {
        issuer: json["issuer"].as_str().unwrap_or("").to_string(),
        authorization_endpoint: auth,
        token_endpoint: token,
        userinfo_endpoint: json["userinfo_endpoint"]
            .as_str()
            .unwrap_or("")
            .to_string(),
    })
}

// ============================================================================
// OIDC endpoint resolution helpers
// ============================================================================

/// Resolve an OIDC discovery URL to explicit endpoints.
///
/// 1. First tries to fetch the `.well-known/openid-configuration` document
/// 2. Falls back to well-known endpoints for common providers (Google, Microsoft)
/// 3. As a last resort, uses the discovery URL itself as both auth + token endpoint
///
/// Returns `(auth_endpoint, token_endpoint, userinfo_endpoint, provider_name)`.
///
/// Detect a human-friendly provider name from a discovery/issuer URL.
fn detect_provider_name(url: &str) -> String {
    let lower = url.to_lowercase();
    if lower.contains("google") || lower.contains("googleapis") {
        "Google".into()
    } else if lower.contains("microsoft") || lower.contains("login.microsoftonline") {
        "Microsoft".into()
    } else if lower.contains("github") {
        "GitHub".into()
    } else if lower.contains("okta") {
        "Okta".into()
    } else if lower.contains("auth0") {
        "Auth0".into()
    } else {
        "OIDC".into()
    }
}

fn resolve_oidc_endpoints(discovery_url: &str) -> (String, String, String, String) {
    let url = discovery_url.trim();

    let provider_name = detect_provider_name(url);

    // Try to fetch the discovery document
    let well_known = if url.contains(".well-known/openid-configuration") {
        url.to_string()
    } else {
        // Append .well-known path
        let base = url.trim_end_matches('/');
        format!("{}/.well-known/openid-configuration", base)
    };

    if let Ok(resp) = reqwest::blocking::get(&well_known) {
        if resp.status().is_success() {
            if let Ok(json) = resp.json::<serde_json::Value>() {
                let auth = json["authorization_endpoint"].as_str().unwrap_or("").to_string();
                let token = json["token_endpoint"].as_str().unwrap_or("").to_string();
                let userinfo = json["userinfo_endpoint"].as_str().unwrap_or("").to_string();

                if !auth.is_empty() && !token.is_empty() {
                    tracing::info!(
                        "Resolved OIDC endpoints from discovery: auth={}, token={}, userinfo={}",
                        auth, token, userinfo
                    );
                    return (auth, token, userinfo, provider_name);
                }
            }
        }
    }

    tracing::warn!("Could not fetch OIDC discovery document at {}, using known provider defaults", well_known);

    // Fallback: well-known endpoints for common providers
    match provider_name.as_str() {
        "Google" => (
            "https://accounts.google.com/o/oauth2/v2/auth".into(),
            "https://oauth2.googleapis.com/token".into(),
            "https://www.googleapis.com/oauth2/v3/userinfo".into(),
            "Google".into(),
        ),
        _ => {
            // Last resort — use the URL as-is (will likely fail but at least config is written)
            tracing::error!("Unknown OIDC provider and discovery fetch failed — config may need manual editing");
            (
                url.to_string(),
                url.to_string(),
                url.to_string(),
                provider_name,
            )
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
