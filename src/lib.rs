//! Project Orchestrator
//!
//! An AI agent orchestrator with:
//! - Neo4j knowledge graph for code structure and relationships
//! - Meilisearch for fast semantic search
//! - Tree-sitter for precise code parsing
//! - Plan management for coordinated multi-agent development
//! - MCP server for Claude Code integration

pub mod api;
pub mod auth;
pub mod chat;
pub mod events;
pub mod mcp;
pub mod meilisearch;
pub mod neo4j;
pub mod notes;
pub mod orchestrator;
pub mod parser;
pub mod plan;
pub mod setup_claude;
pub mod update;

#[cfg(test)]
pub(crate) mod test_helpers;

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

/// Expand tilde (~) in paths to the user's home directory.
/// Shell expansion doesn't happen when paths are passed programmatically.
pub(crate) fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return format!("{}{}", home.display(), &path[1..]);
        }
    } else if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.display().to_string();
        }
    }
    path.to_string()
}

// ============================================================================
// YAML config structs (deserialization targets)
// ============================================================================

/// Top-level YAML configuration file structure
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct YamlConfig {
    /// When false, the app is not yet configured — the frontend shows the setup wizard.
    /// Defaults to true for backward compat (existing config.yaml files without this field).
    #[serde(default = "default_true")]
    pub setup_completed: bool,
    pub server: ServerYamlConfig,
    pub neo4j: Neo4jYamlConfig,
    pub meilisearch: MeilisearchYamlConfig,
    pub nats: NatsYamlConfig,
    pub chat: ChatYamlConfig,
    /// Auth section — if absent, auth_config will be None (deny-by-default)
    pub auth: Option<AuthConfig>,
}

fn default_true() -> bool {
    true
}

/// Server configuration section
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerYamlConfig {
    pub port: u16,
    pub workspace_path: String,
    /// Whether the backend should serve the frontend static files (default: true)
    pub serve_frontend: bool,
    /// Path to the frontend dist/ directory (default: "./dist")
    pub frontend_path: String,
}

impl Default for ServerYamlConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            workspace_path: ".".into(),
            serve_frontend: true,
            frontend_path: "./dist".into(),
        }
    }
}

/// Neo4j configuration section
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Neo4jYamlConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
}

impl Default for Neo4jYamlConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".into(),
            user: "neo4j".into(),
            password: "orchestrator123".into(),
        }
    }
}

/// Meilisearch configuration section
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MeilisearchYamlConfig {
    pub url: String,
    pub key: String,
}

impl Default for MeilisearchYamlConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:7700".into(),
            key: "orchestrator-meili-key-change-me".into(),
        }
    }
}

/// NATS configuration section (optional — enables inter-process event sync)
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct NatsYamlConfig {
    /// NATS server URL (e.g. "nats://localhost:4222")
    pub url: Option<String>,
}

/// Chat configuration section (YAML only — ChatConfig in chat/config.rs handles full setup)
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct ChatYamlConfig {
    pub default_model: Option<String>,
    pub max_sessions: Option<usize>,
    pub session_timeout_secs: Option<u64>,
    pub max_turns: Option<i32>,
    pub prompt_builder_model: Option<String>,
}

/// Authentication configuration — flexible multi-provider auth.
///
/// Supports three modes depending on which sub-sections are present:
/// - **No-auth**: No `auth` section in YAML → `auth_config = None` → open access
/// - **Password**: `root_account` present → email/password login (root from config, others from Neo4j)
/// - **OIDC**: `oidc` present → generic OpenID Connect (Google, Microsoft, Okta, Keycloak…)
///
/// Both `root_account` and `oidc` can coexist for maximum flexibility.
///
/// ### Backward compatibility
/// The legacy Google OAuth fields (`google_client_id`, `google_client_secret`,
/// `google_redirect_uri`) are still accepted. When present, they are automatically
/// mapped to an equivalent `OidcConfig` via [`AuthConfig::effective_oidc()`].
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    // ── Common fields ──────────────────────────────────────────────────
    /// JWT signing secret (HS256, minimum 32 characters)
    pub jwt_secret: String,
    /// JWT token lifetime in seconds (default: 28800 = 8h)
    #[serde(default = "default_jwt_expiry")]
    pub jwt_expiry_secs: u64,
    /// Optional domain restriction (e.g. "ffs.holdings")
    pub allowed_email_domain: Option<String>,
    /// Frontend URL for CORS and redirects (e.g. "http://localhost:3000")
    pub frontend_url: Option<String>,
    /// Allow new user registration via POST /auth/register (default: false)
    #[serde(default)]
    pub allow_registration: bool,

    // ── Password auth (root account from config) ───────────────────────
    /// Root account defined in config.yaml — always available, no DB needed.
    pub root_account: Option<RootAccountConfig>,

    // ── OIDC auth (generic OpenID Connect) ─────────────────────────────
    /// Generic OIDC provider configuration.
    pub oidc: Option<OidcConfig>,

    // ── Legacy Google OAuth fields (backward compat) ───────────────────
    /// Deprecated: use `oidc.client_id` instead.
    #[serde(default)]
    pub google_client_id: Option<String>,
    /// Deprecated: use `oidc.client_secret` instead.
    #[serde(default)]
    pub google_client_secret: Option<String>,
    /// Deprecated: use `oidc.redirect_uri` instead.
    #[serde(default)]
    pub google_redirect_uri: Option<String>,
}

/// Root account configuration — defined in config.yaml, verified in-memory.
///
/// The `password_hash` field can contain either:
/// - A bcrypt hash (starts with `$2b$` or `$2a$`) → used as-is
/// - A plaintext password → hashed with bcrypt at startup (with a warning log)
#[derive(Debug, Clone, Deserialize)]
pub struct RootAccountConfig {
    /// Root account email (used as login identifier)
    pub email: String,
    /// Root account display name
    pub name: String,
    /// Bcrypt hash or plaintext password (hashed at startup if plaintext)
    pub password_hash: String,
}

/// OIDC provider configuration — generic OpenID Connect.
///
/// Works with any OIDC-compliant provider: Google, Microsoft, Okta, Keycloak, etc.
#[derive(Debug, Clone, Deserialize)]
pub struct OidcConfig {
    /// OIDC discovery URL (e.g. "https://accounts.google.com/.well-known/openid-configuration").
    /// If provided, `auth_endpoint` and `token_endpoint` are fetched automatically.
    pub discovery_url: Option<String>,
    /// Authorization endpoint (required if no discovery_url)
    pub auth_endpoint: Option<String>,
    /// Token endpoint (required if no discovery_url)
    pub token_endpoint: Option<String>,
    /// Userinfo endpoint (optional, fetched from discovery if available)
    pub userinfo_endpoint: Option<String>,
    /// OAuth2 client ID
    pub client_id: String,
    /// OAuth2 client secret
    pub client_secret: String,
    /// Redirect URI after auth (e.g. "http://localhost:3000/auth/callback")
    pub redirect_uri: String,
    /// Human-readable provider name shown in the UI (e.g. "Google", "Okta")
    #[serde(default = "default_provider_name")]
    pub provider_name: String,
    /// OAuth2 scopes (default: "openid email profile")
    #[serde(default = "default_scopes")]
    pub scopes: String,
}

fn default_jwt_expiry() -> u64 {
    28800 // 8 hours
}

fn default_provider_name() -> String {
    "SSO".to_string()
}

fn default_scopes() -> String {
    "openid email profile".to_string()
}

// Google OIDC well-known endpoints (used for legacy config mapping)
const GOOGLE_AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_USERINFO_ENDPOINT: &str = "https://www.googleapis.com/oauth2/v3/userinfo";

impl AuthConfig {
    /// Returns the effective OIDC config, preferring the explicit `oidc` section
    /// and falling back to legacy `google_*` fields for backward compatibility.
    pub fn effective_oidc(&self) -> Option<OidcConfig> {
        // Prefer explicit oidc section
        if self.oidc.is_some() {
            return self.oidc.clone();
        }

        // Fall back to legacy Google fields
        match (
            &self.google_client_id,
            &self.google_client_secret,
            &self.google_redirect_uri,
        ) {
            (Some(client_id), Some(client_secret), Some(redirect_uri)) if !client_id.is_empty() => {
                Some(OidcConfig {
                    discovery_url: None,
                    auth_endpoint: Some(GOOGLE_AUTH_ENDPOINT.to_string()),
                    token_endpoint: Some(GOOGLE_TOKEN_ENDPOINT.to_string()),
                    userinfo_endpoint: Some(GOOGLE_USERINFO_ENDPOINT.to_string()),
                    client_id: client_id.clone(),
                    client_secret: client_secret.clone(),
                    redirect_uri: redirect_uri.clone(),
                    provider_name: "Google".to_string(),
                    scopes: "openid email profile".to_string(),
                })
            }
            _ => None,
        }
    }

    /// Returns true if password authentication is available (root account configured).
    pub fn has_password_auth(&self) -> bool {
        self.root_account.is_some()
    }

    /// Returns true if OIDC authentication is available (explicit or legacy Google).
    pub fn has_oidc(&self) -> bool {
        self.effective_oidc().is_some()
    }

    /// Ensure the root account password is bcrypt-hashed.
    ///
    /// If the password_hash field doesn't start with "$2b$" (bcrypt prefix),
    /// it's treated as plaintext and hashed in-place with bcrypt cost 12.
    /// Logs a warning when plaintext is detected to encourage using pre-hashed values.
    pub fn ensure_root_password_hashed(&mut self) -> anyhow::Result<()> {
        if let Some(ref mut root) = self.root_account {
            if !root.password_hash.starts_with("$2b$") && !root.password_hash.starts_with("$2a$") {
                tracing::warn!(
                    "Root account password is stored as plaintext in config.yaml — hashing at startup. \
                     Consider pre-hashing with: echo -n 'password' | htpasswd -bnBC 12 '' | cut -d: -f2"
                );
                root.password_hash = bcrypt::hash(&root.password_hash, 12)
                    .map_err(|e| anyhow::anyhow!("Failed to hash root password: {}", e))?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Runtime config (what the application actually uses)
// ============================================================================

/// Application configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// When false, the setup wizard should be shown instead of the main app.
    /// Defaults to true for backward compat.
    pub setup_completed: bool,
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub meilisearch_url: String,
    pub meilisearch_key: String,
    /// NATS server URL (optional — enables inter-process event sync)
    pub nats_url: Option<String>,
    pub workspace_path: String,
    pub server_port: u16,
    /// Auth config — None means deny-by-default (no auth section in YAML)
    pub auth_config: Option<AuthConfig>,
    /// Whether to serve the frontend static files (default: true)
    pub serve_frontend: bool,
    /// Path to the frontend dist/ directory (default: "./dist")
    pub frontend_path: String,
}

impl Config {
    /// Load configuration from environment variables only (backward compat).
    /// Equivalent to from_yaml_and_env(None).
    pub fn from_env() -> Result<Self> {
        Self::from_yaml_and_env(None)
    }

    /// Load configuration from an optional YAML file, then override with env vars.
    ///
    /// Priority: env var > YAML > default
    ///
    /// If `yaml_path` is None, tries "config.yaml" in CWD. If the file doesn't
    /// exist, falls back to pure env var / defaults (backward compatible).
    pub fn from_yaml_and_env(yaml_path: Option<&Path>) -> Result<Self> {
        // 1. Load YAML config (or defaults if file not found)
        let yaml = Self::load_yaml(yaml_path);

        // 2. Build Config with env var overrides
        Ok(Self {
            setup_completed: yaml.setup_completed,
            neo4j_uri: std::env::var("NEO4J_URI").unwrap_or(yaml.neo4j.uri),
            neo4j_user: std::env::var("NEO4J_USER").unwrap_or(yaml.neo4j.user),
            neo4j_password: std::env::var("NEO4J_PASSWORD").unwrap_or(yaml.neo4j.password),
            meilisearch_url: std::env::var("MEILISEARCH_URL").unwrap_or(yaml.meilisearch.url),
            meilisearch_key: std::env::var("MEILISEARCH_KEY").unwrap_or(yaml.meilisearch.key),
            nats_url: std::env::var("NATS_URL").ok().or(yaml.nats.url),
            workspace_path: std::env::var("WORKSPACE_PATH").unwrap_or(yaml.server.workspace_path),
            server_port: std::env::var("SERVER_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(yaml.server.port),
            auth_config: yaml.auth,
            serve_frontend: std::env::var("SERVE_FRONTEND")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(yaml.server.serve_frontend),
            frontend_path: std::env::var("FRONTEND_PATH")
                .unwrap_or(yaml.server.frontend_path),
        })
    }

    /// Try to load and parse a YAML config file. Returns defaults on any failure.
    fn load_yaml(yaml_path: Option<&Path>) -> YamlConfig {
        let default_path = Path::new("config.yaml");
        let path = yaml_path.unwrap_or(default_path);

        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_yaml::from_str(&contents) {
                Ok(config) => {
                    tracing::info!("Loaded config from {}", path.display());
                    config
                }
                Err(e) => {
                    tracing::warn!("Failed to parse {}: {}. Using defaults.", path.display(), e);
                    YamlConfig::default()
                }
            },
            Err(_) => {
                tracing::debug!(
                    "No config file at {}, using env vars / defaults",
                    path.display()
                );
                YamlConfig::default()
            }
        }
    }
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub neo4j: Arc<dyn neo4j::GraphStore>,
    pub meili: Arc<dyn meilisearch::SearchStore>,
    pub parser: Arc<parser::CodeParser>,
    pub config: Arc<Config>,
}

impl AppState {
    /// Create new application state with all services initialized
    pub async fn new(config: Config) -> Result<Self> {
        let neo4j = Arc::new(
            neo4j::client::Neo4jClient::new(
                &config.neo4j_uri,
                &config.neo4j_user,
                &config.neo4j_password,
            )
            .await?,
        );

        let meili = Arc::new(
            meilisearch::client::MeiliClient::new(&config.meilisearch_url, &config.meilisearch_key)
                .await?,
        );

        let parser = Arc::new(parser::CodeParser::new()?);

        Ok(Self {
            neo4j,
            meili,
            parser,
            config: Arc::new(config),
        })
    }
}

// ============================================================================
// Server entry point (for embedding in Tauri or other hosts)
// ============================================================================

/// Start the orchestrator server with the given configuration.
///
/// This is the main entry point for embedding the server in another application
/// (e.g., Tauri desktop). It initializes all services, creates the Axum router,
/// and binds to the configured port.
///
/// Returns when the server shuts down (or an error occurs during startup).
pub async fn start_server(mut config: Config) -> Result<()> {
    use std::net::SocketAddr;

    // ────────────────────────────────────────────────────────────────────
    // Setup-only mode: if the wizard hasn't been completed yet, start a
    // minimal server that only serves /health and /api/setup-status.
    // This avoids connecting to Neo4j/MeiliSearch (which are likely not
    // configured yet) and lets the frontend show the setup wizard.
    // ────────────────────────────────────────────────────────────────────
    if !config.setup_completed {
        tracing::info!(
            "Setup not completed — starting minimal setup-only server on port {}",
            config.server_port
        );
        return start_setup_server(config.server_port).await;
    }

    // ────────────────────────────────────────────────────────────────────
    // Normal (fully configured) server
    // ────────────────────────────────────────────────────────────────────
    use api::handlers::ServerState;
    use tokio::sync::RwLock;

    // Hash root account password at startup (if plaintext)
    if let Some(ref mut auth) = config.auth_config {
        auth.ensure_root_password_hashed()?;
    }

    // Initialize application state
    tracing::info!("Connecting to Neo4j at {}...", config.neo4j_uri);
    tracing::info!("Connecting to Meilisearch at {}...", config.meilisearch_url);

    let state = AppState::new(config.clone()).await?;
    tracing::info!("Connected to databases");

    // Create local event bus for intra-process broadcast
    let local_bus = Arc::new(events::EventBus::default());

    // Connect to NATS if configured (inter-process event sync)
    let nats_emitter = if let Some(ref nats_url) = config.nats_url {
        match events::connect_nats(nats_url).await {
            Ok(client) => {
                let emitter = Arc::new(events::NatsEmitter::new(client, "events"));
                tracing::info!("NATS connected — inter-process event sync enabled");
                Some(emitter)
            }
            Err(e) => {
                tracing::warn!("Failed to connect to NATS: {} — running in local-only mode", e);
                None
            }
        }
    } else {
        tracing::info!("NATS not configured — running in local-only mode");
        None
    };

    // Create hybrid emitter (local broadcast + optional NATS)
    let event_bus = Arc::new(match &nats_emitter {
        Some(nats) => events::HybridEmitter::with_nats(local_bus, nats.clone()),
        None => events::HybridEmitter::new(local_bus),
    });

    // Create orchestrator with hybrid emitter
    let orchestrator =
        Arc::new(orchestrator::Orchestrator::with_event_bus(state, event_bus.clone()).await?);

    // Create file watcher
    let watcher = orchestrator::FileWatcher::new(orchestrator.clone());

    // Create chat manager (optional — requires Claude CLI)
    let chat_manager = {
        let chat_config = chat::ChatConfig::from_env();
        let mut cm = chat::ChatManager::new(
            orchestrator.neo4j_arc(),
            orchestrator.meili_arc(),
            chat_config,
        )
        .await
        .with_event_emitter(event_bus.clone());
        if let Some(ref nats) = nats_emitter {
            cm = cm.with_nats(nats.clone());
        }
        let cm = Arc::new(cm);
        cm.start_cleanup_task();
        tracing::info!("Chat manager initialized");
        Some(cm)
    };

    // Create server state
    let server_state = Arc::new(ServerState {
        orchestrator,
        watcher: Arc::new(RwLock::new(watcher)),
        chat_manager,
        event_bus,
        nats_emitter,
        auth_config: config.auth_config.clone(),
        serve_frontend: config.serve_frontend,
        frontend_path: config.frontend_path.clone(),
        setup_completed: config.setup_completed,
    });

    // Create router
    let app = api::create_router(server_state);

    // Log frontend serving mode
    if config.serve_frontend {
        tracing::info!(
            "Frontend serving enabled — path: {}",
            config.frontend_path
        );
    } else {
        tracing::info!("Frontend serving disabled (API-only mode)");
    }

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server_port));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start a minimal server for the setup wizard.
///
/// Only serves `/health`, `/api/setup-status`, and `/auth/providers`.
/// No database connections are required — it's a lightweight Axum server
/// that lets the Tauri frontend display the setup wizard.
async fn start_setup_server(port: u16) -> Result<()> {
    use axum::routing::get;
    use serde::Serialize;
    use std::net::SocketAddr;
    use tower_http::cors::{Any, CorsLayer};

    #[derive(Serialize)]
    struct HealthResp {
        status: String,
        version: String,
    }

    #[derive(Serialize)]
    struct SetupStatusResp {
        configured: bool,
    }

    #[derive(Serialize)]
    struct AuthProvidersResp {
        auth_required: bool,
        providers: Vec<String>,
        allow_registration: bool,
    }

    async fn health() -> axum::Json<HealthResp> {
        axum::Json(HealthResp {
            status: "ok".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    async fn setup_status() -> axum::Json<SetupStatusResp> {
        axum::Json(SetupStatusResp { configured: false })
    }

    async fn auth_providers() -> axum::Json<AuthProvidersResp> {
        axum::Json(AuthProvidersResp {
            auth_required: false,
            providers: vec![],
            allow_registration: false,
        })
    }

    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .allow_origin(Any);

    let app = axum::Router::new()
        .route("/health", get(health))
        .route("/api/setup-status", get(setup_status))
        .route("/auth/providers", get(auth_providers))
        .layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Setup-only server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod config_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_yaml_config_loading() {
        let yaml = r#"
server:
  port: 9090
  workspace_path: /tmp/test

neo4j:
  uri: bolt://db:7687
  user: admin
  password: secret

meilisearch:
  url: http://search:7700
  key: test-key

auth:
  google_client_id: "123.apps.googleusercontent.com"
  google_client_secret: "secret123"
  google_redirect_uri: "http://localhost:3000/auth/callback"
  jwt_secret: "super-secret-key-min-32-characters!"
  jwt_expiry_secs: 3600
  allowed_email_domain: "ffs.holdings"
  frontend_url: "http://localhost:3000"
"#;

        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.server.port, 9090);
        assert_eq!(config.server.workspace_path, "/tmp/test");
        assert_eq!(config.neo4j.uri, "bolt://db:7687");
        assert_eq!(config.meilisearch.key, "test-key");

        let auth = config.auth.unwrap();
        // Legacy Google fields are now Option<String>
        assert_eq!(
            auth.google_client_id,
            Some("123.apps.googleusercontent.com".into())
        );
        assert_eq!(auth.jwt_expiry_secs, 3600);
        assert_eq!(auth.allowed_email_domain, Some("ffs.holdings".into()));
        assert_eq!(auth.frontend_url, Some("http://localhost:3000".into()));
        // Backward compat: effective_oidc() builds from legacy fields
        let oidc = auth.effective_oidc().expect("should have effective OIDC");
        assert_eq!(oidc.client_id, "123.apps.googleusercontent.com");
        assert_eq!(oidc.provider_name, "Google");
    }

    #[test]
    fn test_auth_config_absent() {
        let yaml = r#"
server:
  port: 8080
neo4j:
  uri: bolt://localhost:7687
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.auth.is_none());
    }

    #[test]
    fn test_yaml_defaults() {
        let config = YamlConfig::default();
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.workspace_path, ".");
        assert_eq!(config.neo4j.uri, "bolt://localhost:7687");
        assert_eq!(config.neo4j.user, "neo4j");
        assert_eq!(config.meilisearch.url, "http://localhost:7700");
        assert!(config.auth.is_none());
    }

    #[test]
    fn test_jwt_expiry_default() {
        let yaml = r#"
auth:
  google_client_id: "id"
  google_client_secret: "secret"
  google_redirect_uri: "http://localhost/callback"
  jwt_secret: "min-32-chars-secret-key-for-test!"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        let auth = config.auth.unwrap();
        assert_eq!(auth.jwt_expiry_secs, 28800); // 8h default
        assert!(auth.allowed_email_domain.is_none());
        assert!(auth.frontend_url.is_none());
    }

    /// Combined test for YAML file loading, env var overrides, and backward compat.
    /// Runs as a single test to avoid parallel env var race conditions.
    #[test]
    fn test_yaml_and_env_lifecycle() {
        // Helper to clear all config env vars
        fn clear_env() {
            for var in &[
                "NEO4J_URI",
                "NEO4J_USER",
                "NEO4J_PASSWORD",
                "MEILISEARCH_URL",
                "MEILISEARCH_KEY",
                "WORKSPACE_PATH",
                "SERVER_PORT",
                "SERVE_FRONTEND",
                "FRONTEND_PATH",
            ] {
                std::env::remove_var(var);
            }
        }

        // --- Phase 1: YAML values loaded correctly ---
        let yaml = r#"
server:
  port: 9999
neo4j:
  uri: bolt://yaml-host:7687
  user: yaml-user
  password: yaml-pass
meilisearch:
  url: http://yaml-search:7700
  key: yaml-key
"#;
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("config.yaml");
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(yaml.as_bytes()).unwrap();

        clear_env();

        let config = Config::from_yaml_and_env(Some(&file_path)).unwrap();
        assert_eq!(config.server_port, 9999);
        assert_eq!(config.neo4j_uri, "bolt://yaml-host:7687");
        assert_eq!(config.neo4j_user, "yaml-user");
        assert_eq!(config.meilisearch_key, "yaml-key");
        assert!(config.auth_config.is_none());

        // --- Phase 2: Env vars override YAML ---
        std::env::set_var("NEO4J_URI", "bolt://env-host:7687");
        std::env::set_var("SERVER_PORT", "7777");

        let config = Config::from_yaml_and_env(Some(&file_path)).unwrap();
        assert_eq!(config.neo4j_uri, "bolt://env-host:7687");
        assert_eq!(config.server_port, 7777);
        // YAML value still used where no env override
        assert_eq!(config.neo4j_user, "yaml-user");

        clear_env();

        // --- Phase 3: No YAML file → defaults ---
        let nonexistent = Path::new("/tmp/nonexistent-config-12345.yaml");
        let config = Config::from_yaml_and_env(Some(nonexistent)).unwrap();
        assert_eq!(config.server_port, 8080);
        assert_eq!(config.neo4j_uri, "bolt://localhost:7687");
        assert!(config.auth_config.is_none());
        // Frontend defaults when no YAML
        assert!(config.serve_frontend);
        assert_eq!(config.frontend_path, "./dist");

        // --- Phase 4: Frontend env var overrides ---
        let frontend_yaml = r#"
server:
  port: 8080
  serve_frontend: true
  frontend_path: "./dist"
"#;
        let dir2 = tempfile::tempdir().unwrap();
        let frontend_file = dir2.path().join("config.yaml");
        let mut f2 = std::fs::File::create(&frontend_file).unwrap();
        f2.write_all(frontend_yaml.as_bytes()).unwrap();

        clear_env();
        std::env::set_var("SERVE_FRONTEND", "false");
        std::env::set_var("FRONTEND_PATH", "/custom/dist");

        let config = Config::from_yaml_and_env(Some(&frontend_file)).unwrap();
        assert!(!config.serve_frontend);
        assert_eq!(config.frontend_path, "/custom/dist");

        clear_env();

        // Without env overrides → YAML values used
        let config = Config::from_yaml_and_env(Some(&frontend_file)).unwrap();
        assert!(config.serve_frontend);
        assert_eq!(config.frontend_path, "./dist");

        clear_env();
    }

    // ========================================================================
    // New auth config format tests
    // ========================================================================

    #[test]
    fn test_new_auth_format_root_account_only() {
        let yaml = r#"
auth:
  jwt_secret: "super-secret-key-min-32-characters!"
  root_account:
    email: "admin@example.com"
    name: "Admin"
    password_hash: "$2b$12$LJ3m4ys1fFNwNkfMjkLx3u"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        let auth = config.auth.unwrap();
        assert!(auth.has_password_auth());
        assert!(!auth.has_oidc());
        assert!(!auth.allow_registration);

        let root = auth.root_account.unwrap();
        assert_eq!(root.email, "admin@example.com");
        assert_eq!(root.name, "Admin");
        assert!(root.password_hash.starts_with("$2b$"));
    }

    #[test]
    fn test_new_auth_format_oidc_only() {
        let yaml = r#"
auth:
  jwt_secret: "super-secret-key-min-32-characters!"
  oidc:
    discovery_url: "https://accounts.google.com/.well-known/openid-configuration"
    client_id: "my-client-id"
    client_secret: "my-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Google"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        let auth = config.auth.unwrap();
        assert!(!auth.has_password_auth());
        assert!(auth.has_oidc());

        let oidc = auth.effective_oidc().unwrap();
        assert_eq!(oidc.client_id, "my-client-id");
        assert_eq!(oidc.provider_name, "Google");
        assert_eq!(oidc.scopes, "openid email profile"); // default
    }

    #[test]
    fn test_new_auth_format_both_providers() {
        let yaml = r#"
auth:
  jwt_secret: "super-secret-key-min-32-characters!"
  allow_registration: true
  allowed_email_domain: "example.com"
  frontend_url: "http://localhost:3000"
  root_account:
    email: "admin@example.com"
    name: "Admin"
    password_hash: "plaintext-will-be-hashed"
  oidc:
    client_id: "oidc-client"
    client_secret: "oidc-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Okta"
    auth_endpoint: "https://okta.example.com/authorize"
    token_endpoint: "https://okta.example.com/token"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        let auth = config.auth.unwrap();
        assert!(auth.has_password_auth());
        assert!(auth.has_oidc());
        assert!(auth.allow_registration);
        assert_eq!(auth.allowed_email_domain, Some("example.com".into()));
        assert_eq!(auth.frontend_url, Some("http://localhost:3000".into()));

        let oidc = auth.effective_oidc().unwrap();
        assert_eq!(oidc.provider_name, "Okta");
        assert_eq!(
            oidc.auth_endpoint,
            Some("https://okta.example.com/authorize".into())
        );
    }

    #[test]
    fn test_retro_compat_google_format() {
        // Legacy format: google_* fields without oidc section
        let yaml = r#"
auth:
  google_client_id: "legacy-id.apps.googleusercontent.com"
  google_client_secret: "legacy-secret"
  google_redirect_uri: "http://localhost:3000/auth/callback"
  jwt_secret: "super-secret-key-min-32-characters!"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        let auth = config.auth.unwrap();

        // No explicit oidc section
        assert!(auth.oidc.is_none());
        // But effective_oidc() should build from legacy fields
        assert!(auth.has_oidc());
        let oidc = auth.effective_oidc().unwrap();
        assert_eq!(oidc.client_id, "legacy-id.apps.googleusercontent.com");
        assert_eq!(oidc.client_secret, "legacy-secret");
        assert_eq!(oidc.provider_name, "Google");
        assert!(oidc.auth_endpoint.is_some()); // Google hardcoded URLs
    }

    #[test]
    fn test_oidc_takes_priority_over_legacy_google() {
        // When both oidc and google_* are present, oidc wins
        let yaml = r#"
auth:
  jwt_secret: "super-secret-key-min-32-characters!"
  google_client_id: "legacy-id"
  google_client_secret: "legacy-secret"
  google_redirect_uri: "http://localhost/callback"
  oidc:
    client_id: "explicit-oidc-id"
    client_secret: "explicit-oidc-secret"
    redirect_uri: "http://localhost:3000/auth/callback"
    provider_name: "Microsoft"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        let auth = config.auth.unwrap();
        let oidc = auth.effective_oidc().unwrap();
        // Explicit oidc section takes priority
        assert_eq!(oidc.client_id, "explicit-oidc-id");
        assert_eq!(oidc.provider_name, "Microsoft");
    }

    #[test]
    fn test_allow_registration_default_false() {
        let yaml = r#"
auth:
  jwt_secret: "super-secret-key-min-32-characters!"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        let auth = config.auth.unwrap();
        assert!(!auth.allow_registration);
    }

    #[test]
    fn test_no_auth_section_means_open_access() {
        // No auth section at all → auth_config is None → open access mode
        let yaml = r#"
server:
  port: 8080
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.auth.is_none());
        // When auth is None, the middleware should allow all requests (no-auth mode)
    }

    // ========================================================================
    // Frontend serving config tests
    // ========================================================================

    #[test]
    fn test_frontend_config_defaults() {
        let config = ServerYamlConfig::default();
        assert!(config.serve_frontend);
        assert_eq!(config.frontend_path, "./dist");
    }

    #[test]
    fn test_frontend_config_from_yaml() {
        let yaml = r#"
server:
  port: 8080
  serve_frontend: false
  frontend_path: "/var/www/orchestrator/dist"
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!config.server.serve_frontend);
        assert_eq!(config.server.frontend_path, "/var/www/orchestrator/dist");
    }

    #[test]
    fn test_frontend_config_defaults_when_absent() {
        // serve_frontend and frontend_path should default when not in YAML
        let yaml = r#"
server:
  port: 9090
"#;
        let config: YamlConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.server.serve_frontend); // default true
        assert_eq!(config.server.frontend_path, "./dist"); // default
    }

}
