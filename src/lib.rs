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

#[cfg(test)]
pub(crate) mod test_helpers;

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// YAML config structs (deserialization targets)
// ============================================================================

/// Top-level YAML configuration file structure
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct YamlConfig {
    pub server: ServerYamlConfig,
    pub neo4j: Neo4jYamlConfig,
    pub meilisearch: MeilisearchYamlConfig,
    pub chat: ChatYamlConfig,
    /// Auth section — if absent, auth_config will be None (deny-by-default)
    pub auth: Option<AuthConfig>,
}

/// Server configuration section
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerYamlConfig {
    pub port: u16,
    pub workspace_path: String,
}

impl Default for ServerYamlConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            workspace_path: ".".into(),
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

/// Authentication configuration — Google OAuth + JWT
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    pub google_client_id: String,
    pub google_client_secret: String,
    pub google_redirect_uri: String,
    pub jwt_secret: String,
    /// JWT token lifetime in seconds (default: 28800 = 8h)
    #[serde(default = "default_jwt_expiry")]
    pub jwt_expiry_secs: u64,
    /// Optional domain restriction (e.g. "ffs.holdings")
    pub allowed_email_domain: Option<String>,
    /// Frontend URL for CORS and redirects (e.g. "http://localhost:3000")
    pub frontend_url: Option<String>,
}

fn default_jwt_expiry() -> u64 {
    28800 // 8 hours
}

// ============================================================================
// Runtime config (what the application actually uses)
// ============================================================================

/// Application configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub meilisearch_url: String,
    pub meilisearch_key: String,
    pub workspace_path: String,
    pub server_port: u16,
    /// Auth config — None means deny-by-default (no auth section in YAML)
    pub auth_config: Option<AuthConfig>,
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
            neo4j_uri: std::env::var("NEO4J_URI").unwrap_or(yaml.neo4j.uri),
            neo4j_user: std::env::var("NEO4J_USER").unwrap_or(yaml.neo4j.user),
            neo4j_password: std::env::var("NEO4J_PASSWORD").unwrap_or(yaml.neo4j.password),
            meilisearch_url: std::env::var("MEILISEARCH_URL").unwrap_or(yaml.meilisearch.url),
            meilisearch_key: std::env::var("MEILISEARCH_KEY").unwrap_or(yaml.meilisearch.key),
            workspace_path: std::env::var("WORKSPACE_PATH").unwrap_or(yaml.server.workspace_path),
            server_port: std::env::var("SERVER_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(yaml.server.port),
            auth_config: yaml.auth,
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
        assert_eq!(auth.google_client_id, "123.apps.googleusercontent.com");
        assert_eq!(auth.jwt_expiry_secs, 3600);
        assert_eq!(auth.allowed_email_domain, Some("ffs.holdings".into()));
        assert_eq!(auth.frontend_url, Some("http://localhost:3000".into()));
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
    }
}
