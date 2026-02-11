//! Chat configuration

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for the chat system
#[derive(Debug, Clone)]
pub struct ChatConfig {
    /// Path to the MCP server binary
    pub mcp_server_path: PathBuf,
    /// Default model to use when not specified in request
    pub default_model: String,
    /// Maximum number of concurrent active sessions
    pub max_sessions: usize,
    /// Timeout after which inactive sessions are closed (subprocess freed)
    pub session_timeout: Duration,
    /// Neo4j connection details for MCP server env
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    /// Meilisearch connection details for MCP server env
    pub meilisearch_url: String,
    pub meilisearch_key: String,
    /// NATS URL for inter-process event sync (MCP ↔ desktop)
    pub nats_url: Option<String>,
    /// Maximum number of agentic turns (tool calls) per message
    pub max_turns: i32,
    /// Model used for the oneshot prompt builder (context refinement)
    pub prompt_builder_model: String,
}

impl ChatConfig {
    /// Create config from environment, auto-detecting the mcp_server binary path
    pub fn from_env() -> Self {
        let mcp_server_path = Self::detect_mcp_server_path();

        Self {
            mcp_server_path,
            default_model: std::env::var("CHAT_DEFAULT_MODEL")
                .unwrap_or_else(|_| "claude-opus-4-6".into()),
            max_sessions: std::env::var("CHAT_MAX_SESSIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            session_timeout: Duration::from_secs(
                std::env::var("CHAT_SESSION_TIMEOUT_SECS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1800), // 30 minutes
            ),
            neo4j_uri: std::env::var("NEO4J_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".into()),
            neo4j_user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
            neo4j_password: std::env::var("NEO4J_PASSWORD")
                .unwrap_or_else(|_| "orchestrator123".into()),
            meilisearch_url: std::env::var("MEILISEARCH_URL")
                .unwrap_or_else(|_| "http://localhost:7700".into()),
            meilisearch_key: std::env::var("MEILISEARCH_KEY")
                .unwrap_or_else(|_| "orchestrator-meili-key-change-me".into()),
            nats_url: std::env::var("NATS_URL").ok(),
            max_turns: std::env::var("CHAT_MAX_TURNS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(50),
            prompt_builder_model: std::env::var("PROMPT_BUILDER_MODEL")
                .unwrap_or_else(|_| "claude-opus-4-6".into()),
        }
    }

    fn detect_mcp_server_path() -> PathBuf {
        // Try environment variable first
        if let Ok(path) = std::env::var("MCP_SERVER_PATH") {
            return PathBuf::from(path);
        }

        // Try relative to current executable
        if let Ok(exe) = std::env::current_exe() {
            let dir = exe.parent().unwrap_or(exe.as_ref());
            let candidate = dir.join("mcp_server");
            if candidate.exists() {
                return candidate;
            }
        }

        // Fallback
        PathBuf::from("mcp_server")
    }

    /// Build the MCP server config JSON for ClaudeCodeOptions
    pub fn mcp_server_config(&self) -> serde_json::Value {
        let mut env = serde_json::json!({
            "NEO4J_URI": self.neo4j_uri,
            "NEO4J_USER": self.neo4j_user,
            "NEO4J_PASSWORD": self.neo4j_password,
            "MEILISEARCH_URL": self.meilisearch_url,
            "MEILISEARCH_KEY": self.meilisearch_key
        });

        // Forward NATS_URL so the spawned MCP server can sync events back
        // to the desktop app. Without this, CRUD events from chat sessions
        // are invisible to the UI when launched from the macOS dock.
        if let Some(ref nats_url) = self.nats_url {
            env["NATS_URL"] = serde_json::Value::String(nats_url.clone());
        }

        serde_json::json!({
            "project-orchestrator": {
                "command": self.mcp_server_path.to_string_lossy(),
                "env": env
            }
        })
    }
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ChatConfig {
            mcp_server_path: PathBuf::from("/usr/bin/mcp_server"),
            default_model: "claude-opus-4-6".into(),
            max_sessions: 10,
            session_timeout: Duration::from_secs(1800),
            neo4j_uri: "bolt://localhost:7687".into(),
            neo4j_user: "neo4j".into(),
            neo4j_password: "test".into(),
            meilisearch_url: "http://localhost:7700".into(),
            meilisearch_key: "test-key".into(),
            nats_url: None,
            max_turns: 10,
            prompt_builder_model: "claude-opus-4-6".into(),
        };

        assert_eq!(config.default_model, "claude-opus-4-6");
        assert_eq!(config.max_sessions, 10);
        assert_eq!(config.session_timeout.as_secs(), 1800);
    }

    /// Combined env var test to avoid parallel test race conditions.
    /// Tests from_env() defaults, custom overrides, invalid fallback, and Default trait.
    #[test]
    fn test_from_env_lifecycle() {
        // Phase 1: defaults (clear any chat env vars first)
        std::env::remove_var("CHAT_DEFAULT_MODEL");
        std::env::remove_var("CHAT_MAX_SESSIONS");
        std::env::remove_var("CHAT_SESSION_TIMEOUT_SECS");
        std::env::remove_var("MCP_SERVER_PATH");

        let config = ChatConfig::from_env();
        assert_eq!(config.default_model, "claude-opus-4-6");
        assert_eq!(config.max_sessions, 10);
        assert_eq!(config.session_timeout.as_secs(), 1800);

        // Phase 2: custom values
        std::env::set_var("CHAT_DEFAULT_MODEL", "claude-sonnet-4-20250514");
        std::env::set_var("CHAT_MAX_SESSIONS", "5");
        std::env::set_var("CHAT_SESSION_TIMEOUT_SECS", "600");
        std::env::set_var("MCP_SERVER_PATH", "/custom/path/mcp_server");

        let config = ChatConfig::from_env();
        assert_eq!(config.default_model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_sessions, 5);
        assert_eq!(config.session_timeout.as_secs(), 600);
        assert_eq!(
            config.mcp_server_path,
            PathBuf::from("/custom/path/mcp_server")
        );

        // Phase 3: invalid value falls back to default
        std::env::set_var("CHAT_MAX_SESSIONS", "not_a_number");
        let config = ChatConfig::from_env();
        assert_eq!(config.max_sessions, 10);

        // Phase 4: Default trait
        let config = ChatConfig::default();
        assert!(!config.default_model.is_empty());
        assert!(config.max_sessions > 0);

        // Cleanup
        std::env::remove_var("CHAT_DEFAULT_MODEL");
        std::env::remove_var("CHAT_MAX_SESSIONS");
        std::env::remove_var("CHAT_SESSION_TIMEOUT_SECS");
        std::env::remove_var("MCP_SERVER_PATH");
    }

    #[test]
    fn test_mcp_server_config_json() {
        let config = ChatConfig {
            mcp_server_path: PathBuf::from("/path/to/mcp_server"),
            default_model: "claude-opus-4-6".into(),
            max_sessions: 10,
            session_timeout: Duration::from_secs(1800),
            neo4j_uri: "bolt://localhost:7687".into(),
            neo4j_user: "neo4j".into(),
            neo4j_password: "pass".into(),
            meilisearch_url: "http://localhost:7700".into(),
            meilisearch_key: "key".into(),
            nats_url: Some("nats://localhost:4222".into()),
            max_turns: 10,
            prompt_builder_model: "claude-opus-4-6".into(),
        };

        let json = config.mcp_server_config();
        let server = &json["project-orchestrator"];
        assert_eq!(server["command"], "/path/to/mcp_server");
        assert_eq!(server["env"]["NEO4J_URI"], "bolt://localhost:7687");
        assert_eq!(server["env"]["NATS_URL"], "nats://localhost:4222");
    }

    #[test]
    fn test_mcp_server_config_without_nats() {
        let config = ChatConfig {
            mcp_server_path: PathBuf::from("/path/to/mcp_server"),
            default_model: "claude-opus-4-6".into(),
            max_sessions: 10,
            session_timeout: Duration::from_secs(1800),
            neo4j_uri: "bolt://localhost:7687".into(),
            neo4j_user: "neo4j".into(),
            neo4j_password: "pass".into(),
            meilisearch_url: "http://localhost:7700".into(),
            meilisearch_key: "key".into(),
            nats_url: None,
            max_turns: 10,
            prompt_builder_model: "claude-opus-4-6".into(),
        };

        let json = config.mcp_server_config();
        let server = &json["project-orchestrator"];
        assert_eq!(server["command"], "/path/to/mcp_server");
        // NATS_URL should not be present when not configured
        assert!(server["env"]["NATS_URL"].is_null());
    }
}
