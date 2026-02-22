//! Chat configuration

use nexus_claude::PermissionMode;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Permission configuration for the chat system.
/// Groups the permission mode and tool allow/disallow patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionConfig {
    /// Permission mode: "default", "acceptEdits", "plan", "bypassPermissions"
    #[serde(default = "PermissionConfig::default_mode")]
    pub mode: String,
    /// Tool patterns to explicitly allow (e.g. "Bash(git *)", "mcp__project-orchestrator__*").
    /// Defaults to [`DEFAULT_ALLOWED_TOOLS`] when absent from config, ensuring MCP tools
    /// are usable out of the box.
    #[serde(default = "PermissionConfig::default_allowed_tools")]
    pub allowed_tools: Vec<String>,
    /// Tool patterns to explicitly disallow (e.g. "Bash(rm -rf *)", "Bash(sudo *)")
    #[serde(default)]
    pub disallowed_tools: Vec<String>,
}

impl PermissionConfig {
    fn default_mode() -> String {
        "default".into()
    }

    /// Convert the string mode to the Nexus SDK `PermissionMode` enum.
    /// Falls back to `Default` for unknown values (safe-by-default).
    pub fn to_nexus_mode(&self) -> PermissionMode {
        match self.mode.as_str() {
            "default" => PermissionMode::Default,
            "acceptEdits" => PermissionMode::AcceptEdits,
            "plan" => PermissionMode::Plan,
            "bypassPermissions" => PermissionMode::BypassPermissions,
            _ => {
                tracing::warn!(
                    mode = %self.mode,
                    "Unknown permission mode, falling back to Default"
                );
                PermissionMode::Default
            }
        }
    }

    /// List of valid permission mode strings.
    pub fn valid_modes() -> &'static [&'static str] {
        &["default", "acceptEdits", "plan", "bypassPermissions"]
    }

    /// Check if the given mode string is valid.
    pub fn is_valid_mode(mode: &str) -> bool {
        Self::valid_modes().contains(&mode)
    }

    /// Default allowed tool patterns (MCP tools pre-approved out of the box).
    pub fn default_allowed_tools() -> Vec<String> {
        DEFAULT_ALLOWED_TOOLS
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
}

/// Default allowed tool patterns applied when no explicit configuration is provided.
///
/// These ensure that MCP tools from the Project Orchestrator server are usable
/// out of the box, without requiring the user to manually add them via the
/// chat settings page or config.yaml.
pub const DEFAULT_ALLOWED_TOOLS: &[&str] = &["mcp__project-orchestrator__*"];

impl Default for PermissionConfig {
    fn default() -> Self {
        Self {
            mode: Self::default_mode(),
            allowed_tools: DEFAULT_ALLOWED_TOOLS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            disallowed_tools: Vec::new(),
        }
    }
}

/// Retry configuration for transient API errors (5xx).
///
/// When the Anthropic API returns a retryable error (e.g., 500 api_error,
/// 529 overloaded_error), the chat system automatically retries with
/// exponential backoff. Only retries when no tokens have been emitted yet.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (default: 3)
    pub max_attempts: u32,
    /// Initial delay in milliseconds before the first retry (default: 1000)
    pub initial_delay_ms: u64,
    /// Backoff multiplier applied after each retry (default: 2.0).
    /// Delay = initial_delay_ms × multiplier^(attempt-1)
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Read retry config from environment variables with fallback to defaults.
    pub fn from_env() -> Self {
        Self {
            max_attempts: std::env::var("CHAT_RETRY_MAX_ATTEMPTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
            initial_delay_ms: std::env::var("CHAT_RETRY_INITIAL_DELAY_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000),
            backoff_multiplier: std::env::var("CHAT_RETRY_BACKOFF_MULTIPLIER")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2.0),
        }
    }

    /// Calculate the delay for a given attempt (1-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        (self.initial_delay_ms as f64 * self.backoff_multiplier.powi(attempt as i32 - 1)) as u64
    }
}

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
    /// Permission configuration (mode + allowed/disallowed tool patterns)
    pub permission: PermissionConfig,
    /// Whether to use an oneshot LLM call to refine project context in the system prompt.
    /// When `false`, falls back to static markdown rendering (useful in tests to avoid
    /// spawning a real Claude CLI subprocess).
    pub enable_oneshot_refinement: bool,
    /// Whether auto-continue is enabled by default for new sessions.
    /// When `true`, the backend automatically sends "Continue" after error_max_turns.
    /// Can be toggled per-session via WebSocket.
    pub auto_continue: bool,
    /// Retry configuration for transient API errors (5xx)
    pub retry: RetryConfig,
}

impl ChatConfig {
    /// Create config from environment, auto-detecting the mcp_server binary path
    pub fn from_env() -> Self {
        let mcp_server_path = Self::detect_mcp_server_path();

        Self {
            mcp_server_path,
            default_model: std::env::var("CHAT_DEFAULT_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-6".into()),
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
            permission: PermissionConfig {
                mode: std::env::var("CHAT_PERMISSION_MODE").unwrap_or_else(|_| "default".into()),
                allowed_tools: std::env::var("CHAT_ALLOWED_TOOLS")
                    .ok()
                    .map(|s| {
                        s.split(',')
                            .map(|t| t.trim().to_string())
                            .filter(|t| !t.is_empty())
                            .collect()
                    })
                    .unwrap_or_else(PermissionConfig::default_allowed_tools),
                disallowed_tools: std::env::var("CHAT_DISALLOWED_TOOLS")
                    .ok()
                    .map(|s| {
                        s.split(',')
                            .map(|t| t.trim().to_string())
                            .filter(|t| !t.is_empty())
                            .collect()
                    })
                    .unwrap_or_default(),
            },
            enable_oneshot_refinement: std::env::var("CHAT_ENABLE_ONESHOT_REFINEMENT")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            auto_continue: std::env::var("CHAT_AUTO_CONTINUE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            retry: RetryConfig::from_env(),
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
            default_model: "claude-sonnet-4-6".into(),
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
            permission: PermissionConfig::default(),
            enable_oneshot_refinement: true,
            auto_continue: false,
            retry: RetryConfig::default(),
        };

        assert_eq!(config.default_model, "claude-sonnet-4-6");
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
        std::env::remove_var("CHAT_PERMISSION_MODE");
        std::env::remove_var("CHAT_ALLOWED_TOOLS");
        std::env::remove_var("CHAT_DISALLOWED_TOOLS");

        let config = ChatConfig::from_env();
        assert_eq!(config.default_model, "claude-sonnet-4-6");
        assert_eq!(config.max_sessions, 10);
        assert_eq!(config.session_timeout.as_secs(), 1800);
        // Permission defaults — MCP tools are pre-approved out of the box
        assert_eq!(config.permission.mode, "default");
        assert_eq!(
            config.permission.allowed_tools,
            vec!["mcp__project-orchestrator__*"]
        );
        assert!(config.permission.disallowed_tools.is_empty());

        // Phase 2: custom values
        std::env::set_var("CHAT_DEFAULT_MODEL", "claude-sonnet-4-20250514");
        std::env::set_var("CHAT_MAX_SESSIONS", "5");
        std::env::set_var("CHAT_SESSION_TIMEOUT_SECS", "600");
        std::env::set_var("MCP_SERVER_PATH", "/custom/path/mcp_server");
        std::env::set_var("CHAT_PERMISSION_MODE", "default");
        std::env::set_var(
            "CHAT_ALLOWED_TOOLS",
            "Bash(git *),Read,mcp__project-orchestrator__*",
        );
        std::env::set_var("CHAT_DISALLOWED_TOOLS", "Bash(rm -rf *), Bash(sudo *)");

        let config = ChatConfig::from_env();
        assert_eq!(config.default_model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_sessions, 5);
        assert_eq!(config.session_timeout.as_secs(), 600);
        assert_eq!(
            config.mcp_server_path,
            PathBuf::from("/custom/path/mcp_server")
        );
        assert_eq!(config.permission.mode, "default");
        assert_eq!(
            config.permission.allowed_tools,
            vec!["Bash(git *)", "Read", "mcp__project-orchestrator__*"]
        );
        assert_eq!(
            config.permission.disallowed_tools,
            vec!["Bash(rm -rf *)", "Bash(sudo *)"]
        );

        // Phase 2b: CSV parsing edge cases for tools
        std::env::set_var("CHAT_ALLOWED_TOOLS", "Bash(git *), Read, Edit");
        std::env::remove_var("CHAT_DISALLOWED_TOOLS");
        let config = ChatConfig::from_env();
        assert_eq!(
            config.permission.allowed_tools,
            vec!["Bash(git *)", "Read", "Edit"]
        );
        assert!(config.permission.disallowed_tools.is_empty());

        // Empty value should produce empty vec
        std::env::set_var("CHAT_ALLOWED_TOOLS", "");
        let config = ChatConfig::from_env();
        assert!(config.permission.allowed_tools.is_empty());

        // Phase 3: invalid value falls back to default
        std::env::set_var("CHAT_MAX_SESSIONS", "not_a_number");
        let config = ChatConfig::from_env();
        assert_eq!(config.max_sessions, 10);

        // Phase 4: Default trait (clear custom env vars first)
        std::env::remove_var("CHAT_DEFAULT_MODEL");
        std::env::remove_var("CHAT_MAX_SESSIONS");
        std::env::remove_var("CHAT_SESSION_TIMEOUT_SECS");
        std::env::remove_var("MCP_SERVER_PATH");
        std::env::remove_var("CHAT_PERMISSION_MODE");
        std::env::remove_var("CHAT_ALLOWED_TOOLS");
        std::env::remove_var("CHAT_DISALLOWED_TOOLS");
        let config = ChatConfig::default();
        assert!(!config.default_model.is_empty());
        assert!(config.max_sessions > 0);
        assert_eq!(config.permission.mode, "default");

        // Cleanup
        std::env::remove_var("CHAT_DEFAULT_MODEL");
        std::env::remove_var("CHAT_MAX_SESSIONS");
        std::env::remove_var("CHAT_SESSION_TIMEOUT_SECS");
        std::env::remove_var("MCP_SERVER_PATH");
        std::env::remove_var("CHAT_PERMISSION_MODE");
        std::env::remove_var("CHAT_ALLOWED_TOOLS");
        std::env::remove_var("CHAT_DISALLOWED_TOOLS");
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
            permission: PermissionConfig::default(),
            enable_oneshot_refinement: true,
            auto_continue: false,
            retry: RetryConfig::default(),
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
            permission: PermissionConfig::default(),
            enable_oneshot_refinement: true,
            auto_continue: false,
            retry: RetryConfig::default(),
        };

        let json = config.mcp_server_config();
        let server = &json["project-orchestrator"];
        assert_eq!(server["command"], "/path/to/mcp_server");
        // NATS_URL should not be present when not configured
        assert!(server["env"]["NATS_URL"].is_null());
    }

    #[test]
    fn test_permission_config_defaults() {
        let config = PermissionConfig::default();
        assert_eq!(config.mode, "default");
        // MCP tools are pre-approved by default
        assert_eq!(config.allowed_tools, vec!["mcp__project-orchestrator__*"]);
        assert!(config.disallowed_tools.is_empty());
    }

    #[test]
    fn test_permission_config_to_nexus_mode() {
        use nexus_claude::PermissionMode;

        // All 4 known modes
        let config = PermissionConfig {
            mode: "default".into(),
            ..Default::default()
        };
        assert!(matches!(config.to_nexus_mode(), PermissionMode::Default));

        let config = PermissionConfig {
            mode: "acceptEdits".into(),
            ..Default::default()
        };
        assert!(matches!(
            config.to_nexus_mode(),
            PermissionMode::AcceptEdits
        ));

        let config = PermissionConfig {
            mode: "plan".into(),
            ..Default::default()
        };
        assert!(matches!(config.to_nexus_mode(), PermissionMode::Plan));

        let config = PermissionConfig {
            mode: "bypassPermissions".into(),
            ..Default::default()
        };
        assert!(matches!(
            config.to_nexus_mode(),
            PermissionMode::BypassPermissions
        ));

        // Unknown mode falls back to Default (safe-by-default)
        let config = PermissionConfig {
            mode: "nonsense".into(),
            ..Default::default()
        };
        assert!(matches!(config.to_nexus_mode(), PermissionMode::Default));

        let config = PermissionConfig {
            mode: "".into(),
            ..Default::default()
        };
        assert!(matches!(config.to_nexus_mode(), PermissionMode::Default));
    }

    #[test]
    fn test_permission_config_valid_modes() {
        assert!(PermissionConfig::is_valid_mode("default"));
        assert!(PermissionConfig::is_valid_mode("acceptEdits"));
        assert!(PermissionConfig::is_valid_mode("plan"));
        assert!(PermissionConfig::is_valid_mode("bypassPermissions"));
        assert!(!PermissionConfig::is_valid_mode("unknown"));
        assert!(!PermissionConfig::is_valid_mode(""));
        assert!(!PermissionConfig::is_valid_mode("Default")); // case-sensitive
    }

    #[test]
    fn test_permission_config_serde_roundtrip() {
        let config = PermissionConfig {
            mode: "acceptEdits".into(),
            allowed_tools: vec!["Bash(git *)".into(), "Read".into()],
            disallowed_tools: vec!["Bash(rm -rf *)".into()],
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PermissionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.mode, "acceptEdits");
        assert_eq!(deserialized.allowed_tools, vec!["Bash(git *)", "Read"]);
        assert_eq!(deserialized.disallowed_tools, vec!["Bash(rm -rf *)"]);
    }

    #[test]
    fn test_permission_config_serde_defaults() {
        // Empty JSON should use defaults — MCP tools pre-approved
        let config: PermissionConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.mode, "default");
        assert_eq!(config.allowed_tools, vec!["mcp__project-orchestrator__*"]);
        assert!(config.disallowed_tools.is_empty());

        // Partial JSON should fill defaults
        let config: PermissionConfig = serde_json::from_str(r#"{"mode":"default"}"#).unwrap();
        assert_eq!(config.mode, "default");
        assert_eq!(config.allowed_tools, vec!["mcp__project-orchestrator__*"]);

        // Explicit empty array should be respected (user intent to disable)
        let config: PermissionConfig =
            serde_json::from_str(r#"{"mode":"default","allowed_tools":[]}"#).unwrap();
        assert_eq!(config.mode, "default");
        assert!(config.allowed_tools.is_empty());
    }

    // ====================================================================
    // RetryConfig
    // ====================================================================

    #[test]
    fn test_retry_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay_ms, 1000);
        assert!((config.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_retry_config_delay_calculation() {
        let config = RetryConfig {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
        };
        assert_eq!(config.delay_for_attempt(1), 1000); // 1000 * 2^0
        assert_eq!(config.delay_for_attempt(2), 2000); // 1000 * 2^1
        assert_eq!(config.delay_for_attempt(3), 4000); // 1000 * 2^2
    }

    #[test]
    fn test_retry_config_from_env() {
        // Clear any existing vars
        std::env::remove_var("CHAT_RETRY_MAX_ATTEMPTS");
        std::env::remove_var("CHAT_RETRY_INITIAL_DELAY_MS");
        std::env::remove_var("CHAT_RETRY_BACKOFF_MULTIPLIER");

        // Defaults
        let config = RetryConfig::from_env();
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.initial_delay_ms, 1000);

        // Custom values
        std::env::set_var("CHAT_RETRY_MAX_ATTEMPTS", "5");
        std::env::set_var("CHAT_RETRY_INITIAL_DELAY_MS", "500");
        std::env::set_var("CHAT_RETRY_BACKOFF_MULTIPLIER", "1.5");
        let config = RetryConfig::from_env();
        assert_eq!(config.max_attempts, 5);
        assert_eq!(config.initial_delay_ms, 500);
        assert!((config.backoff_multiplier - 1.5).abs() < f64::EPSILON);

        // Cleanup
        std::env::remove_var("CHAT_RETRY_MAX_ATTEMPTS");
        std::env::remove_var("CHAT_RETRY_INITIAL_DELAY_MS");
        std::env::remove_var("CHAT_RETRY_BACKOFF_MULTIPLIER");
    }
}
