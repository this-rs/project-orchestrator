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
        // Legacy Claude strings AND the neutral names of the agent contract
        // (decision A43): `ask`, `auto_edits`, `plan_only`, `trust`.
        match self.mode.as_str() {
            "default" | "ask" | "manual" | "dontAsk" => PermissionMode::Default,
            "acceptEdits" | "auto_edits" | "auto" => PermissionMode::AcceptEdits,
            "plan" | "plan_only" => PermissionMode::Plan,
            "bypassPermissions" | "trust" => PermissionMode::BypassPermissions,
            _ => {
                tracing::warn!(
                    mode = %self.mode,
                    "Unknown permission mode, falling back to Default"
                );
                PermissionMode::Default
            }
        }
    }

    /// Accepted permission mode strings: the Claude strings the API always
    /// took, plus the neutral names of the agent contract (A43). The legacy
    /// `auto`, `dontAsk` and `manual` are accepted too (Claude Code knows them).
    pub fn valid_modes() -> &'static [&'static str] {
        &[
            "default",
            "acceptEdits",
            "plan",
            "bypassPermissions",
            "auto",
            "dontAsk",
            "manual",
            "ask",
            "auto_edits",
            "plan_only",
            "trust",
        ]
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
    /// Which engine drives a session: the historical Claude Code client
    /// (`legacy`, default) or the provider-neutral `AgentSession` (`agent`).
    pub provider_path: ProviderPath,
    /// Path to the MCP server binary
    pub mcp_server_path: PathBuf,
    /// The `nexus-tools` executable (Read, Edit, Bash... over MCP) attached to every
    /// native session as its `nexus` server (B40). `None`: not found at start, the
    /// native sessions have the project-orchestrator tools only.
    pub nexus_tools_path: Option<PathBuf>,
    /// The browser (Obscura, N23) found on the `PATH` at start. Found is not authorised:
    /// a project must authorise it (`nexus_tools::BROWSER_KEY`) before a session gets it.
    pub nexus_browser_path: Option<PathBuf>,
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
    /// Permission configuration (mode + allowed/disallowed tool patterns)
    pub permission: PermissionConfig,
    /// Whether auto-continue is enabled by default for new sessions.
    /// When `true`, the backend automatically sends "Continue" after error_max_turns.
    /// Can be toggled per-session via WebSocket.
    pub auto_continue: bool,
    /// Retry configuration for transient API errors (5xx)
    pub retry: RetryConfig,
    /// PATH to inject into the Claude Code subprocess. When Some, overrides the inherited PATH.
    /// Colon-separated on Unix (e.g. "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin").
    pub process_path: Option<String>,
    /// Absolute path to the Claude CLI binary. When Some, skips find_claude_cli() lookup.
    pub claude_cli_path: Option<String>,
    /// Enable automatic CLI version updates on startup (default: false).
    pub auto_update_cli: bool,
    /// Enable automatic Tauri application updates on startup (default: true).
    pub auto_update_app: bool,
    /// JWT secret for generating session tokens (MCP auth).
    /// When Some (auth enabled), build_options() generates a session JWT
    /// and injects PO_AUTH_TOKEN + PO_SERVER_URL into the MCP server env.
    pub jwt_secret: Option<String>,
    /// Server port for PO_SERVER_URL derivation (always 127.0.0.1:{port}).
    pub server_port: u16,
    /// Expiration duration for MCP session tokens in seconds (default: 24h = 86400s).
    pub session_token_expiry_secs: u64,
    /// The third-party MCP tools the operator declares READ-ONLY, by exact name
    /// (`mcp__<server>__<tool>`, no pattern): the only third-party MCP tools a permission
    /// approved "for the session" may cover (`chat::session_grants`, the identical call).
    /// Any other is allowed once at a time. From `CHAT_READ_ONLY_MCP_TOOLS` (comma
    /// separated); empty by default.
    ///
    /// A declaration binds a NAME, not a server: the Claude Code CLI also loads the
    /// project's `.mcp.json`, where a server may take the name of one configured elsewhere.
    /// Declare only tools of servers no project can redefine (the backend's own, or names no
    /// project uses). No strict MCP configuration is passed to the CLI: it would drop every
    /// server configured outside the backend, the very ones a declaration is for
    /// (`docs/guides/chat-websocket.md`).
    pub read_only_mcp_tools: Vec<String>,
}

/// Environment variable selecting the [`ProviderPath`].
pub const PROVIDER_PATH_VAR: &str = "CHAT_PROVIDER_PATH";

/// Variable naming the `nexus-tools` executable of the native sessions (B40).
pub const NEXUS_TOOLS_PATH_VAR: &str = "NEXUS_TOOLS_PATH";

/// Variable listing the third-party MCP tools declared read-only
/// ([`ChatConfig::read_only_mcp_tools`]).
pub const READ_ONLY_MCP_TOOLS_VAR: &str = "CHAT_READ_ONLY_MCP_TOOLS";

/// Engine that drives Claude Code sessions (decision A18 / task B38).
///
/// Routing is HYBRID: Claude Code stays on the historical engine (hooks, message
/// queue, auto-continue, retry, compaction, NATS, images) by default, and every
/// third-party provider is served by the agent engine whatever this says.
/// `Agent` only FORCES Claude Code onto the agent engine, to try it: the session
/// then says what it lost (`system_init.degraded_features`) and the server logs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProviderPath {
    /// The historical path: `InteractiveClient` straight on the Claude CLI.
    #[default]
    Legacy,
    /// The provider-neutral path: an `AgentSession` from the nexus registry.
    Agent,
}

impl ProviderPath {
    /// `legacy` or `agent` (case-insensitive). Anything else, or nothing, is
    /// `legacy`: a typo must never switch the engine.
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("agent") => Self::Agent,
            Some("legacy") | None => Self::Legacy,
            Some(other) => {
                tracing::warn!(value = other, "unknown CHAT_PROVIDER_PATH, using legacy");
                Self::Legacy
            }
        }
    }

    /// Stable name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Agent => "agent",
        }
    }
}

impl ChatConfig {
    /// Create config from environment, auto-detecting the mcp_server binary path
    pub fn from_env() -> Self {
        let mcp_server_path = Self::detect_mcp_server_path();

        Self {
            provider_path: ProviderPath::parse(std::env::var(PROVIDER_PATH_VAR).ok().as_deref()),
            mcp_server_path,
            nexus_tools_path: Self::detect_nexus_tools_path(),
            nexus_browser_path: nexus_claude::providers::native::BrowserTools::locate()
                .map(|browser| browser.program),
            default_model: std::env::var("CHAT_DEFAULT_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-5".into()),
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
            auto_continue: std::env::var("CHAT_AUTO_CONTINUE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            retry: RetryConfig::from_env(),
            process_path: std::env::var("CHAT_PROCESS_PATH").ok(),
            claude_cli_path: std::env::var("CLAUDE_CLI_PATH").ok(),
            auto_update_cli: std::env::var("CHAT_AUTO_UPDATE_CLI")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            auto_update_app: std::env::var("CHAT_AUTO_UPDATE_APP")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            jwt_secret: None, // Injected from Config.auth_config in lib.rs
            server_port: std::env::var("SERVER_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8080),
            session_token_expiry_secs: std::env::var("CHAT_SESSION_TOKEN_EXPIRY_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(86400), // 24 hours
            read_only_mcp_tools: Self::parse_read_only_mcp_tools(
                std::env::var(READ_ONLY_MCP_TOOLS_VAR).ok().as_deref(),
            ),
        }
    }

    /// The declared read-only MCP tools of [`READ_ONLY_MCP_TOOLS_VAR`]: comma separated,
    /// each an exact `mcp__<server>__<tool>` name. Anything else (a pattern, a server
    /// alone, a tool of the `nexus` server, whose built-ins have their own rules) is
    /// dropped with a warning: a declaration never widens beyond one named tool.
    pub fn parse_read_only_mcp_tools(value: Option<&str>) -> Vec<String> {
        let Some(value) = value else {
            return Vec::new();
        };
        value
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .filter(|name| {
                let exact = name
                    .strip_prefix("mcp__")
                    .and_then(|rest| rest.split_once("__"))
                    .is_some_and(|(server, tool)| {
                        !server.is_empty()
                            && !tool.is_empty()
                            && server != nexus_claude::providers::native::NEXUS_TOOLS_SERVER
                            && name.chars().all(|c| c.is_ascii_alphanumeric() || "_-.".contains(c))
                    });
                if !exact {
                    tracing::warn!(
                        tool = %name,
                        "{READ_ONLY_MCP_TOOLS_VAR}: not an exact mcp__<server>__<tool> name, ignored"
                    );
                }
                exact
            })
            .map(str::to_string)
            .collect()
    }

    /// Public accessor for MCP server path detection.
    ///
    /// Used by `setup_claude` to auto-detect the binary path when configuring
    /// Claude Code's mcp.json from the CLI `setup-claude` subcommand.
    pub fn detect_mcp_server_path_public() -> PathBuf {
        Self::detect_mcp_server_path()
    }

    /// Where `nexus-tools` is: `NEXUS_TOOLS_PATH` when set (an operator's choice,
    /// taken as is), else next to the server's executable, else on the `PATH`
    /// (`DefaultTools::locate` of nexus). Nothing downloads it: every release
    /// channel ships it next to the server's executable (archives, Homebrew,
    /// .deb/.rpm `/usr/bin`, Docker `/app`; the desktop app sets
    /// `NEXUS_TOOLS_PATH`), built at the pinned nexus revision with `tls` by
    /// `scripts/build-nexus-tools.sh`, which an operator can run too.
    pub fn detect_nexus_tools_path() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os(NEXUS_TOOLS_PATH_VAR).filter(|p| !p.is_empty()) {
            return Some(PathBuf::from(path));
        }
        nexus_claude::providers::native::DefaultTools::locate().map(|tools| tools.program)
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
    fn provider_path_defaults_to_legacy_and_a_typo_never_switches() {
        assert_eq!(ProviderPath::parse(None), ProviderPath::Legacy);
        assert_eq!(ProviderPath::parse(Some("legacy")), ProviderPath::Legacy);
        assert_eq!(ProviderPath::parse(Some(" AGENT ")), ProviderPath::Agent);
        assert_eq!(ProviderPath::parse(Some("agnet")), ProviderPath::Legacy);
        assert_eq!(ProviderPath::default().as_str(), "legacy");
    }

    #[test]
    fn a_read_only_mcp_tool_is_declared_by_its_exact_name_only() {
        assert!(ChatConfig::parse_read_only_mcp_tools(None).is_empty());
        assert_eq!(
            ChatConfig::parse_read_only_mcp_tools(Some(
                " mcp__acme__list , ,mcp__docs-srv__search_docs,mcp__acme__*,mcp__acme,\
                 mcp____x,mcp__nexus__Bash,Bash,acme__list,mcp__a__b c"
            )),
            vec!["mcp__acme__list", "mcp__docs-srv__search_docs"]
        );
    }

    #[test]
    fn test_default_config() {
        let config = ChatConfig {
            provider_path: Default::default(),
            mcp_server_path: PathBuf::from("/usr/bin/mcp_server"),
            nexus_tools_path: None,
            nexus_browser_path: None,
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
            permission: PermissionConfig::default(),
            auto_continue: false,
            retry: RetryConfig::default(),
            process_path: None,
            claude_cli_path: None,
            auto_update_cli: false,
            auto_update_app: true,
            jwt_secret: None,
            server_port: 8080,
            session_token_expiry_secs: 86400,
            read_only_mcp_tools: Vec::new(),
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
        std::env::remove_var("CHAT_PROCESS_PATH");
        std::env::remove_var("CLAUDE_CLI_PATH");
        std::env::remove_var("CHAT_AUTO_UPDATE_CLI");

        let config = ChatConfig::from_env();
        // Tracks the `default_model` literal in `from_env`; bump both together.
        assert_eq!(config.default_model, "claude-sonnet-5");
        assert_eq!(config.max_sessions, 10);
        assert_eq!(config.session_timeout.as_secs(), 1800);
        // Permission defaults — MCP tools are pre-approved out of the box
        assert_eq!(config.permission.mode, "default");
        assert_eq!(
            config.permission.allowed_tools,
            vec!["mcp__project-orchestrator__*"]
        );
        assert!(config.permission.disallowed_tools.is_empty());
        // New fields — defaults when env vars are absent
        assert!(config.process_path.is_none());
        assert!(config.claude_cli_path.is_none());
        assert!(!config.auto_update_cli);

        // Phase 2: custom values
        std::env::set_var("CHAT_DEFAULT_MODEL", "claude-sonnet-4-6");
        std::env::set_var("CHAT_MAX_SESSIONS", "5");
        std::env::set_var("CHAT_SESSION_TIMEOUT_SECS", "600");
        std::env::set_var("MCP_SERVER_PATH", "/custom/path/mcp_server");
        std::env::set_var("CHAT_PERMISSION_MODE", "default");
        std::env::set_var(
            "CHAT_ALLOWED_TOOLS",
            "Bash(git *),Read,mcp__project-orchestrator__*",
        );
        std::env::set_var("CHAT_DISALLOWED_TOOLS", "Bash(rm -rf *), Bash(sudo *)");
        std::env::set_var(
            "CHAT_PROCESS_PATH",
            "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        );
        std::env::set_var("CLAUDE_CLI_PATH", "/opt/homebrew/bin/claude");
        std::env::set_var("CHAT_AUTO_UPDATE_CLI", "true");

        let config = ChatConfig::from_env();
        assert_eq!(config.default_model, "claude-sonnet-4-6");
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
        // New fields — custom values from env
        assert_eq!(
            config.process_path.as_deref(),
            Some("/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
        );
        assert_eq!(
            config.claude_cli_path.as_deref(),
            Some("/opt/homebrew/bin/claude")
        );
        assert!(config.auto_update_cli);

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
        std::env::remove_var("CHAT_PROCESS_PATH");
        std::env::remove_var("CLAUDE_CLI_PATH");
        std::env::remove_var("CHAT_AUTO_UPDATE_CLI");
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
    fn neutral_mode_names_are_accepted_next_to_the_legacy_strings() {
        // Decision A43: the API accepts both forms.
        for (neutral, legacy, expected) in [
            ("ask", "default", PermissionMode::Default),
            ("auto_edits", "acceptEdits", PermissionMode::AcceptEdits),
            ("plan_only", "plan", PermissionMode::Plan),
            (
                "trust",
                "bypassPermissions",
                PermissionMode::BypassPermissions,
            ),
        ] {
            assert!(PermissionConfig::is_valid_mode(neutral), "{neutral}");
            let as_neutral = PermissionConfig {
                mode: neutral.into(),
                ..Default::default()
            };
            let as_legacy = PermissionConfig {
                mode: legacy.into(),
                ..Default::default()
            };
            assert_eq!(as_neutral.to_nexus_mode(), expected, "{neutral}");
            assert_eq!(as_legacy.to_nexus_mode(), expected, "{legacy}");
        }
        // The other legacy strings Claude Code knows are accepted too, and
        // never widen the policy beyond what they mean.
        assert_eq!(
            PermissionConfig {
                mode: "dontAsk".into(),
                ..Default::default()
            }
            .to_nexus_mode(),
            PermissionMode::Default
        );
        assert_eq!(
            PermissionConfig {
                mode: "auto".into(),
                ..Default::default()
            }
            .to_nexus_mode(),
            PermissionMode::AcceptEdits
        );
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
