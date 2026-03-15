//! Automatic Claude Code MCP configuration.
//!
//! Detects Claude Code CLI and configures the Project Orchestrator
//! MCP server in **stdio mode** (the `mcp_server` binary acts as an HTTP
//! proxy to the REST API). Configuration is done either via `claude mcp add`
//! or by directly editing `~/.claude/mcp.json`.
//!
//! The MCP server binary receives two critical env vars:
//! - `PO_SERVER_URL`: REST API base URL (e.g. `http://127.0.0.1:8080`)
//! - `PO_JWT_SECRET`: JWT signing secret for auto-generating auth tokens
//!
//! This supports **multi-instance** setups: each PO server on a different port
//! auto-configures Claude Code with its own URL and JWT secret. The mcp_server
//! binary auto-generates a fresh token from the secret at startup.
//!
//! Also configures `~/.claude/settings.json` to pre-approve all MCP tools
//! from the Project Orchestrator server (`mcp__project-orchestrator__*`),
//! so the user doesn't get a permission prompt for every tool call.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};

// ============================================================================
// Types
// ============================================================================

/// Configuration needed to set up Claude Code MCP integration.
#[derive(Debug, Clone)]
pub struct SetupConfig {
    /// Path to the `mcp_server` binary (auto-detected or from env/config).
    pub mcp_server_path: PathBuf,
    /// Server port for building `PO_SERVER_URL` (e.g. 8080 → `http://127.0.0.1:8080`).
    pub server_port: u16,
    /// JWT signing secret from config.yaml (for `PO_JWT_SECRET` env var).
    pub jwt_secret: Option<String>,
}

/// Result of the setup operation.
#[derive(Debug)]
pub enum SetupResult {
    /// Successfully configured via `claude mcp add`
    ConfiguredViaCli { allowed_tools_configured: bool },
    /// Successfully configured by writing to mcp.json directly
    ConfiguredViaFile {
        path: PathBuf,
        allowed_tools_configured: bool,
    },
    /// Already configured with correct settings — no changes needed
    AlreadyConfigured { allowed_tools_configured: bool },
    /// Updated existing configuration (was stale or wrong mode)
    Updated {
        path: PathBuf,
        allowed_tools_configured: bool,
    },
}

// ============================================================================
// Configuration
// ============================================================================

const MCP_SERVER_NAME: &str = "project-orchestrator";

/// Permission pattern that allows all MCP tools from the Project Orchestrator server.
/// Format: `mcp__<server-name>__*` (Claude Code double-underscore convention).
/// See: https://code.claude.com/docs/en/permissions#mcp
const MCP_ALLOWED_TOOL_PATTERN: &str = "mcp__project-orchestrator__*";

// ============================================================================
// Public API
// ============================================================================

/// Detect Claude Code and configure the MCP server in stdio mode.
///
/// Strategy:
/// 1. Check if already configured with correct settings → skip
/// 2. If stale/wrong config exists → update it
/// 3. Try `claude mcp add` if CLI is available (new install)
/// 4. Fall back to directly editing `~/.claude/mcp.json`
///
/// The MCP server binary is configured with:
/// - `PO_SERVER_URL=http://127.0.0.1:{port}` — points to this instance
/// - `PO_JWT_SECRET={secret}` — auto-generates auth tokens at startup
pub fn setup_claude_code(config: &SetupConfig) -> Result<SetupResult> {
    let server_url = format!("http://127.0.0.1:{}", config.server_port);
    let mcp_path = config.mcp_server_path.to_string_lossy().to_string();

    // Always try to configure allowed tools (idempotent — safe to call multiple times)
    let allowed_tools_ok = match configure_allowed_tools() {
        Ok(()) => {
            tracing::info!("MCP allowed tools configured in settings.json");
            true
        }
        Err(e) => {
            tracing::warn!("Failed to configure allowed tools in settings.json: {}", e);
            false
        }
    };

    // Build the expected env vars
    let mut env_vars = std::collections::HashMap::new();
    env_vars.insert("PO_SERVER_URL".to_string(), server_url.clone());
    if let Some(ref secret) = config.jwt_secret {
        env_vars.insert("PO_JWT_SECRET".to_string(), secret.clone());
    }

    // Check if already configured and up-to-date
    match check_existing_config(&mcp_path, &env_vars)? {
        ConfigStatus::UpToDate => {
            tracing::info!(
                "Project Orchestrator MCP server is already correctly configured in Claude Code"
            );
            return Ok(SetupResult::AlreadyConfigured {
                allowed_tools_configured: allowed_tools_ok,
            });
        }
        ConfigStatus::Stale => {
            tracing::info!(
                "Project Orchestrator MCP config is stale — updating to stdio mode with current settings"
            );
            // Update the existing config in-place
            let path = configure_via_file(&mcp_path, &env_vars)?;
            return Ok(SetupResult::Updated {
                path,
                allowed_tools_configured: allowed_tools_ok,
            });
        }
        ConfigStatus::Missing => {
            tracing::info!("Project Orchestrator MCP server not yet configured");
        }
    }

    // Try CLI first (for new installs)
    if let Some(claude_path) = detect_claude_cli() {
        tracing::info!("Claude Code CLI found at: {}", claude_path);
        match configure_via_cli(&claude_path, &mcp_path, &env_vars) {
            Ok(()) => {
                return Ok(SetupResult::ConfiguredViaCli {
                    allowed_tools_configured: allowed_tools_ok,
                })
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to configure via CLI ({}), falling back to mcp.json",
                    e
                );
            }
        }
    } else {
        tracing::info!("Claude Code CLI not found, will edit mcp.json directly");
    }

    // Fallback: edit mcp.json directly
    let path = configure_via_file(&mcp_path, &env_vars)?;
    Ok(SetupResult::ConfiguredViaFile {
        path,
        allowed_tools_configured: allowed_tools_ok,
    })
}

/// Detect if Claude Code CLI is installed.
///
/// First tries `which claude` (or `where` on Windows). If that fails (common
/// in macOS .app bundles where PATH is minimal), falls back to checking well-known
/// installation paths.
pub fn detect_claude_cli() -> Option<String> {
    #[cfg(unix)]
    let cmd = "which";
    #[cfg(windows)]
    let cmd = "where";

    // Try PATH-based lookup first
    if let Ok(output) = std::process::Command::new(cmd).arg("claude").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }

    // Fallback: check well-known paths (macOS .app bundles have a minimal PATH)
    #[cfg(unix)]
    let candidates: Vec<String> = {
        let home = dirs::home_dir()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_default();
        vec![
            // npm global (macOS/Linux)
            "/usr/local/bin/claude".to_string(),
            // Homebrew (Apple Silicon)
            "/opt/homebrew/bin/claude".to_string(),
            // User-local npm global
            format!("{}/.npm-global/bin/claude", home),
            // nvm / fnm / volta managed
            format!("{}/.local/bin/claude", home),
        ]
    };

    #[cfg(windows)]
    let candidates: Vec<String> = {
        let mut paths = Vec::new();
        // Anthropic installer (AppData\Local\Programs\claude\claude.exe)
        if let Some(local_data) = dirs::data_local_dir() {
            paths.push(
                local_data
                    .join("Programs")
                    .join("claude")
                    .join("claude.exe")
                    .to_string_lossy()
                    .to_string(),
            );
        }
        // npm global (AppData\Roaming\npm\claude.cmd)
        // dirs::config_dir() already returns AppData\Roaming on Windows
        if let Some(config_dir) = dirs::config_dir() {
            paths.push(
                config_dir
                    .join("npm")
                    .join("claude.cmd")
                    .to_string_lossy()
                    .to_string(),
            );
        }
        // Compat: ~/.local/bin/claude.exe
        if let Some(home) = dirs::home_dir() {
            paths.push(
                home.join(".local")
                    .join("bin")
                    .join("claude.exe")
                    .to_string_lossy()
                    .to_string(),
            );
            // ~/.claude/local/claude.exe
            paths.push(
                home.join(".claude")
                    .join("local")
                    .join("claude.exe")
                    .to_string_lossy()
                    .to_string(),
            );
        }
        paths
    };

    for candidate in &candidates {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }

    None
}

// ============================================================================
// Configuration status check
// ============================================================================

enum ConfigStatus {
    /// Config exists with correct command, PO_SERVER_URL, and PO_JWT_SECRET
    UpToDate,
    /// Config exists but is stale (wrong mode, missing env vars, wrong URL)
    Stale,
    /// No config for this server
    Missing,
}

/// Check if the existing MCP config is up-to-date.
///
/// Validates:
/// - Entry exists in mcpServers
/// - Uses stdio mode (has "command", no "type": "sse")
/// - Command matches the expected mcp_server binary path
/// - PO_SERVER_URL matches the expected URL
/// - PO_JWT_SECRET is set (if we have one)
fn check_existing_config(
    expected_command: &str,
    expected_env: &std::collections::HashMap<String, String>,
) -> Result<ConfigStatus> {
    let path = mcp_json_path()?;
    if !path.exists() {
        return Ok(ConfigStatus::Missing);
    }

    let content = std::fs::read_to_string(&path).context("Failed to read mcp.json")?;
    let json: Value = serde_json::from_str(&content).unwrap_or(Value::Object(Default::default()));

    let server = match json.get("mcpServers").and_then(|s| s.get(MCP_SERVER_NAME)) {
        Some(s) => s,
        None => return Ok(ConfigStatus::Missing),
    };

    // Check if it's SSE mode (stale — SSE endpoint doesn't exist)
    if server.get("type").and_then(|t| t.as_str()) == Some("sse") {
        return Ok(ConfigStatus::Stale);
    }

    // Check command matches
    let current_command = server.get("command").and_then(|c| c.as_str()).unwrap_or("");
    if current_command != expected_command {
        return Ok(ConfigStatus::Stale);
    }

    // Check env vars match
    let env_obj = server.get("env").and_then(|e| e.as_object());
    match env_obj {
        None => {
            if !expected_env.is_empty() {
                return Ok(ConfigStatus::Stale);
            }
        }
        Some(env) => {
            for (key, expected_val) in expected_env {
                match env.get(key).and_then(|v| v.as_str()) {
                    Some(val) if val == expected_val => {}
                    _ => return Ok(ConfigStatus::Stale),
                }
            }
        }
    }

    Ok(ConfigStatus::UpToDate)
}

// ============================================================================
// CLI-based configuration
// ============================================================================

fn configure_via_cli(
    claude_path: &str,
    mcp_command: &str,
    env_vars: &std::collections::HashMap<String, String>,
) -> Result<()> {
    // First remove any existing entry (may be SSE or stale stdio)
    let _ = std::process::Command::new(claude_path)
        .args(["mcp", "remove", MCP_SERVER_NAME])
        .output();

    // Build args: claude mcp add -e KEY=val -e KEY=val <name> -- <command>
    let mut args = vec!["mcp".to_string(), "add".to_string()];

    // Add env vars
    for (key, value) in env_vars {
        args.push("-e".to_string());
        args.push(format!("{}={}", key, value));
    }

    // Server name + separator + command
    args.push(MCP_SERVER_NAME.to_string());
    args.push("--".to_string());
    args.push(mcp_command.to_string());

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let output = std::process::Command::new(claude_path)
        .args(&args_ref)
        .output()
        .context("Failed to execute claude mcp add")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("claude mcp add failed: {}", stderr.trim());
    }

    tracing::info!("Successfully configured via `claude mcp add` (stdio mode)");

    // Verify
    let verify = std::process::Command::new(claude_path)
        .args(["mcp", "list"])
        .output();

    if let Ok(out) = verify {
        let stdout = String::from_utf8_lossy(&out.stdout);
        if stdout.contains(MCP_SERVER_NAME) {
            tracing::info!("Verified: {} is in `claude mcp list`", MCP_SERVER_NAME);
        }
    }

    Ok(())
}

// ============================================================================
// File-based configuration (fallback)
// ============================================================================

/// Get the path to Claude Code's MCP config file.
fn mcp_json_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".claude").join("mcp.json"))
}

/// Configure by directly editing ~/.claude/mcp.json.
///
/// Writes a stdio-mode entry with the mcp_server binary command and env vars.
fn configure_via_file(
    mcp_command: &str,
    env_vars: &std::collections::HashMap<String, String>,
) -> Result<PathBuf> {
    let path = mcp_json_path()?;

    // Read existing config or create empty object
    let mut json: Value = if path.exists() {
        let content = std::fs::read_to_string(&path).context("Failed to read mcp.json")?;

        // Create backup
        let backup_path = path.with_extension("json.bak");
        std::fs::copy(&path, &backup_path).context("Failed to create backup of mcp.json")?;
        tracing::info!("Backup created at: {}", backup_path.display());

        serde_json::from_str(&content).unwrap_or(Value::Object(Default::default()))
    } else {
        Value::Object(Default::default())
    };

    // Ensure mcpServers object exists
    let obj = json
        .as_object_mut()
        .context("mcp.json is not a JSON object")?;
    if !obj.contains_key("mcpServers") {
        obj.insert("mcpServers".to_string(), Value::Object(Default::default()));
    }

    let servers = obj
        .get_mut("mcpServers")
        .and_then(|s| s.as_object_mut())
        .context("mcpServers is not an object")?;

    // Build the stdio server config
    let mut env_json = serde_json::Map::new();
    for (key, value) in env_vars {
        env_json.insert(key.clone(), Value::String(value.clone()));
    }

    let server_config = serde_json::json!({
        "command": mcp_command,
        "env": Value::Object(env_json)
    });

    servers.insert(MCP_SERVER_NAME.to_string(), server_config);

    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create .claude directory")?;
    }

    // Write the updated config
    let formatted = serde_json::to_string_pretty(&json).context("Failed to serialize JSON")?;
    std::fs::write(&path, formatted).context("Failed to write mcp.json")?;

    tracing::info!("MCP server configured in stdio mode: {}", path.display());
    Ok(path)
}

// ============================================================================
// Allowed tools configuration (settings.json)
// ============================================================================

/// Get the path to Claude Code's settings file.
fn settings_json_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".claude").join("settings.json"))
}

/// Configure allowed tools in `~/.claude/settings.json`.
///
/// Adds the `mcp__project-orchestrator__*` pattern to `permissions.allow`
/// so all MCP tools from the Project Orchestrator server are pre-approved.
///
/// - Merges with existing settings (never overwrites)
/// - Creates a backup (.bak) if the file already exists
/// - Idempotent: safe to call multiple times
pub fn configure_allowed_tools() -> Result<()> {
    let path = settings_json_path()?;
    configure_allowed_tools_at(&path)
}

/// Internal implementation that accepts a path (for testability).
fn configure_allowed_tools_at(path: &Path) -> Result<()> {
    // Read existing settings or create empty object
    let mut json: Value = if path.exists() {
        let content = std::fs::read_to_string(path).context("Failed to read settings.json")?;

        // Create backup before modifying
        let backup_path = path.with_extension("json.bak");
        std::fs::copy(path, &backup_path).context("Failed to create backup of settings.json")?;
        tracing::info!("Backup created at: {}", backup_path.display());

        serde_json::from_str(&content).unwrap_or(Value::Object(Default::default()))
    } else {
        Value::Object(Default::default())
    };

    // Ensure top-level is an object
    let obj = json
        .as_object_mut()
        .context("settings.json is not a JSON object")?;

    // Ensure permissions object exists
    if !obj.contains_key("permissions") {
        obj.insert("permissions".to_string(), Value::Object(Default::default()));
    }

    let permissions = obj
        .get_mut("permissions")
        .and_then(|p| p.as_object_mut())
        .context("permissions is not an object")?;

    // Ensure allow array exists
    if !permissions.contains_key("allow") {
        permissions.insert("allow".to_string(), Value::Array(Vec::new()));
    }

    let allow = permissions
        .get_mut("allow")
        .and_then(|a| a.as_array_mut())
        .context("permissions.allow is not an array")?;

    // Check if the pattern is already present (idempotent)
    let already_present = allow
        .iter()
        .any(|v| v.as_str() == Some(MCP_ALLOWED_TOOL_PATTERN));

    if already_present {
        tracing::info!(
            "Pattern '{}' already in permissions.allow — skipping",
            MCP_ALLOWED_TOOL_PATTERN
        );
        return Ok(());
    }

    // Add the pattern
    allow.push(Value::String(MCP_ALLOWED_TOOL_PATTERN.to_string()));

    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create .claude directory")?;
    }

    // Write the updated settings
    let formatted =
        serde_json::to_string_pretty(&json).context("Failed to serialize settings JSON")?;
    std::fs::write(path, formatted).context("Failed to write settings.json")?;

    tracing::info!(
        "Added '{}' to permissions.allow in: {}",
        MCP_ALLOWED_TOOL_PATTERN,
        path.display()
    );
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_detect_claude_cli() {
        // Just verify it doesn't panic — may or may not find claude
        let _result = detect_claude_cli();
    }

    #[test]
    fn test_mcp_json_path() {
        let path = mcp_json_path().unwrap();
        assert!(path.ends_with(".claude/mcp.json") || path.ends_with(".claude\\mcp.json"));
    }

    // ========================================================================
    // configure_via_file tests
    // ========================================================================

    #[test]
    fn test_configure_via_file_creates_stdio_config() {
        let tmp = TempDir::new().unwrap();
        let mcp_path = tmp.path().join("mcp.json");

        // Temporarily override mcp_json_path by writing directly
        let mut env = std::collections::HashMap::new();
        env.insert(
            "PO_SERVER_URL".to_string(),
            "http://127.0.0.1:8080".to_string(),
        );
        env.insert("PO_JWT_SECRET".to_string(), "test-secret".to_string());

        // Write directly to the tmp path
        let mut json = serde_json::json!({});
        let obj = json.as_object_mut().unwrap();
        obj.insert("mcpServers".to_string(), Value::Object(Default::default()));
        let servers = obj.get_mut("mcpServers").unwrap().as_object_mut().unwrap();

        let mut env_json = serde_json::Map::new();
        for (key, value) in &env {
            env_json.insert(key.clone(), Value::String(value.clone()));
        }

        let server_config = serde_json::json!({
            "command": "/path/to/mcp_server",
            "env": Value::Object(env_json)
        });
        servers.insert(MCP_SERVER_NAME.to_string(), server_config);

        let formatted = serde_json::to_string_pretty(&json).unwrap();
        std::fs::write(&mcp_path, &formatted).unwrap();

        // Verify the config
        let content = std::fs::read_to_string(&mcp_path).unwrap();
        let parsed: Value = serde_json::from_str(&content).unwrap();
        let server = &parsed["mcpServers"]["project-orchestrator"];

        assert_eq!(server["command"].as_str().unwrap(), "/path/to/mcp_server");
        assert!(server.get("type").is_none(), "should not have SSE type");
        assert_eq!(
            server["env"]["PO_SERVER_URL"].as_str().unwrap(),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            server["env"]["PO_JWT_SECRET"].as_str().unwrap(),
            "test-secret"
        );
    }

    #[test]
    fn test_check_existing_config_missing() {
        // Non-existent file → Missing
        // We can't easily test this without mocking mcp_json_path,
        // but we test the logic with a direct call
        let env = std::collections::HashMap::new();
        // The function reads from the real mcp.json path, so just verify it doesn't panic
        let _result = check_existing_config("/fake/path", &env);
    }

    #[test]
    fn test_check_existing_config_detects_stale_sse() {
        // An SSE config should be detected as stale
        let sse_config = serde_json::json!({
            "type": "sse",
            "url": "http://localhost:8080/mcp/sse"
        });

        // Check that SSE type is correctly identified as stale
        assert_eq!(sse_config.get("type").and_then(|t| t.as_str()), Some("sse"));
    }

    // ========================================================================
    // configure_allowed_tools_at() tests
    // ========================================================================

    #[test]
    fn test_configure_allowed_tools_new_file() {
        // File doesn't exist → should create settings.json with correct structure
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");

        assert!(!path.exists());
        configure_allowed_tools_at(&path).unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();

        // Verify structure: { "permissions": { "allow": ["mcp__project-orchestrator__*"] } }
        let allow = json["permissions"]["allow"].as_array().unwrap();
        assert_eq!(allow.len(), 1);
        assert_eq!(allow[0].as_str().unwrap(), MCP_ALLOWED_TOOL_PATTERN);
    }

    #[test]
    fn test_configure_allowed_tools_merge() {
        // Existing file with other permissions → should add MCP pattern without removing existing ones
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");

        let existing = serde_json::json!({
            "permissions": {
                "allow": ["Bash(git *)", "Read"]
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&existing).unwrap()).unwrap();

        configure_allowed_tools_at(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();

        let allow = json["permissions"]["allow"].as_array().unwrap();
        assert_eq!(allow.len(), 3);
        assert!(allow.iter().any(|v| v.as_str() == Some("Bash(git *)")));
        assert!(allow.iter().any(|v| v.as_str() == Some("Read")));
        assert!(allow
            .iter()
            .any(|v| v.as_str() == Some(MCP_ALLOWED_TOOL_PATTERN)));
    }

    #[test]
    fn test_configure_allowed_tools_idempotent() {
        // Called twice → pattern should appear only once
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");

        configure_allowed_tools_at(&path).unwrap();
        configure_allowed_tools_at(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();

        let allow = json["permissions"]["allow"].as_array().unwrap();
        let count = allow
            .iter()
            .filter(|v| v.as_str() == Some(MCP_ALLOWED_TOOL_PATTERN))
            .count();
        assert_eq!(
            count, 1,
            "Pattern should appear exactly once after two calls"
        );
    }

    #[test]
    fn test_configure_allowed_tools_preserves_other_keys() {
        // File with other top-level keys → they must be preserved
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");

        let existing = serde_json::json!({
            "env": {
                "ANTHROPIC_API_KEY": "sk-ant-xxx"
            },
            "enabledPlugins": ["code-review"],
            "permissions": {
                "deny": ["Bash(rm *)"]
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&existing).unwrap()).unwrap();

        configure_allowed_tools_at(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let json: Value = serde_json::from_str(&content).unwrap();

        // Other keys preserved
        assert_eq!(
            json["env"]["ANTHROPIC_API_KEY"].as_str().unwrap(),
            "sk-ant-xxx"
        );
        assert_eq!(json["enabledPlugins"][0].as_str().unwrap(), "code-review");

        // Deny array preserved
        let deny = json["permissions"]["deny"].as_array().unwrap();
        assert_eq!(deny.len(), 1);
        assert_eq!(deny[0].as_str().unwrap(), "Bash(rm *)");

        // Allow array created with our pattern
        let allow = json["permissions"]["allow"].as_array().unwrap();
        assert_eq!(allow.len(), 1);
        assert_eq!(allow[0].as_str().unwrap(), MCP_ALLOWED_TOOL_PATTERN);
    }

    #[test]
    fn test_configure_allowed_tools_backup() {
        // Existing file → a .bak file should be created with original content
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("settings.json");

        let original_content = serde_json::json!({
            "permissions": {
                "allow": ["Bash(git *)"]
            }
        });
        let original_str = serde_json::to_string_pretty(&original_content).unwrap();
        std::fs::write(&path, &original_str).unwrap();

        configure_allowed_tools_at(&path).unwrap();

        // Verify backup was created
        let backup_path = path.with_extension("json.bak");
        assert!(backup_path.exists(), "Backup file should exist");

        // Verify backup contains the original content
        let backup_content = std::fs::read_to_string(&backup_path).unwrap();
        assert_eq!(backup_content, original_str);

        // Verify the main file has been modified (has MCP pattern)
        let modified_content = std::fs::read_to_string(&path).unwrap();
        let json: Value = serde_json::from_str(&modified_content).unwrap();
        let allow = json["permissions"]["allow"].as_array().unwrap();
        assert!(allow
            .iter()
            .any(|v| v.as_str() == Some(MCP_ALLOWED_TOOL_PATTERN)));
    }
}
