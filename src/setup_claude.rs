//! Automatic Claude Code MCP configuration.
//!
//! Detects Claude Code CLI and configures the Project Orchestrator
//! MCP server, either via `claude mcp add` or by directly editing
//! `~/.claude/mcp.json`.
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
    /// Already configured — no changes needed
    AlreadyConfigured { allowed_tools_configured: bool },
}

// ============================================================================
// Configuration
// ============================================================================

const MCP_SERVER_NAME: &str = "project-orchestrator";

/// Permission pattern that allows all MCP tools from the Project Orchestrator server.
/// Format: `mcp__<server-name>__*` (Claude Code double-underscore convention).
/// See: https://code.claude.com/docs/en/permissions#mcp
const MCP_ALLOWED_TOOL_PATTERN: &str = "mcp__project-orchestrator__*";

/// Build the default SSE URL using the given port.
///
/// The MCP server in stdio mode doesn't use this, but the HTTP server's
/// SSE transport is on `/mcp/sse` at the configured server port.
fn default_sse_url(port: u16) -> String {
    format!("http://localhost:{}/mcp/sse", port)
}

// ============================================================================
// Public API
// ============================================================================

/// Detect Claude Code and configure the MCP server.
///
/// Strategy:
/// 1. Try `claude mcp add` if CLI is available
/// 2. Fall back to directly editing `~/.claude/mcp.json`
///
/// If `server_url` is not provided, the default URL is built from the given
/// `port` (defaults to 8080 if not specified).
pub fn setup_claude_code(server_url: Option<&str>, port: Option<u16>) -> Result<SetupResult> {
    let default_url = default_sse_url(port.unwrap_or(8080));
    let url = server_url.unwrap_or(&default_url);

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

    // Check if already configured
    if is_already_configured()? {
        tracing::info!("Project Orchestrator MCP server is already configured in Claude Code");
        return Ok(SetupResult::AlreadyConfigured {
            allowed_tools_configured: allowed_tools_ok,
        });
    }

    // Try CLI first
    if let Some(claude_path) = detect_claude_cli() {
        tracing::info!("Claude Code CLI found at: {}", claude_path);
        match configure_via_cli(&claude_path, url) {
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
    let path = configure_via_file(url)?;
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
    let candidates = [
        // npm global (macOS/Linux)
        "/usr/local/bin/claude",
        // Homebrew (Apple Silicon)
        "/opt/homebrew/bin/claude",
        // Homebrew (Intel)
        "/usr/local/bin/claude",
        // User-local npm global
        &format!(
            "{}/.npm-global/bin/claude",
            std::env::var("HOME").unwrap_or_default()
        ),
        // nvm / fnm / volta managed
        &format!(
            "{}/.local/bin/claude",
            std::env::var("HOME").unwrap_or_default()
        ),
    ];

    for candidate in &candidates {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }

    None
}

// ============================================================================
// CLI-based configuration
// ============================================================================

fn configure_via_cli(claude_path: &str, url: &str) -> Result<()> {
    let output = std::process::Command::new(claude_path)
        .args([
            "mcp",
            "add",
            MCP_SERVER_NAME,
            "--transport",
            "sse",
            "--url",
            url,
        ])
        .output()
        .context("Failed to execute claude mcp add")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("claude mcp add failed: {}", stderr.trim());
    }

    tracing::info!("Successfully configured via `claude mcp add`");

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

/// Check if the MCP server is already configured.
fn is_already_configured() -> Result<bool> {
    let path = mcp_json_path()?;
    if !path.exists() {
        return Ok(false);
    }

    let content = std::fs::read_to_string(&path).context("Failed to read mcp.json")?;
    let json: Value = serde_json::from_str(&content).unwrap_or(Value::Object(Default::default()));

    // Check in mcpServers
    Ok(json
        .get("mcpServers")
        .and_then(|s| s.get(MCP_SERVER_NAME))
        .is_some())
}

/// Configure by directly editing ~/.claude/mcp.json.
fn configure_via_file(url: &str) -> Result<PathBuf> {
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

    // Add our server entry
    let server_config = serde_json::json!({
        "type": "sse",
        "url": url
    });

    servers.insert(MCP_SERVER_NAME.to_string(), server_config);

    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create .claude directory")?;
    }

    // Write the updated config
    let formatted = serde_json::to_string_pretty(&json).context("Failed to serialize JSON")?;
    std::fs::write(&path, formatted).context("Failed to write mcp.json")?;

    tracing::info!("MCP server configured in: {}", path.display());
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
