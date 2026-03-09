//! CLI auth status — check if the user is authenticated via Claude CLI.
//!
//! Spawns `claude auth status` (which outputs JSON) and parses the result
//! to determine authentication state.

use nexus_claude::find_claude_cli;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Authentication status as reported by `claude auth status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliAuthStatus {
    /// Whether the user is currently logged in
    pub logged_in: bool,
    /// Authentication method: "oauth", "api_key", "claude.ai", "none"
    pub auth_type: String,
    /// Account email (if available)
    pub account_email: Option<String>,
    /// Human-readable status message (set on errors / CLI not installed)
    pub message: Option<String>,
}

/// Raw JSON output from `claude auth status`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawAuthStatus {
    logged_in: bool,
    auth_method: Option<String>,
    email: Option<String>,
}

/// Check the CLI authentication status by spawning `claude auth status`.
///
/// - Uses `find_claude_cli()` from the Nexus SDK to locate the binary.
/// - Timeout: 10 seconds.
/// - If the CLI is not installed, returns `logged_in: false` with an explanatory message.
/// - **Security**: Never logs tokens or credentials.
pub async fn check_cli_auth_status() -> CliAuthStatus {
    let cli_path = match find_claude_cli() {
        Ok(path) => path,
        Err(_) => {
            debug!("Claude CLI not found — returning not-authenticated status");
            return CliAuthStatus {
                logged_in: false,
                auth_type: "none".to_string(),
                account_email: None,
                message: Some("Claude CLI is not installed".to_string()),
            };
        }
    };

    debug!(cli = %cli_path.display(), "Checking CLI auth status");

    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new(&cli_path)
            .args(["auth", "status"])
            .stderr(std::process::Stdio::null())
            .output(),
    )
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            warn!(error = %e, "Failed to spawn claude auth status");
            return CliAuthStatus {
                logged_in: false,
                auth_type: "none".to_string(),
                account_email: None,
                message: Some(format!("Failed to run CLI: {}", e)),
            };
        }
        Err(_) => {
            warn!("claude auth status timed out (10s)");
            return CliAuthStatus {
                logged_in: false,
                auth_type: "none".to_string(),
                account_email: None,
                message: Some("CLI auth check timed out".to_string()),
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    debug!(exit_code = ?output.status.code(), "claude auth status finished");

    // Parse JSON output
    match serde_json::from_str::<RawAuthStatus>(&stdout) {
        Ok(raw) => CliAuthStatus {
            logged_in: raw.logged_in,
            auth_type: raw.auth_method.unwrap_or_else(|| "none".to_string()),
            account_email: raw.email,
            message: None,
        },
        Err(e) => {
            warn!(error = %e, "Failed to parse claude auth status JSON");
            // Try to infer from exit code
            let success = output.status.success();
            CliAuthStatus {
                logged_in: success,
                auth_type: if success {
                    "unknown".to_string()
                } else {
                    "none".to_string()
                },
                account_email: None,
                message: Some(format!(
                    "Could not parse auth status (exit code: {:?})",
                    output.status.code()
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_auth_status_serialization() {
        let status = CliAuthStatus {
            logged_in: true,
            auth_type: "claude.ai".to_string(),
            account_email: Some("user@example.com".to_string()),
            message: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"logged_in\":true"));
        assert!(json.contains("\"auth_type\":\"claude.ai\""));
        assert!(json.contains("\"account_email\":\"user@example.com\""));
    }

    #[test]
    fn test_cli_auth_status_not_logged_in() {
        let status = CliAuthStatus {
            logged_in: false,
            auth_type: "none".to_string(),
            account_email: None,
            message: Some("CLI not installed".to_string()),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"logged_in\":false"));
        assert!(json.contains("\"auth_type\":\"none\""));
        assert!(json.contains("\"message\":\"CLI not installed\""));
    }

    #[test]
    fn test_parse_raw_auth_status_json() {
        let raw_json = r#"{
            "loggedIn": true,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "email": "user@example.com",
            "orgId": "abc-123",
            "orgName": null,
            "subscriptionType": "max"
        }"#;
        let raw: RawAuthStatus = serde_json::from_str(raw_json).unwrap();
        assert!(raw.logged_in);
        assert_eq!(raw.auth_method, Some("claude.ai".to_string()));
        assert_eq!(raw.email, Some("user@example.com".to_string()));
    }

    #[test]
    fn test_parse_raw_auth_status_not_logged_in() {
        let raw_json = r#"{"loggedIn": false}"#;
        let raw: RawAuthStatus = serde_json::from_str(raw_json).unwrap();
        assert!(!raw.logged_in);
        assert_eq!(raw.auth_method, None);
        assert_eq!(raw.email, None);
    }

    /// Integration test — requires a real Claude CLI.
    #[tokio::test]
    #[ignore]
    async fn test_check_cli_auth_status_integration() {
        let status = check_cli_auth_status().await;
        // On a machine with Claude CLI installed and logged in
        if status.logged_in {
            assert_ne!(status.auth_type, "none");
        }
    }
}
