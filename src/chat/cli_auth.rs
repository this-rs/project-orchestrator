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
    auth_status_of(
        tokio::process::Command::new(&cli_path),
        std::time::Duration::from_secs(10),
    )
    .await
}

/// Run `<cli> auth status` with the program (and environment) `cli` already
/// carries, and read its answer. Split from [`check_cli_auth_status`] so the
/// tests can point it at `fake_claude` instead of whatever is installed.
async fn auth_status_of(
    mut cli: tokio::process::Command,
    timeout: std::time::Duration,
) -> CliAuthStatus {
    let output = match tokio::time::timeout(
        timeout,
        cli.args(["auth", "status"])
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
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
            warn!(?timeout, "claude auth status timed out");
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

    /// `fake_claude` (the scripted Claude CLI of nexus) playing `transcript`.
    fn fake_cli(
        dir: &tempfile::TempDir,
        transcript: &[serde_json::Value],
    ) -> tokio::process::Command {
        let path = dir.path().join("transcript.jsonl");
        let lines: Vec<String> = transcript.iter().map(|op| op.to_string()).collect();
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();
        let mut cli =
            tokio::process::Command::new(crate::chat::agent_e2e_tests::fake_bin("fake_claude"));
        cli.env("FAKE_CLAUDE_TRANSCRIPT", &path);
        cli
    }

    const PATIENCE: std::time::Duration = std::time::Duration::from_secs(10);

    #[tokio::test]
    async fn a_logged_in_cli_reports_its_method_and_account() {
        let dir = tempfile::tempdir().unwrap();
        let cli = fake_cli(
            &dir,
            &[
                serde_json::json!({"op": "emit_json", "json": {
                    "loggedIn": true, "authMethod": "claude.ai", "email": "dev@example.com"
                }}),
                serde_json::json!({"op": "exit", "code": 0}),
            ],
        );
        let status = auth_status_of(cli, PATIENCE).await;
        assert!(status.logged_in);
        assert_eq!(status.auth_type, "claude.ai");
        assert_eq!(status.account_email.as_deref(), Some("dev@example.com"));
        assert_eq!(status.message, None);
    }

    #[tokio::test]
    async fn a_logged_out_cli_without_method_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let cli = fake_cli(
            &dir,
            &[
                serde_json::json!({"op": "emit_json", "json": {"loggedIn": false}}),
                serde_json::json!({"op": "exit", "code": 1}),
            ],
        );
        let status = auth_status_of(cli, PATIENCE).await;
        assert!(!status.logged_in);
        assert_eq!(status.auth_type, "none");
        assert_eq!(status.account_email, None);
        assert_eq!(status.message, None, "a parsed answer carries no message");
    }

    #[tokio::test]
    async fn an_unreadable_answer_falls_back_on_the_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        let ok = fake_cli(
            &dir,
            &[
                serde_json::json!({"op": "emit", "line": "Logged in."}),
                serde_json::json!({"op": "exit", "code": 0}),
            ],
        );
        let status = auth_status_of(ok, PATIENCE).await;
        assert!(status.logged_in, "exit 0 reads as logged in");
        assert_eq!(status.auth_type, "unknown");
        assert_eq!(
            status.message.as_deref(),
            Some("Could not parse auth status (exit code: Some(0))")
        );

        let dir = tempfile::tempdir().unwrap();
        let failed = fake_cli(
            &dir,
            &[
                serde_json::json!({"op": "emit", "line": "Not logged in."}),
                serde_json::json!({"op": "exit", "code": 3}),
            ],
        );
        let status = auth_status_of(failed, PATIENCE).await;
        assert!(!status.logged_in);
        assert_eq!(status.auth_type, "none");
        assert_eq!(
            status.message.as_deref(),
            Some("Could not parse auth status (exit code: Some(3))")
        );
    }

    #[tokio::test]
    async fn a_cli_that_does_not_answer_in_time_is_reported_as_a_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let cli = fake_cli(
            &dir,
            &[
                serde_json::json!({"op": "sleep", "ms": 5000}),
                serde_json::json!({"op": "exit", "code": 0}),
            ],
        );
        let status = auth_status_of(cli, std::time::Duration::from_millis(200)).await;
        assert!(!status.logged_in);
        assert_eq!(status.auth_type, "none");
        assert_eq!(status.message.as_deref(), Some("CLI auth check timed out"));
    }

    #[tokio::test]
    async fn a_cli_that_cannot_be_started_is_reported_as_such() {
        let cli = tokio::process::Command::new("/nonexistent/claude");
        let status = auth_status_of(cli, PATIENCE).await;
        assert!(!status.logged_in);
        assert_eq!(status.auth_type, "none");
        assert!(
            status
                .message
                .as_deref()
                .is_some_and(|m| m.starts_with("Failed to run CLI: ")),
            "{status:?}"
        );
    }
}
