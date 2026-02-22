//! CLI version management — check installed version, detect updates, install/upgrade.
//!
//! Uses the Nexus SDK's `find_claude_cli`, `get_cli_version`, `check_latest_npm_version`,
//! and `download_cli` functions to provide a unified CLI management service.

use nexus_claude::{
    check_latest_npm_version, download_cli, find_claude_cli, get_cached_cli_path, get_cli_version,
};
use serde::Serialize;
use std::path::Path;
use tracing::{debug, info, warn};

/// Full CLI version status — returned by `check_cli_status()`.
#[derive(Debug, Clone, Serialize)]
pub struct CliVersionStatus {
    /// Whether a Claude CLI binary was found at all
    pub installed: bool,
    /// Installed version string (e.g., "2.5.1") or None if not installed / unparseable
    pub installed_version: Option<String>,
    /// Latest version available on npm (e.g., "2.6.0") or None if npm unreachable
    pub latest_version: Option<String>,
    /// Whether an update is available (installed < latest, or local build)
    pub update_available: bool,
    /// Whether the CLI is a local/custom build (not from npm or cc-sdk cache)
    pub is_local_build: bool,
    /// Path to the CLI binary
    pub cli_path: Option<String>,
}

/// Result of an install/upgrade operation.
#[derive(Debug, Clone, Serialize)]
pub struct CliInstallResult {
    /// Whether the installation succeeded
    pub success: bool,
    /// Installed version string after the operation
    pub version: Option<String>,
    /// Human-readable result message
    pub message: String,
    /// Path to the installed CLI binary
    pub cli_path: Option<String>,
}

/// Check the current CLI installation status.
///
/// This function:
/// 1. Finds the CLI binary via `find_claude_cli()` (PATH + cache search)
/// 2. Gets the installed version via `get_cli_version()`
/// 3. Checks npm for the latest version (10s timeout, returns None on failure)
/// 4. Detects whether the binary is a "local build" (not from npm/cc-sdk cache)
/// 5. Compares versions to determine if an update is available
pub async fn check_cli_status() -> CliVersionStatus {
    // 1. Find the CLI binary
    let cli_path = find_claude_cli().ok();
    let cli_path_str = cli_path.as_ref().map(|p| p.to_string_lossy().to_string());

    if cli_path.is_none() {
        debug!("Claude CLI not found — returning not-installed status");
        return CliVersionStatus {
            installed: false,
            installed_version: None,
            latest_version: None,
            update_available: false,
            is_local_build: false,
            cli_path: None,
        };
    }
    let cli_path_ref = cli_path.as_ref().unwrap();

    // 2. Get installed version
    let installed_version = get_cli_version(cli_path_ref).await;
    let installed_str = installed_version.as_ref().map(|v| v.to_string());
    debug!(
        version = ?installed_str,
        path = %cli_path_ref.display(),
        "Detected CLI version"
    );

    // 3. Check npm for latest version (best-effort, returns None on network error)
    let latest_version = check_latest_npm_version().await;
    let latest_str = latest_version.as_ref().map(|v| v.to_string());

    // 4. Detect local build
    let is_local = is_local_build(cli_path_ref);

    // 5. Determine if update is available
    let update_available = match (&installed_version, &latest_version) {
        // Local builds are always considered outdated (we can't compare versions)
        _ if is_local => true,
        // Both versions available — compare
        (Some(installed), Some(latest)) => installed < latest,
        // Can't determine — no version info available
        _ => false,
    };

    CliVersionStatus {
        installed: true,
        installed_version: installed_str,
        latest_version: latest_str,
        update_available,
        is_local_build: is_local,
        cli_path: cli_path_str,
    }
}

/// Install or upgrade the Claude Code CLI.
///
/// Uses the Nexus SDK's `download_cli()` which:
/// - Downloads from the official Anthropic install script (Unix) or npm (Windows)
/// - Caches the binary in `~/.cache/cc-sdk/cli/` (or platform equivalent)
/// - Returns the path to the installed binary
///
/// `target_version`: specific version to install (e.g., "2.6.0") or None for latest.
pub async fn install_or_upgrade_cli(target_version: Option<&str>) -> CliInstallResult {
    info!(
        version = ?target_version,
        "Starting CLI install/upgrade"
    );

    match download_cli(target_version, None).await {
        Ok(installed_path) => {
            // Verify the installed version
            let version = get_cli_version(&installed_path).await;
            let version_str = version.as_ref().map(|v| v.to_string());
            let path_str = installed_path.to_string_lossy().to_string();

            info!(
                version = ?version_str,
                path = %path_str,
                "CLI install/upgrade succeeded"
            );

            CliInstallResult {
                success: true,
                version: version_str,
                message: format!(
                    "Claude CLI installed successfully at {}",
                    installed_path.display()
                ),
                cli_path: Some(path_str),
            }
        }
        Err(e) => {
            warn!(error = %e, "CLI install/upgrade failed");
            CliInstallResult {
                success: false,
                version: None,
                message: format!("Installation failed: {}", e),
                cli_path: None,
            }
        }
    }
}

/// Detect whether a CLI binary is a "local build" (not from npm or cc-sdk cache).
///
/// A binary is considered "local" if its path is NOT in:
/// - The cc-sdk cache directory (`~/.cache/cc-sdk/` or `~/Library/Caches/cc-sdk/`)
/// - A standard npm global install location (`node_modules/.bin/`, `/usr/local/bin/`)
/// - A standard Homebrew location (`/opt/homebrew/bin/`, `/usr/local/Cellar/`)
///
/// Local builds cannot be auto-updated and are always considered "outdated"
/// (since we can't verify their provenance).
fn is_local_build(cli_path: &Path) -> bool {
    let path_str = cli_path.to_string_lossy();

    // Check known standard install locations
    let standard_patterns = [
        // cc-sdk cache (primary auto-download location)
        "cc-sdk",
        // npm global installs
        "node_modules/.bin/",
        "node_modules/",
        // Standard system paths (usually from npm global)
        "/usr/local/bin/claude",
        "/usr/bin/claude",
        // Homebrew
        "/opt/homebrew/bin/claude",
        "/opt/homebrew/Cellar/",
        "/usr/local/Cellar/",
    ];

    // Also check if it's the cc-sdk cached path specifically
    if let Some(cached_path) = get_cached_cli_path() {
        if cli_path == cached_path {
            return false; // It's the cached version — not local
        }
    }

    // If the path matches any standard pattern, it's NOT a local build
    for pattern in &standard_patterns {
        if path_str.contains(pattern) {
            return false;
        }
    }

    // Doesn't match any known standard location — consider it local
    debug!(
        path = %path_str,
        "CLI detected as local build (not in standard install locations)"
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_local_build_standard_paths() {
        // Standard npm global installs are NOT local builds
        assert!(
            !is_local_build(Path::new("/usr/local/bin/claude")),
            "/usr/local/bin/claude should not be local"
        );
        assert!(
            !is_local_build(Path::new("/usr/local/lib/node_modules/.bin/claude")),
            "npm global .bin should not be local"
        );
        assert!(
            !is_local_build(Path::new("/opt/homebrew/bin/claude")),
            "Homebrew path should not be local"
        );
    }

    #[test]
    fn test_is_local_build_cc_sdk_cache() {
        // cc-sdk cache path is NOT a local build
        assert!(
            !is_local_build(Path::new("/Users/me/Library/Caches/cc-sdk/cli/claude")),
            "cc-sdk cache should not be local"
        );
        assert!(
            !is_local_build(Path::new("/home/me/.cache/cc-sdk/cli/claude")),
            "Linux cc-sdk cache should not be local"
        );
    }

    #[test]
    fn test_is_local_build_custom_paths() {
        // Custom paths ARE local builds
        assert!(
            is_local_build(Path::new(
                "/home/me/projects/claude-dev/target/release/claude"
            )),
            "Custom project build should be local"
        );
        assert!(
            is_local_build(Path::new("/tmp/claude")),
            "/tmp/claude should be local"
        );
        assert!(
            is_local_build(Path::new("/home/me/bin/claude")),
            "~/bin/claude should be local"
        );
    }

    #[test]
    fn test_cli_version_status_serialization() {
        let status = CliVersionStatus {
            installed: true,
            installed_version: Some("2.5.1".into()),
            latest_version: Some("2.6.0".into()),
            update_available: true,
            is_local_build: false,
            cli_path: Some("/usr/local/bin/claude".into()),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"installed\":true"));
        assert!(json.contains("\"installed_version\":\"2.5.1\""));
        assert!(json.contains("\"update_available\":true"));
    }

    #[test]
    fn test_cli_install_result_serialization() {
        let result = CliInstallResult {
            success: true,
            version: Some("2.6.0".into()),
            message: "Installed successfully".into(),
            cli_path: Some("/path/to/claude".into()),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"version\":\"2.6.0\""));
    }

    /// Integration test: check_cli_status returns a valid struct.
    /// Marked #[ignore] because it requires network access for npm check
    /// and a real Claude CLI installation.
    #[tokio::test]
    #[ignore]
    async fn test_check_cli_status_integration() {
        let status = check_cli_status().await;
        // On a machine with Claude CLI installed, this should be true
        if status.installed {
            assert!(
                status.installed_version.is_some(),
                "Installed CLI should have a version"
            );
            assert!(
                status.cli_path.is_some(),
                "Installed CLI should have a path"
            );
        }
    }
}
