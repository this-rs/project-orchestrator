//! Self-update mechanism for the standalone binary.
//!
//! Checks GitHub Releases for newer versions, downloads the appropriate
//! binary for the current OS/arch, verifies its SHA-256 checksum, and
//! performs an atomic binary replacement.

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::io::Write;
use std::path::PathBuf;

// ============================================================================
// Configuration
// ============================================================================

const GITHUB_REPO_OWNER: &str = "this-rs";
const GITHUB_REPO_NAME: &str = "project-orchestrator";
const BINARY_NAME: &str = "orchestrator";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

// ============================================================================
// Types
// ============================================================================

/// Information about an available update.
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub current_version: String,
    pub latest_version: String,
    pub release_notes: Option<String>,
    pub download_url: String,
    pub expected_checksum: Option<String>,
    pub html_url: String,
}

/// Status returned after performing an update.
#[derive(Debug)]
pub enum UpdateStatus {
    Updated { from: String, to: String },
    AlreadyUpToDate,
}

/// GitHub API release response (subset).
#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    body: Option<String>,
    html_url: String,
    assets: Vec<GitHubAsset>,
}

/// GitHub API asset response (subset).
#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

// ============================================================================
// Platform detection
// ============================================================================

/// Get the archive filename suffix for the current platform.
fn platform_archive_suffix() -> Result<(&'static str, &'static str)> {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        bail!("Unsupported operating system for self-update");
    };

    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        bail!("Unsupported architecture for self-update");
    };

    Ok((os, arch))
}

// ============================================================================
// Version comparison
// ============================================================================

/// Parse a semver string into (major, minor, patch) tuple.
fn parse_semver(version: &str) -> Result<(u64, u64, u64)> {
    let v = version.strip_prefix('v').unwrap_or(version);
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() != 3 {
        bail!("Invalid semver: {}", version);
    }
    Ok((
        parts[0].parse().context("Invalid major version")?,
        parts[1].parse().context("Invalid minor version")?,
        parts[2]
            .split('-')
            .next()
            .unwrap_or("0")
            .parse()
            .context("Invalid patch version")?,
    ))
}

/// Check if `latest` is newer than `current`.
fn is_newer(current: &str, latest: &str) -> Result<bool> {
    let current = parse_semver(current)?;
    let latest = parse_semver(latest)?;
    Ok(latest > current)
}

// ============================================================================
// Update check
// ============================================================================

/// Check GitHub Releases for a newer version.
pub async fn check_for_update() -> Result<Option<UpdateInfo>> {
    let url = format!(
        "https://api.github.com/repos/{}/{}/releases/latest",
        GITHUB_REPO_OWNER, GITHUB_REPO_NAME
    );

    let client = reqwest::Client::builder()
        .user_agent(format!("{}/{}", BINARY_NAME, CURRENT_VERSION))
        .build()?;

    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to reach GitHub API")?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        // No releases yet
        return Ok(None);
    }

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("GitHub API returned {}: {}", status, body);
    }

    let release: GitHubRelease = response
        .json()
        .await
        .context("Failed to parse GitHub release")?;

    let latest_version = release
        .tag_name
        .strip_prefix('v')
        .unwrap_or(&release.tag_name);

    if !is_newer(CURRENT_VERSION, latest_version)? {
        return Ok(None);
    }

    // Find the download URL for our platform
    let (os, arch) = platform_archive_suffix()?;
    let extension = if cfg!(target_os = "windows") {
        "zip"
    } else {
        "tar.gz"
    };
    // Look for the "full" variant (with embedded frontend) first, then fall back to light
    let archive_pattern_full = format!(
        "{}-full-{}-{}-{}.{}",
        BINARY_NAME, latest_version, os, arch, extension
    );
    let archive_pattern_light = format!(
        "{}-{}-{}-{}.{}",
        BINARY_NAME, latest_version, os, arch, extension
    );

    let download_asset = release
        .assets
        .iter()
        .find(|a| a.name == archive_pattern_full)
        .or_else(|| {
            release
                .assets
                .iter()
                .find(|a| a.name == archive_pattern_light)
        });

    let download_url = match download_asset {
        Some(asset) => asset.browser_download_url.clone(),
        None => {
            tracing::warn!(
                "No binary found for {}-{} in release {} (looked for {} and {})",
                os,
                arch,
                latest_version,
                archive_pattern_full,
                archive_pattern_light,
            );
            bail!(
                "No binary available for your platform ({}-{}) in release {}",
                os,
                arch,
                latest_version
            );
        }
    };

    // Try to find checksums file
    let checksums_asset = release
        .assets
        .iter()
        .find(|a| a.name == "checksums-sha256.txt");
    let expected_checksum = if let Some(checksums) = checksums_asset {
        let checksums_text = client
            .get(&checksums.browser_download_url)
            .send()
            .await?
            .text()
            .await?;

        let archive_name = download_asset.map(|a| &a.name).unwrap();
        checksums_text
            .lines()
            .find(|line| line.contains(archive_name.as_str()))
            .and_then(|line| line.split_whitespace().next())
            .map(|s| s.to_string())
    } else {
        None
    };

    Ok(Some(UpdateInfo {
        current_version: CURRENT_VERSION.to_string(),
        latest_version: latest_version.to_string(),
        release_notes: release.body,
        download_url,
        expected_checksum,
        html_url: release.html_url,
    }))
}

// ============================================================================
// Perform update
// ============================================================================

/// Download and install the update, replacing the current binary.
pub async fn perform_update(info: &UpdateInfo) -> Result<UpdateStatus> {
    let client = reqwest::Client::builder()
        .user_agent(format!("{}/{}", BINARY_NAME, CURRENT_VERSION))
        .build()?;

    // Download the archive
    tracing::info!("Downloading {}...", info.download_url);
    let response = client
        .get(&info.download_url)
        .send()
        .await
        .context("Failed to download update")?;

    if !response.status().is_success() {
        bail!("Download failed with status: {}", response.status());
    }

    let archive_bytes = response.bytes().await.context("Failed to read download")?;

    // Verify checksum if available
    if let Some(expected) = &info.expected_checksum {
        use sha2::{Digest, Sha256};
        let actual = hex::encode(Sha256::digest(&archive_bytes));
        if actual != *expected {
            bail!(
                "Checksum mismatch!\n  Expected: {}\n  Got:      {}",
                expected,
                actual
            );
        }
        tracing::info!("Checksum verified: {}", actual);
    } else {
        tracing::warn!("No checksum available — skipping verification");
    }

    // Extract the binary from the archive
    let binary_bytes = extract_binary_from_archive(&archive_bytes)?;

    // Perform atomic replacement of the current binary
    let current_exe = std::env::current_exe().context("Failed to get current executable path")?;
    atomic_replace(&current_exe, &binary_bytes)?;

    tracing::info!(
        "Updated from v{} to v{}",
        info.current_version,
        info.latest_version
    );

    Ok(UpdateStatus::Updated {
        from: info.current_version.clone(),
        to: info.latest_version.clone(),
    })
}

/// Extract the orchestrator binary from a tar.gz or zip archive.
fn extract_binary_from_archive(archive_bytes: &[u8]) -> Result<Vec<u8>> {
    #[cfg(not(target_os = "windows"))]
    {
        extract_from_tar_gz(archive_bytes)
    }
    #[cfg(target_os = "windows")]
    {
        extract_from_zip(archive_bytes)
    }
}

#[cfg(not(target_os = "windows"))]
fn extract_from_tar_gz(archive_bytes: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;

    let decoder = flate2::read::GzDecoder::new(archive_bytes);
    let mut archive = tar::Archive::new(decoder);

    let binary_name = BINARY_NAME;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        if file_name == binary_name {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            return Ok(buf);
        }
    }

    bail!("Binary '{}' not found in archive", binary_name);
}

#[cfg(target_os = "windows")]
fn extract_from_zip(archive_bytes: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;

    let cursor = std::io::Cursor::new(archive_bytes);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let binary_name = format!("{}.exe", BINARY_NAME);

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();
        let file_name = std::path::Path::new(&name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();

        if file_name == binary_name {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            return Ok(buf);
        }
    }

    bail!("Binary '{}' not found in archive", binary_name);
}

/// Atomically replace a binary file.
///
/// Strategy:
/// 1. Write new binary to a temporary file next to the target
/// 2. Set executable permissions
/// 3. Rename old binary to .old backup
/// 4. Rename new binary into place
/// 5. Remove .old backup
fn atomic_replace(target: &PathBuf, new_bytes: &[u8]) -> Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("Cannot determine parent directory of {}", target.display()))?;

    let temp_path = parent.join(format!(".{}.new", BINARY_NAME));
    let backup_path = parent.join(format!(".{}.old", BINARY_NAME));

    // Write new binary
    {
        let mut file = std::fs::File::create(&temp_path)
            .context("Failed to create temporary file for update")?;
        file.write_all(new_bytes)?;
        file.flush()?;
    }

    // Set executable permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(0o755))?;
    }

    // Move current binary to backup
    if target.exists() {
        // Remove old backup if it exists
        let _ = std::fs::remove_file(&backup_path);
        std::fs::rename(target, &backup_path).context("Failed to backup current binary")?;
    }

    // Move new binary into place
    match std::fs::rename(&temp_path, target) {
        Ok(()) => {
            // Clean up backup
            let _ = std::fs::remove_file(&backup_path);
            Ok(())
        }
        Err(e) => {
            // Rollback: restore backup
            tracing::error!("Failed to place new binary, rolling back: {}", e);
            if backup_path.exists() {
                let _ = std::fs::rename(&backup_path, target);
            }
            let _ = std::fs::remove_file(&temp_path);
            Err(e).context("Failed to replace binary with update")
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_semver() {
        assert_eq!(parse_semver("0.1.0").unwrap(), (0, 1, 0));
        assert_eq!(parse_semver("v1.2.3").unwrap(), (1, 2, 3));
        assert_eq!(parse_semver("1.0.0-beta").unwrap(), (1, 0, 0));
    }

    #[test]
    fn test_is_newer() {
        assert!(is_newer("0.1.0", "0.2.0").unwrap());
        assert!(is_newer("0.1.0", "1.0.0").unwrap());
        assert!(is_newer("1.0.0", "1.0.1").unwrap());
        assert!(!is_newer("1.0.0", "1.0.0").unwrap());
        assert!(!is_newer("2.0.0", "1.0.0").unwrap());
    }

    #[test]
    fn test_platform_detection() {
        let result = platform_archive_suffix();
        assert!(result.is_ok());
        let (os, arch) = result.unwrap();
        assert!(!os.is_empty());
        assert!(!arch.is_empty());
    }
}
