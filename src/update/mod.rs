//! Self-update mechanism.
//!
//! - [`version`]: what the running build is (release tag or source checkout).
//! - [`deployment`]: standalone / source / desktop / docker / package manager.
//! - [`service`]: startup + periodic GitHub release check, API status, and
//!   (standalone only, when `chat.auto_update_app` is on) staged self-update.
//!
//! This file holds the GitHub plumbing shared by the service and the
//! `orchestrator update` CLI: release fetch, asset selection, checksum
//! verification and atomic binary replacement.

pub mod deployment;
pub mod service;
pub mod version;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

pub use service::{global, UpdateService, UpdateStatus};

// ============================================================================
// Configuration
// ============================================================================

const GITHUB_REPO_OWNER: &str = "this-rs";
const GITHUB_REPO_NAME: &str = "project-orchestrator";
const BINARY_NAME: &str = "orchestrator";
const CHECKSUMS_ASSET: &str = "checksums-sha256.txt";

// ============================================================================
// Types
// ============================================================================

/// Information about an available update (CLI flow).
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
pub enum UpdateOutcome {
    Updated { from: String, to: String },
    AlreadyUpToDate,
}

/// A published release, as far as the updater cares.
#[derive(Debug, Clone, PartialEq)]
pub struct ReleaseInfo {
    pub tag: String,
    pub version: semver::Version,
    pub html_url: String,
    pub body: Option<String>,
    pub published_at: Option<String>,
    pub assets: Vec<ReleaseAsset>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ReleaseAsset {
    pub name: String,
    #[serde(rename = "browser_download_url")]
    pub url: String,
}

/// GitHub API release response (subset).
#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    body: Option<String>,
    html_url: String,
    published_at: Option<String>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    assets: Vec<ReleaseAsset>,
}

impl GitHubRelease {
    fn into_release(self) -> Result<ReleaseInfo> {
        let version = version::parse_version(&self.tag_name)
            .ok_or_else(|| anyhow!("Release tag '{}' is not a semver version", self.tag_name))?;
        Ok(ReleaseInfo {
            tag: self.tag_name,
            version,
            html_url: self.html_url,
            body: self.body,
            published_at: self.published_at,
            assets: self.assets,
        })
    }
}

// ============================================================================
// Pluggable I/O (injected into the service; mocked in tests)
// ============================================================================

/// Where releases come from.
#[async_trait]
pub trait ReleaseSource: Send + Sync {
    /// Latest stable release, `None` if the repository has none.
    async fn latest(&self) -> Result<Option<ReleaseInfo>>;
}

/// Installs a release over the running executable.
#[async_trait]
pub trait Installer: Send + Sync {
    async fn install(&self, release: &ReleaseInfo) -> Result<()>;
}

fn http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent(format!(
            "{}/{}",
            BINARY_NAME,
            version::current_build().display()
        ))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .build()?)
}

/// `GET /repos/{owner}/{repo}/releases/latest` (GitHub excludes drafts and
/// pre-releases from this endpoint).
pub struct GitHubReleaseSource {
    client: reqwest::Client,
    url: String,
}

impl GitHubReleaseSource {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: http_client()?,
            url: format!(
                "https://api.github.com/repos/{}/{}/releases/latest",
                GITHUB_REPO_OWNER, GITHUB_REPO_NAME
            ),
        })
    }

    #[cfg(test)]
    fn with_url(url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            url,
        }
    }
}

#[async_trait]
impl ReleaseSource for GitHubReleaseSource {
    async fn latest(&self) -> Result<Option<ReleaseInfo>> {
        let response = self
            .client
            .get(&self.url)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .context("Failed to reach GitHub API")?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None); // no releases yet
        }
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let body: String = body.chars().take(200).collect();
            bail!("GitHub API returned {}: {}", status, body);
        }
        let release: GitHubRelease = response
            .json()
            .await
            .context("Failed to parse GitHub release")?;
        if release.draft || release.prerelease {
            return Ok(None);
        }
        release.into_release().map(Some)
    }
}

/// Real installer for the standalone binary: platform archive + mandatory
/// SHA-256 verification + sanity check + atomic swap of `current_exe()`.
pub struct SelfInstaller {
    client: reqwest::Client,
}

impl SelfInstaller {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: http_client()?,
        })
    }
}

#[async_trait]
impl Installer for SelfInstaller {
    async fn install(&self, release: &ReleaseInfo) -> Result<()> {
        let (os, arch) = platform_archive_suffix()?;
        let asset = select_archive(&release.assets, &release.version.to_string(), os, arch)
            .ok_or_else(|| anyhow!("No binary for {}-{} in release {}", os, arch, release.tag))?;
        let checksum = fetch_checksum(&self.client, &release.assets, &asset.name)
            .await?
            .ok_or_else(|| {
                anyhow!(
                    "No SHA-256 checksum published for {} — refusing unattended install",
                    asset.name
                )
            })?;
        download_verify_install(&self.client, &asset.url, Some(&checksum)).await
    }
}

// ============================================================================
// Platform & asset selection
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

fn archive_extension(os: &str) -> &'static str {
    if os == "windows" {
        "zip"
    } else {
        "tar.gz"
    }
}

/// Pick the release archive for `os`/`arch`: the "full" variant (embedded
/// frontend) first, then the light one. Names follow release.yml:
/// `orchestrator[-full]-<version>-<os>-<arch>.<ext>`.
fn select_archive<'a>(
    assets: &'a [ReleaseAsset],
    version: &str,
    os: &str,
    arch: &str,
) -> Option<&'a ReleaseAsset> {
    let ext = archive_extension(os);
    let full = format!("{BINARY_NAME}-full-{version}-{os}-{arch}.{ext}");
    let light = format!("{BINARY_NAME}-{version}-{os}-{arch}.{ext}");
    assets
        .iter()
        .find(|a| a.name == full)
        .or_else(|| assets.iter().find(|a| a.name == light))
}

/// Find `file_name`'s hash in a `sha256sum` listing (`<hash>  <name>`, names
/// may contain spaces, `*` marks binary mode).
fn checksum_for(listing: &str, file_name: &str) -> Option<String> {
    listing.lines().find_map(|line| {
        let (hash, name) = line.trim().split_once(char::is_whitespace)?;
        let name = name.trim_start();
        let name = name.strip_prefix('*').unwrap_or(name);
        let name = name.strip_prefix("./").unwrap_or(name);
        (name == file_name && hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()))
            .then(|| hash.to_ascii_lowercase())
    })
}

async fn fetch_checksum(
    client: &reqwest::Client,
    assets: &[ReleaseAsset],
    file_name: &str,
) -> Result<Option<String>> {
    let Some(sums) = assets.iter().find(|a| a.name == CHECKSUMS_ASSET) else {
        return Ok(None);
    };
    let text = client
        .get(&sums.url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    Ok(checksum_for(&text, file_name))
}

// ============================================================================
// CLI flow (`orchestrator update`)
// ============================================================================

/// Check GitHub Releases for a newer version.
pub async fn check_for_update() -> Result<Option<UpdateInfo>> {
    let build = version::current_build();
    let Some(release) = GitHubReleaseSource::new()?.latest().await? else {
        return Ok(None);
    };
    if !build.is_behind(&release.version) {
        return Ok(None);
    }
    let (os, arch) = platform_archive_suffix()?;
    let latest_version = release.version.to_string();
    let asset = select_archive(&release.assets, &latest_version, os, arch).ok_or_else(|| {
        anyhow!(
            "No binary available for your platform ({}-{}) in release {}",
            os,
            arch,
            latest_version
        )
    })?;
    let expected_checksum = fetch_checksum(&http_client()?, &release.assets, &asset.name).await?;

    Ok(Some(UpdateInfo {
        current_version: build.display(),
        latest_version,
        release_notes: release.body.clone(),
        download_url: asset.url.clone(),
        expected_checksum,
        html_url: release.html_url.clone(),
    }))
}

/// Download and install the update, replacing the current binary.
pub async fn perform_update(info: &UpdateInfo) -> Result<UpdateOutcome> {
    download_verify_install(
        &http_client()?,
        &info.download_url,
        info.expected_checksum.as_deref(),
    )
    .await?;
    tracing::info!(
        "Updated from v{} to v{}",
        info.current_version,
        info.latest_version
    );
    Ok(UpdateOutcome::Updated {
        from: info.current_version.clone(),
        to: info.latest_version.clone(),
    })
}

// ============================================================================
// Download → verify → extract → swap
// ============================================================================

async fn download_verify_install(
    client: &reqwest::Client,
    url: &str,
    expected_checksum: Option<&str>,
) -> Result<()> {
    tracing::info!("Downloading {}...", url);
    let response = client
        .get(url)
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .context("Failed to download update")?;
    if !response.status().is_success() {
        bail!("Download failed with status: {}", response.status());
    }
    let archive_bytes = response.bytes().await.context("Failed to read download")?;

    match expected_checksum {
        Some(expected) => {
            verify_sha256(&archive_bytes, expected)?;
            tracing::info!("Checksum verified");
        }
        None => tracing::warn!("No checksum available — skipping verification"),
    }

    let binary_bytes = extract_binary_from_archive(&archive_bytes)?;
    validate_executable(&binary_bytes)?;

    let current_exe = std::env::current_exe().context("Failed to get current executable path")?;
    let current_exe = std::fs::canonicalize(&current_exe).unwrap_or(current_exe);
    atomic_replace(&current_exe, &binary_bytes)
}

fn verify_sha256(bytes: &[u8], expected: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let actual = hex::encode(Sha256::digest(bytes));
    if !actual.eq_ignore_ascii_case(expected.trim()) {
        bail!(
            "Checksum mismatch!\n  Expected: {}\n  Got:      {}",
            expected,
            actual
        );
    }
    Ok(())
}

/// Reject obviously broken payloads before they replace a working binary.
fn validate_executable(bytes: &[u8]) -> Result<()> {
    const ELF: &[u8] = b"\x7fELF";
    const MACHO: [&[u8]; 4] = [
        b"\xcf\xfa\xed\xfe", // 64-bit LE
        b"\xce\xfa\xed\xfe", // 32-bit LE
        b"\xca\xfe\xba\xbe", // universal
        b"\xfe\xed\xfa\xcf", // 64-bit BE
    ];
    const PE: &[u8] = b"MZ";
    if bytes.len() < 1024 {
        bail!("Downloaded binary is too small ({} bytes)", bytes.len());
    }
    let ok = if cfg!(target_os = "macos") {
        MACHO.iter().any(|m| bytes.starts_with(m))
    } else if cfg!(target_os = "windows") {
        bytes.starts_with(PE)
    } else {
        bytes.starts_with(ELF)
    };
    if !ok {
        bail!("Downloaded file is not an executable for this platform");
    }
    Ok(())
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

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        if file_name == BINARY_NAME {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf)?;
            return Ok(buf);
        }
    }

    bail!("Binary '{}' not found in archive", BINARY_NAME);
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
/// 1. Write the new binary next to the target (same filesystem) and fsync it
/// 2. Set executable permissions
/// 3. Rename old binary to .old backup
/// 4. Rename new binary into place (rollback on failure)
/// 5. Remove .old backup
///
/// On Unix the running process keeps executing the old inode; the new binary
/// takes effect on the next start.
fn atomic_replace(target: &Path, new_bytes: &[u8]) -> Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("Cannot determine parent directory of {}", target.display()))?;

    let temp_path = parent.join(format!(".{}.new", BINARY_NAME));
    let backup_path = parent.join(format!(".{}.old", BINARY_NAME));

    {
        let mut file = std::fs::File::create(&temp_path)
            .context("Failed to create temporary file for update")?;
        file.write_all(new_bytes)?;
        file.sync_all()?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(0o755))?;
    }

    if target.exists() {
        let _ = std::fs::remove_file(&backup_path);
        std::fs::rename(target, &backup_path).context("Failed to backup current binary")?;
    }

    match std::fs::rename(&temp_path, target) {
        Ok(()) => {
            let _ = std::fs::remove_file(&backup_path);
            Ok(())
        }
        Err(e) => {
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

    fn asset(name: &str) -> ReleaseAsset {
        ReleaseAsset {
            name: name.into(),
            url: format!("https://example.invalid/{name}"),
        }
    }

    #[test]
    fn test_platform_detection() {
        let (os, arch) = platform_archive_suffix().unwrap();
        assert!(!os.is_empty());
        assert!(!arch.is_empty());
    }

    #[test]
    fn selects_full_archive_then_light() {
        // Real asset names from the v0.0.15 release.
        let assets = vec![
            asset("checksums-sha256.txt"),
            asset("orchestrator-0.0.15-macos-arm64.tar.gz"),
            asset("orchestrator-full-0.0.15-macos-arm64.tar.gz"),
            asset("orchestrator-0.0.15-linux-x86_64.tar.gz"),
            asset("orchestrator-full-0.0.15-windows-x86_64.zip"),
        ];
        assert_eq!(
            select_archive(&assets, "0.0.15", "macos", "arm64")
                .unwrap()
                .name,
            "orchestrator-full-0.0.15-macos-arm64.tar.gz"
        );
        assert_eq!(
            select_archive(&assets, "0.0.15", "linux", "x86_64")
                .unwrap()
                .name,
            "orchestrator-0.0.15-linux-x86_64.tar.gz"
        );
        assert_eq!(
            select_archive(&assets, "0.0.15", "windows", "x86_64")
                .unwrap()
                .name,
            "orchestrator-full-0.0.15-windows-x86_64.zip"
        );
        assert!(select_archive(&assets, "0.0.15", "linux", "arm64").is_none());
    }

    #[test]
    fn checksum_lookup_is_exact() {
        let h1 = "a".repeat(64);
        let h2 = "B".repeat(64);
        let listing = format!(
            "{h1}  orchestrator-full-0.0.15-macos-arm64.tar.gz\n\
             {h2} *orchestrator-0.0.15-macos-arm64.tar.gz\n\
             {h1}  Project Orchestrator_0.0.15_aarch64.dmg\n\
             nothex  broken.tar.gz\n"
        );
        assert_eq!(
            checksum_for(&listing, "orchestrator-0.0.15-macos-arm64.tar.gz"),
            Some("b".repeat(64))
        );
        assert_eq!(
            checksum_for(&listing, "Project Orchestrator_0.0.15_aarch64.dmg"),
            Some(h1.clone())
        );
        assert_eq!(checksum_for(&listing, "0.0.15-macos-arm64.tar.gz"), None);
        assert_eq!(checksum_for(&listing, "broken.tar.gz"), None);
    }

    #[test]
    fn sha256_verification() {
        let digest = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"; // "hello"
        assert!(verify_sha256(b"hello", digest).is_ok());
        assert!(verify_sha256(b"hello", &digest.to_uppercase()).is_ok());
        assert!(verify_sha256(b"hell0", digest).is_err());
    }

    #[test]
    fn rejects_non_executables() {
        assert!(validate_executable(b"\x7fELF").is_err()); // too small
        assert!(validate_executable(&[b'<'; 4096]).is_err()); // HTML error page
        let mut native = if cfg!(target_os = "macos") {
            b"\xcf\xfa\xed\xfe".to_vec()
        } else if cfg!(target_os = "windows") {
            b"MZ".to_vec()
        } else {
            b"\x7fELF".to_vec()
        };
        native.resize(4096, 0);
        assert!(validate_executable(&native).is_ok());
    }

    #[test]
    fn atomic_replace_swaps_file() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join(BINARY_NAME);
        std::fs::write(&target, b"old").unwrap();
        atomic_replace(&target, b"new").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
        assert!(!dir.path().join(format!(".{BINARY_NAME}.old")).exists());
        assert!(!dir.path().join(format!(".{BINARY_NAME}.new")).exists());
    }

    #[tokio::test]
    async fn github_source_parses_latest_release() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "tag_name": "v0.0.16",
                "html_url": "https://github.com/this-rs/project-orchestrator/releases/tag/v0.0.16",
                "body": "## What's new",
                "published_at": "2026-09-01T00:00:00Z",
                "draft": false,
                "prerelease": false,
                "assets": [{"name": "latest.json", "browser_download_url": "https://x/latest.json"}]
            })))
            .mount(&server)
            .await;
        let src = GitHubReleaseSource::with_url(format!("{}/latest", server.uri()));
        let r = src.latest().await.unwrap().unwrap();
        assert_eq!(r.version, semver::Version::new(0, 0, 16));
        assert_eq!(r.tag, "v0.0.16");
        assert_eq!(r.assets.len(), 1);

        Mock::given(method("GET"))
            .and(path("/missing"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        let src = GitHubReleaseSource::with_url(format!("{}/missing", server.uri()));
        assert!(src.latest().await.unwrap().is_none());

        Mock::given(method("GET"))
            .and(path("/limited"))
            .respond_with(ResponseTemplate::new(403).set_body_string("rate limited"))
            .mount(&server)
            .await;
        let src = GitHubReleaseSource::with_url(format!("{}/limited", server.uri()));
        assert!(src.latest().await.is_err());
    }
}
