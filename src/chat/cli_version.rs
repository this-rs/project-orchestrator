//! CLI version management — check installed version, detect updates, install/upgrade.
//!
//! Uses the Nexus SDK's `find_claude_cli`, `get_cli_version`, `check_latest_npm_version`,
//! and `download_cli` functions to provide a unified CLI management service.

use nexus_claude::{
    check_latest_npm_version, download_cli, find_claude_cli, get_cached_cli_path, get_cli_version,
    SemVer,
};
use serde::Serialize;
use std::path::Path;
use tracing::{debug, info, warn};

/// Run `<cli_path> --version` and parse the raw output ourselves.
///
/// The CLI outputs strings like "2.1.50 (Claude Code)\n". Nexus's `SemVer::parse`
/// cannot handle the trailing `(Claude Code)` suffix — it parses "50 (Claude Code)"
/// as patch and `u32::parse` fails, falling back to 0. So we strip the suffix first.
///
/// Falls back to Nexus's `get_cli_version()` if our parsing fails.
async fn get_cli_version_robust(cli_path: &Path) -> Option<SemVer> {
    // Try running the binary directly
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::process::Command::new(cli_path)
            .arg("--version")
            .stderr(std::process::Stdio::null())
            .output(),
    )
    .await
    .ok()?
    .ok()?;

    let raw = String::from_utf8_lossy(&output.stdout);
    let raw = raw.trim();
    debug!(raw_version = %raw, "Raw CLI --version output");

    // Extract just the version part: take everything before the first space.
    // "2.1.50 (Claude Code)" → "2.1.50"
    // "2.1.50" → "2.1.50"
    let version_str = raw.split_whitespace().next().unwrap_or(raw);

    // Try parsing with Nexus SemVer (now that we stripped the suffix)
    if let Some(v) = SemVer::parse(version_str) {
        return Some(v);
    }

    // Last resort: let Nexus try its own parsing on the full output
    debug!("Robust parsing failed, falling back to nexus get_cli_version");
    get_cli_version(cli_path).await
}

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

    // 2. Get installed version (using our robust parser that handles "X.Y.Z (Claude Code)" suffix)
    let installed_version = get_cli_version_robust(cli_path_ref).await;
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
            // Verify the installed version (robust parsing for "(Claude Code)" suffix)
            let version = get_cli_version_robust(&installed_path).await;
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
            warn!(error = %e, "CLI install/upgrade failed — checking if CLI was installed anyway");

            // Safety net: the official install script may have succeeded but the SDK's
            // post-install checks failed (e.g. ~/.local/bin not in process PATH).
            // Use detect_claude_cli() which checks well-known locations directly.
            if let Some(found_path) = crate::setup_claude::detect_claude_cli() {
                let path = std::path::PathBuf::from(&found_path);
                let version = get_cli_version_robust(&path).await;
                let version_str = version.as_ref().map(|v| v.to_string());

                warn!(
                    path = %found_path,
                    version = ?version_str,
                    "CLI install reported failure but binary found via detect_claude_cli"
                );

                return CliInstallResult {
                    success: true,
                    version: version_str,
                    message: format!(
                        "Claude CLI detected at {} (install script succeeded but post-install check failed)",
                        found_path
                    ),
                    cli_path: Some(found_path),
                };
            }

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
        // Official Anthropic installer (curl https://claude.ai/install.sh | bash)
        // Installs to ~/.local/share/claude/versions/<ver>/ with symlink at ~/.local/bin/claude
        ".local/bin/claude",
        ".local/share/claude/",
        // Windows: npm global (backslash and forward slash variants)
        "AppData\\Roaming\\npm",
        "AppData/Roaming/npm",
        // Windows: Nexus SDK cache (backslash and forward slash variants)
        "AppData\\Local\\cc-sdk",
        "AppData/Local/cc-sdk",
        // Windows: Anthropic installer (backslash and forward slash variants)
        "AppData\\Local\\Programs\\claude",
        "AppData/Local/Programs/claude",
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

/// Whether the Claude Code CLI is on this machine: the configured path when one is set
/// (`CLAUDE_CLI_PATH`, `chat.claude_cli_path`), else nexus's search (PATH, the SDK cache,
/// the usual install locations). Looks at the filesystem only: no process, no download.
pub fn claude_cli_present(configured: Option<&str>) -> bool {
    match configured.map(str::trim).filter(|p| !p.is_empty()) {
        Some(path) => resolve_configured_cli(path).is_some(),
        None => find_claude_cli().is_ok(),
    }
}

/// The file a configured CLI path names, as a spawn would find it: a value with a path
/// separator is that file; a bare name (`claude`) is looked up in the server's PATH, the
/// way `Command::new` does (review of #722, finding 2).
pub fn resolve_configured_cli(configured: &str) -> Option<std::path::PathBuf> {
    resolve_cli_in(configured, std::env::var_os("PATH"))
}

fn resolve_cli_in(
    configured: &str,
    path_var: Option<std::ffi::OsString>,
) -> Option<std::path::PathBuf> {
    let configured = configured.trim();
    if configured.is_empty() {
        return None;
    }
    let has_separator = configured.contains('/') || configured.contains(std::path::MAIN_SEPARATOR);
    if has_separator {
        let path = Path::new(configured);
        return path.is_file().then(|| path.to_path_buf());
    }
    std::env::split_paths(&path_var?)
        .map(|dir| dir.join(configured))
        .find(|candidate| is_executable_file(candidate))
}

/// What execvp accepts while it walks the PATH: a regular file with an execute bit (a
/// non-executable `claude` earlier in the PATH is skipped, review of #722).
fn is_executable_file(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// The Claude Code CLI to configure or run: the configured path (a bare name through the
/// PATH), else the one on the machine (PATH, the usual install places, the SDK cache).
pub fn locate_claude_cli(configured: Option<&str>) -> Option<String> {
    match configured.map(str::trim).filter(|p| !p.is_empty()) {
        Some(configured) => resolve_configured_cli(configured).map(|p| p.display().to_string()),
        None => crate::setup_claude::detect_claude_cli()
            .or_else(|| find_claude_cli().ok().map(|p| p.display().to_string())),
    }
}

/// The CLI path the server is configured with (`CLAUDE_CLI_PATH` over `chat.claude_cli_path`),
/// recorded at boot; `None` before boot or when nothing is configured.
static CONFIGURED_CLI_PATH: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// Records the configured CLI path (at boot, and when it is changed at runtime), for the
/// code that has no config at hand (the feature-graph proposal of the code scan).
pub fn set_configured_cli_path(path: Option<String>) {
    *CONFIGURED_CLI_PATH
        .write()
        .unwrap_or_else(|e| e.into_inner()) = path.filter(|p| !p.trim().is_empty());
}

/// The configured CLI path: what boot recorded, else `CLAUDE_CLI_PATH`.
pub fn configured_cli_path() -> Option<String> {
    CONFIGURED_CLI_PATH
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .or_else(|| std::env::var("CLAUDE_CLI_PATH").ok())
        .filter(|p| !p.trim().is_empty())
}

/// How long a detection of the CLI is trusted before the machine is looked at again.
pub const CLI_DETECTION_TTL: std::time::Duration = std::time::Duration::from_secs(60);

/// What the chat manager knows of the Claude Code CLI (B2): the built-in instance is
/// usable only when its CLI is on the server, and a default chat must not be routed to a
/// missing CLI when another provider is usable.
#[derive(Debug, Default)]
pub enum ClaudeCliProbe {
    /// Never looked up: the built-in instance counts as usable, its absence surfaces when
    /// a session opens (424 `cli_not_found`). The default of a manager nobody asked to
    /// detect (tests, embedders that inject their own Claude Code).
    #[default]
    Assumed,
    /// Looked up on the machine ([`claude_cli_present`]), the answer kept
    /// [`CLI_DETECTION_TTL`]. What the server runs with.
    Detect(std::sync::Mutex<Option<(std::time::Instant, bool)>>),
    /// A fixed answer (tests).
    Fixed(bool),
}

impl ClaudeCliProbe {
    /// The probe that looks at the machine.
    pub fn detecting() -> Self {
        Self::Detect(std::sync::Mutex::new(None))
    }

    /// Whether the CLI is there, `configured` being the path the server is set to use.
    pub fn present(&self, configured: Option<&str>) -> bool {
        match self {
            Self::Assumed => true,
            Self::Fixed(present) => *present,
            Self::Detect(cache) => {
                let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
                if let Some((at, present)) = *cache {
                    if at.elapsed() < CLI_DETECTION_TTL {
                        return present;
                    }
                }
                let present = claude_cli_present(configured);
                *cache = Some((std::time::Instant::now(), present));
                present
            }
        }
    }

    /// Drops what was detected: the next question looks at the machine again (the CLI was
    /// just installed, or its configured path changed).
    pub fn forget(&self) {
        if let Self::Detect(cache) = self {
            *cache.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_configured_cli_path_is_present_only_when_the_file_is_there() {
        let dir = tempfile::TempDir::new().unwrap();
        let cli = dir.path().join("claude");
        let configured = cli.to_str().unwrap();
        assert!(!claude_cli_present(Some(configured)), "no file yet");
        std::fs::write(&cli, "#!/bin/sh\n").unwrap();
        assert!(claude_cli_present(Some(configured)));
        // A directory is not a CLI.
        assert!(!claude_cli_present(Some(dir.path().to_str().unwrap())));
    }

    #[test]
    fn a_bare_configured_cli_name_is_looked_up_in_the_path_like_a_spawn_would() {
        // CLAUDE_CLI_PATH=claude: no separator, so the PATH decides (review of #722).
        let bin = tempfile::TempDir::new().unwrap();
        let empty = tempfile::TempDir::new().unwrap();
        let path = std::env::join_paths([empty.path(), bin.path()]).unwrap();
        assert_eq!(resolve_cli_in("claude", Some(path.clone())), None);
        let cli = bin.path().join("claude");
        std::fs::write(&cli, "#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // Not executable: skipped like execvp skips it.
            std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o644)).unwrap();
            assert_eq!(resolve_cli_in("claude", Some(path.clone())), None);
            // A non-executable `claude` EARLIER in the PATH does not hide the real one.
            std::fs::write(empty.path().join("claude"), "#!/bin/sh\n").unwrap();
            std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert_eq!(resolve_cli_in("claude", Some(path.clone())), Some(cli));
        #[cfg(unix)]
        std::fs::remove_file(empty.path().join("claude")).unwrap();
        // No PATH at all: not found, never a guess.
        assert_eq!(resolve_cli_in("claude", None), None);
        // A value with a separator is that file, never searched.
        let elsewhere = empty.path().join("claude");
        assert_eq!(
            resolve_cli_in(elsewhere.to_str().unwrap(), Some(path)),
            None
        );
    }

    #[test]
    fn the_detection_is_cached_until_forgotten() {
        let dir = tempfile::TempDir::new().unwrap();
        let cli = dir.path().join("claude");
        let configured = cli.to_str().unwrap();
        let probe = ClaudeCliProbe::detecting();
        assert!(!probe.present(Some(configured)));
        std::fs::write(&cli, "#!/bin/sh\n").unwrap();
        assert!(!probe.present(Some(configured)), "still the cached answer");
        probe.forget();
        assert!(
            probe.present(Some(configured)),
            "looked at again once forgotten"
        );
        assert!(ClaudeCliProbe::Assumed.present(Some("/nowhere/claude")));
        assert!(!ClaudeCliProbe::Fixed(false).present(None));
    }

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
    fn test_is_local_build_anthropic_installer() {
        // Official Anthropic installer paths are NOT local builds
        assert!(
            !is_local_build(Path::new("/home/me/.local/bin/claude")),
            "~/.local/bin/claude (Anthropic installer symlink) should not be local"
        );
        assert!(
            !is_local_build(Path::new("/Users/me/.local/bin/claude")),
            "macOS ~/.local/bin/claude should not be local"
        );
        assert!(
            !is_local_build(Path::new(
                "/home/me/.local/share/claude/versions/2.1.50/claude"
            )),
            "~/.local/share/claude/versions/<ver>/claude should not be local"
        );
        assert!(
            !is_local_build(Path::new(
                "/Users/me/.local/share/claude/versions/2.1.50/claude"
            )),
            "macOS ~/.local/share/claude/versions/<ver>/claude should not be local"
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
    fn test_semver_parse_with_claude_code_suffix() {
        // The real CLI outputs "2.1.50 (Claude Code)\n"
        // SemVer::parse must work after we strip the suffix via split_whitespace
        let raw = "2.1.50 (Claude Code)";
        let version_str = raw.split_whitespace().next().unwrap();
        let v = SemVer::parse(version_str).expect("Should parse '2.1.50'");
        assert_eq!(v.to_string(), "2.1.50", "Patch must be 50, not 0");

        // Also test plain version without suffix (npm-style)
        let v2 = SemVer::parse("2.6.0").expect("Should parse '2.6.0'");
        assert_eq!(v2.to_string(), "2.6.0");

        // Nexus SemVer::parse now handles the raw CLI output itself
        // (this-rs/nexus#40); it used to yield 2.1.0.
        let raw_parsed = SemVer::parse(raw).expect("Should parse the raw CLI output");
        assert_eq!(raw_parsed.to_string(), "2.1.50");
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

    #[test]
    fn test_is_local_build_windows_paths() {
        // Windows npm global installs are NOT local builds
        assert!(
            !is_local_build(Path::new(
                "C:\\Users\\me\\AppData\\Roaming\\npm\\claude.cmd"
            )),
            "Windows npm global should not be local"
        );
        assert!(
            !is_local_build(Path::new("C:/Users/me/AppData/Roaming/npm/claude.cmd")),
            "Windows npm global (forward slash) should not be local"
        );
        // Windows cc-sdk cache is NOT a local build
        assert!(
            !is_local_build(Path::new(
                "C:\\Users\\me\\AppData\\Local\\cc-sdk\\cli\\claude.exe"
            )),
            "Windows cc-sdk cache should not be local"
        );
        // Windows Anthropic installer is NOT a local build
        assert!(
            !is_local_build(Path::new(
                "C:\\Users\\me\\AppData\\Local\\Programs\\claude\\claude.exe"
            )),
            "Windows Anthropic installer should not be local"
        );
        // Custom Windows path IS a local build
        assert!(
            is_local_build(Path::new("C:\\dev\\claude.exe")),
            "Custom Windows path should be local"
        );
    }

    /// The version is read from the CLI itself, `X.Y.Z (Claude Code)` suffix
    /// included: `fake_claude` (the scripted Claude CLI of nexus) answers
    /// `--version` the way the real one does.
    #[tokio::test]
    async fn the_installed_version_is_read_from_the_cli_output() {
        let fake = crate::chat::agent_e2e_tests::fake_bin("fake_claude");
        let v = get_cli_version_robust(&fake)
            .await
            .expect("fake_claude answers --version");
        let raw = std::process::Command::new(&fake)
            .arg("--version")
            .output()
            .unwrap();
        let raw = String::from_utf8_lossy(&raw.stdout).trim().to_string();
        assert!(
            raw.ends_with("(Claude Code)"),
            "the fake prints the suffix the parser strips: {raw}"
        );
        assert_eq!(
            Some(v.to_string().as_str()),
            raw.split_whitespace().next(),
            "the whole X.Y.Z, patch included ({raw})"
        );
    }

    #[tokio::test]
    async fn a_cli_that_cannot_be_run_has_no_version() {
        assert!(get_cli_version_robust(Path::new("/nonexistent/claude"))
            .await
            .is_none());
    }

    /// Integration test: check_cli_status returns a valid struct.
    /// Stays `#[ignore]`: it asks npm for the latest version (network), which
    /// no fake stands in for. Its CLI half is covered above on `fake_claude`.
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
