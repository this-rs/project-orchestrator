//! Deployment mode detection — decides what "auto update" can mean here.
//!
//! | mode              | who updates the binary                         |
//! |-------------------|------------------------------------------------|
//! | `standalone`      | this server (download + verify + atomic swap)  |
//! | `source`          | the operator (`git pull && cargo build`)       |
//! | `desktop`         | the Tauri updater (driven by the frontend)     |
//! | `docker`          | the operator (`docker compose pull`)           |
//! | `package_manager` | Homebrew / apt / dnf                           |
//!
//! Only `standalone` may replace its own executable. Every mode still gets the
//! "new version available" notification.

use serde::Serialize;
use std::path::{Component, Path};

/// Env override: `PO_DEPLOYMENT=standalone|source|desktop|docker|package_manager`.
pub const DEPLOYMENT_ENV: &str = "PO_DEPLOYMENT";
/// Env override for restart supervision: `PO_SUPERVISED=launchd|systemd|1|0`.
pub const SUPERVISED_ENV: &str = "PO_SUPERVISED";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentMode {
    Standalone,
    Source,
    Desktop,
    Docker,
    PackageManager,
}

impl DeploymentMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "standalone" | "binary" => Some(Self::Standalone),
            "source" => Some(Self::Source),
            "desktop" | "tauri" => Some(Self::Desktop),
            "docker" | "container" => Some(Self::Docker),
            "package_manager" | "package" | "homebrew" | "brew" | "deb" | "rpm" => {
                Some(Self::PackageManager)
            }
            _ => None,
        }
    }

    /// Whether the server may download and swap its own executable.
    pub fn supports_self_update(self) -> bool {
        matches!(self, Self::Standalone)
    }

    /// How the operator updates this deployment when we cannot.
    pub fn update_hint(self) -> &'static str {
        match self {
            Self::Standalone => "orchestrator update",
            Self::Source => "git pull && cargo build --release, then restart the service",
            Self::Desktop => "Use the in-app updater or download the installer",
            Self::Docker => "docker compose pull && docker compose up -d",
            Self::PackageManager => {
                "Upgrade with your package manager (brew upgrade / apt upgrade / dnf upgrade)"
            }
        }
    }
}

/// Classify an executable path. Pure — see [`detect_current`] for the wiring.
pub fn classify(
    env_override: Option<&str>,
    exe: Option<&Path>,
    in_container: bool,
) -> DeploymentMode {
    if let Some(mode) = env_override.and_then(DeploymentMode::parse) {
        return mode;
    }
    if in_container {
        return DeploymentMode::Docker;
    }
    let Some(exe) = exe else {
        return DeploymentMode::Standalone;
    };
    let s = exe.to_string_lossy();
    if s.contains(".app/Contents/MacOS/") {
        return DeploymentMode::Desktop;
    }
    if s.contains("/Cellar/") || s.starts_with("/usr/bin/") || s.starts_with("/usr/sbin/") {
        return DeploymentMode::PackageManager;
    }
    // `…/target/release/orchestrator`, `…/target/<triple>/debug/orchestrator`,
    // or a custom CARGO_TARGET_DIR named `target*`.
    let comps: Vec<String> = exe
        .components()
        .filter_map(|c| match c {
            Component::Normal(os) => Some(os.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let in_target = comps.iter().enumerate().any(|(i, c)| {
        c.starts_with("target")
            && comps[i + 1..]
                .iter()
                .take(2)
                .any(|p| p == "release" || p == "debug")
    });
    if in_target {
        return DeploymentMode::Source;
    }
    DeploymentMode::Standalone
}

/// Detect the mode of the running process.
pub fn detect_current() -> DeploymentMode {
    let env_override = std::env::var(DEPLOYMENT_ENV).ok();
    let exe = std::env::current_exe()
        .ok()
        .map(|p| std::fs::canonicalize(&p).unwrap_or(p));
    let in_container =
        Path::new("/.dockerenv").exists() || Path::new("/run/.containerenv").exists();
    classify(env_override.as_deref(), exe.as_deref(), in_container)
}

/// Process supervisor that restarts us after a non-zero exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Supervisor {
    Launchd,
    Systemd,
    Other,
}

/// Pure supervisor detection from environment lookups and the parent pid.
pub fn classify_supervisor(
    env: impl Fn(&str) -> Option<String>,
    parent_pid: Option<u32>,
    is_macos: bool,
) -> Option<Supervisor> {
    if let Some(v) = env(SUPERVISED_ENV) {
        return match v.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "no" | "off" | "" => None,
            "launchd" => Some(Supervisor::Launchd),
            "systemd" => Some(Supervisor::Systemd),
            _ => Some(Supervisor::Other),
        };
    }
    if env("INVOCATION_ID").is_some() {
        return Some(Supervisor::Systemd);
    }
    // launchd agents/daemons are direct children of launchd (pid 1).
    if is_macos && parent_pid == Some(1) {
        return Some(Supervisor::Launchd);
    }
    None
}

pub fn detect_supervisor() -> Option<Supervisor> {
    #[cfg(unix)]
    let ppid = Some(std::os::unix::process::parent_id());
    #[cfg(not(unix))]
    let ppid = None;
    classify_supervisor(|k| std::env::var(k).ok(), ppid, cfg!(target_os = "macos"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn c(p: &str) -> DeploymentMode {
        classify(None, Some(&PathBuf::from(p)), false)
    }

    #[test]
    fn classifies_known_layouts() {
        assert_eq!(
            c("/Users/me/project-orchestrator/target/release/orchestrator"),
            DeploymentMode::Source
        );
        assert_eq!(
            c("/src/po/target-update/aarch64-apple-darwin/release/orchestrator"),
            DeploymentMode::Source
        );
        assert_eq!(
            c("/Applications/Project Orchestrator.app/Contents/MacOS/project-orchestrator-desktop"),
            DeploymentMode::Desktop
        );
        assert_eq!(
            c("/opt/homebrew/Cellar/orchestrator/0.0.15/bin/orchestrator"),
            DeploymentMode::PackageManager
        );
        assert_eq!(c("/usr/bin/orchestrator"), DeploymentMode::PackageManager);
        assert_eq!(c("/usr/local/bin/orchestrator"), DeploymentMode::Standalone);
        assert_eq!(c("/home/me/bin/orchestrator"), DeploymentMode::Standalone);
        // A directory merely named "target" without release/debug below it.
        assert_eq!(
            c("/srv/target/bin/orchestrator"),
            DeploymentMode::Standalone
        );
    }

    #[test]
    fn env_override_and_container_take_precedence() {
        let p = PathBuf::from("/usr/local/bin/orchestrator");
        assert_eq!(
            classify(Some("desktop"), Some(&p), true),
            DeploymentMode::Desktop
        );
        assert_eq!(
            classify(Some("bogus"), Some(&p), true),
            DeploymentMode::Docker
        );
        assert_eq!(classify(None, None, false), DeploymentMode::Standalone);
    }

    #[test]
    fn only_standalone_self_updates() {
        assert!(DeploymentMode::Standalone.supports_self_update());
        for m in [
            DeploymentMode::Source,
            DeploymentMode::Desktop,
            DeploymentMode::Docker,
            DeploymentMode::PackageManager,
        ] {
            assert!(!m.supports_self_update(), "{m:?}");
            assert!(!m.update_hint().is_empty());
        }
    }

    #[test]
    fn supervisor_detection() {
        let none = |_: &str| None;
        assert_eq!(
            classify_supervisor(none, Some(1), true),
            Some(Supervisor::Launchd)
        );
        assert_eq!(classify_supervisor(none, Some(4242), true), None);
        assert_eq!(classify_supervisor(none, Some(1), false), None);
        let systemd = |k: &str| (k == "INVOCATION_ID").then(|| "abc".to_string());
        assert_eq!(
            classify_supervisor(systemd, Some(77), false),
            Some(Supervisor::Systemd)
        );
        let off = |k: &str| (k == SUPERVISED_ENV).then(|| "0".to_string());
        assert_eq!(classify_supervisor(off, Some(1), true), None);
        let forced = |k: &str| (k == SUPERVISED_ENV).then(|| "launchd".to_string());
        assert_eq!(
            classify_supervisor(forced, Some(99), false),
            Some(Supervisor::Launchd)
        );
    }
}
