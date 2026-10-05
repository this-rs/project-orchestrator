//! Background update service.
//!
//! - Checks GitHub releases shortly after startup, then every ~6 h (with
//!   jitter). Never blocks or fails server startup; offline / rate-limited
//!   checks are recorded in `last_error` and retried on the next tick.
//! - Exposes a snapshot through `GET /api/version` (`update` field) so the
//!   frontend can show a "new version available" notification.
//! - When `chat.auto_update_app` is true AND the deployment can replace its
//!   own binary (standalone only), downloads + verifies + atomically swaps the
//!   binary and marks the update as *staged*. It never restarts on its own:
//!   running chat sessions are only interrupted by an explicit user action
//!   (`POST /api/update/restart`), or by the next natural restart.

use super::deployment::{DeploymentMode, Supervisor};
use super::version::BuildVersion;
use super::{Installer, ReleaseInfo, ReleaseSource};
use chrono::{DateTime, Utc};
use semver::Version;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

/// Exit code used to ask the supervisor (launchd `KeepAlive.SuccessfulExit =
/// false`, systemd `Restart=on-failure`) to start us again. EX_TEMPFAIL.
pub const RESTART_EXIT_CODE: i32 = 75;

/// Env switch to disable release checks entirely (air-gapped installs):
/// `PO_UPDATE_CHECK=0|false|off`.
pub const UPDATE_CHECK_ENV: &str = "PO_UPDATE_CHECK";

const NOTES_EXCERPT_CHARS: usize = 400;

/// Timing of the periodic check.
#[derive(Debug, Clone, Copy)]
pub struct Schedule {
    pub initial_delay: Duration,
    pub interval: Duration,
    /// Up to this much random delay is added to every wait.
    pub jitter: Duration,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_secs(15),
            interval: Duration::from_secs(6 * 3600),
            jitter: Duration::from_secs(20 * 60),
        }
    }
}

/// Static configuration of the service.
#[derive(Debug, Clone)]
pub struct UpdateServiceConfig {
    pub build: BuildVersion,
    pub mode: DeploymentMode,
    pub supervisor: Option<Supervisor>,
    /// Initial value of `chat.auto_update_app` (can be toggled at runtime).
    pub auto_update: bool,
    pub check_enabled: bool,
}

#[derive(Debug, Default, Clone)]
struct State {
    latest: Option<ReleaseInfo>,
    checked_at: Option<DateTime<Utc>>,
    last_error: Option<String>,
    staged: Option<Version>,
    installing: bool,
    install_error: Option<String>,
}

/// Snapshot served by the API (`GET /api/version` → `update`).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct UpdateStatus {
    /// Version the running build corresponds to (e.g. `0.0.15`).
    pub current: String,
    /// Display string including commits ahead for source builds.
    pub current_build: String,
    pub latest: Option<String>,
    pub update_available: bool,
    pub release_url: Option<String>,
    pub notes_excerpt: Option<String>,
    pub published_at: Option<String>,
    pub checked_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
    pub check_enabled: bool,
    /// `chat.auto_update_app`.
    pub auto_update_enabled: bool,
    pub deployment_mode: DeploymentMode,
    /// Whether this server can install the update itself.
    pub self_update_supported: bool,
    /// What the operator should run when we cannot self-update.
    pub update_hint: &'static str,
    pub installing: bool,
    pub install_error: Option<String>,
    /// Version already written to disk, waiting for a restart.
    pub staged_version: Option<String>,
    pub restart_required: bool,
    /// A supervisor will bring us back after `POST /api/update/restart`.
    pub restart_supported: bool,
}

/// Why a manual install request was refused.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum InstallError {
    #[error("self-update is not supported for this deployment ({0:?}): {1}")]
    NotSupported(DeploymentMode, &'static str),
    #[error("no newer release is known yet")]
    NoUpdate,
    #[error("an update is already being installed")]
    AlreadyRunning,
    #[error("install failed: {0}")]
    Failed(String),
}

pub struct UpdateService {
    cfg: UpdateServiceConfig,
    auto_update: AtomicBool,
    source: Arc<dyn ReleaseSource>,
    installer: Arc<dyn Installer>,
    state: RwLock<State>,
    install_lock: tokio::sync::Mutex<()>,
}

fn excerpt(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() <= NOTES_EXCERPT_CHARS {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(NOTES_EXCERPT_CHARS).collect();
    format!("{}…", cut.trim_end())
}

impl UpdateService {
    pub fn new(
        cfg: UpdateServiceConfig,
        source: Arc<dyn ReleaseSource>,
        installer: Arc<dyn Installer>,
    ) -> Self {
        Self {
            auto_update: AtomicBool::new(cfg.auto_update),
            cfg,
            source,
            installer,
            state: RwLock::new(State::default()),
            install_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn set_auto_update(&self, enabled: bool) {
        self.auto_update.store(enabled, Ordering::SeqCst);
    }

    pub fn auto_update_enabled(&self) -> bool {
        self.auto_update.load(Ordering::SeqCst)
    }

    pub fn deployment_mode(&self) -> DeploymentMode {
        self.cfg.mode
    }

    pub fn supervisor(&self) -> Option<Supervisor> {
        self.cfg.supervisor
    }

    fn read(&self) -> State {
        self.state.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn write<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let mut guard = self.state.write().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }

    fn newer_release(&self, state: &State) -> Option<ReleaseInfo> {
        state
            .latest
            .clone()
            .filter(|r| self.cfg.build.is_behind(&r.version))
    }

    pub fn status(&self) -> UpdateStatus {
        let s = self.read();
        let update_available = self.newer_release(&s).is_some();
        UpdateStatus {
            current: self.cfg.build.base.to_string(),
            current_build: self.cfg.build.display(),
            latest: s.latest.as_ref().map(|r| r.version.to_string()),
            update_available,
            release_url: s.latest.as_ref().map(|r| r.html_url.clone()),
            notes_excerpt: s
                .latest
                .as_ref()
                .and_then(|r| r.body.as_deref())
                .map(excerpt)
                .filter(|e| !e.is_empty()),
            published_at: s.latest.as_ref().and_then(|r| r.published_at.clone()),
            checked_at: s.checked_at,
            last_error: s.last_error.clone(),
            check_enabled: self.cfg.check_enabled,
            auto_update_enabled: self.auto_update_enabled(),
            deployment_mode: self.cfg.mode,
            self_update_supported: self.cfg.mode.supports_self_update(),
            update_hint: self.cfg.mode.update_hint(),
            installing: s.installing,
            install_error: s.install_error.clone(),
            staged_version: s.staged.as_ref().map(|v| v.to_string()),
            restart_required: s.staged.is_some(),
            restart_supported: self.cfg.supervisor.is_some(),
        }
    }

    /// One check cycle. Never returns an error: failures are recorded in the
    /// status and the previous known release is kept.
    pub async fn check_now(&self) -> UpdateStatus {
        match self.source.latest().await {
            Ok(latest) => self.write(|s| {
                s.latest = latest;
                s.checked_at = Some(Utc::now());
                s.last_error = None;
            }),
            Err(e) => {
                tracing::info!("Update check failed (will retry later): {e:#}");
                self.write(|s| {
                    s.checked_at = Some(Utc::now());
                    s.last_error = Some(format!("{e:#}"));
                });
                return self.status();
            }
        }

        let status = self.status();
        if status.update_available {
            tracing::info!(
                current = %status.current_build,
                latest = ?status.latest,
                mode = ?self.cfg.mode,
                auto_update = status.auto_update_enabled,
                "New Project Orchestrator release available"
            );
        }
        if self.should_auto_install() {
            match self.install_latest().await {
                Ok(v) => tracing::info!("Update v{v} staged — restart to apply"),
                Err(e) => tracing::warn!("Automatic update not applied: {e}"),
            }
        }
        self.status()
    }

    /// The `auto_update_app` rule: only when enabled, supported, newer, and
    /// not already staged/failed for that same version.
    fn should_auto_install(&self) -> bool {
        if !self.auto_update_enabled() || !self.cfg.mode.supports_self_update() {
            return false;
        }
        let s = self.read();
        let Some(release) = self.newer_release(&s) else {
            return false;
        };
        if s.staged.as_ref() == Some(&release.version) {
            return false;
        }
        // Don't hammer a broken release every cycle: after a failure, retry
        // only once a different version shows up (manual install still works).
        s.install_error.is_none()
    }

    /// Download, verify and stage the latest release (standalone only).
    pub async fn install_latest(&self) -> Result<Version, InstallError> {
        if !self.cfg.mode.supports_self_update() {
            return Err(InstallError::NotSupported(
                self.cfg.mode,
                self.cfg.mode.update_hint(),
            ));
        }
        let Ok(_guard) = self.install_lock.try_lock() else {
            return Err(InstallError::AlreadyRunning);
        };
        let release = self
            .newer_release(&self.read())
            .ok_or(InstallError::NoUpdate)?;
        if self.read().staged.as_ref() == Some(&release.version) {
            return Ok(release.version);
        }

        self.write(|s| {
            s.installing = true;
            s.install_error = None;
        });
        let result = self.installer.install(&release).await;
        self.write(|s| {
            s.installing = false;
            match &result {
                Ok(()) => s.staged = Some(release.version.clone()),
                Err(e) => s.install_error = Some(format!("{e:#}")),
            }
        });
        result
            .map(|()| release.version)
            .map_err(|e| InstallError::Failed(format!("{e:#}")))
    }

    /// Run `check_now` on the schedule until the task is aborted.
    pub fn spawn_periodic(self: &Arc<Self>, schedule: Schedule) -> tokio::task::JoinHandle<()> {
        let svc = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(schedule.initial_delay + jitter(schedule.jitter)).await;
            loop {
                svc.check_now().await;
                tokio::time::sleep(schedule.interval + jitter(schedule.jitter)).await;
            }
        })
    }
}

/// Cheap non-cryptographic jitter in `[0, max)`.
fn jitter(max: Duration) -> Duration {
    let max_ms = max.as_millis() as u64;
    if max_ms == 0 {
        return Duration::ZERO;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ (std::process::id() as u64).wrapping_mul(2654435761))
        .unwrap_or(0);
    Duration::from_millis(nanos % max_ms)
}

// ============================================================================
// Process-wide instance (read by /api/version and the update endpoints)
// ============================================================================

static GLOBAL: OnceLock<Arc<UpdateService>> = OnceLock::new();

/// The service started by `start_server`, if any.
pub fn global() -> Option<&'static Arc<UpdateService>> {
    GLOBAL.get()
}

/// Apply a runtime change of `chat.auto_update_app` to the running service.
/// Returns whether a service was reached (false when the checker is disabled
/// or failed to start, in which case there is nothing to toggle).
pub fn apply_auto_update(enabled: bool) -> bool {
    match global() {
        Some(svc) => {
            svc.set_auto_update(enabled);
            true
        }
        None => false,
    }
}

/// Whether release checks are enabled (`PO_UPDATE_CHECK` not set to off).
pub fn check_enabled_from_env() -> bool {
    !matches!(
        std::env::var(UPDATE_CHECK_ENV)
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Ok("0" | "false" | "off" | "no")
    )
}

/// Build the real service, register it globally and start the periodic
/// checker. Safe to call once; later calls are ignored. Never fails startup.
pub fn start_global(auto_update: bool) -> Option<Arc<UpdateService>> {
    if let Some(existing) = GLOBAL.get() {
        return Some(existing.clone());
    }
    let cfg = UpdateServiceConfig {
        build: super::version::current_build(),
        mode: super::deployment::detect_current(),
        supervisor: super::deployment::detect_supervisor(),
        auto_update,
        check_enabled: check_enabled_from_env(),
    };
    let source = match super::GitHubReleaseSource::new() {
        Ok(s) => Arc::new(s),
        Err(e) => {
            tracing::warn!("Update checker disabled: {e:#}");
            return None;
        }
    };
    let installer = match super::SelfInstaller::new() {
        Ok(i) => Arc::new(i),
        Err(e) => {
            tracing::warn!("Update checker disabled: {e:#}");
            return None;
        }
    };
    tracing::info!(
        version = %cfg.build.display(),
        mode = ?cfg.mode,
        supervisor = ?cfg.supervisor,
        auto_update,
        check_enabled = cfg.check_enabled,
        "Update service initialized"
    );
    let check_enabled = cfg.check_enabled;
    let svc = Arc::new(UpdateService::new(cfg, source, installer));
    let svc = GLOBAL.get_or_init(|| svc).clone();
    if check_enabled {
        svc.spawn_periodic(Schedule::default());
    }
    Some(svc)
}

/// Exit with [`RESTART_EXIT_CODE`] after `delay` (lets the HTTP response
/// flush). The supervisor starts the freshly installed binary.
pub fn schedule_restart(delay: Duration) {
    tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        tracing::warn!("Restarting to apply update (exit {RESTART_EXIT_CODE})");
        std::process::exit(RESTART_EXIT_CODE);
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::version::resolve_build_version;
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Mutex;

    /// Scripted release source: pops one response per call (last one sticks).
    struct FakeSource {
        responses: Mutex<Vec<Result<Option<ReleaseInfo>, String>>>,
        calls: AtomicUsize,
    }

    impl FakeSource {
        fn new(responses: Vec<Result<Option<ReleaseInfo>, String>>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses),
                calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl ReleaseSource for FakeSource {
        async fn latest(&self) -> Result<Option<ReleaseInfo>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut r = self.responses.lock().unwrap();
            let next = if r.len() > 1 {
                r.remove(0)
            } else {
                r[0].clone()
            };
            next.map_err(|e| anyhow!(e))
        }
    }

    #[derive(Default)]
    struct FakeInstaller {
        calls: AtomicUsize,
        fail: bool,
    }

    #[async_trait]
    impl Installer for FakeInstaller {
        async fn install(&self, _r: &ReleaseInfo) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                Err(anyhow!("checksum mismatch"))
            } else {
                Ok(())
            }
        }
    }

    fn release(v: &str) -> ReleaseInfo {
        ReleaseInfo {
            tag: format!("v{v}"),
            version: Version::parse(v).unwrap(),
            html_url: format!("https://github.com/this-rs/project-orchestrator/releases/tag/v{v}"),
            body: Some("## Highlights\n- faster".into()),
            published_at: Some("2026-09-01T00:00:00Z".into()),
            assets: vec![],
        }
    }

    fn svc(
        mode: DeploymentMode,
        auto: bool,
        source: Arc<FakeSource>,
        installer: Arc<FakeInstaller>,
    ) -> UpdateService {
        UpdateService::new(
            UpdateServiceConfig {
                build: resolve_build_version("0.0.15", ""),
                mode,
                supervisor: Some(Supervisor::Launchd),
                auto_update: auto,
                check_enabled: true,
            },
            source,
            installer,
        )
    }

    #[tokio::test]
    async fn reports_available_update_with_api_shape() {
        let s = svc(
            DeploymentMode::Source,
            false,
            FakeSource::new(vec![Ok(Some(release("0.0.16")))]),
            Arc::default(),
        );
        let before = s.status();
        assert!(!before.update_available);
        assert!(before.checked_at.is_none());

        let st = s.check_now().await;
        assert!(st.update_available);
        assert_eq!(st.current, "0.0.15");
        assert_eq!(st.latest.as_deref(), Some("0.0.16"));
        assert!(st.release_url.unwrap().ends_with("/v0.0.16"));
        assert_eq!(st.notes_excerpt.as_deref(), Some("## Highlights\n- faster"));
        assert!(st.checked_at.is_some());
        assert!(!st.auto_update_enabled);
        assert!(!st.self_update_supported);

        // Serialized field names are the frontend contract.
        let json = serde_json::to_value(s.status()).unwrap();
        for key in [
            "current",
            "current_build",
            "latest",
            "update_available",
            "release_url",
            "notes_excerpt",
            "checked_at",
            "auto_update_enabled",
            "deployment_mode",
            "self_update_supported",
            "update_hint",
            "staged_version",
            "restart_required",
            "restart_supported",
            "installing",
            "install_error",
            "last_error",
            "check_enabled",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["deployment_mode"], "source");
    }

    #[tokio::test]
    async fn same_or_older_release_is_not_an_update() {
        for v in ["0.0.15", "0.0.14"] {
            let s = svc(
                DeploymentMode::Standalone,
                true,
                FakeSource::new(vec![Ok(Some(release(v)))]),
                Arc::default(),
            );
            assert!(!s.check_now().await.update_available, "{v}");
        }
    }

    #[tokio::test]
    async fn auto_update_disabled_never_installs() {
        let installer = Arc::new(FakeInstaller::default());
        let s = svc(
            DeploymentMode::Standalone,
            false,
            FakeSource::new(vec![Ok(Some(release("0.0.16")))]),
            installer.clone(),
        );
        let st = s.check_now().await;
        s.check_now().await;
        assert!(st.update_available);
        assert!(!st.restart_required);
        assert_eq!(installer.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn auto_update_enabled_stages_once_on_standalone() {
        let installer = Arc::new(FakeInstaller::default());
        let s = svc(
            DeploymentMode::Standalone,
            true,
            FakeSource::new(vec![Ok(Some(release("0.0.16")))]),
            installer.clone(),
        );
        let st = s.check_now().await;
        assert_eq!(st.staged_version.as_deref(), Some("0.0.16"));
        assert!(st.restart_required);
        assert!(st.update_available); // still running the old binary
        s.check_now().await; // already staged → no second download
        assert_eq!(installer.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn auto_update_never_touches_non_standalone_binaries() {
        for mode in [
            DeploymentMode::Source,
            DeploymentMode::Desktop,
            DeploymentMode::Docker,
            DeploymentMode::PackageManager,
        ] {
            let installer = Arc::new(FakeInstaller::default());
            let s = svc(
                mode,
                true,
                FakeSource::new(vec![Ok(Some(release("0.0.16")))]),
                installer.clone(),
            );
            s.check_now().await;
            assert_eq!(installer.calls.load(Ordering::SeqCst), 0, "{mode:?}");
            assert!(matches!(
                s.install_latest().await,
                Err(InstallError::NotSupported(..))
            ));
        }
    }

    #[test]
    fn apply_auto_update_without_a_running_service_is_a_reported_noop() {
        // No `start_global` in unit tests: nothing to toggle, and the caller is told.
        assert!(global().is_none());
        assert!(!apply_auto_update(true));
    }

    #[tokio::test]
    async fn runtime_toggle_is_respected() {
        let installer = Arc::new(FakeInstaller::default());
        let s = svc(
            DeploymentMode::Standalone,
            true,
            FakeSource::new(vec![Ok(Some(release("0.0.16")))]),
            installer.clone(),
        );
        s.set_auto_update(false);
        s.check_now().await;
        assert_eq!(installer.calls.load(Ordering::SeqCst), 0);
        // Manual install is still allowed when auto-update is off.
        assert_eq!(s.install_latest().await.unwrap(), Version::new(0, 0, 16));
        assert_eq!(installer.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_install_is_reported_and_not_retried_automatically() {
        let installer = Arc::new(FakeInstaller {
            fail: true,
            ..Default::default()
        });
        let s = svc(
            DeploymentMode::Standalone,
            true,
            FakeSource::new(vec![Ok(Some(release("0.0.16")))]),
            installer.clone(),
        );
        let st = s.check_now().await;
        assert!(st.install_error.unwrap().contains("checksum"));
        assert!(!st.restart_required);
        s.check_now().await;
        assert_eq!(installer.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn offline_check_keeps_last_known_release() {
        let s = svc(
            DeploymentMode::Source,
            false,
            FakeSource::new(vec![
                Ok(Some(release("0.0.16"))),
                Err("Failed to reach GitHub API".into()),
            ]),
            Arc::default(),
        );
        s.check_now().await;
        let st = s.check_now().await;
        assert!(st.update_available);
        assert_eq!(st.latest.as_deref(), Some("0.0.16"));
        assert!(st.last_error.unwrap().contains("GitHub"));
    }

    #[tokio::test]
    async fn install_without_known_update_is_refused() {
        let s = svc(
            DeploymentMode::Standalone,
            false,
            FakeSource::new(vec![Ok(None)]),
            Arc::default(),
        );
        s.check_now().await;
        assert_eq!(s.install_latest().await, Err(InstallError::NoUpdate));
    }

    #[tokio::test(start_paused = true)]
    async fn periodic_checker_runs_on_schedule() {
        let source = FakeSource::new(vec![Ok(None)]);
        let s = Arc::new(svc(
            DeploymentMode::Source,
            false,
            source.clone(),
            Arc::default(),
        ));
        let handle = s.spawn_periodic(Schedule {
            initial_delay: Duration::from_secs(10),
            interval: Duration::from_secs(3600),
            jitter: Duration::ZERO,
        });
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert_eq!(
            source.calls.load(Ordering::SeqCst),
            0,
            "startup not blocked"
        );
        tokio::time::sleep(Duration::from_secs(6)).await;
        assert_eq!(source.calls.load(Ordering::SeqCst), 1);
        tokio::time::sleep(Duration::from_secs(3600)).await;
        assert_eq!(source.calls.load(Ordering::SeqCst), 2);
        handle.abort();
    }

    #[test]
    fn jitter_is_bounded() {
        assert_eq!(jitter(Duration::ZERO), Duration::ZERO);
        for _ in 0..100 {
            assert!(jitter(Duration::from_secs(60)) < Duration::from_secs(60));
        }
    }

    #[test]
    fn notes_excerpt_is_truncated_on_char_boundary() {
        let long = "é".repeat(1000);
        let e = excerpt(&long);
        assert_eq!(e.chars().count(), NOTES_EXCERPT_CHARS + 1);
        assert!(e.ends_with('…'));
    }
}
