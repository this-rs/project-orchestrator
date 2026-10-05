//! Update endpoints: check now, install, restart.
//!
//! The status itself is read from `GET /api/version` (`update` field, public).
//! These endpoints *act*, so they need a person: installing swaps the server
//! binary and restarting kills every running chat session. An agent holding a
//! session, vault or MCP token must not be able to do either.
//!
//! - `POST /api/update/check`   — query GitHub now, return the status.
//! - `POST /api/update/install` — download + verify + stage the latest release
//!   (standalone deployments only). Returns `202` with the status: the download
//!   keeps running in the background and the caller polls `GET /api/version`
//!   (`installing` → `staged_version` / `install_error`).
//! - `POST /api/update/restart` — exit with the supervisor restart code so the
//!   staged binary starts. Refused unless an update is staged AND a supervisor
//!   will bring the server back.

use axum::{extract::State, http::StatusCode, Extension, Json};
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;

use super::handlers::{AppError, OrchestratorState};
use crate::auth::jwt::Claims;
use crate::update::service::{schedule_restart, InstallError};
use crate::update::{UpdateService, UpdateStatus};

/// Lets the HTTP response flush before the process exits.
const RESTART_DELAY: Duration = Duration::from_millis(750);

/// How long `install` waits for a fast install (cached/small) before answering
/// `202` and leaving the work to the background task.
const INSTALL_GRACE: Duration = Duration::from_millis(300);

fn require_human(state: &OrchestratorState, claims: &Claims) -> Result<(), AppError> {
    guard(state.auth_config.is_some(), claims)
}

/// Without auth there are no people to tell apart: the server is trusted as a
/// whole, as everywhere else in that mode.
fn guard(auth_enabled: bool, claims: &Claims) -> Result<(), AppError> {
    if !auth_enabled || claims.is_human() {
        Ok(())
    } else {
        Err(AppError::Forbidden(
            "updating requires a user session, not an agent token".to_string(),
        ))
    }
}

fn service() -> Result<Arc<UpdateService>, AppError> {
    crate::update::global().cloned().ok_or_else(|| {
        AppError::NotImplemented(
            "the update service is not running (disabled with PO_UPDATE_CHECK, or failed to start)"
                .to_string(),
        )
    })
}

#[derive(Debug, Serialize)]
pub struct RestartResponse {
    pub restarting: bool,
    pub in_ms: u64,
}

// ----------------------------------------------------------------------------
// Logic (takes the service, so tests do not need the process-wide singleton)
// ----------------------------------------------------------------------------

pub(crate) async fn check_logic(svc: &UpdateService) -> Result<UpdateStatus, AppError> {
    if !svc.status().check_enabled {
        return Err(AppError::Conflict(
            "release checks are disabled (PO_UPDATE_CHECK)".to_string(),
        ));
    }
    Ok(svc.check_now().await)
}

/// Validate, then run the install in the background. Returns the status as it
/// is after at most [`INSTALL_GRACE`].
pub(crate) async fn install_logic(
    svc: &Arc<UpdateService>,
    grace: Duration,
) -> Result<UpdateStatus, AppError> {
    let status = svc.status();
    if !status.self_update_supported {
        return Err(AppError::Conflict(format!(
            "this deployment ({:?}) cannot update itself: {}",
            status.deployment_mode, status.update_hint
        )));
    }
    if status.installing {
        return Err(AppError::Conflict(
            "an update is already being installed".to_string(),
        ));
    }
    if !status.update_available {
        return Err(AppError::Conflict("no newer release is known".to_string()));
    }
    if status.staged_version.is_some() && status.staged_version == status.latest {
        return Ok(status); // already on disk, waiting for a restart
    }

    let task = {
        let svc = svc.clone();
        tokio::spawn(async move { svc.install_latest().await })
    };
    tokio::pin!(task);
    match tokio::time::timeout(grace, &mut task).await {
        // Finished within the grace period: report the outcome as the status.
        Ok(Ok(Err(InstallError::AlreadyRunning))) => {
            return Err(AppError::Conflict(
                "an update is already being installed".to_string(),
            ))
        }
        Ok(Ok(Err(InstallError::NoUpdate))) => {
            return Err(AppError::Conflict("no newer release is known".to_string()))
        }
        Ok(Ok(Err(InstallError::NotSupported(mode, hint)))) => {
            return Err(AppError::Conflict(format!(
                "this deployment ({mode:?}) cannot update itself: {hint}"
            )))
        }
        // `Failed` is recorded in the status (`install_error`); the caller reads it there.
        Ok(Ok(Ok(_) | Err(InstallError::Failed(_)))) => {}
        Ok(Err(join_error)) => {
            return Err(AppError::Internal(anyhow::anyhow!(
                "install task failed: {join_error}"
            )))
        }
        // Still running: it goes on in the background; `installing` is already set.
        Err(_) => {}
    }
    Ok(svc.status())
}

pub(crate) fn restart_logic(status: &UpdateStatus) -> Result<(), AppError> {
    if !status.restart_required {
        return Err(AppError::Conflict(
            "no update is staged; nothing to restart for".to_string(),
        ));
    }
    if !status.restart_supported {
        return Err(AppError::Conflict(
            "no supervisor would restart the server; restart it manually to apply the update"
                .to_string(),
        ));
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Handlers
// ----------------------------------------------------------------------------

pub async fn check_update(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
) -> Result<Json<UpdateStatus>, AppError> {
    require_human(&state, &claims)?;
    let svc = service()?;
    Ok(Json(check_logic(&svc).await?))
}

pub async fn install_update(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
) -> Result<(StatusCode, Json<UpdateStatus>), AppError> {
    require_human(&state, &claims)?;
    let svc = service()?;
    let status = install_logic(&svc, INSTALL_GRACE).await?;
    Ok((StatusCode::ACCEPTED, Json(status)))
}

pub async fn restart_update(
    State(state): State<OrchestratorState>,
    Extension(claims): Extension<Claims>,
) -> Result<(StatusCode, Json<RestartResponse>), AppError> {
    require_human(&state, &claims)?;
    let svc = service()?;
    restart_logic(&svc.status())?;
    tracing::warn!("Restart requested through the API to apply a staged update");
    schedule_restart(RESTART_DELAY);
    Ok((
        StatusCode::ACCEPTED,
        Json(RestartResponse {
            restarting: true,
            in_ms: RESTART_DELAY.as_millis() as u64,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::deployment::{DeploymentMode, Supervisor};
    use crate::update::service::UpdateServiceConfig;
    use crate::update::version::resolve_build_version;
    use crate::update::{Installer, ReleaseInfo, ReleaseSource};
    use anyhow::{anyhow, Result};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Source(Option<ReleaseInfo>);

    #[async_trait]
    impl ReleaseSource for Source {
        async fn latest(&self) -> Result<Option<ReleaseInfo>> {
            Ok(self.0.clone())
        }
    }

    struct Inst {
        calls: AtomicUsize,
        delay: Duration,
        fail: bool,
    }

    #[async_trait]
    impl Installer for Inst {
        async fn install(&self, _r: &ReleaseInfo) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            if self.fail {
                Err(anyhow!("disk full"))
            } else {
                Ok(())
            }
        }
    }

    fn release(v: &str) -> ReleaseInfo {
        ReleaseInfo {
            tag: format!("v{v}"),
            version: semver::Version::parse(v).unwrap(),
            html_url: format!("https://example.invalid/{v}"),
            body: None,
            published_at: None,
            assets: vec![],
        }
    }

    fn service_with(
        mode: DeploymentMode,
        supervisor: Option<Supervisor>,
        check_enabled: bool,
        latest: Option<ReleaseInfo>,
        inst: Arc<Inst>,
    ) -> Arc<UpdateService> {
        Arc::new(UpdateService::new(
            UpdateServiceConfig {
                build: resolve_build_version("0.0.15", ""),
                mode,
                supervisor,
                auto_update: false,
                check_enabled,
            },
            Arc::new(Source(latest)),
            inst,
        ))
    }

    fn inst(delay_ms: u64, fail: bool) -> Arc<Inst> {
        Arc::new(Inst {
            calls: AtomicUsize::new(0),
            delay: Duration::from_millis(delay_ms),
            fail,
        })
    }

    fn conflict(e: AppError) -> String {
        match e {
            AppError::Conflict(m) => m,
            other => panic!("expected Conflict, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn check_reports_a_newer_release() {
        let svc = service_with(
            DeploymentMode::Standalone,
            None,
            true,
            Some(release("0.0.16")),
            inst(0, false),
        );
        let st = check_logic(&svc).await.unwrap();
        assert!(st.update_available);
        assert_eq!(st.latest.as_deref(), Some("0.0.16"));
    }

    #[tokio::test]
    async fn check_is_refused_when_checks_are_disabled() {
        let svc = service_with(
            DeploymentMode::Standalone,
            None,
            false,
            None,
            inst(0, false),
        );
        assert!(conflict(check_logic(&svc).await.unwrap_err()).contains("disabled"));
    }

    #[tokio::test]
    async fn install_is_refused_on_deployments_that_cannot_self_update() {
        for mode in [
            DeploymentMode::Docker,
            DeploymentMode::PackageManager,
            DeploymentMode::Source,
            DeploymentMode::Desktop,
        ] {
            let i = inst(0, false);
            let svc = service_with(mode, None, true, Some(release("0.0.16")), i.clone());
            svc.check_now().await;
            let msg = conflict(
                install_logic(&svc, Duration::from_millis(50))
                    .await
                    .unwrap_err(),
            );
            assert!(msg.contains("cannot update itself"), "{mode:?}: {msg}");
            assert_eq!(
                i.calls.load(Ordering::SeqCst),
                0,
                "{mode:?} must not install"
            );
        }
    }

    #[tokio::test]
    async fn install_is_refused_without_a_known_newer_release() {
        let svc = service_with(
            DeploymentMode::Standalone,
            None,
            true,
            Some(release("0.0.15")),
            inst(0, false),
        );
        svc.check_now().await;
        assert!(conflict(
            install_logic(&svc, Duration::from_millis(50))
                .await
                .unwrap_err()
        )
        .contains("no newer release"));
    }

    #[tokio::test]
    async fn fast_install_is_staged_before_the_answer() {
        let i = inst(0, false);
        let svc = service_with(
            DeploymentMode::Standalone,
            Some(Supervisor::Launchd),
            true,
            Some(release("0.0.16")),
            i.clone(),
        );
        svc.check_now().await;
        let st = install_logic(&svc, Duration::from_secs(2)).await.unwrap();
        assert_eq!(st.staged_version.as_deref(), Some("0.0.16"));
        assert!(st.restart_required && st.restart_supported);
        assert!(!st.installing);
        // A second request is a no-op, not a second download.
        let again = install_logic(&svc, Duration::from_secs(2)).await.unwrap();
        assert_eq!(again.staged_version.as_deref(), Some("0.0.16"));
        assert_eq!(i.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn slow_install_answers_while_still_installing_then_completes() {
        let svc = service_with(
            DeploymentMode::Standalone,
            None,
            true,
            Some(release("0.0.16")),
            inst(400, false),
        );
        svc.check_now().await;
        let st = install_logic(&svc, Duration::from_millis(50))
            .await
            .unwrap();
        assert!(
            st.installing,
            "the answer must say the install is under way"
        );
        assert!(st.staged_version.is_none());
        // A concurrent request is refused while it runs.
        assert!(conflict(
            install_logic(&svc, Duration::from_millis(10))
                .await
                .unwrap_err()
        )
        .contains("already being installed"));
        tokio::time::sleep(Duration::from_millis(700)).await;
        assert_eq!(svc.status().staged_version.as_deref(), Some("0.0.16"));
        assert!(!svc.status().installing);
    }

    #[tokio::test]
    async fn failed_install_is_visible_in_the_status() {
        let svc = service_with(
            DeploymentMode::Standalone,
            None,
            true,
            Some(release("0.0.16")),
            inst(0, true),
        );
        svc.check_now().await;
        let st = install_logic(&svc, Duration::from_secs(2)).await.unwrap();
        assert!(st.staged_version.is_none());
        assert!(st
            .install_error
            .as_deref()
            .unwrap_or("")
            .contains("disk full"));
        assert!(!st.installing);
    }

    fn status(restart_required: bool, restart_supported: bool) -> UpdateStatus {
        let svc = service_with(DeploymentMode::Standalone, None, true, None, inst(0, false));
        let mut st = svc.status();
        st.restart_required = restart_required;
        st.restart_supported = restart_supported;
        st
    }

    #[test]
    fn restart_needs_a_staged_update_and_a_supervisor() {
        assert!(restart_logic(&status(true, true)).is_ok());
        assert!(conflict(restart_logic(&status(false, true)).unwrap_err())
            .contains("no update is staged"));
        assert!(conflict(restart_logic(&status(true, false)).unwrap_err()).contains("supervisor"));
    }

    fn claims(token_type: Option<&str>) -> Claims {
        Claims {
            sub: uuid::Uuid::new_v4().to_string(),
            email: "a@b.c".into(),
            name: "n".into(),
            iat: 0,
            exp: i64::MAX,
            token_type: token_type.map(str::to_string),
            scope: None,
            jti: None,
        }
    }

    #[test]
    fn agents_are_refused_but_people_and_unauthenticated_servers_are_not() {
        assert!(guard(true, &claims(None)).is_ok());
        for t in ["agent_session", "mcp", "vault"] {
            assert!(
                matches!(guard(true, &claims(Some(t))), Err(AppError::Forbidden(_))),
                "{t} token must be refused"
            );
        }
        // No auth configured: trusted as a whole.
        assert!(guard(false, &claims(Some("agent_session"))).is_ok());
    }
}
