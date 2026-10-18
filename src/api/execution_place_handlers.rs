//! « Exécute dans X » over HTTP (T5): a delegated task that writes in an ephemeral worktree.
//!
//! - `POST /api/plans/{plan_id}/tasks/{task_id}/delegate` with `execution_place:
//!   "ephemeral_worktree"` opens nothing: PO resolves X (`runner::ephemeral_worktree::plan`),
//!   keeps the placement pending (`chat::place_confirmation`), writes `requested` to the audit and
//!   answers 202 `confirmation_required`. A caller nobody is in front of is refused (403
//!   `execution_place_no_person`), and nothing is kept.
//! - `POST /api/execution-places/{id}/confirm` `{ "path": X }`: a PERSON (never an agent token),
//!   the owner of the placement, naming X exactly. `confirmed` is written to the audit BEFORE
//!   anything is made (no audit, no worktree); then the worktree is made from the frozen base,
//!   the session opens in it (`launched`), and when the task is over the session is closed and
//!   the worktree released (`removed`, or `kept` with the reason). A failure after the
//!   confirmation releases the worktree at once (`launch_failed`, then `removed` / `kept`).
//! - `POST /api/execution-places/{id}/reject`: the person says no (`rejected`).
//! - `GET /api/execution-places/{id}`: a pending placement.
//! - `GET /api/tasks/{task_id}/execution-places`: the audit of a task, oldest first.

use std::sync::LazyLock;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::handlers::{
    launch_delegation, AppError, DelegateTaskOutcome, OrchestratorState, PreparedDelegation,
};
use crate::auth::jwt::Claims;
use crate::chat::envelope::{self, ParentEnvelope, SpawnCaller};
use crate::chat::place_confirmation::{
    person_in_front, Pending, PlacementBook, PlacementError, Requested,
};
use crate::neo4j::models::{ExecutionPlaceEvent, ExecutionPlaceEventKind};
use crate::neo4j::GraphStore;
use crate::runner::ephemeral_worktree::{self, EphemeralWorktree, PlannedWorktree, ReleaseOutcome};

/// The value of `execution_place` that asks for an ephemeral worktree.
pub const EPHEMERAL_WORKTREE: &str = "ephemeral_worktree";

/// Where a delegation asked to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegationPlace {
    /// In the request's `cwd`, as always.
    AsGiven,
    /// In an ephemeral worktree a person confirms first.
    EphemeralWorktree,
}

impl DelegationPlace {
    pub fn parse(raw: Option<&str>) -> Result<Self, AppError> {
        match raw.map(str::trim).filter(|s| !s.is_empty()) {
            None | Some("project") => Ok(Self::AsGiven),
            Some(EPHEMERAL_WORKTREE) => Ok(Self::EphemeralWorktree),
            Some(other) => Err(AppError::BadRequest(format!(
                "execution_place_unknown: {other:?} (expected \"project\" or \"{EPHEMERAL_WORKTREE}\")"
            ))),
        }
    }
}

/// The pending placements of the server: what a confirmation launches.
static PLACEMENTS: LazyLock<PlacementBook<PreparedDelegation>> = LazyLock::new(PlacementBook::new);

/// Where the ephemeral worktrees go. Tests never write in the person's application directory.
fn place_root() -> std::path::PathBuf {
    #[cfg(test)]
    {
        std::env::temp_dir().join("po-t5-test-task-worktrees")
    }
    #[cfg(not(test))]
    {
        ephemeral_worktree::root()
    }
}

/// A place as the person is shown it, and as the API answers it.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionPlaceView {
    pub placement_id: Uuid,
    pub kind: &'static str,
    /// X: the path the task writes in. The confirmation must name it exactly.
    pub path: String,
    pub repository: String,
    pub base_ref: String,
    pub base_sha: String,
    pub branch: String,
    /// The person who must confirm (`sub`).
    pub owner: String,
    /// `session:<uuid>` or `person:<sub>`.
    pub requested_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
}

impl ExecutionPlaceView {
    fn of(meta: &PlacementMeta, expires_at: Option<DateTime<Utc>>) -> Self {
        Self {
            placement_id: meta.placement_id,
            kind: EPHEMERAL_WORKTREE,
            path: meta.place.path.to_string_lossy().to_string(),
            repository: meta.place.repository.to_string_lossy().to_string(),
            base_ref: meta.place.base_ref.clone(),
            base_sha: meta.place.base_sha.clone(),
            branch: meta.place.branch.clone(),
            owner: meta.owner.clone(),
            requested_by: meta.requested_by.clone(),
            expires_at,
        }
    }
}

/// The answer to a delegation that waits for its person.
#[derive(Debug, Clone, Serialize)]
pub struct ConfirmationRequired {
    pub status: &'static str,
    pub task_id: Uuid,
    pub plan_id: Uuid,
    pub execution_place: ExecutionPlaceView,
    /// What the person is asked, and how the answer is given.
    pub message: String,
}

/// What every audit entry of one placement shares.
#[derive(Debug, Clone)]
pub(crate) struct PlacementMeta {
    pub placement_id: Uuid,
    pub task_id: Uuid,
    pub plan_id: Uuid,
    pub owner: String,
    pub requested_by: String,
    pub place: PlannedWorktree,
}

impl PlacementMeta {
    fn of_pending<T>(p: &Pending<T>) -> Self {
        Self {
            placement_id: p.id,
            task_id: p.task_id,
            plan_id: p.plan_id,
            owner: p.owner.clone(),
            requested_by: p.requester.audit_label(),
            place: p.place.clone(),
        }
    }

    fn event(
        &self,
        kind: ExecutionPlaceEventKind,
        actor: Option<&str>,
        session_id: Option<&str>,
        detail: Option<String>,
    ) -> ExecutionPlaceEvent {
        ExecutionPlaceEvent {
            id: Uuid::new_v4(),
            placement_id: self.placement_id,
            kind,
            task_id: self.task_id,
            plan_id: self.plan_id,
            requested_by: self.requested_by.clone(),
            owner: self.owner.clone(),
            actor: actor.map(str::to_string),
            session_id: session_id.and_then(|s| Uuid::parse_str(s).ok()),
            path: self.place.path.to_string_lossy().to_string(),
            repository: self.place.repository.to_string_lossy().to_string(),
            base_ref: self.place.base_ref.clone(),
            base_sha: self.place.base_sha.clone(),
            branch: self.place.branch.clone(),
            detail,
            at: Utc::now(),
        }
    }

    /// Write an entry; the caller decides whether a failure stops it.
    async fn record(
        &self,
        graph: &dyn GraphStore,
        kind: ExecutionPlaceEventKind,
        actor: Option<&str>,
        session_id: Option<&str>,
        detail: Option<String>,
    ) -> anyhow::Result<()> {
        graph
            .create_execution_place_event(&self.event(kind, actor, session_id, detail))
            .await
    }

    /// Write an entry nothing waits on: a failure is logged loudly, never silent.
    async fn record_or_log(
        &self,
        graph: &dyn GraphStore,
        kind: ExecutionPlaceEventKind,
        session_id: Option<&str>,
        detail: Option<String>,
    ) {
        if let Err(e) = self.record(graph, kind, None, session_id, detail).await {
            tracing::error!(
                placement_id = %self.placement_id,
                task_id = %self.task_id,
                kind = kind.as_str(),
                "execution-place audit entry could not be written: {e:#}"
            );
        }
    }
}

/// An ephemeral worktree a confirmed delegation runs in: made, and released exactly once by
/// [`Self::launch_failed`] or [`Self::finish`] (both consume it).
pub(crate) struct LaunchedPlace {
    worktree: EphemeralWorktree,
    meta: PlacementMeta,
}

impl LaunchedPlace {
    /// The session opened in X.
    pub(crate) async fn launched(
        &self,
        graph: &dyn GraphStore,
        session_id: &str,
    ) -> ExecutionPlaceView {
        self.meta
            .record_or_log(
                graph,
                ExecutionPlaceEventKind::Launched,
                Some(session_id),
                None,
            )
            .await;
        ExecutionPlaceView::of(&self.meta, None)
    }

    /// The session could not open: the worktree is released at once.
    pub(crate) async fn launch_failed(self, graph: &dyn GraphStore, error: &AppError) {
        self.meta
            .record_or_log(
                graph,
                ExecutionPlaceEventKind::LaunchFailed,
                None,
                Some(format!("{error:?}")),
            )
            .await;
        self.release(graph, None).await;
    }

    /// The task is over.
    pub(crate) async fn finish(self, graph: &dyn GraphStore, session_id: &str) {
        self.release(graph, Some(session_id)).await;
    }

    async fn release(self, graph: &dyn GraphStore, session_id: Option<&str>) {
        let outcome = self.worktree.release().await;
        let (kind, detail) = match &outcome {
            ReleaseOutcome::Removed => (ExecutionPlaceEventKind::Removed, None),
            ReleaseOutcome::Kept { .. } => (
                ExecutionPlaceEventKind::Kept,
                serde_json::to_string(&outcome).ok(),
            ),
        };
        if kind == ExecutionPlaceEventKind::Kept {
            tracing::warn!(
                path = %self.meta.place.path.display(),
                branch = %self.meta.place.branch,
                detail = detail.as_deref().unwrap_or(""),
                "ephemeral worktree kept: removing it could lose work"
            );
        }
        self.meta
            .record_or_log(graph, kind, session_id, detail)
            .await;
    }
}

/// The session that asked, read from the graph (its owner and lineage decide who answers).
async fn asking_session(
    graph: &dyn GraphStore,
    parent: Option<&ParentEnvelope>,
) -> Result<Option<crate::neo4j::models::ChatSessionNode>, AppError> {
    match parent {
        None => Ok(None),
        Some(env) => graph
            .get_chat_session(env.parent_session_id)
            .await
            .map_err(AppError::Internal)?
            .map(Some)
            .ok_or_else(|| {
                PlacementError::NoPersonInFront("the asking session is not recorded")
                    .into_app_error()
            }),
    }
}

/// Keep a delegation pending until its person confirms (see the module).
pub(crate) async fn request_placement(
    state: &OrchestratorState,
    claims: Option<&Claims>,
    parent: Option<&ParentEnvelope>,
    prepared: PreparedDelegation,
) -> Result<ConfirmationRequired, AppError> {
    let graph = state.orchestrator.neo4j_arc();
    // Nobody in front: refused now, no question kept, nothing planned.
    let session = asking_session(graph.as_ref(), parent).await?;
    let (owner, requester) =
        person_in_front(claims, session.as_ref()).map_err(PlacementError::into_app_error)?;
    // X is PO's: resolved from the repository of the (envelope-checked) cwd.
    let place =
        ephemeral_worktree::plan(&prepared.chat_request.cwd, prepared.task_id, &place_root())
            .await
            .map_err(|e| match e {
                ephemeral_worktree::PlaceError::Git(_) => {
                    AppError::Internal(anyhow::anyhow!("{e}"))
                }
                _ => AppError::BadRequest(e.to_string()),
            })?;
    let (task_id, plan_id) = (prepared.task_id, prepared.plan_id);
    let requested = PLACEMENTS
        .request(
            task_id,
            plan_id,
            owner,
            requester,
            place,
            prepared,
            Utc::now(),
        )
        .map_err(PlacementError::into_app_error)?;
    let pending = PLACEMENTS
        .get(requested.id(), Utc::now())
        .ok_or_else(|| PlacementError::NotFound(requested.id()).into_app_error())?;
    let meta = PlacementMeta::of_pending(&pending);
    if let Requested::New(id) = requested {
        // No audit, no question: a placement nobody can trace is not kept.
        if let Err(e) = meta
            .record(
                graph.as_ref(),
                ExecutionPlaceEventKind::Requested,
                None,
                None,
                None,
            )
            .await
        {
            PLACEMENTS.discard(id);
            return Err(AppError::Internal(e.context("execution-place audit")));
        }
        emit_question(state, &meta);
    }
    let view = ExecutionPlaceView::of(&meta, Some(pending.expires_at));
    Ok(ConfirmationRequired {
        status: "confirmation_required",
        task_id,
        plan_id,
        message: format!(
            "Nothing runs yet. The person in front of this conversation must confirm « exécute dans {} » \
             (POST /api/execution-places/{}/confirm with {{\"path\": \"{}\"}}). An agent cannot confirm.",
            view.path, view.placement_id, view.path
        ),
        execution_place: view,
    })
}

/// Tell the clients a person has a question waiting (a UI shows « exécute dans X »).
fn emit_question(state: &OrchestratorState, meta: &PlacementMeta) {
    use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};
    state.event_bus.emit(CrudEvent {
        entity_type: EntityType::Task,
        action: CrudAction::Updated,
        entity_id: meta.task_id.to_string(),
        related: None,
        payload: serde_json::json!({
            "event": "execution_place_confirmation_required",
            "placement_id": meta.placement_id,
            "task_id": meta.task_id,
            "plan_id": meta.plan_id,
            "path": meta.place.path.to_string_lossy(),
            "owner": meta.owner,
            "requested_by": meta.requested_by,
        }),
        timestamp: Utc::now().to_rfc3339(),
        project_id: None,
    });
}

/// The caller's own token, refused when it is an agent's (whatever person it acts for).
fn answering_person<'a>(
    state: &OrchestratorState,
    claims: Option<&'a Claims>,
    headers: &axum::http::HeaderMap,
) -> Result<Option<&'a Claims>, AppError> {
    match envelope::identify_caller(
        claims,
        envelope::session_header(headers),
        state.auth_config.is_some(),
    ) {
        Ok(SpawnCaller::Human) => Ok(claims),
        _ => Err(PlacementError::NotAPerson.into_app_error()),
    }
}

/// Body of a confirmation: the place the person was shown.
#[derive(Debug, Deserialize)]
pub struct ConfirmPlaceRequest {
    pub path: String,
}

/// POST /api/execution-places/{id}/confirm — the person confirms « exécute dans X ».
pub async fn confirm_execution_place(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<Claims>>,
    headers: axum::http::HeaderMap,
    Path(placement_id): Path<Uuid>,
    Json(body): Json<ConfirmPlaceRequest>,
) -> Result<(StatusCode, Json<DelegateTaskOutcome>), AppError> {
    let chat_manager = state
        .chat_manager
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;
    let claims = claims.as_ref().map(|c| &c.0);
    let person = answering_person(&state, claims, &headers)?;
    let pending = PLACEMENTS
        .confirm(placement_id, person, body.path.trim(), Utc::now())
        .map_err(PlacementError::into_app_error)?;
    let actor = person.map(|c| c.sub.clone()).unwrap_or_default();
    let meta = PlacementMeta::of_pending(&pending);
    let graph = state.orchestrator.neo4j_arc();
    // The audit comes first: no trace of who confirmed what, no worktree.
    meta.record(
        graph.as_ref(),
        ExecutionPlaceEventKind::Confirmed,
        Some(&actor),
        None,
        None,
    )
    .await
    .map_err(|e| AppError::Internal(e.context("execution-place audit")))?;
    let worktree = match ephemeral_worktree::create(&meta.place).await {
        Ok(wt) => wt,
        Err(e) => {
            meta.record_or_log(
                graph.as_ref(),
                ExecutionPlaceEventKind::LaunchFailed,
                None,
                Some(e.to_string()),
            )
            .await;
            return Err(AppError::Conflict(e.to_string()));
        }
    };
    let mut prepared = pending.launch;
    prepared.chat_request.cwd = worktree.cwd();
    let place = LaunchedPlace { worktree, meta };
    let launched = launch_delegation(&state, &chat_manager, prepared, Some(place)).await?;
    Ok((
        StatusCode::ACCEPTED,
        Json(DelegateTaskOutcome::Launched(launched)),
    ))
}

/// POST /api/execution-places/{id}/reject — the person says no.
pub async fn reject_execution_place(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<Claims>>,
    headers: axum::http::HeaderMap,
    Path(placement_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let claims = claims.as_ref().map(|c| &c.0);
    let person = answering_person(&state, claims, &headers)?;
    let pending = PLACEMENTS
        .reject(placement_id, person, Utc::now())
        .map_err(PlacementError::into_app_error)?;
    let meta = PlacementMeta::of_pending(&pending);
    let actor = person.map(|c| c.sub.as_str());
    let graph = state.orchestrator.neo4j_arc();
    if let Err(e) = meta
        .record(
            graph.as_ref(),
            ExecutionPlaceEventKind::Rejected,
            actor,
            None,
            None,
        )
        .await
    {
        tracing::error!(placement_id = %placement_id, "execution-place audit entry could not be written: {e:#}");
    }
    Ok(Json(serde_json::json!({
        "status": "rejected",
        "placement_id": placement_id,
    })))
}

/// GET /api/execution-places/{id} — a pending placement.
pub async fn get_execution_place(
    Path(placement_id): Path<Uuid>,
) -> Result<Json<ExecutionPlaceView>, AppError> {
    let pending = PLACEMENTS
        .get(placement_id, Utc::now())
        .ok_or_else(|| PlacementError::NotFound(placement_id).into_app_error())?;
    Ok(Json(ExecutionPlaceView::of(
        &PlacementMeta::of_pending(&pending),
        Some(pending.expires_at),
    )))
}

/// GET /api/tasks/{task_id}/execution-places — the audit of a task, oldest first.
pub async fn list_execution_place_events(
    State(state): State<OrchestratorState>,
    Path(task_id): Path<Uuid>,
) -> Result<Json<Vec<ExecutionPlaceEvent>>, AppError> {
    let events = state
        .orchestrator
        .neo4j_arc()
        .list_execution_place_events(task_id)
        .await
        .map_err(AppError::Internal)?;
    Ok(Json(events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::{delegate_task, DelegateTaskRequest};
    use std::sync::Arc;

    const OWNER: &str = "22222222-2222-2222-2222-222222222222";

    fn git(cwd: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(["-c", "user.email=t@t", "-c", "user.name=t"])
            .args(args)
            .current_dir(cwd)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}");
    }

    struct Fixture {
        state: OrchestratorState,
        mock: Arc<crate::neo4j::mock::MockGraphStore>,
        plan_id: Uuid,
        task_id: Uuid,
        parent: crate::neo4j::models::ChatSessionNode,
        _repo: tempfile::TempDir,
    }

    /// A temporary repository (never the real one), a plan, a task, and the session that asks,
    /// working in that repository.
    async fn fixture(owner: Option<&str>, spawned_by: Option<&str>) -> Fixture {
        let repo = tempfile::tempdir().unwrap();
        git(repo.path(), &["init", "-q", "-b", "main"]);
        git(
            repo.path(),
            &["commit", "-q", "--allow-empty", "-m", "init"],
        );
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let project = crate::test_helpers::test_project();
        mock.create_project(&project).await.unwrap();
        let plan = crate::test_helpers::test_plan_for_project(project.id);
        mock.create_plan(&plan).await.unwrap();
        let task = crate::test_helpers::test_task();
        mock.create_task(plan.id, &task).await.unwrap();
        let mut parent = crate::test_helpers::test_chat_session(Some(&project.slug));
        parent.cwd = std::fs::canonicalize(repo.path())
            .unwrap()
            .to_string_lossy()
            .to_string();
        parent.owner = owner.map(str::to_string);
        parent.spawned_by = spawned_by.map(str::to_string);
        mock.create_chat_session(&parent).await.unwrap();
        let dyn_graph: Arc<dyn GraphStore> = mock.clone();
        let manager = Arc::new(crate::chat::manager::ChatManager::new_without_memory(
            dyn_graph,
            crate::test_helpers::mock_app_state().meili,
            Default::default(),
        ));
        let state = crate::test_helpers::server_state_with_chat(manager, mock.clone()).await;
        Fixture {
            state,
            mock,
            plan_id: plan.id,
            task_id: task.id,
            parent,
            _repo: repo,
        }
    }

    fn as_session(parent: Uuid) -> axum::http::HeaderMap {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            envelope::SESSION_ID_HEADER,
            parent.to_string().parse().unwrap(),
        );
        headers
    }

    fn request(cwd: &str) -> DelegateTaskRequest {
        DelegateTaskRequest {
            cwd: cwd.to_string(),
            project_slug: None,
            custom_sections: Vec::new(),
            parent_session_id: None,
            // An engine nobody registered: the session never opens in a test.
            provider: Some("nobody-registered".to_string()),
            model: None,
            task_class: None,
            execution_place: Some(EPHEMERAL_WORKTREE.to_string()),
        }
    }

    async fn ask(f: &Fixture) -> Result<ConfirmationRequired, AppError> {
        match delegate_task(
            State(f.state.clone()),
            None,
            as_session(f.parent.id),
            Path((f.plan_id, f.task_id)),
            Json(request(&f.parent.cwd)),
        )
        .await
        {
            Ok((_, Json(DelegateTaskOutcome::ConfirmationRequired(c)))) => Ok(c),
            Ok((_, Json(DelegateTaskOutcome::Launched(l)))) => {
                panic!("an ephemeral worktree must wait for its person, launched {l:?}")
            }
            Err(e) => Err(e),
        }
    }

    async fn confirm(
        f: &Fixture,
        claims: Claims,
        placement: Uuid,
        path: &str,
    ) -> Result<(StatusCode, Json<DelegateTaskOutcome>), AppError> {
        confirm_execution_place(
            State(f.state.clone()),
            Some(axum::Extension(claims)),
            axum::http::HeaderMap::new(),
            Path(placement),
            Json(ConfirmPlaceRequest {
                path: path.to_string(),
            }),
        )
        .await
    }

    async fn kinds(f: &Fixture) -> Vec<ExecutionPlaceEventKind> {
        f.mock
            .list_execution_place_events(f.task_id)
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.kind)
            .collect()
    }

    #[tokio::test]
    async fn a_session_nobody_is_in_front_of_is_refused_without_a_question() {
        let f = fixture(Some(OWNER), Some(r#"{"type":"delegation"}"#)).await;
        match ask(&f).await {
            Err(AppError::Forbidden(m)) => {
                assert!(m.starts_with("execution_place_no_person"), "{m}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(kinds(&f).await.is_empty(), "no question may be kept");
    }

    #[tokio::test]
    async fn the_place_waits_for_its_person_and_names_x_resolved_by_po() {
        let f = fixture(Some(OWNER), None).await;
        let pending = ask(&f).await.unwrap();
        let place = &pending.execution_place;
        assert_eq!(pending.status, "confirmation_required");
        assert_eq!(place.owner, OWNER);
        assert_eq!(place.requested_by, format!("session:{}", f.parent.id));
        let root = std::fs::canonicalize(place_root()).unwrap();
        assert!(
            place.path.starts_with(&*root.to_string_lossy()),
            "{}",
            place.path
        );
        assert!(
            !std::path::Path::new(&place.path).exists(),
            "nothing made yet"
        );
        assert_eq!(place.repository, f.parent.cwd);
        assert_eq!(kinds(&f).await, vec![ExecutionPlaceEventKind::Requested]);
        // Asking again is the same question (idempotent), not a second one.
        let again = ask(&f).await.unwrap();
        assert_eq!(
            again.execution_place.placement_id,
            pending.execution_place.placement_id
        );
        assert_eq!(kinds(&f).await.len(), 1);
    }

    #[tokio::test]
    async fn an_agent_token_cannot_confirm_for_its_person() {
        let f = fixture(Some(OWNER), None).await;
        let pending = ask(&f).await.unwrap();
        let id = pending.execution_place.placement_id;
        let mut agent = Claims::person(OWNER, "p@example.org", "P");
        agent.token_type = Some(crate::auth::jwt::TOKEN_TYPE_AGENT_SESSION.to_string());
        match confirm(&f, agent, id, &pending.execution_place.path).await {
            Err(AppError::Forbidden(m)) => {
                assert!(m.starts_with("execution_place_not_a_person"), "{m}")
            }
            other => panic!("an agent token must not confirm, got {other:?}"),
        }
        // Another person cannot either.
        match confirm(
            &f,
            Claims::person("someone-else", "o@example.org", "O"),
            id,
            &pending.execution_place.path,
        )
        .await
        {
            Err(AppError::Forbidden(m)) => {
                assert!(m.starts_with("execution_place_not_the_owner"), "{m}")
            }
            other => panic!("another person must not confirm, got {other:?}"),
        }
        assert_eq!(kinds(&f).await, vec![ExecutionPlaceEventKind::Requested]);
        assert!(!std::path::Path::new(&pending.execution_place.path).exists());
        assert!(PLACEMENTS.get(id, Utc::now()).is_some(), "still pending");
        PLACEMENTS.discard(id);
    }

    /// The person confirms; the session then cannot open (no such engine here): the worktree
    /// that was made for it is released at once, and the audit says who, what X, and what
    /// became of it.
    #[tokio::test]
    async fn a_confirmed_place_is_audited_made_and_released_when_the_launch_fails() {
        let f = fixture(Some(OWNER), None).await;
        let pending = ask(&f).await.unwrap();
        let x = pending.execution_place.path.clone();
        let id = pending.execution_place.placement_id;
        // A confirmation of another path is refused and keeps the question.
        assert!(matches!(
            confirm(
                &f,
                Claims::person(OWNER, "p@example.org", "P"),
                id,
                "/elsewhere"
            )
            .await,
            Err(AppError::Conflict(_))
        ));
        let result = confirm(&f, Claims::person(OWNER, "p@example.org", "P"), id, &x).await;
        assert!(result.is_err(), "no engine is registered in this test");
        assert!(
            !std::path::Path::new(&x).exists(),
            "the worktree must be removed when the launch fails"
        );
        let events = f.mock.list_execution_place_events(f.task_id).await.unwrap();
        let kinds: Vec<_> = events.iter().map(|e| e.kind).collect();
        assert_eq!(
            kinds,
            vec![
                ExecutionPlaceEventKind::Requested,
                ExecutionPlaceEventKind::Confirmed,
                ExecutionPlaceEventKind::LaunchFailed,
                ExecutionPlaceEventKind::Removed,
            ]
        );
        let confirmed = &events[1];
        assert_eq!(confirmed.actor.as_deref(), Some(OWNER));
        assert_eq!(confirmed.path, x);
        assert_eq!(confirmed.placement_id, id);
        assert_eq!(confirmed.requested_by, format!("session:{}", f.parent.id));
        // One confirmation, one launch.
        assert!(matches!(
            confirm(&f, Claims::person(OWNER, "p@example.org", "P"), id, &x).await,
            Err(AppError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn the_person_may_say_no() {
        let f = fixture(Some(OWNER), None).await;
        let pending = ask(&f).await.unwrap();
        let id = pending.execution_place.placement_id;
        let _ = reject_execution_place(
            State(f.state.clone()),
            Some(axum::Extension(Claims::person(OWNER, "p@example.org", "P"))),
            axum::http::HeaderMap::new(),
            Path(id),
        )
        .await
        .unwrap();
        assert_eq!(
            kinds(&f).await,
            vec![
                ExecutionPlaceEventKind::Requested,
                ExecutionPlaceEventKind::Rejected
            ]
        );
        assert!(PLACEMENTS.get(id, Utc::now()).is_none());
    }

    #[test]
    fn an_unknown_place_is_a_bad_request() {
        assert!(matches!(
            DelegationPlace::parse(Some("/tmp/anywhere")),
            Err(AppError::BadRequest(_))
        ));
        assert_eq!(
            DelegationPlace::parse(None).unwrap(),
            DelegationPlace::AsGiven
        );
        assert_eq!(
            DelegationPlace::parse(Some("project")).unwrap(),
            DelegationPlace::AsGiven
        );
    }
}
