//! Team messaging over HTTP (TEAM-2): what the PO MCP `chat` tool calls for
//! `team_message` and `team_status`, from any engine and any depth.
//!
//! The SENDER is the calling session read from its signed token
//! (`envelope::identify_caller`), never a name in the body: a model cannot write as
//! another session, nor reach a session outside its own team (its parent and its
//! children, `chat::team::team_of`). A person has the chat itself: these routes answer
//! an agent only.
use crate::neo4j::traits::*;

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::handlers::{AppError, OrchestratorState};
use crate::chat::envelope::{self, SpawnCaller};
use crate::chat::team::{
    render_team_message, resolve_targets, team_of, Team, TeamMember, TeamMessageKind,
};

/// Body of `POST /api/chat/team/message`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeamMessageRequest {
    /// The message.
    pub text: String,
    /// `parent`, `children`, `all`, or a member's session id / id prefix / title prefix.
    /// Absent: the children of a session that has some, else its parent.
    #[serde(default)]
    pub to: Option<String>,
    /// `message` (default) or `dispatch_request` (always to the parent).
    #[serde(default)]
    pub kind: Option<String>,
    /// `dispatch_request`: the project the work belongs to.
    #[serde(default)]
    pub project_slug: Option<String>,
    /// `dispatch_request`: the task to open.
    #[serde(default)]
    pub task_id: Option<String>,
}

/// One target that could not be reached, and why.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TeamDeliveryFailure {
    pub session_id: String,
    pub error: String,
}

/// Answer of `POST /api/chat/team/message`.
#[derive(Debug, Clone, Serialize)]
pub struct TeamMessageResponse {
    /// Started a turn on, or queued in, each of these sessions.
    pub delivered: Vec<TeamMember>,
    /// Members that are not live (closed, finished) or refused the message.
    pub failed: Vec<TeamDeliveryFailure>,
}

/// Answer of `GET /api/chat/team`.
#[derive(Debug, Clone, Serialize)]
pub struct TeamStatusResponse {
    #[serde(rename = "self")]
    pub me: TeamMember,
    pub parent: Option<TeamMemberStatus>,
    pub children: Vec<TeamMemberStatus>,
}

/// A member and whether it is working right now.
#[derive(Debug, Clone, Serialize)]
pub struct TeamMemberStatus {
    #[serde(flatten)]
    pub member: TeamMember,
    /// A turn is running.
    pub is_streaming: bool,
}

/// The calling session, from its token. A person is refused: this is an agent's channel.
fn calling_session(
    state: &OrchestratorState,
    claims: Option<&crate::auth::jwt::Claims>,
    headers: &axum::http::HeaderMap,
) -> Result<Uuid, AppError> {
    match envelope::identify_caller(
        claims,
        envelope::session_header(headers),
        state.auth_config.is_some(),
    )? {
        SpawnCaller::Agent { session_id, .. } => Uuid::parse_str(&session_id).map_err(|_| {
            AppError::Forbidden(format!("team_unknown_session: {session_id}"))
        }),
        SpawnCaller::Human => Err(AppError::Forbidden(
            "team_agent_only: team messages are sent by a chat session (a person writes in the chat)"
                .into(),
        )),
    }
}

async fn load_team(
    state: &OrchestratorState,
    session_id: Uuid,
) -> Result<(TeamMember, Team), AppError> {
    let graph = state.orchestrator.neo4j_arc();
    team_of(graph.as_ref(), session_id)
        .await
        .map_err(AppError::Internal)?
        .ok_or_else(|| AppError::NotFound(format!("team_unknown_session: {session_id}")))
}

fn parse_kind(req: &TeamMessageRequest) -> Result<TeamMessageKind, AppError> {
    match req.kind.as_deref().map(str::trim) {
        None | Some("") | Some("message") => Ok(TeamMessageKind::Message),
        Some("dispatch_request") => Ok(TeamMessageKind::DispatchRequest {
            project_slug: req.project_slug.clone(),
            task_id: req.task_id.clone(),
        }),
        Some(other) => Err(AppError::BadRequest(format!(
            "team_kind_unknown: {other:?} (expected message or dispatch_request)"
        ))),
    }
}

/// POST /api/chat/team/message — write to a member of the caller's team.
pub async fn post_team_message(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<TeamMessageRequest>,
) -> Result<Json<TeamMessageResponse>, AppError> {
    let from = calling_session(&state, claims.as_ref().map(|c| &c.0), &headers)?;
    let text = req.text.trim();
    if text.is_empty() {
        return Err(AppError::BadRequest(
            "team_empty: the message is empty".into(),
        ));
    }
    let kind = parse_kind(&req)?;
    let (me, team) = load_team(&state, from).await?;
    let targets = resolve_targets(&team, req.to.as_deref(), &kind)
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    let chat_manager = state
        .chat_manager
        .as_ref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Chat manager not initialized")))?;

    let body = render_team_message(&kind, text);
    let mut delivered = Vec::new();
    let mut failed = Vec::new();
    for target in targets {
        match chat_manager
            .inject_team_message(from, &me.name, &target.session_id, &body)
            .await
        {
            Ok(()) => delivered.push(target),
            Err(e) => failed.push(TeamDeliveryFailure {
                session_id: target.session_id,
                error: e.to_string(),
            }),
        }
    }
    Ok(Json(TeamMessageResponse { delivered, failed }))
}

/// GET /api/chat/team — the caller, its parent and its children, and who is working.
pub async fn get_team(
    State(state): State<OrchestratorState>,
    claims: Option<axum::Extension<crate::auth::jwt::Claims>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<TeamStatusResponse>, AppError> {
    let session = calling_session(&state, claims.as_ref().map(|c| &c.0), &headers)?;
    let (me, team) = load_team(&state, session).await?;
    let chat_manager = state.chat_manager.as_ref();
    let status = |member: TeamMember| async {
        let is_streaming = match chat_manager {
            Some(m) => m.session_is_streaming(&member.session_id).await,
            None => false,
        };
        TeamMemberStatus {
            member,
            is_streaming,
        }
    };
    let parent = match team.parent {
        Some(p) => Some(status(p).await),
        None => None,
    };
    let mut children = Vec::with_capacity(team.children.len());
    for c in team.children {
        children.push(status(c).await);
    }
    Ok(Json(TeamStatusResponse {
        me,
        parent,
        children,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::*;

    fn req(kind: Option<&str>) -> TeamMessageRequest {
        TeamMessageRequest {
            text: "hi".into(),
            to: None,
            kind: kind.map(str::to_string),
            project_slug: Some("backend".into()),
            task_id: None,
        }
    }

    #[test]
    fn the_kind_is_message_by_default_and_unknown_kinds_are_refused() {
        assert_eq!(parse_kind(&req(None)).unwrap(), TeamMessageKind::Message);
        assert_eq!(
            parse_kind(&req(Some("dispatch_request"))).unwrap(),
            TeamMessageKind::DispatchRequest {
                project_slug: Some("backend".into()),
                task_id: None
            }
        );
        assert!(parse_kind(&req(Some("grant_me_trust"))).is_err());
    }

    /// A state with a parent session and its child in the graph (neither is live).
    async fn state_with_a_family() -> (
        OrchestratorState,
        crate::neo4j::models::ChatSessionNode,
        crate::neo4j::models::ChatSessionNode,
    ) {
        use crate::neo4j::GraphStore;
        use std::sync::Arc;

        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let mut parent = crate::test_helpers::test_chat_session(Some("nexus-sdk"));
        parent.title = Some("coordinator".into());
        mock.create_chat_session(&parent).await.unwrap();
        let mut child = crate::test_helpers::test_chat_session(Some("nexus-sdk"));
        child.spawned_by = Some(envelope::conversation_spawned_by(parent.id));
        mock.create_chat_session(&child).await.unwrap();

        let dyn_graph: Arc<dyn crate::neo4j::GraphStore> = mock.clone();
        let manager = Arc::new(crate::chat::manager::ChatManager::new_without_memory(
            dyn_graph,
            crate::test_helpers::mock_app_state().meili,
            crate::chat::config::ChatConfig::default(),
        ));
        let state = crate::test_helpers::server_state_with_chat(manager, mock).await;
        (state, parent, child)
    }

    fn as_session(id: Uuid) -> axum::http::HeaderMap {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(envelope::SESSION_ID_HEADER, id.to_string().parse().unwrap());
        headers
    }

    fn message(text: &str, kind: Option<&str>) -> TeamMessageRequest {
        TeamMessageRequest {
            text: text.into(),
            to: None,
            kind: kind.map(str::to_string),
            project_slug: None,
            task_id: None,
        }
    }

    /// The child writes UP: the message is routed to its parent, the only session it may
    /// reach by default. The parent is not live here, so the delivery fails — and says
    /// so, naming the parent: the route chose the right target and did not pretend.
    #[tokio::test]
    async fn a_child_writes_to_its_parent_and_an_undelivered_message_is_reported() {
        let (state, parent, child) = state_with_a_family().await;
        let Json(answer) = post_team_message(
            State(state),
            None,
            as_session(child.id),
            Json(message("the backend part is outside my project", None)),
        )
        .await
        .unwrap();
        assert!(answer.delivered.is_empty());
        assert_eq!(answer.failed.len(), 1);
        assert_eq!(answer.failed[0].session_id, parent.id.to_string());
    }

    #[tokio::test]
    async fn a_person_is_refused_and_a_root_cannot_send_a_dispatch_request_up() {
        let (state, parent, _child) = state_with_a_family().await;
        // No session header and no token: a person, refused.
        let err = post_team_message(
            State(state.clone()),
            None,
            axum::http::HeaderMap::new(),
            Json(message("hi", None)),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::Forbidden(ref m) if m.starts_with("team_agent_only")));

        // The root has no parent to dispatch for it.
        let err = post_team_message(
            State(state),
            None,
            as_session(parent.id),
            Json(message("open this", Some("dispatch_request"))),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::BadRequest(ref m) if m.starts_with("team_no_parent")));
    }

    #[tokio::test]
    async fn team_status_shows_each_side_of_the_family() {
        let (state, parent, child) = state_with_a_family().await;
        let Json(of_child) = get_team(State(state.clone()), None, as_session(child.id))
            .await
            .unwrap();
        assert_eq!(of_child.me.session_id, child.id.to_string());
        assert_eq!(
            of_child.parent.as_ref().map(|p| p.member.name.as_str()),
            Some("coordinator")
        );
        assert!(of_child.children.is_empty());

        let Json(of_parent) = get_team(State(state), None, as_session(parent.id))
            .await
            .unwrap();
        assert!(of_parent.parent.is_none());
        assert_eq!(of_parent.children.len(), 1);
        assert!(!of_parent.children[0].is_streaming);
    }

    /// A session of ANOTHER conversation is out of reach, even named by its exact id:
    /// the team is read from the graph for the session of the token, nothing else.
    #[tokio::test]
    async fn a_session_outside_the_team_is_out_of_reach_even_by_its_exact_id() {
        let (state, _parent, child) = state_with_a_family().await;
        let stranger = crate::test_helpers::test_chat_session(Some("nexus-sdk"));
        state
            .orchestrator
            .neo4j()
            .create_chat_session(&stranger)
            .await
            .unwrap();

        let mut req = message("do this for me", None);
        req.to = Some(stranger.id.to_string());
        let err = post_team_message(State(state), None, as_session(child.id), Json(req))
            .await
            .unwrap_err();
        assert!(
            matches!(err, AppError::BadRequest(ref m) if m.starts_with("team_no_match")),
            "{err:?}"
        );
    }

    /// A team message never widens anything. The body cannot carry a permission mode
    /// (refused before any delivery); a dispatch request opens NO session (the parent
    /// decides, with its own envelope); the receiver's stored mode does not move.
    #[tokio::test]
    async fn a_team_message_cannot_widen_the_receivers_access_nor_open_a_session() {
        for widening in [
            serde_json::json!({"text": "go", "permission_mode": "bypassPermissions"}),
            serde_json::json!({"text": "go", "add_dirs": ["/"]}),
            serde_json::json!({"text": "go", "cwd": "/"}),
        ] {
            assert!(
                serde_json::from_value::<TeamMessageRequest>(widening.clone()).is_err(),
                "{widening}"
            );
        }

        let (state, parent, child) = state_with_a_family().await;
        let graph = state.orchestrator.neo4j_arc();
        let before_children = graph.get_session_children(parent.id).await.unwrap().len();
        let before_mode = graph
            .get_chat_session(parent.id)
            .await
            .unwrap()
            .unwrap()
            .permission_mode;

        let mut req = message("open the backend task", Some("dispatch_request"));
        req.project_slug = Some("project-orchestrator-backend".into());
        req.task_id = Some("905f3700".into());
        let Json(answer) = post_team_message(State(state), None, as_session(child.id), Json(req))
            .await
            .unwrap();
        assert!(answer.delivered.is_empty(), "the parent is not live here");

        assert_eq!(
            graph.get_session_children(parent.id).await.unwrap().len(),
            before_children,
            "a dispatch request opens nothing by itself"
        );
        assert_eq!(
            graph
                .get_chat_session(parent.id)
                .await
                .unwrap()
                .unwrap()
                .permission_mode,
            before_mode,
            "the receiver's mode does not move"
        );
    }

    #[test]
    fn the_body_names_no_sender() {
        // The sender comes from the token: a body that tries to name one is refused.
        let err = serde_json::from_value::<TeamMessageRequest>(serde_json::json!({
            "text": "hi", "from": "someone-else"
        }));
        assert!(err.is_err());
    }
}
