//! `TeamHooks` — the `SessionHooks` impl that delivers `TeamMessage` and `TeamStatus`
//! via the sub-agent intercept path.
//!
//! ## Protocol (TEAM-1)
//!
//! A root session (spawned by nothing) has the Agent tool available. To send a team
//! message the model calls:
//!
//! ```json
//! {"tool": "Agent", "subagent_type": "team_message", "prompt": "the message",
//!  "description": "to:alice" }   ← "to:" prefix in description → single target
//!                                  OR omit → broadcast to all children
//! ```
//!
//! `TeamStatus` works the same way:
//! ```json
//! {"tool": "Agent", "subagent_type": "team_status", "prompt": ""}
//! ```
//!
//! ## Two ways, every engine (TEAM-2)
//!
//! The `Agent` intercept above only reaches a ROOT session of the native engine. The
//! two-way channel goes through the PO MCP `chat` tool instead (`team_message`,
//! `team_status`), which every engine and every child has. The SENDER is the session of
//! the signed token (`envelope::identify_caller`), never a name the model writes, and a
//! session reaches only its own team: the session that spawned it and the sessions it
//! spawned ([`team_of`]). Siblings talk through their parent, which is the dispatcher: a
//! child that needs work outside its envelope (another project of the workspace) sends a
//! `dispatch_request` UP, and the parent opens it with ITS envelope. The envelope itself
//! is never widened.

use std::sync::{Arc, Weak};

use async_trait::async_trait;
use nexus_claude::agent::{
    CompactionInfo, HookVerdict, SessionHooks, SubagentRequest, ToolCallInfo, ToolResultInfo,
    TurnContext, TurnDirective,
};
use tracing::info;
use uuid::Uuid;

use super::manager::ChatManager;

pub(crate) struct TeamHooks {
    /// Inner hooks (SubagentHooks or GraphSessionHooks), if any.
    inner: Option<Arc<dyn SessionHooks>>,
    manager: Weak<ChatManager>,
    session_id: String,
}

impl TeamHooks {
    pub(crate) fn new(
        inner: Option<Arc<dyn SessionHooks>>,
        manager: Weak<ChatManager>,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            manager,
            session_id: session_id.into(),
        }
    }

    /// Handle `Agent(subagent_type: "team_message", prompt: "...", description: "to:NAME")`.
    ///
    /// `description` prefix `to:<name>` routes to a named session; anything else broadcasts
    /// to every direct child of this session.
    async fn handle_team_message(&self, request: &SubagentRequest) -> Result<String, String> {
        let Some(manager) = self.manager.upgrade() else {
            return Err("server shutting down".to_owned());
        };

        let text = request.prompt.trim();
        if text.is_empty() {
            return Err("team_message: prompt is empty".to_owned());
        }

        // Resolve target(s).
        // `input.to` overrides `description`; then `description` prefix "to:NAME"; else all.
        let to_override = request
            .input
            .get("to")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let desc_prefix = request
            .description
            .as_deref()
            .and_then(|d| d.strip_prefix("to:"))
            .map(str::to_string);
        let target = to_override.or(desc_prefix);

        // Resolve the sender name: the session's metadata or a short UUID prefix.
        let from_session_id = Uuid::parse_str(&self.session_id)
            .map_err(|_| format!("team_message: invalid session id {}", self.session_id))?;
        let from_name = {
            // Use the session's title when available; fall back to UUID prefix.
            if let Ok(Some(node)) = manager.graph.get_chat_session(from_session_id).await {
                node.title
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| format!("agent-{}", &self.session_id[..8]))
            } else {
                format!("agent-{}", &self.session_id[..8])
            }
        };

        // Find target session(s): direct children of this session.
        let children = manager.direct_children(&self.session_id).await;

        if children.is_empty() {
            return Ok(r#"{"sent":false,"reason":"no team members"}"#.to_owned());
        }

        let targets: Vec<String> = match target.as_deref() {
            Some("all") | None => children.clone(),
            Some(name) => {
                // Match by session name prefix or id prefix.
                let mut matched: Vec<String> = Vec::new();
                for id in &children {
                    if id.starts_with(name) {
                        matched.push(id.clone());
                    } else if let Ok(uuid) = Uuid::parse_str(id) {
                        if let Ok(Some(node)) = manager.graph.get_chat_session(uuid).await {
                            if node.title.as_deref().is_some_and(|n| n.starts_with(name)) {
                                matched.push(id.clone());
                            }
                        }
                    }
                }
                if matched.is_empty() {
                    return Err(format!("team_message: no member matching {:?}", name));
                }
                matched
            }
        };

        let mut sent = 0usize;
        for to_id in &targets {
            match manager
                .inject_team_message(from_session_id, &from_name, to_id, text)
                .await
            {
                Ok(()) => {
                    info!(
                        from = %self.session_id,
                        to = %to_id,
                        "TeamMessage delivered"
                    );
                    sent += 1;
                }
                Err(e) => {
                    info!(
                        from = %self.session_id,
                        to = %to_id,
                        error = %e,
                        "TeamMessage delivery failed"
                    );
                }
            }
        }

        Ok(format!(
            r#"{{"sent":{sent},"to":{count},"text":{text_json}}}"#,
            count = targets.len(),
            text_json = serde_json::to_string(text).unwrap_or_default(),
        ))
    }

    /// Handle `Agent(subagent_type: "team_status")`.
    async fn handle_team_status(&self, _request: &SubagentRequest) -> Result<String, String> {
        let Some(manager) = self.manager.upgrade() else {
            return Err("server shutting down".to_owned());
        };

        let children = manager.direct_children(&self.session_id).await;
        if children.is_empty() {
            return Ok(r#"{"members":[]}"#.to_owned());
        }

        let mut members = Vec::new();
        for id in &children {
            if let Ok(uuid) = Uuid::parse_str(id) {
                if let Ok(Some(node)) = manager.graph.get_chat_session(uuid).await {
                    members.push(serde_json::json!({
                        "session_id": id,
                        "name": node.title.unwrap_or_default(),
                        "is_streaming": manager.session_is_streaming(id).await,
                    }));
                }
            }
        }

        Ok(serde_json::json!({ "members": members }).to_string())
    }
}

#[async_trait]
impl SessionHooks for TeamHooks {
    async fn before_tool(&self, call: &ToolCallInfo) -> HookVerdict {
        match &self.inner {
            Some(inner) => inner.before_tool(call).await,
            None => HookVerdict::Continue,
        }
    }

    async fn after_tool(&self, result: &ToolResultInfo) -> Option<String> {
        match &self.inner {
            Some(inner) => inner.after_tool(result).await,
            None => None,
        }
    }

    async fn before_compaction(&self, info: &CompactionInfo) -> Option<String> {
        match &self.inner {
            Some(inner) => inner.before_compaction(info).await,
            None => None,
        }
    }

    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        match &self.inner {
            Some(inner) => inner.before_turn(ctx).await,
            None => TurnDirective::default(),
        }
    }

    fn runs_subagents(&self) -> bool {
        // Preserve whatever the inner hooks say (SubagentHooks returns true for root sessions;
        // if there are no SubagentHooks, the session has no Agent tool and this returns false).
        self.inner
            .as_ref()
            .is_some_and(|inner| inner.runs_subagents())
    }

    async fn run_subagent(&self, request: &SubagentRequest) -> Result<String, String> {
        // Intercept team-only subagent types BEFORE delegating to the real subagent runner.
        match request.subagent_type.as_deref() {
            Some("team_message") => return self.handle_team_message(request).await,
            Some("team_status") => return self.handle_team_status(request).await,
            _ => {}
        }
        match &self.inner {
            Some(inner) => inner.run_subagent(request).await,
            None => Err("this host runs no sub-agent".to_owned()),
        }
    }
}

// ============================================================================
// TEAM-2 — the team of a session, its targets, and the shape of a request
// ============================================================================

/// One session a team member may write to.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TeamMember {
    pub session_id: String,
    pub name: String,
}

/// Who a session may talk to: the session that spawned it, and the sessions it spawned.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct Team {
    pub parent: Option<TeamMember>,
    pub children: Vec<TeamMember>,
}

fn member_of(node: &crate::neo4j::models::ChatSessionNode) -> TeamMember {
    let id = node.id.to_string();
    TeamMember {
        name: node
            .title
            .clone()
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| format!("agent-{}", &id[..8])),
        session_id: id,
    }
}

/// The session itself and its team, read from the graph. `None` = unknown session.
pub(crate) async fn team_of(
    graph: &dyn crate::neo4j::GraphStore,
    session_id: Uuid,
) -> anyhow::Result<Option<(TeamMember, Team)>> {
    let Some(me) = graph.get_chat_session(session_id).await? else {
        return Ok(None);
    };
    let parent_id = me
        .spawned_by
        .as_deref()
        .and_then(super::manager::parse_spawned_by)
        .and_then(|ctx| ctx.parent_session_id)
        .and_then(|id| Uuid::parse_str(&id).ok());
    let parent = match parent_id {
        Some(id) => graph.get_chat_session(id).await?.map(|n| member_of(&n)),
        None => None,
    };
    let children = graph
        .get_session_children(session_id)
        .await?
        .iter()
        .map(member_of)
        .collect();
    Ok(Some((member_of(&me), Team { parent, children })))
}

/// Why a team message has nobody to go to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TeamTargetError {
    /// The session has neither a parent nor a child.
    Alone,
    /// `to` names nobody of this team.
    Unknown(String),
    /// A dispatch request goes UP: a root session has nobody to dispatch for it.
    RootCannotDispatch,
}

impl std::fmt::Display for TeamTargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Alone => write!(f, "team_no_member: this session has no parent and no child"),
            Self::Unknown(to) => write!(f, "team_no_match: nobody of this team matches {to:?}"),
            Self::RootCannotDispatch => write!(
                f,
                "team_no_parent: a dispatch request goes to the parent, and this session has none"
            ),
        }
    }
}

/// What is being sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TeamMessageKind {
    /// Plain text to a team member.
    Message,
    /// "Open this work where I cannot": a child asks its parent to delegate it, with the
    /// parent's own envelope. The parent decides; nothing is opened by the request.
    DispatchRequest {
        project_slug: Option<String>,
        task_id: Option<String>,
    },
}

/// The members a message goes to.
///
/// `to`: absent = the children of a session that has some, else its parent; `parent`;
/// `children`; `all` (parent and children); or a member's session id, id prefix or title
/// prefix. A dispatch request always goes to the parent, whatever `to` says.
pub(crate) fn resolve_targets(
    team: &Team,
    to: Option<&str>,
    kind: &TeamMessageKind,
) -> Result<Vec<TeamMember>, TeamTargetError> {
    if matches!(kind, TeamMessageKind::DispatchRequest { .. }) {
        return team
            .parent
            .clone()
            .map(|p| vec![p])
            .ok_or(TeamTargetError::RootCannotDispatch);
    }
    if team.parent.is_none() && team.children.is_empty() {
        return Err(TeamTargetError::Alone);
    }
    let everyone = || -> Vec<TeamMember> {
        team.parent
            .iter()
            .chain(team.children.iter())
            .cloned()
            .collect()
    };
    match to.map(str::trim).filter(|s| !s.is_empty()) {
        None if team.children.is_empty() => Ok(team.parent.iter().cloned().collect()),
        None | Some("children") if !team.children.is_empty() => Ok(team.children.clone()),
        Some("children") => Err(TeamTargetError::Unknown("children".into())),
        Some("parent") => team
            .parent
            .clone()
            .map(|p| vec![p])
            .ok_or_else(|| TeamTargetError::Unknown("parent".into())),
        Some("all") => Ok(everyone()),
        Some(name) => {
            let all = everyone();
            if let Some(exact) = all.iter().find(|m| m.session_id == name) {
                return Ok(vec![exact.clone()]);
            }
            let matched: Vec<TeamMember> = all
                .into_iter()
                .filter(|m| m.session_id.starts_with(name) || m.name.starts_with(name))
                .collect();
            if matched.is_empty() {
                Err(TeamTargetError::Unknown(name.to_string()))
            } else {
                Ok(matched)
            }
        }
        None => unreachable!("handled by the two guarded arms above"),
    }
}

/// The text the receiving model reads (the `[team <name>]` prefix is added on delivery).
pub(crate) fn render_team_message(kind: &TeamMessageKind, text: &str) -> String {
    match kind {
        TeamMessageKind::Message => text.to_string(),
        TeamMessageKind::DispatchRequest {
            project_slug,
            task_id,
        } => {
            let mut head = String::from("[dispatch_request");
            if let Some(p) = project_slug.as_deref().filter(|s| !s.is_empty()) {
                head.push_str(&format!(" project={p}"));
            }
            if let Some(t) = task_id.as_deref().filter(|s| !s.is_empty()) {
                head.push_str(&format!(" task={t}"));
            }
            head.push_str("] ");
            head.push_str(text);
            head
        }
    }
}

#[cfg(test)]
mod team2_tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::GraphStore;
    use crate::test_helpers::test_chat_session;

    fn member(id: &str, name: &str) -> TeamMember {
        TeamMember {
            session_id: id.into(),
            name: name.into(),
        }
    }

    fn team(parent: Option<TeamMember>, children: Vec<TeamMember>) -> Team {
        Team { parent, children }
    }

    #[tokio::test]
    async fn a_child_sees_its_parent_and_a_parent_sees_its_children() {
        let graph = MockGraphStore::new();
        let mut root = test_chat_session(Some("nexus-sdk"));
        root.title = Some("coordinator".into());
        graph.create_chat_session(&root).await.unwrap();
        let mut child = test_chat_session(Some("nexus-sdk"));
        child.spawned_by = Some(crate::chat::envelope::conversation_spawned_by(root.id));
        graph.create_chat_session(&child).await.unwrap();

        let (me, t) = team_of(&graph, child.id).await.unwrap().unwrap();
        assert_eq!(me.session_id, child.id.to_string());
        assert_eq!(
            t.parent.as_ref().map(|p| p.name.as_str()),
            Some("coordinator")
        );
        assert!(t.children.is_empty());

        let (_, t) = team_of(&graph, root.id).await.unwrap().unwrap();
        assert!(t.parent.is_none());
        assert_eq!(t.children.len(), 1);
        assert_eq!(t.children[0].session_id, child.id.to_string());

        assert!(team_of(&graph, Uuid::new_v4()).await.unwrap().is_none());
    }

    #[test]
    fn a_child_writes_up_by_default_and_a_parent_writes_down() {
        let p = member("p-1", "parent");
        let c = member("c-1", "child");
        // The child (no children of its own): to its parent.
        let child_team = team(Some(p.clone()), vec![]);
        assert_eq!(
            resolve_targets(&child_team, None, &TeamMessageKind::Message).unwrap(),
            vec![p.clone()]
        );
        // The parent: to its children.
        let parent_team = team(None, vec![c.clone()]);
        assert_eq!(
            resolve_targets(&parent_team, None, &TeamMessageKind::Message).unwrap(),
            vec![c.clone()]
        );
        // A middle session can name either side.
        let middle = team(Some(p.clone()), vec![c.clone()]);
        assert_eq!(
            resolve_targets(&middle, Some("parent"), &TeamMessageKind::Message).unwrap(),
            vec![p.clone()]
        );
        assert_eq!(
            resolve_targets(&middle, Some("all"), &TeamMessageKind::Message).unwrap(),
            vec![p, c]
        );
    }

    #[test]
    fn nobody_outside_the_team_is_reachable() {
        let t = team(Some(member("p-1", "parent")), vec![member("c-1", "child")]);
        assert_eq!(
            resolve_targets(&t, Some("someone-else"), &TeamMessageKind::Message),
            Err(TeamTargetError::Unknown("someone-else".into()))
        );
        assert_eq!(
            resolve_targets(&team(None, vec![]), None, &TeamMessageKind::Message),
            Err(TeamTargetError::Alone)
        );
        // A member is found by id, id prefix or title prefix.
        assert_eq!(
            resolve_targets(&t, Some("chi"), &TeamMessageKind::Message).unwrap(),
            vec![member("c-1", "child")]
        );
    }

    #[test]
    fn a_dispatch_request_always_goes_up() {
        let p = member("p-1", "parent");
        let kind = TeamMessageKind::DispatchRequest {
            project_slug: Some("project-orchestrator-backend".into()),
            task_id: Some("905f3700".into()),
        };
        let t = team(Some(p.clone()), vec![member("c-1", "child")]);
        // Even when `to` names a child: a request to open work goes to the dispatcher.
        assert_eq!(resolve_targets(&t, Some("c-1"), &kind).unwrap(), vec![p]);
        // A root has nobody to dispatch for it.
        assert_eq!(
            resolve_targets(&team(None, vec![member("c-1", "child")]), None, &kind),
            Err(TeamTargetError::RootCannotDispatch)
        );
        assert_eq!(
            render_team_message(&kind, "the parity matrix is backend work"),
            "[dispatch_request project=project-orchestrator-backend task=905f3700] the parity matrix is backend work"
        );
        assert_eq!(render_team_message(&TeamMessageKind::Message, "hi"), "hi");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use uuid::Uuid;

    use crate::chat::manager::{test_support, ChatManager};
    use crate::test_helpers::mock_app_state;

    fn make_manager() -> ChatManager {
        let state = mock_app_state();
        ChatManager::new_without_memory(state.neo4j, state.meili, test_support::chat_config())
    }

    /// An idle target session: `inject_team_message` wins the compare_exchange
    /// and sets `is_streaming = true` (a stream_response is spawned).
    #[tokio::test]
    async fn a_team_message_to_an_idle_session_starts_streaming() {
        let manager = make_manager();
        let to_id = Uuid::new_v4().to_string();
        test_support::insert_live_session_without_cli(&manager, &to_id).await;

        let from_id = Uuid::new_v4();
        manager
            .inject_team_message(from_id, "sender", &to_id, "hello from peer")
            .await
            .expect("inject succeeds");

        // compare_exchange(false → true) was claimed by inject_team_message.
        assert!(
            manager.session_is_streaming(&to_id).await,
            "is_streaming must be set after idle delivery"
        );

        // The pending queue must be empty: the message was sent as a live turn.
        let sessions = manager.active_sessions.read().await;
        let queue = sessions[&to_id].pending_messages.lock().await;
        assert!(queue.is_empty(), "no pending messages for an idle delivery");
    }

    /// A streaming target session: `inject_team_message` loses the compare_exchange
    /// and parks the message in the pending queue for drain.
    #[tokio::test]
    async fn a_team_message_to_a_streaming_session_is_queued() {
        use crate::chat::types::PendingMessageKind;

        let manager = make_manager();
        let to_id = Uuid::new_v4().to_string();
        test_support::insert_live_session_without_cli(&manager, &to_id).await;
        // Manually flip is_streaming to true BEFORE inject.
        {
            let sessions = manager.active_sessions.read().await;
            sessions[&to_id].is_streaming.store(true, Ordering::SeqCst);
        }

        let from_id = Uuid::new_v4();
        manager
            .inject_team_message(from_id, "peer", &to_id, "queued text")
            .await
            .expect("inject succeeds");

        let sessions = manager.active_sessions.read().await;
        let queue = sessions[&to_id].pending_messages.lock().await;
        assert_eq!(queue.len(), 1, "one message must be queued");
        let msg = &queue[0];
        assert!(
            matches!(
                &msg.kind,
                PendingMessageKind::TeamMessage { from_session_id, from_name }
                if *from_session_id == from_id && from_name == "peer"
            ),
            "queued message must be a TeamMessage from the correct sender: {:?}",
            msg.kind
        );
    }

    /// Rate limit: after 10 messages the 11th is silently dropped.
    #[tokio::test]
    async fn team_message_rate_limit_drops_after_10_per_minute() {
        let manager = make_manager();
        let to_id = Uuid::new_v4().to_string();
        test_support::insert_live_session_without_cli(&manager, &to_id).await;

        let from_id = Uuid::new_v4();
        // Seed 10 recent instants for this (from, to) pair.
        {
            let mut rate = manager.team_msg_rate.lock().await;
            let key = (from_id, to_id.clone());
            let window = rate.entry(key).or_default();
            for _ in 0..10 {
                window.push(std::time::Instant::now());
            }
        }

        let result = manager
            .inject_team_message(from_id, "sender", &to_id, "over the limit")
            .await;
        assert!(result.is_ok(), "rate-limit drop must return Ok, not Err");

        // Nothing was queued.
        let sessions = manager.active_sessions.read().await;
        let queue = sessions[&to_id].pending_messages.lock().await;
        assert!(queue.is_empty(), "rate-limited message must not be queued");
    }

    /// Injecting to a session that does not exist returns an error.
    #[tokio::test]
    async fn team_message_to_unknown_session_returns_error() {
        let manager = make_manager();
        let result = manager
            .inject_team_message(Uuid::new_v4(), "sender", "nonexistent-id", "hi")
            .await;
        assert!(result.is_err(), "must fail for an unknown session");
    }
}
