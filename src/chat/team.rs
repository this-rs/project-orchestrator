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
//! ## Limitations (TEAM-1)
//!
//! - Only ROOT sessions can SEND (the `Agent` tool is not offered to children).
//! - Children can RECEIVE via `inject_team_message` (the queue path).
//! - TEAM-2 will add a dedicated MCP tool endpoint for children.

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
