//! Sub-agents of a native session (P18): the `Agent` tool of the nexus harness, run as a
//! child session of project-orchestrator (decision A18: a provider without sub-agents of its
//! own gets the host's child session).
//!
//! The native harness offers its `Agent` tool when the session's hooks say
//! `runs_subagents()`, and hands every call to `SessionHooks::run_subagent`. [`SubagentHooks`]
//! is that host: it wraps the session's other hooks (forwarded untouched) and, on a call,
//! [`ChatManager::run_subagent`] opens the child exactly as `chat(send_message)` opens one
//! for an agent caller (A17, `envelope.rs`):
//!
//! * the envelope of the parent: depth 1 (a session that was itself spawned cannot spawn),
//!   at most [`MAX_LIVE_CHILDREN`](super::envelope::MAX_LIVE_CHILDREN) children alive, a
//!   permission mode never wider than the parent's, the parent's project and directories;
//! * `spawned_by = conversation { parent_session_id, tool_use_id }`: the child is in the
//!   parent's session tree, and its token is restricted (H6: its parent is a third party);
//! * the parent's provider and model, the parent's claims;
//! * the prompt as an inert message (no reference block may ride on a model's text).
//!
//! The call waits for the child's first `result` and answers its text; the child is then
//! closed (it stays resumable and visible). A dropped call (the parent's turn interrupted or
//! the call cancelled) closes the child too.
//!
//! Depth is held twice: the backend gives [`SubagentHooks`] only to a native session whose
//! record has no `spawned_by` (`ChatManager::build_agent_spec_with_access`), and the envelope
//! refuses a parent that has a parent.

use std::sync::{Arc, Weak};

use async_trait::async_trait;
use nexus_claude::agent::{
    CompactionInfo, HookVerdict, SessionHooks, SubagentRequest, ToolCallInfo, ToolResultInfo,
    TurnContext, TurnDirective,
};
use uuid::Uuid;

use super::manager::ChatManager;
use super::types::{ChatEvent, ChatRequest, SpawnedBy};

/// The hooks of a native session that may run sub-agents: its other hooks, plus the host of
/// the `Agent` tool.
pub(crate) struct SubagentHooks {
    inner: Option<Arc<dyn SessionHooks>>,
    manager: Weak<ChatManager>,
    parent_session_id: String,
    user_claims: Option<crate::auth::jwt::Claims>,
}

impl SubagentHooks {
    pub(crate) fn new(
        inner: Option<Arc<dyn SessionHooks>>,
        manager: Weak<ChatManager>,
        parent_session_id: impl Into<String>,
        user_claims: Option<crate::auth::jwt::Claims>,
    ) -> Self {
        Self {
            inner,
            manager,
            parent_session_id: parent_session_id.into(),
            user_claims,
        }
    }
}

#[async_trait]
impl SessionHooks for SubagentHooks {
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
        true
    }

    async fn run_subagent(&self, request: &SubagentRequest) -> Result<String, String> {
        let Some(manager) = self.manager.upgrade() else {
            return Err("the server is shutting down: no sub-agent can start".to_owned());
        };
        manager
            .run_subagent(&self.parent_session_id, self.user_claims.clone(), request)
            .await
    }
}

/// Closes the child session if the call ends before the child did (the future dropped).
struct CloseOnDrop {
    manager: Weak<ChatManager>,
    child: Option<String>,
}

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        let (Some(child), Some(manager)) = (self.child.take(), self.manager.upgrade()) else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                if let Err(e) = manager.close_session(&child).await {
                    tracing::debug!(child = %child, error = %e, "sub-agent close after a cut call");
                }
            });
        }
    }
}

/// The answer of a sub-agent from its `result`: the result text, else what it said.
fn answer_of(events: &[ChatEvent]) -> Option<Result<String, String>> {
    let (is_error, result_text, subtype) = events.iter().find_map(|e| match e {
        ChatEvent::Result {
            is_error,
            result_text,
            subtype,
            ..
        } => Some((*is_error, result_text.clone(), subtype.clone())),
        _ => None,
    })?;
    let said: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            ChatEvent::AssistantText {
                content,
                parent_tool_use_id: None,
            } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    let text = result_text
        .filter(|t| !t.trim().is_empty())
        .or_else(|| said.last().map(|s| s.to_string()))
        .unwrap_or_default();
    Some(if is_error {
        Err(format!(
            "the sub-agent failed ({subtype}){}",
            if text.is_empty() {
                String::new()
            } else {
                format!(": {text}")
            }
        ))
    } else {
        Ok(text)
    })
}

impl ChatManager {
    /// Lets this manager hand itself to what outlives a call: the sub-agents of native
    /// sessions (P18) open their child sessions through it. Called once the manager is in its
    /// `Arc`; without it, native sessions are given no `Agent` tool.
    pub fn enable_subagents(self: &Arc<Self>) {
        let _ = self.self_ref.set(Arc::downgrade(self));
    }

    /// The manager itself, once [`Self::enable_subagents`] was called.
    pub(crate) fn weak_self(&self) -> Option<Weak<ChatManager>> {
        self.self_ref.get().cloned()
    }

    /// Runs one sub-agent of `parent_session_id` as its child session and returns what it
    /// answered (module documentation). `Err` is what the model reads when it could not run.
    pub(crate) async fn run_subagent(
        self: Arc<Self>,
        parent_session_id: &str,
        user_claims: Option<crate::auth::jwt::Claims>,
        request: &SubagentRequest,
    ) -> Result<String, String> {
        use super::envelope;
        let refused = |why: String| format!("the sub-agent could not start: {why}");
        let parent_uuid = Uuid::parse_str(parent_session_id).map_err(|e| refused(e.to_string()))?;
        let parent = self
            .graph
            .get_chat_session(parent_uuid)
            .await
            .map_err(|e| refused(e.to_string()))?
            .ok_or_else(|| refused("the parent session is unknown".to_owned()))?;
        let env =
            envelope::resolve_envelope(self.graph.as_ref(), self.as_ref(), parent_session_id, None)
                .await
                .map_err(|e| refused(e.code().to_owned()))?;
        let mut child = ChatRequest {
            message: crate::refs::compose::inert(&request.prompt),
            cwd: parent.cwd.clone(),
            model: Some(parent.model.clone()),
            provider: parent.provider_id.clone(),
            add_dirs: parent.add_dirs.clone(),
            user_claims,
            task_context: request.description.clone(),
            ..Default::default()
        };
        let default_mode = self.default_permission_mode().await;
        env.apply(&mut child, &default_mode)
            .map_err(|e| refused(e.code().to_owned()))?;
        child.spawned_by = Some(
            SpawnedBy::Conversation {
                parent_session_id: env.parent_session_id,
                tool_use_id: Some(request.tool_call_id.clone()),
            }
            .to_json_string(),
        );
        let opened = self
            .create_session(&child)
            .await
            .map_err(|e| refused(format!("{e:#}")))?;
        let child_id = opened.session_id;
        let mut guard = CloseOnDrop {
            manager: Arc::downgrade(&self),
            child: Some(child_id.clone()),
        };
        tracing::info!(parent = %parent_session_id, child = %child_id, tool_use_id = %request.tool_call_id, "sub-agent started");
        let answer = self.wait_for_answer(&child_id).await;
        // The child ended: close it (it stays resumable); the guard has nothing left to do.
        guard.child = None;
        if let Err(e) = self.close_session(&child_id).await {
            tracing::debug!(child = %child_id, error = %e, "sub-agent close");
        }
        answer
    }

    /// The first `result` of the session `sid`, live or already persisted.
    async fn wait_for_answer(&self, sid: &str) -> Result<String, String> {
        let mut rx = self
            .subscribe(sid)
            .await
            .map_err(|e| format!("the sub-agent cannot be followed: {e:#}"))?;
        let mut seen: Vec<ChatEvent> = Vec::new();
        // A turn that ended before the subscription is in the store.
        if let Ok(uuid) = Uuid::parse_str(sid) {
            let persisted: Vec<ChatEvent> = self
                .graph
                .get_chat_events(uuid, 0, 5_000)
                .await
                .unwrap_or_default()
                .iter()
                .filter_map(|r| serde_json::from_str(&r.data).ok())
                .collect();
            if let Some(answer) = answer_of(&persisted) {
                return answer;
            }
        }
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let done = matches!(event, ChatEvent::Result { .. });
                    let closed = matches!(event, ChatEvent::SessionClosed { .. });
                    seen.push(event);
                    if done {
                        return answer_of(&seen).unwrap_or_else(|| Ok(String::new()));
                    }
                    if closed {
                        return Err("the sub-agent's session was closed before it answered".into());
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err("the sub-agent's session ended before it answered".into())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_is_the_result_text_else_the_last_thing_said() {
        let result = |is_error: bool, text: Option<&str>| ChatEvent::Result {
            session_id: "s".into(),
            duration_ms: 1,
            cost_usd: None,
            subtype: if is_error {
                "error_during_execution"
            } else {
                "success"
            }
            .into(),
            is_error,
            num_turns: None,
            result_text: text.map(str::to_owned),
            cost: None,
            usage: None,
            model: None,
            stop_reason: None,
        };
        let said = |t: &str| ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        };
        assert_eq!(answer_of(&[said("x")]), None, "no result yet");
        assert_eq!(
            answer_of(&[said("first"), said("last"), result(false, Some("FINAL"))]),
            Some(Ok("FINAL".into()))
        );
        assert_eq!(
            answer_of(&[said("first"), said("last"), result(false, None)]),
            Some(Ok("last".into()))
        );
        let failed = answer_of(&[result(true, Some("boom"))])
            .unwrap()
            .unwrap_err();
        assert!(
            failed.contains("failed") && failed.contains("boom"),
            "{failed}"
        );
    }
}
