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
