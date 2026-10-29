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
//! * the parent's provider and model, the parent's claims — and, when the parent has none
//!   (reopened by the restart recovery, which keeps no claims), the parent's stored OWNER:
//!   the child belongs to the same person (P18b round 3, fix A). No claims mints no token,
//!   so the child's MCP access stays as restricted as before;
//! * the parent's workspace (inherited by the envelope), with its directories clamped to
//!   what the parent could already see (P18b round 2, fix 1);
//! * the prompt as an inert message (no reference block may ride on a model's text).
//!
//! The call waits for the child's first `result` and answers its text; the child is then
//! closed (it stays resumable and visible). A dropped call (the parent's turn interrupted or
//! the call cancelled) closes the child too — at ANY point of the call, the opening
//! included: [`ChatManager::run_subagent`] opens, awaits and closes the child in a task of
//! its own and only AWAITS that task, so a drop cannot cut the opening halfway. That matters
//! because `create_session` runs as long as the provider's open timeout and has already sent
//! the child's first message when it returns: a future dropped inside it used to leave a
//! child running that nobody knew the id of. The caller's drop closes a one-shot channel the
//! task watches; the task then closes the child as soon as the opening gives its id.
//!
//! The child's answer is read from its live event stream, and from its stored thread
//! whenever the stream cannot be trusted: before the first wait (a turn that ended before
//! the subscription), once the stream is closed, and — from the first `Lagged` on — every
//! second until the thread shows the `result` or the close, the live stream still watched
//! alongside (P18b round 3, fix C: the `result` may have been among the dropped events and
//! reach the thread arbitrarily later; a fixed number of re-reads left the parent waiting
//! until its turn timeout). The tick spaces the re-reads: a child that lags again and again
//! costs one tail read per second. What is kept from the stream is bounded
//! ([`keep_for_answer`]): a chatty child cannot grow it.
//!
//! The close of the child is backed by a guard armed from the moment the child exists
//! until its explicit `close_session` has RETURNED (round 3, fix E: a panic in the close
//! itself is covered; a second close is a harmless "not found or inactive"); the guard
//! owns the parent's slot, so a panic gives it back only after the detached close.
//!
//! Depth is held three times: the backend gives [`SubagentHooks`] only to a native session
//! whose record has no `spawned_by` (`ChatManager::build_agent_spec_with_access`); a spawned
//! native session is opened with `SessionSpec::lineage` at depth 1, so the harness itself
//! refuses it the tool; and the envelope refuses a parent that has a parent. A child keeps its
//! `spawned_by` when it moves to another provider (relay).
//!
//! The cap of live children holds under parallel calls (the harness runs the calls of one step
//! concurrently): a slot of the parent is taken BEFORE the envelope counts its children, in
//! [`ChatManager::run_subagent`], and held by [`ChatManager::open_wait_close`]'s OWN task —
//! not by `run_subagent`'s frame — until the sub-agent actually ends, right alongside its
//! `close_session` ([`MAX_LIVE_CHILDREN`](super::envelope::MAX_LIVE_CHILDREN) slots per
//! parent; P18b round 2, fix 2: a dropped `run_subagent` future used to free the slot as soon
//! as it was dropped, which could be well before the task it spawned had even learned the
//! child's id, let alone closed it). The slot counts only `Agent` calls; the cap counts EVERY
//! child of the parent: the slot also carries the opening's ticket (`envelope::ChildOpenings`,
//! P18-suite point 6), which the envelope adds to the live children — the children that
//! `chat(send_message)` and `plan(delegate_task)` open hold one too — until the child is live.
//! The ceiling is the parent's CURRENT mode: a native
//! session's mode changes are kept live and persisted (`set_session_permission_mode`).
//!
//! What a child may not do: ask the user (`AskUserQuestion` is denied to a spawned native
//! session; a question seen anyway fails the call at once); wait for a permission (round 3,
//! fix B, made deterministic by the review of #724: a request the backend does not answer
//! itself — no grant of the child, no lasting rule of its owner covers it, which the child's
//! session decides BEFORE it shows the request — fails the call at once; the child's waiting
//! requests are then answered `deny` through the path a user's answer takes, so none is
//! left in its owner's cockpit, and the child is closed like on every other end — the request
//! would otherwise hold the parent until its 30-minute turn timeout. A person who opens the
//! child (its run is listed, its state live) sees the request but cannot answer it: that is
//! the choice made, not an oversight); return more than [`MAX_ANSWER_CHARS`] characters (the rest is
//! cut, and said). Failing fast was chosen over opening the child with a
//! deny-instead-of-ask policy: the harness has no such mode (`PolicyMode` is plan / ask /
//! auto-edits / trust), and a fail needs no new policy plumbing and no new prompt — the
//! parent's model reads why and does the step itself. A covered request never fails the
//! call, however long its automatic answer takes (unless that answer fails).
//! The call otherwise waits as long as the child works, bounded by the parent's turn
//! timeout. What a child spent is added to its parent's CHILDREN spend when the call ends
//! ([`ChatManager::aggregated_cost_usd`] = own + children); the parent's own
//! `total_cost_usd` is untouched, so its routing decision is not charged twice (PYSR-3).

use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use nexus_claude::agent::{
    CompactionInfo, HookVerdict, SessionHooks, SubagentRequest, ToolCallInfo, ToolResultInfo,
    TurnContext, TurnDirective,
};
use uuid::Uuid;

use super::manager::ChatManager;
use super::provider::cognitive::signature::TaskClass;
use super::provider::policy::SessionAccess;
use super::session_pool::{WarmFacts, WarmKey};
use super::types::{ChatEvent, ChatRequest, SpawnedBy};

// ── Sub-agent type registry (ISO2) ──────────────────────────────────────────

/// A known agent type: class for cognitive routing, tool restrictions for the child's
/// session policy.  CC types carry no host-imposed restriction; PO types do.
pub(crate) struct SubagentTypeDef {
    /// The [`TaskClass`] the type implies for cognitive routing.
    #[allow(dead_code)]
    pub class: TaskClass,
    /// Tool names the child is denied (applied as `spec.policy.deny`).  Empty = no restriction.
    pub denied_tools: &'static [&'static str],
}

/// Resolves a `subagent_type` name to its definition, or `None` for unknown types.
///
/// CC types (explore, general-purpose, plan, claude-code-guide, fork, web-fetch) carry no
/// tool restriction from the host.  PO types (research, code-reviewer) deny write tools.
/// An unknown type gets `None` and the caller rejects the call.
pub(crate) fn resolve_subagent_type(name: &str) -> Option<SubagentTypeDef> {
    Some(match name.trim().to_lowercase().as_str() {
        // ── CC types — no tool restriction from the host ─────────
        "explore" | "claude-code-guide" => SubagentTypeDef {
            class: TaskClass::Simple,
            denied_tools: &[],
        },
        "general-purpose" | "claude" => SubagentTypeDef {
            class: TaskClass::Complex,
            denied_tools: &[],
        },
        "plan" => SubagentTypeDef {
            class: TaskClass::Creative,
            denied_tools: &[],
        },
        "fork" | "web-fetch" => SubagentTypeDef {
            class: TaskClass::Complex,
            denied_tools: &[],
        },
        // ── PO types — each with explicit tool restrictions ──────
        "research" => SubagentTypeDef {
            class: TaskClass::Simple,
            denied_tools: &["Edit", "Write", "NotebookEdit"],
        },
        "code-reviewer" => SubagentTypeDef {
            class: TaskClass::Complex,
            denied_tools: &["Edit", "Write", "NotebookEdit"],
        },
        _ => return None,
    })
}

/// Comma-separated list of known agent type names, for error messages.
pub(crate) fn known_subagent_type_list() -> String {
    [
        "explore",
        "claude-code-guide",
        "general-purpose",
        "claude",
        "plan",
        "fork",
        "web-fetch",
        "research",
        "code-reviewer",
    ]
    .join(", ")
}

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

    async fn compaction_directive(
        &self,
        info: &CompactionInfo,
    ) -> nexus_claude::agent::CompactionDirective {
        match &self.inner {
            Some(inner) => inner.compaction_directive(info).await,
            None => nexus_claude::agent::CompactionDirective::none(),
        }
    }

    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        match &self.inner {
            Some(inner) => inner.before_turn(ctx).await,
            None => TurnDirective::default(),
        }
    }

    // Claude Code's own events (contract v7, ISO1) belong to the inner hooks: a wrapper that
    // dropped them would hide the graph's answers on the agent engine.
    async fn session_start(
        &self,
        info: &nexus_claude::agent::SessionStartInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.session_start(info).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn user_prompt_submit(
        &self,
        prompt: &nexus_claude::agent::PromptInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.user_prompt_submit(prompt).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn before_stop(
        &self,
        info: &nexus_claude::agent::StopInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.before_stop(info).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn before_subagent_stop(
        &self,
        info: &nexus_claude::agent::SubagentStopInfo,
    ) -> nexus_claude::agent::HookReply {
        match &self.inner {
            Some(inner) => inner.before_subagent_stop(info).await,
            None => nexus_claude::agent::HookReply::none(),
        }
    }

    async fn notification(&self, info: &nexus_claude::agent::NotificationInfo) {
        if let Some(inner) = &self.inner {
            inner.notification(info).await;
        }
    }

    fn runs_subagents(&self) -> bool {
        true
    }

    fn runs_background_subagents(&self) -> bool {
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

/// One running sub-agent of a parent (module documentation): taken before the envelope
/// counts the parent's live children, given back when the sub-agent ends or its call is dropped.
pub(crate) struct SubagentSlot {
    manager: Weak<ChatManager>,
    parent: String,
    /// The opening of the child in flight (`envelope::ChildOpenings`, P18-suite point 6):
    /// counted by the envelope of the parent's other openings until the child is live.
    opening: Option<super::envelope::OpeningTicket>,
}

impl SubagentSlot {
    /// A slot of `parent`, or `None` when `max_live_children` slots are taken.
    /// The cap is [`ChatManager::max_live_children`]: the stored setting (PLAFOND-RT), default
    /// [`super::envelope::MAX_LIVE_CHILDREN`].
    pub(crate) fn take(manager: &Arc<ChatManager>, parent: &str) -> Option<Self> {
        let cap = manager.max_live_children();
        let mut slots = manager
            .subagent_slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let taken = slots.entry(parent.to_string()).or_insert(0);
        if *taken >= cap {
            return None;
        }
        *taken += 1;
        drop(slots);
        Some(Self {
            manager: Arc::downgrade(manager),
            parent: parent.to_string(),
            opening: Some(manager.child_openings.reserve(parent)),
        })
    }

    /// The opening this slot's child holds, for the envelope's count.
    pub(crate) fn opening(&self) -> Option<&super::envelope::OpeningTicket> {
        self.opening.as_ref()
    }

    /// The child is live (or its opening failed): it no longer counts as an opening; the
    /// slot itself stays taken until the sub-agent ends.
    fn opened(&mut self) {
        self.opening = None;
    }
}

impl Drop for SubagentSlot {
    fn drop(&mut self) {
        let Some(manager) = self.manager.upgrade() else {
            return;
        };
        let mut slots = manager
            .subagent_slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(taken) = slots.get_mut(&self.parent) {
            *taken = taken.saturating_sub(1);
            if *taken == 0 {
                slots.remove(&self.parent);
            }
        }
    }
}

/// What the warm-session pool says about one delegation (CACHE-POOL): the key and the facts
/// the child must be served with — computed for THIS delegation, never read from the warm
/// session — and the warm session it may resume, when the pool had an eligible one.
struct Reuse {
    key: super::session_pool::WarmKey,
    facts: super::session_pool::WarmFacts,
    warm: Option<super::session_pool::WarmSession>,
}

impl Reuse {
    /// What becomes of the child once its delegation ended. A child that ANSWERED goes back
    /// to the pool warm. One that did not — the call was cancelled, the child errored, could
    /// not be followed — is never offered again: its thread ends on a turn nobody finished,
    /// and resuming it would hand the next delegation that unfinished state (CACHE-POOL,
    /// "a contaminated session is never reused"). It is dropped from the pool instead.
    fn finish(
        self,
        pool: &super::session_pool::SessionPool,
        session_id: &str,
        answered: bool,
        now: std::time::Instant,
    ) {
        if answered {
            self.release(pool, session_id, now);
        } else {
            pool.abandon(session_id);
        }
    }

    /// Gives the child back to the pool, warm, once it is closed: the next delegation of the
    /// same conversation, on the same engine and model, with the same facts, resumes it.
    fn release(
        self,
        pool: &super::session_pool::SessionPool,
        session_id: &str,
        now: std::time::Instant,
    ) {
        let opened_at = self.warm.as_ref().map_or(now, |warm| warm.opened_at);
        let uses = self.warm.as_ref().map_or(1, |warm| warm.uses + 1);
        pool.release(
            super::session_pool::WarmSession {
                session_id: session_id.to_owned(),
                key: self.key,
                facts: self.facts,
                opened_at,
                last_used: now,
                uses,
            },
            now,
        );
    }
}

/// A backstop for exactly one window: between a resource existing and the END of the
/// explicit code that releases it. [`Self::new`] arms it with what to run if it is ever
/// dropped armed; [`Self::disarm`] is called right AFTER that explicit release returned, so
/// a panic inside the release itself is still covered (the release must then be safe to
/// run twice). Used by [`ChatManager::open_wait_close`] (P18b round 2, fix 3; round 3,
/// fix E) for the window between `create_session` succeeding and the child's explicit
/// `close_session` having returned: a panic there (none today) would otherwise leave the
/// child open with nothing left to close it until the next restart's recovery scan.
struct ClosePanicGuard {
    armed: bool,
    on_drop: Option<Box<dyn FnOnce() + Send>>,
}

impl ClosePanicGuard {
    fn new(on_drop: impl FnOnce() + Send + 'static) -> Self {
        Self {
            armed: true,
            on_drop: Some(Box::new(on_drop)),
        }
    }

    /// The ordinary path took over: this guard's backstop must not run any more.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ClosePanicGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Some(f) = self.on_drop.take() {
            f();
        }
    }
}

/// Longest answer of a sub-agent given back to its parent, in characters: a sub-agent that
/// copies a whole file must not fill the parent's context.
pub(crate) const MAX_ANSWER_CHARS: usize = 64 * 1024;

/// `text` cut to [`MAX_ANSWER_CHARS`], saying how much was left out.
fn bounded(text: String) -> String {
    let total = text.chars().count();
    if total <= MAX_ANSWER_CHARS {
        return text;
    }
    let kept: String = text.chars().take(MAX_ANSWER_CHARS).collect();
    format!(
        "{kept}\n[… {} characters of the sub-agent's answer left out]",
        total - MAX_ANSWER_CHARS
    )
}

/// A question the child asked: the call fails on it.
fn is_question(event: &ChatEvent) -> bool {
    matches!(event, ChatEvent::AskUserQuestion { .. })
}

/// Something the child said in its own voice (not inside one of its own tool calls).
fn is_said(event: &ChatEvent) -> bool {
    matches!(
        event,
        ChatEvent::AssistantText {
            parent_tool_use_id: None,
            ..
        }
    )
}

/// The end of the child's turn.
fn is_result(event: &ChatEvent) -> bool {
    matches!(event, ChatEvent::Result { .. })
}

/// Adds `event` to what [`answer_of`] will be given, keeping only what it actually reads:
/// the first question, the LAST thing said, and the first `result` — at most three events,
/// whatever the child emits. Without this bound a chatty child (thousands of deltas and
/// tool results) grew the parent's vector for the whole call.
fn keep_for_answer(seen: &mut Vec<ChatEvent>, event: ChatEvent) {
    if is_said(&event) {
        // `answer_of` falls back to the LAST thing said: replace it in place.
        match seen.iter_mut().find(|e| is_said(e)) {
            Some(held) => *held = event,
            None => seen.push(event),
        }
    } else if (is_question(&event) && !seen.iter().any(is_question))
        || (is_result(&event) && !seen.iter().any(is_result))
    {
        seen.push(event);
    }
}

/// The answer of a sub-agent from its `result`: the result text, else what it said. A
/// sub-agent that asked the user a question fails the call: nobody sees a child's question.
fn answer_of(events: &[ChatEvent]) -> Option<Result<String, String>> {
    if events.iter().any(is_question) {
        return Some(Err(
            "the sub-agent asked the user a question; a sub-agent cannot ask: give it what it needs in the prompt".to_owned(),
        ));
    }
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
        Ok(bounded(text))
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
        // ── Validate subagent_type and resolve tool restrictions (ISO2) ──────
        let type_def = request.subagent_type.as_deref().map(resolve_subagent_type);
        if let Some(None) = type_def {
            // The model named a type not in the registry.
            let kind = request.subagent_type.as_deref().unwrap_or("");
            return Err(refused(format!(
                "unknown subagent_type \"{kind}\": known types are [{}]",
                known_subagent_type_list(),
            )));
        }
        let denied_tools: Vec<String> = type_def.flatten().map_or_else(Vec::new, |td| {
            td.denied_tools.iter().map(|s| (*s).to_owned()).collect()
        });
        // Taken before the envelope counts the parent's children (A17); moved into
        // `open_wait_close`'s own task below and held there for the child's whole real
        // lifetime, NOT in this frame — a drop of this future (the parent's turn
        // interrupted) must not free the slot before the spawned task has actually
        // closed the child (P18b round 2, fix 2).
        let max = self.max_live_children();
        let slot = SubagentSlot::take(&self, parent_session_id).ok_or_else(|| {
            refused(format!(
                "envelope_too_many_children ({max} sub-agents of one session are already running; \
                 raise it in Settings or with PUT /api/chat/delegation-limits)"
            ))
        })?;
        let env = envelope::resolve_envelope_for_opening(
            self.graph.as_ref(),
            self.as_ref(),
            parent_session_id,
            None,
            max,
            slot.opening(),
        )
        .await
        .map_err(|e| refused(e.code().to_owned()))?;
        // AUTO-SUB: a parent conversation in Auto (mode `full`, or a pool of two models or
        // more) routes every `Agent` call over every candidate its person and project allow —
        // Claude Code included — instead of the child inheriting the parent's provider and
        // model. A pinned parent (a provider named by the request, or a single ticked model)
        // keeps the inheritance: its choice is never substituted. The decision is taken and
        // applied by the open-time routing (`cognitive_decision`), which the child reaches by
        // clearing `provider`/`model` and asking for `full` routing.
        let routing_mode: Option<super::provider::cognitive::ConversationRoutingMode> = parent
            .routing_mode
            .as_deref()
            .and_then(|m| serde_json::from_value(serde_json::Value::String(m.to_owned())).ok());
        let routing_pool: Option<Vec<super::types::RoutingPoolEntry>> = parent
            .routing_pool
            .as_deref()
            .and_then(|json| serde_json::from_str(json).ok());
        let pool_len = routing_pool.as_ref().map_or(0, Vec::len);
        let parent_routes = super::provider::cognitive::conversation_routes(routing_mode, pool_len);
        // A model pinned BY HAND (`set_session_model(manual = true)` ->
        // `pin_session_model`) is never moved, everywhere else in this host: the manual
        // change updates neither `routed_by` nor `routing_mode`, so the two other pin
        // tests miss it (review point 4). Its `Agent` calls inherit like any other pin's.
        let pinned = super::provider::cognitive::engine_is_pinned(
            parent.routed_by.as_deref(),
            pool_len,
            self.model_is_pinned(parent_session_id).await,
        );
        let route_child = super::provider::cognitive::take_over_subagent_tool(
            parent_routes,
            pinned,
            self.cognitive_routing.is_some(),
        );
        tracing::debug!(
            route_child = route_child,
            parent_routes = parent_routes,
            pinned = pinned,
            cognitive = self.cognitive_routing.is_some(),
            "sub-agent opening: auto routing"
        );
        let mut child = ChatRequest {
            message: crate::refs::compose::inert(&request.prompt),
            cwd: parent.cwd.clone(),
            model: if route_child {
                None
            } else {
                Some(parent.model.clone())
            },
            provider: if route_child {
                None
            } else {
                parent.provider_id.clone()
            },
            add_dirs: parent.add_dirs.clone(),
            user_claims,
            task_context: request.description.clone(),
            subagent_type: request.subagent_type.clone(),
            route_auto: route_child,
            // Never wider than the parent (a read-only parent has no `Agent`; defence in depth).
            access: Some(parent.access),
            // ISO2: tool restrictions from the agent type.
            denied_tools,
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
        // The opening must not be droppable (module documentation): it is done in a task of
        // its own, and this future only awaits that task. `alive` is dropped with this
        // future — whether it ends normally (the task is over by then) or is cut — and that
        // is how the task learns nobody awaits the child any more.
        // CACHE-POOL: the warm session this delegation may RESUME instead of opening a new
        // child, and the facts it must match to be allowed to (session_pool). Claimed here,
        // before the task is spawned, so a delegation that never runs does not hold it.
        let reuse = self.reuse_for(&child, parent.owner.as_deref(), parent_session_id);
        let (alive, cancelled) = tokio::sync::oneshot::channel::<std::convert::Infallible>();
        let opening = {
            let manager = Arc::clone(&self);
            let parent = parent_session_id.to_string();
            let tool_use_id = request.tool_call_id.clone();
            tokio::spawn(async move {
                manager
                    .open_wait_close(child, cancelled, &parent, &tool_use_id, slot, reuse)
                    .await
            })
        };
        // ── Background (ISO2) ──────────────────────────────────────────────
        // The child runs in a detached task: its answer, or its failure, is
        // injected into the parent's next turn as a `task_notification` hint.
        if request.run_in_background {
            let parent_id = parent_session_id.to_string();
            let tool_call_id = request.tool_call_id.clone();
            let mgr = Arc::clone(&self);
            tokio::spawn(async move {
                // Hold the sender so `cancelled` never fires: the child runs
                // to completion, not bounded by the parent's turn.
                let _keep = alive;
                let result = opening
                    .await
                    .unwrap_or_else(|e| Err(format!("the sub-agent ended: {e}")));
                let (status, summary) = match &result {
                    Ok(answer) => ("completed", bounded(answer.clone())),
                    Err(err) => ("failed", err.clone()),
                };
                let notification = format!(
                    "<task-notification tool_use_id=\"{tool_call_id}\" \
                     status=\"{status}\">\n{summary}\n</task-notification>",
                );
                if let Err(e) = mgr.inject_hint(&parent_id, &notification).await {
                    tracing::warn!(
                        parent = %parent_id,
                        tool_use_id = %tool_call_id,
                        error = %e,
                        "could not inject background sub-agent completion",
                    );
                }
            });
            return Ok(format!(
                "Async agent launched successfully (tool_use_id: {}): {}",
                request.tool_call_id,
                request
                    .description
                    .as_deref()
                    .unwrap_or("background sub-agent"),
            ));
        }
        // ── Foreground (unchanged) ─────────────────────────────────────────
        let _alive = alive;
        opening
            .await
            .unwrap_or_else(|e| Err(refused(format!("the sub-agent's opening ended: {e}"))))
    }

    /// What the warm-session pool says about one delegation (CACHE-POOL, module
    /// [`super::session_pool`]): the key and the facts the child must be served with, and
    /// the warm session it may resume — `None` when the pool has no eligible one.
    ///
    /// `None` (no pool lookup at all) when the child's engine and model are not settled
    /// before the opening: a child the cognitive router opens (`route_child`) has neither,
    /// and the pool is keyed on them. The facts are the ones the delegation needs, computed
    /// the way the opening computes them — never read from the warm session.
    fn reuse_for(
        &self,
        child: &ChatRequest,
        parent_owner: Option<&str>,
        parent_session_id: &str,
    ) -> Option<Reuse> {
        let provider = child.provider.clone().filter(|p| !p.trim().is_empty())?;
        let model = child.model.clone().filter(|m| !m.trim().is_empty())?;
        let owner = child
            .user_claims
            .as_ref()
            .map(|claims| claims.sub.clone())
            .or_else(|| parent_owner.map(str::to_owned));
        let key = WarmKey {
            provider,
            model,
            parent_session_id: parent_session_id.to_owned(),
            project: child.project_slug.clone(),
            owner: owner.clone(),
        };
        let mut declared = self.config.read_only_mcp_tools.clone();
        declared.sort();
        let facts = WarmFacts {
            cwd: child.cwd.clone(),
            access: child.access.unwrap_or_default(),
            permission_mode: child.permission_mode.clone(),
            add_dirs: child.add_dirs.clone().unwrap_or_default(),
            workspace: child.workspace_slug.clone(),
            consent: self.consent_fingerprint(owner.as_deref()),
            tool_policy: super::session_pool::fingerprint(
                [
                    Some(match child.access.unwrap_or_default() {
                        SessionAccess::Normal => "normal",
                        SessionAccess::ReadOnly => "read_only",
                    }),
                    child.permission_mode.as_deref(),
                ]
                .into_iter()
                .chain(declared.iter().map(|tool| Some(tool.as_str()))),
            ),
        };
        let claim = self
            .session_pool
            .claim(&key, &facts, std::time::Instant::now());
        tracing::debug!(parent = %parent_session_id, "sub-agent opening: {}", claim.summary());
        Some(Reuse {
            key,
            facts,
            warm: claim.held,
        })
    }

    /// Fingerprint of the person's lasting approvals ([`super::lasting_rules`]): a warm
    /// session is only resumed while they are exactly what they were when it was opened.
    /// Reusing a session opened before an approval was granted (or after one was revoked)
    /// would carry a consent this delegation never got — a security bug, not a saving.
    fn consent_fingerprint(&self, owner: Option<&str>) -> String {
        let rules = owner
            .and_then(|owner| self.lasting_rule_store().map(|store| store.rules(owner)))
            .unwrap_or_default();
        let mut ids: Vec<String> = rules
            .iter()
            .map(|rule| format!("{}:{}", rule.project, rule.id))
            .collect();
        ids.sort();
        super::session_pool::fingerprint(ids.iter().map(|id| Some(id.as_str())))
    }

    /// Opens the child of one sub-agent call, waits for its answer and closes it. Runs in a
    /// task of its own ([`Self::run_subagent`]) so no drop can cut the opening halfway;
    /// `cancelled` resolves when nobody awaits the call any more, and the child is then
    /// closed as soon as the opening has given its id. `slot` is the parent's reservation
    /// (module documentation): held by this task, then by its close guard, so it is given
    /// back only once the child this task owns is actually closed — on the ordinary path,
    /// on a dropped caller (P18b round 2, fix 2) and on a panic (round 3, fix E).
    ///
    /// `reuse` carries the warm session the pool offered (CACHE-POOL): the child RESUMES it
    /// instead of being opened, and is given back warm once closed. Empty, or refused by the
    /// provider, the child is opened fresh — exactly as before the pool existed.
    async fn open_wait_close(
        self: Arc<Self>,
        child: ChatRequest,
        cancelled: tokio::sync::oneshot::Receiver<std::convert::Infallible>,
        parent_session_id: &str,
        tool_use_id: &str,
        mut slot: SubagentSlot,
        reuse: Option<Reuse>,
    ) -> Result<String, String> {
        // A resumed warm session already carries the cost of its earlier turns: only what it
        // spends from here on is this call's (PYSR-3).
        let baseline = match reuse.as_ref().and_then(|reuse| reuse.warm.as_ref()) {
            Some(warm) => self.own_cost_usd(&warm.session_id).await,
            None => 0.0,
        };
        let opened = self
            .open_child(&child, reuse.as_ref(), parent_session_id, tool_use_id)
            .await;
        // Live (or failed): the parent's other openings now count the child itself.
        slot.opened();
        let (child_id, resumed_from) = opened?;
        tracing::info!(parent = %parent_session_id, child = %child_id, tool_use_id = %tool_use_id, resumed = resumed_from.is_some(), "sub-agent started");
        // The child exists from here: whatever ends this task, it is closed first (it stays
        // resumable and visible). Nothing between here and the end of the explicit close
        // below panics today, but nothing rules one out after this module grows — `guard`
        // is the backstop for exactly that window, the close call itself INCLUDED: it is
        // disarmed only once `close_session` has returned (P18b round 3, fix E; a second
        // close of the same child is a harmless "not found or inactive"). The guard owns the
        // slot: a panic gives it back only after the detached close, never before.
        let manager = Arc::clone(&self);
        let backstop_id = child_id.clone();
        let mut guard = ClosePanicGuard::new(move || {
            tokio::spawn(async move {
                if let Err(e) = manager.close_session(&backstop_id).await {
                    tracing::debug!(child = %backstop_id, error = %e, "sub-agent close");
                }
                drop(slot);
            });
        });
        let answer = tokio::select! {
            biased;
            _ = cancelled => Err("the sub-agent's call was cancelled".to_owned()),
            // A resumed session's thread already holds the answer of the turn it ran
            // before: only what came after `resumed_from` can be this call's answer.
            answer = self.wait_for_answer_after(&child_id, resumed_from.unwrap_or(0)) => answer,
        };
        #[cfg(test)]
        panic_injection::panic_if_injected(parent_session_id);
        self.deny_waiting_permissions(&child_id).await;
        if let Err(e) = self.close_session(&child_id).await {
            tracing::debug!(child = %child_id, error = %e, "sub-agent close");
        }
        // What the child spent is the parent's too, however the call ended (PYSR-3).
        self.charge_children_cost(parent_session_id, &child_id, baseline)
            .await;
        // Closed: the backstop is not needed any more, and the session becomes the
        // conversation's warm prefix for its next delegation of the same engine and model.
        if let Some(reuse) = reuse {
            reuse.finish(
                &self.session_pool,
                &child_id,
                answer.is_ok(),
                std::time::Instant::now(),
            );
        }
        // Dropping the guard gives the slot back — after the close, as on every other path.
        guard.disarm();
        drop(guard);
        answer
    }

    /// Opens the child a delegation runs in: the warm session `reuse` offers, RESUMED, or a
    /// fresh session. Returns the child's id and, when it was resumed, the last event its
    /// thread already held (the floor under what counts as this call's answer).
    ///
    /// A resume the provider refuses does not fail the delegation: the warm entry is dropped
    /// and the child is opened fresh, exactly as before the pool existed. The pool is an
    /// optimisation, never a new failure mode.
    async fn open_child(
        &self,
        child: &ChatRequest,
        reuse: Option<&Reuse>,
        parent_session_id: &str,
        tool_use_id: &str,
    ) -> Result<(String, Option<i64>), String> {
        if let Some(warm) = reuse.and_then(|reuse| reuse.warm.as_ref()) {
            let floor = self.last_event_seq(&warm.session_id).await;
            // A delegation's resume: checked against the parent's tombstone like a fresh child
            // (B36-RACE-RESUME: a person's resume of the same session would pass).
            match super::cancelled_parents::as_delegation(self.resume_session(
                &warm.session_id,
                &child.message,
                child.user_claims.as_ref(),
            ))
            .await
            {
                Ok(()) => {
                    tracing::info!(
                        parent = %parent_session_id,
                        child = %warm.session_id,
                        tool_use_id = %tool_use_id,
                        uses = warm.uses,
                        "sub-agent resumes a warm session"
                    );
                    return Ok((warm.session_id.clone(), Some(floor)));
                }
                Err(e) => {
                    // B36-RACE: the parent was cancelled while this delegation was opening:
                    // the warm child was refused at its activation, and a fresh one would be
                    // too. The delegation ends on the typed refusal.
                    if let Some(refused) = super::cancelled_parents::ParentCancelled::in_chain(&e) {
                        self.session_pool.abandon(&warm.session_id);
                        return Err(format!("the sub-agent could not start: {refused}"));
                    }
                    // The prefix is worthless (the provider's session is gone) or the
                    // session can no longer be reopened: never offered again.
                    tracing::info!(
                        child = %warm.session_id,
                        error = %e,
                        "sub-agent cannot resume its warm session: opening a fresh one"
                    );
                    self.session_pool.abandon(&warm.session_id);
                }
            }
        }
        let opened = self
            .create_session(child)
            .await
            .map_err(|e| format!("the sub-agent could not start: {e:#}"))?;
        Ok((opened.session_id, None))
    }

    /// The OWN cost of a session (`0.0` when unknown).
    async fn own_cost_usd(&self, sid: &str) -> f64 {
        let Ok(id) = Uuid::parse_str(sid) else {
            return 0.0;
        };
        match self.graph.get_chat_session(id).await {
            Ok(Some(session)) => session
                .total_cost_usd
                .filter(|c| c.is_finite())
                .unwrap_or(0.0),
            _ => 0.0,
        }
    }

    /// Adds what the child spent since `baseline` to the CHILDREN spend of its parent. The
    /// parent's own `total_cost_usd` is left alone: the routing decision of the parent is closed
    /// on its own cost, the child's being carried by the child's own decision (AUTO-SUB) —
    /// otherwise the two arms would both be charged.
    async fn charge_children_cost(&self, parent_session_id: &str, child_id: &str, baseline: f64) {
        let spent = (self.own_cost_usd(child_id).await - baseline).max(0.0);
        if spent <= 0.0 {
            return;
        }
        let Ok(parent) = Uuid::parse_str(parent_session_id) else {
            return;
        };
        if let Err(e) = self
            .graph
            .add_chat_session_children_cost(parent, spent)
            .await
        {
            tracing::warn!(parent = %parent_session_id, child = %child_id, error = %e, "the sub-agent's cost could not be added to its parent");
        }
    }

    /// The cost of a session WITH its sub-agents: its own plus what its children spent
    /// (PYSR-3). `total_cost_usd` stays the own cost.
    pub async fn aggregated_cost_usd(&self, session_id: &str) -> Option<f64> {
        let id = Uuid::parse_str(session_id).ok()?;
        let own = self
            .graph
            .get_chat_session(id)
            .await
            .ok()??
            .total_cost_usd
            .unwrap_or(0.0);
        let children = self
            .graph
            .get_chat_session_children_cost(id)
            .await
            .unwrap_or(0.0);
        Some(own + children)
    }

    /// The last event sequence the thread of `sid` holds (0 when it has none): the floor a
    /// resumed child's answer is read from.
    async fn last_event_seq(&self, sid: &str) -> i64 {
        match Uuid::parse_str(sid) {
            Ok(uuid) => self
                .graph
                .get_latest_chat_event_seq(uuid)
                .await
                .unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Answers `deny` to every permission request the child `sid` still waits on, through
    /// the path a user's answer takes ([`ChatManager::route_permission_response`]): its
    /// `permission_decision` lands on the child's thread, so a request nobody will ever
    /// answer (the call failed on it, fix B) is no orphan in the cockpit of the child's
    /// owner for 30 days once the child is closed (review of #724, point 1). The requests
    /// are the live session's, and those its stored thread shows undecided.
    ///
    /// Every request named ends DECIDED on the thread (lot review of #721): when the child
    /// is not live on this server (the restart recovery closing it), or when its provider
    /// refuses the deny (a request it no longer knows), the refusal is written on the
    /// thread by this server, with a warning naming why — the child is closed right after,
    /// so nothing of it can run the call any more. It used to write nothing then, and the
    /// request stayed pending in the owner's attention.
    pub(crate) async fn deny_waiting_permissions(&self, sid: &str) {
        let live = self.agent_runtime.get(sid).await;
        let mut ids: Vec<String> = match &live {
            Some(handle) => handle.waiting_permission_ids(),
            None => Vec::new(),
        };
        let persisted = self.persisted_events(sid).await;
        let mut decided = std::collections::HashSet::new();
        for event in &persisted {
            if let ChatEvent::PermissionDecision { id, .. } = event {
                decided.insert(id.clone());
            }
        }
        for event in &persisted {
            if let ChatEvent::PermissionRequest { id, .. } = event {
                if !decided.contains(id) && !ids.contains(id) {
                    ids.push(id.clone());
                }
            }
        }
        let routable = live.is_some() || self.is_session_active(sid).await;
        for id in ids {
            if routable {
                match self
                    .route_permission_response(
                        sid,
                        &id,
                        false,
                        super::types::PermissionAnswerScope::Once,
                        false,
                    )
                    .await
                {
                    Ok(_) => continue,
                    Err(e) => tracing::warn!(
                        child = %sid, request = %id, error = %e,
                        "the provider did not take the sub-agent's deny: the refusal is written on its thread, and it is closed"
                    ),
                }
            } else {
                tracing::info!(
                    child = %sid, request = %id,
                    "sub-agent not live here: its undecided permission request is refused on its thread"
                );
            }
            self.deny_on_thread_if_undecided(sid, &id, live.as_deref())
                .await;
        }
    }

    /// The fallback refusal of [`Self::deny_waiting_permissions`]: writes
    /// `permission_decision {allow: false}` for `id` on the thread of `sid` ONLY if the
    /// request is still undecided right now (review nexus-41 of #727, finding 2, and the
    /// second review's finding 2). Between the collection of the requests and this write,
    /// an allow may have won the race — the automatic answer of a grant, or the user's
    /// answer whose decision is stored only after its lasting rule is written — and that
    /// is the very reason the provider refused the deny: a refusal written then would
    /// leave two contradictory decisions on the thread for a call that ran. Undecided
    /// means: the live session (when there is one) neither answers it itself nor had an
    /// answer to it taken (`answers_itself`, marked the moment the provider takes it), and
    /// the thread, read again, holds no decision for it. Returns whether it was written.
    pub(super) async fn deny_on_thread_if_undecided(
        &self,
        sid: &str,
        id: &str,
        live: Option<&super::agent_runtime::AgentSessionHandle>,
    ) -> bool {
        if live.is_some_and(|handle| handle.answers_itself(id)) {
            tracing::info!(
                child = %sid, request = %id,
                "the sub-agent's request was answered meanwhile: no refusal written"
            );
            return false;
        }
        let decided = self.persisted_events(sid).await.iter().any(|event| {
            matches!(event, ChatEvent::PermissionDecision { id: decided, .. } if decided == id)
        });
        if decided {
            tracing::info!(
                child = %sid, request = %id,
                "the sub-agent's request is decided on its thread already: no refusal written"
            );
            return false;
        }
        self.emit_on_session(
            sid,
            ChatEvent::PermissionDecision {
                id: id.to_owned(),
                allow: false,
                scope: None,
                rule: None,
            },
        )
        .await;
        true
    }

    /// The first `result` of the session `sid` after `floor`, live or already persisted.
    /// `floor` is the sequence the thread had reached when the child was opened: a
    /// delegation that RESUMED a warm session (CACHE-POOL) shares its thread with the turn
    /// that session ran before, and without the floor the previous `result` would be read
    /// as this call's answer the instant the wait began.
    async fn wait_for_answer_after(&self, sid: &str, floor: i64) -> Result<String, String> {
        let rx = self
            .subscribe(sid)
            .await
            .map_err(|e| format!("the sub-agent cannot be followed: {e:#}"))?;
        let initial = self.agent_runtime.get(sid).await;
        let answers_itself = live_answers_itself(Arc::clone(&self.agent_runtime), sid, initial);
        self.answer_from_floor(sid, rx, floor, &answers_itself)
            .await
    }

    /// [`Self::wait_for_answer_after`] on a subscription already taken — the seam the tests
    /// of a lagged subscription and of a permission request drive. `answers_itself` says
    /// whether the backend answers a request of the child itself (a grant or a lasting rule
    /// covers it).
    #[cfg(test)]
    pub(super) async fn answer_from(
        &self,
        sid: &str,
        rx: tokio::sync::broadcast::Receiver<ChatEvent>,
        answers_itself: &(dyn Fn(&str) -> bool + Send + Sync),
    ) -> Result<String, String> {
        self.answer_from_floor(sid, rx, 0, answers_itself).await
    }

    /// [`Self::answer_from`] with the `floor` a resumed child starts from: nothing the
    /// thread held at or before `floor` is this call's answer, and requests stored then are
    /// not this call's requests either.
    async fn answer_from_floor(
        &self,
        sid: &str,
        mut rx: tokio::sync::broadcast::Receiver<ChatEvent>,
        floor: i64,
        answers_itself: &(dyn Fn(&str) -> bool + Send + Sync),
    ) -> Result<String, String> {
        let mut watch = Watch::default();
        // A turn that ended before the subscription is in the store.
        if let Some(end) = self.read_back(sid, &mut watch, floor).await {
            return end;
        }
        // Once a `Lagged` was seen, the stream alone cannot be trusted any more: the thread
        // is re-read on every tick until it shows the end (P18b round 3, fix C) — no fixed
        // number of attempts, since the child's events may land on its thread long after
        // the drop (a verbose child is exactly the one that lags). The tick spaces those
        // reads one second apart however often the stream lags, so a chatty child does not
        // make the parent hammer the store; it also wakes the loop while a request the
        // backend answers itself waits for that answer (its answer may fail: then nobody
        // answers it).
        let mut tick = tokio::time::interval(READ_BACK_EVERY);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut lagged = false;
        loop {
            if let Some(end) = watch.unanswered_ask(answers_itself) {
                return end;
            }
            tokio::select! {
                received = rx.recv() => match received {
                    Ok(event) => {
                        if let Some(end) = watch.observe(event) {
                            return end;
                        }
                    }
                    // Events were dropped, and the child's `result` may have been among
                    // them: waiting on the stream alone could last until the parent's
                    // 30-minute turn timeout. The thread is the truth (P18b, fix 4).
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(dropped)) => {
                        if !lagged {
                            tracing::warn!(child = %sid, dropped, "sub-agent events dropped: its thread is read back until it ends");
                        }
                        lagged = true;
                    }
                    // The stream is over: what it did not carry, the thread may still hold.
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        return self.read_back(sid, &mut watch, floor).await.unwrap_or_else(|| {
                            Err("the sub-agent's session ended before it answered".into())
                        });
                    }
                },
                _ = tick.tick(), if lagged || watch.is_asking() => {
                    if lagged {
                        if let Some(end) = self.read_back(sid, &mut watch, floor).await {
                            return end;
                        }
                    }
                }
            }
        }
    }

    /// The end of `sid` as its STORED thread tells it — its answer, or its close — and
    /// what it still asks for there, into `watch` (a decision the stream dropped is seen
    /// here). `None` while the thread shows no end. Nothing at or before `floor` counts: it
    /// belongs to the turn a resumed warm session ran before this call (CACHE-POOL).
    async fn read_back(
        &self,
        sid: &str,
        watch: &mut Watch,
        floor: i64,
    ) -> Option<Result<String, String>> {
        let persisted = self.persisted_events_after(sid, floor).await;
        watch.reconcile(&persisted);
        if let Some(answer) = answer_of(&persisted) {
            return Some(answer);
        }
        persisted
            .iter()
            .any(|e| matches!(e, ChatEvent::SessionClosed { .. }))
            .then(|| Err(CLOSED_BEFORE_ANSWER.to_owned()))
    }

    /// The answer of `sid` as its STORED thread tells it, when its `result` is already
    /// there.
    #[cfg(test)]
    async fn persisted_answer(&self, sid: &str) -> Option<Result<String, String>> {
        answer_of(&self.persisted_events(sid).await)
    }

    /// The most recent events of `sid`'s STORED thread. Reads the TAIL of the thread, not
    /// the head (P18b round 2, fix 5): a long child's `result` can land after more than
    /// [`PERSISTED_READ_WINDOW`] events, and `get_chat_events` only reads forward from
    /// `after_seq` — starting at 0 would never reach it. `floor` raises that starting point:
    /// a delegation that resumed a warm session only reads what its own turn wrote
    /// (CACHE-POOL).
    async fn persisted_events(&self, sid: &str) -> Vec<ChatEvent> {
        self.persisted_events_after(sid, 0).await
    }

    /// [`Self::persisted_events`], ignoring everything at or before `floor`.
    async fn persisted_events_after(&self, sid: &str, floor: i64) -> Vec<ChatEvent> {
        let Ok(uuid) = Uuid::parse_str(sid) else {
            return Vec::new();
        };
        let latest_seq = self
            .graph
            .get_latest_chat_event_seq(uuid)
            .await
            .unwrap_or(0);
        let after_seq = (latest_seq - PERSISTED_READ_WINDOW).max(0).max(floor);
        self.graph
            .get_chat_events(uuid, after_seq, PERSISTED_READ_WINDOW)
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|r| serde_json::from_str(&r.data).ok())
            .collect()
    }
}

/// What the parent's wait on one child holds: what [`answer_of`] will read, and the
/// permission requests of the child still waiting for an answer (P18b round 3, fix B).
#[derive(Default)]
struct Watch {
    seen: Vec<ChatEvent>,
    /// Request id → (tool, order in which the parent first saw it waiting).
    asks: std::collections::HashMap<String, (String, usize)>,
    /// How many requests were seen: the order of the next one.
    asked: usize,
}

impl Watch {
    /// Takes one live event; the end of the call when this event ends it.
    fn observe(&mut self, event: ChatEvent) -> Option<Result<String, String>> {
        match &event {
            ChatEvent::PermissionRequest { id, tool, .. } => self.ask(id, tool),
            ChatEvent::PermissionDecision { id, .. } => {
                self.asks.remove(id);
            }
            _ => {}
        }
        let done = is_result(&event);
        let asked = is_question(&event);
        let closed = matches!(event, ChatEvent::SessionClosed { .. });
        keep_for_answer(&mut self.seen, event);
        if done || asked {
            // A question fails the call at once: nobody would ever answer it.
            return Some(answer_of(&self.seen).unwrap_or_else(|| Ok(String::new())));
        }
        if closed {
            // A child can also close without this event reaching the parent: when it was
            // dropped, the read-back after the `Lagged` sees the close on the thread; when
            // the stream just ends, the read-back on `Closed` does. A close the stream never
            // carried while it stays open (a clone of `events_tx` alive) and never lags is
            // NOT caught: the parent then waits until its turn timeout.
            return Some(Err(CLOSED_BEFORE_ANSWER.into()));
        }
        None
    }

    /// Brings the waiting requests in line with the stored thread: what it shows decided
    /// is not waiting any more, what it shows asked and undecided is.
    ///
    /// Contract with `answers_itself` (review nexus-41 of #727, nit 3; second review,
    /// finding 3): a request the live session saw asked and whose answer it took counts
    /// as answered even before its decision is stored; one it never saw — asked before a
    /// resume under the same id — does NOT: nobody here will answer it, so
    /// [`Self::unanswered_ask`] fails the call on it at once, as on any undecided request.
    fn reconcile(&mut self, persisted: &[ChatEvent]) {
        for event in persisted {
            if let ChatEvent::PermissionRequest { id, tool, .. } = event {
                self.ask(id, tool);
            }
        }
        for event in persisted {
            if let ChatEvent::PermissionDecision { id, .. } = event {
                self.asks.remove(id);
            }
        }
    }

    /// A request of the child waits from now on (a second sight of it changes nothing).
    fn ask(&mut self, id: &str, tool: &str) {
        if !self.asks.contains_key(id) {
            self.asks
                .insert(id.to_owned(), (tool.to_owned(), self.asked));
            self.asked += 1;
        }
    }

    fn is_asking(&self) -> bool {
        !self.asks.is_empty()
    }

    /// The failure of the call when a permission request of the child waits that the
    /// backend does not answer itself (`answers_itself`: no grant, no lasting rule covers
    /// it, or its automatic answer failed): a sub-agent does not wait for a person, and an
    /// unanswered request would hold the parent until its turn timeout. Decided at once, with no grace
    /// (review of #724): the backend knows whether it answers a request before the request
    /// is shown. Several such requests: the first one seen names the tool.
    fn unanswered_ask(
        &self,
        answers_itself: &(dyn Fn(&str) -> bool + Send + Sync),
    ) -> Option<Result<String, String>> {
        let (tool, _) = self
            .asks
            .iter()
            .filter(|(id, _)| !answers_itself(id))
            .map(|(_, ask)| ask)
            .min_by_key(|(_, order)| *order)?;
        Some(Err(format!(
            "the sub-agent needed a permission for `{tool}`, and a sub-agent cannot ask for one \
             (nobody answers in its session): do that step yourself, or give it a task that \
             needs no permission"
        )))
    }
}

/// Whether the backend answers a request of the child `sid` itself, read from the child's
/// LIVE session each time it is asked (`AgentSessionHandle::answers_itself`, decided before
/// the request is broadcast). The handle used to be captured once, when the wait began
/// (lot review of #721, nit (a) of the follow-up review of #724): a child whose session was
/// replaced in the runtime meanwhile (re-adopted under the same id: a resume) was judged
/// on the OLD handle, whose sets never see the new session's requests — a covered request
/// then failed the call, or one still waiting looked answered. The runtime is read without
/// waiting; at the instant it is being written, the handle last seen answers.
fn live_answers_itself(
    runtime: Arc<super::agent_runtime::AgentRuntime>,
    sid: &str,
    initial: Option<Arc<super::agent_runtime::AgentSessionHandle>>,
) -> impl Fn(&str) -> bool + Send + Sync {
    let sid = sid.to_owned();
    let last = std::sync::Mutex::new(initial);
    move |id: &str| {
        let mut last = last.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(now) = runtime.try_get(&sid) {
            *last = now;
        }
        last.as_ref().is_some_and(|h| h.answers_itself(id))
    }
}

/// What the call answers when the child closed before its `result`.
const CLOSED_BEFORE_ANSWER: &str = "the sub-agent's session was closed before it answered";

/// How often the parent re-reads the thread of a child whose stream lagged, and checks the
/// permission requests of the child the backend answers itself (P18b round 3, fix C).
const READ_BACK_EVERY: Duration = Duration::from_secs(1);

/// How many of a child's most recent persisted events `ChatManager::persisted_events`
/// reads back (P18b round 2, fix 5): enough for `answer_of` to find a `result` that is
/// not buried under a very chatty child's thread, without reading the whole thing.
const PERSISTED_READ_WINDOW: i64 = 5_000;

/// A panic injected into [`ChatManager::open_wait_close`] between the child's answer and
/// its explicit close, keyed by parent session id (unique per test): the proof that the
/// close guard is wired into the real task, not just that the struct works (P18b round 3,
/// fix E). That point is the one the guard did NOT cover before: it was disarmed there.
#[cfg(test)]
pub(crate) mod panic_injection {
    static BEFORE_CLOSE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    /// The next sub-agent of `parent` panics right before its child's `close_session`.
    pub(crate) fn before_close(parent: &str) {
        BEFORE_CLOSE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(parent.to_string());
    }

    pub(super) fn panic_if_injected(parent: &str) {
        let hit = {
            let mut armed = BEFORE_CLOSE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match armed.iter().position(|p| p == parent) {
                Some(i) => {
                    armed.remove(i);
                    true
                }
                None => false,
            }
        };
        if hit {
            panic!("injected panic before the sub-agent's close (test)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::traits::GraphStore;
    use crate::neo4j::traits::*;

    fn bare_manager() -> (Arc<crate::neo4j::mock::MockGraphStore>, Arc<ChatManager>) {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let manager = Arc::new(ChatManager::new_without_memory(
            dyn_graph,
            crate::test_helpers::mock_app_state().meili,
            Default::default(),
        ));
        (graph, manager)
    }

    /// PYSR-3 — what a sub-agent spent shows in its parent's AGGREGATE (own + children), and
    /// the parent's OWN cost is left alone: its routing decision is closed on the own cost, the
    /// child's being carried by the child's own decision (AUTO-SUB), so no arm is charged twice.
    /// Red before the fix: the child's cost was nowhere on the parent.
    #[tokio::test]
    async fn a_sub_agents_cost_is_in_the_parents_aggregate_not_in_its_own_cost() {
        let (graph, manager) = bare_manager();
        let mut parent = crate::test_helpers::test_chat_session(None);
        parent.total_cost_usd = Some(1.0);
        let mut child = crate::test_helpers::test_chat_session(None);
        child.total_cost_usd = Some(2.0);
        graph.create_chat_session(&parent).await.unwrap();
        graph.create_chat_session(&child).await.unwrap();
        let (parent_id, child_id) = (parent.id.to_string(), child.id.to_string());

        manager
            .charge_children_cost(&parent_id, &child_id, 0.0)
            .await;

        assert_eq!(manager.aggregated_cost_usd(&parent_id).await, Some(3.0));
        let own = graph.get_chat_session(parent.id).await.unwrap().unwrap();
        assert_eq!(
            own.total_cost_usd,
            Some(1.0),
            "the own cost is what the routing decision is closed on"
        );
    }

    /// PYSR-3 — a resumed warm child carries the cost of its earlier turns: only what it spent
    /// since the baseline is this call's; two calls add up.
    #[tokio::test]
    async fn a_resumed_children_earlier_turns_are_not_charged_again() {
        let (graph, manager) = bare_manager();
        let parent = crate::test_helpers::test_chat_session(None);
        let mut child = crate::test_helpers::test_chat_session(None);
        child.total_cost_usd = Some(5.0);
        graph.create_chat_session(&parent).await.unwrap();
        graph.create_chat_session(&child).await.unwrap();
        let (parent_id, child_id) = (parent.id.to_string(), child.id.to_string());
        // The warm child had spent 4.0 before this call; it is at 5.0 after it.
        manager
            .charge_children_cost(&parent_id, &child_id, 4.0)
            .await;
        assert_eq!(manager.aggregated_cost_usd(&parent_id).await, Some(1.0));
        // A free call adds nothing.
        manager
            .charge_children_cost(&parent_id, &child_id, 5.0)
            .await;
        assert_eq!(manager.aggregated_cost_usd(&parent_id).await, Some(1.0));
    }

    fn said(t: &str) -> ChatEvent {
        ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        }
    }

    fn asked(id: &str) -> ChatEvent {
        ChatEvent::PermissionRequest {
            id: id.into(),
            tool: "Bash".into(),
            input: serde_json::json!({"command": "rm -rf build"}),
            parent_tool_use_id: None,
            category: None,
            canonical: None,
            tool_use_id: None,
            always: None,
            consent: None,
        }
    }

    /// P18b round 3, fix C: the stream lags, and the child's `result` reaches its thread
    /// only 6 s later — past the ~4–5 s the round-2 poll gave it. The parent still gets the
    /// answer, about when it is stored, not at its turn timeout. Red before the fix: after
    /// its 5 reads the poll went back to `recv()` for good, and the stream (still open,
    /// quiet) never said anything again.
    #[tokio::test(start_paused = true)]
    async fn a_result_stored_long_after_a_lag_still_reaches_the_parent() {
        let (graph, manager) = bare_manager();
        let uuid = Uuid::new_v4();
        let sid = uuid.to_string();
        let (tx, rx) = tokio::sync::broadcast::channel::<ChatEvent>(2);
        tx.send(said("working")).unwrap();
        let started = tokio::time::Instant::now();
        let waiting = {
            let manager = manager.clone();
            let sid = sid.clone();
            tokio::spawn(async move { manager.answer_from(&sid, rx, &|_| false).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Overrun: eight events into two slots, without yielding to the reader.
        for i in 0..8 {
            tx.send(said(&format!("chatter {i}"))).unwrap();
        }
        tokio::time::sleep(Duration::from_secs(6)).await;
        assert!(!waiting.is_finished(), "nothing stored yet: it still waits");
        graph
            .store_chat_events(
                uuid,
                vec![crate::neo4j::models::ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: 1,
                    event_type: "result".into(),
                    data: serde_json::json!({
                        "type": "result", "session_id": sid, "duration_ms": 1,
                        "subtype": "success", "is_error": false, "result_text": "LATE-ANSWER",
                    })
                    .to_string(),
                    created_at: chrono::Utc::now(),
                }],
            )
            .await
            .unwrap();
        // The stream stays open and quiet: only the read-back can end the wait.
        let out = tokio::time::timeout(Duration::from_secs(60), waiting)
            .await
            .expect("the parent must not wait for its turn timeout")
            .unwrap();
        assert_eq!(out, Ok("LATE-ANSWER".to_string()));
        assert!(
            started.elapsed() < Duration::from_secs(9),
            "answered about when stored: {:?}",
            started.elapsed()
        );
        drop(tx);
    }

    /// P18b round 3, fix B, made deterministic by the review of #724: a permission request
    /// of the child the backend does not answer itself fails the call AT ONCE, not after a
    /// grace nor at the parent's turn timeout; one it answers itself (a grant or a lasting
    /// rule covers it) never fails it, however late its `permission_decision` comes —
    /// unless that automatic answer fails, and then the request fails the call too. Red
    /// before: an uncovered request waited 3 s, and a covered one whose answer took longer
    /// than that failed the call.
    #[tokio::test(start_paused = true)]
    async fn an_uncovered_permission_fails_the_call_at_once_and_a_covered_one_never() {
        let (_graph, manager) = bare_manager();
        let sid = Uuid::new_v4().to_string();

        // Uncovered: failed at once (paused time: no tick, no grace elapsed).
        let (tx, rx) = tokio::sync::broadcast::channel::<ChatEvent>(16);
        let started = tokio::time::Instant::now();
        let waiting = {
            let manager = manager.clone();
            let sid = sid.clone();
            tokio::spawn(async move { manager.answer_from(&sid, rx, &|_| false).await })
        };
        tx.send(asked("p1")).unwrap();
        let out = tokio::time::timeout(Duration::from_secs(60), waiting)
            .await
            .expect("an unanswered permission must not hold the parent")
            .unwrap();
        assert!(
            out.as_ref()
                .is_err_and(|e| e.contains("permission") && e.contains("Bash")),
            "{out:?}"
        );
        assert!(
            started.elapsed() < READ_BACK_EVERY,
            "{:?}",
            started.elapsed()
        );

        // Covered: its automatic answer is late (30 s, ten times the old grace); the call
        // waits, then the child goes on and answers.
        let covered = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let (tx, rx) = tokio::sync::broadcast::channel::<ChatEvent>(16);
        let waiting = {
            let manager = manager.clone();
            let sid = sid.clone();
            let covered = covered.clone();
            tokio::spawn(async move {
                let answers_itself =
                    move |id: &str| id == "p2" && covered.load(std::sync::atomic::Ordering::SeqCst);
                manager.answer_from(&sid, rx, &answers_itself).await
            })
        };
        tx.send(asked("p2")).unwrap();
        tokio::time::sleep(Duration::from_secs(30)).await;
        assert!(!waiting.is_finished(), "a covered request does not fail it");
        tx.send(ChatEvent::PermissionDecision {
            id: "p2".into(),
            allow: true,
            scope: None,
            rule: None,
        })
        .unwrap();
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert!(!waiting.is_finished(), "a decided request does not fail it");
        tx.send(ChatEvent::Result {
            session_id: sid.clone(),
            duration_ms: 1,
            cost_usd: None,
            subtype: "success".into(),
            is_error: false,
            num_turns: None,
            result_text: Some("DONE".into()),
            cost: None,
            usage: None,
            model: None,
            stop_reason: None,
            error_code: None,
            error_message: None,
        })
        .unwrap();
        assert_eq!(waiting.await.unwrap(), Ok("DONE".to_string()));

        // Covered, but its automatic answer failed: nobody answers it any more.
        let (tx, rx) = tokio::sync::broadcast::channel::<ChatEvent>(16);
        let waiting = {
            let manager = manager.clone();
            let sid = sid.clone();
            let covered = covered.clone();
            tokio::spawn(async move {
                let answers_itself =
                    move |id: &str| id == "p2" && covered.load(std::sync::atomic::Ordering::SeqCst);
                manager.answer_from(&sid, rx, &answers_itself).await
            })
        };
        tx.send(asked("p2")).unwrap();
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert!(!waiting.is_finished());
        covered.store(false, std::sync::atomic::Ordering::SeqCst);
        let out = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("a request whose automatic answer failed fails the call")
            .unwrap();
        assert!(out.is_err_and(|e| e.contains("Bash")));
        drop(tx);
    }

    /// Lot review of #721 (nit (a) of the follow-up review of #724): what the parent asks
    /// about a child's request is read from the child's LIVE session, not from the handle
    /// seen when the wait began. The child is re-adopted under the same id (a resume); its
    /// new session holds a grant and answers a covered request itself: the parent must not
    /// see it as unanswered. Red before: the handle captured at the start never saw the
    /// request, so the call failed with "needed a permission".
    #[tokio::test]
    async fn whether_a_child_answers_itself_is_read_from_its_live_session() {
        use super::super::agent_runtime::{fake::FakeProvider, AgentRuntime};
        use super::super::types::PermissionAnswerScope;
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = Arc::new(AgentRuntime::new(graph));
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes = vec![
            nexus_claude::agent::PermissionScope::Once,
            nexus_claude::agent::PermissionScope::Session,
        ];
        let adopt = || {
            runtime.adopt(
                "child",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
        };
        let write = |id: &str| ChatEvent::PermissionRequest {
            id: id.to_string(),
            tool: "mcp__nexus__Write".into(),
            input: serde_json::json!({ "file_path": "a.txt", "content": "x" }),
            parent_tool_use_id: None,
            category: Some("edit".into()),
            canonical: Some("Write".into()),
            tool_use_id: None,
            always: None,
            consent: None,
        };
        let first = adopt().await;
        let answers_itself = live_answers_itself(Arc::clone(&runtime), "child", Some(first));
        let second = adopt().await;
        second.emit(write("w0")).await;
        assert!(!answers_itself("w0"), "the live session waits on its user");
        second
            .answer_permission_scoped("w0", true, PermissionAnswerScope::Session)
            .await
            .unwrap();
        second.emit(write("w1")).await;
        assert!(
            answers_itself("w1"),
            "the live session answered it itself (its grant covers it)"
        );
        second.emit(asked("bash-1")).await;
        assert!(!answers_itself("bash-1"), "nothing covers it: it waits");
        // The child is gone from the runtime: nothing answers for it.
        let _ = runtime.close("child").await;
        assert!(!answers_itself("w1"));
    }

    /// P18b round 2, fix 3: a guard still armed when it drops runs its backstop exactly
    /// once. Forcing a real panic through the whole `create_session` → `close_session`
    /// async chain is not clean to set up, so this drives the guard's own `Drop` directly —
    /// the seam `open_wait_close` relies on.
    #[test]
    fn an_armed_guard_runs_its_backstop_on_drop() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        {
            let _guard = ClosePanicGuard::new(move || {
                tx.send(()).unwrap();
            });
            // Dropped here, still armed: the backstop must run.
        }
        rx.try_recv().expect("the backstop ran on drop");
    }

    /// The ordinary path (`disarm` before the guard drops, exactly what `open_wait_close`
    /// does right before its own explicit `close_session`) must not ALSO run the backstop:
    /// the child would be closed twice.
    #[test]
    fn a_disarmed_guard_does_not_run_its_backstop() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        {
            let mut guard = ClosePanicGuard::new(move || {
                tx.send(()).unwrap();
            });
            guard.disarm();
        }
        assert!(
            rx.try_recv().is_err(),
            "a disarmed guard must not run its backstop"
        );
    }

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
            error_code: None,
            error_message: None,
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

    fn ok_result(text: &str) -> ChatEvent {
        ChatEvent::Result {
            session_id: "s".into(),
            duration_ms: 1,
            cost_usd: None,
            subtype: "success".into(),
            is_error: false,
            num_turns: None,
            result_text: Some(text.to_owned()),
            cost: None,
            usage: None,
            model: None,
            stop_reason: None,
            error_code: None,
            error_message: None,
        }
    }

    /// N3: a sub-agent that copies a whole file does not fill its parent's context.
    #[test]
    fn a_long_answer_is_cut_and_says_so() {
        let long = "x".repeat(1024 * 1024);
        let answer = answer_of(&[ok_result(&long)]).unwrap().unwrap();
        assert!(
            answer.chars().count() < MAX_ANSWER_CHARS + 100,
            "{}",
            answer.len()
        );
        assert!(answer.ends_with(&format!(
            "[… {} characters of the sub-agent's answer left out]",
            1024 * 1024 - MAX_ANSWER_CHARS
        )));
        let short = answer_of(&[ok_result("short")]).unwrap().unwrap();
        assert_eq!(short, "short");
    }

    /// N2: a question a child asks reaches nobody: the call fails, saying why.
    #[test]
    fn a_child_that_asks_the_user_fails_the_call() {
        let asked = ChatEvent::AskUserQuestion {
            id: "q".into(),
            tool_call_id: "cq".into(),
            questions: serde_json::json!([{"question": "Which?"}]),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
            synthetic: Some(true),
        };
        let out = answer_of(&[asked, ok_result("asked")]).unwrap();
        assert!(out.is_err_and(|e| e.contains("cannot ask")));
    }

    /// P18b, fix 4: what the parent holds of a child's stream is bounded — at most the
    /// question, the last thing said and the `result` — and `answer_of` still reads the
    /// same answer out of it.
    #[test]
    fn what_is_kept_of_a_chatty_child_is_bounded() {
        let said = |t: &str| ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        };
        let inside = ChatEvent::AssistantText {
            content: "inside a tool call".into(),
            parent_tool_use_id: Some("toolu_1".into()),
        };
        let mut seen = Vec::new();
        for i in 0..5_000 {
            keep_for_answer(&mut seen, said(&format!("line {i}")));
            keep_for_answer(&mut seen, inside.clone());
            keep_for_answer(
                &mut seen,
                ChatEvent::ToolUse {
                    id: "t".into(),
                    tool: "Read".into(),
                    input: serde_json::json!({}),
                    parent_tool_use_id: None,
                    category: None,
                    canonical: None,
                },
            );
        }
        assert_eq!(seen.len(), 1, "only the last thing said is held: {seen:?}");
        keep_for_answer(&mut seen, ok_result("FINAL"));
        keep_for_answer(&mut seen, ok_result("A SECOND RESULT"));
        assert_eq!(seen.len(), 2);
        assert_eq!(answer_of(&seen), Some(Ok("FINAL".into())));

        // Without a result text, the fallback is the LAST thing said, not the first.
        let mut seen = Vec::new();
        keep_for_answer(&mut seen, said("first"));
        keep_for_answer(&mut seen, said("last"));
        keep_for_answer(
            &mut seen,
            ChatEvent::Result {
                session_id: "s".into(),
                duration_ms: 1,
                cost_usd: None,
                subtype: "success".into(),
                is_error: false,
                num_turns: None,
                result_text: None,
                cost: None,
                usage: None,
                model: None,
                stop_reason: None,
                error_code: None,
                error_message: None,
            },
        );
        assert_eq!(answer_of(&seen), Some(Ok("last".into())));

        // A question is kept once, whatever follows.
        let question = ChatEvent::AskUserQuestion {
            id: "q".into(),
            tool_call_id: "cq".into(),
            questions: serde_json::json!([{"question": "Which?"}]),
            input: serde_json::json!({}),
            parent_tool_use_id: None,
            synthetic: Some(true),
        };
        let mut seen = Vec::new();
        keep_for_answer(&mut seen, question.clone());
        keep_for_answer(&mut seen, question);
        keep_for_answer(&mut seen, ok_result("whatever"));
        assert_eq!(seen.len(), 2);
        assert!(answer_of(&seen)
            .unwrap()
            .is_err_and(|e| e.contains("cannot ask")));
    }

    /// P18b round 2, fix 5: a child's `result` landing well past
    /// [`PERSISTED_READ_WINDOW`] events is still found — `persisted_answer` reads the
    /// TAIL of the thread. Red before the fix: `get_chat_events(uuid, 0, 5_000)` reads
    /// forward from the head and never reaches a `result` stored at seq 6001.
    #[tokio::test]
    async fn persisted_answer_reads_the_tail_not_the_head() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let manager = ChatManager::new_without_memory(
            dyn_graph,
            crate::test_helpers::mock_app_state().meili,
            Default::default(),
        );
        let sid = Uuid::new_v4();
        let said = |t: &str| ChatEvent::AssistantText {
            content: t.into(),
            parent_tool_use_id: None,
        };
        let mut events: Vec<crate::neo4j::models::ChatEventRecord> = (1..=6_000i64)
            .map(|i| crate::neo4j::models::ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: sid,
                seq: i,
                event_type: "assistant_text".into(),
                data: serde_json::to_string(&said(&format!("line {i}"))).unwrap(),
                created_at: chrono::Utc::now(),
            })
            .collect();
        events.push(crate::neo4j::models::ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: sid,
            seq: 6_001,
            event_type: "result".into(),
            data: serde_json::json!({
                "type": "result", "session_id": sid.to_string(), "duration_ms": 1,
                "subtype": "success", "is_error": false, "result_text": "TAIL-ANSWER",
            })
            .to_string(),
            created_at: chrono::Utc::now(),
        });
        graph.store_chat_events(sid, events).await.unwrap();
        let answer = manager.persisted_answer(&sid.to_string()).await;
        assert_eq!(answer, Some(Ok("TAIL-ANSWER".to_string())));
    }

    // ---------------------------------------------------------------------
    // CACHE-POOL (session_pool): what the delegation path asks the pool, and
    // what it does with the answer. The token counts themselves need the real
    // CLI; what is proven here is the MECHANISM and the eligibility rules.
    // ---------------------------------------------------------------------

    /// A delegation as `run_subagent` builds one, pinned to one engine and model.
    fn delegation(project: &str, model: &str) -> ChatRequest {
        ChatRequest {
            message: "summarise src/chat/subagents.rs".into(),
            cwd: "/work/proj".into(),
            provider: Some("claude-code".into()),
            model: Some(model.into()),
            project_slug: Some(project.into()),
            ..Default::default()
        }
    }

    /// What `run_subagent` asks before it opens the child.
    fn claim_for(manager: &ChatManager, child: &ChatRequest) -> Reuse {
        manager
            .reuse_for(child, None, "parent-1")
            .expect("a delegation pinned to one engine and model asks the pool")
    }

    fn a_person() -> Option<crate::auth::jwt::Claims> {
        Some(crate::auth::jwt::Claims::person("person-1", "a@b.c", "A"))
    }

    /// CACHE-POOL criterion 1, the MECHANISM: the second and third delegations of one
    /// conversation, on the same engine and model, are handed the SAME warm session and
    /// resume it instead of opening a new one. Red without the pool: every delegation
    /// would be a fresh `create_session`.
    #[tokio::test]
    async fn a_second_delegation_of_the_same_conversation_reuses_the_warm_session() {
        let (_graph, manager) = bare_manager();
        let child = delegation("proj", "opus");
        let first = claim_for(&manager, &child);
        assert!(
            first.warm.is_none(),
            "the pool starts empty: a fresh session"
        );
        // Exactly what `open_wait_close` does once the child is closed.
        first.release(&manager.session_pool, "warm-1", std::time::Instant::now());
        assert_eq!(manager.session_pool.held(), vec!["warm-1".to_owned()]);

        let second = claim_for(&manager, &child);
        assert_eq!(
            second.warm.as_ref().map(|warm| warm.session_id.as_str()),
            Some("warm-1"),
            "the same engine and model are handed the same warm handle"
        );
        assert!(
            manager.session_pool.is_empty(),
            "a claimed session is not lent to a concurrent delegation"
        );

        second.release(&manager.session_pool, "warm-1", std::time::Instant::now());
        let third = claim_for(&manager, &child);
        let warm = third.warm.as_ref().expect("still warm");
        assert_eq!(warm.session_id, "warm-1");
        assert_eq!(
            warm.uses, 2,
            "two delegations have already run in this session"
        );
    }

    /// CACHE-POOL, "a contaminated session is never reused": a child whose delegation did NOT
    /// answer (cancelled, errored) is dropped, not given back warm — the next delegation
    /// opens a fresh session. A child that answered still goes back. Red if `finish`
    /// releases whatever the outcome.
    #[tokio::test]
    async fn a_child_that_did_not_answer_is_never_given_back_warm() {
        let (_graph, manager) = bare_manager();
        let child = delegation("proj", "opus");
        let now = std::time::Instant::now();

        // A delegation resumes a warm session, then fails: nothing goes back.
        claim_for(&manager, &child).finish(&manager.session_pool, "warm-1", true, now);
        assert_eq!(manager.session_pool.held(), vec!["warm-1".to_owned()]);
        let resumed = claim_for(&manager, &child);
        assert!(resumed.warm.is_some(), "it resumes the warm session");
        resumed.finish(&manager.session_pool, "warm-1", false, now);
        assert!(
            manager.session_pool.is_empty(),
            "a failed child is not given back"
        );
        assert!(
            claim_for(&manager, &child).warm.is_none(),
            "the next delegation opens a fresh session"
        );

        // A fresh child that fails never enters the pool either.
        claim_for(&manager, &child).finish(&manager.session_pool, "fresh-1", false, now);
        assert!(manager.session_pool.is_empty());
    }

    /// Another model (or another engine) is another warm session: the pool is keyed on the
    /// pair, so a delegation routed elsewhere opens its own.
    #[tokio::test]
    async fn another_model_gets_no_warm_session() {
        let (_graph, manager) = bare_manager();
        let first = claim_for(&manager, &delegation("proj", "opus"));
        first.release(&manager.session_pool, "warm-1", std::time::Instant::now());
        let other = claim_for(&manager, &delegation("proj", "haiku"));
        assert!(other.warm.is_none());
        assert_eq!(
            manager.session_pool.held(),
            vec!["warm-1".to_owned()],
            "the entry stays for the model it belongs to"
        );
    }

    /// CACHE-POOL criterion 3, another PROJECT: the warm session of one project is never
    /// resumed for a delegation of another. Red if `reuse_for` stops keying on the project.
    #[tokio::test]
    async fn another_project_gets_no_warm_session() {
        let (_graph, manager) = bare_manager();
        let first = claim_for(&manager, &delegation("proj", "opus"));
        first.release(&manager.session_pool, "warm-1", std::time::Instant::now());
        let other = claim_for(&manager, &delegation("autre-projet", "opus"));
        assert!(other.warm.is_none(), "a warm session crosses no project");
        assert_eq!(manager.session_pool.held(), vec!["warm-1".to_owned()]);
    }

    /// CACHE-POOL criterion 2 and 3, another TOOL POLICY: a child opened read-only never
    /// resumes the session a `normal` child ran in, and another permission mode is refused
    /// too. Red if the facts are read from the warm session instead of compared.
    #[tokio::test]
    async fn another_tool_policy_gets_no_warm_session() {
        let (_graph, manager) = bare_manager();
        let first = claim_for(&manager, &delegation("proj", "opus"));
        first.release(&manager.session_pool, "warm-1", std::time::Instant::now());

        let mut read_only = delegation("proj", "opus");
        read_only.access = Some(SessionAccess::ReadOnly);
        assert!(claim_for(&manager, &read_only).warm.is_none());

        let mut edits = delegation("proj", "opus");
        edits.permission_mode = Some("acceptEdits".into());
        assert!(claim_for(&manager, &edits).warm.is_none());

        // The entry is untouched for the delegation whose facts it does match.
        assert_eq!(manager.session_pool.held(), vec!["warm-1".to_owned()]);
    }

    /// CACHE-POOL criterion 2, CONSENT: a person who approves (or revokes) a lasting rule
    /// between two delegations makes the warm session ineligible — it was opened under
    /// another consent, and a resume would carry approvals this delegation never got. Red
    /// if `reuse_for` stops fingerprinting the lasting rules.
    #[tokio::test]
    async fn a_changed_consent_gets_no_warm_session() {
        let dir = std::env::temp_dir().join(format!("po-warm-pool-{}", Uuid::new_v4()));
        let config = crate::chat::config::ChatConfig {
            lasting_rules_dir: Some(dir.clone()),
            ..Default::default()
        };
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let manager = ChatManager::new_without_memory(
            dyn_graph,
            crate::test_helpers::mock_app_state().meili,
            config,
        );
        let mut child = delegation("proj", "opus");
        child.user_claims = a_person();
        let first = claim_for(&manager, &child);
        first.release(&manager.session_pool, "warm-1", std::time::Instant::now());

        // The person answers "always" to a call in this project meanwhile.
        manager
            .lasting_rule_store()
            .expect("the configuration names a store")
            .add(
                "person-1",
                super::super::lasting_rules::LastingRule {
                    id: Uuid::new_v4(),
                    project: "proj".into(),
                    root: "/work/proj".into(),
                    asker: "native".into(),
                    tool: "Read".into(),
                    canonical: Some("Read".into()),
                    input: serde_json::json!({"file_path": "a.txt"}),
                    rule: "Read: a.txt".into(),
                    created_at: chrono::Utc::now(),
                    session_id: None,
                },
            )
            .expect("the rule is kept");

        let second = claim_for(&manager, &child);
        assert!(
            second.warm.is_none(),
            "a consent that changed since the opening is never inherited"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// B36-RACE, criterion 4 — the warm pool follows the same rule: a delegation handed
    /// the warm session of a parent that a cascade closed meanwhile does not resume it (the
    /// child is refused at its activation, under the engine's lock) and does not fall back
    /// to a fresh session either: it ends on the typed refusal. Red without the tombstone:
    /// the warm child is resumed, live, after its run was cancelled.
    #[tokio::test]
    async fn a_warm_child_of_a_cancelled_parent_is_not_resumed() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let config = crate::chat::config::ChatConfig {
            provider_path: crate::chat::config::ProviderPath::Agent,
            ..Default::default()
        };
        let fake = super::super::agent_runtime::fake::FakeProvider::new();
        let manager = Arc::new(
            ChatManager::new_without_memory(
                dyn_graph,
                crate::test_helpers::mock_app_state().meili,
                config,
            )
            .with_provider_source(Arc::new(fake.clone())),
        );
        let parent = crate::test_helpers::test_chat_session(None);
        graph.create_chat_session(&parent).await.unwrap();
        let parent_id = parent.id.to_string();
        super::super::manager::test_support::insert_live_session_without_cli(&manager, &parent_id)
            .await;
        let mut child = delegation("proj", "opus");
        child.provider = None;
        child.model = None;
        child.permission_mode = Some("acceptEdits".into());
        child.spawned_by = Some(
            serde_json::json!({"type": "conversation", "parent_session_id": parent_id}).to_string(),
        );
        // A first delegation ran and went back to the pool, warm.
        let warm = manager
            .create_session(&child)
            .await
            .expect("the parent is live: the child opens")
            .session_id;
        manager.close_session(&warm).await.unwrap();
        let pinned = delegation("proj", "opus");
        manager
            .reuse_for(&pinned, None, &parent_id)
            .expect("pinned")
            .release(&manager.session_pool, &warm, std::time::Instant::now());

        // The run is cancelled; a delegation already on its way claims the warm child.
        manager
            .close_session_with_cascade(&parent_id, true)
            .await
            .unwrap();
        let reuse = manager
            .reuse_for(&pinned, None, &parent_id)
            .expect("pinned");
        assert_eq!(
            reuse.warm.as_ref().map(|w| w.session_id.as_str()),
            Some(warm.as_str())
        );
        let err = manager
            .open_child(&child, Some(&reuse), &parent_id, "tu-1")
            .await
            .expect_err("a child of a cancelled parent is not resumed");
        assert!(err.contains("was cancelled"), "{err}");
        assert!(!manager.is_session_active(&warm).await, "not live again");
        assert_eq!(
            graph.get_session_children(parent.id).await.unwrap().len(),
            1,
            "no fresh child opened instead"
        );
        assert!(manager.session_pool.is_empty(), "dropped from the pool");
    }

    /// A child whose engine and model the cognitive router picks (`route_child`) has no key
    /// to ask the pool with: the pool is not consulted at all, and the delegation opens
    /// exactly as before.
    #[tokio::test]
    async fn a_routed_delegation_never_consults_the_pool() {
        let (_graph, manager) = bare_manager();
        let mut routed = delegation("proj", "opus");
        routed.provider = None;
        routed.model = None;
        assert!(manager.reuse_for(&routed, None, "parent-1").is_none());
        assert!(manager
            .reuse_for(&delegation("proj", "opus"), None, "parent-1")
            .is_some());
    }

    /// A resumed child shares its thread with the turn the warm session ran before: only
    /// what came AFTER the floor is this call's answer. Red without the floor — the wait
    /// would return the PREVIOUS delegation's answer at once, before the child did anything.
    #[tokio::test]
    async fn a_resumed_child_reads_only_its_own_turn() {
        let (graph, manager) = bare_manager();
        let sid = Uuid::new_v4();
        let record = |seq: i64, event: ChatEvent| crate::neo4j::models::ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: sid,
            seq,
            event_type: "event".into(),
            data: serde_json::to_string(&event).unwrap(),
            created_at: chrono::Utc::now(),
        };
        graph
            .store_chat_events(
                sid,
                vec![
                    record(1, said("premier tour")),
                    record(2, ok_result("PREMIERE-REPONSE")),
                ],
            )
            .await
            .unwrap();
        let id = sid.to_string();
        // Without a floor (a fresh child), the thread's end is the answer.
        assert_eq!(
            manager.read_back(&id, &mut Watch::default(), 0).await,
            Some(Ok("PREMIERE-REPONSE".to_string()))
        );
        // Resumed from the floor the thread had reached: nothing to answer yet.
        assert_eq!(manager.read_back(&id, &mut Watch::default(), 2).await, None);
        // The resumed turn answers.
        graph
            .store_chat_events(
                sid,
                vec![
                    record(3, said("second tour")),
                    record(4, ok_result("DEUXIEME-REPONSE")),
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            manager.read_back(&id, &mut Watch::default(), 2).await,
            Some(Ok("DEUXIEME-REPONSE".to_string()))
        );
    }

    // ── ISO2 — sub-agent type registry + background ─────────────────────

    /// ISO2, step 1: an unknown subagent_type is refused with a named error that lists
    /// the known types.  RED without the validation.
    #[test]
    fn unknown_subagent_type_is_refused() {
        assert!(
            resolve_subagent_type("nonexistent").is_none(),
            "unknown type must resolve to None"
        );
        // The error message must contain the type name and the known list.
        let list = known_subagent_type_list();
        assert!(list.contains("explore"));
        assert!(list.contains("research"));
        assert!(list.contains("code-reviewer"));
    }

    /// ISO2, step 1: every known type (CC + PO) resolves.
    #[test]
    fn all_known_types_resolve() {
        for name in [
            "explore",
            "claude-code-guide",
            "general-purpose",
            "plan",
            "web-fetch",
            "claude",
            "fork",
            "research",
            "code-reviewer",
        ] {
            assert!(
                resolve_subagent_type(name).is_some(),
                "type '{name}' should resolve"
            );
        }
        // Case-insensitive
        assert!(resolve_subagent_type("EXPLORE").is_some());
        assert!(resolve_subagent_type("Research").is_some());
    }

    /// ISO2, step 2: PO types that deny write tools populate denied_tools.
    #[test]
    fn po_types_carry_denied_tools() {
        let research = resolve_subagent_type("research").unwrap();
        assert!(
            research.denied_tools.contains(&"Edit"),
            "research must deny Edit"
        );
        assert!(
            research.denied_tools.contains(&"Write"),
            "research must deny Write"
        );
        assert!(
            research.denied_tools.contains(&"NotebookEdit"),
            "research must deny NotebookEdit"
        );

        let reviewer = resolve_subagent_type("code-reviewer").unwrap();
        assert!(!reviewer.denied_tools.is_empty());

        // CC types carry no denied_tools.
        let explore = resolve_subagent_type("explore").unwrap();
        assert!(
            explore.denied_tools.is_empty(),
            "CC types carry no host-imposed restrictions"
        );
    }

    /// ISO2: the denied_tools from a PO type flow into the ChatRequest.
    #[test]
    fn denied_tools_populate_chat_request() {
        let td = resolve_subagent_type("research").unwrap();
        let denied: Vec<String> = td.denied_tools.iter().map(|s| (*s).to_owned()).collect();
        assert!(denied.contains(&"Edit".to_owned()));
        assert!(denied.contains(&"Write".to_owned()));

        // A CC type produces an empty vec.
        let td = resolve_subagent_type("explore").unwrap();
        let denied: Vec<String> = td.denied_tools.iter().map(|s| (*s).to_owned()).collect();
        assert!(denied.is_empty());
    }
}
