//! The knowledge-graph hooks of the Claude Code engine, served to any provider.
//!
//! ## Why
//!
//! The Claude Code engine builds its hooks once per session as a table
//! `event name → [HookMatcher]` (skill activation before a tool, redirect advice
//! after a noisy one, the compaction notifier and the injection ledger reset at
//! compaction). Those hooks are where the project's graph enters the agent's
//! context while it works. The agent engine used to start every session with no
//! hook at all, so a native session never saw a skill, a persona, a note or the
//! compaction guidance, whatever the model behind it.
//!
//! [`GraphSessionHooks`] is the provider-neutral face of the SAME table: it
//! implements the contract's [`SessionHooks`] on top of the very callbacks the
//! Claude path registers. There is one logic, two doors.
//!
//! ## Mapping
//!
//! | `SessionHooks` | Claude event | What comes back |
//! |---|---|---|
//! | `before_tool` | `PreToolUse` | `additionalContext` → `AddContext`; `decision: block` or `permissionDecision: deny` → `Deny`; `updatedInput` → `ReplaceInput` |
//! | `after_tool` | `PostToolUse` | `additionalContext` |
//! | `before_compaction` | `PreCompact` | the callbacks' `reason` (where the notifier puts the instructions it built) |
//!
//! The tool is named as Claude Code names it (`canonical`, else the provider's
//! name) and its input is given as is: that is the vocabulary the hooks were
//! written in (`Read`, `Edit`, `Bash`… with `file_path`, `command`…).
//!
//! A hook that fails is skipped (logged): a hook observes and advises, it never
//! takes the turn down.
//!
//! ## Per-turn routing (`before_turn`)
//!
//! In routing mode `full` with the learning stage `auto`, the model of a turn is chosen
//! by the cognitive router. [`TurnRouter`] holds what that needs for one session and
//! [`directive_for_turn`] is the one function that changes the model of a turn, among the
//! models of the session's OWN provider (one session = one provider): the agent engine
//! reaches it through [`GraphSessionHooks::before_turn`], the legacy engine (Claude Code
//! CLI) calls it from `ChatManager::send_message` and sends the `set_model` control frame
//! itself.
//!
//! In mode `full`, the decision may name ANOTHER provider. That is decided before the turn
//! starts, by [`plan_provider_move`], which the host's send path calls on both engines
//! (`ChatManager::send_message`): a move opens a new session through the relay
//! (`moved_by: auto`) and the message is sent there, never to the old session. The hook
//! then uses the decision already taken for its turn instead of asking again.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use nexus_claude::agent::{
    CompactionInfo, HookVerdict, SessionHooks, ToolCallInfo, ToolResultInfo, TurnContext,
    TurnDirective,
};
use nexus_claude::{
    HookContext, HookInput, HookJSONOutput, HookMatcher, PostToolUseHookInput, PreCompactHookInput,
    PreToolUseHookInput,
};
use serde_json::Value;
use tracing::{debug, warn};
use uuid::Uuid;

use super::provider::cognitive::candidates::{self, ModelFacts, Slot};
use super::provider::cognitive::decider::HYSTERESIS_MARGIN;
use super::provider::cognitive::decision::{CognitiveDecision, DecideRequest, Decider, Pick};
use super::provider::cognitive::signature::{ContextHints, TaskSignature};
use super::provider::cognitive::{LearningStage, ProviderRoutingMode, RoutingSettings};
use super::types::{EffectiveCapability, EffectiveCause, EffectiveSource, RoutingPoolEntry};

/// The table the Claude path hands to its CLI: event name → matchers.
pub(crate) type HookTable = HashMap<String, Vec<HookMatcher>>;

pub(crate) const PRE_TOOL_USE: &str = "PreToolUse";
pub(crate) const POST_TOOL_USE: &str = "PostToolUse";
pub(crate) const PRE_COMPACT: &str = "PreCompact";

/// [`SessionHooks`] over a [`HookTable`].
pub(crate) struct GraphSessionHooks {
    table: HookTable,
    session_id: String,
    cwd: String,
    permission_mode: Option<String>,
    /// Per-turn model routing; `None` when no decider is configured.
    router: Option<Arc<TurnRouter>>,
}

impl GraphSessionHooks {
    pub(crate) fn new(
        table: HookTable,
        session_id: impl Into<String>,
        cwd: impl Into<String>,
        permission_mode: Option<String>,
    ) -> Self {
        Self {
            table,
            session_id: session_id.into(),
            cwd: cwd.into(),
            permission_mode,
            router: None,
        }
    }

    /// Lets the cognitive router choose the model of each turn.
    pub(crate) fn with_turn_router(mut self, router: Option<Arc<TurnRouter>>) -> Self {
        self.router = router;
        self
    }

    /// The callbacks registered for `event` whose matcher accepts `tool`.
    fn callbacks<'a>(
        &'a self,
        event: &str,
        tool: Option<&'a str>,
    ) -> impl Iterator<Item = &'a Arc<dyn nexus_claude::HookCallback>> + 'a {
        self.table
            .get(event)
            .into_iter()
            .flatten()
            .filter(move |matcher| matches(matcher.matcher.as_ref(), tool))
            .flat_map(|matcher| matcher.hooks.iter())
    }

    async fn run(
        &self,
        hook: &Arc<dyn nexus_claude::HookCallback>,
        input: &HookInput,
        tool_use_id: Option<&str>,
        event: &str,
    ) -> Option<HookJSONOutput> {
        let context = HookContext { signal: None };
        match hook.execute(input, tool_use_id, &context).await {
            Ok(output) => Some(output),
            Err(error) => {
                warn!(session_id = %self.session_id, event, error = %error, "graph hook failed (skipped)");
                None
            }
        }
    }
}

/// Claude's matcher: absent or empty matches everything; a string is a list of
/// tool names separated by `|` (`*` matches everything).
fn matches(matcher: Option<&Value>, tool: Option<&str>) -> bool {
    let Some(pattern) = matcher.and_then(Value::as_str).filter(|p| !p.is_empty()) else {
        return true;
    };
    let Some(tool) = tool else {
        return true;
    };
    pattern
        .split('|')
        .any(|name| name == "*" || name.trim() == tool)
}

/// What a callback said, flattened.
#[derive(Default)]
struct Said {
    context: Option<String>,
    deny: Option<String>,
    replaced: Option<Value>,
    reason: Option<String>,
}

fn read(output: HookJSONOutput) -> Said {
    use nexus_claude::HookSpecificOutput;
    let HookJSONOutput::Sync(sync) = output else {
        return Said::default();
    };
    let mut said = Said {
        reason: sync.reason.clone(),
        ..Said::default()
    };
    if sync.decision.as_deref() == Some("block") {
        said.deny = Some(
            sync.reason
                .clone()
                .unwrap_or_else(|| "blocked by a hook".to_owned()),
        );
    }
    match sync.hook_specific_output {
        Some(HookSpecificOutput::PreToolUse(pre)) => {
            if pre.permission_decision.as_deref() == Some("deny") {
                said.deny = Some(
                    pre.permission_decision_reason
                        .unwrap_or_else(|| "denied by a hook".to_owned()),
                );
            }
            said.replaced = pre.updated_input;
            said.context = pre.additional_context;
        }
        Some(HookSpecificOutput::PostToolUse(post)) => said.context = post.additional_context,
        _ => {}
    }
    said
}

/// Texts joined by a blank line, empty ones dropped.
fn joined(parts: Vec<String>) -> Option<String> {
    let parts: Vec<String> = parts.into_iter().filter(|p| !p.trim().is_empty()).collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn tool_name(call: &ToolCallInfo) -> String {
    call.canonical.clone().unwrap_or_else(|| call.name.clone())
}

#[async_trait]
impl SessionHooks for GraphSessionHooks {
    async fn before_tool(&self, call: &ToolCallInfo) -> HookVerdict {
        let name = tool_name(call);
        let input = HookInput::PreToolUse(PreToolUseHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            tool_name: name.clone(),
            tool_input: call.input.clone(),
            agent_id: None,
            agent_type: None,
        });
        let mut contexts = Vec::new();
        let mut replaced = None;
        for hook in self.callbacks(PRE_TOOL_USE, Some(&name)) {
            let Some(output) = self
                .run(hook, &input, call.id.as_deref(), PRE_TOOL_USE)
                .await
            else {
                continue;
            };
            let said = read(output);
            if let Some(reason) = said.deny {
                return HookVerdict::Deny { reason };
            }
            if said.replaced.is_some() {
                replaced = said.replaced;
            }
            contexts.extend(said.context);
        }
        // A replaced input and added context cannot both travel in one verdict: the
        // replacement wins (it changes what runs), the context is not lost when
        // there is no replacement.
        match (replaced, joined(contexts)) {
            (Some(input), _) => HookVerdict::ReplaceInput(input),
            (None, Some(text)) => HookVerdict::AddContext(text),
            (None, None) => HookVerdict::Continue,
        }
    }

    async fn after_tool(&self, result: &ToolResultInfo) -> Option<String> {
        let name = tool_name(&result.call);
        let input = HookInput::PostToolUse(PostToolUseHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            tool_name: name.clone(),
            tool_input: result.call.input.clone(),
            tool_response: result.output.clone(),
            agent_id: None,
            agent_type: None,
        });
        let mut contexts = Vec::new();
        for hook in self.callbacks(POST_TOOL_USE, Some(&name)) {
            if let Some(output) = self
                .run(hook, &input, result.call.id.as_deref(), POST_TOOL_USE)
                .await
            {
                contexts.extend(read(output).context);
            }
        }
        joined(contexts)
    }

    async fn before_compaction(&self, info: &CompactionInfo) -> Option<String> {
        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            trigger: info.trigger.clone(),
            custom_instructions: info.custom_instructions.clone(),
        });
        let mut reasons = Vec::new();
        for hook in self.callbacks(PRE_COMPACT, None) {
            if let Some(output) = self.run(hook, &input, None, PRE_COMPACT).await {
                reasons.extend(read(output).reason);
            }
        }
        joined(reasons)
    }

    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        match &self.router {
            Some(router) => directive_for_turn(router, ctx).await,
            None => TurnDirective::none(),
        }
    }
}

/// How long the decider may take before the turn goes on without it.
pub(crate) const DECIDE_TIMEOUT: Duration = Duration::from_secs(2);

/// The models the per-turn decision chooses among.
#[async_trait]
pub trait PoolSource: Send + Sync {
    /// Facts of the models of `provider_id` only (the candidates of a turn that stays on
    /// its session's provider).
    async fn pool(&self, provider_id: &str) -> Vec<ModelFacts>;

    /// Facts of every model of every instance, with the consent of `project_slug` stated
    /// on each (`allowed_for_project`): the candidates of a turn in mode `full`, which may
    /// move the conversation to another provider. Empty when the source knows no more than
    /// one provider at a time: no move is then considered.
    async fn project_pool(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
        Vec::new()
    }
}

struct TurnState {
    current_model: String,
    last_change_turn: Option<u32>,
    /// Index of the next turn: counted here for the legacy engine, which has no harness
    /// to count turns, and copied from the harness on the agent engine.
    next_turn: u32,
}

/// A decision taken BEFORE its turn started (the provider check of mode `full`): the turn
/// it was taken for uses it instead of asking again.
#[derive(Debug, Clone)]
struct PreDecided {
    turn_index: u32,
    pick: Option<Pick>,
    /// The pick may be applied (stage `auto`, not right after a change).
    apply: bool,
}

/// What the per-turn decision of ONE session needs.
pub(crate) struct TurnRouter {
    decider: Arc<dyn Decider>,
    pool: Arc<dyn PoolSource>,
    /// Mode, stage, pin and pool of the conversation: read when the session opened,
    /// replaced by `PUT /api/chat/sessions/{id}/routing` (never by the global settings).
    choice: Mutex<RouterChoice>,
    provider_id: String,
    session_id: Option<Uuid>,
    project_slug: Option<String>,
    trust: bool,
    /// The session can switch model between turns (known once it is open).
    set_model_live: AtomicBool,
    /// The user changed the model by hand: no more automatic change.
    manual: AtomicBool,
    /// The conversation reached this session by a provider move: the next provider check
    /// is recorded, never applied (no two moves in a row).
    moved_in: AtomicBool,
    predecided: Mutex<Option<PreDecided>>,
    last_message: Mutex<Option<String>>,
    /// The next turn carries an image (`set_turn_input`): its decision excludes the models
    /// that cannot read one (`needs_images`, `RejectReason::NoImages`).
    last_has_images: AtomicBool,
    state: Mutex<TurnState>,
    timeout: Duration,
}

/// What the conversation asked of its routing, as the router holds it.
#[derive(Debug, Clone)]
struct RouterChoice {
    routing: RoutingSettings,
    /// The model is imposed (named by the request, or one model ticked): never replaced.
    explicit_model: bool,
    /// The provider is imposed (named by the request, `routed_by: request`): the
    /// conversation is never moved off it automatically.
    provider_imposed: bool,
    /// Models of this provider the conversation may be routed among (a pool); `None` = all.
    allowed_models: Option<Vec<String>>,
    /// Every pair of the conversation's pool, whatever its provider: the candidates of a
    /// provider check in mode `full`. `None` = no pool.
    routing_pool: Option<Vec<RoutingPoolEntry>>,
}

/// What opens a [`TurnRouter`].
pub(crate) struct TurnRouterSpec {
    pub decider: Arc<dyn Decider>,
    pub pool: Arc<dyn PoolSource>,
    pub routing: RoutingSettings,
    pub provider_id: String,
    pub session_id: Option<Uuid>,
    pub project_slug: Option<String>,
    pub trust: bool,
    pub explicit_model: bool,
    pub provider_imposed: bool,
    pub allowed_models: Option<Vec<String>>,
    pub routing_pool: Option<Vec<RoutingPoolEntry>>,
    pub current_model: String,
    /// Index the next turn counted by the router itself gets (legacy engine).
    pub next_turn: u32,
    /// The session continues a conversation the router moved here (`moved_by: auto`).
    pub moved_in: bool,
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl TurnRouter {
    pub(crate) fn new(spec: TurnRouterSpec) -> Self {
        // The relayed turn (index 0) runs on the pair the move chose: nothing to ask.
        let predecided = spec.moved_in.then_some(PreDecided {
            turn_index: 0,
            pick: None,
            apply: false,
        });
        Self {
            decider: spec.decider,
            pool: spec.pool,
            choice: Mutex::new(RouterChoice {
                routing: spec.routing,
                explicit_model: spec.explicit_model,
                provider_imposed: spec.provider_imposed,
                allowed_models: spec.allowed_models,
                routing_pool: spec.routing_pool,
            }),
            provider_id: spec.provider_id,
            session_id: spec.session_id,
            project_slug: spec.project_slug,
            trust: spec.trust,
            set_model_live: AtomicBool::new(false),
            manual: AtomicBool::new(false),
            moved_in: AtomicBool::new(spec.moved_in),
            predecided: Mutex::new(predecided),
            last_message: Mutex::new(None),
            last_has_images: AtomicBool::new(false),
            state: Mutex::new(TurnState {
                current_model: spec.current_model,
                last_change_turn: None,
                next_turn: spec.next_turn,
            }),
            timeout: DECIDE_TIMEOUT,
        }
    }

    /// The text the next turn will carry: the hook only sees its length.
    pub(crate) fn set_last_message(&self, message: &str) {
        *locked(&self.last_message) = Some(message.to_owned());
    }

    /// The next turn, from its STORED form (`<po-attachments>`/`<po-refs>` blocks
    /// included): the routing reads what the user typed, and whether an image goes with it
    /// (F-R4: the decision then keeps only the candidates that read images).
    pub(crate) fn set_turn_input(&self, stored: &str) {
        let (_, attachments) = super::message_attachments::split(stored);
        let images = attachments.iter().any(super::message_attachments::is_image);
        self.last_has_images.store(images, Ordering::SeqCst);
        self.set_last_message(&crate::refs::turn::visible_text(stored));
    }

    /// Whether the next turn carries an image, as last set.
    pub(crate) fn turn_has_images(&self) -> bool {
        self.last_has_images.load(Ordering::SeqCst)
    }

    /// The text the next turn will carry, as last set.
    #[cfg(test)]
    pub(crate) fn last_message(&self) -> Option<String> {
        locked(&self.last_message).clone()
    }

    /// Whether the session can switch model between turns.
    pub(crate) fn set_model_live(&self, live: bool) {
        self.set_model_live.store(live, Ordering::SeqCst);
    }

    /// The conversation's routing changed (`PUT .../routing`): from the next turn on, `mode`
    /// replaces the one the session opened with, `allowed_models` (this provider's models of
    /// the pool) restricts the candidates, `explicit_model` pins the current model. Whatever
    /// was set by hand before is forgotten: this IS the new choice by hand, and a provider
    /// imposed at the opening is released (`auto` lets PO choose the provider again; the
    /// other choices never move the conversation).
    pub(crate) fn reroute(
        &self,
        mode: ProviderRoutingMode,
        allowed_models: Option<Vec<String>>,
        explicit_model: bool,
    ) {
        let mut choice = locked(&self.choice);
        choice.routing.mode = mode;
        choice.allowed_models = allowed_models;
        choice.explicit_model = explicit_model;
        choice.provider_imposed = false;
        choice.routing_pool = None;
        self.manual.store(false, Ordering::SeqCst);
    }

    /// The user changed the model by hand: the router leaves the session alone.
    pub(crate) fn mark_manual(&self) {
        self.manual.store(true, Ordering::SeqCst);
    }

    /// The context of the next turn of a session whose turns the router counts itself.
    pub(crate) fn next_turn_context(&self, input_chars: usize) -> TurnContext {
        let mut state = locked(&self.state);
        let mut ctx = TurnContext::new(state.next_turn, state.current_model.clone());
        ctx.input_chars = input_chars;
        state.next_turn += 1;
        ctx
    }

    /// An applied change that could not be sent: the model did not change.
    pub(crate) fn forget_change(&self, model_before: &str) {
        let mut state = locked(&self.state);
        state.current_model = model_before.to_owned();
        state.last_change_turn = None;
    }

    /// The signature of the turn about to start, from the text it carries.
    fn signature(&self) -> TaskSignature {
        let message = locked(&self.last_message).clone().unwrap_or_default();
        TaskSignature::from_chat_request(
            &message,
            self.turn_has_images(),
            self.project_slug.as_deref(),
            ContextHints::default(),
        )
    }

    /// Asks the decider within the router's timeout; `None` (logged) when it fails or is slow.
    async fn ask(&self, request: DecideRequest) -> Option<CognitiveDecision> {
        match tokio::time::timeout(self.timeout, self.decider.decide(&request)).await {
            Ok(Ok(decision)) => Some(decision),
            Ok(Err(error)) => {
                warn!(provider = %self.provider_id, error = %error, "turn routing failed: the model stays");
                None
            }
            Err(_) => {
                warn!(provider = %self.provider_id, "turn routing timed out: the model stays");
                None
            }
        }
    }
}

/// The settings a decision is asked with: `stage` drops from `auto` to `shadow` when the
/// pick may not be applied, so the decision is recorded as what it is.
fn settings_for(routing: &RoutingSettings, apply: bool) -> RoutingSettings {
    let mut settings = routing.clone();
    if !apply && settings.stage == LearningStage::Auto {
        settings.stage = LearningStage::Shadow;
    }
    settings
}

/// What the turn about to start does about its provider.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProviderPlan {
    /// The turn runs on the session's provider (its model is still routed by
    /// [`directive_for_turn`]).
    Stay,
    /// The decision names another provider: the conversation moves there before the turn
    /// starts, and the turn runs on the new session. `decision` is the stored decision.
    Move {
        pick: Pick,
        decision: Box<CognitiveDecision>,
    },
}

/// Whether the turn about to start moves the conversation to ANOTHER provider (mode
/// `full`). Called before the turn starts, by the host's send path: a session runs on one
/// provider, so a move is never a [`TurnDirective`].
///
/// * not `full` (`mixed` stays on the session's provider), a model or a provider imposed
///   (`routed_by: request`, pinned), a model changed by hand, a source that knows one
///   provider only: nothing is asked, [`ProviderPlan::Stay`];
/// * otherwise the decider chooses among every model the project consented to (the
///   conversation's pool when it has one), the current pair in force. The decision is kept
///   for the turn (the hook does not ask again) and recorded applied or not;
/// * a move needs the stage `auto`, no move or model change on the previous turn, a pick
///   that is not an exploration draw and that beats the current pair by the hysteresis
///   margin. Anything else stays.
pub(crate) async fn plan_provider_move(router: &TurnRouter) -> ProviderPlan {
    let choice = locked(&router.choice).clone();
    if choice.routing.mode != ProviderRoutingMode::Full
        || choice.explicit_model
        || choice.provider_imposed
        || router.manual.load(Ordering::SeqCst)
    {
        return ProviderPlan::Stay;
    }
    let mut pool = match tokio::time::timeout(
        router.timeout,
        router.pool.project_pool(router.project_slug.as_deref()),
    )
    .await
    {
        Ok(pool) => pool,
        Err(_) => return ProviderPlan::Stay,
    };
    if let Some(entries) = &choice.routing_pool {
        pool.retain(|facts| {
            entries
                .iter()
                .any(|e| e.provider == facts.provider_id && e.model == facts.model)
        });
    }
    if !pool
        .iter()
        .any(|facts| facts.provider_id != router.provider_id)
    {
        return ProviderPlan::Stay;
    }
    let (turn, current_model, just_changed) = {
        let state = locked(&router.state);
        let just_changed = state
            .last_change_turn
            .is_some_and(|t| t.checked_add(1) == Some(state.next_turn));
        (state.next_turn, state.current_model.clone(), just_changed)
    };
    // No two moves in a row: the turn right after a move (or a model change) is only recorded.
    let just_moved = router.moved_in.swap(false, Ordering::SeqCst);
    // F-R4: a turn the pair in force cannot read must leave it, whatever the anti-flapping.
    let blind = cannot_read_turn(router, &pool, &router.provider_id, &current_model);
    let apply =
        choice.routing.stage == LearningStage::Auto && (blind || (!just_moved && !just_changed));
    let current = Pick::new(&router.provider_id, &current_model);
    let mut request = DecideRequest::new(
        router.signature(),
        settings_for(&choice.routing, apply),
        pool,
    );
    request.trust = router.trust;
    request.current = Some(current.clone());
    request.session_id = router.session_id;
    request.turn_index = Some(turn);
    let Some(decision) = router.ask(request).await else {
        return ProviderPlan::Stay;
    };
    let apply = apply && decision.applied;
    *locked(&router.predecided) = Some(PreDecided {
        turn_index: turn,
        pick: decision.chosen.clone(),
        apply,
    });
    let pick = match &decision.chosen {
        Some(pick) if apply && pick.provider_id != router.provider_id => pick.clone(),
        _ => return ProviderPlan::Stay,
    };
    if decision.explored && !blind {
        debug!(provider = %pick.provider_id, "an exploration draw never moves a conversation");
        return ProviderPlan::Stay;
    }
    let score_of = |wanted: &Pick| {
        decision
            .alternatives
            .iter()
            .find(|a| a.pick == *wanted && a.rejected.is_none())
            .and_then(|a| a.score)
    };
    if let (Some(best), Some(now)) = (
        decision.score.or_else(|| score_of(&pick)),
        score_of(&current),
    ) {
        if best - now < HYSTERESIS_MARGIN {
            debug!(provider = %pick.provider_id, gap = best - now, "the gap is under the hysteresis margin: the provider stays");
            return ProviderPlan::Stay;
        }
    }
    ProviderPlan::Move {
        pick,
        decision: Box::new(decision),
    }
}

/// The model of the turn about to start, `none` when it stays as it is.
///
/// * not `full` mode, a model named by the request, a model changed by hand, a session
///   that cannot switch model live: nothing is asked and nothing changes;
/// * a decision already taken for this turn by [`plan_provider_move`] is used as is (a
///   pick on another provider changes nothing here: the move was the host's to make);
/// * `full` mode before the `auto` stage (and the turn right after a change): the decider
///   is still asked, so the decision is recorded, but nothing is applied;
/// * `full` + `auto`: the pick is used when the decision is applied and differs from the
///   current model. A failing or slow decider (2 s) never fails the turn.
pub(crate) async fn directive_for_turn(router: &TurnRouter, ctx: &TurnContext) -> TurnDirective {
    {
        // The harness knows the model the turn would run on, whoever changed it, and the
        // index of the turn.
        let mut state = locked(&router.state);
        state.current_model = ctx.current_model.clone();
        state.next_turn = ctx.turn_index.saturating_add(1);
    }
    let predecided = locked(&router.predecided)
        .take()
        .filter(|pre| pre.turn_index == ctx.turn_index);
    let choice = locked(&router.choice).clone();
    // `full` routes everything; a pool (mixed, models ticked in the menu) routes among them.
    let routed =
        choice.routing.mode == ProviderRoutingMode::Full || choice.allowed_models.is_some();
    if !routed
        || choice.explicit_model
        // A pool none of whose models is on this provider: nothing to route among here.
        || choice.allowed_models.as_ref().is_some_and(Vec::is_empty)
        || router.manual.load(Ordering::SeqCst)
        || !router.set_model_live.load(Ordering::SeqCst)
    {
        return TurnDirective::none();
    }
    let pick = match predecided {
        Some(pre) => pre.pick.filter(|_| pre.apply),
        None => {
            // Never two changes in consecutive turns: the turn right after one is only recorded.
            let just_changed = locked(&router.state)
                .last_change_turn
                .is_some_and(|turn| turn.checked_add(1) == Some(ctx.turn_index));
            let mut pool = provider_pool(router).await;
            if let Some(allowed) = &choice.allowed_models {
                pool.retain(|facts| allowed.iter().any(|m| m == &facts.model));
            }
            // F-R4: a turn the model in force cannot read must leave it, even right after a change.
            let blind = cannot_read_turn(router, &pool, &router.provider_id, &ctx.current_model);
            let apply = choice.routing.stage == LearningStage::Auto && (blind || !just_changed);
            let mut settings = settings_for(&choice.routing, apply);
            // A pool is routed like `full` inside it, whatever the settings' mode.
            if choice.allowed_models.is_some() {
                settings.mode = ProviderRoutingMode::Full;
            }
            let mut request = DecideRequest::new(router.signature(), settings, pool);
            request.trust = router.trust;
            request.restrict_provider = Some(router.provider_id.clone());
            request.current = Some(Pick::new(&router.provider_id, &ctx.current_model));
            request.session_id = router.session_id;
            request.turn_index = Some(ctx.turn_index);
            let Some(decision) = router.ask(request).await else {
                return TurnDirective::none();
            };
            if !apply || !decision.applied {
                return TurnDirective::none();
            }
            if let Some(pick) = &decision.chosen {
                debug!(model = %pick.model, turn = ctx.turn_index, reason = %decision.reason, "turn routing decided");
            }
            decision.chosen
        }
    };
    match pick {
        Some(pick) if pick.provider_id == router.provider_id && pick.model != ctx.current_model => {
            debug!(model = %pick.model, turn = ctx.turn_index, "turn routing changes the model");
            let mut state = locked(&router.state);
            state.current_model = pick.model.clone();
            state.last_change_turn = Some(ctx.turn_index);
            TurnDirective::model(pick.model)
        }
        _ => TurnDirective::none(),
    }
}

/// The models of the session's provider, with the consent of the session's PROJECT stated
/// on each. [`PoolSource::pool`] knows no project: the manager's answers it without one,
/// and without a project no stored instance is allowed, so every model came back
/// `not_allowed` and a turn decided here was `no_candidate` (the opening turn, and every
/// turn of a conversation routed among models ticked in the menu). The project's pool,
/// narrowed to the provider, is asked first; a source that has none answers `pool`.
async fn provider_pool(router: &TurnRouter) -> Vec<ModelFacts> {
    let project = tokio::time::timeout(
        router.timeout,
        router.pool.project_pool(router.project_slug.as_deref()),
    )
    .await
    .unwrap_or_default();
    let mine: Vec<ModelFacts> = project
        .into_iter()
        .filter(|facts| facts.provider_id == router.provider_id)
        .collect();
    if mine.is_empty() {
        router.pool.pool(&router.provider_id).await
    } else {
        mine
    }
}

/// The turn about to start carries an image and the pair in force cannot read it (or is not
/// in the pool, so nothing says it can). Staying is then no option: the anti-flapping rules
/// (no two changes in a row, no move on an exploration draw) give way to the hard constraint.
fn cannot_read_turn(router: &TurnRouter, pool: &[ModelFacts], provider: &str, model: &str) -> bool {
    router.turn_has_images()
        && !pool
            .iter()
            .any(|f| f.provider_id == provider && f.model == model && f.supports_images)
}

/// Whether PO chooses the MODEL of this conversation's turns on its provider (the rules of
/// [`directive_for_turn`], at the stage that applies them).
fn routes_model(router: &TurnRouter, choice: &RouterChoice) -> bool {
    (choice.routing.mode == ProviderRoutingMode::Full || choice.allowed_models.is_some())
        && choice.routing.stage == LearningStage::Auto
        && !choice.explicit_model
        && !choice.allowed_models.as_ref().is_some_and(Vec::is_empty)
        && !router.manual.load(Ordering::SeqCst)
        && router.set_model_live.load(Ordering::SeqCst)
}

/// Whether PO may move this conversation to ANOTHER provider (the rules of
/// [`plan_provider_move`], at the stage that applies them).
fn routes_provider(router: &TurnRouter, choice: &RouterChoice) -> bool {
    choice.routing.mode == ProviderRoutingMode::Full
        && choice.routing.stage == LearningStage::Auto
        && !choice.explicit_model
        && !choice.provider_imposed
        && !router.manual.load(Ordering::SeqCst)
}

/// The model of the session's provider an image turn must run on, chosen BEFORE the turn
/// is sent (F-R4). The native harness checks the vision of its ACTIVE model when the turn is
/// sent, before `before_turn` could change it: the host makes this model active first, and
/// the hook of the turn finds the decision taken (it does not ask again).
///
/// `None`: nothing to change — no image, PO does not choose this conversation's model, the
/// model in force reads images, or no candidate does (the provider then refuses the turn
/// and says so: a pool without vision is never hidden). A decision already taken for this
/// turn (the provider check of mode `full`) is used as is.
pub(crate) async fn model_for_images(router: &TurnRouter) -> Option<String> {
    if !router.turn_has_images() {
        return None;
    }
    let choice = locked(&router.choice).clone();
    if !routes_model(router, &choice) {
        return None;
    }
    let (turn, current_model) = {
        let state = locked(&router.state);
        (state.next_turn, state.current_model.clone())
    };
    let predecided = locked(&router.predecided)
        .clone()
        .filter(|pre| pre.turn_index == turn);
    let pick = match predecided {
        Some(pre) => pre.pick.filter(|_| pre.apply)?,
        None => {
            let mut pool = provider_pool(router).await;
            if let Some(allowed) = &choice.allowed_models {
                pool.retain(|facts| allowed.iter().any(|m| m == &facts.model));
            }
            if !cannot_read_turn(router, &pool, &router.provider_id, &current_model) {
                return None;
            }
            let mut settings = settings_for(&choice.routing, true);
            if choice.allowed_models.is_some() {
                settings.mode = ProviderRoutingMode::Full;
            }
            let mut request = DecideRequest::new(router.signature(), settings, pool);
            request.trust = router.trust;
            request.restrict_provider = Some(router.provider_id.clone());
            request.current = Some(Pick::new(&router.provider_id, &current_model));
            request.session_id = router.session_id;
            request.turn_index = Some(turn);
            let decision = router.ask(request).await?;
            if !decision.applied {
                return None;
            }
            let pick = decision.chosen.clone()?;
            debug!(model = %pick.model, turn, reason = %decision.reason, "an image turn is routed to a model that reads images");
            *locked(&router.predecided) = Some(PreDecided {
                turn_index: turn,
                pick: Some(pick.clone()),
                apply: true,
            });
            pick
        }
    };
    if pick.provider_id != router.provider_id || pick.model == current_model {
        return None;
    }
    let mut state = locked(&router.state);
    state.current_model = pick.model.clone();
    state.last_change_turn = Some(turn);
    Some(pick.model)
}

/// Whether the next turn of this conversation may carry images, as PO can really serve it
/// (F-R4, decision 11cefdb2): the candidates PO may route the turn to govern, not the
/// snapshot of the model the session opened on.
///
/// * the snapshot says yes: yes (`model_has_it`);
/// * PO does not route this conversation (primary, a model imposed or changed by hand, a
///   stage that applies nothing, a session that cannot switch model): the snapshot
///   (`not_routed`);
/// * PO routes but its pool lists no model at all: the snapshot (`pool_unbuilt`) — not
///   probed is not absent;
/// * otherwise: yes when at least one candidate passes the filter of an image turn
///   (`pool_has_it`, `via` names them), no when none does (`pool_lacks_it`).
pub(crate) async fn effective_images(router: &TurnRouter, snapshot: bool) -> EffectiveCapability {
    if snapshot {
        return EffectiveCapability::snapshot(true, EffectiveCause::ModelHasIt);
    }
    let choice = locked(&router.choice).clone();
    let by_model = routes_model(router, &choice);
    let by_provider = routes_provider(router, &choice);
    if !by_model && !by_provider {
        return EffectiveCapability::snapshot(false, EffectiveCause::NotRouted);
    }
    let mut pool = Vec::new();
    if by_model {
        let mut mine = provider_pool(router).await;
        if let Some(allowed) = &choice.allowed_models {
            mine.retain(|facts| allowed.iter().any(|m| m == &facts.model));
        }
        pool.extend(mine);
    }
    if by_provider {
        let mut others = tokio::time::timeout(
            router.timeout,
            router.pool.project_pool(router.project_slug.as_deref()),
        )
        .await
        .unwrap_or_default();
        others.retain(|facts| facts.provider_id != router.provider_id);
        if let Some(entries) = &choice.routing_pool {
            others.retain(|facts| {
                entries
                    .iter()
                    .any(|e| e.provider == facts.provider_id && e.model == facts.model)
            });
        }
        pool.extend(others);
    }
    if pool.is_empty() {
        return EffectiveCapability::snapshot(false, EffectiveCause::PoolUnbuilt);
    }
    let signature = TaskSignature::from_chat_request(
        "",
        true,
        router.project_slug.as_deref(),
        ContextHints::default(),
    );
    let via: Vec<RoutingPoolEntry> = candidates::apply(Slot::Automatic, &signature, &pool)
        .map(|filtered| filtered.eligible)
        .unwrap_or_default()
        .into_iter()
        .map(|facts| RoutingPoolEntry {
            provider: facts.provider_id,
            model: facts.model,
        })
        .collect();
    EffectiveCapability {
        value: !via.is_empty(),
        source: EffectiveSource::RoutingPool,
        cause: if via.is_empty() {
            EffectiveCause::PoolLacksIt
        } else {
            EffectiveCause::PoolHasIt
        },
        via,
    }
}

/// Records the cognitive decision for the compaction summary about to be written, in
/// shadow: what the router would choose to summarise this conversation. It is never
/// applied, because neither engine lets the caller choose the summary model (the Claude Code
/// CLI compacts on the session's model; nexus' native compaction calls the session's own
/// endpoint with its active model, and `CompactionConfig` has no model field). The stored
/// decision says so (`compaction_model_not_selectable`) and names the pair the summary runs
/// on (`used`). Every mode records it, like the opening decision; a failing or slow decider
/// (2 s) only loses the record.
pub(crate) async fn record_compaction_decision(router: &TurnRouter) -> Option<CognitiveDecision> {
    use super::provider::cognitive::decider::COMPACTION_MODEL_NOT_SELECTABLE;
    use super::provider::cognitive::signature::TaskClass;

    let choice = locked(&router.choice).clone();
    let current = Pick::new(&router.provider_id, &locked(&router.state).current_model);
    // Every model the project consented to (the summary needs no tool and could run
    // anywhere), else the session's provider's: the alternatives worth learning about.
    let mut pool = match tokio::time::timeout(
        router.timeout,
        router.pool.project_pool(router.project_slug.as_deref()),
    )
    .await
    {
        Ok(pool) if !pool.is_empty() => pool,
        _ => router.pool.pool(&router.provider_id).await,
    };
    if let Some(entries) = &choice.routing_pool {
        pool.retain(|facts| {
            entries
                .iter()
                .any(|e| e.provider == facts.provider_id && e.model == facts.model)
        });
    }
    // The summary reads what the current model held when compaction started: about 80 %
    // of its window (the threshold of both engines; the signature clamps it to its floor).
    let need = pool
        .iter()
        .find(|f| f.provider_id == current.provider_id && f.model == current.model)
        .and_then(|f| f.context_window)
        .map_or(0, |window| window / 5 * 4);
    let signature = TaskSignature::utility(
        TaskClass::UtilityCompaction,
        need,
        router.project_slug.as_deref(),
    );
    let mut settings = choice.routing.clone();
    settings.stage = LearningStage::Shadow;
    let mut request = DecideRequest::new(signature, settings, pool);
    request.trust = router.trust;
    request.current = Some(current);
    request.session_id = router.session_id;
    request.not_selectable = Some(COMPACTION_MODEL_NOT_SELECTABLE);
    let decision = router.ask(request).await?;
    debug!(provider = %router.provider_id, reason = %decision.reason, "compaction decision recorded in shadow");
    Some(decision)
}

/// The decider shared by the sessions and the pool it chooses in.
pub(crate) type SharedDecider = (Arc<dyn Decider>, Arc<dyn PoolSource>);

/// The routers of the live sessions, and the decider they share.
#[derive(Default)]
pub(crate) struct TurnRouting {
    decider: Mutex<Option<SharedDecider>>,
    routers: Mutex<HashMap<String, Arc<TurnRouter>>>,
}

impl TurnRouting {
    pub(crate) fn configure(&self, decider: Arc<dyn Decider>, pool: Arc<dyn PoolSource>) {
        *locked(&self.decider) = Some((decider, pool));
    }

    pub(crate) fn configured(&self) -> Option<SharedDecider> {
        locked(&self.decider).clone()
    }

    pub(crate) fn insert(&self, session_id: &str, router: Arc<TurnRouter>) {
        locked(&self.routers).insert(session_id.to_owned(), router);
    }

    pub(crate) fn get(&self, session_id: &str) -> Option<Arc<TurnRouter>> {
        locked(&self.routers).get(session_id).cloned()
    }

    pub(crate) fn remove(&self, session_id: &str) {
        locked(&self.routers).remove(session_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_claude::agent::ToolCategory;
    use nexus_claude::{
        HookCallback, HookSpecificOutput, PostToolUseHookSpecificOutput,
        PreToolUseHookSpecificOutput, SyncHookJSONOutput,
    };
    use serde_json::json;
    use std::sync::Mutex;

    /// Answers with a fixed output and remembers what it was given.
    struct Scripted {
        output: Mutex<Option<HookJSONOutput>>,
        seen: Mutex<Vec<HookInput>>,
        fail: bool,
    }

    impl Scripted {
        fn answering(output: SyncHookJSONOutput) -> Arc<Self> {
            Arc::new(Self {
                output: Mutex::new(Some(HookJSONOutput::Sync(output))),
                seen: Mutex::new(Vec::new()),
                fail: false,
            })
        }

        fn failing() -> Arc<Self> {
            Arc::new(Self {
                output: Mutex::new(None),
                seen: Mutex::new(Vec::new()),
                fail: true,
            })
        }
    }

    #[async_trait]
    impl HookCallback for Scripted {
        async fn execute(
            &self,
            input: &HookInput,
            _tool_use_id: Option<&str>,
            _context: &HookContext,
        ) -> Result<HookJSONOutput, nexus_claude::SdkError> {
            self.seen.lock().unwrap().push(input.clone());
            if self.fail {
                return Err(nexus_claude::SdkError::InvalidState {
                    message: "boom".into(),
                });
            }
            Ok(self
                .output
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| HookJSONOutput::Sync(SyncHookJSONOutput::default())))
        }
    }

    fn table(event: &str, matcher: Option<Value>, hooks: Vec<Arc<dyn HookCallback>>) -> HookTable {
        let mut table = HookTable::new();
        table.insert(event.to_owned(), vec![HookMatcher { matcher, hooks }]);
        table
    }

    fn hooks(table: HookTable) -> GraphSessionHooks {
        GraphSessionHooks::new(table, "s1", "/work/app", Some("default".into()))
    }

    fn call(name: &str, canonical: Option<&str>, input: Value) -> ToolCallInfo {
        ToolCallInfo {
            id: Some("c1".into()),
            name: name.into(),
            canonical: canonical.map(str::to_owned),
            category: ToolCategory::Mcp,
            input,
        }
    }

    fn pre_context(text: &str) -> SyncHookJSONOutput {
        SyncHookJSONOutput {
            continue_: Some(true),
            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                PreToolUseHookSpecificOutput {
                    permission_decision: None,
                    permission_decision_reason: None,
                    updated_input: None,
                    additional_context: Some(text.into()),
                },
            )),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn before_tool_gives_the_hook_the_claude_vocabulary_and_returns_its_context() {
        let hook = Scripted::answering(pre_context("SKILL: use find_references"));
        let hooks = hooks(table(PRE_TOOL_USE, None, vec![hook.clone()]));
        let verdict = hooks
            .before_tool(&call(
                "mcp__nexus__read",
                Some("Read"),
                json!({"file_path": "src/a.rs"}),
            ))
            .await;
        assert_eq!(
            verdict,
            HookVerdict::AddContext("SKILL: use find_references".into())
        );
        let seen = hook.seen.lock().unwrap();
        let HookInput::PreToolUse(pre) = &seen[0] else {
            panic!("a PreToolUse input is expected");
        };
        // Named as Claude Code names it, input untouched, session facts filled in.
        assert_eq!(pre.tool_name, "Read");
        assert_eq!(pre.tool_input, json!({"file_path": "src/a.rs"}));
        assert_eq!(
            (pre.session_id.as_str(), pre.cwd.as_str()),
            ("s1", "/work/app")
        );
    }

    #[tokio::test]
    async fn a_tool_without_canonical_name_is_given_under_its_own_name() {
        let hook = Scripted::answering(pre_context("x"));
        let hooks = hooks(table(PRE_TOOL_USE, None, vec![hook.clone()]));
        hooks
            .before_tool(&call("mcp__project-orchestrator__note", None, json!({})))
            .await;
        let seen = hook.seen.lock().unwrap();
        let HookInput::PreToolUse(pre) = &seen[0] else {
            panic!()
        };
        assert_eq!(pre.tool_name, "mcp__project-orchestrator__note");
    }

    #[tokio::test]
    async fn a_blocking_or_denying_hook_denies_with_its_reason() {
        let block = Scripted::answering(SyncHookJSONOutput {
            decision: Some("block".into()),
            reason: Some("hot bridge".into()),
            ..Default::default()
        });
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![block]))
            .before_tool(&call("t", None, json!({})))
            .await;
        assert_eq!(
            verdict,
            HookVerdict::Deny {
                reason: "hot bridge".into()
            }
        );

        let deny = Scripted::answering(SyncHookJSONOutput {
            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                PreToolUseHookSpecificOutput {
                    permission_decision: Some("deny".into()),
                    permission_decision_reason: Some("not here".into()),
                    updated_input: None,
                    additional_context: None,
                },
            )),
            ..Default::default()
        });
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![deny]))
            .before_tool(&call("t", None, json!({})))
            .await;
        assert_eq!(
            verdict,
            HookVerdict::Deny {
                reason: "not here".into()
            }
        );
    }

    #[tokio::test]
    async fn an_updated_input_replaces_the_input() {
        let rewrite = Scripted::answering(SyncHookJSONOutput {
            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                PreToolUseHookSpecificOutput {
                    permission_decision: None,
                    permission_decision_reason: None,
                    updated_input: Some(json!({"command": "ls"})),
                    additional_context: None,
                },
            )),
            ..Default::default()
        });
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![rewrite]))
            .before_tool(&call("Bash", Some("Bash"), json!({"command": "ls -R /"})))
            .await;
        assert_eq!(verdict, HookVerdict::ReplaceInput(json!({"command": "ls"})));
    }

    #[tokio::test]
    async fn contexts_of_several_hooks_are_joined_and_a_failing_hook_is_skipped() {
        let a = Scripted::answering(pre_context("ONE"));
        let b = Scripted::answering(pre_context("TWO"));
        let verdict = hooks(table(PRE_TOOL_USE, None, vec![a, Scripted::failing(), b]))
            .before_tool(&call("t", None, json!({})))
            .await;
        assert_eq!(verdict, HookVerdict::AddContext("ONE\n\nTWO".into()));
    }

    #[tokio::test]
    async fn without_a_registered_hook_the_call_goes_on() {
        let hooks = hooks(HookTable::new());
        assert_eq!(
            hooks.before_tool(&call("t", None, json!({}))).await,
            HookVerdict::Continue
        );
        assert_eq!(
            hooks
                .after_tool(&ToolResultInfo {
                    call: call("t", None, json!({})),
                    output: json!("x"),
                    is_error: false,
                })
                .await,
            None
        );
        assert_eq!(
            hooks
                .before_compaction(&CompactionInfo {
                    trigger: "auto".into(),
                    custom_instructions: None,
                })
                .await,
            None
        );
    }

    #[tokio::test]
    async fn a_matcher_restricts_a_hook_to_its_tools() {
        let hook = Scripted::answering(pre_context("ONLY BASH"));
        let hooks = hooks(table(
            PRE_TOOL_USE,
            Some(json!("Bash|Grep")),
            vec![hook.clone()],
        ));
        assert_eq!(
            hooks.before_tool(&call("x", Some("Read"), json!({}))).await,
            HookVerdict::Continue
        );
        assert!(hook.seen.lock().unwrap().is_empty());
        assert_eq!(
            hooks.before_tool(&call("x", Some("Grep"), json!({}))).await,
            HookVerdict::AddContext("ONLY BASH".into())
        );
    }

    #[tokio::test]
    async fn after_tool_hands_over_the_response_and_returns_the_context() {
        let hook = Scripted::answering(SyncHookJSONOutput {
            hook_specific_output: Some(HookSpecificOutput::PostToolUse(
                PostToolUseHookSpecificOutput {
                    additional_context: Some("use find_references instead".into()),
                },
            )),
            ..Default::default()
        });
        let hooks = hooks(table(POST_TOOL_USE, None, vec![hook.clone()]));
        let said = hooks
            .after_tool(&ToolResultInfo {
                call: call("mcp__nexus__grep", Some("Grep"), json!({"pattern": "foo"})),
                output: json!("60 matches"),
                is_error: false,
            })
            .await;
        assert_eq!(said.as_deref(), Some("use find_references instead"));
        let seen = hook.seen.lock().unwrap();
        let HookInput::PostToolUse(post) = &seen[0] else {
            panic!("a PostToolUse input is expected");
        };
        assert_eq!(post.tool_name, "Grep");
        assert_eq!(post.tool_response, json!("60 matches"));
    }

    #[tokio::test]
    async fn before_compaction_returns_the_reason_the_notifier_builds() {
        let notifier = Scripted::answering(SyncHookJSONOutput {
            continue_: Some(true),
            reason: Some("Preserve: decision on the hot bridge".into()),
            ..Default::default()
        });
        let reset = Scripted::answering(SyncHookJSONOutput {
            continue_: Some(true),
            ..Default::default()
        });
        let hooks = hooks(table(
            PRE_COMPACT,
            None,
            vec![notifier.clone(), reset.clone()],
        ));
        let said = hooks
            .before_compaction(&CompactionInfo {
                trigger: "auto".into(),
                custom_instructions: None,
            })
            .await;
        assert_eq!(
            said.as_deref(),
            Some("Preserve: decision on the hot bridge")
        );
        // Both ran: the ledger reset must see every compaction.
        assert_eq!(notifier.seen.lock().unwrap().len(), 1);
        assert_eq!(reset.seen.lock().unwrap().len(), 1);
        let seen = notifier.seen.lock().unwrap();
        let HookInput::PreCompact(pre) = &seen[0] else {
            panic!("a PreCompact input is expected");
        };
        assert_eq!(pre.trigger, "auto");
    }
}

/// The provider check of mode `full` ([`plan_provider_move`]) and how the hook of the same
/// turn uses it, on a fake decider and a fake pool.
#[cfg(test)]
mod provider_move_tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::DecisionAlternative;
    use chrono::Utc;
    use nexus_claude::agent::CostBasis;
    use std::sync::Mutex as StdMutex;

    fn facts(provider: &str, model: &str, allowed: bool) -> ModelFacts {
        ModelFacts {
            provider_id: provider.into(),
            model: model.into(),
            supports_tools: true,
            supports_images: true,
            context_window: Some(200_000),
            window_unknown: None,
            price: None,
            cost_basis: CostBasis::Unknown,
            healthy: Some(true),
            allowed_for_project: allowed,
            sandboxed: false,
        }
    }

    /// `local/m` (the session's), `local2/m`, `local3/m` (no consent).
    struct ThreeProviders;

    #[async_trait]
    impl PoolSource for ThreeProviders {
        async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
            self.project_pool(None)
                .await
                .into_iter()
                .filter(|f| f.provider_id == provider_id)
                .collect()
        }
        async fn project_pool(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
            vec![
                facts("local", "m", true),
                facts("local2", "m", true),
                facts("local3", "m", false),
            ]
        }
    }

    /// Knows one provider at a time (the default `project_pool`).
    struct OneProvider;

    #[async_trait]
    impl PoolSource for OneProvider {
        async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
            vec![facts(provider_id, "m", true)]
        }
    }

    /// Picks `pick` with `score`, the current pair scored `current_score`; applied in
    /// `full` + `auto` like the real decider (pilot role).
    struct Fake {
        pick: Pick,
        score: f64,
        current_score: f64,
        explored: bool,
        requests: StdMutex<Vec<DecideRequest>>,
    }

    impl Fake {
        fn picking(provider: &str, score: f64, current_score: f64) -> Arc<Self> {
            Arc::new(Self {
                pick: Pick::new(provider, "m"),
                score,
                current_score,
                explored: false,
                requests: StdMutex::new(Vec::new()),
            })
        }
        fn calls(&self) -> usize {
            self.requests.lock().unwrap().len()
        }
        fn last(&self) -> DecideRequest {
            self.requests.lock().unwrap().last().cloned().unwrap()
        }
    }

    #[async_trait]
    impl Decider for Fake {
        async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
            self.requests.lock().unwrap().push(request.clone());
            let mut alternatives = vec![DecisionAlternative {
                pick: self.pick.clone(),
                score: Some(self.score),
                rejected: None,
            }];
            if let Some(current) = request.current.clone().filter(|c| *c != self.pick) {
                alternatives.push(DecisionAlternative {
                    pick: current,
                    score: Some(self.current_score),
                    rejected: None,
                });
            }
            Ok(CognitiveDecision {
                id: Uuid::new_v4(),
                at: Utc::now(),
                signature: request.signature.clone(),
                chosen: Some(self.pick.clone()),
                score: Some(self.score),
                explored: self.explored,
                reason: "fake".into(),
                alternatives,
                applied: request.settings.mode == ProviderRoutingMode::Full
                    && request.settings.stage == LearningStage::Auto,
                mode: request.settings.mode,
                stage: request.settings.stage,
                session_id: request.session_id,
                task_id: None,
                run_id: None,
                turn_index: request.turn_index,
                outcome: None,
                used: None,
            })
        }
    }

    fn spec(
        decider: Arc<Fake>,
        pool: Arc<dyn PoolSource>,
        mode: ProviderRoutingMode,
        stage: LearningStage,
    ) -> TurnRouterSpec {
        TurnRouterSpec {
            decider,
            pool,
            routing: RoutingSettings {
                mode,
                stage,
                ..RoutingSettings::default()
            },
            provider_id: "local".into(),
            session_id: Some(Uuid::new_v4()),
            project_slug: Some("proj".into()),
            trust: false,
            explicit_model: false,
            provider_imposed: false,
            allowed_models: None,
            routing_pool: None,
            current_model: "m".into(),
            next_turn: 1,
            moved_in: false,
        }
    }

    fn router(spec: TurnRouterSpec) -> TurnRouter {
        let router = TurnRouter::new(spec);
        router.set_model_live(true);
        router.set_last_message("why does this crash with a stack trace error");
        router
    }

    fn full_auto(decider: Arc<Fake>) -> TurnRouter {
        router(spec(
            decider,
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        ))
    }

    fn is_move_to(plan: &ProviderPlan, provider: &str) -> bool {
        matches!(plan, ProviderPlan::Move { pick, .. } if pick.provider_id == provider)
    }

    #[tokio::test]
    async fn full_auto_a_pick_on_another_provider_is_a_move_decided_once_for_the_turn() {
        let decider = Fake::picking("local2", 0.9, 0.5);
        let router = full_auto(decider.clone());
        let plan = plan_provider_move(&router).await;
        assert!(is_move_to(&plan, "local2"), "{plan:?}");
        let request = decider.last();
        assert_eq!(
            request.restrict_provider, None,
            "every provider is a candidate"
        );
        assert_eq!(
            request.pool.len(),
            3,
            "the project's instances, consent stated"
        );
        assert_eq!(request.current, Some(Pick::new("local", "m")));
        assert_eq!(request.turn_index, Some(1));
        // The hook of that turn (were it to start here) uses the same decision.
        let directive = directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        assert_eq!(directive.model, None, "a move is never a model directive");
        assert_eq!(decider.calls(), 1, "one decision per turn");
    }

    #[tokio::test]
    async fn mixed_never_leaves_the_sessions_provider() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Mixed,
            LearningStage::Auto,
        );
        s.allowed_models = Some(vec!["m".into()]);
        let router = router(s);
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.calls(), 0, "mixed asks no provider question");
        // The model decision of the turn stays restricted to the session's provider.
        directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        assert_eq!(decider.last().restrict_provider.as_deref(), Some("local"));
        assert!(decider.last().pool.iter().all(|f| f.provider_id == "local"));
    }

    #[tokio::test]
    async fn an_imposed_model_or_provider_is_never_moved_until_auto_is_chosen_again() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut pinned = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        pinned.explicit_model = true;
        assert_eq!(
            plan_provider_move(&router(pinned)).await,
            ProviderPlan::Stay
        );
        let mut named = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        named.provider_imposed = true;
        let named = router(named);
        assert_eq!(plan_provider_move(&named).await, ProviderPlan::Stay);
        assert_eq!(
            decider.calls(),
            0,
            "an imposed pair is not even asked about"
        );
        // `PUT .../routing {auto: true}`: PO chooses the provider again.
        named.reroute(ProviderRoutingMode::Full, None, false);
        assert!(is_move_to(&plan_provider_move(&named).await, "local2"));
    }

    #[tokio::test]
    async fn the_shadow_stage_records_the_provider_decision_and_moves_nothing() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let router = router(spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Shadow,
        ));
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.calls(), 1, "asked, so recorded");
        assert_eq!(decider.last().settings.stage, LearningStage::Shadow);
        assert_eq!(decider.last().restrict_provider, None);
        directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        assert_eq!(decider.calls(), 1, "the hook does not ask again");
    }

    #[tokio::test]
    async fn a_gap_under_the_hysteresis_margin_or_an_exploration_draw_never_moves() {
        let close = Fake::picking("local2", 0.65, 0.6);
        assert_eq!(
            plan_provider_move(&full_auto(close)).await,
            ProviderPlan::Stay
        );
        let clear = Fake::picking("local2", 0.75, 0.6);
        assert!(is_move_to(
            &plan_provider_move(&full_auto(clear)).await,
            "local2"
        ));
        let explored = Arc::new(Fake {
            explored: true,
            ..Arc::try_unwrap(Fake::picking("local2", 0.9, 0.1))
                .ok()
                .unwrap()
        });
        assert_eq!(
            plan_provider_move(&full_auto(explored)).await,
            ProviderPlan::Stay
        );
    }

    #[tokio::test]
    async fn the_turn_after_a_move_is_only_recorded_then_the_router_may_move_again() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        s.moved_in = true;
        let router = router(s);
        // The relayed turn (0) runs on the pair the move chose: nothing is asked.
        directive_for_turn(&router, &TurnContext::new(0, "m")).await;
        assert_eq!(decider.calls(), 0);
        // The next turn: recorded, never applied (no two moves in a row).
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.last().settings.stage, LearningStage::Shadow);
        directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        // The one after may move.
        assert!(is_move_to(&plan_provider_move(&router).await, "local2"));
    }

    #[tokio::test]
    async fn the_conversations_pool_bounds_the_providers_checked() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        s.routing_pool = Some(vec![
            RoutingPoolEntry {
                provider: "local".into(),
                model: "m".into(),
            },
            RoutingPoolEntry {
                provider: "local2".into(),
                model: "m".into(),
            },
        ]);
        let router = router(s);
        assert!(is_move_to(&plan_provider_move(&router).await, "local2"));
        let providers: Vec<_> = decider
            .last()
            .pool
            .iter()
            .map(|f| f.provider_id.clone())
            .collect();
        assert_eq!(providers, ["local", "local2"]);
    }

    #[tokio::test]
    async fn a_source_that_knows_one_provider_asks_no_provider_question() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let router = router(spec(
            decider.clone(),
            Arc::new(OneProvider),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        ));
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.calls(), 0);
    }
}

/// F-R4: the capability shown follows the routing candidates, and an image turn is routed
/// to one that reads images (never substituted: the pool is constrained).
#[cfg(test)]
mod image_routing_tests {
    use super::*;
    use crate::chat::message_attachments::{self, MessageAttachment};
    use crate::chat::provider::cognitive::candidates::RejectReason;
    use crate::chat::provider::cognitive::decider::decide_with;
    use crate::chat::provider::cognitive::scorer::PriorHints;
    use chrono::Utc;
    use nexus_claude::agent::CostBasis;
    use std::sync::Mutex as StdMutex;

    fn facts(provider: &str, model: &str, images: bool) -> ModelFacts {
        ModelFacts {
            provider_id: provider.into(),
            model: model.into(),
            supports_tools: true,
            supports_images: images,
            context_window: Some(1_000_000),
            window_unknown: None,
            price: None,
            cost_basis: CostBasis::Unknown,
            healthy: Some(true),
            allowed_for_project: true,
            sandboxed: false,
        }
    }

    /// A fixed project pool; `pool(provider)` is its part on that provider.
    struct Pool(Vec<ModelFacts>);

    #[async_trait]
    impl PoolSource for Pool {
        async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
            self.0
                .iter()
                .filter(|f| f.provider_id == provider_id)
                .cloned()
                .collect()
        }
        async fn project_pool(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
            self.0.clone()
        }
    }

    /// The real decision (candidates, scoring, hysteresis, reason), no store.
    #[derive(Default)]
    struct Real {
        decisions: StdMutex<Vec<CognitiveDecision>>,
    }

    #[async_trait]
    impl Decider for Real {
        async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
            let decision =
                decide_with(Uuid::new_v4(), Utc::now(), request, &[], &PriorHints::new());
            self.decisions.lock().unwrap().push(decision.clone());
            Ok(decision)
        }
    }

    fn router(
        decider: Arc<Real>,
        pool: Vec<ModelFacts>,
        mode: ProviderRoutingMode,
        allowed_models: Option<Vec<String>>,
    ) -> TurnRouter {
        let router = TurnRouter::new(TurnRouterSpec {
            decider,
            pool: Arc::new(Pool(pool)),
            routing: RoutingSettings {
                mode,
                stage: LearningStage::Auto,
                exploration_epsilon: 0.0,
                ..RoutingSettings::default()
            },
            provider_id: "deepseek".into(),
            session_id: Some(Uuid::new_v4()),
            project_slug: Some("proj".into()),
            trust: false,
            explicit_model: false,
            provider_imposed: false,
            allowed_models,
            routing_pool: None,
            current_model: "flash".into(),
            next_turn: 1,
            moved_in: false,
        });
        router.set_model_live(true);
        router
    }

    /// The stored form of a message with one picture attached.
    fn with_picture(text: &str) -> String {
        message_attachments::encode(
            text,
            &[MessageAttachment {
                id: Uuid::new_v4(),
                filename: "shot.png".into(),
                mime_type: "image/png".into(),
                size_bytes: 4,
            }],
        )
    }

    fn deepseek_with_vision() -> Vec<ModelFacts> {
        vec![
            facts("deepseek", "flash", false),
            facts("deepseek", "vision", true),
        ]
    }

    #[test]
    fn the_turn_input_says_whether_an_image_goes_with_it() {
        let router = router(
            Arc::default(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        router.set_turn_input(&with_picture("what is this?"));
        assert!(router.turn_has_images());
        assert_eq!(router.last_message().as_deref(), Some("what is this?"));
        assert!(router.signature().needs_images);
        router.set_turn_input("just text");
        assert!(!router.turn_has_images());
        assert!(!router.signature().needs_images);
    }

    #[tokio::test]
    async fn a_pool_with_a_vision_model_makes_images_reachable_and_says_through_whom() {
        let router = router(
            Arc::default(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        let images = effective_images(&router, false).await;
        assert!(images.value);
        assert_eq!(images.source, EffectiveSource::RoutingPool);
        assert_eq!(images.cause, EffectiveCause::PoolHasIt);
        assert_eq!(
            images.via,
            [RoutingPoolEntry {
                provider: "deepseek".into(),
                model: "vision".into()
            }]
        );
    }

    #[tokio::test]
    async fn another_provider_with_vision_counts_in_full_but_not_in_a_pool_of_this_provider() {
        let pool = vec![facts("deepseek", "flash", false), facts("glm", "v", true)];
        let full = router(
            Arc::default(),
            pool.clone(),
            ProviderRoutingMode::Full,
            None,
        );
        let images = effective_images(&full, false).await;
        assert!(images.value, "{images:?}");
        assert_eq!(images.via[0].provider, "glm");

        let mixed = router(
            Arc::default(),
            pool,
            ProviderRoutingMode::Mixed,
            Some(vec!["flash".into()]),
        );
        let images = effective_images(&mixed, false).await;
        assert!(!images.value);
        assert_eq!(images.cause, EffectiveCause::PoolLacksIt);
    }

    #[tokio::test]
    async fn a_pool_without_any_vision_model_still_says_images_will_not_pass() {
        let router = router(
            Arc::default(),
            vec![
                facts("deepseek", "flash", false),
                facts("deepseek", "pro", false),
            ],
            ProviderRoutingMode::Full,
            None,
        );
        let images = effective_images(&router, false).await;
        assert!(!images.value);
        assert_eq!(images.source, EffectiveSource::RoutingPool);
        assert_eq!(images.cause, EffectiveCause::PoolLacksIt);
        assert!(images.via.is_empty());
        // And the turn is not routed anywhere: the provider refuses it, said so.
        router.set_turn_input(&with_picture("look"));
        assert_eq!(model_for_images(&router).await, None);
    }

    #[tokio::test]
    async fn an_empty_pool_falls_back_on_the_snapshot_with_its_cause() {
        let router = router(Arc::default(), Vec::new(), ProviderRoutingMode::Full, None);
        let images = effective_images(&router, false).await;
        assert!(!images.value);
        assert_eq!(images.source, EffectiveSource::Snapshot);
        assert_eq!(images.cause, EffectiveCause::PoolUnbuilt);
    }

    #[tokio::test]
    async fn without_routing_the_snapshot_stands_as_before() {
        // primary: PO does not choose the model.
        let primary = router(
            Arc::default(),
            deepseek_with_vision(),
            ProviderRoutingMode::Primary,
            None,
        );
        let images = effective_images(&primary, false).await;
        assert_eq!(
            (images.value, images.source, images.cause),
            (false, EffectiveSource::Snapshot, EffectiveCause::NotRouted)
        );
        // A model changed by hand: the same.
        let manual = router(
            Arc::default(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        manual.mark_manual();
        assert_eq!(
            effective_images(&manual, false).await.cause,
            EffectiveCause::NotRouted
        );
        // A model that reads images: nothing to route for.
        let images = effective_images(&primary, true).await;
        assert_eq!(
            (images.value, images.cause),
            (true, EffectiveCause::ModelHasIt)
        );
    }

    #[tokio::test]
    async fn an_image_turn_is_routed_to_a_vision_candidate_and_the_reason_says_so() {
        let decider = Arc::new(Real::default());
        let router = router(
            decider.clone(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        router.set_turn_input(&with_picture("what is on this screenshot?"));
        assert_eq!(model_for_images(&router).await.as_deref(), Some("vision"));
        let decision = decider.decisions.lock().unwrap().last().cloned().unwrap();
        assert_eq!(decision.chosen, Some(Pick::new("deepseek", "vision")));
        assert!(decision.reason.contains("images"), "{}", decision.reason);
        assert!(decision
            .alternatives
            .iter()
            .any(|a| a.pick.model == "flash" && a.rejected == Some(RejectReason::NoImages)));
        // The hook of that turn uses the decision taken: no second question, no change back.
        let directive =
            directive_for_turn(&router, &TurnContext::new(1, "vision".to_string())).await;
        assert_eq!(directive.model, None);
        assert_eq!(decider.decisions.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn the_hook_routes_an_image_turn_even_right_after_a_change() {
        let decider = Arc::new(Real::default());
        let router = router(
            decider.clone(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        // The model changed on the previous turn: a text turn would only be recorded.
        locked(&router.state).last_change_turn = Some(1);
        router.set_turn_input(&with_picture("and this one?"));
        let directive =
            directive_for_turn(&router, &TurnContext::new(2, "flash".to_string())).await;
        assert_eq!(directive.model.as_deref(), Some("vision"));
        let decision = decider.decisions.lock().unwrap().last().cloned().unwrap();
        assert!(decision.signature.needs_images);
    }

    #[tokio::test]
    async fn a_text_turn_keeps_the_model_in_force() {
        let decider = Arc::new(Real::default());
        let router = router(
            decider.clone(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        router.set_turn_input("why does this crash with a stack trace error");
        assert_eq!(model_for_images(&router).await, None);
        assert!(decider.decisions.lock().unwrap().is_empty());
    }
}
