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

use super::provider::cognitive::candidates::ModelFacts;
use super::provider::cognitive::decider::HYSTERESIS_MARGIN;
use super::provider::cognitive::decision::{CognitiveDecision, DecideRequest, Decider, Pick};
use super::provider::cognitive::signature::{ContextHints, TaskSignature};
use super::provider::cognitive::{
    conversation_stage, LearningStage, ProviderRoutingMode, RoutingSettings,
};
use super::types::RoutingPoolEntry;

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
    /// The last decision asked for a turn, with the turn's index: the decision a change of
    /// model comes from, which the host marks not applied when the change cannot be sent
    /// ([`TurnRouter::decision_of_turn`]).
    decided: Mutex<Option<(u32, CognitiveDecision)>>,
    last_message: Mutex<Option<String>>,
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
    /// The settings' stage the session opened with. `routing.stage` is the stage in force
    /// for THIS conversation: `auto` when the user handed its routing to PO (Auto, or a
    /// pool), this one otherwise (decision R-S1, [`conversation_stage`]). Kept so that
    /// [`TurnRouter::reroute`] can fall back to it when the user picks one model again.
    settings_stage: LearningStage,
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
    /// The conversation asked for Auto or a pool (decision R-S1).
    pub conversation_routes: bool,
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
        let settings_stage = spec.routing.stage;
        let mut routing = spec.routing;
        routing.stage = conversation_stage(settings_stage, spec.conversation_routes);
        Self {
            decider: spec.decider,
            pool: spec.pool,
            choice: Mutex::new(RouterChoice {
                routing,
                explicit_model: spec.explicit_model,
                provider_imposed: spec.provider_imposed,
                allowed_models: spec.allowed_models,
                routing_pool: spec.routing_pool,
                settings_stage,
            }),
            provider_id: spec.provider_id,
            session_id: spec.session_id,
            project_slug: spec.project_slug,
            trust: spec.trust,
            set_model_live: AtomicBool::new(false),
            manual: AtomicBool::new(false),
            moved_in: AtomicBool::new(spec.moved_in),
            predecided: Mutex::new(predecided),
            decided: Mutex::new(None),
            last_message: Mutex::new(None),
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
        // Auto, or a pool (`allowed_models` is only given for two models or more): the
        // user's choice, applied at the `auto` stage (decision R-S1).
        let conversation_routes =
            mode == ProviderRoutingMode::Full || choice.allowed_models.is_some();
        choice.routing.stage = conversation_stage(choice.settings_stage, conversation_routes);
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

    /// The decision asked for turn `turn_index`, if it is the last one asked: the decision
    /// a change of that turn came from.
    pub(crate) fn decision_of_turn(&self, turn_index: u32) -> Option<CognitiveDecision> {
        locked(&self.decided)
            .take()
            .filter(|(turn, _)| *turn == turn_index)
            .map(|(_, decision)| decision)
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
            false,
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
    let apply = choice.routing.stage == LearningStage::Auto && !just_moved && !just_changed;
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
    *locked(&router.decided) = Some((turn, decision.clone()));
    *locked(&router.predecided) = Some(PreDecided {
        turn_index: turn,
        pick: decision.chosen.clone(),
        apply,
    });
    let pick = match &decision.chosen {
        Some(pick) if apply && pick.provider_id != router.provider_id => pick.clone(),
        _ => return ProviderPlan::Stay,
    };
    if decision.explored {
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
            let apply = choice.routing.stage == LearningStage::Auto && !just_changed;
            let mut settings = settings_for(&choice.routing, apply);
            // A pool is routed like `full` inside it, whatever the settings' mode.
            if choice.allowed_models.is_some() {
                settings.mode = ProviderRoutingMode::Full;
            }
            let mut pool = provider_pool(router).await;
            if let Some(allowed) = &choice.allowed_models {
                pool.retain(|facts| allowed.iter().any(|m| m == &facts.model));
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
            *locked(&router.decided) = Some((ctx.turn_index, decision.clone()));
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

    /// Removes the session's router only if it is still `router` (`Arc::ptr_eq`): an
    /// opening that failed never takes away the router of one that replaced it.
    pub(crate) fn remove_if(&self, session_id: &str, router: &Arc<TurnRouter>) {
        let mut routers = locked(&self.routers);
        if routers
            .get(session_id)
            .is_some_and(|current| Arc::ptr_eq(current, router))
        {
            routers.remove(session_id);
        }
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
            conversation_routes: false,
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

    /// Decision R-S1: the settings' stage governs a session that asked nothing (`full` from
    /// the settings, `shadow`: recorded, never applied); the user's Auto on the conversation
    /// decides it at the `auto` stage; one model ticked afterwards pins it again.
    #[tokio::test]
    async fn the_conversations_auto_is_decided_at_the_auto_stage_whatever_the_settings() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let router = router(spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Shadow,
        ));
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.last().settings.stage, LearningStage::Shadow);

        let router = router_after_auto(decider.clone());
        // The stage in force lives in `choice.routing.stage` itself: every reader of it
        // (F-R4's effective capabilities, `model_for_images`) gets the conversation's.
        assert_eq!(locked(&router.choice).routing.stage, LearningStage::Auto);
        assert!(is_move_to(&plan_provider_move(&router).await, "local2"));
        assert_eq!(decider.last().settings.stage, LearningStage::Auto);

        // One model ticked: strict, nothing is asked any more, the settings' stage is back.
        router.reroute(ProviderRoutingMode::Primary, None, true);
        assert_eq!(locked(&router.choice).routing.stage, LearningStage::Shadow);
        let calls = decider.calls();
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        let directive = directive_for_turn(&router, &TurnContext::new(2, "m")).await;
        assert_eq!(directive.model, None);
        assert_eq!(decider.calls(), calls);
    }

    /// A pool (two models ticked) chosen on a conversation of a `primary` + `shadow` setting
    /// changes the model between turns: the decision is asked, and stored, at `auto`.
    #[tokio::test]
    async fn a_pool_chosen_on_the_conversation_is_decided_at_the_auto_stage() {
        let decider = Arc::new(Fake {
            pick: Pick::new("local", "big"),
            score: 0.9,
            current_score: 0.1,
            explored: false,
            requests: StdMutex::new(Vec::new()),
        });
        let router = router(spec(
            decider.clone(),
            Arc::new(OneProvider),
            ProviderRoutingMode::Primary,
            LearningStage::Shadow,
        ));
        router.reroute(
            ProviderRoutingMode::Mixed,
            Some(vec!["m".into(), "big".into()]),
            false,
        );
        let directive = directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        assert_eq!(directive.model.as_deref(), Some("big"));
        let asked = decider.last();
        assert_eq!(
            (asked.settings.mode, asked.settings.stage),
            (ProviderRoutingMode::Full, LearningStage::Auto)
        );
    }

    /// A router of a `full` + `shadow` setting whose conversation was handed to PO (Auto).
    fn router_after_auto(decider: Arc<Fake>) -> TurnRouter {
        let router = router(spec(
            decider,
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Primary,
            LearningStage::Shadow,
        ));
        router.reroute(ProviderRoutingMode::Full, None, false);
        router
    }
}
