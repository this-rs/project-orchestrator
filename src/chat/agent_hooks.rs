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
//! | `before_compaction` | `PreCompact` | the callbacks' `systemMessage` (where the notifier puts the instructions it built: the field the CLI appends to the summary's instructions and also shows to the user in its "Compacted…" line; it ignores `reason`, M2c) |
//! | `session_start` | `SessionStart` | `hookSpecificOutput.additionalContext`, `systemMessage` (contract v7) |
//! | `user_prompt_submit` | `UserPromptSubmit` | `decision: block` + `reason`, `hookSpecificOutput.additionalContext`, `systemMessage` |
//! | `before_stop` / `before_subagent_stop` | `Stop` / `SubagentStop` | `decision: block` + `reason` (the turn goes on with the reason), `systemMessage` |
//! | `notification` | `Notification` | nothing (observation only, as on the CLI) |
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
    CompactionInfo, HookReply, HookVerdict, NotificationInfo, PromptInfo, SessionHooks,
    SessionStartInfo, StopInfo, SubagentStopInfo, ToolCallInfo, ToolResultInfo, TurnContext,
    TurnDirective,
};
use nexus_claude::{
    HookContext, HookInput, HookJSONOutput, HookMatcher, NotificationHookInput,
    PostToolUseHookInput, PreCompactHookInput, PreToolUseHookInput, SessionStartHookInput,
    StopHookInput, SubagentStopHookInput, UserPromptSubmitHookInput,
};
use serde_json::Value;
use tracing::{debug, warn};
use uuid::Uuid;

use super::provider::cognitive::auto_models;
use super::provider::cognitive::candidates::{self, ModelFacts, Slot};
use super::provider::cognitive::decider::HYSTERESIS_MARGIN;
use super::provider::cognitive::decision::{CognitiveDecision, DecideRequest, Decider, Pick};
use super::provider::cognitive::observe;
use super::provider::cognitive::segment::{
    Boundary, Dimension, SegmentError, SegmentPolicy, Segmenter, SegmentsSnapshot,
};
use super::provider::cognitive::segment_store::SegmentStore;
use super::provider::cognitive::signature::{ContextHints, TaskSignature};
use super::provider::cognitive::topic;
use super::provider::cognitive::{
    conversation_stage, ConversationRoutingMode, LearningStage, ProviderRoutingMode,
    RoutingSettings,
};
use super::types::{
    pool_spans_providers, EffectiveCapability, EffectiveCause, EffectiveSource, RoutingPoolEntry,
};

/// The table the Claude path hands to its CLI: event name → matchers.
pub(crate) type HookTable = HashMap<String, Vec<HookMatcher>>;

pub(crate) const PRE_TOOL_USE: &str = "PreToolUse";
pub(crate) const POST_TOOL_USE: &str = "PostToolUse";
pub(crate) const PRE_COMPACT: &str = "PreCompact";
pub(crate) const SESSION_START: &str = "SessionStart";
pub(crate) const USER_PROMPT_SUBMIT: &str = "UserPromptSubmit";
pub(crate) const STOP: &str = "Stop";
pub(crate) const SUBAGENT_STOP: &str = "SubagentStop";
pub(crate) const NOTIFICATION: &str = "Notification";

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
    /// Compaction instructions: a `PreCompact` callback's `systemMessage`.
    instructions: Option<String>,
}

fn read(output: HookJSONOutput) -> Said {
    use nexus_claude::HookSpecificOutput;
    let HookJSONOutput::Sync(sync) = output else {
        return Said::default();
    };
    let mut said = Said {
        instructions: sync.system_message.clone(),
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
        let mut instructions = Vec::new();
        for hook in self.callbacks(PRE_COMPACT, None) {
            if let Some(output) = self.run(hook, &input, None, PRE_COMPACT).await {
                instructions.extend(read(output).instructions);
            }
        }
        joined(instructions)
    }

    async fn before_turn(&self, ctx: &TurnContext) -> TurnDirective {
        match &self.router {
            Some(router) => directive_for_turn(router, ctx).await,
            None => TurnDirective::none(),
        }
    }

    async fn session_start(&self, info: &SessionStartInfo) -> HookReply {
        let input = HookInput::SessionStart(SessionStartHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            source: info.source.clone(),
        });
        self.lifecycle(SESSION_START, &input).await
    }

    async fn user_prompt_submit(&self, prompt: &PromptInfo) -> HookReply {
        let input = HookInput::UserPromptSubmit(UserPromptSubmitHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            prompt: prompt.prompt.clone(),
        });
        self.lifecycle(USER_PROMPT_SUBMIT, &input).await
    }

    async fn before_stop(&self, info: &StopInfo) -> HookReply {
        let input = HookInput::Stop(StopHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            stop_hook_active: info.stop_hook_active,
            last_assistant_message: info.last_assistant_message.clone(),
        });
        self.lifecycle(STOP, &input).await
    }

    async fn before_subagent_stop(&self, info: &SubagentStopInfo) -> HookReply {
        let input = HookInput::SubagentStop(SubagentStopHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            stop_hook_active: info.stop_hook_active,
            last_assistant_message: info.last_assistant_message.clone(),
            agent_id: info.agent_id.clone(),
            agent_type: info.agent_type.clone(),
        });
        self.lifecycle(SUBAGENT_STOP, &input).await
    }

    async fn notification(&self, info: &NotificationInfo) {
        let input = HookInput::Notification(NotificationHookInput {
            session_id: self.session_id.clone(),
            transcript_path: String::new(),
            cwd: self.cwd.clone(),
            permission_mode: self.permission_mode.clone(),
            message: info.message.clone(),
            title: info.title.clone(),
            notification_type: info.notification_type.clone(),
        });
        // Observation only, as on the CLI: what the callbacks answer is not read.
        self.lifecycle(NOTIFICATION, &input).await;
    }
}

impl GraphSessionHooks {
    /// Runs the table's callbacks for one of Claude Code's own events (contract v7) and
    /// reads what the CLI would read: `decision: "block"` + `reason`, the event's
    /// `additionalContext`, `systemMessage`. Several callbacks: the texts are joined; one
    /// block blocks. A failing callback is skipped.
    async fn lifecycle(&self, event: &str, input: &HookInput) -> HookReply {
        use nexus_claude::HookSpecificOutput;
        let (mut blocks, mut contexts, mut messages) = (Vec::new(), Vec::new(), Vec::new());
        for hook in self.callbacks(event, None) {
            let Some(HookJSONOutput::Sync(sync)) = self.run(hook, input, None, event).await else {
                continue;
            };
            if sync.decision.as_deref() == Some("block") {
                blocks.push(
                    sync.reason
                        .clone()
                        .unwrap_or_else(|| "blocked by a hook".to_owned()),
                );
            }
            messages.extend(sync.system_message);
            match sync.hook_specific_output {
                Some(HookSpecificOutput::UserPromptSubmit(output)) => {
                    contexts.extend(output.additional_context)
                }
                Some(HookSpecificOutput::SessionStart(output)) => {
                    contexts.extend(output.additional_context)
                }
                _ => {}
            }
        }
        let mut reply = HookReply::none();
        reply.block = joined(blocks);
        reply.additional_context = joined(contexts);
        reply.system_message = joined(messages);
        reply
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
    /// Class (`TaskSignature::arm_key`) of the turn whose decision made the last change of
    /// model: the next turn of the SAME class is decided without exploration, so that only
    /// the hysteresis margin can move it again (R-S3, [`held_by_class`]).
    last_change_class: Option<String>,
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
    /// The pick may be applied (stage `auto`).
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
    /// is decided among this provider's models only (no two PROVIDER moves in a row, R-S3);
    /// its model may still change.
    moved_in: AtomicBool,
    predecided: Mutex<Option<PreDecided>>,
    /// The last decision asked for a turn, with the turn's index: the decision a change of
    /// model comes from, which the host marks not applied when the change cannot be sent
    /// ([`TurnRouter::decision_of_turn`]).
    decided: Mutex<Option<(u32, CognitiveDecision)>>,
    last_message: Mutex<Option<String>>,
    /// The next turn carries an image (`set_turn_input`): its decision excludes the models
    /// that cannot read one (`needs_images`, `RejectReason::NoImages`).
    last_has_images: AtomicBool,
    /// The turn [`model_for_images`] changed the model for, and the model in force before:
    /// what [`TurnRouter::forget_image_change`] puts back when the host could not apply it.
    image_change: Mutex<Option<(u32, String)>>,
    state: Mutex<TurnState>,
    /// The conversation's segmenter: one [`Observation`] per turn, built from the offline
    /// signature and the topic labels of the stored message (see [`TurnRouter::observe_turn`]).
    /// Its boundaries are the finer-grained "the conversation changed direction" signal that
    /// augments the coarse [`TurnRouter::moved_in`] flag: a boundary on any axis but the
    /// harness relaxes the R-S3 "no two provider moves in a row" guard, while a harness
    /// boundary is an explicit provider change (which keeps that guard).
    segmenter: Mutex<Segmenter>,
    /// The segmenter's own turn base, decoupled from the harness's `ctx.turn_index` (which
    /// resets to 0 on a provider move). It is the turn id the segmenter assigns to the
    /// harness's turn 0: `0` on a fresh conversation (the harness's index IS the segmenter's),
    /// and the carried snapshot's `last_turn` after a move. [`TurnRouter::observe_turn`]
    /// feeds the harness's turn shifted by this base, so observations stay strictly
    /// increasing across a move — the segmenter keeps its ids/`PRECEDES` stable and never
    /// sees `OutOfOrder`.
    segment_turn: Mutex<u32>,
    /// Where the confirmed boundaries (and the segment snapshot) are persisted, when a
    /// graph-backed store is attached (`CognitiveRouting.segment_store`). `None` keeps them
    /// in the segmenter only (the in-memory default). Writes are best-effort and
    /// non-blocking; replay/hydration from the store is a follow-up (out of scope).
    segment_store: Option<Arc<dyn SegmentStore>>,
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
    /// The models the conversation's PERSON lets Auto choose among (AUTO-L), `None` = every
    /// available model. Applied only to an Auto conversation without a pool of its own: a
    /// conversation's ticked models always win ([`person_list`]).
    auto_models: Option<Vec<RoutingPoolEntry>>,
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
    /// The person's "models Auto may use" (AUTO-L), `None` = all.
    pub auto_models: Option<Vec<RoutingPoolEntry>>,
    /// The conversation asked for Auto or a pool (decision R-S1).
    pub conversation_routes: bool,
    pub current_model: String,
    /// Index the next turn counted by the router itself gets (legacy engine).
    pub next_turn: u32,
    /// The session continues a conversation the router moved here (`moved_by: auto`).
    pub moved_in: bool,
    /// Where the conversation's segments and boundaries are persisted on a confirmed
    /// boundary (`None` = in-memory only). Threaded from `CognitiveRouting.segment_store`.
    pub segment_store: Option<Arc<dyn SegmentStore>>,
    /// The conversation's persisted segmentation, read once when its router is built
    /// (`None` = a fresh conversation, or nothing to resume). [`Segmenter::resume`]
    /// rebuilds the segmenter from it, so a conversation reopened after a restart
    /// continues where it stopped.
    pub segments: Option<SegmentsSnapshot>,
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The last turn a snapshot segmented, if any: the max of every segment's `last_turn` and
/// every boundary's `at_turn`, the same notion [`Segmenter::resume`] restores as its own
/// `last_turn`. `None` when the snapshot is empty (nothing to continue from).
fn snapshot_last_turn(snapshot: &SegmentsSnapshot) -> Option<u32> {
    snapshot
        .segments
        .iter()
        .map(|segment| segment.last_turn)
        .chain(snapshot.boundaries.iter().map(|boundary| boundary.at_turn))
        .max()
}

impl TurnRouter {
    pub(crate) fn new(spec: TurnRouterSpec) -> Self {
        // The relayed turn runs on the pair the move chose: nothing to ask. It is always the
        // target provider's FIRST turn (index 0, whatever the conversation's carried counter
        // was), so the predecision keys on 0 — the same index the host's hook sends — and the
        // R-S3 guard below keys on the provider's 0-based turn, not the carried counter.
        let predecided = spec.moved_in.then_some(PreDecided {
            turn_index: 0,
            pick: None,
            apply: false,
        });
        let settings_stage = spec.routing.stage;
        let mut routing = spec.routing;
        routing.stage = conversation_stage(settings_stage, spec.conversation_routes);
        // The segment ids are derived deterministically from the session id, so two routers
        // of the same session segment byte-for-byte the same; without a session (a legacy
        // engine run without a conversation id), the seed collapses to 0.
        // Fold all 128 bits of the session id into 64, so two distinct sessions do not
        // collapse to the same seed (and thus the same segment ids) by a low-bits collision.
        let seed = spec
            .session_id
            .map(|id| {
                let v = id.as_u128();
                (v as u64) ^ (v >> 64) as u64
            })
            .unwrap_or(0);
        // The segmenter's turn base: the carried snapshot's `last_turn` (NOT `last_turn + 1`).
        // The relayed turn re-observes the message the move's own turn already carried, so it
        // must land on the SAME segment id as that turn (the segmenter then tolerates it as a
        // repeat); `last_turn + 1` made it a fresh turn, which misread the relay as a new
        // segment and relaxed the "no two moves in a row" guard with a false direction change.
        let segment_turn = spec
            .segments
            .as_ref()
            .and_then(snapshot_last_turn)
            .map_or(0, |last| last);
        Self {
            decider: spec.decider,
            pool: spec.pool,
            choice: Mutex::new(RouterChoice {
                routing,
                explicit_model: spec.explicit_model,
                provider_imposed: spec.provider_imposed,
                allowed_models: spec.allowed_models,
                routing_pool: spec.routing_pool,
                auto_models: spec.auto_models,
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
            last_has_images: AtomicBool::new(false),
            image_change: Mutex::new(None),
            state: Mutex::new(TurnState {
                current_model: spec.current_model,
                last_change_turn: None,
                last_change_class: None,
                next_turn: spec.next_turn,
            }),
            // S4: resume the conversation's persisted segmentation instead of starting
            // from an empty history. A snapshot the model refuses (an incoherent one)
            // starts fresh, with a warning — never a failed turn.
            segmenter: Mutex::new(match spec.segments {
                Some(snapshot) => Segmenter::resume(SegmentPolicy::default(), seed, &snapshot)
                    .unwrap_or_else(|error| {
                        warn!(%error, "unreadable conversation segments: fresh segmentation");
                        Segmenter::new(SegmentPolicy::default(), seed)
                    }),
                None => Segmenter::new(SegmentPolicy::default(), seed),
            }),
            segment_turn: Mutex::new(segment_turn),
            segment_store: spec.segment_store,
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
    /// the pool) restricts the candidates, `routing_pool` (every ticked pair, whatever its
    /// provider) is what a provider move among the pool is decided from (R-S2), and
    /// `explicit_model` pins the current model. Whatever was set by hand before is
    /// forgotten: this IS the new choice by hand, and a provider imposed at the opening is
    /// released (`auto` lets PO choose the provider again; a pool ticked across providers
    /// lets it move among those; the other choices never move the conversation).
    pub(crate) fn reroute(
        &self,
        mode: ConversationRoutingMode,
        allowed_models: Option<Vec<String>>,
        routing_pool: Option<Vec<RoutingPoolEntry>>,
        explicit_model: bool,
    ) {
        let mut choice = locked(&self.choice);
        choice.routing.mode = mode.pilot_mode();
        choice.allowed_models = allowed_models;
        choice.routing_pool = routing_pool;
        choice.explicit_model = explicit_model;
        choice.provider_imposed = false;
        // Auto, or a pool (`allowed_models` is only given for two models or more): the
        // user's choice, applied at the `auto` stage (decision R-S1).
        let conversation_routes =
            mode == ConversationRoutingMode::Full || choice.allowed_models.is_some();
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

    /// The SEGMENTER's next turn id (the conversation's `last_turn + 1`), carried across a
    /// provider move so the reopened router's segmenter continues where this one stopped
    /// (its ids/`PRECEDES` stay byte-identical under the unchanged session seed). This is NOT
    /// the R-S3 seed: the relayed turn of a moved-in router keys on the provider's own
    /// 0-based turn (see [`TurnRouter::new`] and [`TurnRouter::seed_relayed_turn`]).
    pub(crate) fn next_turn(&self) -> u32 {
        snapshot_last_turn(&self.snapshot_segments()).map_or(0, |last| last + 1)
    }

    /// The conversation's segmentation as it stands (its segments and boundaries), for
    /// carrying across a provider move: the new router resumes from it instead of starting
    /// an empty history, so a move does not reset the cognitive segments.
    pub(crate) fn snapshot_segments(&self) -> SegmentsSnapshot {
        let segmenter = locked(&self.segmenter);
        SegmentsSnapshot {
            segments: segmenter.segments(),
            boundaries: segmenter.boundaries().to_vec(),
        }
    }

    /// The model [`model_for_images`] chose could not be made active (the session refused
    /// it, or closed): the model did not change, and the decision kept for the turn is dropped,
    /// so the next turn (with or without an image) is decided afresh, never handed a change
    /// nobody decided for it.
    ///
    /// Returns the decision the change came from, which the host stores back as NOT applied
    /// (`not_changed:`, as a refused move or a legacy `set_model` that failed): the decision
    /// said `applied: true`, and the model did not change.
    pub(crate) fn forget_image_change(&self) -> Option<CognitiveDecision> {
        let (turn, before) = locked(&self.image_change).take()?;
        self.forget_change(&before);
        *locked(&self.predecided) = None;
        self.decision_of_turn(turn)
    }

    /// The decision asked for turn `turn_index`, if it is the last one asked: the decision
    /// a change of that turn came from.
    pub(crate) fn decision_of_turn(&self, turn_index: u32) -> Option<CognitiveDecision> {
        locked(&self.decided)
            .take()
            .filter(|(turn, _)| *turn == turn_index)
            .map(|(_, decision)| decision)
    }

    /// The conversation reached this session by a provider move (`moved_in`): its relayed
    /// turn runs on the pair the move chose, which is a change of model, recorded with the
    /// class of that turn's message (set just before, [`Self::set_last_message`]). Turn 1 of
    /// the same class is then steady (R-S3, [`held_by_class`]), on top of being decided on
    /// this provider only.
    ///
    /// The relayed turn is always the target provider's FIRST turn, index 0 (the provider
    /// resets its own turn counter, whatever the conversation's carried counter was): the
    /// change is recorded AT 0, so `held_by_class` fires for turn 1 of the same class.
    pub(crate) fn seed_relayed_turn(&self) {
        if !self.moved_in.load(Ordering::SeqCst) {
            return;
        }
        let class = self.signature().arm_key();
        let mut state = locked(&self.state);
        let model = state.current_model.clone();
        // The provider's 0-based index, not the carried `state.next_turn`: the R-S3 guard
        // keys on the turn the host actually sends (0), so the next turn (1) is guarded.
        let turn = 0;
        record_change(&mut state, turn, &model, class);
    }

    /// An applied change that could not be sent: the model did not change.
    pub(crate) fn forget_change(&self, model_before: &str) {
        let mut state = locked(&self.state);
        state.current_model = model_before.to_owned();
        state.last_change_turn = None;
        state.last_change_class = None;
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

    /// Feeds the turn into the session's segmenter and returns the boundaries it confirmed
    /// (the segmenter keeps the whole history; this is what the routing of the turn reads).
    /// The observation is built from the offline signature and the topic labels of the stored
    /// message: the harness axis carries `provider_id`, the session's own in production. A
    /// turn that does not strictly increase (the same turn seen by both the provider check and
    /// the hook) is tolerated: logged and dropped, it changes nothing and never breaks routing.
    fn observe_turn(&self, turn_index: u32, provider_id: &str) -> Vec<Boundary> {
        let message = locked(&self.last_message).clone().unwrap_or_default();
        let topic = topic::topic_labels(&message);
        // The segmenter's own turn id: the harness's index shifted by the segmenter's base.
        // On a fresh router the base is 0 (the harness's index IS the segmenter's, unchanged);
        // after a move the base is the carried `last_turn`, so the harness's reset index
        // (0, 1, …) still segments as `last_turn, last_turn + 1, …`. The segmenter's own
        // `last_turn` check still dedups a turn seen by both the provider check and the hook
        // (both yield the same shifted id).
        let segment_turn = turn_index.saturating_add(*locked(&self.segment_turn));
        let observation =
            observe::observation(segment_turn, provider_id, &self.signature(), &topic);
        let boundaries = match locked(&self.segmenter).observe(&observation) {
            Ok(boundaries) => boundaries,
            Err(SegmentError::OutOfOrder { turn, last }) => {
                debug!(provider = %self.provider_id, turn, last, "the turn was already observed: the segmenter is not re-fed");
                Vec::new()
            }
            Err(error) => {
                warn!(provider = %self.provider_id, error = %error, "segment observation refused");
                Vec::new()
            }
        };
        // A confirmed boundary means the conversation's episodic memory changed: snapshot
        // the segments and boundaries and hand them to the store. Best-effort and
        // non-blocking (the store is async): the turn never waits on the write, and a
        // failing write only logs. Persisted on the boundary only; replay/hydration from
        // the store is a follow-up (out of scope).
        if !boundaries.is_empty() {
            self.persist_segments();
        }
        boundaries
    }

    /// Snapshots the segmenter and persists it on a separate task, best-effort: the turn is
    /// never blocked on the graph write. Segments are written before boundaries so the
    /// store's `PRECEDES` links between segment nodes have their endpoints.
    fn persist_segments(&self) {
        let Some(store) = self.segment_store.clone() else {
            return;
        };
        // The conversation key is the session id; without one (a legacy engine run with no
        // conversation id) there is no identity to key the store by, so nothing is written
        // (the segment ids collapse to seed 0 there too).
        let Some(conversation) = self.session_id.map(|id| id.to_string()) else {
            return;
        };
        let segments = locked(&self.segmenter).segments();
        let boundaries = locked(&self.segmenter).boundaries().to_vec();
        tokio::spawn(async move {
            if let Err(error) = store.put_segments(&conversation, &segments).await {
                warn!(conversation = %conversation, error = %error, "conversation segments not persisted");
            }
            if let Err(error) = store.put_boundaries(&conversation, &boundaries).await {
                warn!(conversation = %conversation, error = %error, "conversation boundaries not persisted");
            }
        });
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

/// Forgets, when dropped, that the turn carries images.
struct ImagesOfTheTurn<'a>(&'a TurnRouter);

impl Drop for ImagesOfTheTurn<'_> {
    fn drop(&mut self) {
        self.0.last_has_images.store(false, Ordering::SeqCst);
        locked(&self.0.image_change).take();
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

/// The per-class guard of turn `turn` (R-S3): the PREVIOUS turn changed the model (a
/// provider move included: the relayed turn is a change, [`TurnRouter::seed_relayed_turn`]),
/// and it was of the same class as this one.
///
/// There is no "never two changes in a row" any more: a conversation that alternates a
/// simple question and a heavy task is routed on the turn itself, not one turn late. A turn
/// of ANOTHER class than the change is routed as any turn. A turn of the SAME class is a
/// steady decision ([`narrow`]): no exploration, and the decider keeps the pair in force
/// unless a challenger beats it by [`HYSTERESIS_MARGIN`] on its posterior MEAN — a bound
/// that does not depend on the draws, so two same-class turns never go A → B → A.
fn held_by_class(state: &TurnState, turn: u32, class: &str) -> bool {
    state
        .last_change_turn
        .is_some_and(|t| t.checked_add(1) == Some(turn))
        && state.last_change_class.as_deref() == Some(class)
}

/// Reason suffix of a turn [`held_by_class`].
pub(crate) const SAME_CLASS_NO_EXPLORE: &str = "same_class_no_explore";
/// Reason suffix of the turn right after a provider move (no second move in a row).
pub(crate) const AFTER_PROVIDER_MOVE: &str = "after_provider_move";

/// Narrows `request` by the R-S3 guards, and says so in the reason it will be stored with:
/// a turn [`held_by_class`] is steady (no exploration, hysteresis on the posterior means);
/// the turn right after a provider move is decided among this provider's models only.
fn narrow(request: &mut DecideRequest, held: bool, after_move: Option<&str>) {
    if held {
        request.settings.exploration_epsilon = 0.0;
        request.steady = true;
        request.narrowed.push(SAME_CLASS_NO_EXPLORE);
    }
    if let Some(provider) = after_move {
        request.restrict_provider = Some(provider.to_owned());
        request.narrowed.push(AFTER_PROVIDER_MOVE);
    }
}

/// Records a change of model made on `turn` by a decision of class `class`.
fn record_change(state: &mut TurnState, turn: u32, model: &str, class: String) {
    state.current_model = model.to_owned();
    state.last_change_turn = Some(turn);
    state.last_change_class = Some(class);
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
/// `full`, or a pool ticked across two providers or more — decision R-S2). Called before
/// the turn starts, by the host's send path: a session runs on one provider, so a move is
/// never a [`TurnDirective`].
///
/// * neither `full` nor a pool spanning providers (`mixed` on one provider stays there), a
///   model or a provider imposed (`routed_by: request`, pinned), a model changed by hand, a
///   source that knows one provider only: nothing is asked, [`ProviderPlan::Stay`];
/// * otherwise the decider chooses among every model the project consented to (the
///   conversation's pool when it has one), the current pair in force. The decision is kept
///   for the turn (the hook does not ask again) and recorded applied or not;
/// * a move needs the stage `auto`, no move on the previous turn (the turn right after a
///   move is decided among the models of its provider: its model may change, its provider
///   not), a pick that is not an exploration draw and that beats the current pair by the
///   hysteresis margin. Anything else stays. Right after a change of model, a turn of the
///   same class is decided without exploration ([`held_by_class`], R-S3).
pub(crate) async fn plan_provider_move(router: &TurnRouter) -> ProviderPlan {
    let choice = locked(&router.choice).clone();
    let routing = routing_of(router, &choice);
    if !routing.provider {
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
    let signature = router.signature();
    // AUTO-L: an Auto conversation without a pool of its own chooses among its person's list.
    let person = apply_person_list(&choice, &mut pool, &signature);
    if !pool
        .iter()
        .any(|facts| facts.provider_id != router.provider_id)
    {
        return ProviderPlan::Stay;
    }
    let (turn, current_model) = {
        let state = locked(&router.state);
        (state.next_turn, state.current_model.clone())
    };
    // No two PROVIDER moves in a row (decision R-S3: a move relays the history to a new
    // session, a model change does not): the turn right after a move is decided among the
    // models of the provider it moved to, so its model may still change.
    let just_moved = router.moved_in.swap(false, Ordering::SeqCst);
    // The segmenter sees this turn once (a turn already observed by the hook is a no-op):
    // a boundary on any axis but the harness means the conversation changed direction, and
    // that relaxes the "no two moves in a row" guard below — a turn that really changed
    // direction may re-evaluate the provider even right after a move. A harness boundary is
    // an explicit provider change, so it keeps the guard (`direction_changed` excludes it).
    let boundaries = router.observe_turn(turn, &router.provider_id);
    let direction_changed = boundaries
        .iter()
        .any(|boundary| boundary.dimension != Dimension::Harness);
    // R-S3: `held_by_class` is keyed on the CLASS (`arm_key`), which a Problem/Resolution
    // boundary already reflects (both carry `class.key()`): a class change lifts the steady
    // guard naturally. `direction_changed` only relaxes `may_move` below (the provider
    // question re-opens right after a move) — it must NOT lift `held_by_class`, because a
    // TOPIC-only change keeps the same class and thus the same steady model choice.
    let held = held_by_class(&locked(&router.state), turn, &signature.arm_key());
    let segment_note = (!boundaries.is_empty()).then(|| {
        boundaries
            .iter()
            .map(|boundary| {
                format!(
                    "boundary({})@{}",
                    boundary.dimension.as_str(),
                    boundary.at_turn
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    });
    // F-R4: a turn the pair in force cannot read must leave it, whatever the anti-flapping.
    let blind = cannot_read_turn(router, &pool, &router.provider_id, &current_model);
    if blind && !router.set_model_live.load(Ordering::SeqCst) {
        // The model of THIS provider cannot be changed live: its other models are no way to
        // read the image, only a move is (the effective capability counts them out too).
        pool.retain(|f| {
            f.provider_id != router.provider_id || same_model(&f.model, &current_model)
        });
    }
    let may_move = blind || !just_moved || direction_changed;
    let apply = routing.auto;
    let current = Pick::new(&router.provider_id, &current_model);
    let mut settings = settings_for(&choice.routing, apply);
    // A pool ticked across two providers or more is routed like `full` restricted to the
    // pool (R-S2): the decision is applied (a move) whatever mode the ticks map to, exactly
    // as a model change inside a pool is (`directive_for_turn`). Without this the decider
    // answers `applied: false` under `mixed` (`is_applied`: `primary` applies nothing) and
    // `routing_of`'s opening above would be inert.
    if pool_spans_providers(choice.routing_pool.as_deref()) {
        settings.mode = ProviderRoutingMode::Full;
    }
    let mut request = DecideRequest::new(signature, settings, pool);
    request.narrowed.extend(person);
    narrow(
        &mut request,
        held,
        (!may_move).then_some(router.provider_id.as_str()),
    );
    request.trust = router.trust;
    request.current = Some(current.clone());
    request.session_id = router.session_id;
    request.turn_index = Some(turn);
    request.segment_note = segment_note;
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
        Some(pick) if apply && may_move && pick.provider_id != router.provider_id => pick.clone(),
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
/// * `full` mode before the `auto` stage: the decider is still asked, so the decision is
///   recorded, but nothing is applied;
/// * `full` + `auto`: the pick is used when the decision is applied and differs from the
///   current model, on the turn itself, even right after a change (R-S3: a turn of the same
///   class as that change is decided without exploration, so only the hysteresis margin can
///   move it again). A failing or slow decider (2 s) never fails the turn.
pub(crate) async fn directive_for_turn(router: &TurnRouter, ctx: &TurnContext) -> TurnDirective {
    // What the turn carries is the turn's own: whatever this returns, a later decision of the
    // same session (a compaction during the turn, the next turn before its input is set) never
    // reads this turn's images.
    let _images_read = ImagesOfTheTurn(router);
    {
        // The harness knows the model the turn would run on, whoever changed it, and the
        // index of the turn.
        let mut state = locked(&router.state);
        state.current_model = ctx.current_model.clone();
        state.next_turn = ctx.turn_index.saturating_add(1);
    }
    // Every turn feeds the segmenter exactly once. A turn already observed by the provider
    // check (mode `full`) is a repeated turn here and is tolerated as a no-op; the sessions
    // the provider check never runs for (mode `mixed`, one provider) are observed here.
    // Observe here for mode `mixed` (the provider check never runs); it feeds the segmenter
    // once and persists segments on a confirmed boundary. The boundary list itself is not
    // needed for the model decision: `held_by_class` is keyed on the class (arm_key).
    let _ = router.observe_turn(ctx.turn_index, &router.provider_id);
    let predecided = locked(&router.predecided)
        .take()
        .filter(|pre| pre.turn_index == ctx.turn_index);
    let choice = locked(&router.choice).clone();
    let routing = routing_of(router, &choice);
    if !routing.model {
        return TurnDirective::none();
    }
    let signature = router.signature();
    let class = signature.arm_key();
    let pick = match predecided {
        Some(pre) => pre.pick.filter(|_| pre.apply),
        None => {
            // R-S3: right after a change, a turn of the same class is decided without
            // exploration (the hysteresis margin alone may move it); another class is routed
            // as any turn. A class change (Problem/Resolution boundary) already lifts
            // `held_by_class` through the class itself.
            let held = held_by_class(&locked(&router.state), ctx.turn_index, &class);
            let (mut pool, project) = provider_pool(router).await;
            if let Some(allowed) = &choice.allowed_models {
                pool.retain(|facts| allowed.iter().any(|m| m == &facts.model));
            }
            // AUTO-L: the person's list; none of it on this provider leaves nothing to change
            // here (a provider move is the way to it).
            let person = apply_person_list_here(&choice, &mut pool, &project, &signature);
            if person == Some(auto_models::NARROWED_BY_PERSON) && pool.is_empty() {
                return TurnDirective::none();
            }
            let apply = routing.auto;
            let mut settings = settings_for(&choice.routing, apply);
            // A pool is routed like `full` inside it, whatever the settings' mode.
            if choice.allowed_models.is_some() {
                settings.mode = ProviderRoutingMode::Full;
            }
            let mut request = DecideRequest::new(signature, settings, pool);
            request.trust = router.trust;
            request.restrict_provider = Some(router.provider_id.clone());
            request.current = Some(Pick::new(&router.provider_id, &ctx.current_model));
            request.narrowed.extend(person);
            narrow(&mut request, held, None);
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
            debug!(model = %pick.model, turn = ctx.turn_index, class = %class, "turn routing changes the model");
            record_change(
                &mut locked(&router.state),
                ctx.turn_index,
                &pick.model,
                class,
            );
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
/// narrowed to the provider, is asked first; a source that has none answers `pool`. Also
/// returns the project's pool (every provider), against which the person's list is read.
async fn provider_pool(router: &TurnRouter) -> (Vec<ModelFacts>, Vec<ModelFacts>) {
    let project = tokio::time::timeout(
        router.timeout,
        router.pool.project_pool(router.project_slug.as_deref()),
    )
    .await
    .unwrap_or_default();
    let mine: Vec<ModelFacts> = project
        .iter()
        .filter(|facts| facts.provider_id == router.provider_id)
        .cloned()
        .collect();
    if mine.is_empty() {
        (router.pool.pool(&router.provider_id).await, project)
    } else {
        (mine, project)
    }
}

/// Whether two model ids name the same model: the context-window suffix (`[1m]`) and case
/// are not part of the name.
fn same_model(a: &str, b: &str) -> bool {
    fn base(id: &str) -> &str {
        let id = id.trim();
        id.strip_suffix("[1m]").unwrap_or(id).trim_end()
    }
    base(a).eq_ignore_ascii_case(base(b))
}

/// The turn about to start carries an image and the pool KNOWS the pair in force cannot read
/// it. Staying is then no option: the anti-flapping rules (no two provider moves in a row, no
/// move on an exploration draw) give way to the hard constraint. A pair the pool does not list (an
/// alias, a catalogue that does not name it, a pool not built) is NOT blind: nothing says it
/// cannot read the image, so nothing overrides the guards — the provider refuses honestly if
/// it really cannot.
fn cannot_read_turn(router: &TurnRouter, pool: &[ModelFacts], provider: &str, model: &str) -> bool {
    router.turn_has_images()
        && pool
            .iter()
            .any(|f| f.provider_id == provider && same_model(&f.model, model) && !f.supports_images)
}

/// What PO may do on this conversation's turns — the ONE reading of the routing rules, shared by
/// [`directive_for_turn`], [`plan_provider_move`], [`model_for_images`] and [`effective_images`],
/// so the capability promised is exactly what the routing applies.
#[derive(Debug, Clone, Copy)]
struct Routing {
    /// PO chooses the MODEL of the turns on the session's provider: `full` or a pool ticked
    /// (with a model of this provider), nothing imposed nor changed by hand, a session that can
    /// switch model live.
    model: bool,
    /// PO may move the conversation to ANOTHER provider: `full`, or a pool ticked across two
    /// providers or more (routed like `full` restricted to the pool, decision R-S2); neither
    /// model nor provider imposed, nothing changed by hand.
    provider: bool,
    /// The stage in force for this conversation applies the decisions (`auto`); otherwise they
    /// are only recorded.
    auto: bool,
}

/// The stage this conversation's decisions are taken at. Decision R-S1 (#702) makes it the
/// conversation's own: `choice.routing.stage` holds it (`auto` for an Auto conversation or a
/// pool, the settings' stage otherwise, [`conversation_stage`]); read here only.
fn effective_stage(choice: &RouterChoice) -> LearningStage {
    choice.routing.stage
}

fn routing_of(router: &TurnRouter, choice: &RouterChoice) -> Routing {
    let free = !choice.explicit_model && !router.manual.load(Ordering::SeqCst);
    let full = choice.routing.mode == ProviderRoutingMode::Full;
    Routing {
        model: free
            && (full || choice.allowed_models.is_some())
            // A pool none of whose models is on this provider: nothing to route among here.
            && !choice.allowed_models.as_ref().is_some_and(Vec::is_empty)
            && router.set_model_live.load(Ordering::SeqCst),
        provider: free
            && !choice.provider_imposed
            && (full || pool_spans_providers(choice.routing_pool.as_deref())),
        auto: effective_stage(choice) == LearningStage::Auto,
    }
}

/// The conversation's own pool (two models ticked or more), when it has one.
fn conversation_pool(choice: &RouterChoice) -> Option<&[RoutingPoolEntry]> {
    choice.routing_pool.as_deref().filter(|pool| pool.len() > 1)
}

/// The person's "models Auto may use" (AUTO-L), when they apply to this conversation: Auto
/// (`full`) and no pool of its own (a conversation's ticks always win).
fn person_list(choice: &RouterChoice) -> Option<&[RoutingPoolEntry]> {
    if choice.routing.mode != ProviderRoutingMode::Full
        || conversation_pool(choice).is_some()
        || choice.allowed_models.is_some()
    {
        return None;
    }
    choice
        .auto_models
        .as_deref()
        .filter(|list| !list.is_empty())
}

/// Narrows `pool` (every provider's models) to the person's list when it applies, and says so
/// on `request` (`narrowed: auto_models`, or `narrowed: auto_models_unavailable` when nothing
/// listed can serve and every model stays a candidate).
fn apply_person_list(
    choice: &RouterChoice,
    pool: &mut Vec<ModelFacts>,
    signature: &TaskSignature,
) -> Option<&'static str> {
    let list = person_list(choice)?;
    Some(auto_models::restrict(pool, list, signature))
}

/// The person's list applied to a decision on the session's provider only: `pool` is that
/// provider's models; `project` every provider's. When the list can serve somewhere, this
/// provider keeps only its listed models (possibly none: only a move reaches the list); when it
/// can serve nowhere, every model stays (the fallback, stated).
fn apply_person_list_here(
    choice: &RouterChoice,
    pool: &mut Vec<ModelFacts>,
    project: &[ModelFacts],
    signature: &TaskSignature,
) -> Option<&'static str> {
    let list = person_list(choice)?;
    let mut everywhere = project.to_vec();
    let code = auto_models::restrict(&mut everywhere, list, signature);
    if code == auto_models::NARROWED_BY_PERSON {
        pool.retain(|f| {
            list.iter()
                .any(|e| e.provider == f.provider_id && e.model == f.model)
        });
    }
    Some(code)
}

/// The model of the session's provider an image turn must run on, chosen BEFORE the turn
/// is sent (F-R4). The native harness checks the vision of its ACTIVE model when the turn is
/// sent, before `before_turn` could change it: the host makes this model active first, and
/// the hook of the turn finds the decision taken (it does not ask again). When the host
/// cannot make it active, it calls [`TurnRouter::forget_image_change`].
///
/// `None`: nothing to change — no image, PO does not choose this conversation's model, the
/// model in force reads images (or the pool does not say it cannot), or no candidate does
/// (the provider then refuses the turn and says so: a pool without vision is never hidden).
/// A decision already taken for this turn (the provider check of mode `full`) is used as is.
pub(crate) async fn model_for_images(router: &TurnRouter) -> Option<String> {
    if !router.turn_has_images() {
        return None;
    }
    let choice = locked(&router.choice).clone();
    let routing = routing_of(router, &choice);
    if !routing.model || !routing.auto {
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
            let (mut pool, project) = provider_pool(router).await;
            if let Some(allowed) = &choice.allowed_models {
                pool.retain(|facts| allowed.iter().any(|m| m == &facts.model));
            }
            let person = apply_person_list_here(&choice, &mut pool, &project, &router.signature());
            if !cannot_read_turn(router, &pool, &router.provider_id, &current_model) {
                return None;
            }
            let mut settings = settings_for(&choice.routing, true);
            if choice.allowed_models.is_some() {
                settings.mode = ProviderRoutingMode::Full;
            }
            let signature = router.signature();
            // R-S3 (review of #710, 5): an image turn right after a change of its class is
            // steady too: never an exploration draw among the models that read images.
            let held = held_by_class(&locked(&router.state), turn, &signature.arm_key());
            let mut request = DecideRequest::new(signature, settings, pool);
            request.narrowed.extend(person);
            request.trust = router.trust;
            request.restrict_provider = Some(router.provider_id.clone());
            request.current = Some(Pick::new(&router.provider_id, &current_model));
            narrow(&mut request, held, None);
            request.session_id = router.session_id;
            request.turn_index = Some(turn);
            let decision = router.ask(request).await?;
            // Kept as the decision of the turn: the one the host marks not applied when the
            // model cannot be made active ([`TurnRouter::forget_image_change`]).
            *locked(&router.decided) = Some((turn, decision.clone()));
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
    if pick.provider_id != router.provider_id || same_model(&pick.model, &current_model) {
        return None;
    }
    *locked(&router.image_change) = Some((turn, current_model));
    let class = router.signature().arm_key();
    record_change(&mut locked(&router.state), turn, &pick.model, class);
    Some(pick.model)
}

/// Whether the next turn of this conversation may carry images, as PO can really serve it
/// (F-R4, decision 11cefdb2): the candidates PO may route the turn to govern, not the
/// snapshot of the model the session opened on. `snapshot` is what that snapshot says
/// (`None`: no snapshot stored — the legacy Claude Code engine stores none).
///
/// * the snapshot says yes: yes (`model_has_it`);
/// * PO does not apply routing on this conversation ([`routing_of`]: primary, a model imposed
///   or changed by hand, a stage that only records, a session that cannot switch model):
///   the snapshot (`not_routed`);
/// * PO routes but its pool lists no model, or could not be built in time: the snapshot
///   (`pool_unbuilt`) — not probed is not absent;
/// * otherwise: yes when at least one candidate passes the filter of an image turn
///   (`pool_has_it`, `via` names them), no when none does (`pool_lacks_it`).
///
/// `None`: nothing to say — no snapshot and no candidate reached says yes. An unknown snapshot
/// is never turned into a "no" (a Claude Code session reads images whatever its pool).
///
/// The pool is read ONCE (`project_pool`, the project's consent stated on each model), and the
/// whole answer is bounded by the router's timeout.
pub(crate) async fn effective_images(
    router: &TurnRouter,
    snapshot: Option<bool>,
) -> Option<EffectiveCapability> {
    let images = effective_images_bounded(router, snapshot == Some(true)).await;
    (snapshot.is_some() || images.value).then_some(images)
}

async fn effective_images_bounded(router: &TurnRouter, snapshot: bool) -> EffectiveCapability {
    if snapshot {
        return EffectiveCapability::snapshot(true, EffectiveCause::ModelHasIt);
    }
    let choice = locked(&router.choice).clone();
    let routing = routing_of(router, &choice);
    let (by_model, by_provider) = (
        routing.model && routing.auto,
        routing.provider && routing.auto,
    );
    if !by_model && !by_provider {
        return EffectiveCapability::snapshot(false, EffectiveCause::NotRouted);
    }
    let Ok(project) = tokio::time::timeout(
        router.timeout,
        router.pool.project_pool(router.project_slug.as_deref()),
    )
    .await
    else {
        return EffectiveCapability::snapshot(false, EffectiveCause::PoolUnbuilt);
    };
    let mut pool: Vec<ModelFacts> = project
        .into_iter()
        .filter(|facts| {
            if facts.provider_id == router.provider_id {
                // This provider's models: reachable only by a live model change.
                by_model
                    && choice
                        .allowed_models
                        .as_ref()
                        .is_none_or(|allowed| allowed.iter().any(|m| m == &facts.model))
            } else {
                by_provider
                    && choice.routing_pool.as_ref().is_none_or(|entries| {
                        entries
                            .iter()
                            .any(|e| e.provider == facts.provider_id && e.model == facts.model)
                    })
            }
        })
        .collect();
    // AUTO-L: the person's list bounds what Auto may reach (with the same fallback).
    if let Some(list) = person_list(&choice) {
        auto_models::restrict(&mut pool, list, &router.signature());
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
    async fn before_compaction_returns_the_system_message_the_notifier_builds() {
        let notifier = Scripted::answering(SyncHookJSONOutput {
            continue_: Some(true),
            system_message: Some("Preserve: decision on the hot bridge".into()),
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

    // ------------------------------------------------------------------
    // Claude Code's own events (contract v7, ISO1): the same table, two doors.
    // ------------------------------------------------------------------

    fn blocking(reason: &str) -> SyncHookJSONOutput {
        SyncHookJSONOutput {
            decision: Some("block".into()),
            reason: Some(reason.into()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn user_prompt_submit_reads_the_block_the_context_and_the_message_of_the_table() {
        use nexus_claude::agent::PromptInfo;
        use nexus_claude::UserPromptSubmitHookSpecificOutput;
        let context = Scripted::answering(SyncHookJSONOutput {
            system_message: Some("checked".into()),
            hook_specific_output: Some(HookSpecificOutput::UserPromptSubmit(
                UserPromptSubmitHookSpecificOutput {
                    additional_context: Some("GRAPH: plan 5ad54c48".into()),
                },
            )),
            ..Default::default()
        });
        let hooks_ = hooks(table(USER_PROMPT_SUBMIT, None, vec![context.clone()]));
        let reply = hooks_.user_prompt_submit(&PromptInfo::new("fix it")).await;
        assert_eq!(reply.block, None);
        assert_eq!(
            reply.additional_context.as_deref(),
            Some("GRAPH: plan 5ad54c48")
        );
        assert_eq!(reply.system_message.as_deref(), Some("checked"));
        let seen = context.seen.lock().unwrap()[0].clone();
        let HookInput::UserPromptSubmit(input) = &seen else {
            panic!("a UserPromptSubmit input is expected");
        };
        assert_eq!(input.prompt, "fix it");
        assert_eq!(input.cwd, "/work/app");

        let block = Scripted::answering(blocking("no bananas"));
        let hooks_ = hooks(table(USER_PROMPT_SUBMIT, None, vec![block]));
        let reply = hooks_.user_prompt_submit(&PromptInfo::new("bananas")).await;
        assert_eq!(reply.block.as_deref(), Some("no bananas"));
    }

    #[tokio::test]
    async fn the_stop_hooks_hand_over_the_flag_and_the_last_answer_and_return_the_block() {
        use nexus_claude::agent::{StopInfo, SubagentStopInfo};
        let stop = Scripted::answering(blocking("write the tests first"));
        let hooks_ = hooks(table(STOP, None, vec![stop.clone()]));
        let reply = hooks_
            .before_stop(&StopInfo::new(true, Some("done".into())))
            .await;
        assert_eq!(reply.block.as_deref(), Some("write the tests first"));
        let seen = stop.seen.lock().unwrap()[0].clone();
        let HookInput::Stop(input) = &seen else {
            panic!("a Stop input is expected");
        };
        assert!(input.stop_hook_active);
        assert_eq!(input.last_assistant_message.as_deref(), Some("done"));

        let sub = Scripted::answering(blocking("one more line"));
        let hooks_ = hooks(table(SUBAGENT_STOP, None, vec![sub.clone()]));
        let mut info = SubagentStopInfo::new(false, Some("sub-done".into()));
        info.agent_type = Some("general-purpose".into());
        let reply = hooks_.before_subagent_stop(&info).await;
        assert_eq!(reply.block.as_deref(), Some("one more line"));
        let seen = sub.seen.lock().unwrap()[0].clone();
        let HookInput::SubagentStop(input) = &seen else {
            panic!("a SubagentStop input is expected");
        };
        assert_eq!(input.agent_type.as_deref(), Some("general-purpose"));
        // A Stop callback is not a SubagentStop one: each event reads its own entry.
        assert_eq!(
            hooks_.before_stop(&StopInfo::new(false, None)).await.block,
            None
        );
    }

    #[tokio::test]
    async fn session_start_and_notification_reach_the_callbacks_of_their_event() {
        use nexus_claude::agent::{NotificationInfo, SessionStartInfo};
        use nexus_claude::SessionStartHookSpecificOutput;
        let start = Scripted::answering(SyncHookJSONOutput {
            hook_specific_output: Some(HookSpecificOutput::SessionStart(
                SessionStartHookSpecificOutput {
                    additional_context: Some("GRAPH: project nexus".into()),
                },
            )),
            ..Default::default()
        });
        let notified = Scripted::answering(SyncHookJSONOutput::default());
        let mut table_ = table(SESSION_START, None, vec![start.clone()]);
        table_.insert(
            NOTIFICATION.to_owned(),
            vec![HookMatcher {
                matcher: None,
                hooks: vec![notified.clone()],
            }],
        );
        let hooks_ = hooks(table_);
        let reply = hooks_.session_start(&SessionStartInfo::new("resume")).await;
        assert_eq!(
            reply.additional_context.as_deref(),
            Some("GRAPH: project nexus")
        );
        let seen = start.seen.lock().unwrap()[0].clone();
        let HookInput::SessionStart(input) = &seen else {
            panic!("a SessionStart input is expected");
        };
        assert_eq!(input.source, "resume");
        hooks_
            .notification(&NotificationInfo::permission_prompt("Bash"))
            .await;
        let seen = notified.seen.lock().unwrap()[0].clone();
        let HookInput::Notification(input) = &seen else {
            panic!("a Notification input is expected");
        };
        assert_eq!(input.notification_type, "permission_prompt");
        // Nothing registered for an event: nothing said, nothing blocked.
        let empty = hooks(HookTable::new());
        assert_eq!(
            empty.session_start(&SessionStartInfo::new("startup")).await,
            nexus_claude::agent::HookReply::none()
        );
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
            auto_models: None,
            conversation_routes: false,
            current_model: "m".into(),
            next_turn: 1,
            moved_in: false,
            segment_store: None,
            segments: None,
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

    fn entry(provider: &str, model: &str) -> RoutingPoolEntry {
        RoutingPoolEntry {
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// `mixed` WITHOUT a pool spanning providers: the conversation stays (R-S2 opens the
    /// provider only to a pool ticked across providers, see the R-S2 tests below).
    #[tokio::test]
    async fn mixed_without_a_cross_provider_pool_never_leaves_the_sessions_provider() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ConversationRoutingMode::Mixed.pilot_mode(),
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

    /// AUTO-L: an Auto conversation chooses among its person's list only, on the provider
    /// check (the relay) and on the model of the turn, and says so in the reason.
    #[tokio::test]
    async fn auto_never_chooses_outside_the_persons_list() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        s.auto_models = Some(vec![entry("local2", "m")]);
        let router = router(s);
        assert!(is_move_to(&plan_provider_move(&router).await, "local2"));
        let asked = decider.last();
        assert!(
            asked.pool.iter().all(|f| f.provider_id == "local2"),
            "{:?}",
            asked.pool
        );
        assert_eq!(asked.narrowed, vec![auto_models::NARROWED_BY_PERSON]);
    }

    /// AUTO-L: without a list, Auto sees every model, as before.
    #[tokio::test]
    async fn auto_without_a_list_sees_every_available_model() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let router = full_auto(decider.clone());
        plan_provider_move(&router).await;
        assert_eq!(decider.last().pool.len(), 3);
        assert!(decider.last().narrowed.is_empty());
    }

    /// AUTO-L: on the session's provider, the model of the turn is chosen among the listed
    /// models of that provider; none of them here means no change of model (only a move
    /// reaches the list).
    #[tokio::test]
    async fn the_model_of_a_turn_stays_within_the_persons_list() {
        let decider = Fake::picking("local", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        s.auto_models = Some(vec![entry("local2", "m")]);
        let router = router(s);
        let directive = directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        assert_eq!(directive.model, None);
        assert_eq!(decider.calls(), 0, "nothing listed here: nothing to ask");
    }

    /// AUTO-L: a list none of whose models can serve (gone, consent withdrawn) falls back to
    /// every available model, with a notice in the reason, never a blocked turn.
    #[tokio::test]
    async fn a_list_with_nothing_usable_falls_back_to_every_model_and_says_so() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        // `gone` was deleted; `local3` lost the project's consent.
        s.auto_models = Some(vec![entry("gone", "x"), entry("local3", "m")]);
        let router = router(s);
        assert!(is_move_to(&plan_provider_move(&router).await, "local2"));
        let asked = decider.last();
        assert_eq!(asked.pool.len(), 3, "every available model again");
        assert_eq!(asked.narrowed, vec![auto_models::PERSON_LIST_UNAVAILABLE]);
    }

    /// AUTO-L: the conversation's own ticks win over the person's list: a pool's turn is
    /// decided among the pool, the list is not read.
    #[tokio::test]
    async fn a_conversations_pool_wins_over_the_persons_list() {
        let decider = Fake::picking("local", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ConversationRoutingMode::Mixed.pilot_mode(),
            LearningStage::Auto,
        );
        s.allowed_models = Some(vec!["m".into()]);
        s.routing_pool = Some(vec![entry("local", "m"), entry("local", "big")]);
        s.conversation_routes = true;
        s.auto_models = Some(vec![entry("local2", "m")]);
        let router = router(s);
        directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        assert_eq!(decider.calls(), 1, "the pool's turn is decided");
        assert!(decider.last().narrowed.is_empty(), "the list is not read");
        assert!(decider.last().pool.iter().all(|f| f.provider_id == "local"));
    }

    /// AUTO-L composes with a pool that spans providers (R-S2): an Auto conversation whose
    /// own ticks name models on two providers is moved among THOSE, through the relay; the
    /// person's list (here: only the session's provider) is not read and narrows nothing.
    #[tokio::test]
    async fn a_cross_provider_pool_wins_over_the_persons_list_on_a_move() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        s.allowed_models = Some(vec!["m".into()]);
        s.routing_pool = Some(vec![entry("local", "m"), entry("local2", "m")]);
        s.conversation_routes = true;
        s.auto_models = Some(vec![entry("local", "m")]);
        let router = router(s);
        let plan = plan_provider_move(&router).await;
        assert!(is_move_to(&plan, "local2"), "{plan:?}");
        let asked = decider.last();
        assert!(
            asked.narrowed.is_empty(),
            "the list is not read: {:?}",
            asked.narrowed
        );
        let mut providers: Vec<&str> = asked.pool.iter().map(|f| f.provider_id.as_str()).collect();
        providers.sort_unstable();
        assert_eq!(
            providers,
            vec!["local", "local2"],
            "the conversation's pool, both providers"
        );
    }

    /// R-S2, the one lock (`routing_of`): under `mixed` (its pilot mode is `primary`), a
    /// pool ticked across two providers opens the provider to a move; a pool on ONE
    /// provider, an imposed model or provider, or a choice by hand keeps it closed; `full`
    /// is exactly what it was.
    #[test]
    fn routing_of_opens_the_provider_to_a_mixed_pool_spanning_two_providers() {
        let cross = vec![entry("local", "m"), entry("local2", "m")];
        let mono = vec![entry("local", "m"), entry("local", "big")];
        let mixed = |pool: Vec<RoutingPoolEntry>| {
            let mut s = spec(
                Fake::picking("local2", 0.9, 0.1),
                Arc::new(ThreeProviders),
                ProviderRoutingMode::Primary,
                LearningStage::Auto,
            );
            s.allowed_models = Some(vec!["m".into()]);
            s.routing_pool = Some(pool);
            s.conversation_routes = true;
            s
        };
        let provider_of = |r: &TurnRouter| routing_of(r, &locked(&r.choice)).provider;

        assert!(
            provider_of(&router(mixed(cross.clone()))),
            "two providers: open"
        );
        assert!(!provider_of(&router(mixed(mono))), "one provider: closed");

        let mut pinned = mixed(cross.clone());
        pinned.explicit_model = true;
        assert!(!provider_of(&router(pinned)), "a model imposed: closed");
        let mut named = mixed(cross.clone());
        named.provider_imposed = true;
        assert!(!provider_of(&router(named)), "a provider imposed: closed");
        let by_hand = router(mixed(cross));
        by_hand.mark_manual();
        assert!(!provider_of(&by_hand), "a choice by hand: closed");

        // `full`: open without any pool, closed when imposed — unchanged.
        let full = |imposed: bool| {
            let mut s = spec(
                Fake::picking("local2", 0.9, 0.1),
                Arc::new(ThreeProviders),
                ProviderRoutingMode::Full,
                LearningStage::Auto,
            );
            s.provider_imposed = imposed;
            router(s)
        };
        assert!(provider_of(&full(false)));
        assert!(!provider_of(&full(true)));
    }

    /// R-S2 end to end in the router: the move is decided among the pool's two providers,
    /// asked as `full` (otherwise the decider answers `applied: false` under `primary`),
    /// and applied. Red before R-S2 on both counts: `routing_of` closed the provider, and
    /// the request kept the pilot mode.
    #[tokio::test]
    async fn a_mixed_pool_spanning_two_providers_moves_the_conversation() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Primary,
            LearningStage::Auto,
        );
        s.allowed_models = Some(vec!["m".into()]);
        s.routing_pool = Some(vec![entry("local", "m"), entry("local2", "m")]);
        s.conversation_routes = true;
        let router = router(s);
        let plan = plan_provider_move(&router).await;
        assert!(is_move_to(&plan, "local2"), "{plan:?}");
        let asked = decider.last();
        assert_eq!(
            (asked.settings.mode, asked.settings.stage),
            (ProviderRoutingMode::Full, LearningStage::Auto),
            "asked as full inside the pool"
        );
        let mut providers: Vec<&str> = asked.pool.iter().map(|f| f.provider_id.as_str()).collect();
        providers.sort_unstable();
        assert_eq!(
            providers,
            vec!["local", "local2"],
            "the pool, both providers"
        );
    }

    /// R-S2: a `mixed` pool on ONE provider never moves — the decider is not even asked
    /// for a move (the model is routed inside the provider by `directive_for_turn`).
    #[tokio::test]
    async fn a_mixed_pool_on_one_provider_never_moves_the_conversation() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Primary,
            LearningStage::Auto,
        );
        s.allowed_models = Some(vec!["m".into(), "big".into()]);
        s.routing_pool = Some(vec![entry("local", "m"), entry("local", "big")]);
        s.conversation_routes = true;
        let router = router(s);
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.calls(), 0, "no move is asked");
    }

    /// R-S2 through `PUT .../routing` (`reroute`): a conversation opened on a provider the
    /// request named (imposed, strict) is re-ticked with a pool spanning two providers —
    /// from the next turn on the router may move it. `reroute` must KEEP the pool it is
    /// given: red when it drops it (`choice.routing_pool = None`, what main did), since
    /// then no pool spans anything and `mixed` stays closed.
    #[tokio::test]
    async fn re_ticking_a_pool_across_providers_opens_a_strict_conversation_to_a_move() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Primary,
            LearningStage::Auto,
        );
        s.explicit_model = true;
        s.provider_imposed = true;
        let router = router(s);
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.calls(), 0, "strict: nothing asked");

        router.reroute(
            ConversationRoutingMode::Mixed,
            Some(vec!["m".into()]),
            Some(vec![entry("local", "m"), entry("local2", "m")]),
            false,
        );
        let plan = plan_provider_move(&router).await;
        assert!(is_move_to(&plan, "local2"), "{plan:?}");
        assert_eq!(decider.calls(), 1);

        // Re-ticked on one provider only: closed again.
        router.reroute(
            ConversationRoutingMode::Mixed,
            Some(vec!["m".into(), "big".into()]),
            Some(vec![entry("local", "m"), entry("local", "big")]),
            false,
        );
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.calls(), 1, "one provider: nothing more asked");
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
        named.reroute(ConversationRoutingMode::Full, None, None, false);
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
    async fn the_turn_after_a_move_may_change_model_but_not_provider_then_the_router_may_move_again(
    ) {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        s.moved_in = true;
        // The moved-to session opens fresh: its counter starts at 0, so the relayed turn's
        // predecision (now keyed on the carried `next_turn`) matches the turn sent at index 0.
        s.next_turn = 0;
        let router = router(s);
        // The relayed turn (0) runs on the pair the move chose: nothing is asked.
        directive_for_turn(&router, &TurnContext::new(0, "m")).await;
        assert_eq!(decider.calls(), 0);
        // The next turn: no second move (R-S3), decided among THIS provider's models, at
        // the auto stage: its model may change.
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        assert_eq!(decider.last().restrict_provider.as_deref(), Some("local"));
        assert_eq!(decider.last().settings.stage, LearningStage::Auto);
        directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        // The one after may move.
        assert!(is_move_to(&plan_provider_move(&router).await, "local2"));
        assert_eq!(decider.last().restrict_provider, None);
    }

    /// Review of #710 (1): a provider move is a change of model. Turn 1 of the class of the
    /// relayed turn is steady (no exploration, margin on the means) on top of being decided
    /// on the arrival provider only; a turn 1 of another class is only narrowed to the
    /// provider. Red without `seed_relayed_turn`: turn 1 explored as any turn.
    #[tokio::test]
    async fn after_a_move_turn_1_of_the_relayed_class_is_steady() {
        let moved = |decider: Arc<Fake>| {
            let mut s = spec(
                decider,
                Arc::new(ThreeProviders),
                ProviderRoutingMode::Full,
                LearningStage::Auto,
            );
            s.moved_in = true;
            s.next_turn = 0;
            s
        };
        let same = Fake::picking("local2", 0.9, 0.1);
        let first = router(moved(same.clone()));
        // What `register_turn_router` does once the relayed message is set.
        first.seed_relayed_turn();
        directive_for_turn(&first, &TurnContext::new(0, "m")).await;
        assert_eq!(plan_provider_move(&first).await, ProviderPlan::Stay);
        let request = same.last();
        assert_eq!(request.turn_index, Some(1));
        assert_eq!(request.settings.exploration_epsilon, 0.0);
        assert!(request.steady);
        assert_eq!(request.restrict_provider.as_deref(), Some("local"));
        assert_eq!(
            request.narrowed,
            [SAME_CLASS_NO_EXPLORE, AFTER_PROVIDER_MOVE]
        );

        let other = Fake::picking("local2", 0.9, 0.1);
        let second = router(moved(other.clone()));
        second.seed_relayed_turn();
        directive_for_turn(&second, &TurnContext::new(0, "m")).await;
        second.set_last_message("rename this variable");
        assert_eq!(plan_provider_move(&second).await, ProviderPlan::Stay);
        let request = other.last();
        assert_eq!(
            request.settings.exploration_epsilon,
            RoutingSettings::default().exploration_epsilon
        );
        assert!(!request.steady);
        assert_eq!(request.restrict_provider.as_deref(), Some("local"));
        assert_eq!(request.narrowed, [AFTER_PROVIDER_MOVE]);
    }

    /// Picks `local/<model>`, the current pair scored low: applied in `full` + `auto`.
    fn picking_model(model: &str) -> Arc<Fake> {
        Arc::new(Fake {
            pick: Pick::new("local", model),
            ..Arc::try_unwrap(Fake::picking("local", 0.9, 0.1))
                .ok()
                .unwrap()
        })
    }

    #[tokio::test]
    async fn the_turn_after_a_move_changes_its_model_on_the_provider_it_moved_to() {
        let decider = picking_model("m2");
        let mut s = spec(
            decider.clone(),
            Arc::new(ThreeProviders),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        );
        s.moved_in = true;
        // The moved-to session opens fresh: its counter starts at 0, so the relayed turn's
        // predecision (now keyed on the carried `next_turn`) matches the turn sent at index 0.
        s.next_turn = 0;
        let router = router(s);
        directive_for_turn(&router, &TurnContext::new(0, "m")).await;
        assert_eq!(plan_provider_move(&router).await, ProviderPlan::Stay);
        let directive = directive_for_turn(&router, &TurnContext::new(1, "m")).await;
        assert_eq!(
            directive.model.as_deref(),
            Some("m2"),
            "the model may change"
        );
        assert_eq!(decider.calls(), 1, "one decision for the turn");
    }

    /// R-S3: right after a change of model, a turn of ANOTHER class is routed as any turn
    /// (it used to be only recorded: one turn late), with the exploration of the settings.
    #[tokio::test]
    async fn right_after_a_change_a_turn_of_another_class_is_routed_on_the_turn() {
        let decider = picking_model("m2");
        let router = router(spec(
            decider.clone(),
            Arc::new(OneProvider),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        ));
        router.set_last_message("why does this crash with a stack trace error");
        let debug = router.signature().arm_key();
        {
            let mut state = locked(&router.state);
            record_change(&mut state, 1, "m", "chat.general".into());
        }
        assert_ne!(debug, "chat.general");
        let directive = directive_for_turn(&router, &TurnContext::new(2, "m")).await;
        assert_eq!(directive.model.as_deref(), Some("m2"));
        let request = decider.last();
        assert_eq!(request.settings.stage, LearningStage::Auto);
        assert_eq!(
            request.settings.exploration_epsilon,
            RoutingSettings::default().exploration_epsilon
        );
        let state = locked(&router.state);
        assert_eq!(state.last_change_turn, Some(2));
        assert_eq!(state.last_change_class.as_deref(), Some(debug.as_str()));
    }

    /// R-S3: right after a change, a turn of the SAME class is decided without exploration:
    /// the decider's hysteresis margin alone decides whether it moves again (it is still
    /// applied when the challenger clears it).
    #[tokio::test]
    async fn right_after_a_change_a_turn_of_the_same_class_is_decided_without_exploration() {
        assert!(RoutingSettings::default().exploration_epsilon > 0.0);
        let decider = picking_model("m2");
        let router = router(spec(
            decider.clone(),
            Arc::new(OneProvider),
            ProviderRoutingMode::Full,
            LearningStage::Auto,
        ));
        let class = router.signature().arm_key();
        record_change(&mut locked(&router.state), 1, "m", class.clone());
        let directive = directive_for_turn(&router, &TurnContext::new(2, "m")).await;
        assert_eq!(directive.model.as_deref(), Some("m2"), "the margin cleared");
        assert_eq!(decider.last().settings.exploration_epsilon, 0.0);
        assert_eq!(decider.last().settings.stage, LearningStage::Auto);
        // Two turns later, the guard is gone.
        record_change(&mut locked(&router.state), 2, "m2", class);
        directive_for_turn(&router, &TurnContext::new(4, "m2")).await;
        assert!(decider.last().settings.exploration_epsilon > 0.0);
    }

    /// R-S3: the provider check is guarded per class too: right after a change of model of
    /// another class, a move is allowed on the turn.
    #[tokio::test]
    async fn right_after_a_change_of_another_class_the_provider_may_move() {
        let decider = Fake::picking("local2", 0.9, 0.1);
        let router = full_auto(decider.clone());
        record_change(&mut locked(&router.state), 0, "m", "chat.general".into());
        assert!(is_move_to(&plan_provider_move(&router).await, "local2"));
        assert!(decider.last().settings.exploration_epsilon > 0.0);
        let same = full_auto(decider.clone());
        let class = same.signature().arm_key();
        record_change(&mut locked(&same.state), 0, "m", class);
        assert!(is_move_to(&plan_provider_move(&same).await, "local2"));
        assert_eq!(decider.last().settings.exploration_epsilon, 0.0);
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
        router.reroute(ConversationRoutingMode::Primary, None, None, true);
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
            ConversationRoutingMode::Mixed,
            Some(vec!["m".into(), "big".into()]),
            Some(vec![entry("local", "m"), entry("local", "big")]),
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
        router.reroute(ConversationRoutingMode::Full, None, None, false);
        router
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
        let router = TurnRouter::new(spec(decider, Arc::new(Pool(pool)), mode, allowed_models));
        router.set_model_live(true);
        router
    }

    fn spec(
        decider: Arc<Real>,
        pool: Arc<dyn PoolSource>,
        mode: ProviderRoutingMode,
        allowed_models: Option<Vec<String>>,
    ) -> TurnRouterSpec {
        TurnRouterSpec {
            decider,
            pool,
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
            auto_models: None,
            conversation_routes: false,
            current_model: "flash".into(),
            next_turn: 1,
            moved_in: false,
            segment_store: None,
            segments: None,
        }
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
        let images = effective_images(&router, Some(false)).await.unwrap();
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
        let images = effective_images(&full, Some(false)).await.unwrap();
        assert!(images.value, "{images:?}");
        assert_eq!(images.via[0].provider, "glm");

        let mixed = router(
            Arc::default(),
            pool,
            ConversationRoutingMode::Mixed.pilot_mode(),
            Some(vec!["flash".into()]),
        );
        let images = effective_images(&mixed, Some(false)).await.unwrap();
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
        let images = effective_images(&router, Some(false)).await.unwrap();
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
        let images = effective_images(&router, Some(false)).await.unwrap();
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
        let images = effective_images(&primary, Some(false)).await.unwrap();
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
            effective_images(&manual, Some(false)).await.unwrap().cause,
            EffectiveCause::NotRouted
        );
        // A model that reads images: nothing to route for.
        let images = effective_images(&primary, Some(true)).await.unwrap();
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

    /// A pool source that answers too late.
    struct Slow;

    #[async_trait]
    impl PoolSource for Slow {
        async fn pool(&self, _provider_id: &str) -> Vec<ModelFacts> {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Vec::new()
        }
        async fn project_pool(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
            tokio::time::sleep(Duration::from_secs(30)).await;
            deepseek_with_vision()
        }
    }

    #[tokio::test]
    async fn a_failed_model_switch_undoes_the_router_and_drops_the_decision_it_kept() {
        let decider = Arc::new(Real::default());
        let router = router(
            decider.clone(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        router.set_turn_input(&with_picture("look"));
        assert_eq!(model_for_images(&router).await.as_deref(), Some("vision"));
        let image_decision = decider.decisions.lock().unwrap()[0].clone();
        assert!(image_decision.applied);
        // The host could not make `vision` active: the turn was refused, never started.
        // The decision the change came from is handed back, for the host to store it as
        // not applied (it said `applied: true`).
        let undone = router
            .forget_image_change()
            .expect("the decision of the image change is handed back");
        assert_eq!(undone.id, image_decision.id);
        // Handed back once: a second undo has nothing left to rewrite.
        assert!(router.forget_image_change().is_none());
        {
            let state = locked(&router.state);
            assert_eq!(state.current_model, "flash");
            assert_eq!(state.last_change_turn, None);
        }
        assert!(locked(&router.predecided).is_none());
        // The next turn, text only, on the same index: decided afresh, no change handed to it.
        router.set_turn_input("why does this crash with a stack trace error");
        let directive = directive_for_turn(&router, &TurnContext::new(1, "flash")).await;
        let decisions = decider.decisions.lock().unwrap().clone();
        assert_eq!(
            decisions.len(),
            2,
            "asked again, not the kept image decision"
        );
        assert!(!decisions[1].signature.needs_images);
        // Whatever changes now is this text turn's own decision, never the image one.
        let own = decisions[1]
            .chosen
            .clone()
            .filter(|pick| pick.model != "flash")
            .map(|pick| pick.model);
        assert_eq!(directive.model, own);
    }

    /// The legacy Claude Code engine stores no snapshot: unknown is never turned into a no,
    /// even when the pool rejects every Claude model (catalogue offline: window unknown).
    #[tokio::test]
    async fn an_unknown_snapshot_is_never_reported_as_a_pool_without_images() {
        let mut facts_offline = facts("deepseek", "claude-opus", true);
        facts_offline.context_window = None;
        facts_offline.window_unknown = Some(candidates::UnknownWindow::CatalogOffline);
        let offline = router(
            Arc::default(),
            vec![facts_offline],
            ProviderRoutingMode::Full,
            None,
        );
        assert_eq!(effective_images(&offline, None).await, None);
        assert_eq!(
            effective_images(&offline, Some(false))
                .await
                .map(|i| i.cause),
            Some(EffectiveCause::PoolLacksIt)
        );
        // Not routed, no snapshot: nothing to say either.
        let primary = router_with(ProviderRoutingMode::Primary, LearningStage::Auto, false);
        assert_eq!(effective_images(&primary, None).await, None);
        // A pool that says yes is said, snapshot or not.
        let full = router(
            Arc::default(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        assert_eq!(
            effective_images(&full, None).await.map(|i| i.value),
            Some(true)
        );
    }

    #[test]
    fn model_ids_are_compared_without_the_window_suffix_and_case() {
        assert!(same_model("flash[1m]", "FLASH"));
        assert!(same_model("claude-opus-4[1m]", "claude-opus-4"));
        assert!(!same_model("opus", "claude-opus-4"));
    }

    /// A pair the pool does not list is not blind: an image turn does not override "no move
    /// right after a move", nor the model change rules.
    #[tokio::test]
    async fn a_model_the_pool_does_not_list_is_not_taken_as_blind() {
        let decider = Arc::new(Real::default());
        let mut spec = spec(
            decider.clone(),
            Arc::new(Pool(vec![
                facts("deepseek", "flash", false),
                facts("glm", "v", true),
            ])),
            ProviderRoutingMode::Full,
            None,
        );
        spec.current_model = "opus".into();
        spec.moved_in = true;
        let unlisted = TurnRouter::new(spec);
        unlisted.set_model_live(true);
        unlisted.set_turn_input(&with_picture("look"));
        assert_eq!(plan_provider_move(&unlisted).await, ProviderPlan::Stay);
        assert_eq!(model_for_images(&unlisted).await, None);
        // The same pair, named with its window suffix, IS known blind.
        let known = router(
            Arc::default(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        known.set_turn_input(&with_picture("look"));
        let pool = deepseek_with_vision();
        assert!(cannot_read_turn(&known, &pool, "deepseek", "flash[1m]"));
        assert!(!cannot_read_turn(&known, &pool, "deepseek", "opus"));
    }

    #[tokio::test]
    async fn a_pool_built_too_late_is_unbuilt_never_lacking() {
        let mut router = TurnRouter::new(spec(
            Arc::new(Real::default()),
            Arc::new(Slow),
            ProviderRoutingMode::Full,
            None,
        ));
        router.set_model_live(true);
        router.timeout = Duration::from_millis(50);
        let started = std::time::Instant::now();
        let images = effective_images(&router, Some(false)).await.unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(
            (images.value, images.source, images.cause),
            (
                false,
                EffectiveSource::Snapshot,
                EffectiveCause::PoolUnbuilt
            )
        );
    }

    /// `full`, but this provider cannot switch model live: its own vision models are no way to
    /// read the image — neither promised, nor chosen by the provider check.
    #[tokio::test]
    async fn without_a_live_model_change_only_another_provider_reads_the_image() {
        let pool = vec![
            facts("deepseek", "flash", false),
            facts("deepseek", "vision", true),
            facts("glm", "v", true),
        ];
        let decider = Arc::new(Real::default());
        let router = TurnRouter::new(spec(
            decider.clone(),
            Arc::new(Pool(pool)),
            ProviderRoutingMode::Full,
            None,
        ));
        let images = effective_images(&router, Some(false)).await.unwrap();
        assert_eq!(
            images.via,
            [RoutingPoolEntry {
                provider: "glm".into(),
                model: "v".into()
            }]
        );
        router.set_turn_input(&with_picture("look"));
        assert!(matches!(
            plan_provider_move(&router).await,
            ProviderPlan::Move { pick, .. } if pick == Pick::new("glm", "v")
        ));
    }

    fn router_with(mode: ProviderRoutingMode, stage: LearningStage, explicit: bool) -> TurnRouter {
        let mut spec = spec(
            Arc::new(Real::default()),
            Arc::new(Pool(deepseek_with_vision())),
            mode,
            None,
        );
        spec.routing.stage = stage;
        spec.explicit_model = explicit;
        let router = TurnRouter::new(spec);
        router.set_model_live(true);
        router.set_turn_input(&with_picture("look"));
        router
    }

    /// R-S1 x F-R4: the user put THIS conversation on Auto while the global stage is
    /// `shadow`. The conversation is routed at the `auto` stage, so the image turn really
    /// goes to the model that reads images (and the capability is promised), where the same
    /// settings without the conversation's Auto only record.
    #[tokio::test]
    async fn an_auto_conversation_under_the_global_shadow_stage_switches_to_the_image_model() {
        let decider = Arc::new(Real::default());
        let mut spec = spec(
            decider.clone(),
            Arc::new(Pool(deepseek_with_vision())),
            ProviderRoutingMode::Full,
            None,
        );
        spec.routing.stage = LearningStage::Shadow;
        spec.conversation_routes = true;
        let router = TurnRouter::new(spec);
        router.set_model_live(true);
        router.set_turn_input(&with_picture("look"));
        let images = effective_images(&router, Some(false)).await.unwrap();
        assert_eq!(
            (images.value, images.source, images.cause),
            (
                true,
                EffectiveSource::RoutingPool,
                EffectiveCause::PoolHasIt
            )
        );
        assert_eq!(model_for_images(&router).await.as_deref(), Some("vision"));
        let decision = decider.decisions.lock().unwrap()[0].clone();
        assert_eq!(decision.stage, LearningStage::Auto, "{decision:#?}");
        assert!(decision.applied, "{decision:#?}");
        // The hook of the turn finds the change made and asks nothing more.
        let directive = directive_for_turn(&router, &TurnContext::new(1, "vision")).await;
        assert_eq!(directive.model, None);
        assert_eq!(decider.decisions.lock().unwrap().len(), 1);
        assert_eq!(router.next_turn_context(0).current_model, "vision");
    }

    /// Shadow and advisory only record; an imposed model is never replaced: no promise, no
    /// change before the turn.
    #[tokio::test]
    async fn shadow_advisory_and_an_imposed_model_promise_nothing_and_change_nothing() {
        for router in [
            router_with(ProviderRoutingMode::Full, LearningStage::Shadow, false),
            router_with(ProviderRoutingMode::Full, LearningStage::Advisory, false),
            router_with(ProviderRoutingMode::Full, LearningStage::Auto, true),
        ] {
            let images = effective_images(&router, Some(false)).await.unwrap();
            assert_eq!(
                (images.value, images.source, images.cause),
                (false, EffectiveSource::Snapshot, EffectiveCause::NotRouted)
            );
            assert_eq!(model_for_images(&router).await, None);
        }
    }

    #[tokio::test]
    async fn the_images_of_a_turn_are_forgotten_once_its_hook_ran() {
        let router = router(
            Arc::default(),
            deepseek_with_vision(),
            ProviderRoutingMode::Full,
            None,
        );
        router.set_turn_input(&with_picture("look"));
        assert!(router.turn_has_images());
        let _ = directive_for_turn(&router, &TurnContext::new(1, "flash")).await;
        assert!(!router.turn_has_images());
        assert!(
            !router.signature().needs_images,
            "a compaction during the turn reads no image"
        );
    }
}

/// S4 integration: the session's segmenter is fed one [`Observation`] per turn and its
/// boundaries (harness = explicit provider change; any other axis = a change of direction)
/// are the finer-grained signal that augments [`TurnRouter::moved_in`].
#[cfg(test)]
mod segment_tests {
    use super::*;
    use crate::chat::provider::cognitive::decision::DecisionAlternative;
    use crate::chat::provider::cognitive::segment::BoundaryReason;
    use crate::chat::provider::cognitive::segment_store::{InMemorySegmentStore, SegmentStore};
    use chrono::Utc;
    use nexus_claude::agent::CostBasis;
    use std::sync::Mutex as StdMutex;

    /// A decider that is never asked: these tests only feed observations.
    struct NoDecide;

    #[async_trait]
    impl Decider for NoDecide {
        async fn decide(&self, _request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
            unreachable!("the segment tests never ask the decider");
        }
    }

    /// A pool that knows nothing: the segment tests never route.
    struct NoPool;

    #[async_trait]
    impl PoolSource for NoPool {
        async fn pool(&self, _provider_id: &str) -> Vec<ModelFacts> {
            Vec::new()
        }
    }

    fn spec(session_id: Uuid) -> TurnRouterSpec {
        TurnRouterSpec {
            decider: Arc::new(NoDecide),
            pool: Arc::new(NoPool),
            routing: RoutingSettings::default(),
            provider_id: "deepseek".into(),
            session_id: Some(session_id),
            project_slug: None,
            trust: false,
            explicit_model: false,
            provider_imposed: false,
            allowed_models: None,
            routing_pool: None,
            auto_models: None,
            conversation_routes: false,
            current_model: "m".into(),
            next_turn: 1,
            moved_in: false,
            segment_store: None,
            segments: None,
        }
    }

    fn router(session_id: Uuid) -> TurnRouter {
        TurnRouter::new(spec(session_id))
    }

    /// Feeds a turn under `provider`, with `text` as the stored message.
    fn feed(router: &TurnRouter, turn: u32, provider: &str, text: &str) -> Vec<Boundary> {
        router.set_last_message(text);
        router.observe_turn(turn, provider)
    }

    #[test]
    fn a_harness_change_opens_a_boundary_at_once() {
        let router = router(Uuid::new_v4());
        // The harness axis confirms a switch at once (a switch is explicit), unlike the
        // other axes that need two deviating turns.
        assert!(feed(&router, 0, "deepseek", "anything").is_empty());
        assert!(feed(&router, 1, "deepseek", "anything").is_empty());
        let boundaries = feed(&router, 2, "claude-code", "anything");
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].dimension, Dimension::Harness);
        assert_eq!(boundaries[0].at_turn, 2);
        let BoundaryReason::Drift { confirmations, .. } = &boundaries[0].reason else {
            panic!("a harness switch is a drift");
        };
        assert_eq!(*confirmations, 1);
        // The segmenter keeps the confirmed boundary, not just the per-turn return.
        assert_eq!(locked(&router.segmenter).boundaries().len(), 1);
    }

    #[test]
    fn a_topic_change_opens_a_boundary_after_two_deviating_turns() {
        let router = router(Uuid::new_v4());
        let routing = "the routing of the provider";
        let other = "neo4j docker tests";
        // Same subject on two turns: the segment absorbs them, no boundary.
        assert!(feed(&router, 0, "deepseek", routing).is_empty());
        assert!(feed(&router, 1, "deepseek", routing).is_empty());
        // One deviating turn is not a change; it opens a pending run only.
        assert!(feed(&router, 2, "deepseek", other).is_empty());
        // The second deviating turn confirms the boundary, placed at the FIRST deviating turn.
        let boundaries = feed(&router, 3, "deepseek", other);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].dimension, Dimension::Topic);
        assert_eq!(boundaries[0].at_turn, 2);
        let BoundaryReason::Drift { confirmations, .. } = &boundaries[0].reason else {
            panic!("a topic drift is a drift");
        };
        assert_eq!(*confirmations, 2);
    }

    #[test]
    fn the_seed_makes_segment_ids_deterministic_across_routers_with_the_same_session() {
        let session = Uuid::new_v4();
        let a = router(session);
        let b = router(session);
        let history = [
            (0, "deepseek", "the routing of the provider"),
            (1, "deepseek", "the routing of the provider"),
            (2, "deepseek", "neo4j docker tests"),
            (3, "deepseek", "neo4j docker tests"),
        ];
        for (turn, provider, text) in history {
            feed(&a, turn, provider, text);
            feed(&b, turn, provider, text);
        }
        // The same session id gives the same seed, hence the same ids, byte for byte.
        assert_eq!(
            locked(&a.segmenter).segments(),
            locked(&b.segmenter).segments()
        );
        assert_eq!(
            locked(&a.segmenter).boundaries().to_vec(),
            locked(&b.segmenter).boundaries().to_vec()
        );
        // Another session (another seed) segments the same turns but with different ids.
        let other = router(Uuid::new_v4());
        for (turn, provider, text) in history {
            feed(&other, turn, provider, text);
        }
        let segs_a = locked(&a.segmenter).segments();
        let segs_other = locked(&other.segmenter).segments();
        assert_eq!(segs_a.len(), segs_other.len());
        assert_ne!(segs_a[0].id, segs_other[0].id);
    }

    #[test]
    fn a_non_increasing_turn_is_tolerated_without_breaking_routing() {
        let router = router(Uuid::new_v4());
        // The first observed turn is 3; turns 2 (before) and 3 (again) are refused by the
        // segmenter, which `observe_turn` tolerates as an empty return instead of an error:
        // routing never breaks on a repeated or out-of-order turn.
        assert!(feed(&router, 3, "deepseek", "the routing of the provider").is_empty());
        assert!(feed(&router, 2, "deepseek", "the routing of the provider").is_empty());
        assert!(feed(&router, 3, "deepseek", "the routing of the provider").is_empty());
        // The next strictly increasing turn still segments normally.
        assert!(feed(&router, 4, "deepseek", "neo4j docker tests").is_empty());
        let boundaries = feed(&router, 5, "deepseek", "neo4j docker tests");
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].dimension, Dimension::Topic);
        assert_eq!(boundaries[0].at_turn, 4);
    }

    // --- S3 decision wiring: fixtures that actually route (unlike `NoDecide`/`NoPool`) ---

    fn routing_facts(provider: &str, model: &str) -> ModelFacts {
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
            allowed_for_project: true,
            sandboxed: false,
        }
    }

    /// Two providers, so a move is possible: `local/m` (the session's) and `local2/m`.
    struct TwoProviders;

    #[async_trait]
    impl PoolSource for TwoProviders {
        async fn pool(&self, provider_id: &str) -> Vec<ModelFacts> {
            self.project_pool(None)
                .await
                .into_iter()
                .filter(|f| f.provider_id == provider_id)
                .collect()
        }
        async fn project_pool(&self, _project_slug: Option<&str>) -> Vec<ModelFacts> {
            vec![routing_facts("local", "m"), routing_facts("local2", "m")]
        }
    }

    /// Picks `local2/m` (a move, when the guards allow it) and records every request.
    #[derive(Default)]
    struct Recording {
        requests: StdMutex<Vec<DecideRequest>>,
    }

    impl Recording {
        fn last(&self) -> DecideRequest {
            self.requests.lock().unwrap().last().cloned().unwrap()
        }
    }

    #[async_trait]
    impl Decider for Recording {
        async fn decide(&self, request: &DecideRequest) -> anyhow::Result<CognitiveDecision> {
            self.requests.lock().unwrap().push(request.clone());
            let pick = Pick::new("local2", "m");
            let mut alternatives = vec![DecisionAlternative {
                pick: pick.clone(),
                score: Some(0.9),
                rejected: None,
            }];
            if let Some(current) = request.current.clone().filter(|c| *c != pick) {
                alternatives.push(DecisionAlternative {
                    pick: current,
                    score: Some(0.1),
                    rejected: None,
                });
            }
            Ok(CognitiveDecision {
                id: Uuid::new_v4(),
                at: Utc::now(),
                signature: request.signature.clone(),
                chosen: Some(pick),
                score: Some(0.9),
                explored: false,
                reason: "recording".into(),
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

    /// A router in `full` + `auto` on `local/m`, over the two-provider pool.
    fn routing_router(decider: Arc<Recording>, moved_in: bool) -> TurnRouter {
        TurnRouter::new(TurnRouterSpec {
            decider,
            pool: Arc::new(TwoProviders),
            routing: RoutingSettings {
                mode: ProviderRoutingMode::Full,
                stage: LearningStage::Auto,
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
            auto_models: None,
            conversation_routes: false,
            current_model: "m".into(),
            next_turn: 1,
            moved_in,
            segment_store: None,
            segments: None,
        })
    }

    /// S3 signal: a TOPIC drift within the SAME class does NOT lift the R-S3 `held_by_class`
    /// steady guard (the class is unchanged, so the model choice stays steady), but it still
    /// relaxes the `after_provider_move` restriction (`direction_changed`) and is recorded in
    /// the reason. Only a CLASS change (Problem/Resolution boundary) lifts the steady guard.
    #[tokio::test]
    async fn a_topic_drift_keeps_the_class_steady_but_still_relaxes_the_move_guard() {
        let decider = Arc::new(Recording::default());
        let router = routing_router(decider.clone(), true);
        feed(&router, 0, "local", "the routing of the provider");
        feed(&router, 1, "local", "the routing of the provider");
        feed(&router, 2, "local", "neo4j docker tests"); // one deviating turn
        let class = router.signature().arm_key();
        {
            let mut state = locked(&router.state);
            state.next_turn = 3;
            record_change(&mut state, 2, "m", class);
        }
        let _ = plan_provider_move(&router).await;
        let request = decider.last();
        assert!(
            request.steady,
            "a same-class topic drift keeps the steady guard: {request:?}"
        );
        assert_eq!(
            request.settings.exploration_epsilon, 0.0,
            "no exploration on a same-class topic drift: {request:?}"
        );
        assert!(
            request.narrowed.contains(&SAME_CLASS_NO_EXPLORE),
            "{request:?}"
        );
        assert_eq!(
            request.restrict_provider, None,
            "the topic drift still relaxes the after-provider-move guard: {request:?}"
        );
        assert!(
            request.segment_note.is_some(),
            "the boundary is recorded in the reason suffix: {request:?}"
        );
    }

    /// The same class with no boundary: the R-S3 `held_by_class` guard behaves exactly as
    /// before (steady, no exploration).
    #[tokio::test]
    async fn a_same_class_turn_without_a_boundary_stays_held() {
        let decider = Arc::new(Recording::default());
        let router = routing_router(decider.clone(), false);
        feed(&router, 0, "local", "the routing of the provider");
        feed(&router, 1, "local", "the routing of the provider");
        feed(&router, 2, "local", "the routing of the provider");
        let class = router.signature().arm_key();
        {
            let mut state = locked(&router.state);
            state.next_turn = 3;
            record_change(&mut state, 2, "m", class);
        }
        let _ = plan_provider_move(&router).await;
        let request = decider.last();
        assert!(request.steady, "{request:?}");
        assert_eq!(request.settings.exploration_epsilon, 0.0);
        assert!(
            request.narrowed.contains(&SAME_CLASS_NO_EXPLORE),
            "{request:?}"
        );
        assert_eq!(request.segment_note, None);
    }

    /// A harness boundary is a provider switch, not a change of direction: it keeps the
    /// "no two provider moves in a row" guard (`direction_changed` excludes the harness axis).
    #[tokio::test]
    async fn a_harness_boundary_keeps_the_move_guard() {
        let decider = Arc::new(Recording::default());
        let router = routing_router(decider.clone(), true);
        // The session arrived from another harness: turn 0 was served there, the turn about
        // to start (1) is on this one — a harness boundary fires, and nothing else.
        feed(&router, 0, "claude-code", "the weather in antibes");
        {
            let mut state = locked(&router.state);
            state.next_turn = 1;
        }
        let plan = plan_provider_move(&router).await;
        let request = decider.last();
        assert_eq!(
            request.restrict_provider.as_deref(),
            Some("local"),
            "the move guard stays: {request:?}"
        );
        assert!(
            request.narrowed.contains(&AFTER_PROVIDER_MOVE),
            "{request:?}"
        );
        assert_eq!(
            plan,
            ProviderPlan::Stay,
            "no second move in a row: {plan:?}"
        );
    }

    /// A confirmed boundary snapshots the segmenter to the store, best-effort and
    /// non-blocking (the write is spawned on a separate task). The store is keyed by the
    /// session id, so the conversation's segments and boundaries land under it.
    #[tokio::test]
    async fn a_boundary_observation_persists_segments_and_boundaries_to_the_store() {
        let session = Uuid::new_v4();
        let conversation = session.to_string();
        let store: Arc<dyn SegmentStore> = Arc::new(InMemorySegmentStore::new());
        let mut router_spec = spec(session);
        router_spec.segment_store = Some(store.clone());
        let router = TurnRouter::new(router_spec);
        feed(&router, 0, "deepseek", "the routing of the provider");
        feed(&router, 1, "deepseek", "the routing of the provider");
        assert!(feed(&router, 2, "deepseek", "neo4j docker tests").is_empty());
        let boundaries = feed(&router, 3, "deepseek", "neo4j docker tests");
        assert_eq!(boundaries.len(), 1);
        // The write is spawned, not awaited: poll the store briefly.
        let deadline = std::time::Instant::now();
        loop {
            let segments = store.segments(&conversation).await.unwrap();
            if !segments.is_empty() {
                assert_eq!(segments, locked(&router.segmenter).segments());
                break;
            }
            assert!(
                deadline.elapsed() < Duration::from_secs(2),
                "the segments were not persisted"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            store.boundaries(&conversation).await.unwrap(),
            locked(&router.segmenter).boundaries().to_vec()
        );
    }

    /// S3 signal, model path: `directive_for_turn` (the hook of a `mixed` or one-provider
    /// session, the only observer of its turns) keeps the R-S3 `held_by_class` steady guard on
    /// a TOPIC drift within the SAME class (the class is unchanged), matching
    /// `plan_provider_move`. A class change lifts it through the class itself.
    #[tokio::test]
    async fn a_topic_drift_keeps_the_steady_guard_in_the_model_path() {
        let decider = Arc::new(Recording::default());
        let router = routing_router(decider.clone(), false);
        router.set_model_live(true);
        feed(&router, 0, "local", "the routing of the provider");
        feed(&router, 1, "local", "the routing of the provider");
        feed(&router, 2, "local", "neo4j docker tests"); // one deviating turn
        let class = router.signature().arm_key();
        {
            let mut state = locked(&router.state);
            record_change(&mut state, 2, "m", class);
        }
        // Turn 3 continues the drift: the second deviating turn confirms a topic boundary, and
        // the hook must keep the class steady (no exploration), not re-open the decision.
        directive_for_turn(&router, &TurnContext::new(3, "m")).await;
        let request = decider.last();
        assert!(
            request.steady,
            "a same-class topic drift keeps the steady guard: {request:?}"
        );
        assert_eq!(request.settings.exploration_epsilon, 0.0, "{request:?}");
        assert!(
            request.narrowed.contains(&SAME_CLASS_NO_EXPLORE),
            "{request:?}"
        );
    }

    /// Slice B: a router rebuilt from a carried snapshot continues segmenting where the old
    /// one stopped even though the provider's harness RESETS its turn index to 0. The carried
    /// snapshot's `last_turn` becomes the segmenter's turn base, so the reset indices
    /// (0, 1, …) still segment as `last_turn, last_turn + 1, …` — no `OutOfOrder`, no
    /// spurious boundary, and the pre-carry boundary stays put.
    #[test]
    fn a_router_rebuilt_from_a_carried_snapshot_continues_segmenting_not_reset() {
        let session = Uuid::new_v4();
        let routing = "the routing of the provider";
        let other = "neo4j docker tests";
        // The first router confirms a topic boundary (two deviating turns).
        let first = router(session);
        feed(&first, 0, "deepseek", routing);
        feed(&first, 1, "deepseek", routing);
        feed(&first, 2, "deepseek", other);
        let confirmed = feed(&first, 3, "deepseek", other);
        assert_eq!(confirmed.len(), 1);
        assert_eq!(confirmed[0].dimension, Dimension::Topic);

        let snapshot = first.snapshot_segments();
        let last_turn = snapshot
            .segments
            .iter()
            .map(|segment| segment.last_turn)
            .chain(snapshot.boundaries.iter().map(|boundary| boundary.at_turn))
            .max()
            .unwrap_or(0);
        assert_eq!(last_turn, 3);
        // The segmenter counter the move carries is `last_turn + 1` (NOT the R-S3 seed).
        assert_eq!(first.next_turn(), last_turn + 1);

        // Rebuild on the same session with the carried snapshot. The R-S3 `next_turn` is the
        // provider's 0-based index (its harness resets), not the carried counter.
        let mut carried = spec(session);
        carried.segments = Some(snapshot.clone());
        carried.next_turn = 0;
        let rebuilt = TurnRouter::new(carried);

        // The pre-carry boundary is restored, not reset.
        assert_eq!(locked(&rebuilt.segmenter).boundaries().len(), 1);
        // The provider's harness restarts at 0: the segmenter shifts each reset index up by
        // `last_turn`, so the continued subject is absorbed, never OutOfOrder, and never
        // mistaken for a new direction.
        assert!(
            feed(&rebuilt, 0, "deepseek", other).is_empty(),
            "a continued subject must not open a spurious boundary"
        );
        assert!(
            feed(&rebuilt, 1, "deepseek", other).is_empty(),
            "the reset index still segments strictly increasing"
        );
        assert_eq!(locked(&rebuilt.segmenter).boundaries().len(), 1);
        // The reopened router's segmenter continued from the carried `last_turn` (its base), not reset to 0.
        assert_eq!(rebuilt.next_turn(), last_turn + 2);
    }

    /// Slice B: the relayed turn of a moved-in router is recorded at the provider's 0-based
    /// index (turn 0), NOT the carried counter, so `held_by_class` fires for the following
    /// turn (1) of the same class. Red with the carried counter (e.g. 5): the next turn's
    /// index (1) would never follow the recorded change and the "no two moves in a row" guard
    /// would die.
    #[test]
    fn a_carried_relayed_turn_seeds_the_guard_at_the_provider_index() {
        let session = Uuid::new_v4();
        let mut carried = spec(session);
        carried.moved_in = true;
        carried.next_turn = 5; // the carried counter must NOT leak into the R-S3 seed
        let router = TurnRouter::new(carried);
        router.set_last_message("the routing of the provider");
        let class = router.signature().arm_key();
        router.seed_relayed_turn();

        let state = locked(&router.state);
        assert_eq!(
            state.last_change_turn,
            Some(0),
            "the relayed turn is recorded at the provider's 0-based index"
        );
        assert_eq!(state.last_change_class.as_deref(), Some(class.as_str()));
        // The following turn (1) of the same class is steady; turn 2 is not.
        assert!(held_by_class(&state, 1, &class));
        assert!(!held_by_class(&state, 2, &class));
    }
}
