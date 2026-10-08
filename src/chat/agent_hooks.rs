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
//! by the cognitive router, among the models of the session's OWN provider (a session
//! keeps its provider). [`TurnRouter`] holds what that needs for one session and
//! [`directive_for_turn`] is the one decision function: the agent engine reaches it
//! through [`GraphSessionHooks::before_turn`], the legacy engine (Claude Code CLI) calls
//! it from `ChatManager::send_message` and sends the `set_model` control frame itself.

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
use super::provider::cognitive::decision::{DecideRequest, Decider, Pick};
use super::provider::cognitive::signature::{ContextHints, TaskSignature};
use super::provider::cognitive::{LearningStage, ProviderRoutingMode, RoutingSettings};

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

/// The models a provider instance offers, with their facts (the candidates of a
/// per-turn decision are restricted to the session's provider).
#[async_trait]
pub trait PoolSource: Send + Sync {
    /// Facts of the models of `provider_id` only.
    async fn pool(&self, provider_id: &str) -> Vec<ModelFacts>;
}

struct TurnState {
    current_model: String,
    last_change_turn: Option<u32>,
    /// Turn counter of the legacy engine, which has no harness to count turns.
    next_turn: u32,
}

/// What the per-turn decision of ONE session needs.
pub(crate) struct TurnRouter {
    decider: Arc<dyn Decider>,
    pool: Arc<dyn PoolSource>,
    /// Mode and stage, read once when the session opened.
    routing: RoutingSettings,
    provider_id: String,
    session_id: Option<Uuid>,
    project_slug: Option<String>,
    trust: bool,
    /// The request named its model: it is never replaced.
    explicit_model: bool,
    /// Models of this provider the conversation may be routed among (mixed); `None` = all.
    allowed_models: Option<Vec<String>>,
    /// The session can switch model between turns (known once it is open).
    set_model_live: AtomicBool,
    /// The user changed the model by hand: no more automatic change.
    manual: AtomicBool,
    last_message: Mutex<Option<String>>,
    state: Mutex<TurnState>,
    timeout: Duration,
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
    pub allowed_models: Option<Vec<String>>,
    pub current_model: String,
    /// Index the next turn counted by the router itself gets (legacy engine).
    pub next_turn: u32,
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl TurnRouter {
    pub(crate) fn new(spec: TurnRouterSpec) -> Self {
        Self {
            decider: spec.decider,
            pool: spec.pool,
            routing: spec.routing,
            provider_id: spec.provider_id,
            session_id: spec.session_id,
            project_slug: spec.project_slug,
            trust: spec.trust,
            explicit_model: spec.explicit_model,
            allowed_models: spec.allowed_models,
            set_model_live: AtomicBool::new(false),
            manual: AtomicBool::new(false),
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
}

/// The model of the turn about to start, `none` when it stays as it is.
///
/// * not `full` mode, a model named by the request, a model changed by hand, a session
///   that cannot switch model live: nothing is asked and nothing changes;
/// * `full` mode before the `auto` stage (and the turn right after a change): the decider
///   is still asked, so the decision is recorded, but nothing is applied;
/// * `full` + `auto`: the pick is used when the decision is applied and differs from the
///   current model. A failing or slow decider (2 s) never fails the turn.
pub(crate) async fn directive_for_turn(router: &TurnRouter, ctx: &TurnContext) -> TurnDirective {
    // The harness knows the model the turn would run on, whoever changed it.
    locked(&router.state).current_model = ctx.current_model.clone();
    // `full` routes everything; a pool (mixed, models ticked in the menu) routes among them.
    let routed =
        router.routing.mode == ProviderRoutingMode::Full || router.allowed_models.is_some();
    if !routed
        || router.explicit_model
        || router.manual.load(Ordering::SeqCst)
        || !router.set_model_live.load(Ordering::SeqCst)
    {
        return TurnDirective::none();
    }
    // Never two changes in consecutive turns: the turn right after one is only recorded.
    let just_changed = locked(&router.state)
        .last_change_turn
        .is_some_and(|turn| turn.checked_add(1) == Some(ctx.turn_index));
    let apply = router.routing.stage == LearningStage::Auto && !just_changed;
    let mut settings = router.routing.clone();
    // A pool is routed like `full` inside it, whatever the settings' mode.
    if router.allowed_models.is_some() {
        settings.mode = ProviderRoutingMode::Full;
    }
    if !apply {
        settings.stage = match settings.stage {
            LearningStage::Auto => LearningStage::Shadow,
            other => other,
        };
    }
    let message = locked(&router.last_message).clone().unwrap_or_default();
    let signature = TaskSignature::from_chat_request(
        &message,
        false,
        router.project_slug.as_deref(),
        ContextHints::default(),
    );
    let decision = tokio::time::timeout(router.timeout, async {
        let mut pool = router.pool.pool(&router.provider_id).await;
        if let Some(allowed) = &router.allowed_models {
            pool.retain(|facts| allowed.iter().any(|m| m == &facts.model));
        }
        let mut request = DecideRequest::new(signature, settings, pool);
        request.trust = router.trust;
        request.restrict_provider = Some(router.provider_id.clone());
        request.current = Some(Pick::new(&router.provider_id, &ctx.current_model));
        request.session_id = router.session_id;
        request.turn_index = Some(ctx.turn_index);
        router.decider.decide(&request).await
    })
    .await;
    let decision = match decision {
        Ok(Ok(decision)) => decision,
        Ok(Err(error)) => {
            warn!(provider = %router.provider_id, error = %error, "turn routing failed: the model stays");
            return TurnDirective::none();
        }
        Err(_) => {
            warn!(provider = %router.provider_id, "turn routing timed out: the model stays");
            return TurnDirective::none();
        }
    };
    if !apply || !decision.applied {
        return TurnDirective::none();
    }
    match decision.chosen {
        Some(pick) if pick.provider_id == router.provider_id && pick.model != ctx.current_model => {
            debug!(model = %pick.model, turn = ctx.turn_index, reason = %decision.reason, "turn routing changes the model");
            let mut state = locked(&router.state);
            state.current_model = pick.model.clone();
            state.last_change_turn = Some(ctx.turn_index);
            TurnDirective::model(pick.model)
        }
        _ => TurnDirective::none(),
    }
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
