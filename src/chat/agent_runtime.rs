//! The provider-neutral session engine (task B13, decision A18).
//!
//! `CHAT_PROVIDER_PATH=agent` makes [`ChatManager`](super::manager::ChatManager)
//! drive a session through an `Arc<dyn AgentSession>` (nexus contract) instead
//! of the Claude `InteractiveClient`. This module owns what that needs and
//! nothing Claude specific: the live sessions, the turn driver that turns the
//! provider's event stream into `ChatEvent`s (through [`EventMapper`]), the
//! out-of-turn pump (spawned via [`super::oob_listener::pump_out_of_band`]),
//! persistence of the events for replay, and the answers to
//! permission requests, interrupts, model and mode changes.
//!
//! The legacy path stays as it was; the manager decides which one serves a
//! session and a session never changes engine.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use futures::StreamExt;
use nexus_claude::agent::{
    AgentEvent, AgentProvider, AgentSession, CancelScope, Capabilities, InterruptScope,
    PermissionDecision, PermissionScope, PolicyMode, ProviderError, TurnInput,
};
use tokio::sync::{broadcast, Mutex, RwLock};
use uuid::Uuid;

use super::config::RetryConfig;
use super::manager::PermissionDeliveryError;
use super::manager::{
    CancelTaskResult, CancelToolsResult, ChatManager, CANCEL_TASK_CAP, CANCEL_TASK_WINDOW_SECS,
    CANCEL_TOOLS_CAP, CANCEL_TOOLS_WINDOW_SECS,
};
use super::provider::circuit_breaker::Outcome;
use super::provider::event_map::{
    default_subtype, out_of_band_to_chat_events, stop_reason_name, EventMapper,
};
use super::provider::outage::{HoldPolicy, OutageNotice, OutagePolicy, TurnHold};
use super::spill::SpillConfig;
use super::types::{
    BackgroundTaskInfo, ChatEvent, PendingMessage, PendingMessageKind, PermissionAnswerScope,
};
use crate::neo4j::models::ChatEventRecord;
#[allow(unused_imports)]
use crate::neo4j::traits::*;
use crate::neo4j::GraphStore;

const BROADCAST_BUFFER: usize = 1024;

/// Replaces every secret value delivered by the vault inside an event of the
/// provider, before anything stores or broadcasts it (same rule as the Claude
/// path: `ChatManager::mask_cli_message`). Fails CLOSED: an event that holds a
/// secret and cannot be masked is replaced by a marker that surfaces as a
/// visible error, never passed on in clear.
pub(crate) fn mask_agent_event(event: AgentEvent) -> AgentEvent {
    let masker = crate::vault::mask::global().snapshot();
    mask_agent_event_with(&masker, event)
}

/// [`mask_agent_event`] against an explicit masker (testable).
pub(crate) fn mask_agent_event_with(
    masker: &crate::vault::mask::Masker,
    event: AgentEvent,
) -> AgentEvent {
    match crate::vault::mask::mask_serde(masker, event) {
        Ok(masked) => masked,
        Err(_unmasked) => {
            tracing::error!(
                "vault: a provider event holding a secret could not be masked; withholding it"
            );
            AgentEvent::ProviderNotice {
                kind: crate::chat::manager::MASKING_FAILED_SUBTYPE.to_string(),
                data: serde_json::Value::Null,
            }
        }
    }
}

/// Epoch milliseconds now.
fn wall_clock_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A rate limit that names its reset, further away than a turn retry may wait
/// ([`MAX_RETRY_DELAY_MS`]): an exhausted quota, not a burst. Running the turn again would only
/// meet the same refusal, so it is said to the person (with the instant it serves again) instead.
fn waits_for_a_reset(error: &ProviderError, now_ms: u64) -> bool {
    matches!(
        error,
        ProviderError::RateLimited { resets_at_ms: Some(at), .. }
            if *at > now_ms.saturating_add(MAX_RETRY_DELAY_MS)
    )
}

/// The failure that ends a turn and is worth trying again: a retryable
/// `done.error` of an error turn, or a retryable terminal `error`.
fn retryable_failure(event: &AgentEvent, now_ms: u64) -> Option<ProviderError> {
    match event {
        // A turn its configured time limit stopped (`error_timeout`) is not a transient
        // failure: running it again would hit the same limit (ISO5). A silent model
        // stream (the endpoint's idle deadline) is `error_during_execution` and retries.
        AgentEvent::Done {
            subtype: Some(subtype),
            ..
        } if subtype == "error_timeout" => None,
        AgentEvent::Done {
            is_error: true,
            error: Some(error),
            ..
        }
        | AgentEvent::Error { error }
            if error.retryable() && !waits_for_a_reset(error, now_ms) =>
        {
            Some(error.clone())
        }
        _ => None,
    }
}

/// The failure of a turn that already showed text but no tool, and is still worth
/// trying again mid-generation: the transient mid-stream errors only — the rate
/// limit (429) and the upstream timeout (504). A turn its configured time limit
/// stopped (`error_timeout`) is not a transient failure (ISO5); the other
/// retryable classes (`Overloaded`, `EndpointUnreachable`) keep the old rule that
/// a turn which already showed something is not replayed, since they are more
/// likely a systemic condition than a mid-stream blip. Quota (402) is handled
/// elsewhere and never arrives as a retryable error here.
fn mid_generation_retryable(event: &AgentEvent, now_ms: u64) -> Option<ProviderError> {
    match event {
        AgentEvent::Done {
            subtype: Some(subtype),
            ..
        } if subtype == "error_timeout" => None,
        AgentEvent::Done {
            is_error: true,
            error: Some(error),
            ..
        }
        | AgentEvent::Error { error }
            if matches!(
                error,
                ProviderError::RateLimited { .. } | ProviderError::Timeout { .. }
            ) && !waits_for_a_reset(error, now_ms) =>
        {
            Some(error.clone())
        }
        _ => None,
    }
}

/// Whether the event ends a turn on a refusal of the ACCOUNT: a rate limit, or no credit left.
/// The model did not fail; the instance is out for a while (QUOTA).
fn is_quota_refusal(event: &AgentEvent) -> bool {
    match event {
        AgentEvent::Done {
            error: Some(error), ..
        }
        | AgentEvent::Error { error } => {
            matches!(error, ProviderError::RateLimited { .. })
                || super::provider::cognitive::decider::is_out_of_credit(error)
        }
        _ => false,
    }
}

/// Whether an event says the provider process is gone: the turn cannot go on, and
/// no later message can be delivered to this session (it must be resumed instead).
fn process_exited(event: &AgentEvent) -> bool {
    let error = match event {
        AgentEvent::Error { error } => Some(error),
        AgentEvent::Done { error, .. } => error.as_ref(),
        _ => None,
    };
    matches!(error, Some(ProviderError::ProcessExited { .. }))
}

/// The terminal event of a turn stopped before the provider answered (a Stop during
/// the pause before a retry).
fn interrupted_done() -> AgentEvent {
    AgentEvent::Done {
        stop_reason: nexus_claude::agent::StopReason::Interrupted,
        subtype: None,
        is_error: false,
        result_text: None,
        usage: Default::default(),
        cost: Default::default(),
        duration_ms: 0,
        duration_api_ms: None,
        num_turns: 0,
        model: None,
        provider_session_id: None,
        structured_output: None,
        error: None,
    }
}

/// Longest the runtime waits for the host's whole end of turn: at most five waits,
/// each bounded by the host to the post-stream step budget — context, re-injection
/// (a turn that compacted), objectives, the ONE write of the held context (its store
/// or its clear, never both), feedback — plus a margin. A backstop: the steps' own
/// budgets come first.
pub const AFTER_TURN_BACKSTOP: Duration =
    Duration::from_secs(5 * super::post_stream::POST_STREAM_STEP_BUDGET.as_secs() + 5);

/// Whether a provider of `kind` keeps a turn it ended `interrupted` (no error) in its
/// history, so what `prepare` put in front of it was delivered ([`TurnOutcome::answered`]).
/// Verified per provider, at the pinned nexus rev (9b5f470b):
/// - `native`: yes. nexus' `drive` (`native/loop.rs`) pushes the user message before
///   the run and commits the history on `StopCause::Interrupted`, the only cause of a
///   `done interrupted` without error.
/// - `codex`: not verified. The `done` maps the app-server's `turn/completed` status
///   `interrupted` (`codex/map.rs`); whether the thread keeps the user item is the
///   app-server's, not seen from here.
/// - `acp`: not verified. `done interrupted` is the agent's `cancelled` stop reason
///   (`acp/map.rs`); the protocol does not say whether the agent keeps the prompt.
/// - `claude_code`: not verified (the CLI owns its transcript).
///
/// Unverified: an interrupted turn is not answered, the context comes again with the
/// next turn — a bounded duplicate (at most the held context's cap) rather than a
/// loss.
pub(crate) fn keeps_interrupted_turns(kind: &str) -> bool {
    kind == "native"
}

/// Longest pause before a retry, whatever the provider or the configuration asks:
/// the pause is cut short by a Stop, but a client waits that long for the next word.
pub const MAX_RETRY_DELAY_MS: u64 = 30_000;

/// What one emitted event tells of the turn ([`TurnOutcome`]), and the tool calls
/// still waiting for their result.
fn track_turn(
    outcome: &mut TurnOutcome,
    tool_ids: &mut Vec<String>,
    pending_tools: &mut Vec<(String, Option<String>)>,
    event: &ChatEvent,
) {
    match event {
        ChatEvent::AssistantText { content, .. } => outcome.assistant_text.push_str(content),
        ChatEvent::ToolUse {
            id,
            tool,
            input,
            parent_tool_use_id,
            ..
        } => {
            outcome.tools.push((tool.clone(), input.clone()));
            tool_ids.push(id.clone());
            pending_tools.push((id.clone(), parent_tool_use_id.clone()));
        }
        // A tool call announced before its input was complete (Claude Code, ACP): its
        // input is the resolved one, so a `git commit` is told from an edit.
        ChatEvent::ToolUseInputResolved { id, input, .. } => {
            if let Some(i) = tool_ids.iter().position(|t| t == id) {
                outcome.tools[i].1 = input.clone();
            }
        }
        ChatEvent::ToolResult { id, .. } | ChatEvent::ToolCancelled { id, .. } => {
            pending_tools.retain(|(pending, _)| pending != id);
        }
        ChatEvent::CompactBoundary { .. } => outcome.compacted = true,
        _ => {}
    }
}

/// Whether an event is something the user has already seen of this turn.
fn shows_content(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::Text { .. }
            | AgentEvent::Thinking { .. }
            | AgentEvent::Delta { .. }
            | AgentEvent::ToolCall { .. }
            | AgentEvent::ToolResult { .. }
            | AgentEvent::PermissionAsk { .. }
            | AgentEvent::Question { .. }
    )
}

/// Delay before attempt `n`: what the provider asked (`retry_after`), else the
/// backoff of the chat's `RetryConfig` (the figures of `stream_response`); never
/// beyond [`MAX_RETRY_DELAY_MS`].
fn retry_delay_ms(error: &ProviderError, attempt: u32, retry: &RetryConfig) -> u64 {
    if let ProviderError::RateLimited {
        retry_after_ms: Some(ms),
        ..
    } = error
    {
        return (*ms).min(MAX_RETRY_DELAY_MS);
    }
    retry.delay_for_attempt(attempt).min(MAX_RETRY_DELAY_MS)
}

/// The input of a turn: its text, then the images the user attached, in order
/// (the ONE place the backend's attached images become the nexus input blocks).
/// The text loses the "no text could be extracted… /raw" line of each image it
/// carries inline (`without_notes_of_sent_images`): the picture itself is there.
fn turn_input(text: String, images: &[super::message_attachments::AttachedImage]) -> TurnInput {
    let text = super::message_attachments::without_notes_of_sent_images(&text, images);
    let mut input = TurnInput::text(text);
    input.blocks.extend(
        images
            .iter()
            .map(|image| nexus_claude::agent::InputBlock::Image {
                media_type: image.media_type.clone(),
                data_base64: image.data_base64.clone(),
            }),
    );
    input
}

/// What the Claude Code engine (`ChatManager::stream_response`, the CLI
/// subprocess) writes on the CLI's stdin for a turn.
///
/// - No image: the prompt as one string ([`InteractiveClient::send_and_receive_stream`]),
///   byte for byte what this engine always wrote.
/// - Images: the blocks of [`turn_input`] — the same input as the agent engine,
///   text then images in the order they were attached — checked and turned into
///   the CLI's blocks by nexus' own `claude_code::input::content_blocks` (type,
///   standard base64, 5 MiB), written by `send_blocks_and_receive_stream`.
///
/// The manager holds it opaque: the SDK block type stays here. A retry sends the
/// same value again (`send` on a clone): blocks stay blocks.
///
/// [`InteractiveClient::send_and_receive_stream`]: nexus_claude::InteractiveClient::send_and_receive_stream
#[derive(Debug, Clone)]
pub(crate) enum CliInput {
    Text(String),
    Blocks(Vec<nexus_claude::UserContentBlock>),
}

impl CliInput {
    /// The input of a turn whose (enriched) prompt is `text` and whose attached
    /// images are `images`.
    ///
    /// # Errors
    ///
    /// The `images_refused` event (`reason: invalid`) of an image nexus will not
    /// hand the CLI: nothing is to be written, the caller says it on the wire.
    pub(crate) fn of(
        text: String,
        images: &[super::message_attachments::AttachedImage],
    ) -> std::result::Result<Self, Box<ChatEvent>> {
        if images.is_empty() {
            return Ok(Self::Text(text));
        }
        nexus_claude::providers::claude_code::input::content_blocks(&turn_input(text, images))
            .map(Self::Blocks)
            .map_err(|error| {
                Box::new(image_refusal(&error).unwrap_or_else(|| {
                    images_refused(
                        "invalid",
                        format!("Error: The attached images could not be sent ({error}): the message was not sent."),
                    )
                }))
            })
    }

    /// Writes the turn on the CLI's stdin and streams its answer.
    pub(crate) async fn send(
        self,
        client: &mut nexus_claude::InteractiveClient,
    ) -> nexus_claude::Result<
        impl futures::Stream<Item = nexus_claude::Result<nexus_claude::Message>> + '_,
    > {
        use futures::future::Either;
        match self {
            Self::Text(prompt) => client
                .send_and_receive_stream(prompt)
                .await
                .map(Either::Left),
            Self::Blocks(blocks) => client
                .send_blocks_and_receive_stream(blocks)
                .await
                .map(Either::Right),
        }
    }
}

/// The error a turn whose images were refused shows: `code: images_refused`, the
/// `reason` (`unsupported`: the model has no vision; `invalid`: a picture the
/// provider will not take; `unreadable`: the backend could not read it).
pub(crate) fn images_refused(reason: &str, message: String) -> ChatEvent {
    ChatEvent::Error {
        message,
        parent_tool_use_id: None,
        code: Some("images_refused".to_string()),
        reason: Some(reason.to_string()),
        index: None,
        request_id: None,
    }
}

/// Whether an `InvalidRequest` detail is one of nexus' refusals of an image
/// (`providers::claude_code::input::check_image`). nexus gives these no stable
/// code: the three messages are matched as nexus words them today, and
/// `image_tests::the_nexus_image_refusals_are_recognised_as_nexus_words_them`
/// produces them with nexus' own `content_blocks`, so a change of wording on
/// the nexus side fails that test instead of silently changing the code.
fn is_nexus_image_refusal(detail: &str) -> bool {
    // "image media type `<type>` is not one of image/png, …"
    (detail.starts_with("image media type `") && detail.contains("` is not one of "))
        // "an image payload must be standard base64, without a `data:` prefix"
        || detail.starts_with("an image payload must be standard base64")
        // "an image is <n> bytes, more than the <max> bytes the CLI accepts"
        || (detail.starts_with("an image is ") && detail.ends_with(" bytes the CLI accepts"))
}

/// What the wire says when the provider refuses a turn that carries images
/// because of them: `Unsupported { images }`, or an `InvalidRequest` whose
/// detail is one of nexus' image checks (type, base64, size —
/// [`is_nexus_image_refusal`]). Any other failure, an unrelated
/// `InvalidRequest` included, is not about the images (`None`): it goes the way
/// of every failed turn.
fn image_refusal(error: &ProviderError) -> Option<ChatEvent> {
    match error {
        ProviderError::Unsupported { capability } if capability == "images" => Some(images_refused(
            "unsupported",
            "Error: The active model does not take images: the message was not sent.".to_string(),
        )),
        ProviderError::InvalidRequest { detail } if is_nexus_image_refusal(detail) => Some(images_refused(
            "invalid",
            format!("Error: The provider refused the attached image ({detail}): the message was not sent."),
        )),
        _ => None,
    }
}

/// The identifier of a native session that has no file, shell or web tool: its
/// `nexus-tools` executable was not found, or is not runnable, when it was opened
/// (`NEXUS_TOOLS_PATH`, next to the server, or on the `PATH`). It is the HOST that
/// knows it, not the provider's capabilities, so it is added at adoption.
pub const NEXUS_TOOLS_FEATURE: &str = "nexus_tools";

/// The identifier of a session that has none of the project-orchestrator tools: it
/// cannot carry an MCP server (`per_session_mcp` false: a remote Claude Code), or the
/// host did not give it one because its agent refuses them (OpenClaw's ACP bridge,
/// `ChatManager::carries_per_session_mcp`).
pub const PO_TOOLS_FEATURE: &str = "project_orchestrator_tools";

/// The identifier of a native session whose conversation is kept in MEMORY only: the
/// directory its transcripts would be written to lies inside a git work tree (a versioned
/// HOME, a data directory placed in a project), where a `git add -A` would publish them
/// (`transcripts::inside_git_worktree`, CTX6). The session works, but does not resume
/// after a server restart. The HOST knows it, so it is added at adoption.
pub const RESUME_AFTER_RESTART_FEATURE: &str = "resume_after_restart";

/// The identifier of a session whose provider has no sub-agents of its own
/// (`SubagentSupport::None`) and that the host gives no fallback: the backend runs the
/// `Agent` calls as child sessions for the native harness only
/// (`super::subagents::SubagentHooks`), so an ACP agent has none (M2, gap D3 of M0).
pub const SUBAGENTS_FEATURE: &str = "subagents";

/// What a session on the agent engine does NOT do, as the identifiers the
/// frontend knows (`hooks`, `compaction`, `images`, [`PO_TOOLS_FEATURE`]; the
/// host adds [`NEXUS_TOOLS_FEATURE`] and [`kind_degraded`]).
///
/// Two sources, kept apart on purpose:
/// - what THIS ENGINE (the backend) has not ported, whatever the provider can do:
///   nothing any more (enrichment, message queue, auto-continue, retry, NATS are
///   ported);
/// - what THE SESSION's capabilities say it cannot do: `images`, and `compaction`
///   when the provider emits no compaction signal.
///
/// `thinking` is never listed: a model that does not reason has nothing the engine
/// drops (the reasoning a provider emits is carried on every engine), and the frozen
/// capabilities already say `thinking: false` (M2, gap D4 of M0: refuted).
pub fn degraded_features(caps: &Capabilities) -> Vec<String> {
    // `retry` is NOT listed: the engine retries a retryable `done.error` (B15).
    // `enrichment` is NOT listed: every turn gets the graph context (`TurnServices::prepare`).
    // `message_queue` is NOT listed: a message sent during a turn is queued (`pending`).
    // `auto_continue` is NOT listed: a turn stopped on its limit is continued (`auto_continue_after`).
    // `nats` is NOT listed: events are published and the session answers the other
    // instances (`TurnServices::publish`, `ChatManager::spawn_agent_nats_listeners`).
    let mut missing: Vec<&str> = Vec::new();
    // The knowledge-graph hooks are served to a provider that runs hooks in its own loop
    // (`GraphSessionHooks`). A session that cannot carry an MCP server is the remote Claude
    // Code, which is given none: it keeps the entry.
    if caps.hooks != nexus_claude::agent::HookSupport::InProtocol || !caps.per_session_mcp {
        missing.push("hooks");
    }
    if !caps.compaction_signal {
        missing.push("compaction");
    }
    if !caps.images {
        missing.push("images");
    }
    // A session that cannot carry an MCP server (a remote Claude Code) has none
    // of the project-orchestrator tools.
    if !caps.per_session_mcp {
        missing.push(PO_TOOLS_FEATURE);
    }
    missing.into_iter().map(str::to_string).collect()
}

/// What a session of `kind` lacks that its capabilities do not say by themselves, as
/// `degraded_features` entries (M2, gaps D2 and D3 of M0):
/// - `compaction` for a native session whose context window is unknown: the native
///   harness compacts against the window (`should_compact` of nexus returns `false`
///   without one), so its `compaction_signal: true` would be a promise nobody keeps.
///   Measured: an OpenAI-compatible endpoint that reports no window (`fake_openai`,
///   `probe.context_window: null`). Claude Code and Codex compact in their own loop;
/// - [`SUBAGENTS_FEATURE`] for a provider without sub-agents of its own
///   (`SubagentSupport::None`) other than the native harness, the only one the host
///   runs child sessions for.
pub fn kind_degraded(kind: nexus_claude::agent::ProviderKind, caps: &Capabilities) -> Vec<String> {
    use nexus_claude::agent::{ProviderKind, SubagentSupport};
    let mut missing = Vec::new();
    if kind == ProviderKind::Native && caps.compaction_signal && caps.context_window.is_none() {
        missing.push("compaction".to_string());
    }
    if kind != ProviderKind::Native && caps.subagents == SubagentSupport::None {
        missing.push(SUBAGENTS_FEATURE.to_string());
    }
    missing
}

/// What a turn of the agent engine did, for what the host does after it
/// ([`TurnServices::after_turn`]) — what `stream_response` gathers for its
/// post-stream steps.
#[derive(Debug, Clone, Default)]
pub struct TurnOutcome {
    /// The assistant's text of the turn.
    pub assistant_text: String,
    /// The tools the turn called: name and input.
    pub tools: Vec<(String, serde_json::Value)>,
    /// The provider compacted the conversation during the turn (`compact_boundary`).
    pub compacted: bool,
    /// The turn was stopped (a Stop, or a message sent now).
    pub interrupted: bool,
    /// The turn stopped on its turn limit.
    pub hit_turn_limit: bool,
    /// Auto-continue was allowed after it (a continuation is on its way).
    pub auto_continue_allowed: bool,
    /// The model's context window as the session knows it (tokens), if known.
    pub context_window: Option<u64>,
    /// The turn ended on a `done` of the provider without error — `interrupted`
    /// included only for a provider known to keep such a turn in its history
    /// ([`keeps_interrupted_turns`]: nexus' native loop, the user message pushed
    /// before the run) — so what `prepare` put in front of it only for one turn may
    /// be dropped now. Not on
    /// `send_turn` accepting it: a provider may accept a turn before any request
    /// (the native harness spawns its run), then fail it (a 429 once the retries
    /// are spent). Not on the `done` the runtime makes itself for a Stop during the
    /// pause before a retry: no attempt of the turn was answered.
    pub answered: bool,
    /// The terminal `result` of the turn, captured from its final `done` (the same
    /// wire form `ChatEvent::Result` carries): its neutral stop reason, its result
    /// subtype and whether it was an error. Empty on a turn with no terminal `done`
    /// (stopped before it was answered), which must never read as a success.
    pub stop_reason: Option<String>,
    /// Result subtype: `success`, `error_during_execution`, … (see
    /// [`ChatEvent::Result`]).
    pub subtype: String,
    /// Whether the turn ended in error.
    pub is_error: bool,
    /// The typed provider error that caused a failure, as its stable code (ERR-1);
    /// `None` for a clean stop or an error without a typed provider cause.
    pub error_code: Option<String>,
    /// The turn ended on a refusal of the ACCOUNT (a rate limit, an exhausted quota, no credit):
    /// the instance is out for a while, which says nothing of how well the model works. The
    /// bandit is not fed it as a failure of the arm (`decision::turn_reward`).
    pub quota_refusal: bool,
}

/// What the host asks of the session after a turn: events to emit, then system
/// hints to queue (played as the next turns, as the Claude Code engine's queue).
#[derive(Debug, Default)]
pub struct AfterTurn {
    pub events: Vec<ChatEvent>,
    pub hints: Vec<String>,
}

/// What the host does around a turn of the agent engine, that the Claude Code
/// engine does around its own (`ChatManager::stream_response`). The manager
/// implements it; the runtime stays free of graph and transport concerns.
#[async_trait::async_trait]
pub trait TurnServices: Send + Sync {
    /// What the model receives for a turn: `sent` (the user's message `shown`,
    /// possibly behind a relayed history) with the knowledge graph's context.
    /// `turn` is the expansion of `shown` (`refs::turn::expand_user_turn`), made
    /// once by the runtime: its `#` references, its attachments.
    async fn prepare(
        &self,
        session_id: &str,
        shown: &str,
        sent: &str,
        turn: &crate::refs::turn::TurnExpansion,
    ) -> String;
    /// The system hint a turn continued automatically starts with
    /// (`post_stream::continuation_message`).
    async fn continuation(&self, session_id: &str) -> String;
    /// The images attached to the user's message `shown` (its stored form), to
    /// be sent inline with the turn. Err: an image that cannot be read, said on
    /// the wire, the turn not sent. Default: none.
    async fn images(
        &self,
        _shown: &str,
    ) -> std::result::Result<Vec<super::message_attachments::AttachedImage>, String> {
        Ok(Vec::new())
    }
    /// The model the turn about to be sent must run on because it carries images the
    /// model in force cannot read (F-R4): PO routes the turn to a candidate that reads
    /// them, BEFORE it is sent (the provider checks the active model's vision when the
    /// turn is sent). `None`: nothing to change. Default: never.
    async fn model_for_images(&self, _session_id: &str) -> Option<String> {
        None
    }
    /// The model [`Self::model_for_images`] named could not be made active: whatever the
    /// host decided for the turn on that ground is undone. Default: nothing.
    async fn image_model_not_applied(&self, _session_id: &str, _model: &str) {}
    /// Hands an event of the session to the other instances (NATS), as the Claude
    /// Code engine publishes each of its events. Default: nowhere.
    fn publish(&self, _session_id: &str, _event: &ChatEvent) {}
    /// Sees each event of the session as it is emitted (the work log of the turn).
    /// Default: nothing.
    fn observe(&self, _session_id: &str, _event: &ChatEvent) {}
    /// Sees the (masked) events the provider sent between turns, before the session emits
    /// them flattened (`background_output`). Whoever follows a Claude Code sub-agent still
    /// running in the background after the turn (`chat::mirror`) reads them in their
    /// structured form (`event_map::structured_out_of_turn`); nothing is emitted nor stored
    /// from here. Default: nothing.
    fn observe_out_of_turn(&self, _session_id: &str, _events: &[AgentEvent]) {}
    /// After each turn played (not one refused before it was sent): the
    /// Claude Code engine's post-stream steps — post-compaction re-injection,
    /// objective tracking, memory, feedback, observations. The host bounds each of
    /// its steps (`post_stream::POST_STREAM_STEP_BUDGET`); the runtime holds the whole
    /// to [`AFTER_TURN_BACKSTOP`] at most, the turn goes on. Default: nothing.
    async fn after_turn(&self, _session_id: &str, _outcome: &TurnOutcome) -> AfterTurn {
        AfterTurn::default()
    }
}

/// Where the runtime finds a provider instance by identifier. The nexus
/// registry plugs in here; until then only the built-in instance exists.
pub trait ProviderSource: Send + Sync {
    /// The instance, when it exists.
    fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>>;

    /// The lifecycle hooks (`UserPromptSubmit`, `Stop`, `SubagentStop`, `Notification`,
    /// `SessionStart`) the HISTORICAL Claude Code engine serves, as the engine hands them to
    /// `agent_hooks::serve_lifecycle`. The agent engine gets its hooks through the provider's
    /// `SessionSpec`, which a source can tap; the historical engine has no provider, so this is
    /// its seam. The default returns them unchanged; a test overrides it to record the calls.
    fn wrap_lifecycle(
        &self,
        hooks: Arc<dyn nexus_claude::agent::SessionHooks>,
    ) -> Arc<dyn nexus_claude::agent::SessionHooks> {
        hooks
    }
}

/// How many answered request ids a session keeps ([`AgentSessionHandle::answers_itself`]).
const ANSWERED_KEPT: usize = 1024;

/// A bounded set of request ids, the oldest forgotten first.
#[derive(Default)]
struct AnsweredIds {
    ids: std::collections::HashSet<String>,
    order: VecDeque<String>,
}

impl AnsweredIds {
    fn insert(&mut self, id: &str) {
        if self.ids.insert(id.to_owned()) {
            self.order.push_back(id.to_owned());
            while self.order.len() > ANSWERED_KEPT {
                if let Some(old) = self.order.pop_front() {
                    self.ids.remove(&old);
                }
            }
        }
    }

    fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }
}

/// One live session of the agent path.
pub struct AgentSessionHandle {
    /// Identifier of the session (the Neo4j `ChatSession` id).
    pub session_id: String,
    /// Provider instance serving it.
    pub provider_id: String,
    /// The provider session.
    pub session: Arc<dyn AgentSession>,
    /// Capabilities frozen at open.
    pub capabilities: Capabilities,
    /// Broadcast of the session's events.
    pub events_tx: broadcast::Sender<ChatEvent>,
    /// A turn is being streamed.
    pub is_streaming: AtomicBool,
    /// Text of the turn in progress (mid-stream joins).
    pub streaming_text: Mutex<String>,
    /// Structured events of the turn in progress (mid-stream joins).
    pub streaming_events: Mutex<Vec<ChatEvent>>,
    /// `{ id, kind }` stamped on `system_init`.
    provider: serde_json::Value,
    /// Tool policy stamped on `system_init`.
    tool_policy: serde_json::Value,
    /// What this session does not do (see [`degraded_features`]).
    degraded: Vec<String>,
    next_seq: AtomicI64,
    mapper: Mutex<EventMapper>,
    graph: Arc<dyn GraphStore>,
    uuid: Option<Uuid>,
    /// What the host does around a turn (`None`: nothing, the bare provider).
    services: Option<Arc<dyn TurnServices>>,
    /// Messages waiting for the running turn to end (`chat::pending_queue`).
    pending: Mutex<std::collections::VecDeque<PendingMessage>>,
    /// The running turn was stopped (by the user, or for a message sent now):
    /// the automated entries of the queue are dropped when it ends.
    interrupted: AtomicBool,
    /// Cancelled when the session closes: ends what listens on its behalf (NATS).
    pub closed: tokio_util::sync::CancellationToken,
    /// Continue a turn that stopped on its turn limit: set when the session opens
    /// (always on, capped — decision 7b95342a — unless the operator's kill switch is
    /// off); `set_auto_continue` no longer changes it.
    pub auto_continue: AtomicBool,
    /// The current chain of continuations in a row (`post_stream::auto_continue_allowed`).
    auto_continue_count: std::sync::atomic::AtomicU32,
    /// Continuations allowed in a row (0: no limit).
    max_auto_continues: std::sync::atomic::AtomicU32,
    /// The resume token (wire form) the graph holds for this session, as far as this
    /// handle knows: what the opener persisted, then each change [`Self::sync_resume_token`]
    /// wrote. Held across the write, so two writers never store an older token last.
    persisted_token: Mutex<Option<String>>,
    /// When `cancel_tools` was asked of this session: the sliding window of the
    /// per-session cap the Claude Code engine applies
    /// (`ActiveSession::cancel_tools_history`), so a Stop clicked in a loop
    /// cannot flood the provider here either.
    cancel_tools_history: Arc<Mutex<VecDeque<Instant>>>,
    /// Calls allowed within `cancel_tools_window` (`CANCEL_TOOLS_CAP`).
    cancel_tools_cap: u32,
    /// The window of the cap (`CANCEL_TOOLS_WINDOW_SECS`).
    cancel_tools_window: Duration,
    /// What the user granted "for the session" (`chat::session_grants`): the backend
    /// answers itself the later requests of THIS session they cover ([`Self::emit`]);
    /// never handed to the provider.
    session_grants: std::sync::Mutex<super::session_grants::SessionGrants>,
    /// The call each permission request still waiting asks for (request id -> call):
    /// what a `session` grant is derived from.
    asked_calls: std::sync::Mutex<HashMap<String, super::session_grants::AskedCall>>,
    /// The permission requests a grant of this session or a lasting rule of its person
    /// covers: decided in [`Self::emit_one`] BEFORE the request is broadcast, so whoever
    /// sees the request can know at once that the backend answers it itself (a sub-agent's
    /// parent: P18b, review of #724). A request whose automatic answer failed leaves the
    /// set (the user is asked).
    self_answered: std::sync::Mutex<std::collections::HashSet<String>>,
    /// The permission requests THIS handle saw asked and whose answer the provider took
    /// (the user's, or the automatic one), marked the moment the provider took it —
    /// before a lasting rule is written or the decision is stored. What
    /// [`Self::answers_itself`] calls answered: an id this handle never saw (one asked
    /// before a resume under the same id) is not (review of #727, finding 3). Bounded
    /// (`ANSWERED_KEPT`, oldest forgotten first).
    answered: std::sync::Mutex<AnsweredIds>,
    /// Who asks this session's permission requests (its provider kind): whether a
    /// request carries the whole call (`session_grants::Asker`).
    asker: super::session_grants::Asker,
    /// When `cancel_task` was asked of this session: the sliding window of the
    /// per-session cap the Claude Code engine applies to it
    /// (`ActiveSession::cancel_task_history`, `CANCEL_TASK_CAP` per
    /// `CANCEL_TASK_WINDOW_SECS`).
    cancel_task_history: Arc<Mutex<VecDeque<Instant>>>,
    /// How the provider states the cost of a turn (`session_record::CostFigure`).
    cost_figure: super::session_record::CostFigure,
    /// The provider is known to keep a turn it ended `interrupted` in its history
    /// ([`keeps_interrupted_turns`]): such a turn counts as answered.
    keeps_interrupted_turns: bool,
    /// Held across each read-then-write of the session record, so two writes of
    /// this session never lose one another's figure.
    record: Mutex<()>,
    /// The next user message was already counted when the session was created
    /// (the opening message: `message_count` starts at 1).
    opening_counted: AtomicBool,
    /// How a turn that failed before showing anything is retried: the chat's
    /// `RetryConfig` (`configure_retry`), as on the Claude Code engine.
    retry: std::sync::RwLock<RetryConfig>,
    /// When this session's instance counts as down, and how long a turn whose retry
    /// budget is spent is held for it (`configure_outage`): the outage detector and
    /// the hold policy of one instance, in one state machine
    /// (`chat::provider::outage::TurnHold`).
    outage: std::sync::Mutex<TurnHold>,
    /// Whether an oversized tool result is spilled out of the transcript
    /// (`super::spill`): the budget and preview sizes this session keeps.
    spill: SpillConfig,
    /// Where what the provider says about its account's limits goes (QUOTA): the registry the
    /// router and the send path read. `None`: nothing is kept.
    quota: Option<Arc<super::provider::quota::QuotaRegistry>>,
    /// When the session last did something (an event, a message): what the idle
    /// cleanup reads (`ChatManager::start_cleanup_task`).
    last_activity: std::sync::Mutex<Instant>,
    /// When its tool calls really ran (`tool_clock`): a `tool_timing` follows each result.
    tool_clock: Arc<super::tool_clock::ToolClock>,
    /// The runtime's live-session registry, so this handle can remove itself once
    /// its provider process is gone: a dead handle must not keep answering (and
    /// re-failing) a session the manager will otherwise resume.
    registry: std::sync::Weak<RwLock<HashMap<String, Arc<AgentSessionHandle>>>>,
    /// The provider said, OUT of any turn, that its process is gone (an idle CLI killed from
    /// outside). Only a turn in flight used to notice a death, so an idle session whose CLI was
    /// killed stayed live, holding its child slot (ZOMBIE-CLI, agent engine). The manager's
    /// dead-CLI watchdog closes such a session with the reason `process_died`.
    provider_dead: AtomicBool,
}

impl AgentSessionHandle {
    /// Persists (except transient events) and broadcasts one event, then the timing
    /// of the tool call it ends, if it ends one. A permission request a grant of this
    /// session covers (an earlier `session` answer) is answered here, `once`, its
    /// decision emitted with the scope `session` and the grant: the user is not asked.
    /// Only requests the provider ASKED reach this point: its policy (read-only access,
    /// denies, trust) has already decided.
    /// What a question about this call is about, as the HARNESS judges it (N23b): the origin the
    /// answer will be kept against, and every destination it covers. `None`: this call names no
    /// destination, so the question stays the ordinary one about the call.
    ///
    /// Two sources, the same on both sides of the question (the wire it decorates and the answer
    /// it records):
    ///
    /// * a page — the `url` of a fetch or a browser navigation, `https://` as nexus-tools reaches
    ///   it ([`nexus_tools::consent_subject`]);
    /// * a SEARCH — which names no destination of its own: what it reaches is the engines the
    ///   session was launched with, all of them, since nexus-tools falls back from one to the next
    ///   ([`nexus_tools::search_destinations`]). Asking about a search is asking this project for
    ///   its search engine, which is exactly what Settings → Network tools records.
    ///
    /// [`nexus_tools::consent_subject`]: super::provider::nexus_tools::consent_subject
    /// [`nexus_tools::search_destinations`]: super::provider::nexus_tools::search_destinations
    fn consent_about(
        &self,
        tool: &str,
        canonical: Option<&str>,
        input: &serde_json::Value,
    ) -> Option<(String, Vec<String>)> {
        if let Some(subject) = super::provider::nexus_tools::consent_subject(tool, canonical, input)
        {
            return Some((
                subject,
                super::provider::nexus_tools::destinations_of(tool, canonical, input),
            ));
        }
        if canonical != Some("WebSearch") {
            return None;
        }
        let engines = super::provider::nexus_tools::search_destinations(
            &super::provider::nexus_tools::search_origins_of(&self.session_id),
        );
        let subject = engines.first()?.clone();
        Some((subject, engines))
    }

    pub async fn emit(&self, event: ChatEvent) {
        let timing = self.tool_clock.observe(&event, chrono::Utc::now());
        let covered = self.emit_one(event).await;
        if let Some(timing) = timing {
            self.emit_one(timing).await;
        }
        let Some((covered, cover)) = covered else {
            return;
        };
        // On the tool clock BEFORE the provider has it, as for an answer of the user,
        // and handed over alone among the answers to this request.
        let answering = self.tool_clock.answering(&covered).await;
        let mark = self.tool_clock.decided(&covered, true, chrono::Utc::now());
        let answered = self
            .session
            .answer_permission(&covered, PermissionDecision::allow_once())
            .await;
        if answered.is_err() {
            self.tool_clock.undecided(&covered, mark);
        }
        drop(answering);
        match answered {
            Ok(()) => {
                self.mark_answered(&covered);
                // Answered: it waits no more, and the set of the requests the backend
                // answers itself forgets it — it used to keep every one for the session's
                // whole life (lot review of #721, finding 2c). `answers_itself` still says
                // `true` for it: a request no longer waiting needs nobody.
                self.asked_calls
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(&covered);
                self.self_answered
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(&covered);
                let _ = self
                    .emit_one(ChatEvent::PermissionDecision {
                        id: covered,
                        allow: true,
                        scope: Some(if cover.lasting {
                            PermissionAnswerScope::Always
                        } else {
                            PermissionAnswerScope::Session
                        }),
                        rule: Some(cover.rule),
                    })
                    .await;
            }
            // Left waiting: the user answers it.
            Err(e) => {
                self.self_answered
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(&covered);
                tracing::warn!(
                    session_id = %self.session_id,
                    error = %e,
                    "a call a session grant covers could not be allowed: the user is asked"
                )
            }
        }
    }

    /// [`Self::emit`] for one event; returns the id of a permission request a grant of
    /// this session covers, and that grant as the user is shown it.
    async fn emit_one(
        &self,
        mut event: ChatEvent,
    ) -> Option<(String, super::session_grants::Cover)> {
        self.touch();
        // Spill an oversized tool result BEFORE it reaches the transcript, the
        // observers and the broadcast: the stored `tool_result` (the model-facing
        // copy) keeps a bounded preview, and the full text lands in a
        // `tool_result_spill` record under a stable key (`super::spill`). Small
        // results, a disabled budget, and a store failure leave it inline.
        if let ChatEvent::ToolResult { id, result, .. } = &mut event {
            super::spill::spill_if_oversized(
                &self.graph,
                self.uuid,
                id,
                &self.next_seq,
                result,
                &self.spill,
            )
            .await;
        }
        // The model IN FORCE is what the session node says: the sessions list and the model
        // menu read it (SDA-2). A change reaches the backend as this event whoever made it: the
        // user, the image router, or the `before_turn` hook of the per-turn router, which the
        // harness applies INSIDE the provider without going through the manager — that last
        // path persisted nothing and left the node on the opening value (`auto`, or the
        // opening model). One write here covers all of them. `routing_mode` is a separate
        // property: the Auto marker survives. A failed write is logged, not an error: the
        // provider already runs the model, and the next change writes again.
        if let ChatEvent::ModelChanged { model } = &event {
            if let Some(uuid) = self.uuid {
                if let Err(e) = self.graph.update_chat_session_model(uuid, model).await {
                    tracing::warn!(
                        session_id = %self.session_id,
                        %model,
                        error = %e,
                        "the active model could not be written on the session"
                    );
                }
            }
        }
        // What only the session owner knows rides on `system_init`.
        if let ChatEvent::SystemInit {
            provider,
            capabilities,
            tool_policy,
            engine,
            degraded_features,
            ..
        } = &mut event
        {
            // The client must be able to tell which engine drives the session,
            // and, when Claude Code was forced here, what it no longer does.
            engine.get_or_insert_with(|| "agent".to_string());
            degraded_features.get_or_insert_with(|| self.degraded.clone());
            provider.get_or_insert_with(|| self.provider.clone());
            capabilities.get_or_insert_with(|| {
                serde_json::to_value(self.offered_capabilities()).unwrap_or_default()
            });
            // A native session whose `nexus-tools` runs with its file scope lifted says so
            // (decision fce98b8a): its file tools reach outside its directories, the deny list
            // apart. Absent otherwise (the scope is the session's directories).
            if crate::auth::agent_tokens::is_scope_lifted(&self.session_id) {
                if let Some(object) = capabilities.as_mut().and_then(|c| c.as_object_mut()) {
                    object.insert("file_scope".to_owned(), serde_json::json!("unscoped"));
                }
            }
            tool_policy.get_or_insert_with(|| self.tool_policy.clone());
        }
        let mut covered = None;
        // A command the shell may have moved with: lasting rules stop covering commands.
        if let Some(call) = super::session_grants::observed_call(self.asker, &event) {
            self.session_grants
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .observe_tool_use(&call);
        }
        if let ChatEvent::PermissionRequest {
            id,
            tool,
            input,
            canonical,
            always,
            consent,
            ..
        } = &mut event
        {
            let call = super::session_grants::AskedCall {
                asker: self.asker,
                tool: tool.clone(),
                canonical: canonical.clone(),
                input: input.clone(),
            };
            // A question about a DESTINATION (N23b, CONSENT-WIRE): its subject is named on the
            // wire by the very function the answer path records with
            // (`deliver_permission_answer`), so what the person is shown is what PO keeps. A
            // search names no page of its own: its question is about the engines of the session.
            *consent = self.consent_about(tool, canonical.as_deref(), input).map(
                |(subject, destinations)| super::types::ConsentSubject {
                    subject,
                    destinations,
                },
            );
            // Whether this very question may be answered `always` (the client offers it then).
            // About a destination: the project's network consent, kept only by a session that
            // keeps lasting rules in a project (`keep_origin_consents`), on a provider that
            // offers `always`. About a call: the backend's comparator for that call.
            *always = if consent.is_some() {
                let offers_always = self
                    .offered_capabilities()
                    .permission_scopes
                    .contains(&PermissionScope::Always);
                let grants = self
                    .session_grants
                    .lock()
                    .unwrap_or_else(|p| p.into_inner());
                (offers_always
                    && grants.keeps_lasting(self.asker)
                    && grants.lasting_scope().is_some())
                .then_some(true)
            } else {
                self.session_grants
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .lasting_grant_for(&call)
                    .map(|_| true)
            };
            covered = Box::pin(super::session_grants::cover(&self.session_grants, &call))
                .await
                .map(|cover| (id.clone(), cover));
            if covered.is_some() {
                self.self_answered
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert(id.clone());
            }
            self.asked_calls
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(id.clone(), call);
        }
        match &event {
            ChatEvent::StreamDelta { text, .. } => self.streaming_text.lock().await.push_str(text),
            // `active_tasks_update` is a snapshot, never replayed (the client reads
            // `GET .../background-tasks` when it joins), as on the Claude Code engine.
            ChatEvent::StreamingStatus { .. }
            | ChatEvent::PendingQueue { .. }
            | ChatEvent::ActiveTasksUpdate { .. } => {}
            other => self.streaming_events.lock().await.push(other.clone()),
        }
        if !matches!(
            event,
            ChatEvent::StreamDelta { .. }
                | ChatEvent::StreamingStatus { .. }
                | ChatEvent::PendingQueue { .. }
                | ChatEvent::ActiveTasksUpdate { .. }
        ) {
            if let Some(uuid) = self.uuid {
                // A stopped turn is stored in its neutral form, as on the Claude Code
                // engine: the reloaded thread shows no error block under it.
                let stored = super::manager::stored_interrupted_result(&event);
                let record = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: self.next_seq.fetch_add(1, Ordering::SeqCst),
                    event_type: event.event_type().to_string(),
                    data: serde_json::to_string(stored.as_ref().unwrap_or(&event))
                        .unwrap_or_default(),
                    created_at: chrono::Utc::now(),
                };
                let _ = self.graph.store_chat_events(uuid, vec![record]).await;
            }
        }
        if let Some(services) = &self.services {
            services.publish(&self.session_id, &event);
            services.observe(&self.session_id, &event);
        }
        let _ = self.events_tx.send(event);
        covered
    }

    /// Persists the provider's resume token when it is not the one the graph holds.
    ///
    /// A provider may name its session only after the session opened: the Claude Code
    /// CLI gives its session id with its first `system/init` and each `result`, so at
    /// open the token is still `None`. Called where the token can appear or change (the
    /// session's start, the end of each turn, a batch of out-of-turn events, the close):
    /// one read of the provider's token each time, a graph write only when it changed.
    /// A failed write is retried at the next call.
    pub(crate) async fn sync_resume_token(&self) {
        let Some(uuid) = self.uuid else { return };
        let Some(token) = self.session.resume_token().map(|t| t.to_wire()) else {
            return;
        };
        let mut persisted = self.persisted_token.lock().await;
        if persisted.as_deref() == Some(token.as_str()) {
            return;
        }
        match self
            .graph
            .update_chat_session_harness(uuid, None, Some(&token))
            .await
        {
            Ok(()) => *persisted = Some(token),
            Err(e) => tracing::warn!(
                session_id = %self.session_id,
                error = %e,
                "Failed to persist the resume token (retried at the next turn)"
            ),
        }
    }

    /// Removes this handle from the runtime's registry once its provider process
    /// has exited. A dead handle must not keep answering (and re-failing) a
    /// session the manager will resume instead: once it is gone, the next message
    /// takes the resume path. Only this handle is removed — a newer handle that
    /// replaced it stays.
    /// Whether the provider reported, out of a turn, that its process is gone.
    pub(crate) fn provider_is_dead(&self) -> bool {
        self.provider_dead.load(Ordering::SeqCst)
    }

    async fn detach_after_process_exit(&self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut sessions = registry.write().await;
        if sessions
            .get(&self.session_id)
            .is_some_and(|live| std::ptr::eq(live.as_ref(), self))
        {
            sessions.remove(&self.session_id);
        }
    }

    /// Retries this session's failed turns as `retry` says (the chat's configuration).
    pub fn configure_retry(&self, retry: RetryConfig) {
        *self.retry.write().unwrap_or_else(|e| e.into_inner()) = retry;
    }

    pub(crate) fn retry_config(&self) -> RetryConfig {
        self.retry.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Sets the outage policies of this session: when its instance is declared down,
    /// and how long a turn is held for it (`configure_retry`'s twin).
    pub fn configure_outage(&self, outage: OutagePolicy, hold: HoldPolicy) {
        self.outage
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_policies(outage, hold);
    }

    /// What the provider says about the limits of its account, and the refusals that date them
    /// (QUOTA), go to the registry as they arrive: the router excludes the instance until its
    /// reset, and the send path says so. Never an effect on the turn itself.
    fn observe_quota(&self, event: &AgentEvent) {
        let Some(quota) = &self.quota else { return };
        let refusal = match event {
            AgentEvent::Quota { quota: snapshot } => {
                quota.observe(&self.provider_id, snapshot);
                return;
            }
            AgentEvent::Done {
                error: Some(error), ..
            }
            | AgentEvent::Error { error } => error,
            _ => return,
        };
        match refusal {
            ProviderError::RateLimited {
                retry_after_ms,
                resets_at_ms,
            } => quota.mark_rate_limited(&self.provider_id, *resets_at_ms, *retry_after_ms),
            error if super::provider::cognitive::decider::is_out_of_credit(error) => {
                quota.mark_out_of_credit(&self.provider_id);
            }
            _ => {}
        }
    }

    /// Feeds one final failed call of this session to the outage detector and emits
    /// the notice when this outcome flipped the instance — the `down` transition
    /// only: a further failure of an instance already down is part of the same
    /// outage and says nothing new.
    async fn record_outage_failure(&self, error: &ProviderError) {
        let cause = super::provider::errors::cause_of(error);
        let notice = {
            let mut hold = self.outage.lock().unwrap_or_else(|e| e.into_inner());
            hold.record(Outcome::Failure, Instant::now(), cause.as_deref())
        };
        if let Some(notice) = notice {
            self.emit_outage(notice).await;
        }
    }

    /// The first success after an outage: the instance is back, the hold is over.
    async fn record_outage_success(&self) {
        let notice = {
            let mut hold = self.outage.lock().unwrap_or_else(|e| e.into_inner());
            hold.record(Outcome::Success, Instant::now(), None)
        };
        if let Some(notice) = notice {
            self.emit_outage(notice).await;
        }
    }

    /// Whether a turn whose retry budget is spent may go on: `Some(wait)` while the
    /// instance is down and the hold budget allows, `None` when the failure stands.
    fn hold_a_failed_turn(&self) -> Option<Duration> {
        self.outage
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hold_on(Instant::now())
    }

    /// Emits the persisted, attributable event of an outage transition.
    async fn emit_outage(&self, notice: OutageNotice) {
        self.emit(notice.to_chat_event(chrono::Utc::now())).await;
    }

    /// The session did something now.
    pub fn touch(&self) {
        *self.last_activity.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    /// How long the session has done nothing.
    pub fn idle_for(&self) -> Duration {
        self.last_activity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .elapsed()
    }

    /// Tests: the session last did something `ago` before now.
    #[cfg(test)]
    pub(crate) fn backdate_activity(&self, ago: Duration) {
        let then = Instant::now()
            .checked_sub(ago)
            .expect("a representable instant");
        *self.last_activity.lock().unwrap_or_else(|e| e.into_inner()) = then;
    }

    /// Tests: the provider's background task snapshot, as its event would set it
    /// (no event emitted, no activity recorded).
    #[cfg(test)]
    pub(crate) async fn set_background_tasks(
        &self,
        tasks: Vec<nexus_claude::agent::BackgroundTask>,
    ) {
        self.mapper
            .lock()
            .await
            .map(&AgentEvent::BackgroundTasks { tasks });
    }

    /// The opening message of the session was counted when the session was created:
    /// its turn does not count it again (`session_record`).
    pub fn opening_message_counted(&self) {
        self.opening_counted.store(true, Ordering::SeqCst);
    }

    /// The opening message was not played after all: the next one counts.
    pub fn forget_opening_count(&self) {
        self.opening_counted.store(false, Ordering::SeqCst);
    }

    /// Counts a user message on the session record — the opening one excepted, already
    /// counted — under [`super::session_record::RECORD_WRITE_BUDGET`].
    async fn count_user_message(&self) {
        if self.opening_counted.swap(false, Ordering::SeqCst) {
            return;
        }
        let Some(uuid) = self.uuid else { return };
        let _held = self.record.lock().await;
        let write = super::session_record::count_user_message(&self.graph, uuid);
        match tokio::time::timeout(super::session_record::RECORD_WRITE_BUDGET, write).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::warn!(session_id = %self.session_id, error = %e, "Failed to count the user message")
            }
            Err(_) => {
                tracing::warn!(session_id = %self.session_id, "Counting the user message took too long: abandoned")
            }
        }
    }

    /// Adds what a turn cost to the session record (nothing for an unknown price),
    /// under [`super::session_record::RECORD_WRITE_BUDGET`].
    async fn add_turn_cost(&self, usd: Option<f64>) {
        let (Some(uuid), Some(_)) = (self.uuid, usd) else {
            return;
        };
        let _held = self.record.lock().await;
        let write = super::session_record::add_turn_cost(&self.graph, uuid, usd, self.cost_figure);
        match tokio::time::timeout(super::session_record::RECORD_WRITE_BUDGET, write).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::warn!(session_id = %self.session_id, error = %e, "Failed to record the cost of the turn")
            }
            Err(_) => {
                tracing::warn!(session_id = %self.session_id, "Recording the cost of the turn took too long: abandoned")
            }
        }
    }

    /// Starts a turn and drives it to its terminal event in the background. A
    /// turn already running does not refuse the message: it is queued and the
    /// running turn interrupted so it is read sooner — what the Claude Code
    /// engine does (`ChatManager::send_message`).
    pub async fn send_message(self: &Arc<Self>, text: &str) -> Result<()> {
        self.send_message_relayed(text, text).await
    }

    /// Sends `sent` to the model while the conversation shows and stores `shown`: a
    /// relayed history (B-SW) goes in front of the user's message without becoming
    /// part of it.
    pub async fn send_message_relayed(self: &Arc<Self>, shown: &str, sent: &str) -> Result<()> {
        // `/compact` the host runs itself (ISO3). Not behind a relayed history: the relay
        // must reach the model, so that message stays a turn.
        let compaction = if shown == sent {
            self.manual_compaction(shown)
        } else {
            None
        };
        {
            // `is_streaming` is read and set under the queue lock, the lock the end
            // of a turn holds to decide there is nothing left to drain: a message
            // is either seen by the running turn or starts its own.
            let mut queue = self.pending.lock().await;
            if self.is_streaming.load(Ordering::SeqCst) {
                if compaction.is_some() {
                    // Never queued behind the turn nor interrupting it: refused, with the
                    // code nexus gives a `compact` during a turn.
                    drop(queue);
                    self.emit(ChatEvent::UserMessage {
                        content: shown.to_string(),
                    })
                    .await;
                    self.count_user_message().await;
                    self.emit(super::manual_compaction::refusal_event(
                        &ProviderError::TurnInProgress,
                    ))
                    .await;
                    return Ok(());
                }
                queue.push_back(PendingMessage::user(shown.to_string()));
                drop(queue);
                self.interrupt_for_the_queue().await;
                return Ok(());
            }
            self.is_streaming.store(true, Ordering::SeqCst);
        }
        self.start(PendingMessageKind::User, shown, sent).await
    }

    /// `Some(instructions)` when `text` is `/compact` and this host runs it: every
    /// provider but Claude Code, whose CLI reads the command itself (it stays a turn).
    fn manual_compaction(&self, text: &str) -> Option<Option<String>> {
        if self.asker == super::session_grants::Asker::ClaudeCode {
            return None;
        }
        super::manual_compaction::compact_command(text)
    }

    /// Runs `/compact` (`is_streaming` held by the caller, no turn of the provider
    /// running): the message is shown and counted like any, then the provider is asked
    /// to compact — when it declares `compaction_manual`, else a typed refusal. The
    /// `compaction` events come out of turn from the provider; the outcome is said here.
    /// A compaction that ran ends like a turn that compacted (`after_turn`): the work
    /// context is restored in front of the next turn, as after an automatic one.
    async fn compact_on_request(&self, shown: &str, instructions: Option<String>) {
        self.interrupted.store(false, Ordering::SeqCst);
        self.emit(ChatEvent::UserMessage {
            content: shown.to_string(),
        })
        .await;
        self.count_user_message().await;
        if !self.capabilities.compaction_manual {
            tracing::info!(session_id = %self.session_id, "/compact refused: the provider does not declare compaction_manual");
            self.emit(super::manual_compaction::refusal_event(
                &ProviderError::unsupported("compaction_manual"),
            ))
            .await;
            return;
        }
        self.emit(ChatEvent::StreamingStatus { is_streaming: true })
            .await;
        match self.session.compact(instructions).await {
            Ok(outcome) => {
                tracing::info!(
                    session_id = %self.session_id,
                    compacted = outcome.compacted,
                    messages_summarised = outcome.messages_summarised,
                    "manual compaction"
                );
                self.emit(super::manual_compaction::outcome_event(&outcome))
                    .await;
                if outcome.compacted {
                    self.after_turn(&TurnOutcome {
                        compacted: true,
                        context_window: self.capabilities.context_window.map(|w| w.value),
                        ..TurnOutcome::default()
                    })
                    .await;
                }
            }
            Err(error) => {
                tracing::warn!(session_id = %self.session_id, %error, "manual compaction failed");
                self.emit(super::manual_compaction::refusal_event(&error))
                    .await;
            }
        }
    }

    /// Holds a user message until the running turn ends — it interrupts nothing
    /// (`ChatManager::queue_user_message`). Returns `true` when the message was
    /// held, `false` when the session was idle and the message simply sent.
    pub async fn queue_message(self: &Arc<Self>, content: &str) -> Result<bool> {
        {
            let mut queue = self.pending.lock().await;
            if self.is_streaming.load(Ordering::SeqCst) {
                queue.push_back(PendingMessage::held_user(content.to_string()));
                let messages = super::pending_queue::snapshot(&queue);
                drop(queue);
                self.emit(ChatEvent::PendingQueue { messages }).await;
                return Ok(true);
            }
            self.is_streaming.store(true, Ordering::SeqCst);
        }
        self.start(PendingMessageKind::User, content, content)
            .await
            .map(|()| false)
    }

    /// Queues a system hint for after the running turn, without interrupting it;
    /// an idle session gets it as a user message (`ChatManager::inject_hint`).
    pub async fn inject_hint(self: &Arc<Self>, content: &str) -> Result<()> {
        {
            let mut queue = self.pending.lock().await;
            if self.is_streaming.load(Ordering::SeqCst) {
                queue.push_back(PendingMessage::system_hint(content.to_string()));
                return Ok(());
            }
        }
        self.send_message(content).await
    }

    /// The held messages, as clients see them.
    pub async fn queue_snapshot(&self) -> Vec<super::types::PendingQueueEntry> {
        super::pending_queue::snapshot(&*self.pending.lock().await)
    }

    /// Edits, drops, moves to the front or sends now one held message, then
    /// publishes the list again (`ChatManager::pending_queue_op`).
    pub async fn queue_op(&self, op: &super::pending_queue::QueueOp) {
        let (outcome, messages) = {
            let mut queue = self.pending.lock().await;
            let outcome = super::pending_queue::apply(&mut queue, op);
            (outcome, super::pending_queue::snapshot(&queue))
        };
        let streaming = self.is_streaming.load(Ordering::SeqCst);
        tracing::info!(
            session_id = %self.session_id,
            // The kind and the id only: an edit carries the user's text.
            op = op.name(),
            id = ?op.id(),
            found = outcome.found,
            interrupt = outcome.interrupt,
            streaming,
            held_left = messages.len(),
            "queue op applied"
        );
        self.emit(ChatEvent::PendingQueue { messages }).await;
        if outcome.interrupt && streaming {
            self.interrupt_for_the_queue().await;
        }
    }

    /// Cuts the running turn short so the message queued in front is read now.
    ///
    /// The message itself is delivered by the run's own loop (`drive` →
    /// `next_queued`) once the turn ends: an interrupt the provider could not apply
    /// (no turn of its own running — the run is between two turns, or in its
    /// end-of-turn steps) only means it waits for those steps, never that it is lost.
    async fn interrupt_for_the_queue(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
        match self.session.interrupt(InterruptScope::TurnAndTools).await {
            Ok(outcome) if outcome.turn_interrupted => {
                tracing::info!(
                    session_id = %self.session_id,
                    turn_interrupted = true,
                    tools_cancelled = outcome.tools_cancelled,
                    "turn interrupted for a queued message"
                );
            }
            Ok(outcome) => {
                tracing::warn!(
                    session_id = %self.session_id,
                    turn_interrupted = false,
                    tools_cancelled = outcome.tools_cancelled,
                    streaming = self.is_streaming.load(Ordering::SeqCst),
                    "the provider had no turn to interrupt for a queued message: it is sent when the run's current step ends"
                );
            }
            Err(e) => {
                tracing::warn!(session_id = %self.session_id, error = %e, "interrupting the turn for a queued message failed");
            }
        }
    }

    /// Opens the first turn of a run (`is_streaming` already claimed) and drives
    /// it, then whatever the queue holds, in the background.
    async fn start(
        self: &Arc<Self>,
        kind: PendingMessageKind,
        shown: &str,
        sent: &str,
    ) -> Result<()> {
        if kind == PendingMessageKind::User && shown == sent {
            if let Some(instructions) = self.manual_compaction(shown) {
                // Like a turn: in the background, `is_streaming` held (a message sent
                // meanwhile is queued, and its interrupt stops the summary as it stops a
                // turn), then the queue is played.
                let me = Arc::clone(self);
                let shown = shown.to_string();
                tokio::spawn(async move {
                    me.compact_on_request(&shown, instructions).await;
                    me.drive(None).await;
                });
                return Ok(());
            }
        }
        match self.open_turn(kind, shown, sent).await {
            // `None`: stopped while it was prepared, only the queue is left to play.
            Ok(first) => {
                let me = Arc::clone(self);
                tokio::spawn(async move { me.drive(first).await });
                Ok(())
            }
            Err(error) => {
                // A message queued while this turn was being opened (the enrichment
                // awaits) still runs: the run goes on without this turn. Nothing
                // queued: the session stops streaming, decided under the queue lock.
                let queued = {
                    let queue = self.pending.lock().await;
                    if queue.is_empty() {
                        self.is_streaming.store(false, Ordering::SeqCst);
                    }
                    !queue.is_empty()
                };
                if queued {
                    let me = Arc::clone(self);
                    tokio::spawn(async move { me.drive(None).await });
                }
                Err(anyhow::Error::new(error))
            }
        }
    }

    /// Shows the message of a turn and sends the turn to the provider (`None`: the
    /// turn was stopped before it could be sent).
    async fn open_turn(
        &self,
        kind: PendingMessageKind,
        shown: &str,
        sent: &str,
    ) -> std::result::Result<Option<(nexus_claude::agent::EventStream, TurnInput)>, ProviderError>
    {
        // A Stop belongs to the turn it stopped: the new turn starts unstopped.
        self.interrupted.store(false, Ordering::SeqCst);
        let from_user = kind == PendingMessageKind::User;
        match kind {
            PendingMessageKind::SystemHint => {
                self.emit(ChatEvent::SystemHint {
                    content: shown.to_string(),
                })
                .await
            }
            // A background output was already shown when it arrived.
            PendingMessageKind::BackgroundOutput => {}
            PendingMessageKind::User => {
                self.emit(ChatEvent::UserMessage {
                    content: shown.to_string(),
                })
                .await;
                self.count_user_message().await;
            }
            // TeamMessage: already persisted and broadcast by drain before
            // this method is called; no re-emit needed here. Does not count
            // as a user message turn.
            PendingMessageKind::TeamMessage { .. } => {}
        }
        // The same expansion as `stream_response` (`refs::turn`), for every turn
        // whatever started it (a message, the queue, auto-continue, another
        // instance): a message with `#` references reaches the model as its visible
        // text plus the pointers, never as the raw block; one without is sent as it
        // always was (a relayed history, if any, stays in front).
        let turn =
            crate::refs::turn::expand_user_turn_in(&self.graph, shown, &self.session_id).await;
        if let Some(event) = turn.event() {
            self.emit(event).await;
        }
        // The knowledge graph's context, as the Claude Code engine gives it to its turns.
        let sent = match &self.services {
            Some(services) => services.prepare(&self.session_id, shown, sent, &turn).await,
            None => turn.native_prompt(shown, sent),
        };
        // A Stop while the turn was prepared (the enrichment awaits) found no turn
        // to interrupt at the provider: it stops this one before it is sent.
        if self.interrupted.load(Ordering::SeqCst) {
            tracing::info!(session_id = %self.session_id, "Turn stopped before it was sent");
            return Ok(None);
        }
        // The images the user attached go with the turn, inline (the provider
        // checks them: a refusal is said on the wire, the turn is not sent).
        let images = match &self.services {
            Some(services) if from_user => match services.images(shown).await {
                Ok(images) => images,
                Err(reason) => {
                    tracing::warn!(session_id = %self.session_id, %reason, "an attached image could not be read");
                    self.emit(images_refused(
                        "unreadable",
                        format!("Error: {reason}: the message was not sent."),
                    ))
                    .await;
                    return Ok(None);
                }
            },
            _ => Vec::new(),
        };
        // F-R4: images the model in force cannot read go to a model PO routes to that
        // reads them, made active before the turn is sent. A change that fails leaves the
        // provider to say it refuses the images.
        if !images.is_empty() {
            if let Some(services) = &self.services {
                if let Some(model) = services.model_for_images(&self.session_id).await {
                    if let Err(error) = self.set_model(&model).await {
                        tracing::warn!(session_id = %self.session_id, %model, %error, "the model that reads the images could not be made active");
                        services
                            .image_model_not_applied(&self.session_id, &model)
                            .await;
                    }
                }
            }
        }
        let input = turn_input(sent, &images);
        let stream = match self.session.send_turn(input.clone()).await {
            Ok(stream) => stream,
            Err(error) if !images.is_empty() => match image_refusal(&error) {
                Some(event) => {
                    tracing::info!(session_id = %self.session_id, %error, "the provider refused the attached images");
                    self.emit(event).await;
                    return Ok(None);
                }
                None => return Err(error),
            },
            Err(error) => return Err(error),
        };
        self.streaming_text.lock().await.clear();
        self.streaming_events.lock().await.clear();
        self.emit(ChatEvent::StreamingStatus { is_streaming: true })
            .await;
        Ok(Some((stream, input)))
    }

    /// Plays turns until the queue is empty: the turn given, then each message
    /// queued meanwhile, highest priority first (`drain::pop_next_after_turn`).
    /// `None`: the first turn could not be opened, only the queue is played.
    async fn drive(self: Arc<Self>, first: Option<(nexus_claude::agent::EventStream, TurnInput)>) {
        let mut turn = first;
        loop {
            if let Some((stream, input)) = turn.take() {
                let (mut outcome, pending_tools) = self.play(stream, input).await;
                outcome.interrupted = self.interrupted.load(Ordering::SeqCst);
                self.cancel_pending_tools(outcome.interrupted, pending_tools)
                    .await;
                outcome.auto_continue_allowed =
                    self.auto_continue_after(outcome.hit_turn_limit).await;
                self.after_turn(&outcome).await;
            }
            let Some(next) = self.next_queued().await else {
                return;
            };
            // A `/compact` held until the turn ended runs now, between two turns.
            if next.kind == PendingMessageKind::User {
                if let Some(instructions) = self.manual_compaction(&next.content) {
                    self.compact_on_request(&next.content, instructions).await;
                    continue;
                }
            }
            match self
                .open_turn(next.kind, &next.content, &next.content)
                .await
            {
                Ok(opened) => turn = opened,
                Err(error) => {
                    let event = AgentEvent::Error { error };
                    // Mapped under the lock, emitted without it (the cockpit's snapshot
                    // reads the mapper).
                    let chat_events = self.mapper.lock().await.map(&event);
                    for chat_event in chat_events {
                        self.emit(chat_event).await;
                    }
                }
            }
        }
    }

    /// The next queued message, or — nothing left — the end of the run: the
    /// session stops streaming, decided under the queue lock.
    async fn next_queued(&self) -> Option<PendingMessage> {
        let (next, held_left) = {
            let mut queue = self.pending.lock().await;
            let interrupted = self.interrupted.load(Ordering::SeqCst);
            let (next, dropped) = super::drain::pop_next_after_turn(&mut queue, interrupted);
            if dropped > 0 {
                tracing::info!(session_id = %self.session_id, dropped, "Stop requested: dropped automated queued messages");
            }
            if let Some(next) = next.as_ref() {
                tracing::info!(
                    session_id = %self.session_id,
                    kind = ?next.kind,
                    held = next.held,
                    interrupted,
                    left = queue.len(),
                    "next queued message taken for a new turn"
                );
            }
            if next.is_none() {
                self.is_streaming.store(false, Ordering::SeqCst);
            }
            // A held message is leaving: the list clients show loses it now.
            let held_left = next
                .as_ref()
                .filter(|m| m.held)
                .map(|_| super::pending_queue::snapshot(&queue));
            (next, held_left)
        };
        if let Some(messages) = held_left {
            self.emit(ChatEvent::PendingQueue { messages }).await;
        }
        if next.is_none() {
            self.streaming_text.lock().await.clear();
            self.streaming_events.lock().await.clear();
            self.emit(ChatEvent::StreamingStatus {
                is_streaming: false,
            })
            .await;
        }
        next
    }

    /// Plays one turn to its terminal event, retrying a failure that showed nothing.
    /// Answers what the turn did, and the tool calls it left without a result
    /// (id, parent).
    async fn play(
        &self,
        stream: nexus_claude::agent::EventStream,
        input: TurnInput,
    ) -> (TurnOutcome, Vec<(String, Option<String>)>) {
        let mut attempt = 0u32;
        let mut outcome = TurnOutcome {
            context_window: self.capabilities.context_window.map(|w| w.value),
            ..TurnOutcome::default()
        };
        let mut pending_tools: Vec<(String, Option<String>)> = Vec::new();
        let mut tool_ids: Vec<String> = Vec::new();
        // What the turn cost, read on its `done` (`None`: no price, or no `done`).
        let mut turn_cost: Option<f64> = None;
        let retry_config = self.retry_config();
        let max_attempts = retry_config.max_attempts;
        // A provider that died mid-turn leaves its handle dead: once the turn is
        // over the handle is removed from the runtime, so the next message resumes
        // the session instead of re-failing on a closed process.
        let mut exited = false;
        // The stream of the attempt in flight. `None` means the attempt has still to
        // be sent: the first is already open, and a held turn whose probe the
        // provider refused goes back through `send_turn` at the top of the loop.
        let mut stream: Option<nexus_claude::agent::EventStream> = Some(stream);
        loop {
            let Some(mut current) = stream.take() else {
                // The probe of a held turn: one more call of the same input. A
                // refusal is one more failed call — the detector sees it, and the
                // hold decides whether the turn goes on (its own budget bounds the
                // loop) or ends here on the real cause.
                match self.session.send_turn(input.clone()).await {
                    Ok(next) => {
                        stream = Some(next);
                        continue;
                    }
                    Err(error) => {
                        exited |= matches!(&error, ProviderError::ProcessExited { .. });
                        self.record_outage_failure(&error).await;
                        let Some(wait) = self.hold_a_failed_turn() else {
                            let event = AgentEvent::Error { error };
                            let chat_events = self.mapper.lock().await.map(&event);
                            for chat_event in chat_events {
                                self.emit(chat_event).await;
                            }
                            break;
                        };
                        if self.pause_unless_stopped(wait).await {
                            let stopped = interrupted_done();
                            let chat_events = self.mapper.lock().await.map(&stopped);
                            for chat_event in chat_events {
                                self.emit(chat_event).await;
                            }
                            break;
                        }
                        continue;
                    }
                }
            };
            // Did the turn already show the user anything? A turn that showed
            // nothing is retried on any retryable failure; a turn that showed
            // text but no tool is retried on a transient mid-generation failure
            // only (rate limit, upstream timeout). A turn that already called a
            // tool is never replayed: the tool may have had a side effect.
            let mut shown = false;
            let mut retry: Option<ProviderError> = None;
            // Set when the attempt goes on past its retry budget, held: how long to
            // wait before the next probe.
            let mut held_for: Option<Duration> = None;
            // Whether this attempt saw a retryable failure at all: an attempt that
            // did, and is not retried, must not read as the instance answering (that
            // would reset the detector and hide the outage).
            let mut failed = false;
            while let Some(event) = current.next().await {
                // Terminal-ness is read on the original: a withheld terminal event
                // still ends the turn.
                let terminal = event.is_terminal();
                self.observe_quota(&event);
                // A quota refusal is only counted when the turn ENDS on it: a 429
                // retried then followed by a real error is not a refusal of the account.
                if terminal {
                    outcome.quota_refusal = is_quota_refusal(&event);
                }
                exited |= process_exited(&event);
                // The provider may have just named its session: stored before the
                // terminal event goes out, so a client that saw the turn end can
                // count on a resume.
                if terminal || matches!(event, AgentEvent::SessionStarted { .. }) {
                    self.sync_resume_token().await;
                }
                let now_ms = self
                    .quota
                    .as_ref()
                    .map_or_else(wall_clock_ms, |q| q.now_ms());
                let failure = if !shown {
                    retryable_failure(&event, now_ms)
                } else if outcome.tools.is_empty() {
                    mid_generation_retryable(&event, now_ms)
                } else {
                    None
                };
                // The outage detector sees this final call's outcome, in exactly one
                // place, and the hold follows from it: a failure whose ordinary retry
                // budget is spent goes on while the instance is down (a held turn)
                // instead of closing the turn on an error the user would read as
                // theirs.
                if let Some(error) = failure.as_ref() {
                    failed = true;
                    self.record_outage_failure(error).await;
                    if attempt >= max_attempts {
                        held_for = self.hold_a_failed_turn();
                    }
                }
                retry = failure.filter(|_| attempt < max_attempts || held_for.is_some());
                if retry.is_some() {
                    break;
                }
                shown |= shows_content(&event);
                outcome.hit_turn_limit |= matches!(
                    event,
                    AgentEvent::Done {
                        stop_reason: nexus_claude::agent::StopReason::MaxTurns,
                        ..
                    }
                );
                if let AgentEvent::Done {
                    cost,
                    is_error,
                    stop_reason,
                    subtype,
                    error,
                    ..
                } = &event
                {
                    turn_cost = cost.usd;
                    // `interrupted` too where the provider is known to keep the turn
                    // in its history ([`keeps_interrupted_turns`]). Only the
                    // provider's `done` comes here; the one made for a Stop during
                    // the pause below does not. A Claude Code CLI `done` with an
                    // error may already have written the prompt to its transcript:
                    // the context then comes once more with the next turn (bounded
                    // duplicate, not a loss).
                    outcome.answered = !is_error
                        && (*stop_reason != nexus_claude::agent::StopReason::Interrupted
                            || self.keeps_interrupted_turns);
                    // R0: capture the terminal result so `after_turn` can feed the
                    // routing bandit the turn's outcome (the same mapping
                    // `EventMapper` applies when it emits `ChatEvent::Result`).
                    outcome.stop_reason = Some(stop_reason_name(*stop_reason));
                    outcome.subtype = subtype
                        .clone()
                        .unwrap_or_else(|| default_subtype(*stop_reason, *is_error).to_string());
                    outcome.is_error = *is_error;
                    outcome.error_code = error.as_ref().map(|e| {
                        super::provider::errors::open_failure(e, None)
                            .code
                            .to_string()
                    });
                }
                let event = mask_agent_event(event);
                let chat_events = self.mapper.lock().await.map(&event);
                for chat_event in chat_events {
                    track_turn(&mut outcome, &mut tool_ids, &mut pending_tools, &chat_event);
                    self.emit(chat_event).await;
                }
                if terminal {
                    break;
                }
            }
            let Some(error) = retry else {
                // The attempt answered without a retryable failure: the instance is
                // up as far as this session knows. An error of its own that is not
                // retryable is not a health failure either (`Outcome`).
                if !failed {
                    self.record_outage_success().await;
                }
                break;
            };
            attempt += 1;
            let delay = match held_for {
                // A held turn probes at the hold's cadence, not on the retry curve
                // the ordinary budget follows.
                Some(probe) => probe.as_millis() as u64,
                None => retry_delay_ms(&error, attempt, &retry_config),
            };
            self.emit(ChatEvent::Retrying {
                attempt,
                max_attempts,
                delay_ms: delay,
                error_message: format!(
                    "Error: {}",
                    super::provider::errors::open_failure(&error, None).message
                ),
            })
            .await;
            // A Stop cuts the pause short, and ends the turn: no retry after it.
            if self
                .pause_unless_stopped(Duration::from_millis(delay))
                .await
            {
                // The turn ends as any stopped turn does: on its terminal event.
                let stopped = interrupted_done();
                let chat_events = self.mapper.lock().await.map(&stopped);
                for chat_event in chat_events {
                    self.emit(chat_event).await;
                }
                break;
            }
            // The next attempt is sent at the top of the loop, so a held turn whose
            // probe the provider refuses is simply probed again — the hold's own
            // budget is what bounds it.
        }
        // The record carries the turn's cost before the session is seen idle.
        self.add_turn_cost(turn_cost).await;
        if exited {
            self.detach_after_process_exit().await;
        }
        (outcome, pending_tools)
    }

    /// Waits `delay`, or less when the turn is stopped meanwhile: `true` when it was.
    async fn pause_unless_stopped(&self, delay: Duration) -> bool {
        let deadline = Instant::now() + delay;
        while Instant::now() < deadline {
            if self.interrupted.load(Ordering::SeqCst) {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            tokio::time::sleep(left.min(Duration::from_millis(50))).await;
        }
        self.interrupted.load(Ordering::SeqCst)
    }

    /// A stopped turn's tool calls left without a result are said cancelled
    /// (`tool_cancelled`, persisted) — the Claude Code engine's
    /// `PostStreamHandler::handle_interrupt_cleanup`.
    async fn cancel_pending_tools(
        &self,
        interrupted: bool,
        pending: Vec<(String, Option<String>)>,
    ) {
        if !interrupted {
            return;
        }
        for (id, parent_tool_use_id) in pending {
            self.emit(ChatEvent::ToolCancelled {
                id,
                parent_tool_use_id,
            })
            .await;
        }
    }

    /// The host's end of turn ([`TurnServices::after_turn`]), bounded: its events
    /// are emitted, its hints queued.
    async fn after_turn(&self, outcome: &TurnOutcome) {
        self.after_turn_bounded(outcome, AFTER_TURN_BACKSTOP).await;
    }

    /// [`Self::after_turn`] under an explicit budget (tests shorten it). When the
    /// host's end of turn does not finish in time it is skipped — recoverable, the
    /// turn already completed — so it is announced as a non-terminal system note,
    /// never as an `Error` that would read as a failed turn.
    async fn after_turn_bounded(&self, outcome: &TurnOutcome, backstop: Duration) {
        let Some(services) = &self.services else {
            return;
        };
        let Some(after) = super::post_stream::bounded(
            &self.session_id,
            "after_turn",
            backstop,
            services.after_turn(&self.session_id, outcome),
        )
        .await
        else {
            self.emit(ChatEvent::SystemHint {
                content: "The end-of-turn processing took too long and was skipped.".into(),
            })
            .await;
            return;
        };
        for event in after.events {
            self.emit(event).await;
        }
        if !after.hints.is_empty() {
            let mut queue = self.pending.lock().await;
            for hint in after.hints {
                queue.push_back(PendingMessage::system_hint(hint));
            }
        }
    }

    /// Sets how this session continues a turn that stopped on its turn limit.
    pub fn configure_auto_continue(&self, enabled: bool, max: u32) {
        self.auto_continue.store(enabled, Ordering::Relaxed);
        self.max_auto_continues.store(max, Ordering::Relaxed);
    }

    /// After a turn: when it stopped on its turn limit and auto-continue allows it,
    /// announce the continuation, wait (a Stop cancels it), and queue the
    /// "continue" hint the queue then plays — the Claude Code engine's
    /// `PostStreamHandler::handle_auto_continue`, with the same decision and message.
    async fn auto_continue_after(&self, hit_turn_limit: bool) -> bool {
        use super::post_stream::{auto_continue_allowed, AUTO_CONTINUE_DELAY_MS};
        let Some(services) = &self.services else {
            return false;
        };
        if !auto_continue_allowed(
            &self.session_id,
            hit_turn_limit,
            &self.auto_continue,
            self.interrupted.load(Ordering::SeqCst),
            &self.auto_continue_count,
            self.max_auto_continues.load(Ordering::Relaxed),
        ) {
            return false;
        }
        self.emit(ChatEvent::AutoContinue {
            session_id: self.session_id.clone(),
            delay_ms: AUTO_CONTINUE_DELAY_MS,
        })
        .await;
        tokio::time::sleep(Duration::from_millis(AUTO_CONTINUE_DELAY_MS)).await;
        if self.interrupted.load(Ordering::SeqCst) {
            tracing::info!(session_id = %self.session_id, "Auto-continue cancelled by interrupt");
            return true;
        }
        let hint = services.continuation(&self.session_id).await;
        self.pending
            .lock()
            .await
            .push_back(PendingMessage::system_hint(hint));
        true
    }

    /// The ids of the permission requests this session still waits on (asked, not yet
    /// answered by the user or by one of its grants).
    pub fn waiting_permission_ids(&self) -> Vec<String> {
        self.asked_calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Whether nobody needs to answer the permission request `request_id`: the backend
    /// answers it itself (a grant or a lasting rule covers it — known as soon as the
    /// request is broadcast, before its automatic answer went out), or this handle saw it
    /// asked and its answer was taken (the decision may not be stored yet). `false` for a
    /// request this session still waits on that no grant answers (one whose automatic
    /// answer failed included) AND for an id this handle never saw asked — one of a
    /// session resumed under the same id (review of #727, finding 3): nobody here will
    /// answer it.
    ///
    /// The "answered" half keeps a parent reading the request before the decision that
    /// follows it seeing it answered, never asked (lot review of #721).
    pub fn answers_itself(&self, request_id: &str) -> bool {
        self.self_answered
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains(request_id)
            || self.was_answered(request_id)
    }

    /// Whether this handle saw `request_id` asked and the provider took its answer.
    pub fn was_answered(&self, request_id: &str) -> bool {
        self.answered
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains(request_id)
    }

    /// Marks `request_id` answered (the provider took an answer to it).
    pub(crate) fn mark_answered(&self, request_id: &str) {
        self.answered
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(request_id);
    }

    /// How many requests this session holds as answered by itself: zero once their
    /// automatic answers went through.
    #[cfg(test)]
    pub(crate) fn self_answered_len(&self) -> usize {
        self.self_answered
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }

    /// Answers a permission request for this call only.
    pub async fn answer_permission(&self, request_id: &str, allow: bool) -> Result<()> {
        self.answer_permission_scoped(request_id, allow, PermissionAnswerScope::Once)
            .await
            .map_err(|e| match e {
                PermissionDeliveryError::Failed(e) => e,
                other => anyhow!(other.to_string()),
            })
    }

    /// Keeps a grant as if the user had answered `session` (tests).
    #[cfg(test)]
    pub(crate) fn seed_session_grant(&self, grant: super::session_grants::SessionGrant) {
        self.session_grants
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .add(grant);
    }

    /// Whether this session's shell may have left its project root
    /// (`SessionGrants::shell_moved`).
    pub fn shell_moved(&self) -> bool {
        self.session_grants
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .shell_moved()
    }

    /// The capabilities this session offers: its provider's, `always` being the BACKEND's
    /// (P11b) whatever the provider declares: offered only when this session keeps
    /// lasting rules (a store, an owner, a project) for an engine that can
    /// (`SessionGrants::keeps_lasting`), and the provider asks for the session at all.
    /// What `system_init` carries.
    pub fn offered_capabilities(&self) -> Capabilities {
        let mut caps = self.capabilities.clone();
        caps.permission_scopes
            .retain(|scope| *scope != PermissionScope::Always);
        let keeps = self
            .session_grants
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keeps_lasting(self.asker);
        if keeps && caps.permission_scopes.contains(&PermissionScope::Session) {
            caps.permission_scopes.push(PermissionScope::Always);
        }
        caps
    }

    /// Answers a permission request; an approval lasts as long as `scope` says. The
    /// provider is ALWAYS answered `once` (never a provider-side session or lasting
    /// rule, whose matching is looser than what the user approved):
    ///
    /// - `session`: the grant of this call (`session_grants::grant_for`: the identical
    ///   call, or any call of a read-only tool of nexus) is kept in this session AFTER the
    ///   provider took the answer; the backend answers the later requests it covers
    ///   ([`Self::emit`]). A call that cannot be granted (a request that does not carry the
    ///   whole call, a tool or a program off the allowlist) is refused.
    /// - `always`: the lasting grant of this call (`session_grants::lasting_grant_for`:
    ///   the identical call, stricter than `session`) is kept for the session's person and
    ///   project (`chat::lasting_rules`) AFTER the provider took the answer; the backend
    ///   answers the later requests it covers, in any of their sessions. Refused when the
    ///   session keeps no lasting rule or the call cannot be approved for good. When the
    ///   rule cannot be written, the call stays granted for this session and the decision
    ///   says `session` (never an `always` that was not kept).
    ///
    /// A refusal ([`PermissionDeliveryError::ScopeUnsupported`]) answers nothing: the
    /// request stays waiting.
    pub async fn answer_permission_scoped(
        &self,
        request_id: &str,
        allow: bool,
        scope: PermissionAnswerScope,
    ) -> std::result::Result<(), PermissionDeliveryError> {
        // The call this request asks for: what a grant is derived from.
        let asked = self
            .asked_calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(request_id)
            .cloned();
        // Is this ask about a DESTINATION (N23b)? The gate of the session
        // (`nexus_tools::ToolAccessHooks`) raised a `HookVerdict::RequireConsent` about what
        // this call would reach, and names the subject with the very function read here: the
        // answer is then about an ORIGIN, and its durable place is the project's network-tools
        // consent — the same document the settings route writes — never a lasting rule about a
        // tool call, which the gate does not read. A search names no page of its own: what it
        // would reach is the engines of the session, and an `always` writes them all.
        let (destination, destinations) = match asked
            .as_ref()
            .and_then(|call| self.consent_about(&call.tool, call.canonical.as_deref(), &call.input))
        {
            Some((subject, destinations)) => (Some(subject), destinations),
            None => (None, Vec::new()),
        };
        // `always` kept as a lasting rule (`false`: as a session grant, the shell having
        // possibly left the project's root; review of #714, A).
        let mut for_good = false;
        let grant = match (allow, scope) {
            (false, _) | (true, PermissionAnswerScope::Once) => None,
            // An origin, for this session: the harness keeps `tool\u{0}origin` itself when it is
            // answered `session` (nexus `providers::native::session::answer_permission`), and its
            // matching is exactly the subject the person was shown. No grant of this backend: an
            // exact-call grant would cover one URL of the origin, not the origin.
            (true, PermissionAnswerScope::Session) if destination.is_some() => {
                if !self
                    .offered_capabilities()
                    .permission_scopes
                    .contains(&PermissionScope::Session)
                {
                    return Err(PermissionDeliveryError::ScopeUnsupported(scope));
                }
                None
            }
            // An origin, for good: the project's consent, by the person who answered, in the one
            // store `PUT /api/projects/{slug}/network-tools/origins` writes. Written BEFORE the
            // provider is answered, so an `always` that could not be kept is never reported as
            // kept.
            (true, PermissionAnswerScope::Always) if destination.is_some() => {
                if !self
                    .offered_capabilities()
                    .permission_scopes
                    .contains(&PermissionScope::Always)
                {
                    return Err(PermissionDeliveryError::ScopeUnsupported(scope));
                }
                self.keep_origin_consents(&destinations).await?;
                for_good = true;
                None
            }
            (true, PermissionAnswerScope::Session) => {
                if !self
                    .offered_capabilities()
                    .permission_scopes
                    .contains(&PermissionScope::Session)
                {
                    return Err(PermissionDeliveryError::ScopeUnsupported(scope));
                }
                let grant = asked.as_ref().and_then(|call| {
                    self.session_grants
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .grant_for(call)
                });
                Some(grant.ok_or(PermissionDeliveryError::ScopeUnsupported(scope))?)
            }
            (true, PermissionAnswerScope::Always) => {
                if !self
                    .offered_capabilities()
                    .permission_scopes
                    .contains(&PermissionScope::Always)
                {
                    return Err(PermissionDeliveryError::ScopeUnsupported(scope));
                }
                let kept = asked.as_ref().and_then(|call| {
                    self.session_grants
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .always_grant(call)
                });
                let (grant, lasting) =
                    kept.ok_or(PermissionDeliveryError::ScopeUnsupported(scope))?;
                for_good = lasting;
                Some(grant)
            }
        };
        let decision = if !allow {
            PermissionDecision::deny()
        } else if destination.is_some() && scope == PermissionAnswerScope::Session {
            // `session` on a destination is remembered by the harness against the SUBJECT (N23b):
            // answering `once` would make the answer cover this call only, and the next page of
            // the same origin would be asked again.
            PermissionDecision::Allow {
                scope: PermissionScope::Session,
                updated_input: None,
            }
        } else {
            // Everything else, an `always` about an origin included: the durable part is now in
            // the project's consent document, and this session approves no more than the call.
            PermissionDecision::allow_once()
        };
        // On the tool clock BEFORE the provider has it: a fast tool's result cannot
        // overtake it.
        // A second answer (double click, two tabs) the provider refuses takes back
        // only its own mark, never the first answer's; and it waits until the first
        // was handed over, so the provider gets them in the order the clock kept.
        let answering = self.tool_clock.answering(request_id).await;
        let mark = self
            .tool_clock
            .decided(request_id, allow, chrono::Utc::now());
        // The grant covers, from now on, the requests that arrive while the provider takes
        // the answer: a CLI asks the identical call again as soon as it has it, before
        // `answer_permission` returns here (measured on CI, review of #726). Kept for good
        // once delivered, dropped if the provider refuses the answer.
        // Only while the request is still waiting: a second answer (another tab) that got
        // here after the first one took it covers nothing, even for the time the provider
        // takes to refuse it.
        let still_asked = self
            .asked_calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains_key(request_id);
        // Rolls the pending grant back if this future is dropped before it is settled (the
        // HTTP or WebSocket request that carries the answer is cancelled while the provider
        // takes it): never a grant left pending for the rest of the session.
        let mut pending = PendingGrant {
            grants: &self.session_grants,
            grant: grant.clone().filter(|_| still_asked),
            lasting: scope == PermissionAnswerScope::Always && for_good,
        };
        pending.begin();
        if let Err(e) = self.session.answer_permission(request_id, decision).await {
            self.tool_clock.undecided(request_id, mark);
            drop(pending);
            return Err(PermissionDeliveryError::Failed(anyhow::Error::new(e)));
        }
        // Taken: answered from now on, though its decision is stored only below.
        if asked.is_some() {
            self.mark_answered(request_id);
        }
        drop(answering);
        // Kept only now that the provider took the answer.
        let mut kept = scope;
        match (&grant, &asked, scope) {
            (Some(grant), Some(call), PermissionAnswerScope::Always) if for_good => {
                let place = {
                    let grants = self
                        .session_grants
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    // The shell may have moved since: then nothing is written.
                    (!grants.cwd_bound(call))
                        .then(|| grants.lasting_scope())
                        .flatten()
                };
                let written =
                    super::lasting_rules::keep(place, call, grant, &self.session_id).await;
                pending.disarm();
                self.session_grants
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .end_lasting(grant, written);
                if !written {
                    kept = PermissionAnswerScope::Session;
                }
            }
            (Some(grant), _, _) => {
                if scope == PermissionAnswerScope::Always {
                    kept = PermissionAnswerScope::Session;
                }
                pending.disarm();
                self.session_grants
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .end_pending(grant, true)
            }
            (None, _, _) => {}
        }
        pending.disarm();
        self.asked_calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(request_id);
        self.emit(ChatEvent::PermissionDecision {
            id: request_id.to_string(),
            allow,
            scope: kept.lasting(allow),
            rule: grant
                .as_ref()
                .map(super::session_grants::SessionGrant::describe)
                // An answer about a destination covers an ORIGIN, not a call (N23b): the client
                // says so, and never reads it as the identical call (P11b).
                .or_else(|| {
                    destination
                        .as_ref()
                        .map(|subject| format!("origin {subject}"))
                }),
        })
        .await;
        Ok(())
    }

    /// Records the person's `always` about a DESTINATION (N23b) where the settings route records
    /// it: the project's network-tools consent, one document per origin, written by
    /// [`nexus_tools::record_origin_consent`](super::provider::nexus_tools::record_origin_consent).
    /// Same store, same key, same writer as `PUT /api/projects/{slug}/network-tools/origins` —
    /// there is no second store, so a consent given in the chat is listed, revocable and honoured
    /// exactly like one given in Settings.
    ///
    /// Every form the gate judges is written ([`destinations_of`]), so a consent just given does
    /// not ask again for the `http://` twin of the `https://` endpoint it was about.
    ///
    /// Typed `ScopeUnsupported` when the session keeps no lasting rule — no owner, no project, or
    /// a session that does not run in its project's root: there is nobody whose `always` this is,
    /// and no name to write next to the consent.
    ///
    /// [`destinations_of`]: super::provider::nexus_tools::destinations_of
    async fn keep_origin_consents(
        &self,
        destinations: &[String],
    ) -> std::result::Result<(), PermissionDeliveryError> {
        let place = {
            let grants = self
                .session_grants
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            grants
                .keeps_lasting(self.asker)
                .then(|| grants.lasting_scope())
                .flatten()
        };
        let Some(place) = place else {
            return Err(PermissionDeliveryError::ScopeUnsupported(
                PermissionAnswerScope::Always,
            ));
        };
        for origin in destinations {
            super::provider::nexus_tools::record_origin_consent(
                self.graph.as_ref(),
                &place.project,
                origin,
                &place.owner,
            )
            .await
            .map_err(|e| PermissionDeliveryError::Failed(anyhow::Error::new(e)))?;
        }
        Ok(())
    }

    /// Interrupts the turn and the tools it runs.
    pub async fn interrupt(&self) -> Result<()> {
        self.interrupt_scoped(InterruptScope::TurnAndTools)
            .await
            .map(|_| ())
    }

    /// Stops the turn (a Stop of the user): what the queue holds of automated
    /// work (hints, a pending auto-continue) is dropped when the turn ends, as on
    /// the Claude Code engine; the messages the user typed still run.
    pub async fn interrupt_scoped(
        &self,
        scope: InterruptScope,
    ) -> Result<nexus_claude::agent::InterruptOutcome> {
        self.interrupted.store(true, Ordering::SeqCst);
        self.session
            .interrupt(scope)
            .await
            .map_err(anyhow::Error::new)
    }

    /// Stops the tools the running turn executes and keeps the turn: the provider
    /// answers each cut call with a cancelled `tool_result` and the model goes on
    /// from there — the `cancel_tools` of the Claude Code engine
    /// (`ChatManager::cancel_running_tools`), whose per-session cap applies here
    /// too: past `cancel_tools_cap` calls within `cancel_tools_window` nothing
    /// reaches the provider and the result says `capped`.
    ///
    /// Announces `tools_cancelled` to every client of the session (stored,
    /// published to the other instances like any event of the session);
    /// `killed_count` is the number of tools the provider stopped.
    pub async fn cancel_tools(&self) -> Result<CancelToolsResult> {
        let allowed = ChatManager::check_and_record_cancel_cap(
            &self.cancel_tools_history,
            self.cancel_tools_cap,
            self.cancel_tools_window,
        )
        .await;
        if !allowed {
            tracing::warn!(
                session_id = %self.session_id,
                cap = self.cancel_tools_cap,
                window_secs = self.cancel_tools_window.as_secs(),
                "cancel_tools: rate cap hit, refusing"
            );
            return Ok(CancelToolsResult {
                cli_pid: None,
                killed_pids: Vec::new(),
                capped: true,
            });
        }
        // Bounded like a cancel relayed from another instance (review N6): a
        // provider that hangs is `owner_timeout` (not retryable for cancel_tools).
        let call = tokio::time::timeout(
            super::cancel_relay::OWNER_CANCEL_BOUND,
            self.session.cancel_tools(CancelScope::All),
        )
        .await
        .map_err(|_| cancel_timeout(super::cancel_relay::CancelKind::Tools))?;
        let outcome = match call {
            Ok(outcome) => outcome,
            Err(error) => return Err(self.cancel_refused(error).await),
        };
        let diagnostic = outcome.diagnostic.unwrap_or_default();
        tracing::info!(
            session_id = %self.session_id,
            tools_cancelled = outcome.tools_cancelled,
            "cancel_tools: the running tools were stopped (turn preserved)"
        );
        self.emit(ChatEvent::ToolsCancelled {
            cli_pid: diagnostic.pid,
            killed_count: outcome.tools_cancelled as usize,
            requested_by: "user".to_string(),
        })
        .await;
        Ok(CancelToolsResult {
            cli_pid: diagnostic.pid,
            killed_pids: diagnostic.killed_pids,
            capped: false,
        })
    }

    /// Stops ONE background task (a `Bash run_in_background`, a `Monitor`) by its
    /// wire id — the `tool_use` that started it, what `active_tasks_update` and the
    /// Claude Code engine call it — through the provider
    /// (`AgentSession::cancel_tools(task { id })`, the provider's own id found in
    /// the last snapshot). The turn, if one runs, goes on. The per-session cap of
    /// the Claude Code engine applies (`CANCEL_TASK_CAP` per window): past it,
    /// nothing reaches the provider and the result says `capped`.
    ///
    /// The provider follows with a `background_tasks` snapshot (the task `killed`),
    /// which goes out as `active_tasks_update` without it. A task that already
    /// ended, or that no snapshot ever named, is the idempotent no-op of the Claude
    /// Code engine (`killed_pids` empty). A provider that cannot stop a task
    /// (`Unsupported`: a Claude Code over SSH has no `tool_cancel`, a session
    /// without background tasks) is refused: typed on the wire
    /// (`error { code: cancel_refused }`) and in the `Err` (the `ProviderError` in
    /// the chain), never a success.
    pub async fn cancel_task(&self, task_id: &str) -> Result<CancelTaskResult> {
        let allowed = ChatManager::check_and_record_cancel_cap(
            &self.cancel_task_history,
            CANCEL_TASK_CAP,
            Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
        )
        .await;
        let result = |killed_pids: Vec<u32>, capped: bool| CancelTaskResult {
            task_id: task_id.to_string(),
            killed_pids,
            capped,
        };
        if !allowed {
            tracing::warn!(session_id = %self.session_id, task_id, "cancel_task: rate cap hit, refusing");
            return Ok(result(Vec::new(), true));
        }
        let known = self.mapper.lock().await.provider_task(task_id);
        let provider_id = match &known {
            // Over already: nothing to stop, nothing changes (as on Claude Code).
            Some((_, false)) => return Ok(result(Vec::new(), false)),
            Some((id, true)) => id.clone(),
            // No snapshot named it (yet): the provider decides.
            None => task_id.to_string(),
        };
        let call = tokio::time::timeout(
            super::cancel_relay::OWNER_CANCEL_BOUND,
            self.session.cancel_tools(CancelScope::Task {
                id: provider_id.clone(),
            }),
        )
        .await
        .map_err(|_| cancel_timeout(super::cancel_relay::CancelKind::Task))?;
        match call {
            Ok(outcome) => {
                let diagnostic = outcome.diagnostic.unwrap_or_default();
                tracing::info!(
                    session_id = %self.session_id,
                    task_id,
                    provider_task_id = %provider_id,
                    tasks_cancelled = outcome.tools_cancelled,
                    killed = ?diagnostic.killed_pids,
                    "cancel_task: background task stopped"
                );
                Ok(result(diagnostic.killed_pids, false))
            }
            // An id the provider does not know (or no longer runs): the idempotent
            // no-op of the Claude Code engine.
            Err(ProviderError::InvalidRequest { detail }) => {
                tracing::debug!(
                    session_id = %self.session_id,
                    task_id,
                    %detail,
                    known_in_snapshot = known.is_some(),
                    "cancel_task: the provider does not know this task (unknown id, already over, or a Stop that came before the first background_tasks snapshot named it): idempotent no-op"
                );
                Ok(result(Vec::new(), false))
            }
            Err(error) => Err(self.cancel_refused(error).await),
        }
    }

    /// The background tasks still running, as the last `active_tasks_update` said.
    pub async fn active_background_tasks(&self) -> Vec<BackgroundTaskInfo> {
        self.mapper.lock().await.active_tasks()
    }

    /// A cancel the provider refused: an `Unsupported` is announced to every
    /// client of the session (`error { code: cancel_refused, reason: <capability> }`)
    /// — the click had no effect and the user must know; the error is returned with
    /// the `ProviderError` in its chain, which the HTTP layer answers typed
    /// (`AppError::from_open_error`: 422 `unsupported`).
    async fn cancel_refused(&self, error: ProviderError) -> anyhow::Error {
        if let ProviderError::Unsupported { capability } = &error {
            tracing::warn!(session_id = %self.session_id, %capability, "cancel refused: the provider cannot do it");
            self.emit(cancel_refused_event(capability)).await;
        }
        anyhow::Error::new(error)
    }

    /// Changes the model live.
    pub async fn set_model(&self, model: &str) -> Result<()> {
        self.session
            .set_model(model)
            .await
            .map_err(anyhow::Error::new)?;
        self.emit(ChatEvent::ModelChanged {
            model: model.to_string(),
        })
        .await;
        Ok(())
    }

    /// Changes the policy mode live.
    pub async fn set_policy_mode(&self, mode: PolicyMode, native: Option<&str>) -> Result<()> {
        self.session
            .set_policy_mode(mode, native)
            .await
            .map_err(anyhow::Error::new)?;
        self.emit(ChatEvent::PermissionModeChanged {
            mode: native
                .map(str::to_string)
                .unwrap_or_else(|| super::provider::policy::legacy_name(mode).to_string()),
            policy_mode: Some(super::provider::policy::neutral_name(mode).to_string()),
        })
        .await;
        Ok(())
    }
}

/// A cancel the provider did not answer within `OWNER_CANCEL_BOUND`.
fn cancel_timeout(kind: super::cancel_relay::CancelKind) -> anyhow::Error {
    anyhow::Error::new(super::cancel_relay::CancelRelayError::OwnerTimeout { kind })
}

/// The wire form of a cancel (`cancel_tools`, `cancel_task`) the provider refused
/// for lack of `capability` (`tool_cancel`: a Claude Code over SSH or off Unix;
/// `background_tasks`: a session that runs none).
pub(crate) fn cancel_refused_event(capability: &str) -> ChatEvent {
    ChatEvent::Error {
        message: format!(
            "Error: This session cannot stop running tools: its provider does not support \
             `{capability}`. Nothing was stopped."
        ),
        parent_tool_use_id: None,
        code: Some("cancel_refused".to_string()),
        reason: Some(capability.to_string()),
        index: None,
        request_id: None,
    }
}

/// The live sessions of the agent path.
pub struct AgentRuntime {
    /// The live sessions, shared with each handle (as a `Weak`) so a handle can
    /// remove itself once its provider process is gone.
    sessions: Arc<RwLock<HashMap<String, Arc<AgentSessionHandle>>>>,
    graph: Arc<dyn GraphStore>,
    /// The third-party MCP tools the operator declared read-only
    /// (`ChatConfig::read_only_mcp_tools`): what each session's grants may cover
    /// (`session_grants::SessionGrants::declaring`).
    read_only_mcp_tools: Vec<String>,
    /// How an oversized tool result is spilled out of the transcript
    /// (`super::spill`), copied onto every session this runtime adopts.
    spill: SpillConfig,
    /// The quota registry handed to every session this runtime adopts (QUOTA).
    quota: std::sync::RwLock<Option<Arc<super::provider::quota::QuotaRegistry>>>,
}

impl AgentRuntime {
    /// A runtime persisting its events in `graph`.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            graph,
            read_only_mcp_tools: Vec::new(),
            spill: SpillConfig::from_env(),
            quota: std::sync::RwLock::new(None),
        }
    }

    /// Sessions adopted from now on report the limits their provider states (and the refusals
    /// that name a reset) to `registry`.
    pub fn set_quota(&self, registry: Arc<super::provider::quota::QuotaRegistry>) {
        *self.quota.write().unwrap_or_else(|e| e.into_inner()) = Some(registry);
    }

    /// This runtime, its sessions declaring these third-party MCP tools read-only (exact
    /// `mcp__<server>__<tool>` names): the only ones a session grant may cover.
    pub fn declaring_read_only_mcp_tools(mut self, tools: Vec<String>) -> Self {
        self.read_only_mcp_tools = tools;
        self
    }

    /// This runtime, its sessions spilling oversized tool results per `spill`
    /// (the default reads the operator's environment).
    pub fn spilling(mut self, spill: SpillConfig) -> Self {
        self.spill = spill;
        self
    }

    /// The live session, if this runtime owns it.
    pub async fn get(&self, session_id: &str) -> Option<Arc<AgentSessionHandle>> {
        self.sessions.read().await.get(session_id).cloned()
    }

    /// [`Self::get`] without waiting, for a caller that cannot `await` (a synchronous
    /// predicate): `None` while the map is being written at this very instant, else the
    /// live session (or its absence).
    pub fn try_get(&self, session_id: &str) -> Option<Option<Arc<AgentSessionHandle>>> {
        self.sessions
            .try_read()
            .ok()
            .map(|sessions| sessions.get(session_id).cloned())
    }

    /// Whether this runtime owns a live session of that id.
    pub async fn owns(&self, session_id: &str) -> bool {
        self.sessions.read().await.contains_key(session_id)
    }

    /// Number of live sessions.
    pub async fn len(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Whether no session is live.
    pub async fn is_empty(&self) -> bool {
        self.sessions.read().await.is_empty()
    }

    /// Every live session.
    pub async fn handles(&self) -> Vec<Arc<AgentSessionHandle>> {
        self.sessions.read().await.values().cloned().collect()
    }

    /// Registers a session just opened by a provider and starts its
    /// out-of-turn pump. `first_seq` is the next event number to persist.
    #[allow(clippy::too_many_arguments)]
    pub async fn adopt(
        &self,
        session_id: &str,
        provider_id: &str,
        session: Arc<dyn AgentSession>,
        first_seq: i64,
        provider_kind: &str,
        tool_policy: serde_json::Value,
        services: Option<Arc<dyn TurnServices>>,
    ) -> Arc<AgentSessionHandle> {
        self.adopt_with(
            session_id,
            provider_id,
            session,
            first_seq,
            provider_kind,
            tool_policy,
            services,
            Vec::new(),
        )
        .await
    }

    /// [`Self::adopt`] for a session the HOST knows lacks something its provider's
    /// capabilities cannot say: `extra_degraded` is added to what the session does not
    /// do (for instance [`NEXUS_TOOLS_FEATURE`], a native session whose `nexus-tools`
    /// could not be attached).
    #[allow(clippy::too_many_arguments)]
    pub async fn adopt_with(
        &self,
        session_id: &str,
        provider_id: &str,
        session: Arc<dyn AgentSession>,
        first_seq: i64,
        provider_kind: &str,
        tool_policy: serde_json::Value,
        services: Option<Arc<dyn TurnServices>>,
        extra_degraded: Vec<String>,
    ) -> Arc<AgentSessionHandle> {
        self.adopt_declaring(
            session_id,
            provider_id,
            session,
            first_seq,
            provider_kind,
            tool_policy,
            services,
            extra_degraded,
            super::session_grants::SessionGrants::declaring(self.read_only_mcp_tools.clone()),
        )
        .await
    }

    /// [`Self::adopt_with`] for a session whose grants start as `grants`: its declared
    /// read-only MCP tools (the runtime's, minus what the session's project redefines:
    /// `session_grants::declarations_for_project`) and the lasting rules it keeps.
    #[allow(clippy::too_many_arguments)]
    pub async fn adopt_declaring(
        &self,
        session_id: &str,
        provider_id: &str,
        session: Arc<dyn AgentSession>,
        first_seq: i64,
        provider_kind: &str,
        tool_policy: serde_json::Value,
        services: Option<Arc<dyn TurnServices>>,
        extra_degraded: Vec<String>,
        grants: super::session_grants::SessionGrants,
    ) -> Arc<AgentSessionHandle> {
        match self
            .adopt_admitting(
                session_id,
                provider_id,
                session,
                first_seq,
                provider_kind,
                tool_policy,
                services,
                extra_degraded,
                grants,
                || Ok::<(), std::convert::Infallible>(()),
            )
            .await
        {
            Ok(handle) => handle,
            Err(never) => match never {},
        }
    }

    /// [`Self::adopt_declaring`] for a session `admit` may still refuse: it is called INSIDE
    /// the critical section that inserts the session in the live map (B36-RACE: a delegated
    /// child whose parent a cascade closed, `super::cancelled_parents`). Refused, the session
    /// is never inserted, its out-of-turn pump never started and its provider session
    /// closed; the refusal is returned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn adopt_admitting<E>(
        &self,
        session_id: &str,
        provider_id: &str,
        session: Arc<dyn AgentSession>,
        first_seq: i64,
        provider_kind: &str,
        tool_policy: serde_json::Value,
        services: Option<Arc<dyn TurnServices>>,
        extra_degraded: Vec<String>,
        grants: super::session_grants::SessionGrants,
        admit: impl FnOnce() -> Result<(), E>,
    ) -> Result<Arc<AgentSessionHandle>, E> {
        let mut degraded = degraded_features(session.capabilities());
        for feature in extra_degraded {
            if !degraded.contains(&feature) {
                degraded.push(feature);
            }
        }
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);
        let handle = Arc::new(AgentSessionHandle {
            session_id: session_id.to_string(),
            provider_id: provider_id.to_string(),
            capabilities: session.capabilities().clone(),
            session: Arc::clone(&session),
            events_tx,
            is_streaming: AtomicBool::new(false),
            streaming_text: Mutex::new(String::new()),
            streaming_events: Mutex::new(Vec::new()),
            provider: serde_json::json!({ "id": provider_id, "kind": provider_kind }),
            tool_policy,
            degraded,
            next_seq: AtomicI64::new(first_seq),
            mapper: Mutex::new(EventMapper::new()),
            graph: Arc::clone(&self.graph),
            uuid: Uuid::parse_str(session_id).ok(),
            services,
            pending: Mutex::new(std::collections::VecDeque::new()),
            interrupted: AtomicBool::new(false),
            closed: tokio_util::sync::CancellationToken::new(),
            auto_continue: AtomicBool::new(false),
            auto_continue_count: std::sync::atomic::AtomicU32::new(0),
            max_auto_continues: std::sync::atomic::AtomicU32::new(0),
            // What the opener persisted with the snapshot (`ChatManager::finish_agent_open`).
            persisted_token: Mutex::new(session.resume_token().map(|t| t.to_wire())),
            cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
            cancel_tools_cap: CANCEL_TOOLS_CAP,
            tool_clock: super::tool_clock::ToolClock::for_session(session_id),
            cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
            session_grants: std::sync::Mutex::new(grants),
            asked_calls: std::sync::Mutex::new(HashMap::new()),
            self_answered: std::sync::Mutex::new(std::collections::HashSet::new()),
            answered: std::sync::Mutex::new(AnsweredIds::default()),
            asker: super::session_grants::Asker::from_provider_kind(provider_kind),
            cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
            cost_figure: super::session_record::CostFigure::of_kind(provider_kind),
            keeps_interrupted_turns: keeps_interrupted_turns(provider_kind),
            record: Mutex::new(()),
            opening_counted: AtomicBool::new(false),
            retry: std::sync::RwLock::new(RetryConfig::default()),
            outage: std::sync::Mutex::new(TurnHold::new(provider_id)),
            spill: self.spill,
            quota: self.quota.read().unwrap_or_else(|e| e.into_inner()).clone(),
            last_activity: std::sync::Mutex::new(Instant::now()),
            registry: Arc::downgrade(&self.sessions),
            provider_dead: AtomicBool::new(false),
        });
        let replaced = {
            let mut sessions = self.sessions.write().await;
            if let Err(refused) = admit() {
                drop(sessions);
                handle.closed.cancel();
                tokio::spawn(async move {
                    let _ = tokio::time::timeout(Duration::from_secs(5), session.close()).await;
                });
                return Err(refused);
            }
            sessions.insert(session_id.to_string(), Arc::clone(&handle))
        };
        // Started once the session is live (a refused one never runs it).
        if let Some(oob) = session.out_of_band() {
            let pump = Arc::clone(&handle);
            tokio::spawn(async move { super::oob_listener::pump_out_of_band(pump, oob).await });
        }
        // Two resumes raced for one session: the handle replaced is ended, or its
        // listeners keep answering the session's NATS subjects next to the new ones
        // (a message from another instance played twice) and its provider session
        // stays open.
        if let Some(old) = replaced {
            tracing::warn!(
                session_id,
                "a live agent session was adopted again: the previous handle is closed"
            );
            old.closed.cancel();
            tokio::spawn(async move {
                let _ = tokio::time::timeout(Duration::from_secs(5), old.session.close()).await;
            });
        }
        Ok(handle)
    }

    /// Closes a session: tells the provider, announces `session_closed` (A45),
    /// forgets it.
    pub async fn close(&self, session_id: &str) -> Result<()> {
        self.close_inner(session_id, true, "closed").await
    }

    /// [`Self::close`] with the reason the clients are told (`process_died` for a provider whose
    /// process is gone).
    pub async fn close_with_reason(&self, session_id: &str, reason: &str) -> Result<()> {
        self.close_inner(session_id, true, reason).await
    }

    /// [`Self::close`] without announcing `session_closed`: an in-place provider switch, whose
    /// same thread must see only `provider_switched`, never `session_closed`.
    pub async fn close_for_switch(&self, session_id: &str) -> Result<()> {
        self.close_inner(session_id, false, "closed").await
    }

    async fn close_inner(&self, session_id: &str, announce: bool, reason: &str) -> Result<()> {
        // The engines this session was launched with are gone with it: a later question about a
        // search must not name an origin nothing runs (`nexus_tools::remember_search_origins`).
        super::provider::nexus_tools::forget_search_origins(session_id);
        let handle = self
            .sessions
            .write()
            .await
            .remove(session_id)
            .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
        handle.closed.cancel();
        // Last chance for a token the provider named outside any turn end.
        handle.sync_resume_token().await;
        let _ = tokio::time::timeout(Duration::from_secs(5), handle.session.close()).await;
        if announce {
            handle
                .emit(ChatEvent::SessionClosed {
                    session_id: session_id.to_string(),
                    reason: Some(reason.to_string()),
                })
                .await;
        }
        Ok(())
    }
}

/// Maps a batch of out-of-turn events to chat events and emits them: masked,
/// mirrored (a sub-agent still running in the background is seen first, in the
/// form a turn shows it), persisted and broadcast. Then the resume token is
/// synced — an out-of-turn `init` (the CLI repeats it) may carry a new session
/// id. The out-of-turn pump (`super::oob_listener::pump_out_of_band`) calls
/// this once per batch, so each event is emitted exactly once.
pub(crate) async fn emit_out_of_band(handle: &AgentSessionHandle, events: Vec<AgentEvent>) {
    if events.is_empty() {
        return;
    }
    for event in &events {
        handle.observe_quota(event);
    }
    if events.iter().any(process_exited) {
        handle.provider_dead.store(true, Ordering::SeqCst);
    }
    let events: Vec<AgentEvent> = events.into_iter().map(mask_agent_event).collect();
    if let Some(services) = &handle.services {
        services.observe_out_of_turn(&handle.session_id, &events);
    }
    let chat_events = {
        let mut mapper = handle.mapper.lock().await;
        out_of_band_to_chat_events(&events, &mut mapper, chrono::Utc::now())
    };
    for event in chat_events {
        handle.emit(event).await;
    }
    handle.sync_resume_token().await;
}

/// A grant made pending while its answer is handed to the provider
/// (`AgentSessionHandle::answer_permission_scoped`): dropped before it is settled — the
/// provider refused the answer, or the future carrying it was cancelled — it is rolled
/// back, so no grant stays pending for the rest of the session.
struct PendingGrant<'a> {
    grants: &'a std::sync::Mutex<super::session_grants::SessionGrants>,
    /// `None` once settled (or when there is nothing to grant).
    grant: Option<super::session_grants::SessionGrant>,
    /// A lasting grant (its rule is written next) rather than a session one.
    lasting: bool,
}

impl PendingGrant<'_> {
    fn begin(&self) {
        if let Some(grant) = &self.grant {
            let mut grants = self.grants.lock().unwrap_or_else(|p| p.into_inner());
            if self.lasting {
                grants.begin_lasting(grant.clone());
            } else {
                grants.begin_pending(grant.clone());
            }
        }
    }

    /// Settled by the caller: nothing to roll back.
    fn disarm(&mut self) {
        self.grant = None;
    }
}

impl Drop for PendingGrant<'_> {
    fn drop(&mut self) {
        if let Some(grant) = self.grant.take() {
            let mut grants = self.grants.lock().unwrap_or_else(|p| p.into_inner());
            if self.lasting {
                grants.abort_lasting(&grant);
            } else {
                grants.end_pending(&grant, false);
            }
        }
    }
}

/// The CLI path of the built-in Claude Code, shared by the chat manager (which changes it
/// at runtime: `ChatManager::update_claude_cli_path`, the CLI installed from the UI) and
/// [`BuiltinProviders`] (which reads it at every opening).
pub type ClaudeCliPath = Arc<std::sync::RwLock<Option<String>>>;

/// The built-in provider instances: Claude Code, on the CLI path the server is
/// configured with NOW. The nexus registry replaces this.
pub struct BuiltinProviders {
    cli_path: ClaudeCliPath,
    /// The provider built for the path it was built with: rebuilt when the path changes
    /// (the historical engine reads the path at every opening; so does this one, M2).
    claude: std::sync::Mutex<(Option<String>, Arc<dyn AgentProvider>)>,
}

impl BuiltinProviders {
    /// The built-in instances, on a path nobody changes afterwards.
    pub fn new(claude_cli_path: Option<String>) -> Self {
        Self::following(Arc::new(std::sync::RwLock::new(claude_cli_path)))
    }

    /// The built-in instances, on the path `cli_path` holds at each opening.
    pub fn following(cli_path: ClaudeCliPath) -> Self {
        let path = cli_path.read().unwrap_or_else(|e| e.into_inner()).clone();
        let claude = Self::claude_on(path.clone());
        Self {
            cli_path,
            claude: std::sync::Mutex::new((path, claude)),
        }
    }

    fn claude_on(path: Option<String>) -> Arc<dyn AgentProvider> {
        use nexus_claude::providers::claude_code::{ClaudeCodeConfig, ClaudeCodeProvider};
        let mut config = ClaudeCodeConfig::default();
        config.cli_path = path.map(std::path::PathBuf::from);
        Arc::new(ClaudeCodeProvider::new(config))
    }
}

impl ProviderSource for BuiltinProviders {
    fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
        if provider_id != super::provider::resolver::CLAUDE_CODE {
            return None;
        }
        let path = self
            .cli_path
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut built = self.claude.lock().unwrap_or_else(|e| e.into_inner());
        if built.0 != path {
            *built = (path.clone(), Self::claude_on(path));
        }
        Some(Arc::clone(&built.1))
    }
}

/// A scriptable provider for tests: no process, no network.
#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use async_trait::async_trait;
    use futures::channel::mpsc::{unbounded, UnboundedSender};
    use nexus_claude::agent::{
        AgentEvent, CancelOutcome, CancelScope, EventStream, InterruptOutcome, ModelInfo,
        ProviderHealth, ProviderKind, QuestionAnswer, ResumeToken, SessionSpec,
    };
    use std::sync::Mutex as StdMutex;

    /// What the fake saw and what the test can push.
    #[derive(Default)]
    pub struct FakeState {
        pub permission_answers: StdMutex<Vec<(String, PermissionDecision)>>,
        pub interrupts: StdMutex<Vec<InterruptScope>>,
        pub models: StdMutex<Vec<String>>,
        /// The next `set_model` fails with this error.
        pub fail_set_model: StdMutex<Option<ProviderError>>,
        pub modes: StdMutex<Vec<PolicyMode>>,
        pub closed: AtomicBool,
        pub turns_started: StdMutex<Vec<String>>,
        pub turn_tx: StdMutex<Option<UnboundedSender<AgentEvent>>>,
        pub oob_tx: StdMutex<Option<UnboundedSender<AgentEvent>>>,
        pub opened_specs: StdMutex<Vec<SessionSpec>>,
        pub resumed_with: StdMutex<Vec<ResumeToken>>,
        /// The next `send_turn` fails with this error.
        pub fail_next_turn: StdMutex<Option<ProviderError>>,
        /// The session id the provider has named, as the Claude Code façade knows it:
        /// none at open, the one of the first `session_started` / `done` that carries
        /// one, the token's at resume.
        pub provider_session_id: StdMutex<Option<String>>,
        /// When set, `answer_permission` waits for this signal before it returns: the
        /// provider is slow to take the answer.
        pub hold_answers: StdMutex<Option<Arc<tokio::sync::Notify>>>,
        /// The same, for the answers that allow only (a denial is taken at once).
        pub hold_allows: StdMutex<Option<Arc<tokio::sync::Notify>>>,
        /// The instructions of every `compact` asked (ISO3).
        pub compactions: StdMutex<Vec<Option<String>>>,
    }

    impl FakeState {
        pub fn push(&self, event: AgentEvent) {
            // As the façade does, the id is known before the event reaches the turn.
            if let AgentEvent::SessionStarted {
                provider_session_id: Some(id),
                ..
            }
            | AgentEvent::Done {
                provider_session_id: Some(id),
                ..
            } = &event
            {
                *self.provider_session_id.lock().unwrap() = Some(id.clone());
            }
            self.turn_tx
                .lock()
                .unwrap()
                .as_ref()
                .expect("a turn is open")
                .unbounded_send(event)
                .unwrap();
        }
        pub fn push_oob(&self, event: AgentEvent) {
            self.oob_tx
                .lock()
                .unwrap()
                .as_ref()
                .expect("oob stream taken")
                .unbounded_send(event)
                .unwrap();
        }
        pub fn end_turn(&self) {
            self.turn_tx.lock().unwrap().take();
        }
    }

    pub struct FakeSession {
        pub state: Arc<FakeState>,
        caps: Capabilities,
        oob: StdMutex<Option<EventStream>>,
    }

    #[async_trait]
    impl AgentSession for FakeSession {
        fn capabilities(&self) -> &Capabilities {
            &self.caps
        }
        fn resume_token(&self) -> Option<ResumeToken> {
            self.state
                .provider_session_id
                .lock()
                .unwrap()
                .clone()
                .map(ResumeToken::claude_code_session)
        }
        async fn send_turn(&self, input: TurnInput) -> Result<EventStream, ProviderError> {
            if let Some(error) = self.state.fail_next_turn.lock().unwrap().take() {
                return Err(error);
            }
            let text = input
                .blocks
                .iter()
                .filter_map(|b| match b {
                    nexus_claude::agent::InputBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<String>();
            self.state.turns_started.lock().unwrap().push(text);
            let (tx, rx) = unbounded();
            *self.state.turn_tx.lock().unwrap() = Some(tx);
            Ok(Box::pin(rx))
        }
        async fn answer_permission(
            &self,
            request_id: &str,
            decision: PermissionDecision,
        ) -> Result<(), ProviderError> {
            let hold = self.state.hold_answers.lock().unwrap().clone();
            if let Some(hold) = hold {
                hold.notified().await;
            }
            let hold_allow = self.state.hold_allows.lock().unwrap().clone();
            if let (Some(hold), PermissionDecision::Allow { .. }) = (hold_allow, &decision) {
                hold.notified().await;
            }
            let mut answers = self.state.permission_answers.lock().unwrap();
            // As nexus does: a request is answered once.
            if answers.iter().any(|(id, _)| id == request_id) {
                return Err(ProviderError::invalid(format!(
                    "no pending permission request {request_id}"
                )));
            }
            answers.push((request_id.to_string(), decision));
            Ok(())
        }
        async fn answer_question(
            &self,
            _question_id: &str,
            _answer: QuestionAnswer,
        ) -> Result<(), ProviderError> {
            Err(ProviderError::unsupported("answer_question"))
        }
        async fn interrupt(
            &self,
            scope: InterruptScope,
        ) -> Result<InterruptOutcome, ProviderError> {
            self.state.interrupts.lock().unwrap().push(scope);
            // Report whether a turn was actually running: the manager surfaces
            // this as `InterruptOutcome.delivered` so an idle stop reads false.
            let turn_interrupted = self.state.turn_tx.lock().unwrap().take().is_some();
            Ok(InterruptOutcome {
                turn_interrupted,
                tools_cancelled: 0,
                diagnostic: None,
            })
        }
        async fn cancel_tools(&self, _scope: CancelScope) -> Result<CancelOutcome, ProviderError> {
            Ok(CancelOutcome::default())
        }
        async fn set_model(&self, model: &str) -> Result<(), ProviderError> {
            if let Some(error) = self.state.fail_set_model.lock().unwrap().take() {
                return Err(error);
            }
            self.state.models.lock().unwrap().push(model.to_string());
            Ok(())
        }
        async fn set_policy_mode(
            &self,
            mode: PolicyMode,
            _native: Option<&str>,
        ) -> Result<(), ProviderError> {
            self.state.modes.lock().unwrap().push(mode);
            Ok(())
        }
        async fn compact(
            &self,
            instructions: Option<String>,
        ) -> Result<nexus_claude::agent::CompactOutcome, ProviderError> {
            self.state.compactions.lock().unwrap().push(instructions);
            if !self.caps.compaction_manual {
                return Err(ProviderError::unsupported("compaction_manual"));
            }
            if self.state.turn_tx.lock().unwrap().is_some() {
                return Err(ProviderError::TurnInProgress);
            }
            Ok(nexus_claude::agent::CompactOutcome {
                compacted: true,
                tokens_before: Some(9000),
                tokens_after: Some(1200),
                messages_summarised: 14,
            })
        }
        fn out_of_band(&self) -> Option<EventStream> {
            self.oob.lock().unwrap().take()
        }
        async fn close(&self) -> Result<(), ProviderError> {
            self.state.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[derive(Clone)]
    pub struct FakeProvider {
        pub state: Arc<FakeState>,
        pub fail_open: Arc<StdMutex<Option<ProviderError>>>,
        /// Every `open` fails with this error, as long as it is set (`fail_open` fails one).
        pub fail_every_open: Arc<StdMutex<Option<ProviderError>>>,
        /// Capabilities the sessions of this provider declare.
        pub caps: Arc<StdMutex<Capabilities>>,
        /// The kind it says it is (Claude Code unless a test plays another).
        pub kind: Arc<StdMutex<ProviderKind>>,
        /// What `health` answers (ok unless a test plays another).
        pub health: Arc<StdMutex<Option<ProviderHealth>>>,
        /// How long `open` and `resume` take (none unless a test needs the opening to
        /// yield, or to be dropped while it waits).
        pub open_delay: Arc<StdMutex<Option<std::time::Duration>>>,
    }

    impl FakeProvider {
        pub fn new() -> Self {
            Self {
                state: Arc::new(FakeState::default()),
                fail_open: Arc::new(StdMutex::new(None)),
                fail_every_open: Arc::new(StdMutex::new(None)),
                caps: Arc::new(StdMutex::new(Capabilities::none())),
                kind: Arc::new(StdMutex::new(ProviderKind::ClaudeCode)),
                health: Arc::new(StdMutex::new(None)),
                open_delay: Arc::new(StdMutex::new(None)),
            }
        }
        pub(crate) fn session(&self) -> Arc<FakeSession> {
            let (tx, rx) = unbounded();
            *self.state.oob_tx.lock().unwrap() = Some(tx);
            Arc::new(FakeSession {
                state: Arc::clone(&self.state),
                caps: self.caps.lock().unwrap().clone(),
                oob: StdMutex::new(Some(Box::pin(rx))),
            })
        }
    }

    #[async_trait]
    impl AgentProvider for FakeProvider {
        fn id(&self) -> &str {
            "claude-code"
        }
        fn kind(&self) -> ProviderKind {
            *self.kind.lock().unwrap()
        }
        async fn health(&self) -> ProviderHealth {
            self.health
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| ProviderHealth::ok(None))
        }
        async fn catalog(&self) -> Result<Vec<ModelInfo>, ProviderError> {
            Ok(Vec::new())
        }
        fn capabilities(&self, _model: Option<&str>) -> Capabilities {
            // A local Claude Code takes MCP servers per session: the host gives it the
            // PO server (`ChatManager::carries_per_session_mcp`).
            let mut caps = Capabilities::none();
            caps.per_session_mcp = true;
            caps
        }
        async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn AgentSession>, ProviderError> {
            let delay = *self.open_delay.lock().unwrap();
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            if let Some(e) = self.fail_every_open.lock().unwrap().clone() {
                return Err(e);
            }
            if let Some(e) = self.fail_open.lock().unwrap().take() {
                return Err(e);
            }
            *self.state.provider_session_id.lock().unwrap() = None;
            self.state.opened_specs.lock().unwrap().push(spec);
            Ok(self.session())
        }
        async fn resume(
            &self,
            spec: SessionSpec,
            token: ResumeToken,
        ) -> Result<Arc<dyn AgentSession>, ProviderError> {
            let delay = *self.open_delay.lock().unwrap();
            if let Some(delay) = delay {
                tokio::time::sleep(delay).await;
            }
            *self.state.provider_session_id.lock().unwrap() = token
                .data()
                .get("session_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            self.state.resumed_with.lock().unwrap().push(token);
            self.state.opened_specs.lock().unwrap().push(spec);
            Ok(self.session())
        }
    }

    impl ProviderSource for FakeProvider {
        fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
            (provider_id == "claude-code").then(|| Arc::new(self.clone()) as Arc<dyn AgentProvider>)
        }
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use crate::vault::mask::Masker;
    use nexus_claude::agent::ToolOutput;

    fn tool_result(text: &str) -> AgentEvent {
        AgentEvent::ToolResult {
            id: "t".into(),
            output: Some(ToolOutput::Text(text.into())),
            is_error: false,
            seq: None,
            parent: None,
        }
    }

    #[test]
    fn a_vault_secret_in_a_provider_event_is_masked() {
        let masker = Masker::from_values([("KEY", "sk-agent-secret-9876")]);
        let out = mask_agent_event_with(&masker, tool_result("the key is sk-agent-secret-9876 ok"));
        let wire = serde_json::to_string(&out).unwrap();
        assert!(!wire.contains("sk-agent-secret-9876"), "{wire}");
        assert!(
            wire.contains("the key is"),
            "the rest of the text is kept: {wire}"
        );
    }

    #[test]
    fn an_event_that_cannot_be_masked_is_withheld_not_passed_in_clear() {
        // The secret collides with a key of the event's own JSON form.
        let secret = "input_complete";
        let masker = Masker::from_values([("KEY", secret)]);
        let event = AgentEvent::ToolCall {
            id: "t".into(),
            name: "x".into(),
            input: serde_json::json!({ "note": format!("value {secret}") }),
            category: Default::default(),
            canonical: None,
            input_complete: true,
            seq: None,
            parent: None,
        };
        let out = mask_agent_event_with(&masker, event);
        assert!(
            matches!(&out, AgentEvent::ProviderNotice { kind, .. } if kind == crate::chat::manager::MASKING_FAILED_SUBTYPE),
            "{out:?}"
        );
        let chat = super::super::provider::event_map::EventMapper::new().map(&out);
        assert!(
            matches!(&chat[0], crate::chat::types::ChatEvent::Error { message, .. }
            if message == crate::chat::manager::MASKING_FAILED_MESSAGE)
        );
    }

    /// QUOTA: a rate limit that names a reset far beyond what a retry waits is an exhausted quota,
    /// not a burst: the turn is not run again into the same refusal, the person is told when the
    /// provider serves again. A near reset, or none, keeps the old retry.
    #[test]
    fn a_rate_limit_with_a_far_reset_is_not_retried_and_a_near_one_is() {
        let now = 1_792_500_000_000_u64;
        let at = |resets: Option<u64>| ProviderError::RateLimited {
            retry_after_ms: None,
            resets_at_ms: resets,
        };
        assert!(waits_for_a_reset(&at(Some(now + 3 * 86_400_000)), now));
        assert!(waits_for_a_reset(
            &at(Some(now + MAX_RETRY_DELAY_MS + 1)),
            now
        ));
        assert!(!waits_for_a_reset(&at(Some(now + MAX_RETRY_DELAY_MS)), now));
        assert!(!waits_for_a_reset(&at(None), now));
        assert!(!waits_for_a_reset(&ProviderError::Overloaded, now));
        // The rule reaches the stream: a far reset makes the terminal error not retryable.
        let far = AgentEvent::Error {
            error: at(Some(now + 3 * 86_400_000)),
        };
        assert!(retryable_failure(&far, now).is_none());
        let burst = AgentEvent::Error { error: at(None) };
        assert!(retryable_failure(&burst, now).is_some());
    }

    #[test]
    fn a_rate_limit_and_no_credit_are_refusals_of_the_account_and_nothing_else_is() {
        let done = |error: ProviderError| AgentEvent::Error { error };
        assert!(is_quota_refusal(&done(ProviderError::RateLimited {
            retry_after_ms: None,
            resets_at_ms: Some(1),
        })));
        assert!(is_quota_refusal(&done(ProviderError::InvalidRequest {
            detail: "HTTP 402: insufficient credits".into()
        })));
        assert!(!is_quota_refusal(&done(ProviderError::Overloaded)));
        assert!(!is_quota_refusal(&done(ProviderError::InvalidRequest {
            detail: "HTTP 400: bad".into()
        })));
    }

    #[test]
    fn the_retry_delay_follows_the_provider_then_the_chat_retry_config() {
        let config = RetryConfig::default();
        let limited = |ms| ProviderError::RateLimited {
            retry_after_ms: Some(ms),
            resets_at_ms: None,
        };
        assert_eq!(retry_delay_ms(&limited(40), 1, &config), 40);
        assert_eq!(retry_delay_ms(&limited(900_000), 1, &config), 30_000);
        // The backoff of the Claude Code engine: initial × multiplier^(n-1).
        assert_eq!(retry_delay_ms(&ProviderError::Overloaded, 1, &config), 1000);
        assert_eq!(retry_delay_ms(&ProviderError::Overloaded, 3, &config), 4000);
        let operator = RetryConfig {
            max_attempts: 5,
            initial_delay_ms: 10,
            backoff_multiplier: 3.0,
        };
        assert_eq!(retry_delay_ms(&ProviderError::Overloaded, 1, &operator), 10);
        assert_eq!(retry_delay_ms(&ProviderError::Overloaded, 3, &operator), 90);
        assert_eq!(
            retry_delay_ms(&ProviderError::Overloaded, 3, &operator),
            operator.delay_for_attempt(3),
            "the same figure as stream_response"
        );
        // Never beyond thirty seconds, whatever the configuration (attempt 10: 512 s).
        assert_eq!(
            retry_delay_ms(&ProviderError::Overloaded, 10, &config),
            30_000
        );
    }

    #[test]
    fn hooks_are_degraded_only_where_the_provider_does_not_run_them() {
        use nexus_claude::agent::HookSupport;
        let mut caps = Capabilities::none();
        caps.per_session_mcp = true;
        for support in [HookSupport::None, HookSupport::Command] {
            caps.hooks = support;
            assert!(
                degraded_features(&caps).iter().any(|f| f == "hooks"),
                "{support:?}"
            );
        }
        caps.hooks = HookSupport::InProtocol;
        assert!(
            !degraded_features(&caps).iter().any(|f| f == "hooks"),
            "a provider that runs hooks in its loop is not told it lost them"
        );
        // ...unless it cannot carry the MCP server either (remote Claude Code: given no hooks).
        caps.per_session_mcp = false;
        assert!(degraded_features(&caps).iter().any(|f| f == "hooks"));
    }

    /// M2 (gaps D2, D3 of M0): compaction and sub-agents, by provider kind.
    #[test]
    fn kind_degraded_says_compaction_without_a_window_and_subagents_without_a_fallback() {
        use nexus_claude::agent::{
            ContextWindow, ContextWindowSource, ProviderKind, SubagentSupport,
        };
        let mut caps = Capabilities::none();
        caps.compaction_signal = true;
        // The native harness compacts against the window: none known, none done.
        assert_eq!(
            kind_degraded(ProviderKind::Native, &caps),
            vec!["compaction"]
        );
        // Claude Code, Codex, ACP compact in their own loop (or say they do not).
        for kind in [
            ProviderKind::ClaudeCode,
            ProviderKind::Codex,
            ProviderKind::Acp,
        ] {
            assert!(
                !kind_degraded(kind, &caps).iter().any(|f| f == "compaction"),
                "{kind:?}"
            );
        }
        caps.context_window = Some(ContextWindow {
            value: 64_000,
            source: ContextWindowSource::Configured,
        });
        assert!(kind_degraded(ProviderKind::Native, &caps).is_empty());
        // No compaction signal at all is `degraded_features`' to say, not listed twice.
        caps.compaction_signal = false;
        caps.context_window = None;
        assert!(kind_degraded(ProviderKind::Native, &caps).is_empty());

        // Sub-agents: an ACP agent has none and the host has no fallback for it...
        assert_eq!(
            kind_degraded(ProviderKind::Acp, &caps),
            vec![SUBAGENTS_FEATURE]
        );
        // ...the native harness gets child sessions from the host (A18)...
        assert!(!kind_degraded(ProviderKind::Native, &caps)
            .iter()
            .any(|f| f == SUBAGENTS_FEATURE));
        // ...and a provider that reports its own is not told it lacks them.
        for (kind, support) in [
            (ProviderKind::ClaudeCode, SubagentSupport::Nested),
            (ProviderKind::Codex, SubagentSupport::SeparateThread),
        ] {
            caps.subagents = support;
            assert!(kind_degraded(kind, &caps).is_empty(), "{kind:?}");
        }
    }

    /// VERIFIER: the engine retries (B15), so `retry` must not be announced as missing.
    #[test]
    fn verifier_degraded_features_does_not_claim_retry_is_missing() {
        let caps = Capabilities::none();
        assert!(
            !degraded_features(&caps).iter().any(|f| f == "retry"),
            "the agent engine retries a retryable done.error: {:?}",
            degraded_features(&caps)
        );
    }

    /// What the engine ports is no longer announced as missing, whatever the provider.
    #[test]
    fn the_ported_features_are_not_announced_as_missing() {
        let caps = Capabilities::none();
        let degraded = degraded_features(&caps);
        let ported = ["enrichment", "message_queue", "auto_continue", "nats"];
        assert!(
            !degraded.iter().any(|f| ported.contains(&f.as_str())),
            "{degraded:?}"
        );
    }

    #[test]
    fn only_a_retryable_failure_of_an_error_turn_is_retried() {
        let now = 1_792_500_000_000_u64;
        let done = |is_error, error| AgentEvent::Done {
            stop_reason: nexus_claude::agent::StopReason::Error,
            subtype: None,
            is_error,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 0,
            duration_api_ms: None,
            num_turns: 0,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error,
        };
        assert!(retryable_failure(&done(true, Some(ProviderError::Overloaded)), now).is_some());
        assert!(retryable_failure(&done(true, Some(ProviderError::Unauthorized)), now).is_none());
        assert!(retryable_failure(&done(false, Some(ProviderError::Overloaded)), now).is_none());
        assert!(retryable_failure(&done(true, None), now).is_none());
        assert!(retryable_failure(
            &AgentEvent::Error {
                error: ProviderError::Overloaded
            },
            now
        )
        .is_some());
        // A turn its configured time limit stopped is not run again (ISO5); a silent model
        // stream (the same typed timeout, `error_during_execution`) still is.
        let timeout = |subtype: &str| {
            let mut event = done(
                true,
                Some(ProviderError::Timeout {
                    after_ms: 1_800_000,
                }),
            );
            if let AgentEvent::Done { subtype: s, .. } = &mut event {
                *s = Some(subtype.to_string());
            }
            event
        };
        assert!(retryable_failure(&timeout("error_timeout"), now).is_none());
        assert!(retryable_failure(&timeout("error_during_execution"), now).is_some());
    }

    /// A2/D1: mid-generation retries are narrower than pre-content retries — only the
    /// transient 429/504, never a turn its own limit stopped or a systemic 503.
    #[test]
    fn only_a_rate_limit_or_timeout_is_retried_mid_generation() {
        let now = 1_792_500_000_000_u64;
        let done = |is_error, error| AgentEvent::Done {
            stop_reason: nexus_claude::agent::StopReason::Error,
            subtype: None,
            is_error,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 0,
            duration_api_ms: None,
            num_turns: 0,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error,
        };
        let limited = ProviderError::RateLimited {
            retry_after_ms: Some(5),
            resets_at_ms: None,
        };
        let timeout = ProviderError::Timeout { after_ms: 60_000 };
        assert!(mid_generation_retryable(&done(true, Some(limited)), now).is_some());
        assert!(mid_generation_retryable(&done(true, Some(timeout.clone())), now).is_some());
        assert!(mid_generation_retryable(&AgentEvent::Error { error: timeout }, now).is_some());
        // The other retryable classes keep the old "already showed something" rule.
        assert!(
            mid_generation_retryable(&done(true, Some(ProviderError::Overloaded)), now).is_none()
        );
        assert!(mid_generation_retryable(
            &done(
                true,
                Some(ProviderError::EndpointUnreachable { detail: "d".into() })
            ),
            now
        )
        .is_none());
        // Non-retryable failures are not retried, and a clean done neither.
        assert!(
            mid_generation_retryable(&done(true, Some(ProviderError::Unauthorized)), now).is_none()
        );
        assert!(mid_generation_retryable(&done(false, None), now).is_none());
        assert!(mid_generation_retryable(&done(true, None), now).is_none());
        // A turn its configured time limit stopped is not run again (ISO5).
        let limit_stopped = {
            let mut event = done(
                true,
                Some(ProviderError::Timeout {
                    after_ms: 1_800_000,
                }),
            );
            if let AgentEvent::Done { subtype: s, .. } = &mut event {
                *s = Some("error_timeout".to_string());
            }
            event
        };
        assert!(mid_generation_retryable(&limit_stopped, now).is_none());
    }

    // ── Fix 3: quota_refusal is terminal-only ────────────────────────────────

    /// A 429 retried then followed by a real protocol error: the turn does NOT count as
    /// `quota_refusal` because the terminal event is the protocol error, not the 429.
    /// Before the fix, `outcome.quota_refusal |=` was sticky: once true, always true.
    #[test]
    fn quota_refusal_is_only_true_when_the_terminal_event_is_a_refusal() {
        let refusal = AgentEvent::Error {
            error: ProviderError::RateLimited {
                retry_after_ms: Some(5),
                resets_at_ms: None,
            },
        };
        let protocol_error = AgentEvent::Error {
            error: ProviderError::InvalidRequest {
                detail: "HTTP 400: bad request".into(),
            },
        };
        // The 429 is indeed a quota refusal…
        assert!(is_quota_refusal(&refusal));
        // …but a protocol error is not.
        assert!(!is_quota_refusal(&protocol_error));
        // The fix: only the TERMINAL event determines `quota_refusal`.
        // Simulate the event loop's logic (Fix 3): `if terminal { outcome.quota_refusal = is_quota_refusal(&event); }`
        // If the last event is a protocol error, quota_refusal must be false even though
        // a 429 came earlier. The old code (`|=`) would have kept it true.
        let events = vec![refusal, protocol_error];
        let mut quota_refusal = false;
        for event in &events {
            // Old code: quota_refusal |= is_quota_refusal(event); ← BUG: sticky
            // New code: only on the terminal (last) event.
            let is_terminal = std::ptr::eq(event, events.last().unwrap());
            if is_terminal {
                quota_refusal = is_quota_refusal(event);
            }
        }
        assert!(
            !quota_refusal,
            "the terminal event is a protocol error, not a quota refusal"
        );
    }

    /// When the terminal event IS a 429, `quota_refusal` is true.
    #[test]
    fn quota_refusal_is_true_when_the_terminal_event_is_a_429() {
        let refusal = AgentEvent::Error {
            error: ProviderError::RateLimited {
                retry_after_ms: None,
                resets_at_ms: Some(99),
            },
        };
        assert!(is_quota_refusal(&refusal));
        // A 402 "insufficient credits" is also a refusal.
        let no_credit = AgentEvent::Error {
            error: ProviderError::InvalidRequest {
                detail: "HTTP 402: insufficient credits".into(),
            },
        };
        assert!(is_quota_refusal(&no_credit));
    }

    // ── Fix 4: waits_for_a_reset uses the injected clock ─────────────────────

    /// `waits_for_a_reset` compares the reset time to the `now_ms` argument (the injected
    /// clock), not to the system clock. This test proves it: the same error is either "far"
    /// or "near" depending solely on the `now_ms` we pass.
    #[test]
    fn waits_for_a_reset_depends_only_on_the_injected_now_ms() {
        let resets_at = 1_792_500_000_000_u64 + 3 * 86_400_000; // some date 3 days hence

        // With now_ms close to resets_at: the remaining wait is short → NOT a quota exhaustion.
        let now_close = resets_at - 5_000; // 5 s before the reset
        let error = ProviderError::RateLimited {
            retry_after_ms: None,
            resets_at_ms: Some(resets_at),
        };
        assert!(
            !waits_for_a_reset(&error, now_close),
            "5 s before reset → near → retryable, not a quota wall"
        );

        // With now_ms far before resets_at: the remaining wait is long → IS a quota exhaustion.
        let now_far = resets_at - 3 * 86_400_000; // 3 days before the reset
        assert!(
            waits_for_a_reset(&error, now_far),
            "3 days before reset → far → quota exhaustion"
        );
    }
}

/// Races between a turn being opened (the enrichment awaits) and what reaches the
/// session meanwhile.
#[cfg(test)]
mod turn_race_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use std::sync::atomic::AtomicBool as StdAtomicBool;
    use tokio::sync::Notify;

    /// Holds the first `prepare` until the test lets it go: the time an
    /// enrichment of the knowledge graph takes.
    #[derive(Default)]
    struct Gate {
        entered: Notify,
        release: Notify,
        passed: StdAtomicBool,
    }

    #[async_trait::async_trait]
    impl TurnServices for Gate {
        async fn prepare(
            &self,
            _session_id: &str,
            _shown: &str,
            sent: &str,
            _turn: &crate::refs::turn::TurnExpansion,
        ) -> String {
            if !self.passed.swap(true, Ordering::SeqCst) {
                self.entered.notify_one();
                self.release.notified().await;
            }
            sent.to_string()
        }

        async fn continuation(&self, _session_id: &str) -> String {
            String::new()
        }
    }

    async fn rig() -> (FakeProvider, Arc<Gate>, Arc<AgentSessionHandle>) {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let gate = Arc::new(Gate::default());
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "claude-code",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                Some(Arc::clone(&gate) as Arc<dyn TurnServices>),
            )
            .await;
        (provider, gate, handle)
    }

    async fn turns_reach(provider: &FakeProvider, n: usize) -> Vec<String> {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let turns = provider.state.turns_started.lock().unwrap().clone();
            if turns.len() >= n || std::time::Instant::now() > deadline {
                return turns;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// A message held while the first turn is being opened waits behind it; when the
    /// provider then refuses that first turn, the held message still runs.
    #[tokio::test]
    async fn a_message_held_while_a_turn_opens_runs_even_when_that_turn_fails() {
        let (provider, gate, handle) = rig().await;
        *provider.state.fail_next_turn.lock().unwrap() = Some(ProviderError::EndpointUnreachable {
            detail: "down for a moment".into(),
        });
        let first = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move { handle.send_message("first").await })
        };
        gate.entered.notified().await;
        assert!(handle.queue_message("second").await.unwrap(), "held");
        gate.release.notify_one();
        assert!(first.await.unwrap().is_err(), "the first turn was refused");

        let turns = turns_reach(&provider, 1).await;
        assert_eq!(turns, ["second"], "the queued message is not stranded");
    }

    /// A Stop while the turn is being prepared (the enrichment awaits, the provider
    /// has no turn to interrupt yet) stops that turn: it is not sent, and the
    /// session stops streaming — the Claude Code engine breaks such a stream at once.
    #[tokio::test]
    async fn a_stop_while_a_turn_is_prepared_stops_that_turn() {
        let (provider, gate, handle) = rig().await;
        let first = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move { handle.send_message("first").await })
        };
        gate.entered.notified().await;
        handle.interrupt().await.unwrap();
        gate.release.notify_one();
        first.await.unwrap().unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while handle.is_streaming.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let turns = provider.state.turns_started.lock().unwrap().clone();
        assert!(
            turns.is_empty(),
            "the stopped turn went to the provider: {turns:?}"
        );
        assert!(
            !handle.is_streaming.load(Ordering::SeqCst),
            "the session stops streaming"
        );
    }
}

/// What happens to a live session when the same id is adopted again (two resumes
/// racing for one session).
#[cfg(test)]
mod adopt_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    /// The handle replaced is ended: its listeners (NATS) stop, its provider
    /// session is closed. Otherwise both handles answer the same NATS subjects and
    /// a message from another instance is played twice.
    #[tokio::test]
    async fn adopting_a_live_session_again_ends_the_handle_it_replaces() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let adopt = || {
            runtime.adopt(
                "s",
                "claude-code",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
        };
        let first = adopt().await;
        let second = adopt().await;
        assert!(
            first.closed.is_cancelled(),
            "the replaced handle's listeners stop"
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !provider.state.closed.load(Ordering::SeqCst) && std::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            provider.state.closed.load(Ordering::SeqCst),
            "the replaced provider session is closed"
        );
        assert!(!second.closed.is_cancelled());
        assert!(Arc::ptr_eq(&runtime.get("s").await.unwrap(), &second));
    }

    /// A native session opened without its `nexus-tools` says so: the host adds the
    /// feature the provider's capabilities cannot report. A plain adoption does not.
    #[tokio::test]
    async fn a_session_the_host_knows_lacks_nexus_tools_reports_it_once() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let plain = runtime
            .adopt(
                "plain",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
        assert!(!plain.degraded.iter().any(|f| f == NEXUS_TOOLS_FEATURE));

        let missing = runtime
            .adopt_with(
                "missing",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
                vec![
                    NEXUS_TOOLS_FEATURE.to_string(),
                    NEXUS_TOOLS_FEATURE.to_string(),
                ],
            )
            .await;
        assert_eq!(
            missing
                .degraded
                .iter()
                .filter(|f| *f == NEXUS_TOOLS_FEATURE)
                .count(),
            1,
            "listed once, whatever the caller passes"
        );
    }

    /// Review of #726 (CI): a CLI asks the identical call again as soon as it has the first
    /// answer, BEFORE `answer_permission` returns to the engine. A `session` grant must
    /// already cover it then — it was kept only once the answer returned, so the second
    /// request waited for a person who had already answered (the CLI then hung).
    #[tokio::test]
    async fn the_identical_call_asked_while_the_session_answer_is_delivered_is_covered() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes =
            vec![PermissionScope::Once, PermissionScope::Session];
        let handle = runtime
            .adopt(
                "native-race",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
        let mut rx = handle.events_tx.subscribe();
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
        handle.emit(write("write-1")).await;
        // The provider takes the answer, and does not return yet.
        let hold = Arc::new(tokio::sync::Notify::new());
        *provider.state.hold_answers.lock().unwrap() = Some(Arc::clone(&hold));
        let answering = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move {
                handle
                    .answer_permission_scoped("write-1", true, PermissionAnswerScope::Session)
                    .await
            })
        };
        // No sleep: the fake holds the answer once it took its own handle on `hold`
        // (current-thread runtime: it is then parked on it).
        while Arc::strong_count(&hold) < 3 {
            tokio::task::yield_now().await;
        }
        // Meanwhile the CLI asks the identical call (answered at once: nothing holds it).
        *provider.state.hold_answers.lock().unwrap() = None;
        handle.emit(write("write-2")).await;
        hold.notify_one();
        answering
            .await
            .unwrap()
            .expect("the first answer is delivered");
        let answered: Vec<String> = provider
            .state
            .permission_answers
            .lock()
            .unwrap()
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        assert!(
            answered.contains(&"write-2".to_string()),
            "the backend answered the identical call: {answered:?}"
        );
        let mut decided_by_backend = false;
        while let Ok(event) = rx.try_recv() {
            if let ChatEvent::PermissionDecision {
                id, rule: Some(_), ..
            } = &event
            {
                decided_by_backend |= id == "write-2";
            }
        }
        assert!(decided_by_backend, "permission_decision with its rule");
    }

    /// The lasting scope of a session of `proj`, owned by `me@example.com`: the shape in which
    /// this backend's `always` is offered at all.
    fn lasting_scope_of(root: &std::path::Path) -> super::super::lasting_rules::LastingScope {
        let store = super::super::lasting_rules::LastingRuleStore::new(root.join("rules"));
        std::fs::create_dir_all(store.root()).unwrap();
        super::super::lasting_rules::LastingScope::of_session(
            Some(&store),
            Some("me@example.com"),
            Some("proj"),
            Some(root),
            root,
        )
        .expect("the session keeps a lasting rule")
    }

    /// The permission request the harness raises about a DESTINATION (N23b): the tool call names
    /// the page it would reach.
    fn fetch_request(id: &str, url: &str) -> ChatEvent {
        ChatEvent::PermissionRequest {
            id: id.to_string(),
            tool: "mcp__nexus__WebFetch".into(),
            input: serde_json::json!({ "url": url }),
            parent_tool_use_id: None,
            category: Some("web".into()),
            canonical: Some("WebFetch".into()),
            tool_use_id: None,
            always: None,
            consent: None,
        }
    }

    fn consent_key(origin: &str) -> (String, String) {
        (
            super::super::provider::settings::project_scope("proj"),
            format!(
                "{}{origin}",
                super::super::provider::nexus_tools::TOOL_ORIGIN_PREFIX
            ),
        )
    }

    /// N23b: an `always` answer to a question about a DESTINATION records the project's consent,
    /// in the very store `PUT /api/projects/{slug}/network-tools/origins` writes — never a lasting
    /// rule about a tool call, which the gate of the next session would not read (one store of
    /// truth, and the same writer).
    #[tokio::test]
    async fn an_always_answer_to_an_origin_question_writes_the_projects_consent() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(Arc::clone(&graph));
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes = vec![
            PermissionScope::Once,
            PermissionScope::Session,
            PermissionScope::Always,
        ];
        let root = tempfile::tempdir().unwrap();
        let handle = runtime
            .adopt_declaring(
                "n23b-always",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
                Vec::new(),
                super::super::session_grants::SessionGrants::declaring(Vec::new())
                    .with_lasting(Some(lasting_scope_of(root.path()))),
            )
            .await;
        handle.emit(fetch_request("p1", "https://docs.rs/x")).await;
        handle
            .answer_permission_scoped("p1", true, PermissionAnswerScope::Always)
            .await
            .expect("the answer is delivered");
        // THE STORE, not a mock of it: the document the settings route reads back.
        let (scope, key) = consent_key("https://docs.rs");
        let stored = graph
            .get_llm_setting(&scope, &key)
            .await
            .unwrap()
            .expect("the project's consent to the origin");
        let consent: super::super::provider::nexus_tools::ToolOriginConsent =
            serde_json::from_str(&stored).unwrap();
        assert_eq!(consent.origin, "https://docs.rs");
        assert_eq!(
            consent.consented_by, "me@example.com",
            "the person who answered, never the model"
        );
        // The provider is answered `once`: the durable part is the project's, not this session's.
        let answers = provider.state.permission_answers.lock().unwrap();
        assert!(
            matches!(
                answers.last(),
                Some((
                    _,
                    PermissionDecision::Allow {
                        scope: PermissionScope::Once,
                        ..
                    }
                ))
            ),
            "the session approves the call only: {:?}",
            answers.last()
        );
    }

    /// N23b: a `session` answer to a question about a DESTINATION is kept by the harness against
    /// the SUBJECT (`tool\u{0}origin`, nexus `providers::native::session`), so approving one
    /// origin says nothing about the next; and it writes NOTHING durable — the project's store
    /// stays empty, which is what makes `always` the only act that grants an origin for good.
    #[tokio::test]
    async fn a_session_answer_to_an_origin_question_is_per_origin_and_writes_no_consent() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(Arc::clone(&graph));
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes =
            vec![PermissionScope::Once, PermissionScope::Session];
        let root = tempfile::tempdir().unwrap();
        let handle = runtime
            .adopt_declaring(
                "n23b-session",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
                Vec::new(),
                super::super::session_grants::SessionGrants::declaring(Vec::new())
                    .with_lasting(Some(lasting_scope_of(root.path()))),
            )
            .await;
        let mut rx = handle.events_tx.subscribe();
        handle.emit(fetch_request("p1", "https://docs.rs/x")).await;
        handle
            .answer_permission_scoped("p1", true, PermissionAnswerScope::Session)
            .await
            .expect("the answer is delivered");
        // The harness is answered `session`, and its key is the tool AND the subject: a later
        // request about ANOTHER origin is a different key, so it is asked again (nexus's own
        // tests pin that rule; this asserts the backend does not answer `once` and lose it).
        let answered = {
            let answers = provider.state.permission_answers.lock().unwrap();
            answers.last().cloned()
        };
        assert!(
            matches!(
                answered,
                Some((
                    _,
                    PermissionDecision::Allow {
                        scope: PermissionScope::Session,
                        ..
                    }
                ))
            ),
            "a destination is remembered against its subject, not this call only: {answered:?}"
        );
        // This backend keeps NO grant of its own for it: a grant would be a second memory of the
        // same approval, matching the identical CALL (`WebFetch {url}`) where the person's answer
        // covers an ORIGIN — and the client is told what it covers (P11b says a `rule` is a call).
        assert!(
            handle
                .session_grants
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .covering(&super::super::session_grants::AskedCall {
                    asker: super::super::session_grants::Asker::Native,
                    tool: "mcp__nexus__WebFetch".into(),
                    canonical: Some("WebFetch".into()),
                    input: serde_json::json!({ "url": "https://docs.rs/x" }),
                })
                .is_none(),
            "the origin is the harness's memory, not a grant of this backend"
        );
        let mut said = None;
        while let Ok(event) = rx.try_recv() {
            if let ChatEvent::PermissionDecision {
                id,
                rule,
                scope,
                allow: true,
            } = &event
            {
                if id == "p1" {
                    said = Some((rule.clone(), scope.map(|s| s.as_str())));
                }
            }
        }
        assert_eq!(
            said,
            Some((Some("origin https://docs.rs".to_string()), Some("session"))),
            "the client reads an ORIGIN, not a call"
        );
        // Nothing durable for either origin.
        for origin in ["https://docs.rs", "https://evil.example"] {
            let (scope, key) = consent_key(origin);
            assert!(
                graph.get_llm_setting(&scope, &key).await.unwrap().is_none(),
                "{origin} must not be consented to by a `session` answer"
            );
        }
    }

    /// The permission request a person sees, as the client receives it (decorated by `emit`).
    async fn emitted_request(rx: &mut broadcast::Receiver<ChatEvent>, id: &str) -> ChatEvent {
        while let Ok(event) = rx.try_recv() {
            if matches!(&event, ChatEvent::PermissionRequest { id: got, .. } if got == id) {
                return event;
            }
        }
        panic!("no permission_request {id} was emitted");
    }

    async fn origin_handle(
        runtime: &AgentRuntime,
        provider: &FakeProvider,
        sid: &str,
        lasting: Option<super::super::lasting_rules::LastingScope>,
    ) -> Arc<AgentSessionHandle> {
        provider.caps.lock().unwrap().permission_scopes = vec![
            PermissionScope::Once,
            PermissionScope::Session,
            PermissionScope::Always,
        ];
        runtime
            .adopt_declaring(
                sid,
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
                Vec::new(),
                super::super::session_grants::SessionGrants::declaring(Vec::new())
                    .with_lasting(lasting),
            )
            .await
    }

    /// CONSENT-WIRE: a question about a DESTINATION names, on the wire, the subject PO asserts
    /// from the call (`https://` once upgraded) and every form `always` will write — and the
    /// answer records EXACTLY those, so what the person approves is what PO keeps.
    #[tokio::test]
    async fn an_origin_question_names_its_subject_and_the_answer_keeps_exactly_it() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(Arc::clone(&graph));
        let provider = FakeProvider::new();
        let root = tempfile::tempdir().unwrap();
        let handle = origin_handle(
            &runtime,
            &provider,
            "consent-wire",
            Some(lasting_scope_of(root.path())),
        )
        .await;
        let mut rx = handle.events_tx.subscribe();
        handle.emit(fetch_request("p1", "http://docs.rs/x")).await;

        let ChatEvent::PermissionRequest {
            consent, always, ..
        } = emitted_request(&mut rx, "p1").await
        else {
            unreachable!()
        };
        let consent = consent.expect("a question about a destination names its subject");
        assert_eq!(
            consent.subject, "https://docs.rs",
            "the endpoint really reached"
        );
        assert_eq!(
            consent.destinations,
            vec!["http://docs.rs".to_string(), "https://docs.rs".to_string()],
            "every form the gate judges"
        );
        assert_eq!(
            always,
            Some(true),
            "a human session in a project may keep the origin for good"
        );

        // The answer path keeps the very destinations the wire named, no more, no less.
        handle
            .answer_permission_scoped("p1", true, PermissionAnswerScope::Always)
            .await
            .expect("the answer is delivered");
        for origin in &consent.destinations {
            let (scope, key) = consent_key(origin);
            assert!(
                graph.get_llm_setting(&scope, &key).await.unwrap().is_some(),
                "{origin} was shown and must be kept"
            );
        }
    }

    /// A SEARCH names no page of its own: what it would reach is the engines the session was
    /// launched with. A question about it is therefore about those origins — all of them, since
    /// nexus-tools falls back from one engine to the next — and `always` writes every one of
    /// them. Without this, a web search the project had not consented to stayed a flat refusal a
    /// person could not answer from the chat at all.
    #[tokio::test]
    async fn a_search_asks_about_its_engines_and_an_always_answer_keeps_them_all() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(Arc::clone(&graph));
        let provider = FakeProvider::new();
        let root = tempfile::tempdir().unwrap();
        let session = "consent-wire-search";
        super::super::provider::nexus_tools::remember_search_origins(
            session,
            &[
                "https://html.duckduckgo.com".to_string(),
                "http://api.search.brave.com".to_string(),
            ],
        );
        let handle = origin_handle(
            &runtime,
            &provider,
            session,
            Some(lasting_scope_of(root.path())),
        )
        .await;
        let mut rx = handle.events_tx.subscribe();
        handle
            .emit(ChatEvent::PermissionRequest {
                id: "s1".into(),
                tool: "mcp__nexus__WebSearch".into(),
                input: serde_json::json!({ "query": "cve" }),
                parent_tool_use_id: None,
                category: Some("web".into()),
                canonical: Some("WebSearch".into()),
                tool_use_id: None,
                always: None,
                consent: None,
            })
            .await;

        let ChatEvent::PermissionRequest {
            consent, always, ..
        } = emitted_request(&mut rx, "s1").await
        else {
            unreachable!()
        };
        let consent = consent.expect("a search is a question about its engines");
        assert_eq!(
            consent.subject, "https://html.duckduckgo.com",
            "the first engine the gate tries, upgraded as nexus-tools reaches it"
        );
        assert_eq!(
            consent.destinations,
            vec![
                "https://html.duckduckgo.com".to_string(),
                "https://api.search.brave.com".to_string(),
            ],
            "every engine of the session, deduplicated and secure"
        );
        assert_eq!(always, Some(true));

        handle
            .answer_permission_scoped("s1", true, PermissionAnswerScope::Always)
            .await
            .expect("the answer is delivered");
        for origin in &consent.destinations {
            let (scope, key) = consent_key(origin);
            assert!(
                graph.get_llm_setting(&scope, &key).await.unwrap().is_some(),
                "an engine the person was shown must be kept: {origin}"
            );
        }
        // A search that reaches no engine is not a question: nothing to consent to.
        super::super::provider::nexus_tools::forget_search_origins(session);
        assert!(super::super::provider::nexus_tools::search_origins_of(session).is_empty());
    }

    /// CONSENT-WIRE: an ordinary tool permission is serialized exactly as before — no `consent`
    /// key at all — so nothing changes for a client that never meets a destination.
    #[tokio::test]
    async fn an_ordinary_tool_permission_carries_no_consent() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(Arc::clone(&graph));
        let provider = FakeProvider::new();
        let handle = origin_handle(&runtime, &provider, "consent-wire-bash", None).await;
        let mut rx = handle.events_tx.subscribe();
        handle
            .emit(ChatEvent::PermissionRequest {
                id: "b1".into(),
                tool: "Bash".into(),
                input: serde_json::json!({ "command": "ls" }),
                parent_tool_use_id: None,
                category: Some("command".into()),
                canonical: Some("Bash".into()),
                tool_use_id: None,
                always: None,
                consent: None,
            })
            .await;
        let event = emitted_request(&mut rx, "b1").await;
        let ChatEvent::PermissionRequest { consent, .. } = &event else {
            unreachable!()
        };
        assert!(consent.is_none());
        let wire = serde_json::to_value(&event).unwrap();
        assert!(wire.get("consent").is_none(), "no key on the wire: {wire}");
    }

    /// CONSENT-WIRE: a session that cannot keep a lasting consent (no project, nobody whose
    /// `always` it would be) still names the subject, but never offers `always`: the client
    /// would offer an answer the backend refuses (`keep_origin_consents`).
    #[tokio::test]
    async fn without_a_project_an_origin_question_never_offers_always() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(Arc::clone(&graph));
        let provider = FakeProvider::new();
        let handle = origin_handle(&runtime, &provider, "consent-wire-nowhere", None).await;
        let mut rx = handle.events_tx.subscribe();
        handle.emit(fetch_request("p1", "https://docs.rs/x")).await;
        let ChatEvent::PermissionRequest {
            consent, always, ..
        } = emitted_request(&mut rx, "p1").await
        else {
            unreachable!()
        };
        assert_eq!(
            consent.map(|c| c.subject).as_deref(),
            Some("https://docs.rs")
        );
        assert_eq!(always, None, "`always` would be refused: not offered");
    }

    /// Follow-up review of #726, point 1: the request carrying a `session` answer is
    /// cancelled (the client dropped its HTTP or WebSocket request) while the provider
    /// takes it. The grant made pending for that time is rolled back: the identical call
    /// asked afterwards waits for the user, it is not answered by a grant nobody kept.
    #[tokio::test]
    async fn a_session_answer_cancelled_in_flight_leaves_no_grant_behind() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes =
            vec![PermissionScope::Once, PermissionScope::Session];
        let handle = runtime
            .adopt(
                "native-cancelled",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
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
        handle.emit(write("write-1")).await;
        let hold = Arc::new(tokio::sync::Notify::new());
        *provider.state.hold_answers.lock().unwrap() = Some(Arc::clone(&hold));
        let answering = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move {
                handle
                    .answer_permission_scoped("write-1", true, PermissionAnswerScope::Session)
                    .await
            })
        };
        // The provider holds the answer (current-thread runtime: parked on `hold`).
        while Arc::strong_count(&hold) < 3 {
            tokio::task::yield_now().await;
        }
        // The client goes away: the future carrying the answer is dropped.
        answering.abort();
        assert!(answering.await.unwrap_err().is_cancelled());
        *provider.state.hold_answers.lock().unwrap() = None;
        handle.emit(write("write-2")).await;
        assert!(
            provider.state.permission_answers.lock().unwrap().is_empty(),
            "nothing answered write-2 for the user"
        );
    }

    /// The review of #681 (finding 1): Codex asks a patch with `{reason}` only (nexus
    /// `codex/map.rs`), no path, no diff. A `session` answer to a first patch is refused,
    /// typed; the second patch, to another file, is asked to the user, never answered by
    /// the backend.
    #[tokio::test]
    async fn a_codex_patch_is_never_granted_for_the_session_and_the_next_one_is_asked() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes =
            vec![PermissionScope::Once, PermissionScope::Session];
        let handle = runtime
            .adopt(
                "codex-patches",
                "codex",
                provider.session(),
                1,
                "codex",
                serde_json::json!({}),
                None,
            )
            .await;
        let patch = |id: &str| ChatEvent::PermissionRequest {
            id: id.to_string(),
            tool: "apply_patch".into(),
            input: serde_json::json!({ "reason": null }),
            parent_tool_use_id: None,
            category: Some("edit".into()),
            canonical: Some("Edit".into()),
            tool_use_id: None,
            always: None,
            consent: None,
        };
        // The first patch (README.md): `session` refused, nothing answered; `once` goes.
        handle.emit(patch("patch-readme")).await;
        let refused = handle
            .answer_permission_scoped("patch-readme", true, PermissionAnswerScope::Session)
            .await;
        assert!(
            matches!(
                refused,
                Err(PermissionDeliveryError::ScopeUnsupported(
                    PermissionAnswerScope::Session
                ))
            ),
            "{refused:?}"
        );
        assert!(provider.state.permission_answers.lock().unwrap().is_empty());
        handle
            .answer_permission_scoped("patch-readme", true, PermissionAnswerScope::Once)
            .await
            .unwrap();
        // The second patch (~/.zshrc) reads the same: it waits for the user.
        handle.emit(patch("patch-zshrc")).await;
        let answered: Vec<String> = provider
            .state
            .permission_answers
            .lock()
            .unwrap()
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        assert_eq!(answered, vec!["patch-readme".to_string()]);

        // The same session kind on the native engine: a `nexus-tools` write IS granted
        // for the identical call (its input is the whole call).
        let native = runtime
            .adopt(
                "native-write",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
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
        native.emit(write("write-1")).await;
        native
            .answer_permission_scoped("write-1", true, PermissionAnswerScope::Session)
            .await
            .unwrap();
        native.emit(write("write-2")).await;
        let answered: Vec<String> = provider
            .state
            .permission_answers
            .lock()
            .unwrap()
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        assert_eq!(
            answered,
            vec![
                "patch-readme".to_string(),
                "write-1".into(),
                "write-2".into()
            ]
        );
    }

    /// Lot review of #721 (finding 2c): a request a session grant covers is answered by the
    /// backend, and then forgotten — the set of the requests it answers itself used to keep
    /// every one for the session's whole life. Each stays "needs nobody" for a parent that
    /// reads it late; a request still waiting for the user does not. Red before: 20 covered
    /// requests left 20 ids in the set.
    #[tokio::test]
    async fn a_request_answered_by_a_grant_does_not_stay_in_the_self_answered_set() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes =
            vec![PermissionScope::Once, PermissionScope::Session];
        let native = runtime
            .adopt(
                "native-forgets",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
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
        native.emit(write("write-0")).await;
        assert!(!native.answers_itself("write-0"), "waits for the user");
        native
            .answer_permission_scoped("write-0", true, PermissionAnswerScope::Session)
            .await
            .unwrap();
        for i in 1..=20 {
            native.emit(write(&format!("write-{i}"))).await;
        }
        assert_eq!(provider.state.permission_answers.lock().unwrap().len(), 21);
        assert_eq!(native.self_answered_len(), 0, "answered, so forgotten");
        assert!(native.waiting_permission_ids().is_empty());
        assert!(
            native.answers_itself("write-20"),
            "answered: nobody needs to answer it"
        );
        // A call no grant covers still waits for the user: not answered by the backend.
        native
            .emit(ChatEvent::PermissionRequest {
                id: "bash-1".into(),
                tool: "Bash".into(),
                input: serde_json::json!({ "command": "rm -rf build" }),
                parent_tool_use_id: None,
                category: None,
                canonical: Some("Bash".into()),
                tool_use_id: None,
                always: None,
                consent: None,
            })
            .await;
        assert!(!native.answers_itself("bash-1"));
    }

    /// Review of #727 (second review, finding 3): "answered" is what THIS handle saw asked
    /// and had answered — never an id it did not see. After a resume under the same id,
    /// a request of the earlier session is unknown to the new handle: nobody here will
    /// answer it, so it is not answered (a parent fails fast on it). Red before: any id
    /// absent from the waiting requests counted as answered.
    #[tokio::test]
    async fn an_id_the_handle_never_saw_is_not_answered() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let adopt = || {
            runtime.adopt(
                "resumed",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
        };
        let bash = |id: &str| ChatEvent::PermissionRequest {
            id: id.into(),
            tool: "Bash".into(),
            input: serde_json::json!({ "command": "make" }),
            parent_tool_use_id: None,
            category: None,
            canonical: Some("Bash".into()),
            tool_use_id: None,
            always: None,
            consent: None,
        };
        let first = adopt().await;
        first.emit(bash("old-1")).await;
        assert!(!first.answers_itself("old-1"), "waits on its user");
        let second = adopt().await;
        assert!(
            !second.answers_itself("old-1"),
            "never seen here: not answered"
        );
        assert!(!second.answers_itself("never-asked"));
        second.emit(bash("new-1")).await;
        assert!(!second.answers_itself("new-1"), "waits on its user");
        second
            .answer_permission_scoped("new-1", true, PermissionAnswerScope::Once)
            .await
            .unwrap();
        assert!(second.answers_itself("new-1"), "seen asked, then answered");
        // An answer the provider refuses marks nothing.
        second.emit(bash("new-2")).await;
        provider
            .state
            .permission_answers
            .lock()
            .unwrap()
            .push(("new-2".into(), PermissionDecision::allow_once()));
        assert!(second
            .answer_permission_scoped("new-2", false, PermissionAnswerScope::Once)
            .await
            .is_err());
        assert!(!second.answers_itself("new-2"));
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn the_turn_input_is_the_text_then_the_images_in_order() {
        let images = [
            super::super::message_attachments::AttachedImage {
                media_type: "image/png".into(),
                data_base64: "AAAA".into(),
                source: attachment("image/png"),
            },
            super::super::message_attachments::AttachedImage {
                media_type: "image/jpeg".into(),
                data_base64: "BBBB".into(),
                source: attachment("image/jpeg"),
            },
        ];
        let input = turn_input("look".into(), &images);
        assert_eq!(input.blocks.len(), 3);
        assert!(
            matches!(&input.blocks[0], nexus_claude::agent::InputBlock::Text { text } if text == "look")
        );
        assert!(matches!(
            &input.blocks[2],
            nexus_claude::agent::InputBlock::Image { media_type, data_base64 }
                if media_type == "image/jpeg" && data_base64 == "BBBB"
        ));
        assert_eq!(turn_input("plain".into(), &[]), TurnInput::text("plain"));
    }

    #[test]
    fn a_provider_refusal_of_the_images_is_a_typed_error_and_nothing_else_is() {
        let reason = |e: &ProviderError| match image_refusal(e) {
            Some(ChatEvent::Error { code, reason, .. }) => {
                assert_eq!(code.as_deref(), Some("images_refused"));
                reason
            }
            Some(other) => panic!("{other:?}"),
            None => None,
        };
        assert_eq!(
            reason(&ProviderError::unsupported("images")).as_deref(),
            Some("unsupported")
        );
        // The Claude Code façade's own checks (type, size) refuse with InvalidRequest;
        // the detail reaches the user.
        let invalid = ProviderError::invalid("image media type `image/bmp` is not one of ...");
        assert_eq!(reason(&invalid).as_deref(), Some("invalid"));
        match image_refusal(&invalid) {
            Some(ChatEvent::Error { message, .. }) => {
                assert!(message.contains("image/bmp"), "{message}")
            }
            other => panic!("{other:?}"),
        }
        // Another failure of the turn is not about the images.
        assert!(image_refusal(&ProviderError::unsupported("hooks")).is_none());
        assert!(image_refusal(&ProviderError::Overloaded).is_none());
    }

    fn attachment(mime: &str) -> super::super::message_attachments::MessageAttachment {
        super::super::message_attachments::MessageAttachment {
            id: uuid::Uuid::new_v4(),
            filename: "x".into(),
            mime_type: mime.into(),
            size_bytes: 4,
        }
    }

    fn image(media_type: &str, data_base64: &str) -> TurnInput {
        TurnInput {
            blocks: vec![
                nexus_claude::agent::InputBlock::Text {
                    text: "look".into(),
                },
                nexus_claude::agent::InputBlock::Image {
                    media_type: media_type.into(),
                    data_base64: data_base64.into(),
                },
            ],
        }
    }

    /// The three refusals of nexus' image checks, PRODUCED by nexus
    /// (`claude_code::input::content_blocks`), are `images_refused` / `invalid`:
    /// a change of their wording in nexus fails here, not silently on the wire.
    #[test]
    fn the_nexus_image_refusals_are_recognised_as_nexus_words_them() {
        use nexus_claude::providers::claude_code::input::{content_blocks, MAX_IMAGE_BYTES};
        // A payload decoding to more than the cap (length a multiple of 4).
        let too_big = "A".repeat((MAX_IMAGE_BYTES / 3 + 2) * 4);
        for input in [
            image("image/bmp", "AAAA"),
            image("image/png", "data:image/png;base64,AAAA"),
            image("image/png", &too_big),
        ] {
            let error = content_blocks(&input).expect_err("nexus refuses this image");
            match image_refusal(&error) {
                Some(ChatEvent::Error { code, reason, .. }) => {
                    assert_eq!(code.as_deref(), Some("images_refused"), "{error}");
                    assert_eq!(reason.as_deref(), Some("invalid"), "{error}");
                }
                other => panic!("{error} -> {other:?}"),
            }
        }
    }

    /// An `InvalidRequest` that is not about the image — on a turn that carries
    /// one — keeps the usual path: never `images_refused`.
    #[test]
    fn an_invalid_request_without_image_wording_on_a_turn_with_an_image_is_not_images_refused() {
        use nexus_claude::providers::claude_code::input::content_blocks;
        let empty = content_blocks(&TurnInput { blocks: vec![] }).expect_err("no content");
        for error in [
            empty,
            ProviderError::invalid("context length exceeded: 140000 tokens, 128000 allowed"),
            ProviderError::invalid("unknown parameter: temperature"),
            ProviderError::invalid("the image of the project is unclear"),
        ] {
            assert!(image_refusal(&error).is_none(), "{error}");
        }
    }

    /// The text of a turn that carries an image inline loses that image's
    /// "no text could be extracted… /raw" line (its heading stays); a turn
    /// without images sends its text untouched.
    #[test]
    fn the_turn_text_drops_the_no_text_line_of_an_image_it_sends_inline() {
        let source = attachment("image/png");
        let note = format!(
            "[no text could be extracted from this image/png file (4 bytes); the original is at GET /api/documents/{}/raw]\n",
            source.id
        );
        let other = attachment("image/png");
        let other_note = format!(
            "[no text could be extracted from this image/png file (4 bytes); the original is at GET /api/documents/{}/raw]\n",
            other.id
        );
        let text = format!(
            "look\n\n### x (id {})\n{note}\n### y (id {})\n{other_note}",
            source.id, other.id
        );
        let sent = super::super::message_attachments::AttachedImage {
            media_type: "image/png".into(),
            data_base64: "AAAA".into(),
            source,
        };
        let input = turn_input(text.clone(), std::slice::from_ref(&sent));
        let nexus_claude::agent::InputBlock::Text { text: carried } = &input.blocks[0] else {
            panic!("text first");
        };
        assert!(!carried.contains(&note), "{carried}");
        assert!(carried.contains(&format!("### x (id {})", sent.source.id)));
        assert!(
            carried.contains(&other_note),
            "an image not sent keeps its line"
        );
        assert_eq!(turn_input(text.clone(), &[]), TurnInput::text(text));
    }

    /// The legacy engine's input: no image → the prompt untouched (the string
    /// path); a valid image → the CLI blocks, text then image; an image nexus
    /// refuses → `images_refused` / `invalid`, nothing to send.
    #[test]
    fn the_cli_input_is_the_string_without_image_and_checked_blocks_with_one() {
        let text = "look at this".to_string();
        assert!(matches!(
            CliInput::of(text.clone(), &[]),
            Ok(CliInput::Text(t)) if t == text
        ));

        let pixel = super::super::message_attachments::AttachedImage {
            media_type: "image/png".into(),
            data_base64: "AAAA".into(),
            source: attachment("image/png"),
        };
        match CliInput::of(text.clone(), std::slice::from_ref(&pixel)) {
            Ok(CliInput::Blocks(blocks)) => assert_eq!(
                blocks,
                vec![
                    nexus_claude::UserContentBlock::text(text.clone()),
                    nexus_claude::UserContentBlock::image_base64("image/png", "AAAA"),
                ]
            ),
            other => panic!("{other:?}"),
        }

        let svg = super::super::message_attachments::AttachedImage {
            media_type: "image/svg+xml".into(),
            ..pixel.clone()
        };
        let not_base64 = super::super::message_attachments::AttachedImage {
            data_base64: "data:image/png;base64,AAAA".into(),
            ..pixel
        };
        for bad in [svg, not_base64] {
            match CliInput::of(text.clone(), std::slice::from_ref(&bad)).map_err(|e| *e) {
                Err(ChatEvent::Error { code, reason, .. }) => {
                    assert_eq!(code.as_deref(), Some("images_refused"));
                    assert_eq!(reason.as_deref(), Some("invalid"));
                }
                other => panic!("{other:?}"),
            }
        }
    }
}

/// The session record the agent engine keeps at the end of its turns, by the rules
/// of the Claude Code engine (`chat::session_record`).
#[cfg(test)]
mod session_record_tests {
    use super::fake::FakeProvider;
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use nexus_claude::agent::{Cost, CostBasis};

    fn done(usd: Option<f64>) -> AgentEvent {
        AgentEvent::Done {
            stop_reason: nexus_claude::agent::StopReason::Completed,
            subtype: None,
            is_error: false,
            result_text: Some("ok".into()),
            usage: Default::default(),
            cost: Cost {
                usd,
                basis: if usd.is_some() {
                    CostBasis::Priced
                } else {
                    CostBasis::Unknown
                },
            },
            duration_ms: 1,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error: None,
        }
    }

    /// A session of `kind` whose node was created as the manager creates it
    /// (`message_count: 1` for the opening message).
    async fn rig(kind: &str) -> (Arc<MockGraphStore>, FakeProvider, Arc<AgentSessionHandle>) {
        let graph = Arc::new(MockGraphStore::new());
        let mut node = crate::test_helpers::test_chat_session(None);
        node.message_count = 1;
        graph.create_chat_session(&node).await.unwrap();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let runtime = AgentRuntime::new(dyn_graph);
        let provider = FakeProvider::new();
        let handle = runtime
            .adopt(
                &node.id.to_string(),
                "local",
                provider.session(),
                1,
                kind,
                serde_json::json!({}),
                None,
            )
            .await;
        (graph, provider, handle)
    }

    /// A session of `kind` whose provider declares (or not) `compaction_manual`, with its
    /// events subscribed.
    async fn compact_rig(
        kind: &str,
        manual: bool,
    ) -> (
        FakeProvider,
        Arc<AgentSessionHandle>,
        broadcast::Receiver<ChatEvent>,
    ) {
        let graph = Arc::new(MockGraphStore::new());
        let node = crate::test_helpers::test_chat_session(None);
        graph.create_chat_session(&node).await.unwrap();
        let dyn_graph: Arc<dyn GraphStore> = graph;
        let runtime = AgentRuntime::new(dyn_graph);
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().compaction_manual = manual;
        let handle = runtime
            .adopt(
                &node.id.to_string(),
                "local",
                provider.session(),
                1,
                kind,
                serde_json::json!({}),
                None,
            )
            .await;
        let rx = handle.events_tx.subscribe();
        (provider, handle, rx)
    }

    /// The events of `rx` until the session is idle again (bounded).
    async fn until_idle(
        handle: &AgentSessionHandle,
        rx: &mut broadcast::Receiver<ChatEvent>,
    ) -> Vec<ChatEvent> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut seen = Vec::new();
        loop {
            while let Ok(event) = rx.try_recv() {
                seen.push(event);
            }
            if !handle.is_streaming.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(20)).await;
                while let Ok(event) = rx.try_recv() {
                    seen.push(event);
                }
                return seen;
            }
            assert!(Instant::now() < deadline, "the session never went idle");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// ISO3: on a provider that declares `compaction_manual`, `/compact <instructions>`
    /// is a compaction with those instructions, never a turn; the outcome is said.
    #[tokio::test]
    async fn slash_compact_compacts_with_its_instructions_instead_of_sending_a_turn() {
        let (provider, handle, mut rx) = compact_rig("native", true).await;
        handle
            .send_message("/compact keep the API decisions")
            .await
            .unwrap();
        let seen = until_idle(&handle, &mut rx).await;
        assert_eq!(
            *provider.state.compactions.lock().unwrap(),
            vec![Some("keep the API decisions".to_string())]
        );
        assert!(
            provider.state.turns_started.lock().unwrap().is_empty(),
            "/compact reached the model as a turn"
        );
        assert!(seen
            .iter()
            .any(|e| matches!(e, ChatEvent::UserMessage { content }
            if content == "/compact keep the API decisions")));
        assert!(
            seen.iter()
                .any(|e| matches!(e, ChatEvent::SystemHint { content }
                if content.contains("14 message(s) summarised"))),
            "{seen:?}"
        );
        // The session takes the next message as a turn.
        handle.send_message("go on").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "the next turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Without the capability (Codex, ACP...): a typed refusal, no turn, no crash.
    #[tokio::test]
    async fn slash_compact_without_the_capability_is_a_typed_refusal() {
        let (provider, handle, mut rx) = compact_rig("codex", false).await;
        handle.send_message("/compact").await.unwrap();
        let seen = until_idle(&handle, &mut rx).await;
        assert!(provider.state.turns_started.lock().unwrap().is_empty());
        assert!(provider.state.compactions.lock().unwrap().is_empty());
        assert!(
            seen.iter().any(
                |e| matches!(e, ChatEvent::Error { code: Some(c), reason: Some(r), .. }
                if c == "compaction_refused" && r == "compaction_manual")
            ),
            "{seen:?}"
        );
    }

    /// On Claude Code the CLI reads `/compact` itself: it stays a turn, as before.
    #[tokio::test]
    async fn slash_compact_on_claude_code_stays_a_turn_for_the_cli() {
        let (provider, handle, _rx) = compact_rig("claude_code", false).await;
        handle.send_message("/compact keep it short").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "the turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(provider.state.turns_started.lock().unwrap()[0].contains("/compact keep it short"));
        assert!(provider.state.compactions.lock().unwrap().is_empty());
    }

    /// While a turn runs, `/compact` is refused with nexus' code: not queued, and the
    /// turn is not interrupted.
    #[tokio::test]
    async fn slash_compact_during_a_turn_is_refused_without_interrupting_it() {
        let (provider, handle, mut rx) = compact_rig("native", true).await;
        handle.send_message("a long turn").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "the turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        handle.send_message("/compact").await.unwrap();
        assert!(provider.state.interrupts.lock().unwrap().is_empty());
        assert!(handle.queue_snapshot().await.is_empty());
        assert!(provider.state.compactions.lock().unwrap().is_empty());
        provider.state.push(done(None));
        let seen = until_idle(&handle, &mut rx).await;
        assert!(
            seen.iter().any(
                |e| matches!(e, ChatEvent::Error { code: Some(c), reason: Some(r), .. }
                if c == "compaction_refused" && r == "turn_in_progress")
            ),
            "{seen:?}"
        );
        assert_eq!(provider.state.turns_started.lock().unwrap().len(), 1);
    }

    /// A `/compact` the user HELD for after the turn (`queue_message`) runs between two
    /// turns, as a compaction: never played as a turn.
    #[tokio::test]
    async fn a_held_slash_compact_runs_as_a_compaction_once_the_turn_ends() {
        let (provider, handle, mut rx) = compact_rig("native", true).await;
        handle.send_message("a long turn").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "the turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(handle.queue_message("/compact after it").await.unwrap());
        provider.state.push(done(None));
        provider.state.end_turn();
        let seen = until_idle(&handle, &mut rx).await;
        assert_eq!(
            *provider.state.compactions.lock().unwrap(),
            vec![Some("after it".to_string())]
        );
        assert_eq!(provider.state.turns_started.lock().unwrap().len(), 1);
        assert!(
            seen.iter()
                .any(|e| matches!(e, ChatEvent::SystemHint { content }
                if content.starts_with("Conversation compacted on request"))),
            "{seen:?}"
        );
    }

    async fn node(
        graph: &MockGraphStore,
        handle: &AgentSessionHandle,
    ) -> crate::neo4j::models::ChatSessionNode {
        graph
            .get_chat_session(Uuid::parse_str(&handle.session_id).unwrap())
            .await
            .unwrap()
            .unwrap()
    }

    /// Plays one turn that ends on `done(usd)` and waits for the session to be idle.
    async fn turn(
        provider: &FakeProvider,
        handle: &Arc<AgentSessionHandle>,
        text: &str,
        usd: Option<f64>,
    ) {
        let before = provider.state.turns_started.lock().unwrap().len();
        handle.send_message(text).await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().len() == before {
            assert!(Instant::now() < deadline, "the turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        provider.state.push(done(usd));
        while handle.is_streaming.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the turn never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Before P8 the record stayed at `message_count: 1` and `total_cost_usd: None`
    /// whatever the conversation did.
    #[tokio::test]
    async fn each_user_message_counts_and_the_turn_costs_add_up() {
        let (graph, provider, handle) = rig("native").await;
        handle.opening_message_counted();
        turn(&provider, &handle, "opening", Some(0.01)).await;
        let n = node(&graph, &handle).await;
        assert_eq!(
            n.message_count, 1,
            "the opening message counted once: {n:?}"
        );
        assert_eq!(n.total_cost_usd, Some(0.01));

        turn(&provider, &handle, "second", Some(0.02)).await;
        let n = node(&graph, &handle).await;
        assert_eq!(n.message_count, 2, "{n:?}");
        assert!((n.total_cost_usd.unwrap() - 0.03).abs() < 1e-9, "{n:?}");

        // An unknown price changes nothing: never an invented zero.
        turn(&provider, &handle, "third", None).await;
        let n = node(&graph, &handle).await;
        assert_eq!(n.message_count, 3, "{n:?}");
        assert!((n.total_cost_usd.unwrap() - 0.03).abs() < 1e-9, "{n:?}");
    }

    /// Claude Code (forced onto the agent engine) reports the session's total on
    /// each `result`: it is kept as is, as the legacy engine keeps it.
    #[tokio::test]
    async fn the_claude_code_figure_is_the_session_total() {
        let (graph, provider, handle) = rig("claude_code").await;
        turn(&provider, &handle, "one", Some(0.01)).await;
        turn(&provider, &handle, "two", Some(0.03)).await;
        let n = node(&graph, &handle).await;
        assert_eq!(n.total_cost_usd, Some(0.03), "{n:?}");
        assert_eq!(
            n.message_count, 3,
            "1 at creation + two messages not flagged as opening"
        );
    }

    /// A system hint queued behind a turn is played as its own turn, and is not a
    /// user message: it does not count.
    #[tokio::test]
    async fn a_system_hint_is_not_counted() {
        let (graph, provider, handle) = rig("native").await;
        let deadline = Instant::now() + Duration::from_secs(5);
        handle.send_message("one").await.unwrap();
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "the turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        handle.inject_hint("a hint").await.unwrap();
        provider.state.push(done(Some(0.0)));
        while provider.state.turns_started.lock().unwrap().len() < 2 {
            assert!(Instant::now() < deadline, "the hint never played");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        provider.state.push(done(Some(0.0)));
        while handle.is_streaming.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the run never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let n = node(&graph, &handle).await;
        assert_eq!(
            n.message_count, 2,
            "1 at creation + one user message: {n:?}"
        );
        assert_eq!(n.total_cost_usd, Some(0.0), "a free turn is a real 0");
    }
}

/// A turn of the agent engine that fails before showing anything is retried as the
/// chat's `RetryConfig` says — the figures the Claude Code engine uses — not by a
/// constant of its own.
#[cfg(test)]
mod retry_config_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn overloaded() -> AgentEvent {
        AgentEvent::Done {
            stop_reason: nexus_claude::agent::StopReason::Error,
            subtype: None,
            is_error: true,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 0,
            duration_api_ms: None,
            num_turns: 0,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error: Some(ProviderError::Overloaded),
        }
    }

    #[tokio::test]
    async fn the_retries_follow_the_chat_retry_config() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
        handle.configure_retry(RetryConfig {
            max_attempts: 1,
            initial_delay_ms: 7,
            backoff_multiplier: 2.0,
        });
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("hello").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut attempts_seen = 0;
        loop {
            let turns = provider.state.turns_started.lock().unwrap().len();
            if turns > attempts_seen {
                attempts_seen = turns;
                provider.state.push(overloaded());
            }
            if !handle.is_streaming.load(Ordering::SeqCst) && attempts_seen > 0 {
                break;
            }
            assert!(Instant::now() < deadline, "the turn never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            attempts_seen, 2,
            "the turn, then ONE retry (max_attempts = 1)"
        );
        let mut retrying = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let ChatEvent::Retrying {
                attempt,
                max_attempts,
                delay_ms,
                ..
            } = event
            {
                retrying.push((attempt, max_attempts, delay_ms));
            }
        }
        assert_eq!(retrying, [(1, 1, 7)]);
    }
}

/// A provider instance that goes down and comes back: the turn that touches it is
/// held (never closed on "the endpoint could not be reached"), and it resumes when
/// the instance answers — without replaying anything the turn had already run.
#[cfg(test)]
mod outage_hold_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn unreachable() -> ProviderError {
        ProviderError::EndpointUnreachable {
            detail: "connection failed: Connection refused (os error 61)".to_string(),
        }
    }

    fn outage_event() -> AgentEvent {
        AgentEvent::Error {
            error: unreachable(),
        }
    }

    fn done(error: Option<ProviderError>) -> AgentEvent {
        AgentEvent::Done {
            stop_reason: nexus_claude::agent::StopReason::Completed,
            subtype: None,
            is_error: error.is_some(),
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 0,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error,
        }
    }

    fn tool_call() -> AgentEvent {
        AgentEvent::ToolCall {
            id: "t1".into(),
            name: "Bash".into(),
            input: serde_json::json!({ "command": "true" }),
            category: nexus_claude::agent::ToolCategory::Other,
            canonical: None,
            input_complete: true,
            seq: None,
            parent: None,
        }
    }

    /// A session on the fake provider whose ordinary retry budget is spent at once —
    /// so the HOLD is what keeps the turn alive — with one failure declaring the
    /// instance down and a short hold window.
    async fn rig() -> (FakeProvider, Arc<AgentSessionHandle>) {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
        handle.configure_retry(RetryConfig {
            max_attempts: 0,
            initial_delay_ms: 1,
            backoff_multiplier: 1.0,
        });
        handle.configure_outage(
            OutagePolicy::new(1, Duration::from_secs(30)),
            HoldPolicy::new(Duration::from_secs(30), Duration::from_millis(5)),
        );
        (provider, handle)
    }

    /// The phases of the `ProviderOutage` events emitted so far, in order.
    fn transitions(rx: &mut broadcast::Receiver<ChatEvent>) -> Vec<String> {
        let mut phases = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let ChatEvent::ProviderOutage { phase, .. } = event {
                phases.push(phase);
            }
        }
        phases
    }

    /// The instance fails every attempt until the test lets it answer: the turn is
    /// held across the failures and resumes when it does.
    #[tokio::test]
    async fn a_down_instance_holds_the_turn_until_it_answers() {
        let (provider, handle) = rig().await;
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("hello").await.unwrap();
        const FAILURES: usize = 2;
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = 0usize;
        loop {
            let turns = provider.state.turns_started.lock().unwrap().len();
            if turns > seen {
                seen = turns;
                if seen <= FAILURES {
                    provider.state.push(outage_event());
                } else {
                    provider.state.push(done(None));
                }
            }
            if !handle.is_streaming.load(Ordering::SeqCst) && seen > 0 {
                break;
            }
            assert!(Instant::now() < deadline, "the turn never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            seen,
            FAILURES + 1,
            "the held turn was probed until the instance answered"
        );
        assert_eq!(
            transitions(&mut rx),
            ["down", "up"],
            "one notice per transition, never per failed call"
        );
    }

    /// A probe the provider *refuses at the send* does not close a held turn: it is
    /// probed again — the hold's own budget is what bounds it.
    #[tokio::test]
    async fn a_refused_probe_does_not_close_a_held_turn() {
        let (provider, handle) = rig().await;
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("hello").await.unwrap();
        // The turn is open; its first probe is refused before any stream exists.
        *provider.state.fail_next_turn.lock().unwrap() = Some(unreachable());
        provider.state.push(outage_event());
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = 0usize;
        loop {
            let turns = provider.state.turns_started.lock().unwrap().len();
            if turns > seen {
                seen = turns;
                if seen >= 2 {
                    provider.state.push(done(None));
                }
            }
            if !handle.is_streaming.load(Ordering::SeqCst) && seen >= 2 {
                break;
            }
            assert!(Instant::now() < deadline, "the turn never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(seen, 2, "the refused probe was retried, not dropped");
        assert_eq!(transitions(&mut rx), ["down", "up"]);
    }

    /// A turn that already called a tool is closed by its failure like any other: the
    /// hold never extends to it, so no tool is ever replayed.
    #[tokio::test]
    async fn a_turn_that_ran_a_tool_is_never_held() {
        let (provider, handle) = rig().await;
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("hello").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "the turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        provider.state.push(tool_call());
        provider.state.push(outage_event());
        provider.state.push(done(Some(unreachable())));
        let deadline = Instant::now() + Duration::from_secs(10);
        while handle.is_streaming.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the turn never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            provider.state.turns_started.lock().unwrap().len(),
            1,
            "a turn that ran a tool is never replayed"
        );
        assert!(
            transitions(&mut rx).is_empty(),
            "a turn that already ran a tool is not even a retryable call"
        );
    }
}

/// The end of a turn of the agent engine: what the Claude Code engine's
/// post-stream does — the tool calls a Stop left are said cancelled, the host's
/// `after_turn` sees what the turn did, its events go out and its hints are played.
#[cfg(test)]
mod after_turn_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use std::sync::Mutex as StdMutex;

    fn done() -> AgentEvent {
        AgentEvent::Done {
            stop_reason: nexus_claude::agent::StopReason::Completed,
            subtype: None,
            is_error: false,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 0,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error: None,
        }
    }

    fn tool_call(id: &str, name: &str) -> AgentEvent {
        AgentEvent::ToolCall {
            id: id.into(),
            name: name.into(),
            input: serde_json::json!({ "file_path": "src/lib.rs" }),
            category: nexus_claude::agent::ToolCategory::Other,
            canonical: None,
            input_complete: true,
            seq: None,
            parent: None,
        }
    }

    /// Records what `after_turn` was told and answers one hint, once.
    #[derive(Default)]
    struct Host {
        seen: StdMutex<Vec<TurnOutcome>>,
    }

    #[async_trait::async_trait]
    impl TurnServices for Host {
        async fn prepare(
            &self,
            _session_id: &str,
            _shown: &str,
            sent: &str,
            _turn: &crate::refs::turn::TurnExpansion,
        ) -> String {
            sent.to_string()
        }
        async fn continuation(&self, _session_id: &str) -> String {
            String::new()
        }
        async fn after_turn(&self, _session_id: &str, outcome: &TurnOutcome) -> AfterTurn {
            let first = {
                let mut seen = self.seen.lock().unwrap();
                seen.push(outcome.clone());
                seen.len() == 1
            };
            if !first {
                return AfterTurn::default();
            }
            AfterTurn {
                events: vec![ChatEvent::CompactionRecovery {
                    hint_tokens: 1,
                    build_latency_ms: 0,
                    recovery_success: true,
                }],
                hints: vec!["THE-HINT".into()],
            }
        }
    }

    async fn rig(
        services: Option<Arc<dyn TurnServices>>,
    ) -> (FakeProvider, Arc<AgentSessionHandle>) {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                services,
            )
            .await;
        (provider, handle)
    }

    async fn until(what: &str, check: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !check() {
            assert!(Instant::now() < deadline, "{what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn the_host_sees_the_turn_and_its_hint_is_played_next() {
        let host = Arc::new(Host::default());
        let (provider, handle) = rig(Some(host.clone() as Arc<dyn TurnServices>)).await;
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("go").await.unwrap();
        until("the turn started", || {
            !provider.state.turns_started.lock().unwrap().is_empty()
        })
        .await;
        provider.state.push(AgentEvent::Text {
            text: "all done".into(),
            seq: None,
            parent: None,
        });
        provider.state.push(tool_call("t1", "Edit"));
        provider.state.push(AgentEvent::ToolResult {
            id: "t1".into(),
            output: None,
            is_error: false,
            seq: None,
            parent: None,
        });
        provider.state.push(AgentEvent::Compaction {
            phase: nexus_claude::agent::CompactionPhase::Completed,
            trigger: None,
            pre_tokens: Some(10),
        });
        provider.state.push(done());
        until("the hint was played", || {
            provider.state.turns_started.lock().unwrap().len() == 2
        })
        .await;
        assert_eq!(provider.state.turns_started.lock().unwrap()[1], "THE-HINT");
        provider.state.push(done());
        until("the run ended", || {
            !handle.is_streaming.load(Ordering::SeqCst)
        })
        .await;

        let seen = host.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "after each turn played: {seen:?}");
        assert_eq!(seen[0].assistant_text, "all done");
        assert_eq!(seen[0].tools.len(), 1);
        assert_eq!(seen[0].tools[0].0, "Edit");
        assert!(seen[0].compacted, "{:?}", seen[0]);
        assert!(!seen[0].interrupted);
        assert!(!seen[1].compacted, "the hint's turn did not compact");
        let mut recovered = false;
        while let Ok(event) = rx.try_recv() {
            recovered |= matches!(event, ChatEvent::CompactionRecovery { .. });
        }
        assert!(recovered, "the host's event went out");
    }

    #[tokio::test]
    async fn a_stop_says_the_tools_left_without_result_cancelled() {
        let (provider, handle) = rig(None).await;
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("go").await.unwrap();
        until("the turn started", || {
            !provider.state.turns_started.lock().unwrap().is_empty()
        })
        .await;
        provider.state.push(tool_call("t-done", "Read"));
        provider.state.push(AgentEvent::ToolResult {
            id: "t-done".into(),
            output: None,
            is_error: false,
            seq: None,
            parent: None,
        });
        provider.state.push(tool_call("t-left", "Bash"));
        until("the tool call went out", || {
            handle
                .streaming_events
                .try_lock()
                .map(|e| {
                    e.iter()
                        .any(|e| matches!(e, ChatEvent::ToolUse { id, .. } if id == "t-left"))
                })
                .unwrap_or(false)
        })
        .await;
        // The fake ends the turn stream on an interrupt, as a provider does.
        handle.interrupt().await.unwrap();
        until("the turn ended", || {
            !handle.is_streaming.load(Ordering::SeqCst)
        })
        .await;
        let mut cancelled = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let ChatEvent::ToolCancelled { id, .. } = event {
                cancelled.push(id);
            }
        }
        assert_eq!(cancelled, ["t-left"], "only the tool left without a result");
    }

    /// F-R4: a turn with an image goes out with an image, and the model PO routes it to is
    /// made active BEFORE it is sent (the provider checks the active model's vision when
    /// the turn is sent, before any hook of the turn runs).
    struct ImageHost {
        state: Arc<super::fake::FakeState>,
        turns_seen_when_asked: StdMutex<Option<usize>>,
        route_to: Option<&'static str>,
        not_applied: StdMutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl TurnServices for ImageHost {
        async fn prepare(
            &self,
            _session_id: &str,
            _shown: &str,
            sent: &str,
            _turn: &crate::refs::turn::TurnExpansion,
        ) -> String {
            sent.to_string()
        }
        async fn continuation(&self, _session_id: &str) -> String {
            String::new()
        }
        async fn images(
            &self,
            _shown: &str,
        ) -> std::result::Result<Vec<super::super::message_attachments::AttachedImage>, String>
        {
            Ok(vec![super::super::message_attachments::AttachedImage {
                media_type: "image/png".into(),
                data_base64: "AAAA".into(),
                source: super::super::message_attachments::MessageAttachment {
                    id: uuid::Uuid::new_v4(),
                    filename: "x.png".into(),
                    mime_type: "image/png".into(),
                    size_bytes: 4,
                },
            }])
        }
        async fn model_for_images(&self, _session_id: &str) -> Option<String> {
            *self.turns_seen_when_asked.lock().unwrap() =
                Some(self.state.turns_started.lock().unwrap().len());
            self.route_to.map(str::to_string)
        }
        async fn image_model_not_applied(&self, _session_id: &str, model: &str) {
            self.not_applied.lock().unwrap().push(model.to_string());
        }
    }

    async fn image_turn(route_to: Option<&'static str>) -> (FakeProvider, Arc<ImageHost>) {
        image_turn_on(FakeProvider::new(), route_to).await
    }

    async fn image_turn_on(
        provider: FakeProvider,
        route_to: Option<&'static str>,
    ) -> (FakeProvider, Arc<ImageHost>) {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let host = Arc::new(ImageHost {
            state: Arc::clone(&provider.state),
            turns_seen_when_asked: StdMutex::new(None),
            route_to,
            not_applied: StdMutex::new(Vec::new()),
        });
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                Some(host.clone() as Arc<dyn TurnServices>),
            )
            .await;
        handle
            .send_message("what is on this picture?")
            .await
            .unwrap();
        until("the turn started", || {
            !provider.state.turns_started.lock().unwrap().is_empty()
        })
        .await;
        (provider, host)
    }

    #[tokio::test]
    async fn an_image_turn_runs_on_the_model_po_routes_it_to_made_active_before_it_is_sent() {
        let (provider, host) = image_turn(Some("vision")).await;
        assert_eq!(*host.turns_seen_when_asked.lock().unwrap(), Some(0));
        assert_eq!(*provider.state.models.lock().unwrap(), ["vision"]);
    }

    /// A model that could not be made active is handed back to the host, which undoes what
    /// it decided for the turn (`TurnRouter::forget_image_change`).
    #[tokio::test]
    async fn a_model_that_could_not_be_made_active_is_handed_back_to_the_host() {
        let provider = FakeProvider::new();
        *provider.state.fail_set_model.lock().unwrap() = Some(ProviderError::Closed);
        let (provider, host) = image_turn_on(provider, Some("vision")).await;
        assert!(provider.state.models.lock().unwrap().is_empty());
        assert_eq!(*host.not_applied.lock().unwrap(), ["vision"]);
    }

    #[tokio::test]
    async fn an_image_turn_with_nothing_to_route_keeps_its_model() {
        let (provider, _host) = image_turn(None).await;
        assert!(provider.state.models.lock().unwrap().is_empty());
    }

    /// A host whose end of turn never finishes: the runtime's backstop skips it.
    struct StuckHost;

    #[async_trait::async_trait]
    impl TurnServices for StuckHost {
        async fn prepare(
            &self,
            _session_id: &str,
            _shown: &str,
            sent: &str,
            _turn: &crate::refs::turn::TurnExpansion,
        ) -> String {
            sent.to_string()
        }
        async fn continuation(&self, _session_id: &str) -> String {
            String::new()
        }
        async fn after_turn(&self, _session_id: &str, _outcome: &TurnOutcome) -> AfterTurn {
            std::future::pending::<AfterTurn>().await
        }
    }

    /// An `after_turn` that never returns is skipped and announced as a recoverable
    /// system note, never as an `Error` that would read as a failed turn (B5).
    #[tokio::test]
    async fn an_abandoned_after_turn_is_a_recoverable_signal_not_an_error() {
        let host = Arc::new(StuckHost);
        let (_provider, handle) = rig(Some(host as Arc<dyn TurnServices>)).await;
        let mut rx = handle.events_tx.subscribe();
        handle
            .after_turn_bounded(&TurnOutcome::default(), Duration::from_millis(20))
            .await;
        let mut note = false;
        let mut error = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                ChatEvent::SystemHint { .. } => note = true,
                ChatEvent::Error { .. } => error = true,
                _ => {}
            }
        }
        assert!(note, "the skipped wrap-up is said as a system note");
        assert!(!error, "no error event for a recoverable skip");
    }
}

#[cfg(test)]
mod retry_stop_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    /// A Stop during the pause before a retry ends the turn at once: no ten-second
    /// wait, no retry sent after the Stop.
    #[tokio::test]
    async fn a_stop_cuts_the_pause_before_a_retry() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
        handle.configure_retry(RetryConfig {
            max_attempts: 3,
            initial_delay_ms: 10_000,
            backoff_multiplier: 1.0,
        });
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("hello").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        provider.state.push(AgentEvent::Error {
            error: ProviderError::Overloaded,
        });
        // The pause has begun once `retrying` is out.
        loop {
            let event = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("retrying")
                .unwrap();
            if matches!(event, ChatEvent::Retrying { .. }) {
                break;
            }
        }
        let stopped = Instant::now();
        handle.interrupt().await.unwrap();
        while handle.is_streaming.load(Ordering::SeqCst) {
            assert!(
                stopped.elapsed() < Duration::from_secs(2),
                "the Stop waited out the pause"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            provider.state.turns_started.lock().unwrap().len(),
            1,
            "no retry after the Stop"
        );
        // The turn ended on its terminal event, as any stopped turn.
        let mut ended = false;
        while let Ok(event) = rx.try_recv() {
            ended |= matches!(
                event,
                ChatEvent::Result { ref stop_reason, .. } if stop_reason.as_deref() == Some("interrupted")
            );
        }
        assert!(ended, "a result with stop_reason interrupted");
    }
}

/// What a provider process that exits does to its live handle (B4): the handle is
/// dropped from the runtime so the next message resumes the session instead of
/// re-failing on a dead process.
#[cfg(test)]
mod process_exit_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    /// A `ProcessExited` terminal event ends the turn, is said as a session error,
    /// and removes the handle from the runtime: the next message takes the resume
    /// path rather than re-failing on the dead process.
    #[tokio::test]
    async fn a_process_exit_removes_the_dead_handle_so_the_next_message_resumes() {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                "native",
                serde_json::json!({}),
                None,
            )
            .await;
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("go").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "the turn never started");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        provider.state.push(AgentEvent::Error {
            error: ProviderError::ProcessExited { code: Some(1) },
        });
        // The turn ends once the terminal error is out; the handle detaches itself.
        let deadline = Instant::now() + Duration::from_secs(5);
        while handle.is_streaming.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the turn never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            runtime.get("not-a-uuid").await.is_none(),
            "the dead handle must leave the registry"
        );
        // The client was told it was the session (not a plain in-stream error).
        let mut session_error = false;
        while let Ok(event) = rx.try_recv() {
            session_error |= matches!(event, ChatEvent::SessionError { .. });
        }
        assert!(
            session_error,
            "the process exit was said as a session error"
        );
    }
}

/// What waits for the next turn after a compaction is dropped only when a turn
/// carrying it was ANSWERED (`TurnOutcome::answered`): a provider `done` without
/// error, `interrupted` included (the provider kept the turn). Not a turn accepted
/// then failed (the native harness accepts before any request), not one whose
/// retries ran out, not one stopped during the pause before a retry.
#[cfg(test)]
mod answered_tests {
    use super::fake::FakeProvider;
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use std::sync::Mutex as StdMutex;

    /// The terminal-result fields of one turn, captured for assertions.
    #[derive(Clone)]
    struct OutcomeFields {
        stop_reason: Option<String>,
        subtype: String,
        is_error: bool,
        error_code: Option<String>,
    }

    /// The `answered` of each turn the host saw end, plus its terminal result.
    #[derive(Default)]
    struct Host {
        answered: StdMutex<Vec<bool>>,
        outcomes: StdMutex<Vec<OutcomeFields>>,
    }

    #[async_trait::async_trait]
    impl TurnServices for Host {
        async fn prepare(
            &self,
            _session_id: &str,
            _shown: &str,
            sent: &str,
            _turn: &crate::refs::turn::TurnExpansion,
        ) -> String {
            sent.to_string()
        }
        async fn continuation(&self, _session_id: &str) -> String {
            String::new()
        }
        async fn after_turn(&self, _session_id: &str, outcome: &TurnOutcome) -> AfterTurn {
            self.answered.lock().unwrap().push(outcome.answered);
            self.outcomes.lock().unwrap().push(OutcomeFields {
                stop_reason: outcome.stop_reason.clone(),
                subtype: outcome.subtype.clone(),
                is_error: outcome.is_error,
                error_code: outcome.error_code.clone(),
            });
            AfterTurn::default()
        }
    }

    impl Host {
        fn seen(&self) -> Vec<bool> {
            self.answered.lock().unwrap().clone()
        }

        fn seen_outcome(&self) -> Vec<OutcomeFields> {
            self.outcomes.lock().unwrap().clone()
        }
    }

    fn done(is_error: bool, error: Option<ProviderError>) -> AgentEvent {
        AgentEvent::Done {
            stop_reason: if is_error {
                nexus_claude::agent::StopReason::Error
            } else {
                nexus_claude::agent::StopReason::Completed
            },
            subtype: None,
            is_error,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 0,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error,
        }
    }

    fn limited() -> AgentEvent {
        done(
            true,
            Some(ProviderError::RateLimited {
                retry_after_ms: Some(5),
                resets_at_ms: None,
            }),
        )
    }

    async fn session(retries: u32) -> (FakeProvider, Arc<AgentSessionHandle>, Arc<Host>) {
        session_of("native", retries).await
    }

    async fn session_of(
        kind: &str,
        retries: u32,
    ) -> (FakeProvider, Arc<AgentSessionHandle>, Arc<Host>) {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let host = Arc::new(Host::default());
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                kind,
                serde_json::json!({}),
                Some(host.clone() as Arc<dyn TurnServices>),
            )
            .await;
        handle.configure_retry(RetryConfig {
            max_attempts: retries,
            initial_delay_ms: 5,
            backoff_multiplier: 1.0,
        });
        (provider, handle, host)
    }

    /// Sends a message and answers each attempt of its turn with the next of `ends`,
    /// until the session is idle again.
    async fn play(
        provider: &FakeProvider,
        handle: &Arc<AgentSessionHandle>,
        ends: Vec<AgentEvent>,
    ) {
        let before = provider.state.turns_started.lock().unwrap().len();
        handle.send_message("go").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut ends = ends.into_iter();
        let mut answered = before;
        loop {
            let started = provider.state.turns_started.lock().unwrap().len();
            if started > answered {
                answered = started;
                provider
                    .state
                    .push(ends.next().expect("an end for each attempt"));
            }
            if answered > before && !handle.is_streaming.load(Ordering::SeqCst) {
                break;
            }
            assert!(Instant::now() < deadline, "the turn never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(ends.next().is_none(), "every attempt was played");
    }

    #[tokio::test]
    async fn a_turn_accepted_then_failed_is_not_answered() {
        let (provider, handle, host) = session(0).await;
        // `send_turn` accepted it, then the endpoint refused it for good.
        play(
            &provider,
            &handle,
            vec![done(true, Some(ProviderError::Unauthorized))],
        )
        .await;
        play(&provider, &handle, vec![done(false, None)]).await;
        assert_eq!(host.seen(), [false, true], "a failed turn keeps it");
    }

    /// A Stop the provider answered (`done interrupted`): nexus' native loop pushed
    /// the user message before the run and keeps it, so the context was delivered.
    #[tokio::test]
    async fn a_turn_the_provider_ended_interrupted_is_answered() {
        let (provider, handle, host) = session(0).await;
        play(&provider, &handle, vec![interrupted_done()]).await;
        assert_eq!(host.seen(), [true]);
    }

    /// R0: the terminal result of a turn is captured from its `done`, so the
    /// manager's `after_turn` can feed the routing bandit (stop_reason/subtype/
    /// is_error/error_code) instead of reading them empty.
    #[tokio::test]
    async fn a_completed_turn_populates_its_terminal_result() {
        let (provider, handle, host) = session(0).await;
        play(&provider, &handle, vec![done(false, None)]).await;
        let fields = &host.seen_outcome()[0];
        assert_eq!(fields.stop_reason.as_deref(), Some("completed"));
        assert_eq!(fields.subtype.as_str(), "success");
        assert!(!fields.is_error);
        assert!(fields.error_code.is_none());
    }

    #[tokio::test]
    async fn a_failed_turn_carries_its_typed_error_code() {
        let (provider, handle, host) = session(0).await;
        play(
            &provider,
            &handle,
            vec![done(true, Some(ProviderError::Unauthorized))],
        )
        .await;
        let fields = &host.seen_outcome()[0];
        assert_eq!(fields.stop_reason.as_deref(), Some("error"));
        assert_eq!(fields.subtype.as_str(), "error_during_execution");
        assert!(fields.is_error);
        assert_eq!(fields.error_code.as_deref(), Some("unauthorized"));
    }

    /// A provider not known to keep an interrupted turn (`keeps_interrupted_turns`):
    /// its `done interrupted` leaves the context for the next turn — a bounded
    /// duplicate if it did keep it, never a loss.
    #[tokio::test]
    async fn a_turn_an_unverified_provider_ended_interrupted_is_not_answered() {
        for kind in ["codex", "acp", "claude_code"] {
            let (provider, handle, host) = session_of(kind, 0).await;
            play(&provider, &handle, vec![interrupted_done()]).await;
            play(&provider, &handle, vec![done(false, None)]).await;
            assert_eq!(host.seen(), [false, true], "{kind}");
        }
    }

    #[tokio::test]
    async fn a_retried_turn_then_answered_is_answered_once() {
        let (provider, handle, host) = session(2).await;
        play(&provider, &handle, vec![limited(), done(false, None)]).await;
        assert_eq!(
            provider.state.turns_started.lock().unwrap().len(),
            2,
            "the turn and its retry"
        );
        assert_eq!(host.seen(), [true], "one turn, answered once");
    }

    #[tokio::test]
    async fn a_turn_whose_retries_ran_out_is_not_answered() {
        let (provider, handle, host) = session(1).await;
        play(&provider, &handle, vec![limited(), limited()]).await;
        assert_eq!(host.seen(), [false]);
    }

    /// The `done interrupted` the runtime makes for a Stop during the pause before a
    /// retry ends the turn, but no attempt of it was answered: the context stays.
    #[tokio::test]
    async fn a_stop_during_the_pause_before_a_retry_is_not_answered() {
        let (provider, handle, host) = session(3).await;
        handle.configure_retry(RetryConfig {
            max_attempts: 3,
            initial_delay_ms: 10_000,
            backoff_multiplier: 1.0,
        });
        let mut rx = handle.events_tx.subscribe();
        handle.send_message("go").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while provider.state.turns_started.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        provider
            .state
            .push(done(true, Some(ProviderError::Overloaded)));
        loop {
            let event = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("retrying")
                .unwrap();
            if matches!(event, ChatEvent::Retrying { .. }) {
                break;
            }
        }
        handle.interrupt().await.unwrap();
        while handle.is_streaming.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(provider.state.turns_started.lock().unwrap().len(), 1);
        assert_eq!(host.seen(), [false]);
    }

    fn text(content: &str) -> AgentEvent {
        AgentEvent::Text {
            text: content.to_string(),
            seq: None,
            parent: None,
        }
    }

    /// Drives one run whose attempts each yield the next of `turns` (a turn = the
    /// events its stream yields, in order), then returns every event the client saw.
    async fn play_events(
        provider: &FakeProvider,
        handle: &Arc<AgentSessionHandle>,
        turns: Vec<Vec<AgentEvent>>,
    ) -> Vec<ChatEvent> {
        let mut rx = handle.events_tx.subscribe();
        let mut answered = provider.state.turns_started.lock().unwrap().len();
        handle.send_message("go").await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        for turn in turns {
            while provider.state.turns_started.lock().unwrap().len() == answered {
                assert!(Instant::now() < deadline, "an attempt never started");
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            answered = provider.state.turns_started.lock().unwrap().len();
            for event in turn {
                provider.state.push(event);
            }
        }
        while handle.is_streaming.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the run never ended");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        // The final `streaming_status { false }` is broadcast right after the flag
        // flips; a short settle lets it and any tail land in the receiver.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let mut seen = Vec::new();
        while let Ok(event) = rx.try_recv() {
            seen.push(event);
        }
        seen
    }

    /// A2/D1: partial text then a transient 429 is retried — the prefix stays and
    /// the assistant content resumes on the same turn, which answers once.
    #[tokio::test]
    async fn partial_text_then_a_rate_limit_is_retried_and_resumes() {
        let (provider, handle, host) = session(1).await;
        let seen = play_events(
            &provider,
            &handle,
            vec![
                vec![
                    text("Hel"),
                    text("lo"),
                    done(
                        true,
                        Some(ProviderError::RateLimited {
                            retry_after_ms: Some(5),
                            resets_at_ms: None,
                        }),
                    ),
                ],
                vec![text(" world"), done(false, None)],
            ],
        )
        .await;
        assert_eq!(
            provider.state.turns_started.lock().unwrap().len(),
            2,
            "the turn and its retry"
        );
        assert_eq!(host.seen(), [true], "one turn, answered once");
        let assistant: String = seen
            .into_iter()
            .filter_map(|event| match event {
                ChatEvent::AssistantText { content, .. } => Some(content),
                _ => None,
            })
            .collect();
        assert_eq!(
            assistant, "Hello world",
            "the prefix stays and the turn resumes"
        );
    }

    /// A session whose runtime reports quota to `registry` (QUOTA), instance id `local`.
    async fn session_with_quota(
        registry: Arc<crate::chat::provider::quota::QuotaRegistry>,
        retries: u32,
    ) -> (FakeProvider, Arc<AgentSessionHandle>, Arc<Host>) {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        runtime.set_quota(registry);
        let provider = FakeProvider::new();
        let host = Arc::new(Host::default());
        let handle = runtime
            .adopt(
                "not-a-uuid",
                "local",
                provider.session(),
                1,
                "claude_code",
                serde_json::json!({}),
                Some(host.clone() as Arc<dyn TurnServices>),
            )
            .await;
        handle.configure_retry(RetryConfig {
            max_attempts: retries,
            initial_delay_ms: 5,
            backoff_multiplier: 1.0,
        });
        (provider, handle, host)
    }

    /// QUOTA: what the provider says about the limits of its account (Claude Code's
    /// `rate_limit_event`, mapped by nexus to `AgentEvent::Quota`) reaches the registry the router
    /// reads, in the middle of a turn that goes on and ends normally; a `rejected` one holds the
    /// instance out until its reset.
    #[tokio::test]
    async fn a_quota_event_of_the_provider_reaches_the_registry_and_the_turn_goes_on() {
        use crate::chat::provider::limits::ManualClock;
        use nexus_claude::agent::{QuotaSnapshot, QuotaStatus, QuotaWindow};
        let clock = Arc::new(ManualClock::at("2026-10-21T10:00:00Z"));
        let registry =
            Arc::new(crate::chat::provider::quota::QuotaRegistry::new().with_clock(clock));
        let (provider, handle, host) = session_with_quota(registry.clone(), 0).await;
        let reset = 1_792_836_000_000_u64; // 2026-10-24T10:00:00Z
        play_events(
            &provider,
            &handle,
            vec![vec![
                AgentEvent::Quota {
                    quota: QuotaSnapshot::live(vec![QuotaWindow {
                        kind: "seven_day".into(),
                        utilization: Some(1.0),
                        resets_at_ms: Some(reset),
                        status: QuotaStatus::Exhausted,
                    }]),
                },
                text("still answered"),
                done(false, None),
            ]],
        )
        .await;
        assert_eq!(host.seen(), [true], "the turn went on and was answered");
        let held = registry
            .exhausted("local")
            .expect("the instance is held out");
        assert_eq!(held.resets_at_ms, reset);
        assert_eq!(held.window.as_deref(), Some("seven_day"));
    }

    /// QUOTA: a rate limit that names a reset three days away is not run again (it would meet the
    /// same refusal), is said to the person with the instant it serves again, and marks the
    /// instance — but it is not the model failing: the turn's ending feeds no failure to the arm.
    #[tokio::test]
    async fn a_rate_limit_with_a_far_reset_ends_the_turn_says_when_and_marks_the_instance() {
        let registry = Arc::new(crate::chat::provider::quota::QuotaRegistry::new());
        let (provider, handle, host) = session_with_quota(registry.clone(), 3).await;
        let reset = wall_clock_ms() + 3 * 86_400_000;
        let seen = play_events(
            &provider,
            &handle,
            vec![vec![done(
                true,
                Some(ProviderError::RateLimited {
                    retry_after_ms: None,
                    resets_at_ms: Some(reset),
                }),
            )]],
        )
        .await;
        assert_eq!(
            provider.state.turns_started.lock().unwrap().len(),
            1,
            "three retries were allowed, none is useful against a quota used up"
        );
        assert_eq!(host.seen(), [false]);
        assert_eq!(registry.exhausted("local").unwrap().resets_at_ms, reset);
        let message = seen
            .iter()
            .find_map(|event| match event {
                ChatEvent::Result {
                    is_error: true,
                    error_message,
                    ..
                } => error_message.clone(),
                _ => None,
            })
            .expect("the failure is said on the wire");
        assert!(message.contains("serves again at"), "{message}");
        assert!(message.contains("nothing is switched for you"), "{message}");
    }

    /// A2/D1: partial text then a non-retryable failure keeps the old behaviour —
    /// the turn ends in error and is not replayed.
    #[tokio::test]
    async fn partial_text_then_a_non_retryable_error_still_ends() {
        let (provider, handle, host) = session(2).await;
        let seen = play_events(
            &provider,
            &handle,
            vec![vec![
                text("Hel"),
                done(true, Some(ProviderError::Unauthorized)),
            ]],
        )
        .await;
        assert_eq!(
            provider.state.turns_started.lock().unwrap().len(),
            1,
            "no retry for a failure that is not transient"
        );
        assert_eq!(
            host.seen(),
            [false],
            "a failed-for-good turn is not answered"
        );
        assert!(
            seen.iter()
                .any(|event| matches!(event, ChatEvent::Result { is_error: true, .. })),
            "the failure is said on the wire"
        );
    }

    /// A2/D1: a turn that already ran a tool is never replayed, whatever the error —
    /// the tool may have had a side effect.
    #[tokio::test]
    async fn a_turn_that_already_ran_a_tool_is_not_retried() {
        let (provider, handle, host) = session(2).await;
        let tool_call = AgentEvent::ToolCall {
            id: "t1".to_string(),
            name: "Bash".to_string(),
            input: serde_json::json!({ "command": "ls" }),
            category: Default::default(),
            canonical: None,
            input_complete: true,
            seq: None,
            parent: None,
        };
        let tool_result = AgentEvent::ToolResult {
            id: "t1".to_string(),
            output: None,
            is_error: false,
            seq: None,
            parent: None,
        };
        let seen = play_events(
            &provider,
            &handle,
            vec![vec![
                text("Hel"),
                tool_call,
                tool_result,
                done(
                    true,
                    Some(ProviderError::RateLimited {
                        retry_after_ms: Some(5),
                        resets_at_ms: None,
                    }),
                ),
            ]],
        )
        .await;
        assert_eq!(
            provider.state.turns_started.lock().unwrap().len(),
            1,
            "no replay after a side-effecting tool"
        );
        assert_eq!(host.seen(), [false]);
        assert!(
            seen.iter()
                .any(|event| matches!(event, ChatEvent::Result { is_error: true, .. })),
            "the failure is said on the wire"
        );
    }
}

#[cfg(test)]
mod tool_timing_tests {
    use super::fake::FakeProvider;
    use super::*;
    use crate::neo4j::traits::GraphStore as _;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    use serde_json::json;

    /// The provider is slow to take the answer and the tool is fast: its result is
    /// emitted before the decision event. The answer was put on the clock before it
    /// was handed over, so the timing still has it, and the run starts there.
    #[tokio::test]
    async fn a_permission_answer_is_timed_even_when_the_result_overtakes_its_event() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph.clone());
        let provider = FakeProvider::new();
        let sid = Uuid::new_v4().to_string();
        let handle = runtime
            .adopt(
                &sid,
                "claude-code",
                provider.session(),
                1,
                "native",
                json!({}),
                None,
            )
            .await;
        handle
            .emit(ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "Bash".into(),
                input: json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            })
            .await;
        handle
            .emit(ChatEvent::PermissionRequest {
                id: "req-1".into(),
                tool: "Bash".into(),
                input: json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
                tool_use_id: Some("t1".into()),
                always: None,
                consent: None,
            })
            .await;
        let hold = Arc::new(tokio::sync::Notify::new());
        *provider.state.hold_answers.lock().unwrap() = Some(Arc::clone(&hold));
        let answering = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move { handle.answer_permission("req-1", true).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        handle
            .emit(ChatEvent::ToolResult {
                id: "t1".into(),
                result: json!("a.rs"),
                is_error: false,
                parent_tool_use_id: None,
            })
            .await;
        hold.notify_one();
        answering.await.unwrap().unwrap();

        let stored = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 50)
            .await
            .unwrap();
        let types: Vec<&str> = stored.iter().map(|r| r.event_type.as_str()).collect();
        let result_at = types.iter().position(|t| *t == "tool_result").unwrap();
        assert_eq!(types[result_at + 1], "tool_timing", "{types:?}");
        let timing: serde_json::Value = serde_json::from_str(&stored[result_at + 1].data).unwrap();
        assert_eq!(timing["permission_outcome"], "allowed", "{timing}");
        assert_eq!(
            timing["run_started_at"], timing["permission_resolved_at"],
            "{timing}"
        );
        assert!(timing["run_started_at"].is_f64(), "{timing}");
        assert_eq!(
            types.iter().filter(|t| **t == "tool_timing").count(),
            1,
            "one timing per call: {types:?}"
        );
    }

    /// Two answers at once (allow in one tab, deny in another): the provider is slow
    /// to take the allow, the deny would overtake it. The deny waits until the allow
    /// was handed over, so the clock's outcome is the one the provider took.
    #[tokio::test]
    async fn two_answers_at_once_reach_the_provider_in_the_order_the_clock_keeps() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph.clone());
        let provider = FakeProvider::new();
        let sid = Uuid::new_v4().to_string();
        let handle = runtime
            .adopt(
                &sid,
                "claude-code",
                provider.session(),
                1,
                "native",
                json!({}),
                None,
            )
            .await;
        handle
            .emit(ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "Bash".into(),
                input: json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            })
            .await;
        handle
            .emit(ChatEvent::PermissionRequest {
                id: "req-1".into(),
                tool: "Bash".into(),
                input: json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
                tool_use_id: Some("t1".into()),
                always: None,
                consent: None,
            })
            .await;
        let hold = Arc::new(tokio::sync::Notify::new());
        *provider.state.hold_allows.lock().unwrap() = Some(Arc::clone(&hold));
        let allow = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move { handle.answer_permission("req-1", true).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        let deny = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move { handle.answer_permission("req-1", false).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        hold.notify_one();
        let _ = allow.await.unwrap();
        let _ = deny.await.unwrap();
        handle
            .emit(ChatEvent::ToolResult {
                id: "t1".into(),
                result: json!("a.rs"),
                is_error: false,
                parent_tool_use_id: None,
            })
            .await;

        let taken = provider.state.permission_answers.lock().unwrap().clone();
        assert_eq!(taken.len(), 1, "one answer taken: {taken:?}");
        let taken = match &taken[0].1 {
            PermissionDecision::Allow { .. } => "allowed",
            PermissionDecision::Deny { .. } => "denied",
            other => panic!("an answer of another kind: {other:?}"),
        };
        let stored = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 50)
            .await
            .unwrap();
        let timing = stored
            .iter()
            .find(|r| r.event_type == "tool_timing")
            .expect("a timing");
        let timing: serde_json::Value = serde_json::from_str(&timing.data).unwrap();
        assert_eq!(timing["permission_outcome"], taken, "{timing}");
        assert_eq!(taken, "allowed", "the first answer is the one handed over");
    }

    /// The backend answers a request a session grant covers, the provider is slow to
    /// take that allow, and the user denies the same request meanwhile: the deny waits
    /// for the backend's answer, so the clock's outcome is the one the provider took.
    #[tokio::test]
    async fn a_user_answer_waits_for_the_backend_answer_to_a_covered_request() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph.clone());
        let provider = FakeProvider::new();
        provider.caps.lock().unwrap().permission_scopes =
            vec![PermissionScope::Once, PermissionScope::Session];
        let sid = Uuid::new_v4().to_string();
        let handle = runtime
            .adopt(
                &sid,
                "local",
                provider.session(),
                1,
                "native",
                json!({}),
                None,
            )
            .await;
        let write = |id: &str, call: &str| ChatEvent::PermissionRequest {
            id: id.to_string(),
            tool: "mcp__nexus__Write".into(),
            input: json!({ "file_path": "a.txt", "content": "x" }),
            parent_tool_use_id: None,
            category: Some("edit".into()),
            canonical: Some("Write".into()),
            tool_use_id: Some(call.to_string()),
            always: None,
            consent: None,
        };
        handle.emit(write("write-1", "t1")).await;
        handle
            .answer_permission_scoped("write-1", true, PermissionAnswerScope::Session)
            .await
            .unwrap();
        handle
            .emit(ChatEvent::ToolUse {
                id: "t2".into(),
                tool: "mcp__nexus__Write".into(),
                input: json!({ "file_path": "a.txt", "content": "x" }),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            })
            .await;
        let hold = Arc::new(tokio::sync::Notify::new());
        *provider.state.hold_allows.lock().unwrap() = Some(Arc::clone(&hold));
        let covered = {
            let handle = Arc::clone(&handle);
            let request = write("write-2", "t2");
            tokio::spawn(async move { handle.emit(request).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        let deny = {
            let handle = Arc::clone(&handle);
            tokio::spawn(async move { handle.answer_permission("write-2", false).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        hold.notify_one();
        covered.await.unwrap();
        let _ = deny.await.unwrap();
        handle
            .emit(ChatEvent::ToolResult {
                id: "t2".into(),
                result: json!("written"),
                is_error: false,
                parent_tool_use_id: None,
            })
            .await;

        let taken: Vec<_> = provider
            .state
            .permission_answers
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, _)| id == "write-2")
            .map(|(_, d)| d.clone())
            .collect();
        assert_eq!(taken.len(), 1, "one answer taken: {taken:?}");
        let taken = match &taken[0] {
            PermissionDecision::Allow { .. } => "allowed",
            PermissionDecision::Deny { .. } => "denied",
            other => panic!("an answer of another kind: {other:?}"),
        };
        let stored = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 50)
            .await
            .unwrap();
        let decisions = stored
            .iter()
            .filter(|r| r.event_type == "permission_decision" && r.data.contains("write-2"))
            .count();
        assert_eq!(decisions, 1, "one decision for the request");
        let timing = stored
            .iter()
            .find(|r| r.event_type == "tool_timing" && r.data.contains("\"t2\""))
            .expect("a timing");
        let timing: serde_json::Value = serde_json::from_str(&timing.data).unwrap();
        assert_eq!(timing["permission_outcome"], taken, "{timing}");
        assert_eq!(
            taken, "allowed",
            "the backend's answer was handed over first"
        );
    }

    /// A double click (or two tabs): the provider refuses the second answer to the
    /// same request. Its failure must not erase the first answer from the clock.
    #[tokio::test]
    async fn a_second_answer_refused_by_the_provider_keeps_the_first_on_the_clock() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph.clone());
        let provider = FakeProvider::new();
        let sid = Uuid::new_v4().to_string();
        let handle = runtime
            .adopt(
                &sid,
                "claude-code",
                provider.session(),
                1,
                "native",
                json!({}),
                None,
            )
            .await;
        handle
            .emit(ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "Bash".into(),
                input: json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            })
            .await;
        handle
            .emit(ChatEvent::PermissionRequest {
                id: "req-1".into(),
                tool: "Bash".into(),
                input: json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
                tool_use_id: Some("t1".into()),
                always: None,
                consent: None,
            })
            .await;
        handle.answer_permission("req-1", true).await.unwrap();
        assert!(
            handle.answer_permission("req-1", false).await.is_err(),
            "the provider refuses a second answer"
        );
        handle
            .emit(ChatEvent::ToolResult {
                id: "t1".into(),
                result: json!("a.rs"),
                is_error: false,
                parent_tool_use_id: None,
            })
            .await;

        let stored = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 50)
            .await
            .unwrap();
        let timing = stored
            .iter()
            .find(|r| r.event_type == "tool_timing")
            .expect("a timing");
        let timing: serde_json::Value = serde_json::from_str(&timing.data).unwrap();
        assert_eq!(timing["permission_outcome"], "allowed", "{timing}");
        assert!(timing["permission_resolved_at"].is_f64(), "{timing}");
        assert_eq!(
            timing["run_started_at"], timing["permission_resolved_at"],
            "{timing}"
        );
    }
}

/// Follow-up review of #711, finding 2, on the agent engine: Claude Code answers an
/// interrupt with `error_during_execution` and `is_error: true`; the turn is STORED in the
/// neutral form the legacy loop stores (`manager::stored_interrupted_result`), so the
/// thread reloads without an error block whichever engine ran it. What is broadcast live
/// keeps the provider's form (the runner and the sub-agents read `is_error` there).
#[cfg(test)]
mod stored_interrupted_result_tests {
    use super::fake::FakeProvider;
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[tokio::test]
    async fn a_stopped_turn_is_stored_neutral_and_broadcast_as_the_provider_said() {
        let graph = Arc::new(MockGraphStore::new());
        let node = crate::test_helpers::test_chat_session(None);
        graph.create_chat_session(&node).await.unwrap();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let runtime = AgentRuntime::new(dyn_graph);
        let provider = FakeProvider::new();
        let handle = runtime
            .adopt(
                &node.id.to_string(),
                "local",
                provider.session(),
                1,
                "claude_code",
                serde_json::json!({}),
                None,
            )
            .await;
        let mut rx = handle.events_tx.subscribe();
        handle
            .emit(ChatEvent::Result {
                session_id: "cc".into(),
                duration_ms: 42,
                cost_usd: None,
                subtype: "error_during_execution".into(),
                is_error: true,
                num_turns: Some(1),
                result_text: Some(String::new()),
                cost: None,
                usage: None,
                model: None,
                stop_reason: Some("interrupted".into()),
                error_code: None,
                error_message: None,
            })
            .await;
        let live = rx.recv().await.unwrap();
        assert!(
            matches!(&live, ChatEvent::Result { subtype, is_error: true, .. } if subtype == "error_during_execution"),
            "{live:?}"
        );
        let stored = graph.get_chat_events(node.id, 0, 50).await.unwrap();
        let ended: serde_json::Value = stored
            .iter()
            .find(|e| e.event_type == "result")
            .map(|e| serde_json::from_str(&e.data).unwrap())
            .expect("the result is stored");
        assert_eq!(ended["subtype"], "interrupted", "{ended}");
        assert_eq!(ended["is_error"], false, "{ended}");
        assert_eq!(ended["stop_reason"], "interrupted", "{ended}");
        assert_eq!(ended["duration_ms"], 42, "{ended}");
        assert!(ended.get("result_text").is_none(), "{ended}");
    }
}
