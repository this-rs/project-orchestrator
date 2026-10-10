//! The provider-neutral session engine (task B13, decision A18).
//!
//! `CHAT_PROVIDER_PATH=agent` makes [`ChatManager`](super::manager::ChatManager)
//! drive a session through an `Arc<dyn AgentSession>` (nexus contract) instead
//! of the Claude `InteractiveClient`. This module owns what that needs and
//! nothing Claude specific: the live sessions, the turn driver that turns the
//! provider's event stream into `ChatEvent`s (through [`EventMapper`]), the
//! out-of-turn pump, persistence of the events for replay, and the answers to
//! permission requests, interrupts, model and mode changes.
//!
//! The legacy path stays as it was; the manager decides which one serves a
//! session and a session never changes engine.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use futures::StreamExt;
use nexus_claude::agent::{
    AgentEvent, AgentProvider, AgentSession, CancelScope, Capabilities, InterruptScope,
    PermissionDecision, PolicyMode, ProviderError, TurnInput,
};
use tokio::sync::{broadcast, Mutex, RwLock};
use uuid::Uuid;

use super::config::RetryConfig;
use super::manager::{CancelToolsResult, ChatManager, CANCEL_TOOLS_CAP, CANCEL_TOOLS_WINDOW_SECS};
use super::provider::event_map::{out_of_band_to_chat_events, EventMapper};
use super::types::{ChatEvent, PendingMessage, PendingMessageKind};
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::GraphStore;

const BROADCAST_BUFFER: usize = 1024;
/// How long the out-of-turn pump waits for the rest of a provider message
/// before it flushes what it has.
const OOB_FLUSH: Duration = Duration::from_millis(50);

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

/// The failure that ends a turn and is worth trying again: a retryable
/// `done.error` of an error turn, or a retryable terminal `error`.
fn retryable_failure(event: &AgentEvent) -> Option<ProviderError> {
    match event {
        AgentEvent::Done {
            is_error: true,
            error: Some(error),
            ..
        }
        | AgentEvent::Error { error }
            if error.retryable() =>
        {
            Some(error.clone())
        }
        _ => None,
    }
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

/// What a session on the agent engine does NOT do, as the identifiers the
/// frontend knows (`hooks`, `message_queue`, `auto_continue`, `compaction`,
/// `nats`, `enrichment`, `images`, and [`NEXUS_TOOLS_FEATURE`] added by the host).
///
/// Two sources, kept apart on purpose:
/// - what THIS ENGINE (the backend) has not ported, whatever the provider can do:
///   nothing any more (enrichment, message queue, auto-continue, NATS are ported);
/// - what THE SESSION's capabilities say it cannot do: `images`, and `compaction`
///   when the provider emits no compaction signal.
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
    /// Hands an event of the session to the other instances (NATS), as the Claude
    /// Code engine publishes each of its events. Default: nowhere.
    fn publish(&self, _session_id: &str, _event: &ChatEvent) {}
    /// Sees each event of the session as it is emitted (the work log of the turn).
    /// Default: nothing.
    fn observe(&self, _session_id: &str, _event: &ChatEvent) {}
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
    /// Continue a turn that stopped on its turn limit (`set_auto_continue`).
    pub auto_continue: AtomicBool,
    auto_continue_count: std::sync::atomic::AtomicU32,
    /// Continuations allowed before auto-continue switches itself off (0: no limit).
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
    /// When the session last did something (an event, a message): what the idle
    /// cleanup reads (`ChatManager::start_cleanup_task`).
    last_activity: std::sync::Mutex<Instant>,
    /// When its tool calls really ran (`tool_clock`): a `tool_timing` follows each result.
    tool_clock: Arc<super::tool_clock::ToolClock>,
}

impl AgentSessionHandle {
    /// Persists (except transient events) and broadcasts one event, then the timing
    /// of the tool call it ends, if it ends one.
    pub async fn emit(&self, event: ChatEvent) {
        let timing = self.tool_clock.observe(&event, chrono::Utc::now());
        self.emit_one(event).await;
        if let Some(timing) = timing {
            self.emit_one(timing).await;
        }
    }

    async fn emit_one(&self, mut event: ChatEvent) {
        self.touch();
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
                serde_json::to_value(&self.capabilities).unwrap_or_default()
            });
            tool_policy.get_or_insert_with(|| self.tool_policy.clone());
        }
        match &event {
            ChatEvent::StreamDelta { text, .. } => self.streaming_text.lock().await.push_str(text),
            ChatEvent::StreamingStatus { .. } | ChatEvent::PendingQueue { .. } => {}
            other => self.streaming_events.lock().await.push(other.clone()),
        }
        if !matches!(
            event,
            ChatEvent::StreamDelta { .. }
                | ChatEvent::StreamingStatus { .. }
                | ChatEvent::PendingQueue { .. }
        ) {
            if let Some(uuid) = self.uuid {
                let record = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: self.next_seq.fetch_add(1, Ordering::SeqCst),
                    event_type: event.event_type().to_string(),
                    data: serde_json::to_string(&event).unwrap_or_default(),
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

    /// Retries this session's failed turns as `retry` says (the chat's configuration).
    pub fn configure_retry(&self, retry: RetryConfig) {
        *self.retry.write().unwrap_or_else(|e| e.into_inner()) = retry;
    }

    pub(crate) fn retry_config(&self) -> RetryConfig {
        self.retry.read().unwrap_or_else(|e| e.into_inner()).clone()
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
        {
            // `is_streaming` is read and set under the queue lock, the lock the end
            // of a turn holds to decide there is nothing left to drain: a message
            // is either seen by the running turn or starts its own.
            let mut queue = self.pending.lock().await;
            if self.is_streaming.load(Ordering::SeqCst) {
                queue.push_back(PendingMessage::user(shown.to_string()));
                drop(queue);
                self.interrupt_for_the_queue().await;
                return Ok(());
            }
            self.is_streaming.store(true, Ordering::SeqCst);
        }
        self.start(PendingMessageKind::User, shown, sent).await
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
        self.emit(ChatEvent::PendingQueue { messages }).await;
        if outcome.interrupt && self.is_streaming.load(Ordering::SeqCst) {
            self.interrupt_for_the_queue().await;
        }
    }

    /// Cuts the running turn short so the message queued in front is read now.
    async fn interrupt_for_the_queue(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
        if let Err(e) = self.session.interrupt(InterruptScope::TurnAndTools).await {
            tracing::warn!(session_id = %self.session_id, error = %e, "interrupting the turn for a queued message failed");
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
            match self
                .open_turn(next.kind, &next.content, &next.content)
                .await
            {
                Ok(opened) => turn = opened,
                Err(error) => {
                    let event = AgentEvent::Error { error };
                    for chat_event in self.mapper.lock().await.map(&event) {
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
        mut stream: nexus_claude::agent::EventStream,
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
        loop {
            // Did the turn already show the user anything? A turn that did is
            // never replayed: it would repeat text or tool calls.
            let mut shown = false;
            let mut retry: Option<ProviderError> = None;
            while let Some(event) = stream.next().await {
                // Terminal-ness is read on the original: a withheld terminal event
                // still ends the turn.
                let terminal = event.is_terminal();
                // The provider may have just named its session: stored before the
                // terminal event goes out, so a client that saw the turn end can
                // count on a resume.
                if terminal || matches!(event, AgentEvent::SessionStarted { .. }) {
                    self.sync_resume_token().await;
                }
                if !shown {
                    retry = retryable_failure(&event).filter(|_| attempt < max_attempts);
                    if retry.is_some() {
                        break;
                    }
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
            let Some(error) = retry else { break };
            attempt += 1;
            let delay = retry_delay_ms(&error, attempt, &retry_config);
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
                for chat_event in self.mapper.lock().await.map(&stopped) {
                    self.emit(chat_event).await;
                }
                break;
            }
            match self.session.send_turn(input.clone()).await {
                Ok(next) => stream = next,
                Err(e) => {
                    let ev = AgentEvent::Error { error: e };
                    for chat_event in self.mapper.lock().await.map(&ev) {
                        self.emit(chat_event).await;
                    }
                    break;
                }
            }
        }
        // The record carries the turn's cost before the session is seen idle.
        self.add_turn_cost(turn_cost).await;
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
        let Some(services) = &self.services else {
            return;
        };
        let Some(after) = super::post_stream::bounded(
            &self.session_id,
            "after_turn",
            AFTER_TURN_BACKSTOP,
            services.after_turn(&self.session_id, outcome),
        )
        .await
        else {
            self.emit(ChatEvent::Error {
                message: "Error: the end-of-turn processing took too long: skipped".into(),
                parent_tool_use_id: None,
                code: Some(super::post_stream::STEP_ABANDONED_CODE.to_string()),
                reason: Some("after_turn".to_string()),
                index: None,
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

    /// Answers a permission request.
    pub async fn answer_permission(&self, request_id: &str, allow: bool) -> Result<()> {
        let decision = if allow {
            PermissionDecision::allow_once()
        } else {
            PermissionDecision::deny()
        };
        // On the tool clock BEFORE the provider has it: a fast tool's result cannot
        // overtake it.
        // A second answer (double click, two tabs) the provider refuses takes back
        // only its own mark, never the first answer's.
        let mark = self
            .tool_clock
            .decided(request_id, allow, chrono::Utc::now());
        if let Err(e) = self.session.answer_permission(request_id, decision).await {
            self.tool_clock.undecided(request_id, mark);
            return Err(anyhow::Error::new(e));
        }
        self.emit(ChatEvent::PermissionDecision {
            id: request_id.to_string(),
            allow,
        })
        .await;
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
        let outcome = self
            .session
            .cancel_tools(CancelScope::All)
            .await
            .map_err(anyhow::Error::new)?;
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

/// The live sessions of the agent path.
pub struct AgentRuntime {
    sessions: RwLock<HashMap<String, Arc<AgentSessionHandle>>>,
    graph: Arc<dyn GraphStore>,
}

impl AgentRuntime {
    /// A runtime persisting its events in `graph`.
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            graph,
        }
    }

    /// The live session, if this runtime owns it.
    pub async fn get(&self, session_id: &str) -> Option<Arc<AgentSessionHandle>> {
        self.sessions.read().await.get(session_id).cloned()
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
            cost_figure: super::session_record::CostFigure::of_kind(provider_kind),
            keeps_interrupted_turns: keeps_interrupted_turns(provider_kind),
            record: Mutex::new(()),
            opening_counted: AtomicBool::new(false),
            retry: std::sync::RwLock::new(RetryConfig::default()),
            last_activity: std::sync::Mutex::new(Instant::now()),
        });
        if let Some(oob) = session.out_of_band() {
            let pump = Arc::clone(&handle);
            tokio::spawn(async move { pump_out_of_band(pump, oob).await });
        }
        let replaced = self
            .sessions
            .write()
            .await
            .insert(session_id.to_string(), Arc::clone(&handle));
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
        handle
    }

    /// Closes a session: tells the provider, announces `session_closed` (A45),
    /// forgets it.
    pub async fn close(&self, session_id: &str) -> Result<()> {
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
        handle
            .emit(ChatEvent::SessionClosed {
                session_id: session_id.to_string(),
                reason: Some("closed".to_string()),
            })
            .await;
        Ok(())
    }
}

/// Reads the out-of-turn stream, regroups what belongs to one provider
/// message and emits it: the batch is flushed when the provider pauses, when
/// it grows large, and when the stream ends.
async fn pump_out_of_band(
    handle: Arc<AgentSessionHandle>,
    mut stream: nexus_claude::agent::EventStream,
) {
    let mut batch = Vec::new();
    loop {
        let item = if batch.is_empty() {
            Some(stream.next().await)
        } else {
            tokio::time::timeout(OOB_FLUSH, stream.next()).await.ok()
        };
        match item {
            Some(Some(event)) => {
                batch.push(event);
                if batch.len() >= 64 {
                    flush(&handle, &mut batch).await;
                }
            }
            Some(None) => {
                flush(&handle, &mut batch).await;
                break;
            }
            None => flush(&handle, &mut batch).await,
        }
    }
}

async fn flush(handle: &AgentSessionHandle, batch: &mut Vec<nexus_claude::agent::AgentEvent>) {
    if batch.is_empty() {
        return;
    }
    let events: Vec<AgentEvent> = std::mem::take(batch)
        .into_iter()
        .map(mask_agent_event)
        .collect();
    let chat_events = {
        let mut mapper = handle.mapper.lock().await;
        out_of_band_to_chat_events(&events, &mut mapper, chrono::Utc::now())
    };
    for event in chat_events {
        handle.emit(event).await;
    }
    // An out-of-turn `init` (the CLI repeats it) may carry a new session id.
    handle.sync_resume_token().await;
}

/// The built-in provider instances: Claude Code, with the CLI path of the
/// server configuration. The nexus registry replaces this.
pub struct BuiltinProviders {
    claude: Arc<dyn AgentProvider>,
}

impl BuiltinProviders {
    /// The built-in instances.
    pub fn new(claude_cli_path: Option<String>) -> Self {
        use nexus_claude::providers::claude_code::{ClaudeCodeConfig, ClaudeCodeProvider};
        let mut config = ClaudeCodeConfig::default();
        config.cli_path = claude_cli_path.map(std::path::PathBuf::from);
        Self {
            claude: Arc::new(ClaudeCodeProvider::new(config)),
        }
    }
}

impl ProviderSource for BuiltinProviders {
    fn get(&self, provider_id: &str) -> Option<Arc<dyn AgentProvider>> {
        (provider_id == super::provider::resolver::CLAUDE_CODE).then(|| Arc::clone(&self.claude))
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
            self.state.end_turn();
            Ok(InterruptOutcome::default())
        }
        async fn cancel_tools(&self, _scope: CancelScope) -> Result<CancelOutcome, ProviderError> {
            Ok(CancelOutcome::default())
        }
        async fn set_model(&self, model: &str) -> Result<(), ProviderError> {
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
        /// Capabilities the sessions of this provider declare.
        pub caps: Arc<StdMutex<Capabilities>>,
        /// The kind it says it is (Claude Code unless a test plays another).
        pub kind: Arc<StdMutex<ProviderKind>>,
    }

    impl FakeProvider {
        pub fn new() -> Self {
            Self {
                state: Arc::new(FakeState::default()),
                fail_open: Arc::new(StdMutex::new(None)),
                caps: Arc::new(StdMutex::new(Capabilities::none())),
                kind: Arc::new(StdMutex::new(ProviderKind::ClaudeCode)),
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
            ProviderHealth::ok(None)
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

    #[test]
    fn the_retry_delay_follows_the_provider_then_the_chat_retry_config() {
        let config = RetryConfig::default();
        let limited = |ms| ProviderError::RateLimited {
            retry_after_ms: Some(ms),
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
        assert!(retryable_failure(&done(true, Some(ProviderError::Overloaded))).is_some());
        assert!(retryable_failure(&done(true, Some(ProviderError::Unauthorized))).is_none());
        assert!(retryable_failure(&done(false, Some(ProviderError::Overloaded))).is_none());
        assert!(retryable_failure(&done(true, None)).is_none());
        assert!(retryable_failure(&AgentEvent::Error {
            error: ProviderError::Overloaded
        })
        .is_some());
    }
}

/// Races between a turn being opened (the enrichment awaits) and what reaches the
/// session meanwhile.
#[cfg(test)]
mod turn_race_tests {
    use super::fake::FakeProvider;
    use super::*;
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
}

#[cfg(test)]
mod image_tests {
    use super::*;

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

/// The end of a turn of the agent engine: what the Claude Code engine's
/// post-stream does — the tool calls a Stop left are said cancelled, the host's
/// `after_turn` sees what the turn did, its events go out and its hints are played.
#[cfg(test)]
mod after_turn_tests {
    use super::fake::FakeProvider;
    use super::*;
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
    }

    async fn image_turn(route_to: Option<&'static str>) -> (FakeProvider, Arc<ImageHost>) {
        let graph: Arc<dyn GraphStore> = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let runtime = AgentRuntime::new(graph);
        let provider = FakeProvider::new();
        let host = Arc::new(ImageHost {
            state: Arc::clone(&provider.state),
            turns_seen_when_asked: StdMutex::new(None),
            route_to,
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

    #[tokio::test]
    async fn an_image_turn_with_nothing_to_route_keeps_its_model() {
        let (provider, _host) = image_turn(None).await;
        assert!(provider.state.models.lock().unwrap().is_empty());
    }
}

#[cfg(test)]
mod retry_stop_tests {
    use super::fake::FakeProvider;
    use super::*;

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

/// What waits for the next turn after a compaction is dropped only when a turn
/// carrying it was ANSWERED (`TurnOutcome::answered`): a provider `done` without
/// error, `interrupted` included (the provider kept the turn). Not a turn accepted
/// then failed (the native harness accepts before any request), not one whose
/// retries ran out, not one stopped during the pause before a retry.
#[cfg(test)]
mod answered_tests {
    use super::fake::FakeProvider;
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// The `answered` of each turn the host saw end.
    #[derive(Default)]
    struct Host {
        answered: StdMutex<Vec<bool>>,
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
            AfterTurn::default()
        }
    }

    impl Host {
        fn seen(&self) -> Vec<bool> {
            self.answered.lock().unwrap().clone()
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
}

#[cfg(test)]
mod tool_timing_tests {
    use super::fake::FakeProvider;
    use super::*;
    use crate::neo4j::traits::GraphStore as _;
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
