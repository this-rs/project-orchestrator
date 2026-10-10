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

/// Retries of a turn that failed before showing anything (`done.error` retryable).
const MAX_RETRIES: u32 = 3;

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

/// Delay before attempt `n`: what the provider asked (`retry_after`), else an
/// exponential backoff from one second, never beyond thirty.
fn retry_delay_ms(error: &ProviderError, attempt: u32) -> u64 {
    if let ProviderError::RateLimited {
        retry_after_ms: Some(ms),
    } = error
    {
        return (*ms).min(30_000);
    }
    (1000u64 << attempt.saturating_sub(1).min(5)).min(30_000)
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
    /// Hands an event of the session to the other instances (NATS), as the Claude
    /// Code engine publishes each of its events. Default: nowhere.
    fn publish(&self, _session_id: &str, _event: &ChatEvent) {}
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
}

impl AgentSessionHandle {
    /// Persists (except transient events) and broadcasts one event.
    pub async fn emit(&self, mut event: ChatEvent) {
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
                .await
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
                let hit_turn_limit = self.play(stream, input).await;
                self.auto_continue_after(hit_turn_limit).await;
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
    /// Answers whether the turn stopped on its turn limit (`max_turns`).
    async fn play(&self, mut stream: nexus_claude::agent::EventStream, input: TurnInput) -> bool {
        let mut attempt = 0u32;
        let mut hit_turn_limit = false;
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
                    retry = retryable_failure(&event).filter(|_| attempt < MAX_RETRIES);
                    if retry.is_some() {
                        break;
                    }
                }
                shown |= shows_content(&event);
                hit_turn_limit |= matches!(
                    event,
                    AgentEvent::Done {
                        stop_reason: nexus_claude::agent::StopReason::MaxTurns,
                        ..
                    }
                );
                let event = mask_agent_event(event);
                let chat_events = self.mapper.lock().await.map(&event);
                for chat_event in chat_events {
                    self.emit(chat_event).await;
                }
                if terminal {
                    break;
                }
            }
            let Some(error) = retry else { break };
            attempt += 1;
            let delay = retry_delay_ms(&error, attempt);
            self.emit(ChatEvent::Retrying {
                attempt,
                max_attempts: MAX_RETRIES,
                delay_ms: delay,
                error_message: format!(
                    "Error: {}",
                    super::provider::errors::open_failure(&error, None).message
                ),
            })
            .await;
            tokio::time::sleep(Duration::from_millis(delay)).await;
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
        hit_turn_limit
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
    async fn auto_continue_after(&self, hit_turn_limit: bool) {
        use super::post_stream::{auto_continue_allowed, AUTO_CONTINUE_DELAY_MS};
        let Some(services) = &self.services else {
            return;
        };
        if !auto_continue_allowed(
            &self.session_id,
            hit_turn_limit,
            &self.auto_continue,
            self.interrupted.load(Ordering::SeqCst),
            &self.auto_continue_count,
            self.max_auto_continues.load(Ordering::Relaxed),
        ) {
            return;
        }
        self.emit(ChatEvent::AutoContinue {
            session_id: self.session_id.clone(),
            delay_ms: AUTO_CONTINUE_DELAY_MS,
        })
        .await;
        tokio::time::sleep(Duration::from_millis(AUTO_CONTINUE_DELAY_MS)).await;
        if self.interrupted.load(Ordering::SeqCst) {
            tracing::info!(session_id = %self.session_id, "Auto-continue cancelled by interrupt");
            return;
        }
        let hint = services.continuation(&self.session_id).await;
        self.pending
            .lock()
            .await
            .push_back(PendingMessage::system_hint(hint));
    }

    /// Answers a permission request.
    pub async fn answer_permission(&self, request_id: &str, allow: bool) -> Result<()> {
        let decision = if allow {
            PermissionDecision::allow_once()
        } else {
            PermissionDecision::deny()
        };
        self.session
            .answer_permission(request_id, decision)
            .await
            .map_err(anyhow::Error::new)?;
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
            cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
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
            self.state
                .permission_answers
                .lock()
                .unwrap()
                .push((request_id.to_string(), decision));
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
    fn the_retry_delay_follows_the_provider_then_backs_off() {
        assert_eq!(
            retry_delay_ms(
                &ProviderError::RateLimited {
                    retry_after_ms: Some(40)
                },
                1
            ),
            40
        );
        assert_eq!(
            retry_delay_ms(
                &ProviderError::RateLimited {
                    retry_after_ms: Some(900_000)
                },
                1
            ),
            30_000
        );
        assert_eq!(retry_delay_ms(&ProviderError::Overloaded, 1), 1000);
        assert_eq!(retry_delay_ms(&ProviderError::Overloaded, 3), 4000);
        assert_eq!(retry_delay_ms(&ProviderError::Overloaded, 20), 30_000);
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
