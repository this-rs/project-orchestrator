//! Read-only MIRROR children of a Claude Code session's sub-agents (P18c, decision 1ec3e540).
//!
//! A sub-agent Claude Code launches with its own `Task` / `Agent` tool runs INSIDE the CLI
//! (its `.claude/agents` definitions, its shared context, its model choice): it has no
//! project-orchestrator session of its own. Its activity reaches the parent's stream with
//! `parent_tool_use_id` = the id of that call, and the parent shows it nested under the
//! call's card. A native sub-agent, by contrast, is a real child session
//! (`chat::subagents`). This module gives the Claude Code sub-agent the same LINK without
//! touching what the CLI does: a child conversation that only MIRRORS it.
//!
//! # Strictly passive
//!
//! A [`MirrorTap`] SEES the parent's events after they were emitted, as the parent's other
//! observers do; it never alters, delays nor intercepts the call or its events in the
//! parent (the parent's thread and its nested display are exactly what they were). The tap
//! only clones the few events it needs into an unbounded channel, in order; a task of its
//! own does every write, so the parent's stream never waits on the mirror.
//!
//! Where the parent's events are seen:
//! - agent engine (the default for Claude Code): `ManagerTurnServices::observe`, which the
//!   runtime calls for every event a session emits (`AgentSessionHandle::emit_one`); only a
//!   session whose provider kind is `claude_code` gets a tap (`finish_agent_open`);
//! - historical engine: the `emit_chat` helper of `ChatManager::stream_response`, through
//!   which every event of a turn is broadcast, plus an explicit end of turn before the turn's
//!   `streaming_status: false` (`ActiveSession::mirror_tap`).
//!
//! A native session never gets a tap: its sub-agents are real children already.
//!
//! # The mirror
//!
//! On the FIRST event carrying `parent_tool_use_id = X` where X is a top-level call of the
//! `Task` or `Agent` tool, ONE child session node is written:
//! `spawned_by = conversation { parent_session_id, tool_use_id: X }` (the join the parent's
//! Agent card already makes), the parent's provider, model, cwd, project, workspace, owner,
//! access and engine, the title from the call's `description` (else its prompt), and the
//! marker [`ChatSessionNode::mirror_of`](crate::neo4j::models::ChatSessionNode::mirror_of)
//! `= "claude_code"` — on the wire `"mirror_of": "claude_code"` in every session response,
//! omitted for every other session. One CRUD `created` is emitted with exactly the payload
//! every session gets (decision ed628dad: `{project_slug, spawned_by: {type,
//! parent_session_id}}`; the client reads `tool_use_id` and `mirror_of` from `/children` and
//! the session), then `updated {is_streaming: true}`.
//!
//! Its thread: the call's prompt as its first `user_message`, then every event carrying
//! `parent_tool_use_id = X`, in order, WITHOUT that parent id (the child shows them at its top
//! level). Permission requests and questions are not copied: they are answered in the parent,
//! and a copy would sit unanswerable in the mirror. Deltas are sent live, never stored (as
//! everywhere).
//!
//! Closed — its `result` written (the call's `tool_result` text, `is_error` kept), then
//! `updated {is_streaming: false}` — when the call's `tool_result` (or its cancel) reaches
//! the parent, else when the parent's turn ends, else when the parent session closes (its
//! tap is dropped). Events of X arriving after the close are dropped. The close also stamps
//! the node's `updated_at` (one write: a session listing shows when the mirror last moved;
//! its `message_count` stays 1, the prompt, as it counts the user's messages).
//!
//! A BACKGROUND call (`run_in_background: true`) answers its `tool_result` at once ("launched"):
//! that result does not close its mirror. The sub-agent goes on after the parent's turn, and
//! Claude Code keeps printing what it does on the SAME stream (CLI 2.1.296, measured with
//! `claude -p --output-format stream-json`): its messages with `parent_tool_use_id` = the call,
//! between the parent's turns, then `system/task_notification {tool_use_id, task_id, status,
//! summary, output_file}` when it ends. That stream is the mirror's source, not the
//! transcript or the `output_file` the notification names: it is already ordered, already
//! masked, already read by one task per session, and it carries the sub-agent's structure
//! (tool calls, results) that the output file flattens; a file would also mean reading the
//! CLI's temporary directory from the backend.
//!
//! - Agent engine: what the CLI prints between turns comes through the runtime's
//!   out-of-turn pump, which shows it to the parent flattened (`background_output`); the
//!   pump hands the tap the same events in their structured form
//!   (`TurnServices::observe_out_of_turn`, `event_map::structured_out_of_turn`), never
//!   emitted. The mirror stays open across the parent's turns and is closed by the
//!   notification that names its call (its `tool_use_id`, else the `task_id` its
//!   `task_started` gave): `completed` closes it as a success, `killed` / `stopped` as
//!   interrupted, anything else as a failure, with the notification's `summary` as its
//!   result. At the latest it closes when its parent session closes (the tap is dropped),
//!   and a restart closes it (`manager::recovery`).
//! - Historical engine: what comes out of turn goes through `oob_listener`, which no tap
//!   sees: the mirror is closed at the end of the parent's turn, saying the rest shows in
//!   the parent.
//!
//! Only FIRST-LEVEL calls of a session are mirrored: a sub-agent's own `Task` call (an
//! event whose parent is a call made inside a mirror) goes into its first-level mirror's
//! thread, keeping its parent id so it shows nested there.
//!
//! # One writer per mirror, and idempotency
//!
//! The manager's [`MirrorRegistry`] holds a CLAIM per (parent, call), taken atomically by one
//! tap (its owner token): a second live tap of the same parent (an old turn's tap still
//! finishing after a resume) is refused and leaves the call alone, and a tap only ever closes
//! or forgets what it owns — a stale tap can never remove a newer tap's entry. Before writing
//! a mirror, the claiming tap asks the graph (the parent's children whose
//! `spawned_by.tool_use_id` is X and that carry the marker): a mirror found is reused (its
//! thread continued) unless it already ends on its `result` (its late events are then
//! dropped). Nothing is created on a reconnect or a replay: those read stored events, they do
//! not emit them again.
//!
//! Bounded: a tap forgets every claim it owns when it ends (its parent closed), and the
//! registry drops the claims of closed mirrors past [`MAX_CLAIMS`]; the graph remains the
//! truth for a claim forgotten. A tap clones only the parent's top-level tool events of the
//! sub-agent calls it watches (never the output of a Read or a Bash).
//!
//! # Read-only
//!
//! A mirror has no process: anything that would write to it is refused
//! ([`MirrorReadOnly`]: HTTP 409 `{error, code: "mirror_read_only", retryable: false}`) — a
//! message (REST, `POST /api/chat/sessions` with its `session_id`, i.e. `chat(send_message)`),
//! a resume, a permission answer, `switch-provider` (refused by `relay_conversation` itself),
//! a change of model or of permission mode; on the WebSocket the flat frame
//! `{type: "error", code: "mirror_read_only", message}` for a message (queued or not), an
//! answer to a permission or a question, a change of model or mode. An interrupt or a cancel
//! is a harmless no-op. Deleting an open mirror makes its tap drop it: never written again,
//! never written anew for its call. The restart recovery never resumes one
//! (`manager::recovery`). The refusal reads no graph for a session live here (it is not a
//! mirror) nor for a mirror a live tap knows.
//!
//! Known limit: a client that joins an OPEN mirror reads its stored events, then follows it
//! live; an event stored between the two is shown only after a reload (a normal session
//! fills that gap from its streaming snapshot, a mirror has none).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

use super::types::{ChatEvent, SpawnedBy};
use crate::neo4j::models::{ChatEventRecord, ChatSessionNode};
use crate::neo4j::GraphStore;

/// The value of `mirror_of` on a mirror of a Claude Code sub-agent.
pub const MIRROR_OF_CLAUDE_CODE: &str = "claude_code";

/// The stable code of the refusal of a message sent to a mirror (HTTP 409).
pub const MIRROR_READ_ONLY: &str = "mirror_read_only";

/// A message was sent to a mirror ([module documentation](self)).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "this conversation mirrors a Claude Code sub-agent and is read-only: write in its parent conversation"
)]
pub struct MirrorReadOnly;

impl MirrorReadOnly {
    /// The typed refusal the API returns: 409, code [`MIRROR_READ_ONLY`].
    pub fn failure() -> super::provider::errors::OpenFailure {
        super::provider::errors::OpenFailure {
            status: 409,
            code: MIRROR_READ_ONLY,
            message: MirrorReadOnly.to_string(),
            provider_id: None,
            action: None,
            retryable: false,
            retry_after_ms: None,
            fallbacks: None,
            consent_scope: None,
            candidates: None,
        }
    }
}

/// Whether `tool` is Claude Code's sub-agent tool (`Task`, renamed `Agent`; nexus'
/// `claude_code::policy_map` puts both in the `agent` category).
pub fn is_subagent_tool(tool: &str) -> bool {
    matches!(tool, "Task" | "Agent")
}

/// The parent call id an event carries, when it has the field.
fn parent_of(event: &ChatEvent) -> Option<&str> {
    match event {
        ChatEvent::AssistantText {
            parent_tool_use_id, ..
        }
        | ChatEvent::Thinking {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUse {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolResult {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUseInputResolved {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolCancelled {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolTiming {
            parent_tool_use_id, ..
        }
        | ChatEvent::PermissionRequest {
            parent_tool_use_id, ..
        }
        | ChatEvent::AskUserQuestion {
            parent_tool_use_id, ..
        }
        | ChatEvent::StreamDelta {
            parent_tool_use_id, ..
        }
        | ChatEvent::Error {
            parent_tool_use_id, ..
        } => parent_tool_use_id.as_deref(),
        _ => None,
    }
}

/// The parent field of an event the mirror copies (`None`: not copied).
fn copied_parent_mut(event: &mut ChatEvent) -> Option<&mut Option<String>> {
    match event {
        ChatEvent::AssistantText {
            parent_tool_use_id, ..
        }
        | ChatEvent::Thinking {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUse {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolResult {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolUseInputResolved {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolCancelled {
            parent_tool_use_id, ..
        }
        | ChatEvent::ToolTiming {
            parent_tool_use_id, ..
        }
        | ChatEvent::StreamDelta {
            parent_tool_use_id, ..
        }
        | ChatEvent::Error {
            parent_tool_use_id, ..
        } => Some(parent_tool_use_id),
        // Answered in the parent: never copied (module documentation).
        _ => None,
    }
}

/// The text of a `tool_result`: a string, or the text blocks of a structured content.
fn result_text(result: &serde_json::Value) -> String {
    match result {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                b.get("text")
                    .and_then(|t| t.as_str())
                    .or_else(|| b.as_str())
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// How many claims the registry keeps at most: past it, the claims of closed mirrors are
/// forgotten first (a forgotten claim is found again in the graph, never written twice).
const MAX_CLAIMS: usize = 4096;

/// Capacity of the live broadcast of one open mirror.
const MIRROR_BROADCAST: usize = 1024;

/// The owner tokens of the taps (one per tap, never reused in a process).
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// One (parent, call) a tap took: by whom, its mirror once written, whether it is over.
struct Claim {
    owner: u64,
    mirror: Option<Uuid>,
    closed: bool,
}

/// What the registry answers a tap that wants the mirror of a call.
enum ClaimOutcome {
    /// Nobody holds it in this process: it is now this tap's (look in the graph, then write).
    Claimed,
    /// Another live tap holds it, or it is over: this tap leaves it alone.
    Refused,
}

#[derive(Default)]
struct RegistryState {
    /// (parent session, call id) -> who holds it: never two mirrors, nor two writers.
    claims: HashMap<(String, String), Claim>,
    /// The open mirrors: owner and live broadcast (what `ChatManager::subscribe` hands).
    open: HashMap<Uuid, (u64, broadcast::Sender<ChatEvent>)>,
    /// The mirrors this process wrote or found, while their parent's tap lives: the
    /// refusal of a write needs no graph read for them (`ChatManager::refuse_if_mirror`).
    known: HashMap<Uuid, u64>,
    /// Open mirrors deleted under their tap: it drops them at its next look.
    deleted: HashSet<Uuid>,
}

/// What the manager knows of the mirrors of this process, shared by every tap. Every
/// operation is ONE critical section (a `std` mutex, never held across an `await`): a claim
/// and its check are atomic, and a tap only ever closes or forgets what it owns.
#[derive(Default)]
pub struct MirrorRegistry {
    state: std::sync::Mutex<RegistryState>,
}

impl MirrorRegistry {
    fn lock(&self) -> std::sync::MutexGuard<'_, RegistryState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A live receiver of the open mirror `session_id`.
    pub fn subscribe(&self, session_id: &str) -> Option<broadcast::Receiver<ChatEvent>> {
        let id = Uuid::parse_str(session_id).ok()?;
        self.lock().open.get(&id).map(|(_, tx)| tx.subscribe())
    }

    /// Whether `session_id` is a mirror still open here.
    pub fn is_open(&self, session_id: &str) -> bool {
        Uuid::parse_str(session_id).is_ok_and(|id| self.lock().open.contains_key(&id))
    }

    /// Whether `session_id` is a mirror this process knows (written or found by a live tap).
    pub fn is_known_mirror(&self, session_id: &str) -> bool {
        Uuid::parse_str(session_id).is_ok_and(|id| self.lock().known.contains_key(&id))
    }

    /// The open mirrors.
    pub fn open_ids(&self) -> Vec<Uuid> {
        self.lock().open.keys().copied().collect()
    }

    /// The mirror `id` is being deleted: it stops being open now, and its tap drops it at its
    /// next look, never writing to it again nor writing another one for its call. `true`:
    /// it was open here.
    pub fn deleted(&self, id: Uuid) -> bool {
        let mut state = self.lock();
        let was_open = state.open.remove(&id).is_some();
        if was_open {
            state.deleted.insert(id);
        }
        for claim in state.claims.values_mut() {
            if claim.mirror == Some(id) {
                claim.closed = true;
            }
        }
        was_open
    }

    fn claim(&self, parent: &str, call: &str, owner: u64) -> ClaimOutcome {
        let mut state = self.lock();
        let key = (parent.to_string(), call.to_string());
        if state.claims.contains_key(&key) {
            return ClaimOutcome::Refused;
        }
        if state.claims.len() >= MAX_CLAIMS {
            state.claims.retain(|_, c| !c.closed);
        }
        state.claims.insert(
            key,
            Claim {
                owner,
                mirror: None,
                closed: false,
            },
        );
        ClaimOutcome::Claimed
    }

    /// The claim of `owner` on (parent, call) failed or found the mirror over: marked over
    /// (`mirror` known) or given back (nothing to remember).
    fn settle(&self, parent: &str, call: &str, owner: u64, mirror: Option<Uuid>) {
        let mut state = self.lock();
        let key = (parent.to_string(), call.to_string());
        if state.claims.get(&key).is_some_and(|c| c.owner == owner) {
            match mirror {
                Some(id) => {
                    let claim = state.claims.get_mut(&key).expect("checked");
                    claim.mirror = Some(id);
                    claim.closed = true;
                }
                None => {
                    state.claims.remove(&key);
                }
            }
        }
    }

    /// `owner` opens the mirror `id` of (parent, call).
    fn bind(
        &self,
        parent: &str,
        call: &str,
        owner: u64,
        id: Uuid,
        tx: broadcast::Sender<ChatEvent>,
    ) {
        let mut state = self.lock();
        if let Some(claim) = state
            .claims
            .get_mut(&(parent.to_string(), call.to_string()))
            .filter(|c| c.owner == owner)
        {
            claim.mirror = Some(id);
        }
        state.open.insert(id, (owner, tx));
        state.known.insert(id, owner);
    }

    /// `owner` closed the mirror `id` of (parent, call). Nothing another owner holds is touched.
    fn closed(&self, parent: &str, call: &str, owner: u64, id: Uuid) {
        let mut state = self.lock();
        if let Some(claim) = state
            .claims
            .get_mut(&(parent.to_string(), call.to_string()))
            .filter(|c| c.owner == owner)
        {
            claim.closed = true;
        }
        if state.open.get(&id).is_some_and(|(o, _)| *o == owner) {
            state.open.remove(&id);
        }
    }

    /// Whether the open mirror `id` was deleted (consumed: asked once).
    fn take_deleted(&self, id: Uuid) -> bool {
        self.lock().deleted.remove(&id)
    }

    /// The tap `owner` ended (its parent session closed): everything it held is forgotten.
    /// Its mirrors are found in the graph again if the parent ever comes back.
    fn forget_owner(&self, owner: u64) {
        let mut state = self.lock();
        let mine: Vec<Uuid> = state
            .claims
            .values()
            .filter(|c| c.owner == owner)
            .filter_map(|c| c.mirror)
            .collect();
        state.claims.retain(|_, c| c.owner != owner);
        state.open.retain(|_, (o, _)| *o != owner);
        state.known.retain(|_, o| *o != owner);
        for id in mine {
            state.deleted.remove(&id);
        }
    }

    #[cfg(test)]
    fn claims_len(&self) -> usize {
        self.lock().claims.len()
    }
}

/// What a tap writes with.
#[derive(Clone)]
pub(crate) struct MirrorDeps {
    pub graph: Arc<dyn GraphStore>,
    pub event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    pub nats: Option<Arc<crate::events::NatsEmitter>>,
    pub registry: Arc<MirrorRegistry>,
}

enum Msg {
    Event(Box<ChatEvent>),
    TurnEnded,
    #[cfg(test)]
    Flush(tokio::sync::oneshot::Sender<()>),
}

/// The passive observer of ONE Claude Code session's events ([module documentation](self)).
/// Dropping the last handle closes the mirrors still open.
pub struct MirrorTap {
    tx: mpsc::UnboundedSender<Msg>,
    /// The top-level sub-agent calls still waiting for their result: what `observe` keeps
    /// of the parent's top-level tool events (the result of a Read or a Bash is never
    /// cloned). Touched by `observe` only, one short lock, never awaited.
    watched: std::sync::Mutex<HashSet<String>>,
}

impl MirrorTap {
    /// Starts the tap of `parent_session_id` (its worker runs on the current runtime).
    /// `follows_out_of_turn`: the tap is also shown what the CLI prints between the
    /// parent's turns (agent engine), so a background sub-agent's mirror stays open past
    /// the turn until the CLI notifies its end ([module documentation](self)).
    pub(crate) fn spawn(
        parent_session_id: &str,
        deps: MirrorDeps,
        follows_out_of_turn: bool,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let worker = Worker {
            parent: parent_session_id.to_string(),
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            follows_out_of_turn,
            deps,
            calls: HashMap::new(),
            nested: HashMap::new(),
            mirrors: HashMap::new(),
            closed: HashSet::new(),
            parent_node: None,
        };
        tokio::spawn(worker.run(rx));
        Arc::new(Self {
            tx,
            watched: std::sync::Mutex::new(HashSet::new()),
        })
    }

    /// Sees one event of the parent, after it was emitted. Never blocks, never fails: what
    /// the mirror does not need is not even cloned.
    pub fn observe(&self, event: &ChatEvent) {
        let wanted = match event {
            _ if parent_of(event).is_some() => true,
            ChatEvent::ToolUse { id, tool, .. } if is_subagent_tool(tool) => {
                self.watch().insert(id.clone());
                true
            }
            ChatEvent::ToolUseInputResolved { id, .. } => self.watch().contains(id),
            ChatEvent::ToolResult { id, .. } | ChatEvent::ToolCancelled { id, .. } => {
                self.watch().remove(id)
            }
            // A background sub-agent's start (its task id) and end, as the CLI notifies them.
            ChatEvent::Workflow { subtype, .. } => {
                matches!(subtype.as_str(), "task_started" | "task_notification")
            }
            ChatEvent::Result { .. }
            | ChatEvent::StreamingStatus {
                is_streaming: false,
            } => {
                let _ = self.tx.send(Msg::TurnEnded);
                return;
            }
            _ => false,
        };
        if wanted {
            let _ = self.tx.send(Msg::Event(Box::new(event.clone())));
        }
    }

    fn watch(&self) -> std::sync::MutexGuard<'_, HashSet<String>> {
        self.watched.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The parent's turn ended (the historical engine says it explicitly).
    pub fn turn_ended(&self) {
        let _ = self.tx.send(Msg::TurnEnded);
    }

    /// Waits until the worker handled everything observed before (tests: no fixed sleep).
    #[cfg(test)]
    pub(crate) async fn flushed(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.tx.send(Msg::Flush(tx)).is_ok() {
            let _ = rx.await;
        }
    }
}

/// One open mirror.
struct Mirror {
    id: Uuid,
    next_seq: i64,
    tx: broadcast::Sender<ChatEvent>,
    opened: Instant,
}

/// A top-level sub-agent call of the parent.
struct Call {
    /// Its input as last known.
    input: serde_json::Value,
    /// Its immediate `tool_result` (a background launch) was seen.
    launched: bool,
    /// The CLI's id of its task (`task_started`): what names it in a notification that
    /// carries no `tool_use_id`.
    task_id: Option<String>,
}

impl Call {
    /// Launched with `run_in_background: true`: its `tool_result` only says it started.
    fn background(&self) -> bool {
        self.input
            .get("run_in_background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}

/// The tap's state, owned by its task: every write happens here, in order.
struct Worker {
    parent: String,
    /// This tap's token in the registry.
    owner: u64,
    /// It sees what the CLI prints between the parent's turns (agent engine).
    follows_out_of_turn: bool,
    deps: MirrorDeps,
    /// Top-level sub-agent calls of the parent not yet over.
    calls: HashMap<String, Call>,
    /// Calls made INSIDE a mirror (any depth): id -> the first-level call.
    nested: HashMap<String, String>,
    /// Open mirrors by first-level call id.
    mirrors: HashMap<String, Mirror>,
    /// First-level sub-agent calls that are over here (their mirror closed, held by another
    /// tap, deleted, or not writable): their later events are dropped. Only sub-agent calls.
    closed: HashSet<String>,
    parent_node: Option<ChatSessionNode>,
}

impl Worker {
    async fn run(mut self, mut rx: mpsc::UnboundedReceiver<Msg>) {
        while let Some(msg) = rx.recv().await {
            match msg {
                Msg::Event(event) => self.handle(*event).await,
                Msg::TurnEnded => self.turn_ended().await,
                #[cfg(test)]
                Msg::Flush(done) => {
                    let _ = done.send(());
                }
            }
        }
        // The parent session is gone (every handle of the tap dropped).
        self.close_all("The parent session closed before the sub-agent answered.")
            .await;
        self.deps.registry.forget_owner(self.owner);
    }

    async fn handle(&mut self, mut event: ChatEvent) {
        match parent_of(&event).map(str::to_string) {
            None => match &event {
                ChatEvent::ToolUse {
                    id, tool, input, ..
                } if is_subagent_tool(tool) => {
                    self.calls.insert(
                        id.clone(),
                        Call {
                            input: input.clone(),
                            launched: false,
                            task_id: None,
                        },
                    );
                }
                ChatEvent::ToolUseInputResolved { id, input, .. } => {
                    if let Some(call) = self.calls.get_mut(id) {
                        call.input = input.clone();
                    }
                }
                ChatEvent::ToolResult {
                    id,
                    result,
                    is_error,
                    ..
                } => {
                    let Some(call) = self.calls.get_mut(id) else {
                        return;
                    };
                    if call.background() && !*is_error {
                        // Only the launch (decision M3): the sub-agent goes on after it. Its
                        // mirror stays open until the CLI notifies its end, or the turn ends.
                        call.launched = true;
                        return;
                    }
                    let (id, text, is_error) = (id.clone(), result_text(result), *is_error);
                    self.close(&id, text, is_error, false).await;
                }
                ChatEvent::ToolCancelled { id, .. } if self.calls.contains_key(id) => {
                    let id = id.clone();
                    self.close(&id, "The sub-agent was cancelled.".to_string(), true, true)
                        .await;
                }
                ChatEvent::Workflow { subtype, data } if subtype == "task_started" => {
                    // A background sub-agent's task id: what a notification without a
                    // `tool_use_id` names it by.
                    let (Some(call), Some(task)) = (
                        data.get("tool_use_id").and_then(|v| v.as_str()),
                        data.get("task_id").and_then(|v| v.as_str()),
                    ) else {
                        return;
                    };
                    if let Some(call) = self.calls.get_mut(call) {
                        call.task_id = Some(task.to_string());
                    }
                }
                ChatEvent::Workflow { data, .. } => {
                    // The end of a background sub-agent, when the notification names its
                    // call: by its id, else by the task its `task_started` gave. A
                    // notification of a call the sub-agent made itself names neither.
                    let by_call = data
                        .get("tool_use_id")
                        .and_then(|v| v.as_str())
                        .filter(|id| self.calls.get(*id).is_some_and(|c| c.launched))
                        .map(str::to_string);
                    let by_task = || {
                        let task = data.get("task_id").and_then(|v| v.as_str())?;
                        self.calls
                            .iter()
                            .find(|(_, c)| c.launched && c.task_id.as_deref() == Some(task))
                            .map(|(id, _)| id.clone())
                    };
                    let Some(id) = by_call.or_else(by_task) else {
                        return;
                    };
                    let text = ["summary", "result", "output"]
                        .iter()
                        .find_map(|k| {
                            data.get(*k)
                                .and_then(|v| v.as_str())
                                .filter(|s| !s.trim().is_empty())
                        })
                        .unwrap_or("The background sub-agent ended.")
                        .to_string();
                    let (is_error, interrupted) = match data.get("status").and_then(|s| s.as_str())
                    {
                        None | Some("completed") => (false, false),
                        Some("killed" | "stopped" | "cancelled") => (true, true),
                        Some(_) => (true, false),
                    };
                    self.close(&id, text, is_error, interrupted).await;
                }
                _ => {}
            },
            Some(parent) => {
                let first = if self.calls.contains_key(&parent) {
                    parent.clone()
                } else if let Some(first) = self.nested.get(&parent) {
                    first.clone()
                } else {
                    // Its call was not seen as a sub-agent call by this tap.
                    return;
                };
                if self.closed.contains(&first) {
                    return;
                }
                if !self.mirrors.contains_key(&first) && !self.open_mirror(&first).await {
                    return;
                }
                if let ChatEvent::ToolUse { id, .. } = &event {
                    self.nested.insert(id.clone(), first.clone());
                }
                let Some(slot) = copied_parent_mut(&mut event) else {
                    return;
                };
                // The first level shows at the mirror's top level; deeper calls stay nested.
                if slot.as_deref() == Some(first.as_str()) {
                    *slot = None;
                }
                self.append(&first, event).await;
            }
        }
    }

    /// The parent's turn ended: every mirror still open is closed, but a launched
    /// background sub-agent's when this tap follows what the CLI prints out of turn (agent
    /// engine): that one goes on, fed by the out-of-turn stream, until the CLI notifies its
    /// end (or the parent session closes). On the historical engine its later events go
    /// through `oob_listener`, which no tap sees: it is closed here, saying the rest shows
    /// in the parent.
    async fn turn_ended(&mut self) {
        let follows = self.follows_out_of_turn;
        let open: Vec<String> = self.mirrors.keys().cloned().collect();
        for call in open {
            match self.calls.get(&call).map(|c| c.launched) {
                Some(true) if follows => {}
                Some(true) => {
                    self.close(
                        &call,
                        "The sub-agent runs in the background: what it does after this turn shows in the parent conversation."
                            .to_string(),
                        false,
                        true,
                    )
                    .await;
                }
                _ => {
                    self.close(
                        &call,
                        "The parent turn ended before the sub-agent answered.".to_string(),
                        true,
                        true,
                    )
                    .await;
                }
            }
        }
        // A call of the ended turn that never got a mirror is over too, but a background
        // sub-agent that goes on (its mirror may open on its first out-of-turn event).
        self.calls.retain(|_, c| follows && c.launched);
        let calls = &self.calls;
        self.nested.retain(|_, first| calls.contains_key(first));
        self.closed.retain(|call| calls.contains_key(call));
    }

    /// The parent's node (read once).
    async fn parent_node(&mut self) -> Option<ChatSessionNode> {
        if self.parent_node.is_none() {
            let uuid = Uuid::parse_str(&self.parent).ok()?;
            self.parent_node = self.deps.graph.get_chat_session(uuid).await.ok().flatten();
        }
        self.parent_node.clone()
    }

    /// The mirror of `call`, found or written, held by this tap alone. `false`: there is none
    /// to write into here (another tap holds it, it is over, or it could not be written).
    async fn open_mirror(&mut self, call: &str) -> bool {
        let registry = Arc::clone(&self.deps.registry);
        if let ClaimOutcome::Refused = registry.claim(&self.parent, call, self.owner) {
            self.closed.insert(call.to_string());
            return false;
        }
        // Claimed: nobody else in this process writes it. A mirror written before (another
        // process, a tap since gone) is in the graph.
        let existing = self.stored_mirror(call).await;
        let id = match existing {
            Some(id) if self.ended_on_its_result(id).await => {
                registry.settle(&self.parent, call, self.owner, Some(id));
                self.closed.insert(call.to_string());
                return false;
            }
            Some(id) => id,
            None => match self.write_mirror(call).await {
                Some(id) => id,
                None => {
                    registry.settle(&self.parent, call, self.owner, None);
                    self.closed.insert(call.to_string());
                    return false;
                }
            },
        };
        let next_seq = self
            .deps
            .graph
            .get_latest_chat_event_seq(id)
            .await
            .unwrap_or(0)
            + 1;
        let (tx, _) = broadcast::channel(MIRROR_BROADCAST);
        registry.bind(&self.parent, call, self.owner, id, tx.clone());
        self.mirrors.insert(
            call.to_string(),
            Mirror {
                id,
                next_seq,
                tx,
                opened: Instant::now(),
            },
        );
        self.streaming(id, true);
        // A mirror written now opens on the call's prompt.
        if existing.is_none() {
            if let Some(prompt) = self.prompt(call) {
                self.append(call, ChatEvent::UserMessage { content: prompt })
                    .await;
            }
        }
        true
    }

    /// A mirror of `call` the graph already holds (written by an earlier tap or process).
    async fn stored_mirror(&self, call: &str) -> Option<Uuid> {
        let parent = Uuid::parse_str(&self.parent).ok()?;
        let children = self.deps.graph.get_session_children(parent).await.ok()?;
        children.into_iter().find_map(|child| {
            let linked = matches!(
                child.spawned_by.as_deref().and_then(SpawnedBy::from_json_str),
                Some(SpawnedBy::Conversation { parent_session_id, tool_use_id: Some(t) })
                    if parent_session_id == parent && t == call
            );
            (linked && child.mirror_of.is_some()).then_some(child.id)
        })
    }

    /// Whether the stored mirror `id` already ends on its `result`.
    async fn ended_on_its_result(&self, id: Uuid) -> bool {
        let Ok(latest) = self.deps.graph.get_latest_chat_event_seq(id).await else {
            return false;
        };
        self.deps
            .graph
            .get_chat_events(id, (latest - 1).max(0), 1)
            .await
            .map(|tail| tail.iter().any(|e| e.event_type == "result"))
            .unwrap_or(false)
    }

    /// The call's `prompt`.
    fn prompt(&self, call: &str) -> Option<String> {
        self.calls
            .get(call)
            .and_then(|c| c.input.get("prompt"))
            .and_then(|p| p.as_str())
            .filter(|p| !p.trim().is_empty())
            .map(str::to_string)
    }

    /// Writes the mirror node of `call`, and announces it.
    async fn write_mirror(&mut self, call: &str) -> Option<Uuid> {
        let parent = self.parent_node().await?;
        let input = self
            .calls
            .get(call)
            .map(|c| c.input.clone())
            .unwrap_or_default();
        let prompt = self.prompt(call);
        let description = input
            .get("description")
            .and_then(|d| d.as_str())
            .map(str::trim)
            .filter(|d| !d.is_empty())
            .map(str::to_string);
        let from_prompt = prompt
            .as_deref()
            .and_then(super::session_record::title_and_preview);
        let title = description
            .or_else(|| from_prompt.as_ref().map(|(t, _)| t.clone()))
            .unwrap_or_else(|| "Claude Code sub-agent".to_string());
        let spawned_by = SpawnedBy::Conversation {
            parent_session_id: parent.id,
            tool_use_id: Some(call.to_string()),
        }
        .to_json_string();
        let now = chrono::Utc::now();
        let node = ChatSessionNode {
            // A mirror has no process and no tools: never lifted.
            file_scope_lifted: false,
            id: Uuid::new_v4(),
            cli_session_id: None,
            project_slug: parent.project_slug.clone(),
            workspace_slug: parent.workspace_slug.clone(),
            cwd: parent.cwd.clone(),
            title: Some(title),
            model: parent.model.clone(),
            created_at: now,
            updated_at: now,
            message_count: i64::from(prompt.is_some()),
            total_cost_usd: None,
            conversation_id: None,
            preview: from_prompt.map(|(_, p)| p),
            permission_mode: parent.permission_mode.clone(),
            add_dirs: parent.add_dirs.clone(),
            spawned_by: Some(spawned_by.clone()),
            provider_id: parent.provider_id.clone(),
            routed_by: None,
            routing_mode: None,
            routing_pool: None,
            capabilities: None,
            engine: parent.engine.clone(),
            resume_token: None,
            execution_place: parent.execution_place,
            access: parent.access,
            owner: parent.owner.clone(),
            mirror_of: Some(MIRROR_OF_CLAUDE_CODE.to_string()),
        };
        if let Err(e) = self.deps.graph.create_chat_session(&node).await {
            tracing::warn!(parent = %self.parent, call, error = %e, "the mirror of a Claude Code sub-agent could not be written");
            return None;
        }
        if let Err(e) = self
            .deps
            .graph
            .create_spawned_by_relation(
                &node.id.to_string(),
                &self.parent,
                "conversation",
                None,
                None,
            )
            .await
        {
            tracing::warn!(mirror = %node.id, error = %e, "SPAWNED_BY of a mirror not written");
        }
        if let Some(emitter) = &self.deps.event_emitter {
            // Exactly the payload of every session (decision ed628dad): the client reads
            // `tool_use_id` and `mirror_of` from `/children` and the session itself.
            emitter.emit_created(
                crate::events::EntityType::ChatSession,
                &node.id.to_string(),
                super::manager::session_created_payload_of(
                    Some(&spawned_by),
                    node.project_slug.as_deref(),
                ),
                None,
            );
        }
        tracing::info!(parent = %self.parent, call, mirror = %node.id, "mirror of a Claude Code sub-agent opened");
        Some(node.id)
    }

    /// Whether the mirror of `call` was deleted under this tap: it is then dropped, never
    /// written again, and its call is over here.
    fn dropped_if_deleted(&mut self, call: &str) -> bool {
        let Some(id) = self.mirrors.get(call).map(|m| m.id) else {
            return false;
        };
        if !self.deps.registry.take_deleted(id) {
            return false;
        }
        self.mirrors.remove(call);
        self.forget_call(call);
        tracing::info!(parent = %self.parent, call, mirror = %id, "deleted mirror dropped by its tap");
        true
    }

    /// Stores (but a delta) and broadcasts one event of the mirror of `call`.
    async fn append(&mut self, call: &str, event: ChatEvent) {
        if self.dropped_if_deleted(call) {
            return;
        }
        let Some(mirror) = self.mirrors.get_mut(call) else {
            return;
        };
        if !matches!(event, ChatEvent::StreamDelta { .. }) {
            let record = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: mirror.id,
                seq: mirror.next_seq,
                event_type: event.event_type().to_string(),
                data: serde_json::to_string(&event).unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            mirror.next_seq += 1;
            if let Err(e) = self
                .deps
                .graph
                .store_chat_events(mirror.id, vec![record])
                .await
            {
                tracing::warn!(mirror = %mirror.id, error = %e, "an event of a mirror was not stored");
            }
        }
        let _ = mirror.tx.send(event.clone());
        if let Some(nats) = &self.deps.nats {
            nats.publish_chat_event(&mirror.id.to_string(), event);
        }
    }

    fn streaming(&self, id: Uuid, is_streaming: bool) {
        if let Some(emitter) = &self.deps.event_emitter {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                &id.to_string(),
                serde_json::json!({ "is_streaming": is_streaming }),
                None,
            );
        }
    }

    /// `call` is over here: its state goes, its later events are dropped.
    fn forget_call(&mut self, call: &str) {
        self.calls.remove(call);
        self.nested.retain(|_, first| first != call);
        self.closed.insert(call.to_string());
    }

    /// Ends the mirror of the sub-agent call `call` on its result.
    async fn close(&mut self, call: &str, text: String, is_error: bool, interrupted: bool) {
        if self.dropped_if_deleted(call) || !self.mirrors.contains_key(call) {
            self.forget_call(call);
            return;
        }
        let (id, elapsed) = {
            let mirror = &self.mirrors[call];
            (mirror.id, mirror.opened.elapsed())
        };
        let result = ChatEvent::Result {
            session_id: id.to_string(),
            duration_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            cost_usd: None,
            subtype: if is_error {
                "error_during_execution"
            } else {
                "success"
            }
            .to_string(),
            is_error,
            num_turns: None,
            result_text: Some(text),
            cost: None,
            usage: None,
            model: None,
            stop_reason: Some(
                if interrupted {
                    "interrupted"
                } else if is_error {
                    "error"
                } else {
                    "completed"
                }
                .to_string(),
            ),
        };
        self.append(call, result).await;
        if let Some(mirror) = self.mirrors.remove(call) {
            let _ = mirror.tx.send(ChatEvent::StreamingStatus {
                is_streaming: false,
            });
        }
        self.deps
            .registry
            .closed(&self.parent, call, self.owner, id);
        self.forget_call(call);
        self.streaming(id, false);
        // When it last moved, for a listing (its events do not touch the node).
        if let Err(e) = self
            .deps
            .graph
            .update_chat_session(id, None, None, None, None, None, None)
            .await
        {
            tracing::warn!(mirror = %id, error = %e, "the end of a mirror was not stamped on its node");
        }
        tracing::info!(parent = %self.parent, call, mirror = %id, "mirror of a Claude Code sub-agent closed");
    }

    /// Ends every mirror still open: the sub-agent was cut.
    async fn close_all(&mut self, why: &str) {
        let open: Vec<String> = self.mirrors.keys().cloned().collect();
        for call in open {
            self.close(&call, why.to_string(), true, true).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_text_of_a_result_is_its_string_or_its_text_blocks() {
        assert_eq!(result_text(&serde_json::json!("done")), "done");
        assert_eq!(
            result_text(&serde_json::json!([
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"}
            ])),
            "a\nb"
        );
        assert_eq!(result_text(&serde_json::Value::Null), "");
    }

    #[test]
    fn only_claude_codes_sub_agent_tools_are_mirrored() {
        assert!(is_subagent_tool("Task"));
        assert!(is_subagent_tool("Agent"));
        assert!(!is_subagent_tool("Bash"));
        assert!(!is_subagent_tool("Read"));
    }

    use super::test_support::{closed_mirrors, mirrors_of, thread, RecordingEmitter};
    use crate::events::CrudAction;
    use crate::neo4j::mock::MockGraphStore;

    const CALL: &str = "toolu_task";

    struct World {
        graph: Arc<MockGraphStore>,
        emitter: Arc<RecordingEmitter>,
        registry: Arc<MirrorRegistry>,
        parent: ChatSessionNode,
    }

    async fn world() -> World {
        let graph = Arc::new(MockGraphStore::new());
        let parent = ChatSessionNode {
            project_slug: Some("proj".into()),
            provider_id: Some("claude-code".into()),
            engine: Some(ChatSessionNode::ENGINE_AGENT.into()),
            owner: Some("owner-1".into()),
            ..crate::test_helpers::test_chat_session(Some("proj"))
        };
        graph.create_chat_session(&parent).await.unwrap();
        World {
            graph,
            emitter: Arc::default(),
            registry: Arc::default(),
            parent,
        }
    }

    impl World {
        /// The tap of the agent engine (it follows what the CLI prints out of turn).
        fn tap(&self) -> Arc<MirrorTap> {
            self.tap_of(true)
        }

        /// The tap of the historical engine (out of turn goes through `oob_listener`).
        fn legacy_tap(&self) -> Arc<MirrorTap> {
            self.tap_of(false)
        }

        fn tap_of(&self, follows_out_of_turn: bool) -> Arc<MirrorTap> {
            MirrorTap::spawn(
                &self.parent.id.to_string(),
                MirrorDeps {
                    graph: self.graph.clone(),
                    event_emitter: Some(self.emitter.clone()),
                    nats: None,
                    registry: Arc::clone(&self.registry),
                },
                follows_out_of_turn,
            )
        }
    }

    fn call(id: &str, tool: &str, input: serde_json::Value, parent: Option<&str>) -> ChatEvent {
        ChatEvent::ToolUse {
            id: id.into(),
            tool: tool.into(),
            input,
            parent_tool_use_id: parent.map(str::to_string),
            category: None,
            canonical: None,
        }
    }

    fn said(text: &str, parent: Option<&str>) -> ChatEvent {
        ChatEvent::AssistantText {
            content: text.into(),
            parent_tool_use_id: parent.map(str::to_string),
        }
    }

    fn answered(id: &str, text: &str, parent: Option<&str>) -> ChatEvent {
        ChatEvent::ToolResult {
            id: id.into(),
            result: serde_json::json!(text),
            is_error: false,
            parent_tool_use_id: parent.map(str::to_string),
        }
    }

    /// The events of a parent turn running one Task call, as both engines emit them: the
    /// call announced with an empty input, then resolved; the sub-agent's activity under
    /// it; the call's result; the parent goes on.
    fn task_turn() -> Vec<ChatEvent> {
        vec![
            call(CALL, "Task", serde_json::json!({}), None),
            ChatEvent::ToolUseInputResolved {
                id: CALL.into(),
                input: serde_json::json!({"description": "find the spec", "prompt": "Look for the spec of X"}),
                parent_tool_use_id: None,
            },
            said("looking", Some(CALL)),
            ChatEvent::StreamDelta {
                text: "live only".into(),
                parent_tool_use_id: Some(CALL.into()),
            },
            call(
                "toolu_read",
                "Read",
                serde_json::json!({"file_path": "a.md"}),
                Some(CALL),
            ),
            ChatEvent::PermissionRequest {
                id: "req-1".into(),
                tool: "Read".into(),
                input: serde_json::json!({}),
                parent_tool_use_id: Some(CALL.into()),
                category: None,
                canonical: None,
                tool_use_id: Some("toolu_read".into()),
                always: None,
            },
            answered("toolu_read", "the spec", Some(CALL)),
            said("found it", Some(CALL)),
            answered(CALL, "The spec is in a.md", None),
            said("the parent goes on", None),
        ]
    }

    /// P18c: one Task call → ONE mirror, linked by `spawned_by.tool_use_id`, marked
    /// `mirror_of: claude_code`, titled from the call's description, holding the prompt then
    /// the sub-agent's events at its top level (no permission request, no delta stored),
    /// closed with the call's result; one `created` (with the call id), then streaming
    /// true / false on the CRUD bus.
    #[tokio::test]
    async fn a_task_call_gets_one_mirror_holding_its_events_closed_by_its_result() {
        let w = world().await;
        let tap = w.tap();
        for event in task_turn() {
            tap.observe(&event);
        }
        let mirrors = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
        assert_eq!(mirrors.len(), 1, "exactly one mirror");
        let mirror = &mirrors[0];
        assert_eq!(mirror.mirror_of.as_deref(), Some(MIRROR_OF_CLAUDE_CODE));
        assert_eq!(mirror.title.as_deref(), Some("find the spec"));
        assert_eq!(
            SpawnedBy::from_json_str(mirror.spawned_by.as_deref().unwrap()),
            Some(SpawnedBy::Conversation {
                parent_session_id: w.parent.id,
                tool_use_id: Some(CALL.into()),
            })
        );
        assert_eq!(mirror.provider_id, w.parent.provider_id);
        assert_eq!(mirror.model, w.parent.model);
        assert_eq!(mirror.cwd, w.parent.cwd);
        assert_eq!(mirror.project_slug, w.parent.project_slug);
        assert_eq!(mirror.owner, w.parent.owner);

        let events = thread(w.graph.as_ref(), mirror.id).await;
        let kinds: Vec<&str> = events.iter().map(ChatEvent::event_type).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "assistant_text",
                "tool_use",
                "tool_result",
                "assistant_text",
                "result"
            ],
            "{events:?}"
        );
        assert!(
            matches!(&events[0], ChatEvent::UserMessage { content } if content == "Look for the spec of X")
        );
        assert!(
            events.iter().all(|e| parent_of(e).is_none()),
            "the sub-agent shows at the mirror's top level: {events:?}"
        );
        match events.last().unwrap() {
            ChatEvent::Result {
                result_text,
                is_error,
                ..
            } => {
                assert_eq!(result_text.as_deref(), Some("The spec is in a.md"));
                assert!(!is_error);
            }
            other => panic!("{other:?}"),
        }

        let crud = w.emitter.of(&mirror.id.to_string());
        assert_eq!(crud.len(), 3, "{crud:?}");
        assert_eq!(crud[0].0, CrudAction::Created);
        assert_eq!(
            crud[0].1,
            serde_json::json!({
                "project_slug": "proj",
                "spawned_by": {
                    "type": "conversation",
                    "parent_session_id": w.parent.id.to_string(),
                },
            }),
            "exactly the payload of every session (decision ed628dad)"
        );
        assert_eq!(
            crud[1],
            (
                CrudAction::Updated,
                serde_json::json!({"is_streaming": true})
            )
        );
        assert_eq!(
            crud[2],
            (
                CrudAction::Updated,
                serde_json::json!({"is_streaming": false})
            )
        );
        assert!(!w.registry.is_open(&mirror.id.to_string()), "closed");
        // Nothing of the parent was written by the tap: it only observes.
        assert!(thread(w.graph.as_ref(), w.parent.id).await.is_empty());
    }

    /// A sub-agent's own Task call is not mirrored on its own: its events go into the
    /// first-level mirror, nested under that call there.
    #[tokio::test]
    async fn a_nested_task_call_stays_in_its_first_level_mirror() {
        let w = world().await;
        let tap = w.tap();
        let events = [
            call(
                CALL,
                "Task",
                serde_json::json!({"description": "outer", "prompt": "p"}),
                None,
            ),
            call(
                "toolu_inner",
                "Task",
                serde_json::json!({"description": "inner", "prompt": "q"}),
                Some(CALL),
            ),
            said("deep", Some("toolu_inner")),
            answered("toolu_inner", "inner done", Some(CALL)),
            answered(CALL, "outer done", None),
        ];
        for event in &events {
            tap.observe(event);
        }
        let mirrors = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
        assert_eq!(mirrors.len(), 1, "only the first level is mirrored");
        let events = thread(w.graph.as_ref(), mirrors[0].id).await;
        assert!(
            events.iter().any(|e| matches!(e,
                ChatEvent::AssistantText { content, parent_tool_use_id: Some(p) }
                    if content == "deep" && p == "toolu_inner")),
            "the deeper event keeps its parent, under the nested call: {events:?}"
        );
    }

    /// Never two mirrors for one call: the same events seen again — by a new tap of the
    /// same process (a resume), then by a new process (a restart: an empty registry) — write
    /// no second mirror, no second `created`, and nothing after the mirror's result.
    #[tokio::test]
    async fn a_replay_never_writes_a_second_mirror() {
        let w = world().await;
        let first = w.tap();
        for event in task_turn() {
            first.observe(&event);
        }
        let mirror = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await[0].clone();
        let stored = thread(w.graph.as_ref(), mirror.id).await.len();

        let again = w.tap();
        for event in task_turn() {
            again.observe(&event);
        }
        let restarted = World {
            graph: w.graph.clone(),
            emitter: w.emitter.clone(),
            registry: Arc::default(),
            parent: w.parent.clone(),
        };
        let fresh = restarted.tap();
        for event in task_turn() {
            fresh.observe(&event);
        }
        again.flushed().await;
        fresh.flushed().await;
        assert_eq!(mirrors_of(w.graph.as_ref(), w.parent.id).await.len(), 1);
        assert_eq!(thread(w.graph.as_ref(), mirror.id).await.len(), stored);
        let created = w
            .emitter
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.action == CrudAction::Created)
            .count();
        assert_eq!(created, 1);
    }

    /// The parent's turn ends (or the session closes) before the call's result: the mirror
    /// is closed anyway, as cut, and stops streaming.
    #[tokio::test]
    async fn the_end_of_the_parent_turn_closes_an_open_mirror() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Agent",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        tap.observe(&ChatEvent::StreamingStatus {
            is_streaming: false,
        });
        let mirror = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await[0].clone();
        match thread(w.graph.as_ref(), mirror.id).await.last().unwrap() {
            ChatEvent::Result {
                is_error,
                stop_reason,
                ..
            } => {
                assert!(is_error);
                assert_eq!(stop_reason.as_deref(), Some("interrupted"));
            }
            other => panic!("{other:?}"),
        }
        let crud = w.emitter.of(&mirror.id.to_string());
        assert_eq!(
            crud.last().unwrap(),
            &(
                CrudAction::Updated,
                serde_json::json!({"is_streaming": false})
            )
        );

        // The session closes (the tap is dropped) while a second call runs.
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        drop(tap);
        closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
    }

    /// Only a sub-agent call is mirrored: events under another tool, or under a call this
    /// tap never saw, write nothing.
    #[tokio::test]
    async fn activity_under_another_tool_is_never_mirrored() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            "toolu_bash",
            "Bash",
            serde_json::json!({"command": "ls"}),
            None,
        ));
        tap.observe(&said("not a sub-agent", Some("toolu_bash")));
        tap.observe(&said("unknown call", Some("toolu_unknown")));
        tap.observe(&answered("toolu_bash", "ok", None));
        tap.flushed().await;
        assert!(mirrors_of(w.graph.as_ref(), w.parent.id).await.is_empty());
        assert!(w.emitter.events.lock().unwrap().is_empty());
    }

    /// The open mirror is followed live and counts as streaming until its result.
    #[tokio::test]
    async fn an_open_mirror_is_live_until_its_result() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        tap.flushed().await;
        let id = mirrors_of(w.graph.as_ref(), w.parent.id).await[0]
            .id
            .to_string();
        assert!(w.registry.is_open(&id));
        let mut rx = w.registry.subscribe(&id).expect("followed live");
        assert!(w.registry.open_ids().iter().any(|o| o.to_string() == id));
        tap.observe(&said("more", Some(CALL)));
        let seen = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(seen, ChatEvent::AssistantText { ref content, parent_tool_use_id: None } if content == "more")
        );
        tap.observe(&answered(CALL, "done", None));
        closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await;
        assert!(!w.registry.is_open(&id));
    }

    /// Two Task calls of one turn, their sub-agents speaking in turns: two mirrors, each
    /// with its own events only, each closed by its own result.
    #[tokio::test]
    async fn two_parallel_task_calls_get_two_mirrors() {
        let w = world().await;
        let tap = w.tap();
        for event in [
            call(
                "toolu_a",
                "Task",
                serde_json::json!({"description": "A", "prompt": "do A"}),
                None,
            ),
            call(
                "toolu_b",
                "Task",
                serde_json::json!({"description": "B", "prompt": "do B"}),
                None,
            ),
            said("a1", Some("toolu_a")),
            said("b1", Some("toolu_b")),
            said("a2", Some("toolu_a")),
            answered("toolu_b", "B done", None),
            said("b-late", Some("toolu_b")),
            said("a3", Some("toolu_a")),
            answered("toolu_a", "A done", None),
        ] {
            tap.observe(&event);
        }
        let mirrors = closed_mirrors(w.graph.as_ref(), w.parent.id, 2).await;
        assert_eq!(mirrors.len(), 2);
        for mirror in mirrors {
            let events = thread(w.graph.as_ref(), mirror.id).await;
            let said: Vec<String> = events
                .iter()
                .filter_map(|e| match e {
                    ChatEvent::AssistantText { content, .. } => Some(content.clone()),
                    _ => None,
                })
                .collect();
            let result = match events.last() {
                Some(ChatEvent::Result { result_text, .. }) => result_text.clone().unwrap(),
                other => panic!("{other:?}"),
            };
            match mirror.title.as_deref() {
                Some("A") => {
                    assert_eq!(said, ["a1", "a2", "a3"]);
                    assert_eq!(result, "A done");
                }
                Some("B") => {
                    assert_eq!(said, ["b1"], "nothing after its result");
                    assert_eq!(result, "B done");
                }
                other => panic!("{other:?}"),
            }
        }
    }

    /// The user stops the parent in the middle of a sub-agent: the call is cancelled, the
    /// mirror ends `interrupted`, stops streaming, and nothing is written after.
    #[tokio::test]
    async fn an_interrupt_in_the_middle_of_a_sub_agent_closes_its_mirror() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        tap.observe(&ChatEvent::ToolCancelled {
            id: CALL.into(),
            parent_tool_use_id: None,
        });
        tap.observe(&said("too late", Some(CALL)));
        tap.observe(&ChatEvent::StreamingStatus {
            is_streaming: false,
        });
        tap.flushed().await;
        let mirror = mirrors_of(w.graph.as_ref(), w.parent.id).await[0].clone();
        let events = thread(w.graph.as_ref(), mirror.id).await;
        assert!(
            matches!(events.last(), Some(ChatEvent::Result { is_error: true, stop_reason: Some(r), .. }) if r == "interrupted"),
            "{events:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, ChatEvent::Result { .. }))
                .count(),
            1
        );
        assert!(!w.registry.is_open(&mirror.id.to_string()));
        let crud = w.emitter.of(&mirror.id.to_string());
        assert_eq!(
            crud.last().unwrap(),
            &(
                CrudAction::Updated,
                serde_json::json!({"is_streaming": false})
            )
        );
    }

    /// A background call (`run_in_background`): its immediate result (the launch) does not
    /// close the mirror; the sub-agent's later events still go in; the CLI's notification
    /// naming the call closes it with its summary. On the historical engine (no out-of-turn
    /// view), the turn's end closes it as a success that says the rest shows in the parent.
    #[tokio::test]
    async fn a_background_agent_call_keeps_its_mirror_open_past_its_launch() {
        let w = world().await;
        let tap = w.tap();
        let input =
            serde_json::json!({"description": "bg", "prompt": "go", "run_in_background": true});
        tap.observe(&call(CALL, "Agent", input.clone(), None));
        tap.observe(&said("started", Some(CALL)));
        tap.observe(&answered(CALL, "Async agent launched successfully", None));
        tap.observe(&said("still working", Some(CALL)));
        tap.flushed().await;
        let mirror = mirrors_of(w.graph.as_ref(), w.parent.id).await[0].clone();
        assert!(
            w.registry.is_open(&mirror.id.to_string()),
            "open past its launch"
        );
        tap.observe(&ChatEvent::Workflow {
            subtype: "task_notification".into(),
            data: serde_json::json!({"tool_use_id": CALL, "status": "completed", "summary": "bg done"}),
        });
        tap.flushed().await;
        let events = thread(w.graph.as_ref(), mirror.id).await;
        assert!(events.iter().any(
            |e| matches!(e, ChatEvent::AssistantText { content, .. } if content == "still working")
        ));
        assert!(
            matches!(events.last(), Some(ChatEvent::Result { is_error: false, result_text: Some(t), stop_reason: Some(r), .. }) if t == "bg done" && r == "completed"),
            "{events:?}"
        );

        // Historical engine, no notification: the end of the turn closes it, as a success.
        let w = world().await;
        let tap = w.legacy_tap();
        tap.observe(&call(CALL, "Agent", input, None));
        tap.observe(&said("started", Some(CALL)));
        tap.observe(&answered(CALL, "Async agent launched successfully", None));
        tap.observe(&ChatEvent::StreamingStatus {
            is_streaming: false,
        });
        let mirror = closed_mirrors(w.graph.as_ref(), w.parent.id, 1).await[0].clone();
        assert!(matches!(
            thread(w.graph.as_ref(), mirror.id).await.last(),
            Some(ChatEvent::Result { is_error: false, result_text: Some(t), .. }) if t.contains("background")
        ));
    }

    /// Agent engine: a background sub-agent outlives the parent's turn. Its mirror stays
    /// open across the turn's end (and the next turn's), takes what the CLI prints out of
    /// turn, and is closed by the notification that names its task (`task_started` gave the
    /// task id; this notification carries no `tool_use_id`), as a failure when the task
    /// failed; the node's `updated_at` moves. A notification of a call the sub-agent made
    /// itself closes nothing.
    #[tokio::test]
    async fn a_background_sub_agent_is_followed_past_the_turn_until_its_notification() {
        let w = world().await;
        let tap = w.tap();
        let input =
            serde_json::json!({"description": "bg", "prompt": "go", "run_in_background": true});
        tap.observe(&call(CALL, "Agent", input, None));
        tap.observe(&ChatEvent::Workflow {
            subtype: "task_started".into(),
            data: serde_json::json!({"task_id": "a1b2", "tool_use_id": CALL, "is_backgrounded": true}),
        });
        tap.observe(&answered(CALL, "Async agent launched successfully", None));
        tap.observe(&said("launched", None));
        tap.observe(&said("started", Some(CALL)));
        tap.observe(&ChatEvent::StreamingStatus {
            is_streaming: false,
        });
        tap.flushed().await;
        let mirror = mirrors_of(w.graph.as_ref(), w.parent.id).await[0].clone();
        assert!(
            w.registry.is_open(&mirror.id.to_string()),
            "open past the parent's turn"
        );

        // Out of turn: the sub-agent works, and the parent may even play another turn.
        tap.observe(&call(
            "toolu_bash",
            "Bash",
            serde_json::json!({"command": "sleep 6"}),
            Some(CALL),
        ));
        tap.observe(&ChatEvent::Workflow {
            subtype: "task_notification".into(),
            data: serde_json::json!({"task_id": "b9", "tool_use_id": "toolu_bash", "status": "completed", "summary": "Sleep"}),
        });
        tap.observe(&answered("toolu_bash", "slept", Some(CALL)));
        tap.observe(&ChatEvent::Result {
            session_id: w.parent.id.to_string(),
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
        });
        tap.observe(&said("it broke", Some(CALL)));
        tap.flushed().await;
        assert!(
            w.registry.is_open(&mirror.id.to_string()),
            "still open: its own notification has not come"
        );
        tap.observe(&ChatEvent::Workflow {
            subtype: "task_notification".into(),
            data: serde_json::json!({"task_id": "a1b2", "status": "failed", "summary": "could not finish"}),
        });
        tap.observe(&said("late", Some(CALL)));
        tap.flushed().await;
        assert!(!w.registry.is_open(&mirror.id.to_string()));

        let events = thread(w.graph.as_ref(), mirror.id).await;
        let words: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ChatEvent::AssistantText { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(words, ["started", "it broke"], "{events:?}");
        assert!(events.iter().any(|e| matches!(e,
            ChatEvent::ToolResult { id, parent_tool_use_id: None, .. } if id == "toolu_bash")));
        assert!(
            matches!(events.last(), Some(ChatEvent::Result { is_error: true, result_text: Some(t), stop_reason: Some(r), .. })
                if t == "could not finish" && r == "error"),
            "{events:?}"
        );
        let node = w.graph.get_chat_session(mirror.id).await.unwrap().unwrap();
        assert!(node.updated_at > node.created_at, "the close is stamped");
        assert_eq!(node.message_count, 1, "the prompt is its only user message");
    }

    /// Two live taps of one parent (an old turn's tap still finishing after a resume): only
    /// the one that claimed the call writes its mirror; the other neither writes into it nor,
    /// when it ends, removes what the first holds. Once the first ends, the registry keeps
    /// nothing of it.
    #[tokio::test]
    async fn a_second_live_tap_never_writes_into_a_mirror_the_first_holds() {
        let w = world().await;
        let first = w.tap();
        first.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        first.observe(&said("from the first", Some(CALL)));
        first.flushed().await;
        let mirror = mirrors_of(w.graph.as_ref(), w.parent.id).await[0].clone();
        let stored = thread(w.graph.as_ref(), mirror.id).await.len();

        let second = w.tap();
        second.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        second.observe(&said("from the second", Some(CALL)));
        second.observe(&answered(CALL, "second's result", None));
        second.flushed().await;
        drop(second);
        // The second ended: the first still holds its open mirror.
        first.flushed().await;
        assert_eq!(thread(w.graph.as_ref(), mirror.id).await.len(), stored);
        assert!(w.registry.is_open(&mirror.id.to_string()));

        first.observe(&answered(CALL, "first's result", None));
        first.flushed().await;
        assert!(matches!(
            thread(w.graph.as_ref(), mirror.id).await.last(),
            Some(ChatEvent::Result { result_text: Some(t), .. }) if t == "first's result"
        ));
        assert_eq!(mirrors_of(w.graph.as_ref(), w.parent.id).await.len(), 1);
        drop(first);
        for _ in 0..400 {
            if w.registry.claims_len() == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(
            w.registry.claims_len(),
            0,
            "a tap that ends forgets its claims"
        );
        assert!(!w.registry.is_known_mirror(&mirror.id.to_string()));
    }

    /// An open mirror deleted under its tap: never written again, not written anew for its
    /// call, no longer open.
    #[tokio::test]
    async fn a_deleted_open_mirror_is_dropped_by_its_tap() {
        let w = world().await;
        let tap = w.tap();
        tap.observe(&call(
            CALL,
            "Task",
            serde_json::json!({"prompt": "go"}),
            None,
        ));
        tap.observe(&said("working", Some(CALL)));
        tap.flushed().await;
        let mirror = mirrors_of(w.graph.as_ref(), w.parent.id).await[0].clone();
        assert!(w.registry.deleted(mirror.id), "it was open");
        w.graph.delete_chat_session(mirror.id).await.unwrap();
        assert!(!w.registry.is_open(&mirror.id.to_string()));
        tap.observe(&said("after the delete", Some(CALL)));
        tap.observe(&answered(CALL, "done", None));
        tap.flushed().await;
        assert!(
            mirrors_of(w.graph.as_ref(), w.parent.id).await.is_empty(),
            "not written anew"
        );
        assert!(
            !thread(w.graph.as_ref(), mirror.id)
                .await
                .iter()
                .any(|e| matches!(e, ChatEvent::AssistantText { content, .. } if content == "after the delete")),
        );
    }

    /// The store refuses the mirror: nothing is written, nothing announced, and the tap
    /// drops the call's events (the parent never sees any of it: the tap only observes).
    #[tokio::test]
    async fn a_mirror_the_store_refuses_is_dropped_quietly() {
        let w = world().await;
        w.graph
            .fail_session_writes
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let tap = w.tap();
        for event in task_turn() {
            tap.observe(&event);
        }
        tap.flushed().await;
        assert!(mirrors_of(w.graph.as_ref(), w.parent.id).await.is_empty());
        assert!(w.emitter.events.lock().unwrap().is_empty());
        assert!(w.registry.open_ids().is_empty());
    }
}

/// What the tests of the mirrors share (`mirror::tests`, the e2e tests of both engines).
#[cfg(test)]
pub(crate) mod test_support {
    use std::time::Duration;

    use super::*;
    use crate::events::{CrudAction, CrudEvent, EntityType, EventEmitter};

    /// Keeps every CRUD event emitted.
    #[derive(Default)]
    pub(crate) struct RecordingEmitter {
        pub events: std::sync::Mutex<Vec<CrudEvent>>,
    }

    impl EventEmitter for RecordingEmitter {
        fn emit(&self, event: CrudEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    impl RecordingEmitter {
        /// The CRUD events of the chat session `id`: (action, payload).
        pub(crate) fn of(&self, id: &str) -> Vec<(CrudAction, serde_json::Value)> {
            self.events
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.entity_type == EntityType::ChatSession && e.entity_id == id)
                .map(|e| (e.action.clone(), e.payload.clone()))
                .collect()
        }
    }

    /// The mirrors the graph holds for `parent`.
    pub(crate) async fn mirrors_of(graph: &dyn GraphStore, parent: Uuid) -> Vec<ChatSessionNode> {
        graph
            .get_session_children(parent)
            .await
            .unwrap()
            .into_iter()
            .filter(|c| c.mirror_of.is_some())
            .collect()
    }

    /// The stored thread of `id`, in order.
    pub(crate) async fn thread(graph: &dyn GraphStore, id: Uuid) -> Vec<ChatEvent> {
        let mut records = graph.get_chat_events(id, 0, 1000).await.unwrap();
        records.sort_by_key(|r| r.seq);
        records
            .iter()
            .map(|r| serde_json::from_str(&r.data).unwrap())
            .collect()
    }

    /// Waits (10 s at most) until `parent` has `n` mirrors whose threads end on a `result`.
    pub(crate) async fn closed_mirrors(
        graph: &dyn GraphStore,
        parent: Uuid,
        n: usize,
    ) -> Vec<ChatSessionNode> {
        for _ in 0..400 {
            let mirrors = mirrors_of(graph, parent).await;
            let mut closed = 0;
            for m in &mirrors {
                if matches!(
                    thread(graph, m.id).await.last(),
                    Some(ChatEvent::Result { .. })
                ) {
                    closed += 1;
                }
            }
            if mirrors.len() >= n && closed >= n {
                return mirrors;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("{n} closed mirror(s) of {parent} expected");
    }
}
