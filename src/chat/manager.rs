//! ChatManager — orchestrates Claude Code CLI sessions via Nexus SDK
//!
//! Manages active InteractiveClient sessions with auto-resume for inactive ones.
//!
//! Architecture:
//! - Each session spawns an `InteractiveClient` (Nexus SDK) subprocess
//! - Messages are streamed via `broadcast::channel` to WebSocket subscribers
//! - Structured events are persisted in Neo4j with sequence numbers for replay
//! - Inactive sessions are persisted in Neo4j with `cli_session_id` for resume
//! - A cleanup task periodically closes timed-out sessions

use super::config::ChatConfig;
use super::post_tool_hook;
use super::skill_hook;
use super::types::{
    classify_api_error, truncate_snippet, BackgroundTaskInfo, BackgroundTaskKind, ChatEvent,
    ChatEventPage, ChatRequest, CreateSessionResponse, MessageSearchHit, MessageSearchResult,
    PendingMessage, SessionActivity, SessionWorkLog,
};
use crate::events::attention::{notify_attention, AttentionReason, AttentionSubject};
use crate::meilisearch::SearchStore;
use crate::neo4j::models::ChatEventRecord;
use crate::neo4j::models::ChatSessionNode;
use crate::neo4j::GraphStore;
use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use nexus_claude::{
    build_hook_response_json, dispatch_hook_from_registry, is_hook_callback,
    memory::{ContextInjector, ConversationMemoryManager, MemoryConfig},
    ClaudeCodeOptions, ContentBlock, ContentValue, InteractiveClient, McpServerConfig, Message,
    StreamDelta, StreamEventData,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::expand_tilde;

/// Broadcast channel buffer size for WebSocket subscribers
const BROADCAST_BUFFER: usize = 256;

/// How long a resumed CLI must stay up after its handshake before the resume is
/// trusted (an unknown `--resume` target makes it exit within a second).
const RESUME_GRACE: std::time::Duration = std::time::Duration::from_millis(1500);

/// OOB-triggered `stream_response` rate cap for interactive sessions.
/// A misbehaving Monitor or background Bash that emits constantly could
/// otherwise loop the session and inflate the LLM bill — this caps the
/// number of OOB-driven LLM turns to 50 over a rolling 5-min window
/// (T7 of plan 9a1684b2).
pub(crate) const OOB_TRIGGER_CAP_INTERACTIVE: u32 = 50;
/// Same cap for runner sessions, more conservative since they run
/// autonomously without a human watching.
pub(crate) const OOB_TRIGGER_CAP_RUNNER: u32 = 5;
/// Rolling window for the OOB trigger cap.
pub(crate) const OOB_TRIGGER_WINDOW_SECS: u64 = 300;

/// User-driven cancel-tools rate cap. The "Stop" button on a running tool
/// could be click-spammed; capping at 10 per minute per session prevents a
/// flood of `pgrep -P` invocations and keeps the CLI healthy. Much more
/// generous than the OOB cap because each cancel is a deliberate user
/// action, not an autonomous LLM trigger (T2 of plan 28e9afe3).
pub(crate) const CANCEL_TOOLS_CAP: u32 = 10;
/// Rolling window for the cancel-tools rate cap (60s).
pub(crate) const CANCEL_TOOLS_WINDOW_SECS: u64 = 60;

/// User-driven cancel-task rate cap. Distinct from `CANCEL_TOOLS_CAP`:
/// `cancel_task` targets a single background subprocess (one Monitor /
/// Bash bg) rather than every descendant. The frontend popover may
/// surface a Stop button per task, so click-spam is more plausible —
/// the cap is more generous (30 per 5 min) but still bounded to keep
/// pgrep-based PID enumeration cheap. Plan 754a1379 (T9).
pub(crate) const CANCEL_TASK_CAP: u32 = 30;
/// Rolling window for the cancel-task rate cap (300s = 5 minutes).
pub(crate) const CANCEL_TASK_WINDOW_SECS: u64 = 300;

/// Grace period applied between marking a `BackgroundTaskInfo` for removal
/// (`pending_removal_at`) and physically purging it from the
/// `active_background_tasks` map. 5 seconds absorbs in-flight
/// `BackgroundOutput` ticks that arrive between the cancel SIGINT and
/// the subprocess actually dying — they still match an entry in the map
/// and are routed correctly to their (now greyed-out) MonitorCard,
/// instead of becoming orphans on the frontend. Plan 754a1379 (T12).
pub(crate) const BACKGROUND_TASK_PURGE_GRACE_SECS: u64 = 5;
/// How often the background-tasks poller wakes up to apply the grace
/// period purge AND the idle-based death detection. 5s matches the
/// grace period — i.e. a marked entry is purged on the next tick
/// after it ages past the grace, with at most an extra tick of
/// latency. Plan 754a1379 (T12 + T3 REMOVE-on-death).
pub(crate) const BACKGROUND_TASKS_POLL_INTERVAL_SECS: u64 = 5;

/// Idle threshold for the death detector (T3 REMOVE-on-death side).
/// An entry whose `last_seen_at` is older than this is marked
/// `pending_removal_at = Some(now)` by the poller, then purged after
/// the usual grace period.
///
/// Default 30 minutes — generous on purpose. Monitors with sparse
/// output (e.g. `tail -F build.log` during a long compile) can be
/// silent for ~10 min legitimately. The death detector errs on the
/// side of false negatives (let live tasks linger) over false
/// positives (reap a still-alive Monitor and confuse the user).
///
/// Subprocesses that died ungracefully and never emit again will
/// still be cleaned up — just with a 30-min lag rather than instantly.
/// Users who want immediate cleanup can `cancel_task` directly.
pub(crate) const BACKGROUND_TASK_IDLE_DEATH_SECS: u64 = 1800;

/// In-memory facts about the running CLIs (see
/// [`ChatManager::live_session_snapshot`]).
#[derive(Debug, Default, Clone)]
pub struct LiveSessionSnapshot {
    /// Sessions whose CLI is alive.
    pub live: std::collections::HashSet<Uuid>,
    /// Among them, those currently streaming a turn.
    pub streaming: std::collections::HashSet<Uuid>,
    /// Per live session, the `request_id`s of permissions still waiting.
    pub pending_permissions: std::collections::HashMap<Uuid, std::collections::HashSet<String>>,
    /// Per live session, the active background subprocesses it is waiting
    /// on, as `(monitors, bash)`. A session can stream nothing for minutes
    /// while one of these runs, so this is what keeps a working session
    /// from looking dead.
    pub background_tasks: std::collections::HashMap<Uuid, (usize, usize)>,
}

/// Count the live `(monitors, bash)` among a session's tracked background
/// tasks.
///
/// Entries carrying `pending_removal_at` are deliberately NOT counted. They
/// are already dying — cancelled, or detected dead — and only linger in the
/// map for the poller's 5 s grace period so late `BackgroundOutput` ticks
/// still route correctly. Advertising them would keep a cancelled watch on
/// screen for five seconds after the user killed it.
pub(crate) fn count_background_tasks<'a>(
    tasks: impl IntoIterator<Item = &'a BackgroundTaskInfo>,
) -> (usize, usize) {
    let mut monitors = 0usize;
    let mut bash = 0usize;
    for info in tasks {
        if info.pending_removal_at.is_some() {
            continue;
        }
        match info.kind {
            BackgroundTaskKind::Monitor => monitors += 1,
            BackgroundTaskKind::BashBackground => bash += 1,
        }
    }
    (monitors, bash)
}

impl LiveSessionSnapshot {
    /// The activity of one session, as the API reports it. A session the
    /// snapshot has never heard of is quiet by definition — which is the
    /// honest answer, not a missing value.
    pub fn activity_for(&self, id: Uuid) -> SessionActivity {
        let (monitors, bash_tasks) = self.background_tasks.get(&id).copied().unwrap_or((0, 0));
        SessionActivity {
            live: self.live.contains(&id),
            streaming: self.streaming.contains(&id),
            pending_permissions: self.pending_permissions.get(&id).map_or(0, |p| p.len()),
            monitors,
            bash_tasks,
        }
    }
}

/// An active chat session with a live Claude CLI subprocess
pub struct ActiveSession {
    /// The anchor mode of the session, fixed when it was opened or resumed, and the
    /// cache its turns share: nothing re-reads the environment per turn.
    pub anchor: super::anchor_resolver::AnchorSession,
    /// Persistent broadcast sender — one per session lifetime, NOT replaced per message
    pub events_tx: broadcast::Sender<ChatEvent>,
    /// When the session was last active
    pub last_activity: Instant,
    /// The CLI session ID (for persistence / resume)
    pub cli_session_id: Option<String>,
    /// Handle to the InteractiveClient (behind Mutex for &mut access)
    pub client: Arc<Mutex<InteractiveClient>>,
    /// Flag to signal the stream loop to stop and release the client lock
    pub interrupt_flag: Arc<AtomicBool>,
    /// Nexus conversation memory manager (records messages for persistence)
    pub memory_manager: Option<Arc<Mutex<ConversationMemoryManager>>>,
    /// Monotonically increasing sequence number for persisted events
    pub next_seq: Arc<AtomicI64>,
    /// Queue of messages waiting to be sent (received while streaming)
    pub pending_messages: Arc<Mutex<VecDeque<PendingMessage>>>,
    /// Whether a stream is currently in progress
    pub is_streaming: Arc<AtomicBool>,
    /// Accumulated text from stream_delta during the current stream (for mid-stream join)
    pub streaming_text: Arc<Mutex<String>>,
    /// Accumulated structured events during the current stream (for mid-stream join).
    /// Contains all non-StreamDelta events (ToolUse, ToolResult, AssistantText, etc.)
    /// that haven't been persisted yet. Cleared at stream start/end.
    pub streaming_events: Arc<Mutex<Vec<ChatEvent>>>,
    /// Current permission mode for this session (updated on mid-session changes)
    pub permission_mode: Option<String>,
    /// Current model for this session (updated on mid-session model changes)
    pub model: Option<String>,
    /// Active protocol run ID — set when this session runs within a protocol FSM context.
    /// Injected into SessionHints on close and used to tag trajectories with DURING_RUN.
    pub protocol_run_id: Option<uuid::Uuid>,
    /// Current protocol state name (e.g., "implement", "review") at session creation time.
    pub protocol_state: Option<String>,
    /// SDK control receiver for permission requests (`can_use_tool`).
    /// Taken once from `InteractiveClient::take_sdk_control_receiver()` at session
    /// creation and reused across all `stream_response` invocations. Wrapped in
    /// `Arc<Mutex<Option<...>>>` so each `stream_response` can temporarily take
    /// ownership during streaming, then put it back when the stream ends.
    pub sdk_control_rx:
        Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::Receiver<serde_json::Value>>>>,
    /// Cloned stdin sender for writing control responses (e.g., permission
    /// allow/deny) to the CLI subprocess **without** taking the `client` lock.
    /// This is critical because `stream_response` holds the client lock for the
    /// entire duration of streaming — if `send_permission_response` tried to
    /// take the same lock, it would deadlock.
    pub stdin_tx: Option<tokio::sync::mpsc::Sender<String>>,
    /// PID of the CLI subprocess. Used to send SIGINT to the process group
    /// during interrupt, which cascades to all child processes (find, sleep,
    /// etc.) that would otherwise survive as orphans.
    /// Captured once after `client.connect()` — stable for the session lifetime.
    pub child_pid: Option<u32>,
    /// Cancellation token for NATS listener tasks spawned by this session.
    /// When the session is replaced (e.g., by `resume_session`), the old token
    /// is cancelled so that stale NATS listeners (interrupt, snapshot, RPC)
    /// shut down instead of accumulating across resumes/restarts.
    pub nats_cancel: CancellationToken,
    /// Cancellation token for the stream_response loop. Cancelled by `interrupt()`
    /// and by the NATS interrupt listener to immediately unblock the `tokio::select!`
    /// in `stream_response`, even when the CLI is silent (e.g., executing `sleep 60`).
    /// A NEW token is created at each stream start (CancellationToken is not resettable).
    pub interrupt_token: CancellationToken,
    /// Stores the original tool input for pending permission requests.
    /// Key: request_id, Value: the tool input JSON.
    /// When the user responds Allow, we include this input in `updatedInput`
    /// so the CLI doesn't lose the original command/parameters.
    pub pending_permission_inputs:
        Arc<tokio::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>>,
    /// Whether auto-continue is enabled for this session.
    /// When `true`, the backend automatically sends "Continue" after error_max_turns.
    /// Toggled via WebSocket `set_auto_continue` message.
    pub auto_continue: Arc<AtomicBool>,
    /// Number of auto-continue cycles consumed so far. Incremented each time
    /// `stream_response` auto-continues after error_max_turns. When this reaches
    /// `max_auto_continues`, auto_continue is disabled to prevent infinite loops.
    pub auto_continue_count: Arc<AtomicU32>,
    /// Maximum number of auto-continue cycles allowed. 0 = unlimited.
    /// Set to `RunnerConfig.max_auto_continues` for runner sessions, 0 for interactive.
    pub max_auto_continues: u32,
    /// RFC accumulator for detecting sustained architectural discussions.
    /// Persists across messages in a session — when 2+ consecutive responses
    /// contain RFC-qualifying patterns, auto-creates an RFC draft note.
    pub rfc_accumulator: Arc<Mutex<super::observation_detector::RfcAccumulator>>,
    /// Reasoning path tracker for Hebbian reinforcement on session close.
    /// Records reasoning tree paths during enrichment; reinforced when the session ends.
    pub reasoning_path_tracker: super::feedback::ReasoningPathTracker,
    /// Whether objective tracking is enabled for this session.
    /// When `true`, the backend checks for pending plan tasks after each stream turn
    /// where the agent did NOT use any tools (= conclusion/summary mode).
    /// If pending tasks exist, a reminder is injected as SystemHint.
    pub objective_tracking: bool,
    /// Cooldown counter for objective reminders. Counts stream turns since the last
    /// reminder was injected. Must reach `OBJECTIVE_REMINDER_COOLDOWN` before another
    /// reminder is sent, to avoid spamming the agent.
    pub objective_reminder_turns_since: Arc<AtomicU32>,
    /// Objective reminders injected back to back with no productive tool use in between.
    /// Capped at `OBJECTIVE_REMINDER_MAX_IN_A_ROW`: past it the tracker stays silent until the
    /// agent works again (a reminder that is answered by text only, every turn, is a loop).
    pub objective_reminders_in_a_row: Arc<AtomicU32>,
    /// Accumulates work performed during the session: files modified, files read,
    /// steps completed, last tool used. Source of truth for resumption after
    /// compaction or max_turns.
    pub work_log: Arc<Mutex<SessionWorkLog>>,
    /// Sliding-window history of OOB-triggered `stream_response` spawn
    /// timestamps (T7 of plan 9a1684b2). Used by `chat::oob_listener` to
    /// enforce a rate cap that prevents a runaway Monitor / background
    /// Bash from looping the session indefinitely.
    pub oob_trigger_history: Arc<Mutex<VecDeque<Instant>>>,
    /// Maximum OOB-triggered turns within `oob_trigger_window`. Set to
    /// `OOB_TRIGGER_CAP_INTERACTIVE` (50) for interactive sessions and
    /// `OOB_TRIGGER_CAP_RUNNER` (5) for runner sessions.
    pub oob_trigger_cap: u32,
    /// Rolling window for the OOB trigger cap. Defaults to 5 minutes.
    pub oob_trigger_window: Duration,
    /// Anti-spam flag: set to true the first time the cap is hit in a
    /// window so we emit the SystemHint warning only once per window.
    /// Reset to false the next time we observe `len < cap` after draining.
    pub oob_capped_warned: Arc<AtomicBool>,
    /// Sliding-window history of `cancel_running_tools` invocations on this
    /// session. Used to enforce a per-session click-spam cap so the user's
    /// "Stop" button cannot saturate `pgrep` or the CLI (T2 of plan
    /// 28e9afe3).
    pub cancel_tools_history: Arc<Mutex<VecDeque<Instant>>>,
    /// Maximum cancel_tools invocations allowed within `cancel_tools_window`.
    /// Defaults to `CANCEL_TOOLS_CAP` (10).
    pub cancel_tools_cap: u32,
    /// Rolling window for the cancel_tools rate cap. Defaults to 60s.
    pub cancel_tools_window: Duration,
    /// Background subprocesses currently attached to this session
    /// (`Monitor`, `Bash run_in_background`). Keyed by `BackgroundTaskInfo.id`
    /// — which equals the SDK `tool_use_id` of the originating invocation
    /// (and the `correlation_id` on every `BackgroundOutput` event from
    /// this subprocess). Plan 754a1379 (T2). Mutated by the OOB lifecycle
    /// hook (T3), the cancel_task path (T7), the grace-period purge (T12),
    /// and the recovery rebuild (T13). The map is **ephemeral** — it is
    /// reconstructed at create_session/resume_session via
    /// `get_descendant_pids(child_pid)` (decision 33bce470, audit note
    /// 2482f4e0).
    pub active_background_tasks: Arc<Mutex<HashMap<String, BackgroundTaskInfo>>>,
    /// How many background tasks the CLI itself says are running, from its
    /// last `background_tasks_changed` system message (0 until one arrives).
    ///
    /// `active_background_tasks` is this server's own bookkeeping and can
    /// only infer liveness from output: a command that runs silently (a
    /// build, a test run redirected to a file) emits nothing for its whole
    /// life. The CLI knows. Written by the OOB listener, read by the
    /// background-task purge and by the idle-session cleanup.
    pub cli_background_tasks: Arc<AtomicUsize>,
    /// Sliding-window history of `cancel_task` invocations on this session.
    /// Same shape and intent as `cancel_tools_history`, but scoped to the
    /// granular per-task cancel path introduced by plan 754a1379 (T9).
    pub cancel_task_history: Arc<Mutex<VecDeque<Instant>>>,
    /// Maximum cancel_task invocations allowed within `cancel_task_window`.
    /// Defaults to `CANCEL_TASK_CAP` (30).
    pub cancel_task_cap: u32,
    /// Rolling window for the cancel_task rate cap. Defaults to 300s (5 min).
    pub cancel_task_window: Duration,
}

/// Where a message or permission answer ended up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryRoute {
    /// Handed to the CLI running in this process.
    Local,
    /// Proxied to the instance that owns the session.
    Remote,
    /// Nobody held the session: the CLI was respawned (`resume_session`).
    Resumed,
    /// The local send failed (dead CLI) and the session was resumed instead.
    ResumedAfterSendFailure,
}

/// Why a permission answer was not delivered.
#[derive(Debug)]
pub enum PermissionDeliveryError {
    /// No instance holds the session: the CLI that asked is gone.
    SessionDead(String),
    /// The CLI is alive but the request is no longer waiting (already
    /// answered, or never asked).
    NotPending,
    Failed(anyhow::Error),
}

impl std::fmt::Display for PermissionDeliveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionDead(id) => write!(f, "Session {id} not found or inactive"),
            Self::NotPending => write!(f, "Permission request is no longer pending"),
            Self::Failed(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PermissionDeliveryError {}

/// Why a user message was not delivered.
#[derive(Debug)]
pub enum MessageDeliveryError {
    /// Nobody held the session and `resume_session` failed.
    Resume(anyhow::Error),
    /// The local send failed and the resume fallback failed too.
    SendAndResume {
        send: anyhow::Error,
        resume: anyhow::Error,
    },
}

/// Result of `ChatManager::cancel_running_tools`. Surfaced to REST/WS
/// callers so the UI can display "killed N processes" feedback or
/// degrade gracefully when the rate cap is hit (T2 of plan 28e9afe3).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CancelToolsResult {
    /// PID of the Claude Code CLI subprocess (for client-side display).
    /// `None` when the platform doesn't support PID capture or the
    /// session has no live subprocess.
    pub cli_pid: Option<u32>,
    /// PIDs that received `SIGINT`. Empty when no descendants existed
    /// at the time of the call (e.g., agent was thinking, no tool
    /// running) — this is **not** an error condition.
    pub killed_pids: Vec<u32>,
    /// `true` when the rate cap was hit and the request was refused
    /// (no SIGINT sent, no NATS publish). Caller should map to HTTP
    /// 429 or display a "slow down" toast.
    pub capped: bool,
}

/// Outcome of [`ChatManager::interrupt_scoped`]. Surfaced to REST callers
/// so a client can tell a real interrupt from a silent no-op: before this,
/// `interrupt()` returned `Ok(())` whether it had stopped a live turn or
/// found nothing at all, and the UI had no way to tell the two apart —
/// it just span on "Stopping…" forever.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterruptOutcome {
    /// `true` when the session was found in `active_sessions` and the
    /// interrupt was actually applied (flag + token + control_request).
    /// `false` means nothing local was stopped — the session may live on
    /// another instance (see `routed`), or not be running at all.
    pub delivered: bool,
    /// Where the interrupt went: `"local"`, `"nats"` (not local, published
    /// for whichever instance owns the session) or `"none"` (nowhere).
    pub routed: String,
    /// PID of the Claude Code CLI subprocess, when known.
    pub cli_pid: Option<u32>,
    /// Descendant PIDs that received `SIGINT`. Always empty when the call
    /// was made with `kill_tools = false`.
    pub killed_pids: Vec<u32>,
}

/// Result of `ChatManager::cancel_task`. Surfaced to REST/WS callers
/// so the UI can update the cancelled task's status and surface rate
/// cap hits gracefully.
///
/// Plan 754a1379 (T7) introduced this with V1 semantics (map-side
/// cancel only, `killed_pids` always empty). Plan fc35b25e (T4)
/// upgraded the implementation to actually SIGINT the subprocess
/// subtree via `kill_subtree`; `killed_pids` now reports the PIDs
/// that received the signal in the common path. Empty `killed_pids`
/// is now an edge case (claim race or subprocess crashed before
/// PID discovery — falls back to V1 map-side cancel).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CancelTaskResult {
    /// `tool_use_id` (≡ map key, ≡ `correlation_id`) of the task that
    /// was cancelled. Echoed back so the caller doesn't have to
    /// remember which click corresponded to which task on a busy UI.
    pub task_id: String,
    /// PIDs (root + descendants) that received `SIGINT` from
    /// `kill_subtree`. Non-empty in the common V2 path. Empty when
    /// the task entry's `pid` was still `None` at cancel time
    /// (claim race or subprocess crashed before discovery — V1
    /// fallback engaged).
    pub killed_pids: Vec<u32>,
    /// `true` when the rate cap was hit and the request was refused
    /// (no map mutation, no broadcast, no kill).
    pub capped: bool,
}

/// Runtime-mutable environment config for Claude CLI subprocess.
///
/// These fields can be changed at runtime via the REST API and are
/// persisted to config.yaml. They live behind `RwLock` on `ChatManager`
/// to allow concurrent read access from `build_options` while supporting
/// write access from update handlers.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeEnvConfig {
    pub process_path: Option<String>,
    pub claude_cli_path: Option<String>,
    pub auto_update_cli: bool,
    pub auto_update_app: bool,
}

/// The concrete native harness of each stored instance, by id.
pub(crate) type NativeProbers =
    HashMap<String, Arc<nexus_claude::providers::native::NativeProvider>>;

/// A built provider and the stored record it was built from.
pub(crate) type NativeCacheEntry = (
    super::provider::settings::InstanceRecord,
    Arc<dyn nexus_claude::agent::AgentProvider>,
);

/// Appended to the system prompt of a session on another machine: the model must
/// not promise tools it was not given.
const REMOTE_SESSION_NOTICE: &str = "This session runs on a remote machine through SSH. The \
project-orchestrator tools are not available in it: work with the files and the shell of that machine only.";

/// What `build_agent_spec` needs to know about a session being opened or resumed.
pub(crate) struct AgentSpecInput<'a> {
    pub cwd: &'a str,
    pub model: &'a str,
    pub system_prompt: &'a str,
    pub permission_mode: Option<&'a str>,
    pub add_dirs: &'a [String],
    pub user_claims: Option<&'a crate::auth::jwt::Claims>,
    pub session_id: &'a str,
    /// A provider other than Claude Code: restricted tool profile in the token,
    /// unless the session is in `trust` ([`third_party_tool_profile`]).
    pub third_party: bool,
    pub max_tokens: Option<u64>,
    pub kind: nexus_claude::agent::ProviderKind,
    /// Working directory ON THE REMOTE machine, for a `claude_code_remote`
    /// instance. `Some` also means: no project-orchestrator MCP server, no extra
    /// directories (the provider refuses both, and the local paths mean nothing
    /// there).
    pub remote_cwd: Option<&'a str>,
    /// The knowledge-graph hooks to give the session (`None`: none).
    pub hooks: Option<AgentHookScope>,
}

/// What [`ChatManager::graph_hook_table`] needs.
pub(crate) struct GraphHookInput {
    pub session_id: String,
    pub context_source: CompactionContextSource,
    pub work_log: Arc<Mutex<SessionWorkLog>>,
    /// Register the per-tool hooks (skill activation, redirect advice). Off for runner
    /// sessions, which already have their task context in the prompt.
    pub tool_knowledge: bool,
    /// Where to announce a compaction. `None`: the engine announces it itself.
    pub announce: Option<broadcast::Sender<ChatEvent>>,
}

/// Which hooks a session of the agent engine gets.
pub(crate) struct AgentHookScope {
    pub project_slug: Option<String>,
    /// The task the session works on (a runner session): its context guides compaction.
    pub task_id: Option<Uuid>,
    /// A runner session: no per-tool knowledge hooks.
    pub runner: bool,
}

/// What `authorize_provider_use` checks before a session's content is sent.
pub(crate) struct ProviderUse<'a> {
    pub provider_id: &'a str,
    pub model: &'a str,
    pub mode: nexus_claude::agent::PolicyMode,
    pub project_slug: Option<&'a str>,
    pub claims: Option<&'a crate::auth::jwt::Claims>,
    pub session_id: &'a str,
}

/// What `open_agent_session` opens.
struct AgentOpen<'a> {
    request: &'a ChatRequest,
    session_id: Uuid,
    provider_id: &'a str,
    model: &'a str,
    system_prompt: &'a str,
    add_dirs: &'a [String],
    project_slug: Option<&'a str>,
    /// History relayed from another provider, sent in front of the first message
    /// and stated on the thread.
    relay: Option<&'a super::relay::RelayedFrom>,
    /// What the session may do (decided by the caller, persisted on the node).
    access: super::provider::policy::SessionAccess,
}

/// What the per-turn router of a session needs to know about its opening.
pub(crate) struct OpeningTurn<'a> {
    /// The request named its model.
    pub explicit_model: bool,
    /// The request named its provider (`routed_by: request`): never moved automatically.
    pub provider_imposed: bool,
    /// The session continues a conversation the router moved here (`moved_by: auto`): its
    /// model was chosen by the router, not imposed, and its next turn does not move again.
    pub moved_in: bool,
    /// The mode the request asked for; it replaces the one of the settings for this conversation.
    pub routing_mode: Option<super::provider::cognitive::ProviderRoutingMode>,
    /// The pairs the request restricted routing to (`mixed`).
    pub routing_pool: Option<Vec<super::types::RoutingPoolEntry>>,
    pub permission_mode: Option<&'a str>,
    /// The message of the turn about to start.
    pub message: &'a str,
    /// Index of the first turn the router counts itself (legacy engine).
    pub next_turn: u32,
}

/// Manages chat sessions and their lifecycle
pub struct ChatManager {
    pub(crate) graph: Arc<dyn GraphStore>,
    #[allow(dead_code)]
    pub(crate) search: Arc<dyn SearchStore>,
    pub(crate) config: ChatConfig,
    pub(crate) active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
    /// Sessions of the provider-neutral path (`CHAT_PROVIDER_PATH=agent`).
    pub(crate) agent_runtime: Arc<super::agent_runtime::AgentRuntime>,
    /// Where the agent path finds a provider instance.
    pub(crate) provider_source: Arc<dyn super::agent_runtime::ProviderSource>,
    /// The cognitive router (R2), when wired. `None` keeps the declarative
    /// resolution exactly as it was.
    pub(crate) cognitive_routing: Option<super::provider::cognitive::decider::CognitiveRouting>,
    /// Open cognitive decisions of live chat sessions (session id -> decision id),
    /// closed with what happened when the session is closed.
    pub(crate) open_decisions: std::sync::Mutex<std::collections::HashMap<String, Uuid>>,
    /// Native providers built for stored instances, by instance id; an entry is
    /// reused while the stored record is unchanged.
    pub(crate) native_cache: Arc<RwLock<HashMap<String, NativeCacheEntry>>>,
    /// The concrete native harness behind a cache entry, for capability probes.
    pub(crate) native_probers: Arc<RwLock<NativeProbers>>,
    /// Nexus memory injector for conversation persistence
    pub(crate) context_injector: Option<Arc<ContextInjector>>,
    /// Memory config (for creating ConversationMemoryManagers)
    pub(crate) memory_config: Option<MemoryConfig>,
    /// Event emitter for CRUD events (streaming status changes)
    pub(crate) event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
    /// Optional NATS emitter for cross-instance chat event publishing
    pub(crate) nats: Option<Arc<crate::events::NatsEmitter>>,
    /// Runtime-mutable permission config (updated via REST API)
    pub(crate) permission_config: Arc<RwLock<super::config::PermissionConfig>>,
    /// Path to config.yaml for persisting permission changes (None = no persistence)
    pub(crate) config_yaml_path: Option<std::path::PathBuf>,
    /// Runtime-mutable environment config (PATH, CLI path, auto-update).
    /// Updated via REST API and persisted to config.yaml.
    pub(crate) env_config: Arc<RwLock<RuntimeEnvConfig>>,
    /// Pre-enrichment pipeline that runs before each LLM call.
    /// Enriches the user message with context from the knowledge graph.
    pub(crate) enrichment_pipeline: Arc<super::enrichment::EnrichmentPipeline>,
    /// Trajectory collector for neural routing feedback loop.
    /// When Some, `close_session()` calls `end_session()` to finalize trajectories.
    /// Behind a RwLock to allow hot-swap when collection is enabled at runtime via API.
    pub(crate) trajectory_collector:
        Arc<std::sync::RwLock<Option<std::sync::Arc<neural_routing_runtime::TrajectoryCollector>>>>,
    /// Optional reasoning tree engine for StatusInjectionStage.
    /// When Some, the status stage can build reasoning trees from the knowledge graph.
    pub(crate) reasoning_engine: Option<Arc<crate::reasoning::ReasoningTreeEngine>>,
    /// Whether neural routing is enabled (hot-swappable at runtime).
    pub(crate) neural_routing_enabled: Arc<std::sync::atomic::AtomicBool>,
    /// Dual-track router that runs both heuristic and neural routing in parallel.
    /// When Some and neural_routing_enabled is true, replaces the default HeuristicRouter.
    pub(crate) dual_track_router: Arc<std::sync::RwLock<Option<super::routing::DualTrackRouter>>>,
    /// Runtime DualTrackRouter (NN router) for trajectory-based action suggestions.
    /// When Some, `build_system_prompt()` queries this router after compose() to populate
    /// dashboard metrics and log NN route matches.
    pub(crate) nn_router: Option<Arc<tokio::sync::RwLock<neural_routing_runtime::DualTrackRouter>>>,
    /// MCP Federation registry for external server connections.
    /// The McpFederationStage injects tool availability into prompts when servers are connected.
    pub(crate) mcp_registry: crate::mcp_federation::registry::SharedRegistry,
    /// Secrets vault: mints the per-session vault token and masks agent output.
    /// None in tests and when the server runs without one.
    pub(crate) vault: Option<Arc<crate::vault::VaultService>>,
    /// Per-turn model routing (`full` mode): the shared decider and the router of each
    /// live session. Empty until [`ChatManager::with_turn_decider`].
    pub(crate) turn_routing: Arc<super::agent_hooks::TurnRouting>,
    /// The `refs_v1` switch (on unless `REFS_V1=0`): whether the API layer folds
    /// `refs` into messages and the server announces the capability.
    pub(crate) refs_v1: bool,
    /// Where the documents attached to a message live: the agent engine reads the
    /// attached images from it to send them inline (`documents.storage_dir`).
    pub(crate) document_store: crate::documents::store::DocumentStore,
    /// Root of the native sessions' transcripts (`<root>/<instance id>/<id>.json`, P14);
    /// `None`: kept in memory, lost with the process (nexus' default, the tests' too).
    pub(crate) native_transcripts: Option<std::path::PathBuf>,
    /// How far the anchor resolver drives the context (`PO_ANCHOR_CONTEXT`, default `shadow`).
    pub(crate) anchor_mode: super::anchor_resolver::AnchorContextMode,
    /// What the turns of every session share in anchor mode (resolutions, shadow runs).
    pub(crate) anchor_cache: Arc<super::anchor_resolver::AnchorCache>,
}

// ============================================================================
// CompactionNotifier — HookCallback that emits ChatEvent::CompactionStarted
// ============================================================================

/// Context source for CompactionNotifier — determines which builder method to call.
#[derive(Debug, Clone)]
pub(crate) enum CompactionContextSource {
    /// Runner mode: task_id known, plan_id resolved at hook time via GraphStore.
    Task(Uuid),
    /// Interactive mode: project slug known.
    Session(String),
    /// No context available (backward-compatible fallback).
    None,
}

/// Hook callback that emits a `ChatEvent::CompactionStarted` when the CLI
/// triggers a `PreCompact` event (context window compaction is about to start).
///
/// This gives the frontend a real-time notification so it can display a spinner
/// instead of leaving the user staring at silence during compaction.
///
/// Additionally, builds contextual `custom_instructions` via `CompactionContextBuilder`
/// to guide Claude's compaction — preserving key concepts (function names, decisions,
/// constraints) in the compacted summary.
///
/// The callback always returns `continue_: true` — it observes, never blocks.
pub(crate) struct CompactionNotifier {
    /// Broadcast sender for chat events (same channel as stream_response uses)
    events_tx: broadcast::Sender<ChatEvent>,
    /// NATS emitter for cross-instance propagation (None in tests or when NATS is disabled)
    nats: Option<Arc<crate::events::NatsEmitter>>,
    /// Session ID for NATS subject routing
    session_id: String,
    /// Graph store for building compaction context (None = no custom_instructions)
    graph: Option<Arc<dyn GraphStore>>,
    /// Context source: task (runner) or session (interactive)
    context_source: CompactionContextSource,
    /// Session work log for enriching custom_instructions during compaction
    work_log: Option<Arc<tokio::sync::Mutex<SessionWorkLog>>>,
}

impl CompactionNotifier {
    pub fn new(
        events_tx: broadcast::Sender<ChatEvent>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
        session_id: String,
    ) -> Self {
        Self {
            events_tx,
            nats,
            session_id,
            graph: None,
            context_source: CompactionContextSource::None,
            work_log: None,
        }
    }

    /// Attach a graph store and context source for building custom_instructions.
    pub fn with_context(
        mut self,
        graph: Arc<dyn GraphStore>,
        source: CompactionContextSource,
    ) -> Self {
        self.graph = Some(graph);
        self.context_source = source;
        self
    }

    /// Attach a session work log for enriching custom_instructions with modified files.
    pub fn with_work_log(mut self, work_log: Arc<tokio::sync::Mutex<SessionWorkLog>>) -> Self {
        self.work_log = Some(work_log);
        self
    }

    /// Build custom instructions string from the context source.
    /// Returns None if no graph/context is available or on any error.
    async fn build_custom_instructions(&self) -> Option<String> {
        let graph = self.graph.as_ref()?;
        let builder = super::compaction_context::CompactionContextBuilder::new(graph.clone());

        let ctx = match &self.context_source {
            CompactionContextSource::Task(task_id) => {
                // Resolve plan_id from task_id
                let plan_id = match graph.get_plan_id_for_task(*task_id).await {
                    Ok(Some(pid)) => pid,
                    _ => {
                        warn!(task_id = %task_id, "Could not resolve plan_id for task — skipping custom_instructions");
                        return None;
                    }
                };
                builder.build_for_task(plan_id, *task_id).await
            }
            CompactionContextSource::Session(slug) => {
                builder.build_for_session(Some(slug.as_str())).await
            }
            CompactionContextSource::None => return None,
        };

        // Take work_log snapshot if available
        let work_log_snapshot = if let Some(ref wl) = self.work_log {
            Some(wl.lock().await.snapshot())
        } else {
            None
        };

        match ctx {
            Ok(context) => {
                let instructions = context.to_custom_instructions(work_log_snapshot.as_ref());
                if instructions.is_empty() {
                    None
                } else {
                    Some(instructions)
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to build compaction context — skipping custom_instructions");
                None
            }
        }
    }
}

#[async_trait::async_trait]
impl nexus_claude::HookCallback for CompactionNotifier {
    async fn execute(
        &self,
        input: &nexus_claude::HookInput,
        _tool_use_id: Option<&str>,
        _context: &nexus_claude::HookContext,
    ) -> std::result::Result<nexus_claude::HookJSONOutput, nexus_claude::SdkError> {
        // Build custom instructions (async, best-effort)
        // This runs BEFORE emitting the event so the instructions are ready for the output.
        let custom_instructions = self.build_custom_instructions().await;

        if let nexus_claude::HookInput::PreCompact(pre_compact) = input {
            let event = ChatEvent::CompactionStarted {
                trigger: pre_compact.trigger.clone(),
            };
            // Best-effort local broadcast — receivers may have been dropped (no subscribers)
            let _ = self.events_tx.send(event.clone());
            // Cross-instance propagation via NATS (fire-and-forget, tokio::spawn inside)
            if let Some(ref nats) = self.nats {
                nats.publish_chat_event(&self.session_id, event);
            }
            info!(
                trigger = %pre_compact.trigger,
                session_id = %self.session_id,
                has_custom_instructions = custom_instructions.is_some(),
                "PreCompact hook fired — emitted CompactionStarted event"
            );
        }
        // Always allow compaction to proceed.
        // custom_instructions is passed via `reason` field — the CLI uses this as
        // feedback context for the compaction summary. The SDK's SyncHookJSONOutput
        // does not have a dedicated `custom_instructions` output field (it's input-only
        // on PreCompactHookInput), so `reason` is the semantic equivalent for output.
        Ok(nexus_claude::HookJSONOutput::Sync(
            nexus_claude::SyncHookJSONOutput {
                continue_: Some(true),
                reason: custom_instructions,
                ..Default::default()
            },
        ))
    }
}

// ============================================================================
// Pure helpers (testable without ChatManager)
// ============================================================================

/// `subtype` of the system message that stands in for a CLI message withheld
/// because it held a secret that could not be masked (fail closed).
pub(crate) const MASKING_FAILED_SUBTYPE: &str = "po_masking_failed";

/// What the user sees in place of a withheld message.
pub(crate) const MASKING_FAILED_MESSAGE: &str =
    "A message from the agent was withheld: it contained a secret that could not be masked.";

/// Extracted protocol context from a `spawned_by` JSON payload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpawnedByContext {
    pub parent_session_id: Option<String>,
    pub spawn_type: String,
    pub run_id: Option<Uuid>,
    pub task_id: Option<Uuid>,
    pub protocol_run_id: Option<Uuid>,
    pub protocol_state: Option<String>,
    /// Inherited scaffolding level from parent session (None = auto-detect).
    pub scaffolding_level: Option<u8>,
}

/// Classify a tool use as "conclusive" (wrapping-up, not productive work).
///
/// Conclusive tools are git finalization commands (commit, push, tag, status-after-commit).
/// When a turn contains ONLY conclusive tools and no productive ones, the objective
/// reminder should still fire — the agent may be concluding prematurely.
///
/// Returns `true` if the tool use is a finalization/wrap-up action.
fn is_conclusive_tool(tool_name: &str, input: &serde_json::Value) -> bool {
    if tool_name == "Bash" {
        if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
            let cmd_trimmed = cmd.trim();
            // Git finalization commands — the agent is wrapping up, not doing productive work
            let conclusive_patterns = [
                "git commit",
                "git push",
                "git tag",
                "git status", // often run after commit to verify
                "git log",    // often run to show what was committed
            ];
            for pattern in &conclusive_patterns {
                if cmd_trimmed.starts_with(pattern)
                    || cmd_trimmed.contains(&format!("&& {}", pattern))
                    || cmd_trimmed.contains(&format!("; {}", pattern))
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Parse a `spawned_by` JSON string and extract parent session info + protocol FSM context.
///
/// Returns `None` if the JSON is invalid.
/// The `protocol_run_id` and `protocol_state` fields are used to tag trajectories
/// with `DURING_RUN` when a session operates within a protocol FSM.
pub fn parse_spawned_by(json_str: &str) -> Option<SpawnedByContext> {
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let parent_session_id = val
        .get("parent_session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let spawn_type = val
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("runner")
        .to_string();
    let run_id = val
        .get("run_id")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<Uuid>().ok());
    let task_id = val
        .get("task_id")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<Uuid>().ok());
    let protocol_run_id = val
        .get("protocol_run_id")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<Uuid>().ok());
    let protocol_state = val
        .get("protocol_state")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let scaffolding_level = val
        .get("scaffolding_level")
        .and_then(|v| v.as_u64())
        .map(|v| v.min(4) as u8);
    Some(SpawnedByContext {
        parent_session_id,
        spawn_type,
        run_id,
        task_id,
        protocol_run_id,
        protocol_state,
        scaffolding_level,
    })
}

/// Project id handed to every enrichment stage, resolved once from the
/// session's slug. Stages that read only `project_id` (reflex) were skipped
/// for every chat message when it was left `None`.
async fn enrichment_project_id(
    graph: &dyn crate::neo4j::traits::GraphStore,
    project_slug: Option<&str>,
) -> Option<Uuid> {
    graph
        .get_project_by_slug(project_slug?)
        .await
        .ok()
        .flatten()
        .map(|p| p.id)
}

/// The protocol context of a turn (what the Claude Code engine knows of a session
/// that runs inside a protocol FSM). Empty for a session that runs in none.
#[derive(Default)]
pub(crate) struct TurnProtocol {
    pub run_id: Option<Uuid>,
    pub state: Option<String>,
    pub reasoning_path_tracker: Option<super::feedback::ReasoningPathTracker>,
}

/// The project of a session that has no explicit `project_slug`, at the places
/// that used to infer it from the cwd (opening, resuming).
///
/// Modes `off` and `shadow`: the historical inference (it is only ever called
/// here and in [`super::anchor_resolver::decide_project`]). Mode `on`: the anchor
/// precedence, whose cwd inference is reserved to sessions with a real cwd.
/// A resolver error in `on` falls back to the historical path: a session never
/// loses its project to a read error.
pub(crate) async fn infer_session_project(
    graph: &dyn GraphStore,
    mode: super::anchor_resolver::AnchorContextMode,
    session_id: Option<Uuid>,
    place: super::neutral_place::ExecutionPlace,
    cwd: &str,
) -> Option<String> {
    use super::anchor_resolver::{
        decide_project, AnchorContextMode, CwdInference, GraphCwdInference, ProjectInputs,
    };
    let infer = GraphCwdInference(graph);
    if mode == AnchorContextMode::On {
        let anchors = match session_id {
            Some(id) => graph.list_session_anchors(id).await,
            None => Ok(Vec::new()),
        };
        let inputs = ProjectInputs {
            explicit_slug: None,
            place,
            cwd,
        };
        match async { decide_project(graph, &inputs, &anchors?, &infer).await }.await {
            Ok(d) => return d.project.map(|p| p.slug),
            Err(e) => {
                warn!(error = %e, "anchor project precedence failed, using the historical inference")
            }
        }
    }
    infer.infer(cwd).await
}

/// Run the resolver beside the historical path (mode `shadow`) without ever
/// delaying or breaking it: a detached task that journals and swallows errors.
/// It is not even started while the last run of the session is recent, and it
/// journals only when the anchors changed (see `AnchorCache`).
pub(crate) fn spawn_anchor_shadow(
    graph: Arc<dyn GraphStore>,
    cache: Arc<super::anchor_resolver::AnchorCache>,
    session_id: Uuid,
    explicit_slug: Option<String>,
    place: super::neutral_place::ExecutionPlace,
    cwd: String,
    legacy_project: Option<String>,
) {
    // Decided before spawning: a turn that is not due costs no task at all.
    if !cache.claim_shadow(session_id) {
        return;
    }
    tokio::spawn(async move {
        let inputs = super::anchor_resolver::ProjectInputs {
            explicit_slug: explicit_slug.as_deref(),
            place,
            cwd: &cwd,
        };
        let _ = super::anchor_resolver::run_shadow_cached(
            &cache,
            graph.as_ref(),
            session_id,
            &inputs,
            legacy_project.as_deref(),
            &super::anchor_resolver::GraphCwdInference(graph.as_ref()),
        )
        .await;
    });
}

/// The knowledge graph's context for one turn whose message is `message`, as the
/// markdown put in front of it (`None`: nothing to add). Both engines call it
/// before every turn — the Claude Code engine in `stream_response`, the agent
/// engine through [`ManagerTurnServices`] — so a message gets the same context
/// whatever drives the session.
///
/// `excluded_note_ids`: the notes the user pointed at with `#` (`refs::turn`): the
/// knowledge injection does not repeat them.
pub(crate) async fn enrichment_for_turn(
    graph: &Arc<dyn GraphStore>,
    pipeline: &super::enrichment::EnrichmentPipeline,
    session_id: &str,
    message: &str,
    protocol: TurnProtocol,
    excluded_note_ids: std::collections::HashSet<String>,
    anchor: &super::anchor_resolver::AnchorSession,
) -> Option<String> {
    use super::anchor_resolver::AnchorContextMode;
    let mode = anchor.mode;
    let uuid = Uuid::parse_str(session_id).ok()?;
    let node = graph.get_chat_session(uuid).await.ok().flatten()?;
    // Mode `on`: the anchor precedence decides the project, and its live block
    // opens the enrichment. Else the historical path decides (a session persisted
    // without a slug is inferred from its cwd). The resolution is cached for the
    // session while its anchors and consent are unchanged, but always checked
    // against the current anchors: an anchor put since the last turn is in this
    // turn's live block.
    let mut live_block = String::new();
    let mut resolved: Option<Option<String>> = None;
    if mode == AnchorContextMode::On {
        match anchor
            .cache
            .resolve(
                graph.as_ref(),
                uuid,
                &super::anchor_resolver::ProjectInputs::of_session(&node),
                &super::anchor_resolver::GraphCwdInference(graph.as_ref()),
            )
            .await
        {
            Ok((r, _)) => {
                live_block = r.live_block();
                let mut slug = r.decision.project.as_ref().map(|p| p.slug.clone());
                // An explicit `project_slug` that does not resolve to a project is
                // kept as it is for the enrichment, as before the resolver existed
                // (a warning, once per session). It never widens the resolver's own
                // scope: the decision above does not use it.
                if let Some(explicit) = node.project_slug.as_deref().filter(|s| !s.is_empty()) {
                    if r.decision.source != super::anchor_resolver::ProjectSource::Explicit {
                        if anchor.cache.first_time(uuid, "unresolved_explicit_slug") {
                            warn!(session_id = %session_id, slug = %explicit, "the project_slug of the session does not resolve to a project: kept as is for the enrichment");
                        }
                        slug = Some(explicit.to_string());
                    }
                }
                resolved = Some(slug);
            }
            Err(e) => {
                warn!(session_id = %session_id, error = %e, "anchor resolver failed, historical context used")
            }
        }
    }
    let project_slug = match resolved {
        Some(slug) => slug,
        None => match node.project_slug.clone() {
            Some(slug) => Some(slug),
            None => {
                super::anchor_resolver::CwdInference::infer(
                    &super::anchor_resolver::GraphCwdInference(graph.as_ref()),
                    &node.cwd,
                )
                .await
            }
        },
    };
    if mode == AnchorContextMode::Shadow {
        spawn_anchor_shadow(
            graph.clone(),
            anchor.cache.clone(),
            uuid,
            node.project_slug.clone(),
            node.execution_place,
            node.cwd.clone(),
            project_slug.clone(),
        );
    }
    // Resolve the project id once for every stage: stages that only read
    // `project_id` (reflex) were skipped for every chat message.
    let project_id = enrichment_project_id(graph.as_ref(), project_slug.as_deref()).await;
    let input = super::enrichment::EnrichmentInput {
        message: message.to_string(),
        session_id: uuid,
        project_slug,
        project_id,
        cwd: Some(node.cwd),
        protocol_run_id: protocol.run_id,
        protocol_state: protocol.state,
        excluded_note_ids,
        reasoning_path_tracker: protocol.reasoning_path_tracker,
    };
    let ctx = pipeline.execute(&input).await;
    if !ctx.has_content() {
        return (!live_block.is_empty()).then_some(live_block);
    }
    debug!(
        "[enrichment] Prompt enriched: {} sections, {}ms (hints: {:?})",
        ctx.sections.len(),
        ctx.total_time_ms,
        ctx.hints.keys().collect::<Vec<_>>()
    );
    // Clean markdown prepended to the user message (replaces the old XML-wrapped
    // <enrichment_context> format).
    let md = ctx.to_system_prompt_markdown();
    if live_block.is_empty() {
        return (!md.is_empty()).then_some(md);
    }
    // The live block opens the enrichment, like the other stages: never the prefix.
    Some(if md.is_empty() {
        live_block
    } else {
        format!("{live_block}\n\n{md}")
    })
}

/// What the manager does around a turn of the agent engine
/// ([`super::agent_runtime::TurnServices`]), with the same functions as the
/// Claude Code engine.
pub(crate) struct ManagerTurnServices {
    graph: Arc<dyn GraphStore>,
    enrichment_pipeline: Arc<super::enrichment::EnrichmentPipeline>,
    turn_routing: Arc<super::agent_hooks::TurnRouting>,
    /// Where the images attached to a message are read.
    documents: crate::documents::store::DocumentStore,
    nats: Option<Arc<crate::events::NatsEmitter>>,
    /// The anchor state of the manager when these services were built.
    anchor: super::anchor_resolver::AnchorSession,
}

#[async_trait::async_trait]
impl super::agent_runtime::TurnServices for ManagerTurnServices {
    async fn prepare(
        &self,
        session_id: &str,
        shown: &str,
        sent: &str,
        turn: &crate::refs::turn::TurnExpansion,
    ) -> String {
        // What follows the enrichment. No references: the attachments expanded
        // (references in the conversation, content for the model), a relayed
        // history kept in front. References: the relay, the visible text, its
        // `<po-context>` pointers, then the documents (`TurnExpansion::native_prompt`).
        let body = if turn.resolved.is_empty() {
            if sent == shown {
                turn.enrichment_text.clone()
            } else {
                super::message_attachments::expand_for_agent(&self.graph, sent).await
            }
        } else {
            turn.native_prompt(shown, sent)
        };
        // The enrichment reads what the user typed (the attachments' text, without
        // references), and does not inject again the notes the user pointed at.
        let prepared = match enrichment_for_turn(
            &self.graph,
            &self.enrichment_pipeline,
            session_id,
            &turn.enrichment_text,
            TurnProtocol::default(),
            turn.excluded_note_ids.clone(),
            &self.anchor,
        )
        .await
        {
            Some(md) => prepend_enrichment(&md, &body),
            None => body,
        };
        // The hook of the turn only sees the length of the text: hand it the text of
        // THIS turn, whatever started it (a message, the queue, a hint, another
        // instance), right before it is sent — what the user typed, not the
        // `<po-refs>`/`<po-attachments>` blocks around it.
        if let Some(router) = self.turn_routing.get(session_id) {
            router.set_last_message(&crate::refs::turn::visible_text(shown));
        }
        prepared
    }

    async fn continuation(&self, session_id: &str) -> String {
        let ctx = super::post_stream::PostStreamContext::build(
            &self.graph,
            Uuid::parse_str(session_id).ok(),
        )
        .await;
        // No work log on this engine: the hint carries the task/step context alone.
        super::post_stream::continuation_message(&self.graph, ctx.project_slug.as_deref(), "").await
    }

    async fn images(
        &self,
        shown: &str,
    ) -> std::result::Result<Vec<super::message_attachments::AttachedImage>, String> {
        super::message_attachments::load_images(&self.graph, &self.documents, shown).await
    }

    fn publish(&self, session_id: &str, event: &ChatEvent) {
        if let Some(nats) = &self.nats {
            nats.publish_chat_event(session_id, event.clone());
        }
    }
}

/// What another instance asks of a session of the agent engine this instance
/// runs (NATS RPC `rpc.chat.{id}.send`): the message types the Claude Code
/// listener (`spawn_nats_rpc_listener`) answers, served by the session handle.
pub(crate) async fn agent_rpc(
    handle: &Arc<super::agent_runtime::AgentSessionHandle>,
    graph: &Arc<dyn GraphStore>,
    request: &crate::events::ChatRpcRequest,
) -> crate::events::ChatRpcResponse {
    let message = &request.message;
    let field = |name: &str| {
        serde_json::from_str::<serde_json::Value>(message)
            .ok()
            .and_then(|v| v.get(name).cloned())
    };
    let outcome: Result<()> = match request.message_type.as_str() {
        "control_response" => {
            let allow = field("allow").and_then(|v| v.as_bool()).unwrap_or(false);
            match field("request_id").and_then(|v| v.as_str().map(str::to_string)) {
                Some(request_id) => handle.answer_permission(&request_id, allow).await,
                None => Err(anyhow!("a permission answer needs its request_id")),
            }
        }
        "set_auto_continue" => {
            let enabled = field("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
            handle.auto_continue.store(enabled, Ordering::Relaxed);
            if let Ok(uuid) = Uuid::parse_str(&handle.session_id) {
                let _ = graph.set_session_auto_continue(uuid, enabled).await;
            }
            Ok(())
        }
        "queue_op" => match serde_json::from_str::<super::pending_queue::QueueOp>(message) {
            Ok(op) => {
                handle.queue_op(&op).await;
                Ok(())
            }
            Err(e) => Err(anyhow!("invalid queue op: {e}")),
        },
        "queued_user_message" => handle.queue_message(message).await.map(|_| ()),
        // user_message, input_response, and what older instances send.
        _ => handle.send_message(message).await,
    };
    match outcome {
        Ok(()) => crate::events::ChatRpcResponse {
            success: true,
            error: None,
        },
        Err(e) => crate::events::ChatRpcResponse {
            success: false,
            error: Some(e.to_string()),
        },
    }
}

/// `prompt` with the turn's enrichment in front of it.
pub(crate) fn prepend_enrichment(enrichment_md: &str, prompt: &str) -> String {
    format!("{}\n\n---\n\n{}", enrichment_md, prompt)
}

/// Server secrets an agent must not inherit, among those present.
///
/// Not listed on purpose: `ANTHROPIC_API_KEY` and `CLAUDE_CODE_OAUTH_TOKEN` —
/// the CLI itself needs them to authenticate.
pub(crate) const SERVER_ONLY_SECRETS: &[&str] = &[
    "NEO4J_PASSWORD",
    "MEILISEARCH_KEY",
    "EMBEDDING_API_KEY",
    "PO_JWT_SECRET",
];

/// Variables of the server's environment an agent process may inherit, on top
/// of the SDK's base list (PATH, HOME, locale, temp dir, proxy, certificates)
/// and the `ANTHROPIC_*` / `CLAUDE_*` the Claude Code CLI authenticates with.
///
/// Developer tooling only — what `git`, `cargo`, `node`, … need to behave in
/// the agent's shell as they do in the operator's. Nothing credential-shaped:
/// a token the operator wants agents to have goes through the vault, or is
/// named explicitly in `CHAT_CHILD_ENV_INHERIT`.
pub(crate) const CHILD_ENV_TOOLING: &[&str] = &[
    "SSH_AUTH_SOCK",
    "GIT_SSH_COMMAND",
    "GIT_EXEC_PATH",
    "EDITOR",
    "VISUAL",
    "PAGER",
    "COLORTERM",
    "NO_COLOR",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
    "GOPATH",
    "GOROOT",
    "GOBIN",
    "JAVA_HOME",
    "NVM_DIR",
    "NVM_BIN",
    "PNPM_HOME",
    "VOLTA_HOME",
    "BUN_INSTALL",
    "PYENV_ROOT",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "HOMEBREW_PREFIX",
    "DOCKER_HOST",
];

/// Operator-chosen additions to the agent environment: a comma-separated list
/// of variable names (e.g. `GH_TOKEN,AWS_PROFILE`).
pub(crate) const CHILD_ENV_INHERIT_VAR: &str = "CHAT_CHILD_ENV_INHERIT";

/// The MCP tool profile a session's own provider and mode grant it, signed into
/// its token (`None`: no profile, the full one). A provider other than Claude
/// Code sees the restricted profile — no tool that opens a session or
/// reconfigures the server (A35) — unless the person opened it in `trust`
/// (bypassPermissions, decision H6): then it is treated like Claude Code. A
/// session opened BY a third-party session is restricted whatever this says
/// (`ChatManager::po_mcp_env`).
pub(crate) fn third_party_tool_profile(
    third_party: bool,
    mode: nexus_claude::agent::PolicyMode,
) -> Option<&'static str> {
    third_party.then_some(if mode == nexus_claude::agent::PolicyMode::Trust {
        crate::auth::tool_profile::FULL
    } else {
        crate::auth::tool_profile::RESTRICTED
    })
}

/// The provider of the session a session was spawned by (H6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpawnParent {
    /// No parent session.
    None,
    ClaudeCode,
    /// Another provider, or a parent that cannot be read.
    ThirdParty,
}

/// Where a session comes from ([`ChatManager::session_origin`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SessionOrigin {
    pub opened_by_third_party: bool,
    pub origin_known: bool,
}

/// Rough size, in tokens, of the project-orchestrator tool schemas a session
/// with `profile` is given (its tool list, as JSON, at four characters a token).
pub(crate) fn tool_schema_tokens(profile: crate::auth::tool_profile::ToolProfile) -> u64 {
    let tools = profile.filter_tools(crate::mcp::tools::all_tools());
    (serde_json::to_string(&tools).map_or(0, |s| s.len()) / 4) as u64
}

/// The tool profile the window check measures for `spec`: the one signed into the
/// token of its project-orchestrator server. Without a token (no signing key, or
/// no caller) the server shows every tool but none of them is authorised: the
/// check keeps measuring the restricted profile there, as before H6.
pub(crate) fn spec_tool_profile(
    spec: &nexus_claude::agent::SessionSpec,
) -> crate::auth::tool_profile::ToolProfile {
    use crate::auth::tool_profile::ToolProfile;
    match spec.mcp_servers.get("project-orchestrator") {
        Some(nexus_claude::agent::McpServerSpec::Stdio { env, .. }) => env
            .get("PO_AUTH_TOKEN")
            .map_or(ToolProfile::Restricted, |t| {
                ToolProfile::from_unverified_token(t)
            }),
        _ => ToolProfile::Restricted,
    }
}

/// Whether a session about to open has no file, shell or web tool because its
/// `nexus-tools` executable is missing: a native provider brings none of its own.
/// Claude Code, Codex and ACP bring theirs, so they are never reported.
///
/// Both must hold: the spec the session is opened with has no `nexus` server (what
/// was really attached), AND `configured` (`NEXUS_TOOLS_PATH`, next to the server, or
/// on the PATH) is not a runnable executable. A session whose policy leaves no
/// `nexus-tools` tool to offer is not attached either, but its installation lacks
/// nothing: that is the policy's choice, not reported.
pub(crate) fn lacks_nexus_tools(
    kind: nexus_claude::agent::ProviderKind,
    spec: &nexus_claude::agent::SessionSpec,
    configured: Option<&std::path::Path>,
) -> bool {
    kind == nexus_claude::agent::ProviderKind::Native
        && !spec
            .mcp_servers
            .contains_key(nexus_claude::providers::native::NEXUS_TOOLS_SERVER)
        && configured
            .and_then(super::provider::native_factory::runnable_nexus_tools)
            .is_none()
}

/// Refuses a model whose context window cannot hold the tool schemas of the
/// session's `profile` with room to work: the schemas must take at most half of
/// it. A window that is not known is not a refusal (nothing is invented).
pub(crate) fn window_holds_the_tools(
    caps: &nexus_claude::agent::Capabilities,
    profile: crate::auth::tool_profile::ToolProfile,
) -> Result<(), nexus_claude::agent::ProviderError> {
    let Some(window) = caps.context_window.as_ref().map(|w| w.value) else {
        return Ok(());
    };
    let needed = tool_schema_tokens(profile) * 2;
    if window < needed {
        return Err(nexus_claude::agent::ProviderError::ContextTooSmall {
            needed: Some(needed),
            available: Some(window),
        });
    }
    Ok(())
}

/// The environment policy of every agent process this server starts (chat
/// sessions, runner tasks, the feature-graph one-shot): clean environment,
/// allowlist only. The server's own secrets are simply not on the list.
pub(crate) fn child_env_policy() -> nexus_claude::EnvPolicy {
    let extra = std::env::var(CHILD_ENV_INHERIT_VAR).unwrap_or_default();
    child_env_policy_with(&extra)
}

/// [`child_env_policy`] with the operator list given explicitly (testable).
pub(crate) fn child_env_policy_with(operator_list: &str) -> nexus_claude::EnvPolicy {
    let operator = operator_list
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        // An operator cannot hand the server's own secrets to agents by listing them.
        .filter(|name| !SERVER_ONLY_SECRETS.contains(name))
        .map(str::to_string);
    nexus_claude::EnvPolicy::claude_code().with_inherited(
        CHILD_ENV_TOOLING
            .iter()
            .map(|name| (*name).to_string())
            .chain(operator),
    )
}

pub(crate) fn server_secrets_to_hide(present: impl Fn(&str) -> bool) -> Vec<&'static str> {
    SERVER_ONLY_SECRETS
        .iter()
        .copied()
        .filter(|k| present(k))
        .collect()
}

impl ChatManager {
    /// Build the standard enrichment pipeline with optional reasoning engine and trajectory collector.
    ///
    /// Centralizes the 7-stage pipeline construction to avoid duplication across
    /// `new()`, `with_reasoning_engine()`, and `with_trajectory_collector()`.
    fn build_enrichment_pipeline(
        graph: &Arc<dyn GraphStore>,
        search: &Arc<dyn SearchStore>,
        reasoning_engine: Option<&Arc<crate::reasoning::ReasoningTreeEngine>>,
        trajectory_collector: Option<&std::sync::Arc<neural_routing_runtime::TrajectoryCollector>>,
        mcp_registry: Option<&crate::mcp_federation::registry::SharedRegistry>,
    ) -> Arc<super::enrichment::EnrichmentPipeline> {
        let mut pipeline = super::enrichment::EnrichmentPipeline::new(
            super::enrichment::EnrichmentConfig::from_env(),
        );
        let skill_stage = super::stages::SkillActivationStage::new(graph.clone());
        let skill_stage = if let Some(tc) = trajectory_collector {
            skill_stage.with_collector(tc.clone())
        } else {
            skill_stage
        };
        pipeline.add_parallel_stage(Box::new(skill_stage));
        pipeline.add_parallel_stage(Box::new(super::stages::BiomimicryStage::new(graph.clone())));
        pipeline.add_parallel_stage(Box::new(super::stages::UserProfileStage::new(
            graph.clone(),
        )));
        pipeline.add_parallel_stage(Box::new(super::stages::PersonaStage::new(graph.clone())));
        let ki_stage = super::stages::KnowledgeInjectionStage::new(graph.clone(), search.clone());
        let ki_stage = if let Some(tc) = trajectory_collector {
            ki_stage.with_collector(tc.clone())
        } else {
            ki_stage
        };
        pipeline.add_parallel_stage(Box::new(ki_stage));
        pipeline.add_parallel_stage(Box::new(super::stages::StatusInjectionStage::with_config(
            graph.clone(),
            reasoning_engine.cloned(),
            Arc::new(super::stages::GraphProtocolProvider::new(graph.clone())),
            super::stages::StatusInjectionConfig::default(),
        )));
        pipeline.add_parallel_stage(Box::new(super::stages::FileContextStage::new(
            graph.clone(),
        )));
        // Reflex stage: injects scar warnings, episode recall, co-change reminders
        // from the autonomous learning loop (T1→T2→T3 materialized knowledge).
        // Controlled by ENRICHMENT_REFLEX env var (default: true).
        pipeline.add_parallel_stage(Box::new(crate::reflex::stage::ReflexStage::new(
            graph.clone(),
        )));
        // MCP Federation stage: injects external tool availability into the prompt.
        // Only adds content when MCP servers are connected (0 overhead otherwise).
        if let Some(registry) = mcp_registry {
            pipeline.add_parallel_stage(Box::new(super::stages::McpFederationStage::new(
                registry.clone(),
            )));
        }
        Arc::new(pipeline)
    }

    /// Create a ChatManager without memory support (for tests or when Meilisearch is unavailable)
    pub fn new_without_memory(
        graph: Arc<dyn GraphStore>,
        search: Arc<dyn SearchStore>,
        config: ChatConfig,
    ) -> Self {
        let permission_config = Arc::new(RwLock::new(config.permission.clone()));
        let env_config = Arc::new(RwLock::new(RuntimeEnvConfig {
            process_path: config.process_path.clone(),
            claude_cli_path: config.claude_cli_path.clone(),
            auto_update_cli: config.auto_update_cli,
            auto_update_app: config.auto_update_app,
        }));
        let enrichment_pipeline =
            Self::build_enrichment_pipeline(&graph, &search, None, None, None);
        let agent_runtime = Arc::new(super::agent_runtime::AgentRuntime::new(graph.clone()));
        let provider_source: Arc<dyn super::agent_runtime::ProviderSource> = Arc::new(
            super::agent_runtime::BuiltinProviders::new(config.claude_cli_path.clone()),
        );
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            agent_runtime,
            provider_source,
            cognitive_routing: None,
            open_decisions: std::sync::Mutex::new(std::collections::HashMap::new()),
            native_cache: Arc::new(RwLock::new(HashMap::new())),
            native_probers: Arc::new(RwLock::new(HashMap::new())),
            context_injector: None,
            memory_config: None,
            event_emitter: None,
            vault: None,
            nats: None,
            permission_config,
            config_yaml_path: None,
            env_config,
            enrichment_pipeline,
            trajectory_collector: Arc::new(std::sync::RwLock::new(None)),
            reasoning_engine: None,
            neural_routing_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dual_track_router: Arc::new(std::sync::RwLock::new(None)),
            nn_router: None,
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            turn_routing: Arc::new(super::agent_hooks::TurnRouting::default()),
            refs_v1: crate::refs::flag::from_env(),
            document_store: crate::documents::store::DocumentStore::new(
                crate::documents::store::default_storage_dir(),
            ),
            native_transcripts: None,
            anchor_mode: super::anchor_resolver::AnchorContextMode::from_env(),
            anchor_cache: Arc::default(),
        }
    }

    /// Create a new ChatManager with conversation memory support
    pub async fn new(
        graph: Arc<dyn GraphStore>,
        search: Arc<dyn SearchStore>,
        config: ChatConfig,
    ) -> Self {
        // Initialize ContextInjector for conversation memory persistence
        let memory_config = MemoryConfig {
            meilisearch_url: config.meilisearch_url.clone(),
            meilisearch_key: Some(config.meilisearch_key.clone()),
            enabled: true,
            ..MemoryConfig::default()
        };
        let context_injector = match ContextInjector::new(memory_config.clone()).await {
            Ok(injector) => {
                info!("ContextInjector initialized for conversation memory");
                Some(Arc::new(injector))
            }
            Err(e) => {
                warn!("Failed to initialize ContextInjector: {} — message history will be unavailable", e);
                None
            }
        };

        let permission_config = Arc::new(RwLock::new(config.permission.clone()));
        let env_config = Arc::new(RwLock::new(RuntimeEnvConfig {
            process_path: config.process_path.clone(),
            claude_cli_path: config.claude_cli_path.clone(),
            auto_update_cli: config.auto_update_cli,
            auto_update_app: config.auto_update_app,
        }));
        // Neutral chat directories a crash left behind (sessions that were never closed).
        tokio::task::spawn_blocking(|| {
            let n = super::neutral_place::sweep_orphans(super::neutral_place::ORPHAN_MAX_AGE);
            if n > 0 {
                info!(removed = n, "swept orphan neutral chat directories");
            }
        });
        let enrichment_pipeline =
            Self::build_enrichment_pipeline(&graph, &search, None, None, None);
        let agent_runtime = Arc::new(super::agent_runtime::AgentRuntime::new(graph.clone()));
        let provider_source: Arc<dyn super::agent_runtime::ProviderSource> = Arc::new(
            super::agent_runtime::BuiltinProviders::new(config.claude_cli_path.clone()),
        );
        Self {
            graph,
            search,
            config,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            agent_runtime,
            provider_source,
            cognitive_routing: None,
            open_decisions: std::sync::Mutex::new(std::collections::HashMap::new()),
            native_cache: Arc::new(RwLock::new(HashMap::new())),
            native_probers: Arc::new(RwLock::new(HashMap::new())),
            context_injector,
            memory_config: Some(memory_config),
            event_emitter: None,
            vault: None,
            nats: None,
            permission_config,
            config_yaml_path: None,
            env_config,
            enrichment_pipeline,
            trajectory_collector: Arc::new(std::sync::RwLock::new(None)),
            reasoning_engine: None,
            neural_routing_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            dual_track_router: Arc::new(std::sync::RwLock::new(None)),
            nn_router: None,
            mcp_registry: crate::mcp_federation::registry::new_shared_registry(),
            turn_routing: Arc::new(super::agent_hooks::TurnRouting::default()),
            refs_v1: crate::refs::flag::from_env(),
            document_store: crate::documents::store::DocumentStore::new(
                crate::documents::store::default_storage_dir(),
            ),
            native_transcripts: None,
            anchor_mode: super::anchor_resolver::AnchorContextMode::from_env(),
            anchor_cache: Arc::default(),
        }
    }

    /// The anchor state a session opened or resumed NOW gets: the mode is fixed here,
    /// once, and the session keeps it for its whole life (the environment is read
    /// only when the manager is built).
    pub(crate) fn anchor_session(&self) -> super::anchor_resolver::AnchorSession {
        super::anchor_resolver::AnchorSession::new(self.anchor_mode, self.anchor_cache.clone())
    }

    /// Set the anchor-context mode (the default comes from `PO_ANCHOR_CONTEXT`).
    pub fn with_anchor_context_mode(
        mut self,
        mode: super::anchor_resolver::AnchorContextMode,
    ) -> Self {
        self.anchor_mode = mode;
        self
    }

    /// Where the attached documents live (`DocumentStore::from_config`).
    pub fn with_document_store(mut self, store: crate::documents::store::DocumentStore) -> Self {
        self.document_store = store;
        self
    }

    /// Keep the native sessions' transcripts under `root` so they resume after a
    /// restart (`provider::transcripts`). Applies to the providers built afterwards.
    pub fn with_native_transcripts(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.native_transcripts = Some(root.into());
        self
    }

    /// Turn `refs_v1` on or off (the default comes from the environment).
    pub fn with_refs_v1(mut self, on: bool) -> Self {
        self.refs_v1 = on;
        self
    }

    /// Whether `refs_v1` is on: the API layer folds `refs` into messages and
    /// `auth_ok` announces the capability.
    pub fn refs_v1_enabled(&self) -> bool {
        self.refs_v1
    }

    /// Lets the cognitive router choose the model of each turn of a session in routing
    /// mode `full` + stage `auto` (a session keeps its provider; `pool` lists the models
    /// of that provider only).
    pub fn with_turn_decider(
        self,
        decider: Arc<dyn super::provider::cognitive::decision::Decider>,
        pool: Arc<dyn super::agent_hooks::PoolSource>,
    ) -> Self {
        self.turn_routing.configure(decider, pool);
        self
    }

    /// Settings scope and key holding the pin of a session's model.
    fn pin_scope(session_id: &str) -> String {
        format!("session:{session_id}")
    }

    const MODEL_PIN_KEY: &'static str = "model_pinned";

    /// Records that the model of the session is not the router's to change.
    async fn pin_session_model(&self, session_id: &str) {
        if let Err(error) = self
            .graph
            .put_llm_setting(&Self::pin_scope(session_id), Self::MODEL_PIN_KEY, "true")
            .await
        {
            warn!(session_id, %error, "model pin not stored: a resume may route it again");
        }
    }

    /// Whether the session's model was pinned (named by the request or set by hand).
    async fn model_is_pinned(&self, session_id: &str) -> bool {
        matches!(
            self.graph
                .get_llm_setting(&Self::pin_scope(session_id), Self::MODEL_PIN_KEY)
                .await,
            Ok(Some(value)) if value == "true"
        )
    }

    /// The models of `provider_id` a pool lets the per-turn decision choose among: `None`
    /// without a pool (or with one model only: strict, a pin), `Some(empty)` when no ticked
    /// model is on this provider (the conversation then stays on its model: moving to another
    /// provider is `switch-provider`, never a per-turn change).
    fn allowed_models_of(
        pool: Option<&[super::types::RoutingPoolEntry]>,
        provider_id: &str,
    ) -> Option<Vec<String>> {
        let pool = pool.filter(|p| p.len() > 1)?;
        Some(
            pool.iter()
                .filter(|e| e.provider == provider_id)
                .map(|e| e.model.clone())
                .collect(),
        )
    }

    /// Builds and registers the per-turn router of a session being opened, `None` without
    /// a decider. The routing mode and stage are read ONCE here for the session's project.
    pub(crate) async fn register_turn_router(
        &self,
        session_id: &str,
        provider_id: &str,
        model: &str,
        project_slug: Option<&str>,
        turn: OpeningTurn<'_>,
    ) -> Option<Arc<super::agent_hooks::TurnRouter>> {
        let (decider, pool) = self.turn_routing.configured()?;
        // A model the request named, or the user set by hand, stays pinned across a
        // resume or a restart: the pin is stored, not only held by the router.
        // A conversation given a pool (two models ticked or more) names its pilot, but PO still
        // routes among the pool: the pilot is not a pin. One model ticked is strict: a pin.
        let allowed_models = Self::allowed_models_of(turn.routing_pool.as_deref(), provider_id);
        let named = turn.explicit_model && !turn.moved_in && allowed_models.is_none();
        let pinned = named || self.model_is_pinned(session_id).await;
        if named {
            self.pin_session_model(session_id).await;
        }
        let routing = match super::provider::cognitive::load_routing(
            self.graph.as_ref(),
            project_slug,
        )
        .await
        {
            Ok((mut settings, _)) => {
                if let Some(mode) = turn.routing_mode {
                    settings.mode = mode;
                }
                settings
            }
            Err(error) => {
                warn!(session_id, error = %error, "routing settings unreadable: no per-turn routing");
                return None;
            }
        };
        let trust = turn.permission_mode.is_some_and(|mode| {
            super::provider::policy::parse_mode(mode)
                .is_some_and(|pair| pair.neutral == nexus_claude::agent::PolicyMode::Trust)
        });
        let router = Arc::new(super::agent_hooks::TurnRouter::new(
            super::agent_hooks::TurnRouterSpec {
                decider,
                pool,
                routing,
                provider_id: provider_id.to_owned(),
                session_id: Uuid::parse_str(session_id).ok(),
                project_slug: project_slug.map(str::to_owned),
                trust,
                explicit_model: pinned,
                provider_imposed: turn.provider_imposed && !turn.moved_in,
                allowed_models,
                routing_pool: turn.routing_pool.clone(),
                current_model: model.to_owned(),
                next_turn: turn.next_turn,
                moved_in: turn.moved_in,
            },
        ));
        router.set_last_message(turn.message);
        self.turn_routing.insert(session_id, Arc::clone(&router));
        Some(router)
    }

    /// Legacy engine: asks the same per-turn decision as the agent engine and, when it
    /// names another model, sends the `set_model` control frame before the message.
    pub(crate) async fn apply_turn_directive(&self, session_id: &str, message: &str) {
        let Some(router) = self.turn_routing.get(session_id) else {
            return;
        };
        // The routing reads what the user typed, not the blocks around it.
        let typed = crate::refs::turn::visible_text(message);
        let message = typed.as_str();
        router.set_last_message(message);
        let ctx = router.next_turn_context(message.chars().count());
        let before = ctx.current_model.clone();
        let directive = super::agent_hooks::directive_for_turn(&router, &ctx).await;
        if let Some(model) = directive.model {
            if let Err(error) = self
                .set_session_model_inner(session_id, &model, false)
                .await
            {
                warn!(session_id, error = %error, "turn routing could not change the model");
                router.forget_change(&before);
            }
        }
    }

    /// Mode `full`, before a turn starts, on both engines: when the router's decision names
    /// ANOTHER provider (`agent_hooks::plan_provider_move`), the conversation moves there
    /// through the relay (`moved_by: auto`) and `message` is the first message of the new
    /// session. Returns `true` when it moved: this session then starts no turn (it is
    /// closed by the relay). A message for a running turn is queued on its session, never
    /// moved. A refused move (the project's consent, the endpoint guard, the security gate,
    /// all applied by the relay's opening) leaves the conversation where it is; the stored
    /// decision then says why (`not_moved: <code>`).
    pub(crate) async fn move_provider_before_turn(&self, session_id: &str, message: &str) -> bool {
        let Some(router) = self.turn_routing.get(session_id) else {
            return false;
        };
        let streaming = match self.agent_runtime.get(session_id).await {
            Some(handle) => handle.is_streaming.load(Ordering::SeqCst),
            None => match self.active_sessions.read().await.get(session_id) {
                Some(session) => session.is_streaming.load(Ordering::SeqCst),
                None => return false,
            },
        };
        if streaming {
            return false;
        }
        // The routing reads what the user typed, not the blocks around it.
        router.set_last_message(&crate::refs::turn::visible_text(message));
        let super::agent_hooks::ProviderPlan::Move { pick, decision } =
            super::agent_hooks::plan_provider_move(&router).await
        else {
            return false;
        };
        match self
            .relay_conversation(
                session_id,
                &pick.provider_id,
                Some(&pick.model),
                message,
                None,
                super::relay::MOVED_BY_AUTO,
            )
            .await
        {
            Ok(moved) => {
                info!(
                    from_session = session_id,
                    to_session = %moved.session_id,
                    provider = %pick.provider_id,
                    model = %pick.model,
                    reason = %decision.reason,
                    "the router moved the conversation to another provider"
                );
                true
            }
            Err(error) => {
                let code =
                    super::provider::errors::classify_open_error(&error, Some(&pick.provider_id))
                        .map(|failure| failure.code)
                        .or_else(|| {
                            error
                                .downcast_ref::<super::types::SwitchProviderError>()
                                .map(|_| "switch_refused")
                        })
                        .unwrap_or("open_failed");
                warn!(
                    session_id,
                    provider = %pick.provider_id,
                    code,
                    error = %error,
                    "the router's provider move was refused: the conversation stays"
                );
                self.record_refused_move(*decision, code).await;
                false
            }
        }
    }

    /// The decision of a move that did not happen says so: not applied, and why.
    async fn record_refused_move(
        &self,
        mut decision: super::provider::cognitive::decision::CognitiveDecision,
        code: &str,
    ) {
        let Some(store) = self
            .cognitive_routing
            .as_ref()
            .and_then(|routing| routing.store.clone())
        else {
            return;
        };
        decision.applied = false;
        decision.reason = format!("{}; not_moved: {code}", decision.reason);
        if let Err(error) = store.put_decision(&decision).await {
            warn!(decision_id = %decision.id, %error, "the refused move was not recorded on its decision");
        }
    }

    /// Emit the light `attention_changed` signal for a session (no-op without
    /// an emitter). Ids only: never the text of a command or question.
    pub(crate) fn notify_attention(&self, session_id: &str, reason: AttentionReason) {
        notify_attention(
            &self.event_emitter,
            AttentionSubject::Session(session_id.to_string()),
            reason,
        );
    }

    /// Set the event emitter for CRUD events (streaming status notifications)
    /// Replaces where the agent path finds provider instances (the nexus
    /// registry, or a fake in tests).
    pub fn with_provider_source(
        mut self,
        source: Arc<dyn super::agent_runtime::ProviderSource>,
    ) -> Self {
        self.provider_source = source;
        self
    }

    /// Wires the cognitive router. Without it the provider is resolved from the
    /// declared rules only.
    pub fn with_cognitive_routing(
        mut self,
        routing: super::provider::cognitive::decider::CognitiveRouting,
    ) -> Self {
        self.cognitive_routing = Some(routing);
        self
    }

    /// The cognitive router, when wired.
    pub fn cognitive_routing(
        &self,
    ) -> Option<&super::provider::cognitive::decider::CognitiveRouting> {
        self.cognitive_routing.as_ref()
    }

    pub fn with_event_emitter(mut self, emitter: Arc<dyn crate::events::EventEmitter>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Set the config.yaml path for persisting permission config changes.
    pub fn with_config_yaml_path(mut self, path: std::path::PathBuf) -> Self {
        self.config_yaml_path = Some(path);
        self
    }

    /// Attach the secrets vault (per-session vault token + output masking).
    pub fn with_vault(mut self, vault: Arc<crate::vault::VaultService>) -> Self {
        self.vault = Some(vault);
        self
    }

    /// Set the NATS emitter for cross-instance chat event publishing.
    ///
    /// When configured, ChatEvents are published to NATS alongside the local
    /// broadcast channel, enabling multi-instance real-time sync.
    /// Interrupts are also propagated via NATS to all instances.
    pub fn with_nats(mut self, nats: Arc<crate::events::NatsEmitter>) -> Self {
        self.nats = Some(nats);
        self
    }

    /// Attach a reasoning tree engine for the StatusInjectionStage.
    ///
    /// Rebuilds the enrichment pipeline so that `StatusInjectionStage` receives
    /// the engine and can build reasoning trees from the knowledge graph.
    pub fn with_reasoning_engine(
        mut self,
        engine: Arc<crate::reasoning::ReasoningTreeEngine>,
    ) -> Self {
        self.reasoning_engine = Some(engine.clone());
        let tc_guard = self.trajectory_collector.read().unwrap();
        self.enrichment_pipeline = Self::build_enrichment_pipeline(
            &self.graph,
            &self.search,
            Some(&engine),
            tc_guard.as_ref(),
            Some(&self.mcp_registry),
        );
        drop(tc_guard);
        self
    }

    /// Attach a trajectory collector to the enrichment pipeline stages.
    ///
    /// Rebuilds the pipeline so that `SkillActivationStage` and
    /// `KnowledgeInjectionStage` fire decision records to the collector.
    pub fn with_trajectory_collector(
        self,
        collector: std::sync::Arc<neural_routing_runtime::TrajectoryCollector>,
    ) -> Self {
        self.set_trajectory_collector(collector);
        self
    }

    /// Hot-swap the trajectory collector at runtime.
    ///
    /// Called when collection is enabled via the API after the ChatManager
    /// is already constructed. Rebuilds the enrichment pipeline and stores
    /// the collector for `close_session()` finalization.
    pub fn set_trajectory_collector(
        &self,
        collector: std::sync::Arc<neural_routing_runtime::TrajectoryCollector>,
    ) {
        *self.trajectory_collector.write().unwrap() = Some(collector);
    }

    /// Attach a dual-track router and optionally enable neural routing.
    ///
    /// The dual-track router runs both heuristic and neural routing in parallel,
    /// using the neural result when `enabled` is true and the model is confident.
    pub fn with_dual_track_router(
        self,
        router: super::routing::DualTrackRouter,
        enabled: bool,
    ) -> Self {
        *self.dual_track_router.write().unwrap() = Some(router);
        self.neural_routing_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        self
    }

    /// Attach the runtime NN router (DualTrackRouter from neural-routing-runtime).
    ///
    /// When set, `build_system_prompt()` queries this router after compose()
    /// to log trajectory-based action suggestions and populate dashboard metrics.
    pub fn with_nn_router(
        mut self,
        router: Arc<tokio::sync::RwLock<neural_routing_runtime::DualTrackRouter>>,
        enabled: bool,
    ) -> Self {
        self.nn_router = Some(router);
        self.neural_routing_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        self
    }

    /// Hot-swap the dual-track router at runtime.
    pub fn set_dual_track_router(&self, router: super::routing::DualTrackRouter) {
        *self.dual_track_router.write().unwrap() = Some(router);
    }

    /// Enable or disable neural routing at runtime.
    pub fn set_neural_routing_enabled(&self, enabled: bool) {
        self.neural_routing_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set a custom reward computer for session reward computation.
    /// Replace the enrichment pipeline with a custom-configured one.
    ///
    /// Use this to configure which enrichment stages are enabled/disabled,
    /// set debug mode, or override the time budget. Stages must be added
    /// to the pipeline before passing it here.
    ///
    /// ```ignore
    /// let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig {
    ///     skill_activation: true,
    ///     knowledge_injection: true,
    ///     debug: true,
    ///     ..Default::default()
    /// });
    /// // pipeline.add_stage(Box::new(SkillActivationStage::new(graph.clone())));
    /// let manager = manager.with_enrichment_pipeline(Arc::new(pipeline));
    /// ```
    pub fn with_enrichment_pipeline(
        mut self,
        pipeline: Arc<super::enrichment::EnrichmentPipeline>,
    ) -> Self {
        self.enrichment_pipeline = pipeline;
        self
    }

    // ========================================================================
    // Runtime permission config (GET / UPDATE via REST API)
    // ========================================================================

    /// Get a clone of the current runtime permission config.
    pub async fn get_permission_config(&self) -> super::config::PermissionConfig {
        self.permission_config.read().await.clone()
    }

    /// Update the runtime permission config.
    ///
    /// Validates the mode string before applying. Returns an error if the mode
    /// is not one of the valid values ("default", "acceptEdits", "plan", "bypassPermissions").
    /// New sessions will pick up the updated config immediately.
    /// Active sessions keep their original config (no mid-session changes).
    ///
    /// When a `config_yaml_path` is set, the updated config is also persisted
    /// to disk atomically (write to .tmp then rename).
    pub async fn update_permission_config(
        &self,
        new_config: super::config::PermissionConfig,
    ) -> Result<super::config::PermissionConfig> {
        if !super::config::PermissionConfig::is_valid_mode(&new_config.mode) {
            return Err(anyhow!(
                "Invalid permission mode '{}'. Valid modes: {:?}",
                new_config.mode,
                super::config::PermissionConfig::valid_modes()
            ));
        }
        let mut perm = self.permission_config.write().await;
        *perm = new_config;
        let result = perm.clone();
        // Drop the lock before doing I/O
        drop(perm);

        // Persist to config.yaml if a path is configured
        if let Some(ref yaml_path) = self.config_yaml_path {
            if let Err(e) = Self::persist_permission_to_yaml(yaml_path, &result) {
                // Log but don't fail the API call — in-memory update succeeded
                error!(
                    path = %yaml_path.display(),
                    error = %e,
                    "Failed to persist permission config to config.yaml"
                );
            } else {
                info!(
                    path = %yaml_path.display(),
                    mode = %result.mode,
                    "Permission config persisted to config.yaml"
                );
            }
        }

        Ok(result)
    }

    /// Persist permission config to config.yaml using surgical YAML modification.
    ///
    /// Reads the existing file as a `serde_yaml::Value` tree, updates only the
    /// `chat.permissions` subtree, and writes back atomically (tmp + rename).
    /// This preserves all other config sections (auth, server, neo4j, etc.)
    /// without needing `Serialize` on those structs.
    fn persist_permission_to_yaml(
        yaml_path: &std::path::Path,
        permission: &super::config::PermissionConfig,
    ) -> Result<()> {
        use std::io::Write;

        // 1. Read existing YAML as a Value tree (or start from empty mapping)
        let mut doc: serde_yaml::Value = if yaml_path.exists() {
            let contents = std::fs::read_to_string(yaml_path)
                .with_context(|| format!("Reading {}", yaml_path.display()))?;
            serde_yaml::from_str(&contents)
                .with_context(|| format!("Parsing {}", yaml_path.display()))?
        } else {
            serde_yaml::Value::Mapping(serde_yaml::Mapping::new())
        };

        // 2. Ensure doc is a mapping
        let root = doc
            .as_mapping_mut()
            .ok_or_else(|| anyhow!("config.yaml root is not a YAML mapping"))?;

        // 3. Ensure chat section exists as a mapping
        let chat_key = serde_yaml::Value::String("chat".into());
        if !root.contains_key(&chat_key) {
            root.insert(
                chat_key.clone(),
                serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
            );
        }
        let chat_section = root
            .get_mut(&chat_key)
            .and_then(|v| v.as_mapping_mut())
            .ok_or_else(|| anyhow!("chat section is not a YAML mapping"))?;

        // 4. Serialize PermissionConfig to a YAML Value and insert
        let perm_value = serde_yaml::to_value(permission)
            .context("Serializing PermissionConfig to YAML value")?;
        chat_section.insert(serde_yaml::Value::String("permissions".into()), perm_value);

        // 5. Serialize the full document back to YAML string
        let yaml_str =
            serde_yaml::to_string(&doc).context("Serializing config document to YAML")?;

        // 6. Atomic write: write to .tmp then rename
        let tmp_path = yaml_path.with_extension("yaml.tmp");
        {
            let mut file = std::fs::File::create(&tmp_path)
                .with_context(|| format!("Creating {}", tmp_path.display()))?;
            file.write_all(yaml_str.as_bytes())
                .with_context(|| format!("Writing {}", tmp_path.display()))?;
            file.sync_all()
                .with_context(|| format!("Syncing {}", tmp_path.display()))?;
        }
        std::fs::rename(&tmp_path, yaml_path).with_context(|| {
            format!("Renaming {} → {}", tmp_path.display(), yaml_path.display())
        })?;

        Ok(())
    }

    // ========================================================================
    // Runtime environment config (PATH, CLI path, auto-update via REST API)
    // ========================================================================

    /// Get a clone of the current runtime environment config.
    pub(crate) async fn get_env_config(&self) -> RuntimeEnvConfig {
        self.env_config.read().await.clone()
    }

    /// Update the process PATH used by Claude CLI subprocesses.
    pub async fn update_process_path(&self, path: Option<String>) {
        self.env_config.write().await.process_path = path;
    }

    /// Update the explicit Claude CLI binary path.
    pub async fn update_claude_cli_path(&self, path: Option<String>) {
        self.env_config.write().await.claude_cli_path = path;
    }

    /// Update the auto-update CLI toggle.
    pub async fn update_auto_update_cli(&self, enabled: bool) {
        self.env_config.write().await.auto_update_cli = enabled;
    }

    /// Update the auto-update app toggle.
    pub async fn update_auto_update_app(&self, enabled: bool) {
        self.env_config.write().await.auto_update_app = enabled;
    }

    /// Persist environment config (process_path, claude_cli_path, auto_update_cli, auto_update_app) to config.yaml.
    ///
    /// Uses the same surgical YAML modification pattern as `persist_permission_to_yaml`:
    /// read existing YAML, update only the relevant fields, write back atomically.
    pub async fn persist_chat_config_to_yaml(&self) -> Result<()> {
        let yaml_path = self
            .config_yaml_path
            .as_ref()
            .ok_or_else(|| anyhow!("No config.yaml path configured — cannot persist"))?;

        let env = self.env_config.read().await.clone();
        Self::persist_env_config_to_yaml(yaml_path, &env)
    }

    /// Persist environment config fields to config.yaml (surgical YAML modification).
    fn persist_env_config_to_yaml(
        yaml_path: &std::path::Path,
        env: &RuntimeEnvConfig,
    ) -> Result<()> {
        use std::io::Write;

        // 1. Read existing YAML as a Value tree (or start from empty mapping)
        let mut doc: serde_yaml::Value = if yaml_path.exists() {
            let contents = std::fs::read_to_string(yaml_path)
                .with_context(|| format!("Reading {}", yaml_path.display()))?;
            serde_yaml::from_str(&contents)
                .with_context(|| format!("Parsing {}", yaml_path.display()))?
        } else {
            serde_yaml::Value::Mapping(serde_yaml::Mapping::new())
        };

        // 2. Ensure doc is a mapping
        let root = doc
            .as_mapping_mut()
            .ok_or_else(|| anyhow!("config.yaml root is not a YAML mapping"))?;

        // 3. Ensure chat section exists as a mapping
        let chat_key = serde_yaml::Value::String("chat".into());
        if !root.contains_key(&chat_key) {
            root.insert(
                chat_key.clone(),
                serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
            );
        }
        let chat_section = root
            .get_mut(&chat_key)
            .and_then(|v| v.as_mapping_mut())
            .ok_or_else(|| anyhow!("chat section is not a YAML mapping"))?;

        // 4. Set/remove individual fields surgically
        let path_key = serde_yaml::Value::String("process_path".into());
        let cli_key = serde_yaml::Value::String("claude_cli_path".into());
        let auto_key = serde_yaml::Value::String("auto_update_cli".into());

        match &env.process_path {
            Some(val) => {
                chat_section.insert(path_key, serde_yaml::Value::String(val.clone()));
            }
            None => {
                chat_section.remove(&path_key);
            }
        }
        match &env.claude_cli_path {
            Some(val) => {
                chat_section.insert(cli_key, serde_yaml::Value::String(val.clone()));
            }
            None => {
                chat_section.remove(&cli_key);
            }
        }
        // Only write auto_update_cli when true (omit when false to keep YAML clean)
        if env.auto_update_cli {
            chat_section.insert(auto_key, serde_yaml::Value::Bool(true));
        } else {
            chat_section.remove(&auto_key);
        }
        // Only write auto_update_app when false (default is true, so omit when true to keep YAML clean)
        let auto_app_key = serde_yaml::Value::String("auto_update_app".into());
        if !env.auto_update_app {
            chat_section.insert(auto_app_key, serde_yaml::Value::Bool(false));
        } else {
            chat_section.remove(&auto_app_key);
        }

        // 5. Serialize the full document back to YAML string
        let yaml_str =
            serde_yaml::to_string(&doc).context("Serializing config document to YAML")?;

        // 6. Atomic write: write to .tmp then rename
        let tmp_path = yaml_path.with_extension("yaml.tmp");
        {
            let mut file = std::fs::File::create(&tmp_path)
                .with_context(|| format!("Creating {}", tmp_path.display()))?;
            file.write_all(yaml_str.as_bytes())
                .with_context(|| format!("Writing {}", tmp_path.display()))?;
            file.sync_all()
                .with_context(|| format!("Syncing {}", tmp_path.display()))?;
        }
        std::fs::rename(&tmp_path, yaml_path).with_context(|| {
            format!("Renaming {} → {}", tmp_path.display(), yaml_path.display())
        })?;

        Ok(())
    }

    /// Spawn a background task that listens for NATS interrupt signals for a session.
    ///
    /// When another instance publishes an interrupt for this session via NATS,
    /// the listener sets the local `interrupt_flag` so the stream loop breaks.
    /// No-op if NATS is not configured.
    fn spawn_nats_interrupt_listener(
        &self,
        session_id: &str,
        interrupt_flag: Arc<AtomicBool>,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        cancel: CancellationToken,
    ) {
        let Some(ref nats) = self.nats else {
            return;
        };

        let nats = nats.clone();
        let session_id = session_id.to_string();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_interrupt(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(
                        "Failed to subscribe to NATS interrupt for session {}: {}",
                        session_id, e
                    );
                    return;
                }
            };

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS interrupt listener cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        let Some(_msg) = msg else { break; };

                        // Check if session is still active — stop listener if session was removed
                        let session_exists = {
                            let sessions = active_sessions.read().await;
                            sessions.contains_key(&session_id)
                        };
                        if !session_exists {
                            debug!(
                                "Session {} no longer active, stopping NATS interrupt listener",
                                session_id
                            );
                            break;
                        }

                        // Guard: don't re-interrupt if the flag is already set
                        if interrupt_flag.load(Ordering::SeqCst) {
                            debug!(
                                "Interrupt flag already set for session {}, ignoring NATS interrupt",
                                session_id
                            );
                            continue;
                        }

                        info!(
                            "NATS interrupt received for session {}, setting interrupt flag and cancelling token",
                            session_id
                        );
                        interrupt_flag.store(true, Ordering::SeqCst);
                        // Cancel the current interrupt_token to unblock stream_response's select!
                        {
                            let sessions = active_sessions.read().await;
                            if let Some(session) = sessions.get(&session_id) {
                                session.interrupt_token.cancel();
                            }
                        }
                    }
                }
            }

            debug!("NATS interrupt listener stopped for session {}", session_id);
        });
    }

    /// Spawn a background task that responds to NATS snapshot requests for a session.
    ///
    /// When a remote instance needs to do a mid-stream join, it sends a NATS request
    /// on `events.chat.{session_id}.snapshot`. This listener replies with the current
    /// streaming snapshot (partial text + structured events).
    /// Stops when the session is removed from active_sessions.
    fn spawn_nats_snapshot_responder(
        &self,
        session_id: &str,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        cancel: CancellationToken,
    ) {
        let Some(ref nats) = self.nats else {
            return;
        };

        let nats = nats.clone();
        let session_id = session_id.to_string();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_snapshot_requests(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(
                        "Failed to subscribe to NATS snapshot requests for session {}: {}",
                        session_id, e
                    );
                    return;
                }
            };

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS snapshot responder cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        let Some(msg) = msg else { break; };

                        // Build snapshot from active session state
                        let snapshot = {
                            let sessions = active_sessions.read().await;
                            match sessions.get(&session_id) {
                                Some(session) => {
                                    let is_streaming = session.is_streaming.load(Ordering::SeqCst);
                                    let text = session.streaming_text.lock().await.clone();
                                    let events = session.streaming_events.lock().await.clone();
                                    Some(crate::events::StreamingSnapshot {
                                        is_streaming,
                                        partial_text: text,
                                        events,
                                    })
                                }
                                None => {
                                    // Session no longer active — stop responder
                                    debug!(
                                        "Session {} no longer active, stopping snapshot responder",
                                        session_id
                                    );
                                    break;
                                }
                            }
                        };

                        if let Some(snapshot) = snapshot {
                            // Reply to the requester
                            if let Some(reply_to) = msg.reply {
                                match serde_json::to_vec(&snapshot) {
                                    Ok(payload) => {
                                        if let Err(e) =
                                            nats.client().publish(reply_to, payload.into()).await
                                        {
                                            warn!(
                                                "Failed to reply with snapshot for session {}: {}",
                                                session_id, e
                                            );
                                        } else {
                                            debug!(
                                                session_id = %session_id,
                                                is_streaming = snapshot.is_streaming,
                                                text_len = snapshot.partial_text.len(),
                                                events_count = snapshot.events.len(),
                                                "Replied with streaming snapshot"
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to serialize snapshot for session {}: {}",
                                            session_id, e
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            debug!("NATS snapshot responder stopped for session {}", session_id);
        });
    }

    /// Spawn a background task that listens for NATS RPC send requests for a session.
    ///
    /// When another instance wants to send a message to a session owned by this instance,
    /// it publishes a `ChatRpcRequest` to `rpc.chat.{session_id}.send`.
    /// This listener processes the request locally (queue if streaming, or persist+stream)
    /// and replies with `ChatRpcResponse`.
    ///
    /// No-op if NATS is not configured.
    fn spawn_nats_rpc_listener(
        &self,
        session_id: &str,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        cancel: CancellationToken,
    ) {
        let Some(ref nats) = self.nats else {
            return;
        };

        let nats = nats.clone();
        let session_id = session_id.to_string();
        let graph = self.graph.clone();
        let context_injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let retry_config = self.config.retry.clone();
        let enrichment_pipeline = self.enrichment_pipeline.clone();
        let search = self.search.clone();
        let documents = self.document_store.clone();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_rpc_send(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(
                        "Failed to subscribe to NATS RPC send for session {}: {}",
                        session_id, e
                    );
                    return;
                }
            };

            info!(
                session_id = %session_id,
                "NATS RPC send listener started"
            );

            loop {
                let msg = tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS RPC listener cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        match msg {
                            Some(m) => m,
                            None => break,
                        }
                    }
                };
                // Parse the RPC request
                let request: crate::events::ChatRpcRequest =
                    match serde_json::from_slice(&msg.payload) {
                        Ok(req) => req,
                        Err(e) => {
                            warn!(
                                "Failed to parse NATS RPC request for session {}: {}",
                                session_id, e
                            );
                            // Reply with error if possible
                            if let Some(reply_to) = msg.reply {
                                let resp = crate::events::ChatRpcResponse {
                                    success: false,
                                    error: Some(format!("Invalid request: {}", e)),
                                };
                                if let Ok(payload) = serde_json::to_vec(&resp) {
                                    let _ = nats.client().publish(reply_to, payload.into()).await;
                                }
                            }
                            continue;
                        }
                    };

                debug!(
                    session_id = %session_id,
                    message_type = %request.message_type,
                    "NATS RPC send request received"
                );

                // Get session state — check if still active locally.
                // DON'T touch interrupt_flag or interrupt_token here — stream_response
                // owns their lifecycle (T1 fix for Gaps 1, 2, 4, 7, 10).
                let session_state = {
                    let mut sessions = active_sessions.write().await;
                    match sessions.get_mut(&session_id) {
                        Some(session) => {
                            session.last_activity = Instant::now();
                            Some((
                                session.client.clone(),
                                session.events_tx.clone(),
                                session.interrupt_flag.clone(),
                                session.memory_manager.clone(),
                                session.next_seq.clone(),
                                session.pending_messages.clone(),
                                session.is_streaming.clone(),
                                session.streaming_text.clone(),
                                session.streaming_events.clone(),
                                session.sdk_control_rx.clone(),
                                session.auto_continue.clone(),
                                session.stdin_tx.clone(),
                                session.interrupt_token.clone(),
                            ))
                        }
                        None => None,
                    }
                };

                let response = match session_state {
                    None => {
                        // Session no longer active — reply error and stop listener
                        debug!(
                            "Session {} no longer active, stopping NATS RPC listener",
                            session_id
                        );
                        let resp = crate::events::ChatRpcResponse {
                            success: false,
                            error: Some("Session not active on this instance".to_string()),
                        };
                        if let Some(reply_to) = msg.reply {
                            if let Ok(payload) = serde_json::to_vec(&resp) {
                                let _ = nats.client().publish(reply_to, payload.into()).await;
                            }
                        }
                        break;
                    }
                    Some((
                        client,
                        events_tx,
                        interrupt_flag,
                        memory_manager,
                        next_seq,
                        pending_messages,
                        is_streaming,
                        streaming_text,
                        streaming_events,
                        sdk_control_rx,
                        auto_continue,
                        stdin_tx,
                        interrupt_token,
                    )) => {
                        let message = &request.message;

                        // Route based on message_type
                        if request.message_type == "control_response" {
                            // Permission control response — send directly to CLI subprocess
                            // via SDK control protocol. Do NOT persist or broadcast.
                            let allow: bool = serde_json::from_str::<serde_json::Value>(message)
                                .ok()
                                .and_then(|v| v.get("allow").and_then(|a| a.as_bool()))
                                .unwrap_or(false);

                            info!(
                                session_id = %session_id,
                                allow,
                                "NATS RPC: Sending permission control response to CLI"
                            );

                            let response_json = serde_json::json!({ "allow": allow });
                            let mut cli = client.lock().await;
                            match cli.send_control_response(response_json).await {
                                Ok(()) => crate::events::ChatRpcResponse {
                                    success: true,
                                    error: None,
                                },
                                Err(e) => crate::events::ChatRpcResponse {
                                    success: false,
                                    error: Some(format!("Failed to send control response: {}", e)),
                                },
                            }
                        } else if request.message_type == "set_auto_continue" {
                            // Toggle auto-continue for this session (no CLI interaction needed)
                            let enabled: bool = serde_json::from_str::<serde_json::Value>(message)
                                .ok()
                                .and_then(|v| v.get("enabled").and_then(|e| e.as_bool()))
                                .unwrap_or(false);

                            auto_continue.store(enabled, Ordering::Relaxed);

                            // Persist to Neo4j
                            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                                if let Err(e) = graph.set_session_auto_continue(uuid, enabled).await
                                {
                                    warn!(
                                        session_id = %session_id,
                                        error = %e,
                                        "Failed to persist auto_continue via NATS RPC (non-fatal)"
                                    );
                                }
                            }

                            info!(
                                session_id = %session_id,
                                enabled = %enabled,
                                "NATS RPC: Auto-continue toggled"
                            );

                            // Broadcast state change event (local + NATS for other instances)
                            let event = ChatEvent::AutoContinueStateChanged {
                                session_id: session_id.clone(),
                                enabled,
                            };
                            let _ = events_tx.send(event.clone());
                            nats.publish_chat_event(&session_id, event);

                            crate::events::ChatRpcResponse {
                                success: true,
                                error: None,
                            }
                        } else if request.message_type == "queue_op" {
                            // An action on the held messages, relayed by the
                            // instance the client is connected to.
                            match serde_json::from_str::<super::pending_queue::QueueOp>(message) {
                                Ok(op) => {
                                    let (outcome, messages) = {
                                        let mut queue = pending_messages.lock().await;
                                        let outcome = super::pending_queue::apply(&mut queue, &op);
                                        (outcome, super::pending_queue::snapshot(&queue))
                                    };
                                    let event = ChatEvent::PendingQueue { messages };
                                    let _ = events_tx.send(event.clone());
                                    nats.publish_chat_event(&session_id, event);
                                    if outcome.interrupt && is_streaming.load(Ordering::SeqCst) {
                                        interrupt_flag.store(true, Ordering::SeqCst);
                                        interrupt_token.cancel();
                                        if let Some(ref tx) = stdin_tx {
                                            let _ =
                                                tx.try_send(
                                                    InteractiveClient::build_interrupt_json(),
                                                );
                                        }
                                    }
                                    crate::events::ChatRpcResponse {
                                        success: true,
                                        error: None,
                                    }
                                }
                                Err(e) => crate::events::ChatRpcResponse {
                                    success: false,
                                    error: Some(format!("Invalid queue operation: {}", e)),
                                },
                            }
                        } else if request.message_type == "queued_user_message" && {
                            // Held only while a turn runs — decided under the
                            // queue lock, like `queue_user_message`. Idle: fall
                            // through to the ordinary send below.
                            let mut queue = pending_messages.lock().await;
                            if is_streaming.load(Ordering::SeqCst) {
                                queue.push_back(PendingMessage::held_user(message.clone()));
                                let event = ChatEvent::PendingQueue {
                                    messages: super::pending_queue::snapshot(&queue),
                                };
                                drop(queue);
                                let _ = events_tx.send(event.clone());
                                nats.publish_chat_event(&session_id, event);
                                true
                            } else {
                                false
                            }
                        } {
                            info!(
                                "Holding user message for session {} (via NATS RPC, no interrupt)",
                                session_id
                            );
                            crate::events::ChatRpcResponse {
                                success: true,
                                error: None,
                            }
                        } else if is_streaming.load(Ordering::SeqCst) {
                            // If streaming → queue the message and interrupt so it's processed sooner (T4, Gap 8)
                            info!(
                                "Stream in progress for session {} (via NATS RPC), queuing message and interrupting",
                                session_id
                            );
                            let mut queue = pending_messages.lock().await;
                            queue.push_back(PendingMessage::user(message.clone()));

                            // Interrupt the stream so the message is processed sooner
                            interrupt_flag.store(true, Ordering::SeqCst);
                            interrupt_token.cancel();

                            // Send interrupt to CLI immediately via stdin_tx (lock-free)
                            if let Some(ref tx) = stdin_tx {
                                let json = InteractiveClient::build_interrupt_json();
                                let _ = tx.try_send(json);
                            }

                            crate::events::ChatRpcResponse {
                                success: true,
                                error: None,
                            }
                        } else {
                            // Not streaming — persist user_message, broadcast, spawn stream
                            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                                // Update message count
                                if let Ok(Some(node)) = graph.get_chat_session(uuid).await {
                                    let _ = graph
                                        .update_chat_session(
                                            uuid,
                                            None,
                                            None,
                                            Some(node.message_count + 1),
                                            None,
                                            None,
                                            None,
                                        )
                                        .await;
                                }

                                // Persist user_message event
                                let user_event = crate::neo4j::models::ChatEventRecord {
                                    id: Uuid::new_v4(),
                                    session_id: uuid,
                                    seq: next_seq.fetch_add(1, Ordering::SeqCst),
                                    event_type: "user_message".to_string(),
                                    data: serde_json::to_string(&ChatEvent::UserMessage {
                                        content: message.to_string(),
                                    })
                                    .unwrap_or_default(),
                                    created_at: chrono::Utc::now(),
                                };
                                let _ = graph.store_chat_events(uuid, vec![user_event]).await;
                            }

                            // Broadcast user_message locally + NATS
                            let user_msg_event = ChatEvent::UserMessage {
                                content: message.clone(),
                            };
                            let _ = events_tx.send(user_msg_event.clone());
                            nats.publish_chat_event(&session_id, user_msg_event);
                            crate::events::attention::notify_attention(
                                &event_emitter,
                                crate::events::attention::AttentionSubject::Session(
                                    session_id.clone(),
                                ),
                                crate::events::attention::AttentionReason::UserMessage,
                            );

                            // Spawn stream_response
                            let session_id_clone = session_id.clone();
                            let graph_clone = graph.clone();
                            let active_sessions_clone = active_sessions.clone();
                            let prompt = message.clone();
                            let injector = context_injector.clone();
                            let event_emitter_clone = event_emitter.clone();
                            let nats_clone = Some(nats.clone());
                            let retry_config_clone = retry_config.clone();
                            let enrichment_pipeline_clone = enrichment_pipeline.clone();
                            let search_clone = search.clone();
                            let documents_clone = documents.clone();

                            tokio::spawn(async move {
                                Self::stream_response(
                                    client,
                                    events_tx,
                                    prompt,
                                    session_id_clone,
                                    graph_clone,
                                    active_sessions_clone,
                                    interrupt_flag,
                                    memory_manager,
                                    injector,
                                    next_seq,
                                    pending_messages,
                                    is_streaming,
                                    streaming_text,
                                    streaming_events,
                                    event_emitter_clone,
                                    nats_clone,
                                    sdk_control_rx,
                                    auto_continue,
                                    retry_config_clone,
                                    enrichment_pipeline_clone,
                                    search_clone,
                                    documents_clone,
                                )
                                .await;
                            });

                            crate::events::ChatRpcResponse {
                                success: true,
                                error: None,
                            }
                        }
                    }
                };

                // Reply to the requester
                if let Some(reply_to) = msg.reply {
                    match serde_json::to_vec(&response) {
                        Ok(payload) => {
                            if let Err(e) = nats.client().publish(reply_to, payload.into()).await {
                                warn!(
                                    "Failed to reply to NATS RPC for session {}: {}",
                                    session_id, e
                                );
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Failed to serialize RPC response for session {}: {}",
                                session_id, e
                            );
                        }
                    }
                }
            }

            debug!("NATS RPC send listener stopped for session {}", session_id);
        });
    }

    /// Resolve the model to use: request > config default
    pub fn resolve_model(&self, request_model: Option<&str>) -> String {
        request_model
            .map(|m| m.to_string())
            .unwrap_or_else(|| self.config.default_model.clone())
    }

    /// [`Self::build_system_prompt`] plus the anchor map of the session, according
    /// to the anchor-context mode:
    /// - `off`: exactly `build_system_prompt`;
    /// - `shadow`: exactly `build_system_prompt` (byte for byte); the resolver runs
    ///   on the side and is journalled;
    /// - `on`: the anchor map (inside the untrusted container, stable for a given
    ///   cache key) is appended after the composed prompt. The composer returns one
    ///   string and the engines set no `cache_control` of their own, so "the prefix"
    ///   is this string: the map goes after the static sections and before nothing
    ///   that changes from one turn to the next (the live block never goes here).
    ///
    /// `project_slug` is the project already decided by the caller.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn build_system_prompt_anchored(
        &self,
        place: super::neutral_place::ExecutionPlace,
        cwd: &str,
        project_slug: Option<&str>,
        user_message: &str,
        model: Option<&str>,
        session_id: &str,
        scaffolding_override: Option<u8>,
    ) -> (String, std::collections::HashSet<String>) {
        use super::anchor_resolver::AnchorContextMode;
        let (prompt, ids) = self
            .build_system_prompt(
                project_slug,
                user_message,
                model,
                Some(session_id),
                scaffolding_override,
            )
            .await;
        let Ok(uuid) = Uuid::parse_str(session_id) else {
            return (prompt, ids);
        };
        match self.anchor_mode {
            AnchorContextMode::Off => (prompt, ids),
            AnchorContextMode::Shadow => {
                spawn_anchor_shadow(
                    self.graph.clone(),
                    self.anchor_cache.clone(),
                    uuid,
                    project_slug.map(str::to_string),
                    place,
                    cwd.to_string(),
                    project_slug.map(str::to_string),
                );
                (prompt, ids)
            }
            AnchorContextMode::On => {
                let inputs = super::anchor_resolver::ProjectInputs {
                    explicit_slug: project_slug,
                    place,
                    cwd,
                };
                match self
                    .anchor_cache
                    .resolve(
                        self.graph.as_ref(),
                        uuid,
                        &inputs,
                        &super::anchor_resolver::GraphCwdInference(self.graph.as_ref()),
                    )
                    .await
                {
                    Ok((r, _)) => (format!("{prompt}\n\n---\n\n{}", r.map()), ids),
                    Err(e) => {
                        warn!(session_id = %session_id, error = %e, "anchor map not built, prompt unchanged");
                        (prompt, ids)
                    }
                }
            }
        }
    }

    /// Build the system prompt with project context.
    ///
    /// Modular architecture via FsmPromptComposer:
    /// 1. Base sections — selected by scaffolding level (0-4)
    /// 2. FSM context — injected from active protocol runs
    /// 3. Dynamic context — markdown rendering of project context + session continuity
    /// 4. Tool reference — selectively rendered by intent + FSM + scaffolding
    ///
    /// Build the system prompt and return the set of note IDs included
    /// (from guidelines/gotchas) for deduplication with the enrichment pipeline.
    pub async fn build_system_prompt(
        &self,
        project_slug: Option<&str>,
        user_message: &str,
        model: Option<&str>,
        session_id: Option<&str>,
        scaffolding_override: Option<u8>,
    ) -> (String, std::collections::HashSet<String>) {
        use super::composer::{ComposerInput, FsmPromptComposer};
        use super::prompt::{context_to_markdown, fetch_project_context};
        use super::routing::HeuristicRouter;
        use super::stages::status_injection::GraphProtocolProvider;
        use super::stages::status_injection::ProtocolStatusProvider;

        let empty_ids = std::collections::HashSet::new();

        // No project → compose with defaults (L0, no FSM, no dynamic context)
        let Some(slug) = project_slug else {
            let input = ComposerInput {
                cite_refs: self.refs_v1,
                ..Default::default()
            };
            return (FsmPromptComposer::compose(&input), empty_ids);
        };

        // Fetch raw context from Neo4j
        let ctx = match fetch_project_context(&self.graph, slug).await {
            Ok(ctx) => ctx,
            Err(e) => {
                warn!(
                    "Failed to fetch project context for '{}': {} — using base prompt only",
                    slug, e
                );
                let input = ComposerInput {
                    cite_refs: self.refs_v1,
                    ..Default::default()
                };
                return (FsmPromptComposer::compose(&input), empty_ids);
            }
        };

        // Collect note IDs from guidelines/gotchas for deduplication with enrichment
        let included_note_ids: std::collections::HashSet<String> = ctx
            .guidelines
            .iter()
            .chain(ctx.gotchas.iter())
            .chain(ctx.global_guidelines.iter())
            .chain(ctx.global_gotchas.iter())
            .map(|n| n.id.to_string())
            .collect();

        // Hook Niveau 3: auto-boost notes included in the system prompt context
        {
            let mut note_ids: Vec<uuid::Uuid> = Vec::new();
            for n in &ctx.guidelines {
                note_ids.push(n.id);
            }
            for n in &ctx.gotchas {
                note_ids.push(n.id);
            }
            for n in &ctx.global_guidelines {
                note_ids.push(n.id);
            }
            for n in &ctx.global_gotchas {
                note_ids.push(n.id);
            }
            if !note_ids.is_empty() {
                let graph = self.graph.clone();
                let boost = 0.05; // context_energy_boost
                tokio::spawn(async move {
                    for id in &note_ids {
                        if let Err(e) = graph.boost_energy(*id, boost).await {
                            tracing::warn!(
                                note_id = %id,
                                error = %e,
                                "Auto-reinforce context: energy boost failed"
                            );
                        }
                    }
                    tracing::debug!(
                        notes = note_ids.len(),
                        "Auto-reinforced notes included in system prompt"
                    );
                });
            }
        }

        // ── Fetch scaffolding level (with 100ms timeout, fallback to L0) ──
        let scaffolding_level = if let Some(ref project) = ctx.project {
            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                self.graph
                    .compute_scaffolding_level(project.id, project.scaffolding_override),
            )
            .await
            {
                Ok(Ok(level)) => {
                    debug!("[composer] Scaffolding L{} for '{}'", level.level, slug);
                    level.level
                }
                Ok(Err(e)) => {
                    warn!(
                        "[composer] Scaffolding computation failed: {} — defaulting to L0",
                        e
                    );
                    0
                }
                Err(_) => {
                    warn!("[composer] Scaffolding computation timed out — defaulting to L0");
                    0
                }
            }
        } else {
            0
        };

        // Override with inherited scaffolding level from parent session
        let scaffolding_level = scaffolding_override.unwrap_or(scaffolding_level);

        // ── Fetch active protocol runs ──────────────────────────────────
        let protocol_runs = if let Some(ref project) = ctx.project {
            let provider = GraphProtocolProvider::new(self.graph.clone());
            match tokio::time::timeout(
                std::time::Duration::from_millis(200),
                provider.get_active_runs(project.id),
            )
            .await
            {
                Ok(Ok(runs)) => {
                    if !runs.is_empty() {
                        debug!("[composer] {} active protocol runs", runs.len());
                    }
                    runs
                }
                Ok(Err(e)) => {
                    warn!("[composer] Failed to fetch protocol runs: {}", e);
                    Vec::new()
                }
                Err(_) => {
                    warn!("[composer] Protocol runs fetch timed out");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // ── Build dynamic context (markdown) ─────────────────────────────
        let dynamic_section = context_to_markdown(&ctx, Some(user_message));

        // ── Load session continuity context ─────────────────────────────
        let continuity_section = {
            let graph = self.graph.clone();
            let slug_owned = slug.to_string();
            match super::continuity::load_session_context(&graph, &slug_owned).await {
                Ok(resume) if resume.has_content() => {
                    debug!(
                        "[continuity] Injecting session context ({}ms)",
                        resume.load_time_ms
                    );
                    resume.to_markdown()
                }
                Ok(_) => String::new(),
                Err(e) => {
                    warn!("[continuity] Failed to load session context: {}", e);
                    String::new()
                }
            }
        };

        // ── Derive context flags from ProjectContext ────────────────────
        let has_active_plan = !ctx.active_plans.is_empty();
        let task_count = ctx.active_plans.len() * 3; // rough estimate
        let is_multi_project = !ctx.sibling_projects.is_empty();

        // ── Pre-compute message embedding for neural routing ─────────────
        let message_embedding = if self
            .neural_routing_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            let builder = neural_routing_runtime::DecisionVectorBuilder::new();
            let ctx = neural_routing_runtime::DecisionContext {
                query_embedding: vec![],
                touched_node_features: vec![],
                previous_embeddings: vec![],
                tool_name: "chat".to_string(),
                action_name: user_message.to_string(),
                params_hash: 0,
                session_meta: neural_routing_runtime::SessionMeta::default(),
            };
            Some(builder.build(&ctx))
        } else {
            None
        };

        // ── Compose via FsmPromptComposer ───────────────────────────────
        // Check if external MCP servers are connected
        let external_tools_available = !self.mcp_registry.read().await.is_empty();

        let input = ComposerInput {
            scaffolding_level,
            protocol_runs: &protocol_runs,
            project_context_markdown: &dynamic_section,
            continuity_markdown: &continuity_section,
            enrichment_markdown: "", // enrichment injected later in stream_response
            user_message,
            routing_hints: None,
            is_multi_project,
            has_active_plan,
            task_count,
            model: model.unwrap_or(""),
            message_embedding: message_embedding.as_ref(),
            external_tools_available,
            cite_refs: self.refs_v1,
        };

        // Use compose_with_record to get the routing decision for trajectory tracking.
        // When neural routing is enabled and a DualTrackRouter is configured, use it
        // instead of the default HeuristicRouter for config-driven router selection.
        let use_neural = self
            .neural_routing_enabled
            .load(std::sync::atomic::Ordering::Relaxed);
        let (prompt, routing_record) = {
            let dtr_guard = self.dual_track_router.read().unwrap();
            if use_neural {
                if let Some(ref dtr) = *dtr_guard {
                    FsmPromptComposer::compose_with_record(&input, dtr)
                } else {
                    FsmPromptComposer::compose_with_record(&input, &HeuristicRouter)
                }
            } else {
                FsmPromptComposer::compose_with_record(&input, &HeuristicRouter)
            }
        };

        // Emit routing decision to trajectory collector (fire-and-forget)
        {
            let tc_guard = self.trajectory_collector.read().unwrap();
            if let (Some(ref collector), Some(sid)) = (&*tc_guard, session_id) {
                // Emit routing decision directly to trajectory collector
                let params = serde_json::to_value(&routing_record).unwrap_or_default();
                collector.record_decision(neural_routing_runtime::DecisionRecord {
                    session_id: sid.to_string(),
                    context_embedding: vec![],
                    action_type: "routing.select_sections".to_string(),
                    action_params: params,
                    alternatives_count: routing_record.selected_sections.len(),
                    chosen_index: 0,
                    confidence: if routing_record.section_weights.is_empty() {
                        0.5
                    } else {
                        let sum: f32 = routing_record.section_weights.iter().map(|(_, w)| w).sum();
                        (sum / routing_record.section_weights.len() as f32) as f64
                    },
                    tool_usages: vec![],
                    touched_entities: vec![],
                    timestamp_ms: 0,
                    query_embedding: vec![],
                    node_features: vec![],
                    protocol_run_id: None,
                    protocol_state: None,
                });
                debug!(
                    "[routing] Emitted routing decision to trajectory: {} sections, {} tool groups",
                    routing_record.selected_sections.len(),
                    routing_record.selected_tool_groups.len()
                );
            }
        }

        // Query NNRouter for trajectory-based action suggestions (populates dashboard metrics)
        if let (Some(nn_router), Some(emb)) = (&self.nn_router, &message_embedding) {
            let emb: &Vec<f32> = emb;
            if !emb.is_empty() {
                let router_guard = nn_router.read().await;
                match tokio::time::timeout(
                    std::time::Duration::from_millis(15),
                    neural_routing_runtime::Router::route(&*router_guard, emb),
                )
                .await
                {
                    Ok(Ok(Some(route))) => {
                        tracing::info!(
                            similarity = %format!("{:.3}", route.similarity),
                            source_trajectory = %route.source_trajectory_id,
                            actions = route.actions.len(),
                            "NN route matched"
                        );
                    }
                    Ok(Ok(None)) => {
                        tracing::debug!("NN route: no match found");
                    }
                    Ok(Err(e)) => {
                        tracing::debug!(error = %e, "NN route query failed");
                    }
                    Err(_) => {
                        tracing::debug!("NN route query timed out (>15ms)");
                    }
                }
            }
        }

        (prompt, included_note_ids)
    }

    /// Sessions whose CLI is alive, those currently streaming, and the
    /// permission requests each live CLI still holds in memory. The sessions
    /// map is read under ONE lock (the cockpit must not take it once per
    /// session); the per-session permission maps are locked after it is
    /// released.
    pub async fn live_session_snapshot(&self) -> LiveSessionSnapshot {
        let mut snap = LiveSessionSnapshot::default();
        let mut inputs = Vec::new();
        let mut tasks = Vec::new();
        {
            let sessions = self.active_sessions.read().await;
            for (id, s) in sessions.iter() {
                if let Ok(id) = id.parse::<Uuid>() {
                    snap.live.insert(id);
                    if s.is_streaming.load(Ordering::SeqCst) {
                        snap.streaming.insert(id);
                    }
                    inputs.push((id, s.pending_permission_inputs.clone()));
                    tasks.push((id, s.active_background_tasks.clone()));
                }
            }
        }
        for (id, pending) in inputs {
            let ids: std::collections::HashSet<String> =
                pending.lock().await.keys().cloned().collect();
            if !ids.is_empty() {
                snap.pending_permissions.insert(id, ids);
            }
        }
        for (id, map) in tasks {
            let counts = count_background_tasks(map.lock().await.values());
            if counts != (0, 0) {
                snap.background_tasks.insert(id, counts);
            }
        }
        snap
    }

    /// Live activity for every session the manager knows about, keyed by id.
    /// Sessions absent from the map are quiet; callers fill them in with
    /// [`SessionActivity::default`] rather than treating absence as unknown.
    pub async fn session_activity_map(&self) -> std::collections::HashMap<Uuid, SessionActivity> {
        let snap = self.live_session_snapshot().await;
        snap.live
            .iter()
            .map(|id| (*id, snap.activity_for(*id)))
            .collect()
    }

    /// Check if a session is currently active (subprocess alive)
    pub async fn is_session_active(&self, session_id: &str) -> bool {
        if self.agent_runtime.owns(session_id).await {
            return true;
        }
        self.active_sessions.read().await.contains_key(session_id)
    }

    /// The live permission mode of an active session (`None` when the session
    /// is not active or runs on the global default).
    pub async fn live_session_permission_mode(&self, session_id: &str) -> Option<String> {
        self.active_sessions
            .read()
            .await
            .get(session_id)
            .and_then(|s| s.permission_mode.clone())
    }

    /// The permission mode a session gets when its request names none.
    pub async fn default_permission_mode(&self) -> String {
        self.permission_config.read().await.mode.clone()
    }

    // ========================================================================
    // ClaudeCodeOptions builder
    // ========================================================================

    /// Resolve additional directories for the Claude CLI `--add-dir` flag.
    ///
    /// Priority:
    /// 1. Explicit `add_dirs` from the request → use as-is (with tilde expansion)
    /// 2. `workspace_slug` → resolve all project root_paths, exclude cwd to avoid duplicates
    /// 3. Neither → empty vec (no additional dirs)
    pub async fn resolve_add_dirs(
        &self,
        cwd: &str,
        add_dirs: Option<&[String]>,
        workspace_slug: Option<&str>,
        _project_slug: Option<&str>,
    ) -> Vec<String> {
        // Case 1: explicit add_dirs
        if let Some(dirs) = add_dirs {
            if !dirs.is_empty() {
                return dirs.iter().map(|d| expand_tilde(d)).collect();
            }
        }

        // Case 2: workspace_slug → resolve projects
        if let Some(ws_slug) = workspace_slug {
            match self.graph.get_workspace_by_slug(ws_slug).await {
                Ok(Some(ws)) => match self.graph.list_workspace_projects(ws.id).await {
                    Ok(projects) => {
                        let expanded_cwd = expand_tilde(cwd);
                        return projects
                            .into_iter()
                            // Projects without a codebase contribute no directory.
                            .filter_map(|p| p.expanded_root_path())
                            .filter(|path| *path != expanded_cwd)
                            .collect();
                    }
                    Err(e) => {
                        tracing::warn!(
                            workspace_slug = ws_slug,
                            "Failed to list workspace projects for add_dirs: {}",
                            e
                        );
                    }
                },
                Ok(None) => {
                    tracing::warn!(
                        workspace_slug = ws_slug,
                        "Workspace not found for add_dirs resolution"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        workspace_slug = ws_slug,
                        "Failed to get workspace for add_dirs: {}",
                        e
                    );
                }
            }
        }

        // Case 3: no add_dirs
        Vec::new()
    }

    /// The provider of the session `session_id` was spawned by
    /// (`spawned_by.parent_session_id`, read from the graph — a resume by a person
    /// still knows). A parent that cannot be read counts as third-party: refusing
    /// a tool is recoverable, a delegation loop is not.
    async fn spawn_parent(&self, session_id: Option<&str>) -> SpawnParent {
        let Some(sid) = session_id.and_then(|s| Uuid::parse_str(s).ok()) else {
            return SpawnParent::None;
        };
        let Ok(Some(node)) = self.graph.get_chat_session(sid).await else {
            return SpawnParent::None;
        };
        let Some(parent) = node
            .spawned_by
            .as_deref()
            .and_then(parse_spawned_by)
            .and_then(|ctx| ctx.parent_session_id)
        else {
            return SpawnParent::None;
        };
        let Ok(parent) = Uuid::parse_str(&parent) else {
            return SpawnParent::ThirdParty;
        };
        match self.graph.get_chat_session(parent).await {
            Ok(Some(parent))
                if parent
                    .provider_id
                    .as_deref()
                    .is_none_or(|p| p == super::provider::resolver::CLAUDE_CODE) =>
            {
                SpawnParent::ClaudeCode
            }
            _ => SpawnParent::ThirdParty,
        }
    }

    /// Where a session comes from, for its tool profile (H6).
    ///
    /// `opened_by_third_party`: its caller's signed token carries the lineage
    /// (`chat send_message`, `plan run`: the request carries the caller's claims),
    /// or its parent session runs on a provider other than Claude Code.
    ///
    /// `origin_known`: a person or a session is behind it. A session opened under
    /// the server's own service account (a protocol run, a plan run resumed after a
    /// restart, a delegation) has lost its caller: only a Claude Code parent
    /// vouches for it. Without that, a third party in `trust` could start a
    /// protocol whose third-party agent is `full` again, and so on.
    pub(crate) async fn session_origin(
        &self,
        user_claims: Option<&crate::auth::jwt::Claims>,
        session_id: Option<&str>,
    ) -> SessionOrigin {
        let parent = self.spawn_parent(session_id).await;
        let lineage = user_claims
            .and_then(crate::auth::jwt::agent_session_binding)
            .is_some_and(|b| b.third_party);
        SessionOrigin {
            opened_by_third_party: lineage || parent == SpawnParent::ThirdParty,
            origin_known: user_claims.is_some_and(|c| !c.is_service_account())
                || parent == SpawnParent::ClaudeCode,
        }
    }

    /// Environment of the project-orchestrator MCP server of one session: the
    /// server URL, the session-BOUND token (registered live, revoked by
    /// `close_session`), the vault token and the session id. Shared by the
    /// Claude path (`build_options`) and the agent path (`build_agent_spec`) so
    /// both hand the agent exactly the same, and nothing else (A33).
    ///
    /// `tool_profile` is what the session's own provider and mode grant it;
    /// `third_party` whether that provider is not Claude Code. A session opened
    /// by a third-party session gets the restricted profile whatever it was
    /// granted, and carries the lineage on; a third party whose origin is unknown
    /// (the service account, no Claude Code parent) is restricted too
    /// ([`Self::session_origin`]).
    pub(crate) async fn po_mcp_env(
        &self,
        permission_mode_override: Option<&str>,
        user_claims: Option<&crate::auth::jwt::Claims>,
        session_id: Option<&str>,
        tool_profile: Option<&str>,
        third_party: bool,
    ) -> HashMap<String, String> {
        let mut env = HashMap::new();
        let SessionOrigin {
            opened_by_third_party,
            origin_known,
        } = self.session_origin(user_claims, session_id).await;
        // A third party gets `full` (trust) only when a person or a Claude Code
        // session is behind it.
        // A read-only session stays read-only: that profile is narrower than the
        // restricted one a third-party lineage would put in its place.
        let read_only = tool_profile == Some(crate::auth::tool_profile::READ_ONLY);
        let tool_profile =
            if !read_only && (opened_by_third_party || (third_party && !origin_known)) {
                Some(crate::auth::tool_profile::RESTRICTED)
            } else {
                tool_profile
            };

        if read_only {
            env.insert(
                crate::auth::tool_profile::TOOL_PROFILE_ENV.into(),
                crate::auth::tool_profile::READ_ONLY.into(),
            );
        }

        // PO_SERVER_URL is always injected — mcp_server runs as an HTTP proxy
        // regardless of whether auth is enabled.
        env.insert(
            "PO_SERVER_URL".into(),
            format!("http://127.0.0.1:{}", self.config.server_port),
        );

        // Inject session token only when auth is enabled (jwt_secret present)
        // AND user claims are available.
        if let (Some(ref secret), Some(claims)) = (&self.config.jwt_secret, user_claims) {
            // The token is BOUND to the session (id + policy ceiling signed in)
            // and registered as live; `close_session` revokes it. The ceiling is
            // the session's effective permission mode at spawn: anything this
            // session spawns can only be as permissive or less.
            let ceiling = match permission_mode_override {
                Some(mode) => mode.to_string(),
                None => self.permission_config.read().await.mode.clone(),
            };
            let binding = session_id.map(|sid| crate::auth::jwt::AgentSessionBinding {
                session_id: sid.to_string(),
                ceiling: Some(ceiling),
                tool_profile: tool_profile.map(str::to_string),
                third_party: third_party || opened_by_third_party,
            });
            match crate::auth::jwt::generate_session_token(
                claims,
                binding.as_ref(),
                secret,
                self.config.session_token_expiry_secs,
            ) {
                Ok((token, jti)) => {
                    crate::auth::agent_tokens::register(&jti, session_id);
                    env.insert("PO_AUTH_TOKEN".into(), token);
                    tracing::debug!(
                        user = %claims.email,
                        expiry_secs = self.config.session_token_expiry_secs,
                        server_port = self.config.server_port,
                        bound_session = ?session_id,
                        "Injected PO_AUTH_TOKEN into MCP env"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to generate MCP session token: {} — MCP will run without auth",
                        e
                    );
                }
            }
        }

        // Vault token: signs the session id, so the server knows which session
        // is asking for a secret without trusting anything the agent can edit.
        // Needs auth (a signing key) and a session; otherwise the agent has no
        // vault access at all.
        let vault_token = match (
            &self.vault,
            &self.config.jwt_secret,
            user_claims,
            session_id,
        ) {
            (Some(_), Some(secret), Some(claims), Some(sid)) => {
                match crate::auth::jwt::generate_vault_token(
                    claims,
                    sid,
                    secret,
                    self.config.session_token_expiry_secs,
                ) {
                    Ok(t) => Some(t),
                    Err(e) => {
                        tracing::warn!("Failed to mint vault token: {e} — no vault access");
                        None
                    }
                }
            }
            _ => None,
        };
        if let Some(ref t) = vault_token {
            env.insert("PO_VAULT_TOKEN".into(), t.clone());
        }

        // Inject session ID so MCP subprocess can send it as X-Session-Id header
        // on all REST API calls — enables server-side auto-linking of sessions
        // to tasks/plans without the agent needing to pass session_id explicitly.
        if let Some(sid) = session_id {
            env.insert("PO_SESSION_ID".into(), sid.to_string());
        }
        env
    }

    /// Build `ClaudeCodeOptions` for a new or resumed session.
    ///
    /// `permission_mode_override`: if Some, overrides the global config permission mode
    /// for this specific session (e.g. user chose a different mode for this session).
    #[allow(deprecated, clippy::too_many_arguments)]
    pub async fn build_options(
        &self,
        cwd: &str,
        model: &str,
        system_prompt: &str,
        resume_id: Option<&str>,
        permission_mode_override: Option<&str>,
        hooks: Option<std::collections::HashMap<String, Vec<nexus_claude::HookMatcher>>>,
        add_dirs: &[String],
        user_claims: Option<&crate::auth::jwt::Claims>,
        session_id: Option<&str>,
    ) -> ClaudeCodeOptions {
        self.build_options_with_access(
            cwd,
            model,
            system_prompt,
            resume_id,
            permission_mode_override,
            hooks,
            add_dirs,
            user_claims,
            session_id,
            super::provider::policy::SessionAccess::Normal,
        )
        .await
    }

    /// [`Self::build_options`] for a session of the given access: the read-only deny list
    /// is added to `disallowed_tools` (the CLI refuses it whatever the permission mode).
    #[allow(deprecated, clippy::too_many_arguments)]
    pub async fn build_options_with_access(
        &self,
        cwd: &str,
        model: &str,
        system_prompt: &str,
        resume_id: Option<&str>,
        permission_mode_override: Option<&str>,
        hooks: Option<std::collections::HashMap<String, Vec<nexus_claude::HookMatcher>>>,
        add_dirs: &[String],
        user_claims: Option<&crate::auth::jwt::Claims>,
        session_id: Option<&str>,
        access: super::provider::policy::SessionAccess,
    ) -> ClaudeCodeOptions {
        // A neutral directory of the host is made on demand: a new session and a resume alike.
        if let Err(e) = super::neutral_place::ensure(cwd) {
            warn!(cwd = %cwd, error = %e, "could not create the neutral chat directory");
        }
        // Expand tilde in cwd (shell doesn't expand ~ when passed via Command)
        let cwd = expand_tilde(cwd);
        let mcp_path = self.config.mcp_server_path.to_string_lossy().to_string();

        // The MCP server is an HTTP proxy to this server (`McpHttpClient`): it
        // needs a URL and a token, nothing else. The database and search
        // credentials used to be copied here "just in case"; they were then
        // readable by the agent (decision A33).
        let env = self
            .po_mcp_env(
                permission_mode_override,
                user_claims,
                session_id,
                access.tool_profile(),
                false,
            )
            .await;

        let vault_token = env.get("PO_VAULT_TOKEN").cloned();

        let mcp_config = McpServerConfig::Stdio {
            command: mcp_path,
            args: None,
            env: Some(env),
        };

        // Read the runtime-mutable permission config (updated via REST API)
        let perm_config = self.permission_config.read().await;

        // Use per-session override if provided, otherwise use global config
        let effective_permission = match permission_mode_override {
            Some(mode_str) => {
                let override_config = super::config::PermissionConfig {
                    mode: mode_str.to_string(),
                    ..Default::default()
                };
                override_config.to_nexus_mode()
            }
            None => perm_config.to_nexus_mode(),
        };

        let mut builder = ClaudeCodeOptions::builder()
            .model(model)
            .cwd(cwd)
            .system_prompt(system_prompt)
            .permission_mode(effective_permission)
            .max_turns(self.config.max_turns)
            .include_partial_messages(true)
            .permission_prompt_tool_name("stdio")
            .cli_channel_buffer_size(8192)
            // The agent starts from a CLEAN environment: only an allowlist of
            // the server's variables reaches the CLI and every shell it opens.
            .env_policy(child_env_policy())
            // The MCP config holds the session token: hand it to the CLI in a
            // 0600 file removed with the session, not on its command line
            // (where `ps` shows it to any process of the same user).
            .mcp_config_via_file(true)
            .add_mcp_server("project-orchestrator", mcp_config);

        // Wire allowed/disallowed tool patterns from config
        if !perm_config.allowed_tools.is_empty() {
            builder = builder.allowed_tools(perm_config.allowed_tools.clone());
        }
        // The harness decides what a read-only session cannot call; the prompt plays no part.
        let disallowed = access.merge_disallowed(&perm_config.disallowed_tools);
        if !disallowed.is_empty() {
            builder = builder.disallowed_tools(disallowed);
        }

        if let Some(id) = resume_id {
            builder = builder.resume(id);
        }

        if let Some(hook_map) = hooks {
            builder = builder.hooks(hook_map);
        }

        // Add additional directories (--add-dir flags)
        for dir in add_dirs {
            builder = builder.add_dir(expand_tilde(dir));
        }

        // Inject custom PATH and CLI path from runtime-mutable env config
        {
            let env = self.env_config.read().await;
            if let Some(ref path) = env.process_path {
                builder = builder.env("PATH", path);
                tracing::debug!("Injecting custom PATH into Claude subprocess");
            }
            if let Some(ref cli) = env.claude_cli_path {
                builder = builder.cli_path(cli);
                tracing::debug!("Using custom Claude CLI path: {}", cli);
            }
        }

        // The agent's shell reads granted secrets with `orchestrator secret get`,
        // which needs these two. The value then flows through a pipe, never
        // through the model's context.
        if let Some(t) = vault_token {
            builder = builder.env("PO_VAULT_TOKEN", &t).env(
                "PO_SERVER_URL",
                format!("http://127.0.0.1:{}", self.config.server_port),
            );
        }

        // The CLI inherits the server's whole environment, and so does every
        // shell the agent opens. When the server was configured through env
        // vars (Docker, .env), its own secrets would sit in the agent's `env`.
        // Override them with an empty value in the CHILD only — mutating the
        // server's environment at runtime would race with other threads.
        for name in server_secrets_to_hide(|k| std::env::var_os(k).is_some()) {
            builder = builder.env(name, "");
        }

        builder.build()
    }

    // ========================================================================
    // Message → ChatEvent conversion
    // ========================================================================

    /// Replace every secret value delivered by the vault inside a CLI message,
    /// before anything reads, stores or broadcasts it.
    ///
    /// Fails CLOSED: a message that cannot be masked is not passed on. It is
    /// replaced by a marker ([`MASKING_FAILED_SUBTYPE`]) that surfaces as a
    /// visible error, never by the original.
    pub(crate) fn mask_cli_message(msg: Message) -> Message {
        let masker = crate::vault::mask::global().snapshot();
        Self::mask_cli_message_with(&masker, msg)
    }

    /// [`Self::mask_cli_message`] against an explicit masker (testable).
    pub(crate) fn mask_cli_message_with(
        masker: &crate::vault::mask::Masker,
        msg: Message,
    ) -> Message {
        match crate::vault::mask::mask_serde(masker, msg) {
            Ok(m) => m,
            // Reached only when a secret value was found AND the masked form no
            // longer reads back (the value collides with the message structure).
            // The original holds the secret in clear: drop it.
            Err(_unmasked) => {
                tracing::error!(
                    "vault: a CLI message holding a secret could not be masked; withholding it"
                );
                Message::System {
                    subtype: MASKING_FAILED_SUBTYPE.to_string(),
                    data: serde_json::Value::Null,
                }
            }
        }
    }

    /// Convert a Nexus SDK `Message` to a list of `ChatEvent`s
    pub fn message_to_events(msg: &Message) -> Vec<ChatEvent> {
        // Extract parent_tool_use_id from the Message (sidechain indicator).
        // When present, this event originated from a sub-agent spawned by a Task tool.
        let parent = msg.parent_tool_use_id().map(|s| s.to_string());

        match msg {
            Message::Assistant { message, .. } => {
                let mut events = Vec::new();
                for block in &message.content {
                    match block {
                        ContentBlock::Text(t) => {
                            events.push(ChatEvent::AssistantText {
                                content: t.text.clone(),
                                parent_tool_use_id: parent.clone(),
                            });
                        }
                        ContentBlock::Thinking(t) => {
                            events.push(ChatEvent::Thinking {
                                content: t.thinking.clone(),
                                parent_tool_use_id: parent.clone(),
                            });
                        }
                        ContentBlock::ToolUse(t) => {
                            events.push(ChatEvent::ToolUse {
                                id: t.id.clone(),
                                tool: t.name.clone(),
                                input: t.input.clone(),
                                parent_tool_use_id: parent.clone(),
                                category: None,
                                canonical: None,
                            });
                        }
                        ContentBlock::ToolResult(t) => {
                            let result = match &t.content {
                                Some(ContentValue::Text(s)) => serde_json::Value::String(s.clone()),
                                Some(ContentValue::Structured(v)) => {
                                    serde_json::Value::Array(v.clone())
                                }
                                None => serde_json::Value::Null,
                            };
                            events.push(ChatEvent::ToolResult {
                                id: t.tool_use_id.clone(),
                                result,
                                is_error: t.is_error.unwrap_or(false),
                                parent_tool_use_id: parent.clone(),
                            });
                        }
                    }
                }
                events
            }
            Message::Result {
                session_id,
                duration_ms,
                total_cost_usd,
                subtype,
                is_error,
                num_turns,
                result,
                ..
            } => {
                vec![ChatEvent::Result {
                    session_id: session_id.clone(),
                    duration_ms: *duration_ms as u64,
                    cost_usd: *total_cost_usd,
                    subtype: subtype.clone(),
                    is_error: *is_error,
                    num_turns: Some(*num_turns),
                    result_text: result.clone(),
                    cost: None,
                    usage: None,
                    model: None,
                    stop_reason: None,
                }]
            }
            Message::StreamEvent { event, .. } => match event {
                StreamEventData::ContentBlockDelta {
                    delta: StreamDelta::TextDelta { text },
                    ..
                } => {
                    vec![ChatEvent::StreamDelta {
                        text: text.clone(),
                        parent_tool_use_id: parent,
                    }]
                }
                // Extract tool_use from ContentBlockStart — this is where tool calls
                // first appear in the stream (before the AssistantMessage is finalized).
                // The content_block is raw JSON: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
                StreamEventData::ContentBlockStart { content_block, .. } => {
                    if content_block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                        let id = content_block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = content_block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = content_block
                            .get("input")
                            .cloned()
                            .unwrap_or(serde_json::json!({}));
                        vec![ChatEvent::ToolUse {
                            id,
                            tool: name,
                            input,
                            parent_tool_use_id: parent,
                            category: None,
                            canonical: None,
                        }]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            },
            Message::System { subtype, data } => {
                match subtype.as_str() {
                    MASKING_FAILED_SUBTYPE => vec![ChatEvent::Error {
                        message: MASKING_FAILED_MESSAGE.to_string(),
                        parent_tool_use_id: None,
                        code: None,
                        reason: None,
                        index: None,
                    }],
                    "init" => {
                        // Extract session metadata from init system message
                        let cli_session_id = data
                            .get("session_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let model = data
                            .get("model")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let tools = data
                            .get("tools")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        let mcp_servers = data
                            .get("mcp_servers")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        let permission_mode = data
                            .get("permissionMode")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        vec![ChatEvent::SystemInit {
                            cli_session_id,
                            model,
                            tools,
                            mcp_servers,
                            permission_mode,
                            provider: None,
                            capabilities: None,
                            tool_policy: None,
                            policy_mode: None,
                            // The historical engine does everything: nothing is missing.
                            engine: Some("legacy".to_string()),
                            degraded_features: Some(Vec::new()),
                        }]
                    }
                    "compact_boundary" => {
                        // Extract compact metadata from data.compact_metadata
                        let metadata = data.get("compact_metadata");
                        let trigger = metadata
                            .and_then(|m| m.get("trigger"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("auto")
                            .to_string();
                        let pre_tokens = metadata
                            .and_then(|m| m.get("pre_tokens"))
                            .and_then(|v| v.as_u64());
                        vec![ChatEvent::CompactBoundary {
                            trigger,
                            pre_tokens,
                        }]
                    }
                    // Claude Code Dynamic Workflow lifecycle events (intra-task
                    // sub-agent fan-out via the `Workflow` tool). These arrive as
                    // flat system messages; forward subtype + full payload so the
                    // frontend can render live fan-out progress. (Requires the SDK
                    // to preserve the flat payload — nexus PR #31.)
                    "task_started" | "task_progress" | "task_updated" | "task_notification" => {
                        vec![ChatEvent::Workflow {
                            subtype: subtype.clone(),
                            data: data.clone(),
                        }]
                    }
                    _ => {
                        debug!("Unhandled system message: {} — {:?}", subtype, data);
                        vec![]
                    }
                }
            }
            Message::User { message, .. } => {
                // User messages with content_blocks contain tool_result blocks
                // from the CLI's tool execution. Extract them as ChatEvent::ToolResult.
                if let Some(blocks) = &message.content_blocks {
                    let mut events = Vec::new();
                    for block in blocks {
                        if let ContentBlock::ToolResult(t) = block {
                            let result = match &t.content {
                                Some(ContentValue::Text(s)) => serde_json::Value::String(s.clone()),
                                Some(ContentValue::Structured(v)) => {
                                    serde_json::Value::Array(v.clone())
                                }
                                None => serde_json::Value::Null,
                            };
                            events.push(ChatEvent::ToolResult {
                                id: t.tool_use_id.clone(),
                                result,
                                is_error: t.is_error.unwrap_or(false),
                                parent_tool_use_id: parent.clone(),
                            });
                        }
                    }
                    events
                } else {
                    vec![]
                }
            }
        }
    }

    // ========================================================================
    // Session lifecycle
    // ========================================================================

    /// Create a new chat session: persist to Neo4j, spawn CLI subprocess, start streaming
    pub async fn create_session(&self, request: &ChatRequest) -> Result<CreateSessionResponse> {
        self.create_session_relayed(request, None).await
    }

    /// [`Self::create_session`] with a history relayed from another provider (B-SW):
    /// the relay is sent to the model in front of `request.message`, but the conversation
    /// stores and shows `request.message` alone, behind a `conversation_relayed` event
    /// that says how much was carried over.
    pub async fn create_session_relayed(
        &self,
        request: &ChatRequest,
        relay: Option<&super::relay::RelayedFrom>,
    ) -> Result<CreateSessionResponse> {
        // The menu's ticks, read once: one model ticked is strict, an empty pool is none.
        let settled = request.settled_routing();
        let request = settled.as_ref().unwrap_or(request);
        let session_id = Uuid::new_v4();
        if !request.cwd.trim().is_empty() {
            return self
                .create_session_in_place(request, relay, session_id)
                .await;
        }
        // No `cwd`: the host gives the session a neutral working directory of its own,
        // derived from the id the session is persisted under (a resume finds it again).
        // Nothing is created here: `build_options` / `build_agent_spec` make it, so a
        // refused open leaves nothing behind.
        let placed = ChatRequest {
            cwd: super::neutral_place::dir_for(session_id)
                .to_string_lossy()
                .into_owned(),
            ..request.clone()
        };
        let mut response = self
            .create_session_in_place(&placed, relay, session_id)
            .await?;
        response.execution_place = super::neutral_place::ExecutionPlace::Neutral;
        if placed.project_slug.is_none() {
            // Never silent: every graph stage is keyed on the project, and none was named.
            warn!(
                session_id = %response.session_id,
                execution_place = "neutral",
                graph_context = "none",
                "neutral session without project_slug: it gets no knowledge-graph context"
            );
            response.notices.push(
                "no cwd and no project_slug: this session is not tied to a project, so it gets no knowledge-graph context (notes, skills, personas, status). Pass project_slug to scope it."
                    .to_string(),
            );
        }
        Ok(response)
    }

    /// [`Self::create_session_relayed`] once `request.cwd` is settled (a project's, or
    /// the neutral directory of `session_id`).
    async fn create_session_in_place(
        &self,
        request: &ChatRequest,
        relay: Option<&super::relay::RelayedFrom>,
        session_id: Uuid,
    ) -> Result<CreateSessionResponse> {
        // Check max sessions
        {
            let sessions = self.active_sessions.read().await;
            if sessions.len() >= self.config.max_sessions {
                return Err(anyhow!(
                    "Maximum number of active sessions reached ({})",
                    self.config.max_sessions
                ));
            }
        }

        // Resolve scaffolding override: explicit field takes priority,
        // fallback to spawned_by JSON if present (for MCP callers)
        let scaffolding_override = request.scaffolding_override.or_else(|| {
            request
                .spawned_by
                .as_deref()
                .and_then(parse_spawned_by)
                .and_then(|ctx| ctx.scaffolding_level)
        });

        // Sessions created in all-projects (workspace) mode carry no
        // project_slug, and every graph stage (system prompt, skills,
        // personas, knowledge injection, status) is keyed on it — so from
        // mid-July 2026 no chat session got ANY graph context. The cwd is the
        // selected project's root: infer the project from it.
        let project_slug = match request.project_slug.clone() {
            Some(slug) => Some(slug),
            None => {
                let place = if super::neutral_place::is_neutral_path(&request.cwd) {
                    super::neutral_place::ExecutionPlace::Neutral
                } else {
                    super::neutral_place::ExecutionPlace::Project
                };
                let inferred = infer_session_project(
                    self.graph.as_ref(),
                    self.anchor_mode,
                    Some(session_id),
                    place,
                    &request.cwd,
                )
                .await;
                if let Some(ref slug) = inferred {
                    info!(session_id = %session_id, slug = %slug, cwd = %request.cwd, "Inferred session project from cwd");
                }
                inferred
            }
        };

        // Which provider serves this session (A16), decided from what is stored
        // (instances, the project's consent, roles, aliases) BEFORE anything is
        // spawned or persisted: a refusal costs nothing.
        let provider_choice = self
            .resolve_provider_choice_for(request, project_slug.as_deref(), Some(session_id))
            .await?;
        let model = match request.model.as_deref().filter(|m| !m.is_empty()) {
            Some(explicit) => explicit.to_string(),
            None => match provider_choice.model.as_deref() {
                Some(chosen) => chosen.to_string(),
                None if provider_choice.provider_id == super::provider::resolver::CLAUDE_CODE => {
                    self.resolve_model(None)
                }
                // Another provider never gets Claude's default model: its own
                // default, or a clear refusal.
                None => super::provider::store::instance(
                    self.graph.as_ref(),
                    &provider_choice.provider_id,
                )
                .await?
                .and_then(|i| i.default_model)
                .ok_or_else(|| {
                    anyhow::Error::new(nexus_claude::agent::ProviderError::invalid(
                        "this provider instance has no default model: name one",
                    ))
                })?,
            },
        };

        // Build system prompt — runner-spawned agents get a dedicated autonomous
        // execution prompt; conversational sessions get the generic PO prompt.
        let (system_prompt, _included_note_ids) =
            if let Some(ref runner_ctx) = request.runner_context {
                // Runner mode: use the runner system prompt (autonomous code execution)
                let prompt_ctx = runner_ctx.to_prompt_context();
                let runner_prompt = crate::runner::prompt::build_runner_system_prompt(&prompt_ctx);
                info!(
                    session_id = %session_id,
                    scaffolding = runner_ctx.scaffolding_level,
                    "Using runner system prompt (autonomous code execution mode)"
                );
                (runner_prompt, std::collections::HashSet::new())
            } else {
                // Conversational mode: use the generic PO system prompt with routing
                let typed_message = crate::refs::turn::visible_text(&request.message);
                let routing_message = if typed_message.is_empty() {
                    request.task_context.as_deref().unwrap_or("")
                } else {
                    &typed_message
                };
                self.build_system_prompt_anchored(
                    if super::neutral_place::is_neutral_path(&request.cwd) {
                        super::neutral_place::ExecutionPlace::Neutral
                    } else {
                        super::neutral_place::ExecutionPlace::Project
                    },
                    &request.cwd,
                    project_slug.as_deref(),
                    routing_message,
                    Some(&model),
                    &session_id.to_string(),
                    scaffolding_override,
                )
                .await
            };

        // Persist session in Neo4j
        // Resolve add_dirs from explicit request, workspace, or empty
        let resolved_add_dirs = self
            .resolve_add_dirs(
                &request.cwd,
                request.add_dirs.as_deref(),
                request.workspace_slug.as_deref(),
                request.project_slug.as_deref(),
            )
            .await;

        // What the session may do: fixed here, persisted on the node, kept by every resume.
        let access = super::provider::policy::SessionAccess::for_open(
            request.access,
            super::neutral_place::is_neutral_path(&request.cwd),
        );
        let session_node = ChatSessionNode {
            id: session_id,
            cli_session_id: None,
            project_slug: project_slug.clone(),
            workspace_slug: request.workspace_slug.clone(),
            cwd: request.cwd.clone(),
            title: None,
            model: model.clone(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            // The opening message is the first user message (persisted below as
            // `user_message`); every later one bumps the count.
            message_count: 1,
            total_cost_usd: None,
            // A relayed session keeps the memory conversation of the one it continues.
            conversation_id: relay.and_then(|r| r.conversation_id.clone()),
            preview: None,
            permission_mode: request.permission_mode.clone(),
            add_dirs: if resolved_add_dirs.is_empty() {
                None
            } else {
                Some(resolved_add_dirs.clone())
            },
            spawned_by: request.spawned_by.clone(),
            provider_id: Some(provider_choice.provider_id.clone()),
            // A move by the router names its target, yet nobody imposed it: `auto`.
            routed_by: Some(if super::relay::moved_by_auto(relay) {
                super::provider::resolver::RoutedBy::Auto
                    .as_str()
                    .to_string()
            } else {
                provider_choice.routed_by.as_str().to_string()
            }),
            routing_mode: request.routing_mode.map(|m| m.as_str().to_owned()),
            routing_pool: request
                .routing_pool
                .as_ref()
                .and_then(|pool| serde_json::to_string(pool).ok()),
            capabilities: None,
            resume_token: None,
            execution_place: if super::neutral_place::is_neutral_path(&request.cwd) {
                super::neutral_place::ExecutionPlace::Neutral
            } else {
                super::neutral_place::ExecutionPlace::Project
            },
            access,
        };
        self.graph
            .create_chat_session(&session_node)
            .await
            .context("Failed to persist chat session")?;
        self.remember_routing_decision(&session_id.to_string(), provider_choice.decision_id)
            .await;

        // The policy rule that applied, and what a `shadow` policy would have
        // chosen, are kept for the execution record (A22): the runner reads
        // them back. A write that fails loses a note, never a session.
        if provider_choice.route_rule.is_some()
            || provider_choice.shadow.is_some()
            || provider_choice.fallback_reason.is_some()
        {
            let note = serde_json::json!({
                "route_rule": provider_choice.route_rule,
                "fallback_reason": provider_choice.fallback_reason,
                "shadow_provider": provider_choice.shadow.as_ref().map(|s| &s.0),
                "shadow_model": provider_choice.shadow.as_ref().and_then(|s| s.1.as_ref()),
            });
            if let Err(e) = self
                .graph
                .put_llm_setting(&format!("routing:{session_id}"), "note", &note.to_string())
                .await
            {
                warn!(session_id = %session_id, error = %e, "Failed to record the routing note (non-fatal)");
            }
        }

        // If this session was spawned by another, create the SPAWNED_BY relation in Neo4j
        // and extract protocol FSM context (run_id + state) for trajectory tagging.
        let spawned_ctx = request.spawned_by.as_deref().and_then(parse_spawned_by);
        let (spawned_protocol_run_id, spawned_protocol_state) = match &spawned_ctx {
            Some(ctx) => {
                // Create SPAWNED_BY relation in graph if parent_session_id is present
                if let Some(ref parent_id) = ctx.parent_session_id {
                    if let Err(e) = self
                        .graph
                        .create_spawned_by_relation(
                            &session_id.to_string(),
                            parent_id,
                            &ctx.spawn_type,
                            ctx.run_id,
                            ctx.task_id,
                        )
                        .await
                    {
                        warn!("Failed to create SPAWNED_BY relation: {e}");
                    }
                }
                (ctx.protocol_run_id, ctx.protocol_state.clone())
            }
            None => (None, None),
        };

        // A plan-runner session has no parent session, so the SPAWNED_BY
        // relation above is never created for it. Link it to its PlanRun here,
        // for EVERY caller (task, retry, wave, resumed run): the cockpit's
        // session -> thread attachment reads this relation (chat::attachment).
        if let Some(spawn) = request
            .spawned_by
            .as_deref()
            .and_then(super::attachment::parse_plan_run_spawn)
        {
            if let Some(run_id) = spawn.run_id {
                match self
                    .graph
                    .link_session_to_run(
                        &session_id.to_string(),
                        run_id,
                        spawn.plan_id,
                        spawn.task_id,
                    )
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => warn!(
                        session_id = %session_id, run_id = %run_id,
                        "Runner session not linked to its run (PlanRun not found)"
                    ),
                    Err(e) => warn!("Failed to link runner session to its run: {e}"),
                }
            }
        }

        // The provider-neutral path takes over here: the session is persisted
        // and the system prompt built; what follows is the Claude CLI engine.
        // Hybrid routing: Claude Code stays on the historical engine (hooks,
        // queue, retry, compaction, NATS, images) unless the operator FORCES it
        // onto the agent engine; every other provider is served by the agent engine.
        if self.engine_is_agent(&provider_choice.provider_id) {
            return self
                .open_agent_session(AgentOpen {
                    request,
                    session_id,
                    provider_id: &provider_choice.provider_id,
                    model: &model,
                    system_prompt: &system_prompt,
                    add_dirs: &resolved_add_dirs,
                    project_slug: project_slug.as_deref(),
                    relay,
                    access,
                })
                .await;
        }

        // The Claude CLI always switches model live; the opening message was turn 0.
        if let Some(router) = self
            .register_turn_router(
                &session_id.to_string(),
                &provider_choice.provider_id,
                &model,
                project_slug.as_deref(),
                OpeningTurn {
                    explicit_model: request.model.is_some(),
                    provider_imposed: request.provider.is_some(),
                    moved_in: super::relay::moved_by_auto(relay),
                    routing_mode: request.routing_mode,
                    routing_pool: request.routing_pool.clone(),
                    permission_mode: request.permission_mode.as_deref(),
                    message: &crate::refs::turn::visible_text(&request.message),
                    next_turn: 1,
                },
            )
            .await
        {
            router.set_model_live(true);
        }

        // Create broadcast channel early so CompactionNotifier can use the sender
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);

        // Create work_log early so CompactionNotifier can reference it
        let work_log = Arc::new(Mutex::new(SessionWorkLog::default()));

        // Build session hooks: PreCompact (compaction notifier) + PreToolUse (skill activation)
        let session_hooks = {
            let context_source = match &spawned_ctx {
                Some(ctx) if ctx.task_id.is_some() => {
                    CompactionContextSource::Task(ctx.task_id.unwrap())
                }
                _ => match project_slug.as_deref() {
                    Some(slug) => CompactionContextSource::Session(slug.to_string()),
                    None => CompactionContextSource::None,
                },
            };
            self.graph_hook_table(GraphHookInput {
                session_id: session_id.to_string(),
                context_source,
                work_log: work_log.clone(),
                // PreToolUse/PostToolUse are skipped for runner sessions — they inject
                // ~1000 chars of context per tool call (persona, notes, redirect suggestions),
                // which accelerates compaction and wastes tokens. Runner agents already have
                // full task context via the prompt.
                tool_knowledge: request.runner_context.is_none(),
                announce: Some(events_tx.clone()),
            })
        };

        // Build options and create InteractiveClient
        let sid_str = session_id.to_string();
        let options = self
            .build_options_with_access(
                &request.cwd,
                &model,
                &system_prompt,
                None,
                request.permission_mode.as_deref(),
                Some(session_hooks),
                &resolved_add_dirs,
                request.user_claims.as_ref(),
                Some(&sid_str),
                access,
            )
            .await;
        let mut client = InteractiveClient::new(options).map_err(|e| {
            super::provider::errors::sdk_open_error("Failed to create InteractiveClient", e)
        })?;

        client.connect().await.map_err(|e| {
            super::provider::errors::sdk_open_error("Failed to connect InteractiveClient", e)
        })?;

        // Initialize hooks with the CLI (sends PreCompact, etc. registrations).
        // Must be called AFTER connect() and BEFORE take_sdk_control_receiver().
        // Graceful: warn on failure but don't abort the session.
        if let Err(e) = client.initialize_hooks().await {
            warn!(
                session_id = %session_id,
                "Failed to initialize hooks with CLI (non-fatal): {}",
                e
            );
        }

        // Create ConversationMemoryManager for message recording
        let memory_manager = if let Some(ref mem_config) = self.memory_config {
            // A relayed session records into the conversation it continues.
            let mm = match relay.and_then(|r| r.conversation_id.clone()) {
                Some(kept) => {
                    ConversationMemoryManager::new(mem_config.clone()).with_conversation_id(kept)
                }
                None => ConversationMemoryManager::new(mem_config.clone()),
            };
            let conversation_id = mm.conversation_id().to_string();
            debug!(
                "Created ConversationMemoryManager for session {} with conversation_id {}",
                session_id, conversation_id
            );

            // Persist conversation_id in Neo4j
            let _ = self
                .graph
                .update_chat_session(
                    session_id,
                    None,
                    None,
                    None,
                    None,
                    Some(conversation_id),
                    None,
                )
                .await;

            Some(Arc::new(Mutex::new(mm)))
        } else {
            None
        };

        // Clone the stdin sender BEFORE wrapping client in Arc<Mutex<>>.
        // This allows send_permission_response to write control responses
        // directly to the CLI subprocess without taking the client lock
        // (which is held by stream_response during streaming → deadlock).
        let stdin_tx = client.clone_stdin_sender().await;

        // Take the SDK control receiver ONCE at session creation and hand it to
        // the session-lifetime control pump. Hook callbacks are answered there
        // for as long as the session lives — background subagents outlive the
        // turn that spawned them — and everything else (`can_use_tool`
        // permission requests…) is forwarded to the receiver `stream_response`
        // reads. It must be taken before wrapping the client in Arc<Mutex<>>.
        // See `control_pump.rs` for the incident behind this.
        let sdk_control_rx = client.take_sdk_control_receiver().await.map(|raw_rx| {
            super::control_pump::spawn(
                session_id.to_string(),
                raw_rx,
                client.hook_callbacks(),
                stdin_tx.clone(),
            )
        });
        let sdk_control_rx = Arc::new(tokio::sync::Mutex::new(sdk_control_rx));

        // Capture the CLI subprocess PID so descendant-PID SIGINT
        // cascade in `interrupt()` and `cancel_running_tools()` works.
        // (Plan 28e9afe3 — without this, kill_descendants always sees
        // None and the per-tool Stop button is a no-op.)
        let child_pid: Option<u32> = client.child_pid().await;

        info!(
            session_id = %session_id,
            has_stdin_tx = stdin_tx.is_some(),
            ?child_pid,
            "Created chat session with model {}",
            model
        );

        let client = Arc::new(Mutex::new(client));

        // Initialize next_seq (new session = start at 1)
        let next_seq = Arc::new(AtomicI64::new(1));
        let pending_messages = Arc::new(Mutex::new(VecDeque::<PendingMessage>::new()));
        let is_streaming = Arc::new(AtomicBool::new(false));
        let streaming_text = Arc::new(Mutex::new(String::new()));
        let streaming_events = Arc::new(Mutex::new(Vec::new()));

        // Register active session — cancel old NATS listeners if session key already exists
        let nats_cancel = CancellationToken::new();
        let interrupt_token = CancellationToken::new();
        // Runner sessions always get auto_continue=true; interactive sessions use config default.
        let is_runner_session = request.runner_context.is_some();
        let auto_continue = Arc::new(AtomicBool::new(if is_runner_session {
            true
        } else {
            self.config.auto_continue
        }));
        let auto_continue_count = Arc::new(AtomicU32::new(0));
        // Runner sessions: limit to 5 auto-continues to prevent infinite loops.
        // Interactive sessions: unlimited (0) — user can always interrupt manually.
        let max_auto_continues: u32 = if is_runner_session { 5 } else { 0 };
        // OOB-trigger cap (T7 of plan 9a1684b2): conservative for runner
        // sessions, generous for interactive (user can always interrupt).
        let oob_trigger_cap: u32 = if is_runner_session {
            OOB_TRIGGER_CAP_RUNNER
        } else {
            OOB_TRIGGER_CAP_INTERACTIVE
        };
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            // Cancel stale NATS listeners from a previous session with the same ID
            if let Some(old_session) = sessions.get(&session_id.to_string()) {
                info!(
                    session_id = %session_id,
                    "Cancelling stale NATS listeners for existing session (create_session replacing)"
                );
                old_session.nats_cancel.cancel();
            }
            let interrupt_flag = Arc::new(AtomicBool::new(false));
            sessions.insert(
                session_id.to_string(),
                ActiveSession {
                    anchor: self.anchor_session(),
                    events_tx: events_tx.clone(),
                    last_activity: Instant::now(),
                    cli_session_id: None,
                    client: client.clone(),
                    interrupt_flag: interrupt_flag.clone(),
                    memory_manager: memory_manager.clone(),
                    next_seq: next_seq.clone(),
                    pending_messages: pending_messages.clone(),
                    is_streaming: is_streaming.clone(),
                    streaming_text: streaming_text.clone(),
                    streaming_events: streaming_events.clone(),
                    permission_mode: request.permission_mode.clone(),
                    model: Some(model.clone()),
                    sdk_control_rx: sdk_control_rx.clone(),
                    stdin_tx,
                    child_pid,
                    nats_cancel: nats_cancel.clone(),
                    interrupt_token: interrupt_token.clone(),
                    pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                        std::collections::HashMap::new(),
                    )),
                    auto_continue: auto_continue.clone(),
                    auto_continue_count: auto_continue_count.clone(),
                    max_auto_continues,
                    rfc_accumulator: Arc::new(Mutex::new(
                        super::observation_detector::RfcAccumulator::new(),
                    )),
                    protocol_run_id: spawned_protocol_run_id,
                    protocol_state: spawned_protocol_state.clone(),
                    reasoning_path_tracker: super::feedback::ReasoningPathTracker::new(),
                    objective_tracking: true,
                    objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
                    objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
                    work_log: work_log.clone(),
                    oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
                    oob_trigger_cap,
                    oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
                    oob_capped_warned: Arc::new(AtomicBool::new(false)),
                    cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
                    cancel_tools_cap: CANCEL_TOOLS_CAP,
                    cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
                    active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
                    cli_background_tasks: Arc::new(AtomicUsize::new(0)),
                    cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
                    cancel_task_cap: CANCEL_TASK_CAP,
                    cancel_task_window: Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
                },
            );
            interrupt_flag
        };
        self.notify_attention(&session_id.to_string(), AttentionReason::SessionActive);

        // Spawn NATS interrupt listener for cross-instance interrupt support
        self.spawn_nats_interrupt_listener(
            &session_id.to_string(),
            interrupt_flag.clone(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS snapshot responder for cross-instance mid-stream join
        self.spawn_nats_snapshot_responder(
            &session_id.to_string(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS RPC send listener for cross-instance message routing
        self.spawn_nats_rpc_listener(
            &session_id.to_string(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS cancel_tools listener (T2 of plan 28e9afe3) for
        // cross-instance routing of user-initiated tool cancellation.
        // Distinct subject from interrupt — different semantics
        // (cf decision d2bf0e7b).
        self.spawn_nats_cancel_tools_listener(
            &session_id.to_string(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn the per-session background-tasks poller (T12 of plan
        // 754a1379). Wakes up every BACKGROUND_TASKS_POLL_INTERVAL_SECS
        // seconds and physically purges any entries whose
        // `pending_removal_at` aged past the grace period — absorbs
        // in-flight `BackgroundOutput` ticks that arrive between the
        // cancel SIGINT and the subprocess actually dying. Tied to
        // `nats_cancel` so it terminates cleanly with the session.
        Self::spawn_background_tasks_poller(
            session_id.to_string(),
            self.active_sessions.clone(),
            events_tx.clone(),
            self.nats.clone(),
            nats_cancel.clone(),
        );

        // Spawn the permanent out-of-band SDK message listener (T4+T5 of
        // plan 9a1684b2). Captures Messages emitted by the CLI subprocess
        // between turns (background tool notifications, etc.) and
        // surfaces them as ChatEvent::BackgroundOutput. When a turn is
        // not active and an OOB event arrives, the listener also pushes
        // a PendingMessage::BackgroundOutput and tries to claim
        // is_streaming via compare_exchange to spawn a fresh
        // stream_response — letting the LLM react autonomously without
        // waiting for the next user turn.
        super::oob_listener::spawn_oob_listener(
            session_id.to_string(),
            client.clone(),
            events_tx.clone(),
            is_streaming.clone(),
            next_seq.clone(),
            nats_cancel,
            super::oob_listener::OobListenerDeps {
                graph: self.graph.clone(),
                active_sessions: self.active_sessions.clone(),
                context_injector: self.context_injector.clone(),
                event_emitter: self.event_emitter.clone(),
                retry_config: self.config.retry.clone(),
                enrichment_pipeline: self.enrichment_pipeline.clone(),
                search: self.search.clone(),
                documents: self.document_store.clone(),
                nats: self.nats.clone(),
            },
        );

        // A relayed conversation says so first: what the model is given in front
        // of the message is stated on the thread, never slipped in silently.
        if let Some(relay) = relay {
            let relayed = relay.event(&session_id.to_string(), &provider_choice.provider_id);
            let record = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: next_seq.fetch_add(1, Ordering::SeqCst),
                event_type: relayed.event_type().to_string(),
                data: serde_json::to_string(&relayed).unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            let _ = self.graph.store_chat_events(session_id, vec![record]).await;
            let _ = events_tx.send(relayed);
        }

        // Persist the initial user_message event
        let user_event = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: "user_message".to_string(),
            data: serde_json::to_string(&ChatEvent::UserMessage {
                content: request.message.clone(),
            })
            .unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        let _ = self
            .graph
            .store_chat_events(session_id, vec![user_event])
            .await;

        // Emit user_message on local broadcast + NATS (so all clients see it)
        let user_msg_event = ChatEvent::UserMessage {
            content: request.message.clone(),
        };
        let _ = events_tx.send(user_msg_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(&session_id.to_string(), user_msg_event);
        }
        self.notify_attention(&session_id.to_string(), AttentionReason::UserMessage);

        // Emit CRUD event so other instances (via NATS) know a session was created
        if let Some(ref emitter) = self.event_emitter {
            emitter.emit_created(
                crate::events::EntityType::ChatSession,
                &session_id.to_string(),
                serde_json::json!({
                    "project_slug": request.project_slug,
                    "execution_place": if super::neutral_place::is_neutral_path(&request.cwd) { "neutral" } else { "project" },
                    "cwd": if super::neutral_place::is_neutral_path(&request.cwd) { serde_json::Value::Null } else { serde_json::json!(request.cwd) },
                    "model": model,
                }),
                None,
            );
        }

        // Auto-generate title and preview from the first user message — what the
        // user typed, never the `<po-refs>`/`<po-attachments>` blocks around it.
        {
            let typed = crate::refs::turn::visible_text(&request.message);
            let msg = &typed;
            let title = if msg.chars().count() > 80 {
                let truncated: String = msg.chars().take(77).collect();
                format!("{}...", truncated.trim_end())
            } else {
                msg.to_string()
            };
            let preview = if msg.chars().count() > 200 {
                let truncated: String = msg.chars().take(197).collect();
                format!("{}...", truncated.trim_end())
            } else {
                msg.to_string()
            };
            let _ = self
                .graph
                .update_chat_session(
                    session_id,
                    None,
                    Some(title),
                    None,
                    None,
                    None,
                    Some(preview),
                )
                .await;
        }

        // Send the initial message and start streaming in a background task
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let message = super::relay::prefixed(relay.and_then(|r| r.text()), &request.message);
        let events_tx_clone = events_tx.clone();
        let injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();
        let retry_config = self.config.retry.clone();
        let enrichment_pipeline = self.enrichment_pipeline.clone();
        let search = self.search.clone();
        let documents = self.document_store.clone();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx_clone,
                message,
                session_id_str.clone(),
                graph,
                active_sessions,
                interrupt_flag,
                memory_manager,
                injector,
                next_seq,
                pending_messages,
                is_streaming,
                streaming_text,
                streaming_events,
                event_emitter,
                nats,
                sdk_control_rx,
                auto_continue,
                retry_config,
                enrichment_pipeline,
                search,
                documents,
            )
            .await;
        });

        Ok(CreateSessionResponse {
            session_id: session_id.to_string(),
            stream_url: format!("/ws/chat/{}", session_id),
            execution_place: Default::default(),
            access,
            notices: Vec::new(),
        })
    }

    /// Track the start of a `Monitor` / `Bash run_in_background` invocation
    /// in `ActiveSession::active_background_tasks`, then broadcast the
    /// updated snapshot via `ChatEvent::ActiveTasksUpdate`.
    ///
    /// Plan 754a1379, T3 (INSERT side of lifecycle hooks). Called from
    /// `stream_response` whenever a `ChatEvent::ToolUse` is observed, so
    /// the entry materialises as soon as the SDK announces a tool call.
    ///
    /// ## Idempotence (two-pass call site)
    ///
    /// `stream_response` may call this helper twice for a single tool_use:
    /// first when `ContentBlockStart` arrives with empty input, then when
    /// `AssistantMessage` arrives with the full input. We use
    /// `HashMap::entry(...).or_insert_with(...)` so the first call inserts
    /// the placeholder entry and the second call only refreshes the
    /// `description` field once it becomes available. Both passes also
    /// emit a fresh `ActiveTasksUpdate` so the frontend reflects every
    /// observable state change.
    ///
    /// ## Filtering
    ///
    /// - `tool_name == "Monitor"` → always tracked. Input is irrelevant
    ///   to the kind decision.
    /// - `tool_name == "Bash"` AND `input.run_in_background == true` →
    ///   tracked as `BashBackground`. Synchronous Bash calls are ignored
    ///   (their lifecycle is bounded by the turn).
    /// - All other tools → ignored, returns `false` without touching
    ///   the map or broadcasting.
    ///
    /// Note that Bash's `run_in_background` flag is only present on the
    /// **second** pass (full input), so the first pass for a Bash call
    /// returns `false` and the second pass performs the actual insert.
    /// For Monitor, both passes succeed (the second one just updates
    /// the description in place).
    ///
    /// ## Returns
    ///
    /// `true` if the tool is tracked (newly inserted or refreshed),
    /// `false` if the tool isn't a tracked kind or the session is gone.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn track_background_task_start(
        session_id: &str,
        active_sessions: &Arc<RwLock<HashMap<String, ActiveSession>>>,
        events_tx: &broadcast::Sender<ChatEvent>,
        nats: &Option<Arc<crate::events::NatsEmitter>>,
        tool_use_id: &str,
        tool_name: &str,
        input: &serde_json::Value,
        parent_tool_use_id: Option<&str>,
    ) -> bool {
        // Decide whether the tool kind is something we track.
        let kind = match tool_name {
            "Monitor" => Some(BackgroundTaskKind::Monitor),
            "Bash"
                if input
                    .get("run_in_background")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false) =>
            {
                Some(BackgroundTaskKind::BashBackground)
            }
            _ => None,
        };
        let Some(kind) = kind else {
            return false;
        };

        // Best-effort description (input field, falls back to a placeholder
        // before the second pass arrives with full input).
        let description = input
            .get("description")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(no description)")
            .to_string();

        // Extract the child_pid and the Arc<Mutex<HashMap>> for tasks under
        // a brief read lock so we can release it before the (potentially
        // slow) `get_descendant_pids` pgrep call below.
        let (tasks_arc, child_pid_for_before) = {
            let sessions = active_sessions.read().await;
            let Some(active) = sessions.get(session_id) else {
                debug!(
                    session_id = %session_id,
                    "track_background_task_start: session not in active_sessions, dropping"
                );
                return false;
            };
            (active.active_background_tasks.clone(), active.child_pid)
        };

        // Capture the descendant snapshot BEFORE the subprocess forks.
        // This must run lock-free because `pgrep -P` may take 10-50 ms.
        // Plan fc35b25e (T2): the async claim spawned below will compare
        // this snapshot to one taken ~1 s later to discover the new PID.
        let before_pids: Vec<u32> = child_pid_for_before
            .map(Self::get_descendant_pids)
            .unwrap_or_default();

        // Now perform the upsert under the tasks lock and capture the
        // fresh snapshot for the broadcast. We use `match entry` instead of
        // `or_insert_with` so we can detect the first-insert case and gate
        // the async PID claim on it (the helper is called twice per ToolUse
        // — first with empty input, then with full input — and we only
        // want to spawn one claim task per tool_use_id).
        let (snapshot, was_vacant) = {
            let mut tasks = tasks_arc.lock().await;
            let was_vacant = !tasks.contains_key(tool_use_id);

            let entry = tasks.entry(tool_use_id.to_string()).or_insert_with(|| {
                let now = chrono::Utc::now();
                BackgroundTaskInfo {
                    id: tool_use_id.to_string(),
                    kind,
                    description: description.clone(),
                    started_at: now,
                    last_seen_at: now,
                    pid: None,
                    parent_tool_use_id: parent_tool_use_id.map(String::from),
                    pending_removal_at: None,
                }
            });

            // On the second pass (full input), refresh the description
            // if we now have a real one.
            if description != "(no description)" && !description.is_empty() {
                entry.description = description;
            }
            // If the entry was previously marked for removal but the SDK
            // surprises us with a fresh ToolUse on the same id, cancel
            // the pending removal — the task is alive after all.
            entry.pending_removal_at = None;

            let snap = tasks.values().cloned().collect::<Vec<BackgroundTaskInfo>>();
            (snap, was_vacant)
        };

        let event = ChatEvent::ActiveTasksUpdate { tasks: snapshot };
        let _ = events_tx.send(event.clone());
        if let Some(ref nats) = nats {
            nats.publish_chat_event(session_id, event);
        }

        info!(
            session_id = %session_id,
            tool_use_id = %tool_use_id,
            kind = ?kind,
            "Tracked background task start (plan 754a1379, T3)"
        );

        // Plan fc35b25e (T2): spawn the async PID claim only on first
        // insert. The second pass (description refresh) reuses whatever
        // PID the first claim populated.
        if was_vacant && child_pid_for_before.is_some() {
            let session_id_owned = session_id.to_string();
            let tool_use_id_owned = tool_use_id.to_string();
            let active_sessions_clone = active_sessions.clone();
            let events_tx_clone = events_tx.clone();
            let nats_clone = nats.clone();
            tokio::spawn(async move {
                Self::async_pid_claim(
                    session_id_owned,
                    active_sessions_clone,
                    events_tx_clone,
                    nats_clone,
                    tool_use_id_owned,
                    before_pids,
                )
                .await;
            });
        }

        true
    }

    /// Resolve the current `child_pid` of a session. Returns `None` if the
    /// session has been removed (e.g., dropped between callsite and the
    /// async PID claim) or if the session has no CLI subprocess yet.
    ///
    /// Plan fc35b25e (T2): companion to `async_pid_claim`. Re-resolves the
    /// PID at sleep wake-up so a `resume_session` that races with the
    /// claim window doesn't silently associate the new tool with an
    /// obsolete CLI.
    async fn lookup_child_pid(
        session_id: &str,
        active_sessions: &Arc<RwLock<HashMap<String, ActiveSession>>>,
    ) -> Option<u32> {
        let sessions = active_sessions.read().await;
        sessions.get(session_id).and_then(|s| s.child_pid)
    }

    /// Async PID claim for a freshly-tracked background task. Sleeps 1 s
    /// to let the subprocess fork, then snapshots descendants again,
    /// computes the diff against `before_pids`, and assigns the newest
    /// new PID to the task entry. Broadcasts a fresh `ActiveTasksUpdate`
    /// so the frontend can surface the PID (useful for debugging).
    ///
    /// Race-safety:
    /// - No locks are held during the `tokio::time::sleep`.
    /// - If the entry was removed during the sleep (rapid cancel) the
    ///   claim no-ops silently.
    /// - If `pid_discovery_diff` returns empty (subprocess crashed before
    ///   the snapshot, or two ToolUses raced and the first claim won
    ///   both candidates), we log a warning and leave `pid: None`. The
    ///   `cancel_task` fallback (T4) handles this case.
    ///
    /// Plan fc35b25e (T2). Tested in `tests::test_pid_claim_*`.
    async fn async_pid_claim(
        session_id: String,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        events_tx: broadcast::Sender<ChatEvent>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
        tool_use_id: String,
        before_pids: Vec<u32>,
    ) {
        // Wait for the subprocess to fork and be visible via pgrep.
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Re-resolve child_pid (a `resume_session` between the callsite
        // and now would have replaced the CLI).
        let Some(child_pid) = Self::lookup_child_pid(&session_id, &active_sessions).await else {
            debug!(
                session_id = %session_id,
                tool_use_id = %tool_use_id,
                "async_pid_claim: session gone, skipping"
            );
            return;
        };

        let after_pids = Self::get_descendant_pids(child_pid);
        let candidates = Self::pid_discovery_diff(&before_pids, &after_pids);
        let Some(claimed) = candidates.first().copied() else {
            warn!(
                session_id = %session_id,
                tool_use_id = %tool_use_id,
                before_count = before_pids.len(),
                after_count = after_pids.len(),
                "async_pid_claim: no new PID in diff (subprocess crashed or claim raced)"
            );
            return;
        };

        // Update the task's pid and capture a fresh snapshot.
        let snapshot = {
            let sessions = active_sessions.read().await;
            let Some(active) = sessions.get(&session_id) else {
                return;
            };
            let mut tasks = active.active_background_tasks.lock().await;
            let Some(task) = tasks.get_mut(&tool_use_id) else {
                // Entry was cancelled/removed during the sleep — no-op.
                debug!(
                    session_id = %session_id,
                    tool_use_id = %tool_use_id,
                    "async_pid_claim: task removed before claim, dropping"
                );
                return;
            };
            task.pid = Some(claimed);
            tasks.values().cloned().collect::<Vec<BackgroundTaskInfo>>()
        };

        info!(
            session_id = %session_id,
            tool_use_id = %tool_use_id,
            pid = claimed,
            "async_pid_claim: claimed PID for background task (plan fc35b25e, T2)"
        );

        let event = ChatEvent::ActiveTasksUpdate { tasks: snapshot };
        let _ = events_tx.send(event.clone());
        if let Some(ref nats) = nats {
            nats.publish_chat_event(&session_id, event);
        }
    }

    /// Touch the activity timestamp on a tracked background task,
    /// AND lazy-recover orphan correlation_ids in the same call.
    ///
    /// Called by the OOB listener for every `BackgroundOutput` tick.
    /// One of three outcomes:
    ///
    /// 1. Already-tracked id (live or `pending_removal_at`): updates
    ///    `last_seen_at = Utc::now()`. No broadcast (the field is on
    ///    the wire format but tick-by-tick refreshes are noise; the
    ///    next state-change broadcast carries the latest value).
    /// 2. Orphan correlation_id + recoverable source ("Monitor",
    ///    "BashOutput"): inserts a recovered entry, broadcasts a fresh
    ///    `ActiveTasksUpdate` (T13 lazy recovery).
    /// 3. Otherwise (None / unknown source): no-op.
    ///
    /// ## Why lazy instead of `pgrep` rebuild at create_session
    ///
    /// See decision `f2151626` on the T13 task. Short version:
    /// pgrep matching `tool_use_id → root PID` is imprecise (cmdline
    /// doesn't carry the id), false-positive prone (random
    /// CLI-internal helpers would also be listed), and platform-fragile
    /// (/proc on Linux vs ps on macOS). Lazy recovery via
    /// `BackgroundOutput` is automatically correct: only subprocesses
    /// that are still emitting events reappear, and dead ones are
    /// never resurrected.
    ///
    /// Plan 754a1379 (T13).
    ///
    /// ## Behaviour
    ///
    /// - `correlation_id == None` → no-op, returns `false`.
    /// - `correlation_id` already in the map → no-op (live or
    ///   `pending_removal_at`), returns `false`.
    /// - `correlation_id` orphan + `source == "Monitor"` →
    ///   insert as `Monitor`, broadcast `ActiveTasksUpdate`,
    ///   returns `true`.
    /// - `correlation_id` orphan + `source == "BashOutput"` →
    ///   insert as `BashBackground`, broadcast, returns `true`.
    /// - Any other source ("system", unknown) → returns `false`. We
    ///   don't speculatively recover unknown kinds — the user can
    ///   still cancel via the global Stop.
    ///
    /// Recovered entries carry `description = "(recovered after
    /// restart)"`, `parent_tool_use_id = Some(correlation_id)`,
    /// `pid = None`, `pending_removal_at = None`. The id field is
    /// the original tool_use_id (= correlation_id), NOT a synthetic
    /// `recovered-{pid}`.
    pub(crate) async fn track_background_task_recovery_if_orphan(
        session_id: &str,
        active_sessions: &Arc<RwLock<HashMap<String, ActiveSession>>>,
        events_tx: &broadcast::Sender<ChatEvent>,
        nats: &Option<Arc<crate::events::NatsEmitter>>,
        correlation_id: Option<&str>,
        source: &str,
    ) -> bool {
        let Some(correlation_id) = correlation_id else {
            return false;
        };

        let kind = match source {
            "Monitor" => BackgroundTaskKind::Monitor,
            "BashOutput" => BackgroundTaskKind::BashBackground,
            _ => return false,
        };

        let snapshot = {
            let tasks_arc = {
                let sessions = active_sessions.read().await;
                let Some(active) = sessions.get(session_id) else {
                    return false;
                };
                active.active_background_tasks.clone()
            };
            let mut tasks = tasks_arc.lock().await;
            if let Some(entry) = tasks.get_mut(correlation_id) {
                // Path 1 — already tracked: refresh activity timestamp
                // for the death detector. No broadcast (a tick-by-tick
                // ActiveTasksUpdate would saturate the WebSocket on
                // chatty Monitors).
                entry.last_seen_at = chrono::Utc::now();
                return false;
            }
            let now = chrono::Utc::now();
            tasks.insert(
                correlation_id.to_string(),
                BackgroundTaskInfo {
                    id: correlation_id.to_string(),
                    kind,
                    description: "(recovered after restart)".to_string(),
                    started_at: now,
                    last_seen_at: now,
                    pid: None,
                    parent_tool_use_id: Some(correlation_id.to_string()),
                    pending_removal_at: None,
                },
            );
            info!(
                session_id = %session_id,
                correlation_id = %correlation_id,
                kind = ?kind,
                source = %source,
                "Lazy-recovered orphan BackgroundOutput as new BackgroundTaskInfo (T13)"
            );
            tasks.values().cloned().collect::<Vec<BackgroundTaskInfo>>()
        };

        let event = ChatEvent::ActiveTasksUpdate { tasks: snapshot };
        let _ = events_tx.send(event.clone());
        if let Some(ref nats) = nats {
            nats.publish_chat_event(session_id, event);
        }
        true
    }

    /// Spawn the per-session background-tasks poller. The poller wakes
    /// up every `BACKGROUND_TASKS_POLL_INTERVAL_SECS` seconds and runs
    /// `tick_purge_background_tasks`, which physically removes any
    /// entries whose `pending_removal_at` aged past
    /// `BACKGROUND_TASK_PURGE_GRACE_SECS`.
    ///
    /// Plan 754a1379 (T12 — race-safe purge). The grace period absorbs
    /// in-flight `BackgroundOutput` ticks that arrive between the
    /// cancel SIGINT and the subprocess actually dying.
    ///
    /// The poller terminates when `cancel_token` is fired — typically
    /// `nats_cancel` of the owning `ActiveSession`, which is also the
    /// cancel token for the OOB listener. Tying both to the same token
    /// guarantees a clean teardown on session end (or `resume_session`,
    /// which cancels the previous session's tokens before spawning new
    /// ones).
    pub(crate) fn spawn_background_tasks_poller(
        session_id: String,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        events_tx: broadcast::Sender<ChatEvent>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
        cancel_token: CancellationToken,
    ) {
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(Duration::from_secs(BACKGROUND_TASKS_POLL_INTERVAL_SECS));
            // Skip the immediate first tick so we don't run a purge
            // before any state has even been mutated.
            interval.tick().await;
            let grace = Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS);

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        debug!(
                            session_id = %session_id,
                            "background_tasks_poller: cancelled, exiting"
                        );
                        return;
                    }
                    _ = interval.tick() => {
                        Self::tick_purge_background_tasks(
                            &session_id,
                            &active_sessions,
                            &events_tx,
                            &nats,
                            grace,
                        )
                        .await;
                    }
                }
            }
        });
    }

    /// One poller pass. Two-phase:
    ///
    /// 1. **Idle-death detection** — entries whose `last_seen_at` is
    ///    older than `idle_death` AND have no `pending_removal_at` get
    ///    marked `pending_removal_at = Some(now)` (T3 REMOVE side).
    /// 2. **Grace-period purge** — entries whose `pending_removal_at`
    ///    is older than `grace` are physically removed (T12).
    ///
    /// Broadcasts a single fresh `ActiveTasksUpdate` if **either**
    /// phase mutated the map. No-op (no broadcast) on a quiet tick.
    /// Public-crate so tests can drive it without waiting for the
    /// real interval.
    pub(crate) async fn tick_purge_background_tasks(
        session_id: &str,
        active_sessions: &Arc<RwLock<HashMap<String, ActiveSession>>>,
        events_tx: &broadcast::Sender<ChatEvent>,
        nats: &Option<Arc<crate::events::NatsEmitter>>,
        grace: Duration,
    ) {
        let (tasks_arc, cli_reports_live_tasks) = {
            let sessions = active_sessions.read().await;
            match sessions.get(session_id) {
                Some(active) => (
                    active.active_background_tasks.clone(),
                    active.cli_background_tasks.load(Ordering::Relaxed) > 0,
                ),
                None => return,
            }
        };

        let now_inst = std::time::Instant::now();
        let now_chrono = chrono::Utc::now();
        let idle_death = chrono::Duration::seconds(BACKGROUND_TASK_IDLE_DEATH_SECS as i64);

        let snapshot = {
            let mut tasks = tasks_arc.lock().await;

            // Phase 1: idle-death detection. Mark entries that haven't
            // emitted in too long AND are not known to be running: silence
            // alone is not death (see `background_task_is_idle_dead`).
            let mut newly_marked = 0usize;
            for info in tasks.values_mut() {
                if info.pending_removal_at.is_some() {
                    continue;
                }
                if background_task_is_idle_dead(
                    now_chrono - info.last_seen_at,
                    idle_death,
                    process_alive(info.pid),
                    cli_reports_live_tasks,
                ) {
                    info.pending_removal_at = Some(now_inst);
                    newly_marked += 1;
                }
            }

            // Phase 2: physical purge of grace-expired entries.
            let before = tasks.len();
            tasks.retain(|_id, info| {
                !matches!(
                    info.pending_removal_at,
                    Some(at) if now_inst.saturating_duration_since(at) >= grace
                )
            });
            let purged = before - tasks.len();

            if newly_marked == 0 && purged == 0 {
                return;
            }
            info!(
                session_id = %session_id,
                idle_marked = newly_marked,
                purged = purged,
                remaining = tasks.len(),
                "background_tasks_poller: idle-marked + purged (T3 + T12)"
            );
            tasks.values().cloned().collect::<Vec<BackgroundTaskInfo>>()
        };

        let event = ChatEvent::ActiveTasksUpdate { tasks: snapshot };
        let _ = events_tx.send(event.clone());
        if let Some(ref nats) = nats {
            nats.publish_chat_event(session_id, event);
        }
    }

    /// Internal: send a message to the client and stream the response to broadcast
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn stream_response(
        client: Arc<Mutex<InteractiveClient>>,
        events_tx: broadcast::Sender<ChatEvent>,
        prompt: String,
        session_id: String,
        graph: Arc<dyn GraphStore>,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        interrupt_flag: Arc<AtomicBool>,
        memory_manager: Option<Arc<Mutex<ConversationMemoryManager>>>,
        context_injector: Option<Arc<ContextInjector>>,
        next_seq: Arc<AtomicI64>,
        pending_messages: Arc<Mutex<VecDeque<PendingMessage>>>,
        is_streaming: Arc<AtomicBool>,
        streaming_text: Arc<Mutex<String>>,
        streaming_events: Arc<Mutex<Vec<ChatEvent>>>,
        event_emitter: Option<Arc<dyn crate::events::EventEmitter>>,
        nats: Option<Arc<crate::events::NatsEmitter>>,
        shared_sdk_control_rx: Arc<
            tokio::sync::Mutex<Option<tokio::sync::mpsc::Receiver<serde_json::Value>>>,
        >,
        auto_continue: Arc<AtomicBool>,
        retry_config: super::config::RetryConfig,
        enrichment_pipeline: Arc<super::enrichment::EnrichmentPipeline>,
        search: Arc<dyn crate::meilisearch::SearchStore>,
        documents: crate::documents::store::DocumentStore,
    ) {
        // Helper closure: emit a ChatEvent to local broadcast + NATS (if configured)
        let emit_chat = |event: ChatEvent,
                         tx: &broadcast::Sender<ChatEvent>,
                         nats: &Option<Arc<crate::events::NatsEmitter>>,
                         sid: &str| {
            let _ = tx.send(event.clone());
            if let Some(ref nats) = nats {
                nats.publish_chat_event(sid, event);
            }
        };

        // Create a NEW CancellationToken and store it in ActiveSession BEFORE
        // setting is_streaming=true. This ensures the token always matches
        // what interrupt() will cancel (fixes Gaps 1, 2, 4, 7, 10).
        let interrupt_token = {
            let mut sessions = active_sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                let token = CancellationToken::new();
                session.interrupt_token = token.clone();
                session.interrupt_flag.store(false, Ordering::SeqCst);
                token
            } else {
                warn!(
                    "Session {} no longer in active_sessions at stream start",
                    session_id
                );
                return;
            }
        };

        is_streaming.store(true, Ordering::SeqCst);
        // Broadcast streaming_status to all connected clients (multi-tab support)
        emit_chat(
            ChatEvent::StreamingStatus { is_streaming: true },
            &events_tx,
            &nats,
            &session_id,
        );
        // Emit CRUD event for session list live refresh
        if let Some(ref emitter) = event_emitter {
            emitter.emit_updated(
                crate::events::EntityType::ChatSession,
                &session_id,
                serde_json::json!({ "is_streaming": true }),
                None,
            );
        }
        // Clear streaming buffers for the new stream
        streaming_text.lock().await.clear();
        streaming_events.lock().await.clear();

        // Track whether this stream ended with error_max_turns (for auto-continue)
        let mut hit_error_max_turns = false;

        // Track whether any *productive* tool_use occurred in this stream turn.
        // "Conclusive" tools (git commit/push/status, git tag) don't count — the agent
        // may be wrapping up without actually finishing the task.
        let mut had_productive_tool_use = false;
        // Track conclusive tool use separately — when BOTH productive and conclusive
        // happen in the same turn (Edit + git commit), the agent may be finishing
        // AND concluding in one shot. We still want to check objectives in that case.
        let mut had_conclusive_tool_use = false;

        // Track whether a CompactBoundary was received during this stream turn.
        // When true, we'll rebuild and re-inject project context after the stream ends.
        let mut needs_post_compaction_injection = false;

        // The ONE expansion of this turn's stored message (`refs::turn`, shared with
        // the agent runtime): the visible text, the `<po-context>` pointers of its
        // `#` references, the attached documents' text. A message without references
        // is expanded exactly as before.
        let turn = crate::refs::turn::expand_user_turn_in(&graph, &prompt, &session_id).await;

        // The images attached to the stored message (`message_attachments`, the
        // loader of the agent engine): sent inline with the turn below. A turn
        // without attachment (continuation, hint, tool output) has none.
        let images = super::message_attachments::load_images(&graph, &documents, &prompt).await;

        // Tell the clients what the references resolved to (persisted for replay).
        if let Some(event) = turn.event() {
            emit_chat(event.clone(), &events_tx, &nats, &session_id);
            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                let record = ChatEventRecord {
                    id: Uuid::new_v4(),
                    session_id: uuid,
                    seq: next_seq.fetch_add(1, Ordering::SeqCst),
                    event_type: event.event_type().to_string(),
                    data: serde_json::to_string(&event).unwrap_or_default(),
                    created_at: chrono::Utc::now(),
                };
                let _ = graph.store_chat_events(uuid, vec![record]).await;
            }
        }

        // Record user message in memory manager (what the user typed)
        if let Some(ref mm) = memory_manager {
            let mut mm = mm.lock().await;
            mm.record_user_message(&turn.memory_text);
        }

        // What the enrichment reads, and what it must not inject twice.
        let prompt = turn.enrichment_text.clone();

        // ===== PRE-ENRICHMENT PIPELINE =====
        // Enrich the prompt with context from the knowledge graph BEFORE the LLM call.
        // If the pipeline has no stages or all fail, the original prompt is used unchanged.
        // The agent engine runs the same function (`enrichment_for_turn`): one logic.
        let prompt = {
            let (protocol, anchor) = {
                let sessions = active_sessions.read().await;
                // The anchor mode is the session's own, fixed when it was opened or
                // resumed: never read from the environment per turn.
                let anchor = sessions
                    .get(&session_id)
                    .map(|s| s.anchor.clone())
                    .unwrap_or_default();
                let protocol = sessions
                    .get(&session_id)
                    .map(|s| TurnProtocol {
                        run_id: s.protocol_run_id,
                        state: s.protocol_state.clone(),
                        reasoning_path_tracker: Some(s.reasoning_path_tracker.clone()),
                    })
                    .unwrap_or_default();
                (protocol, anchor)
            };
            match enrichment_for_turn(
                &graph,
                &enrichment_pipeline,
                &session_id,
                &prompt,
                protocol,
                turn.excluded_note_ids.clone(),
                &anchor,
            )
            .await
            {
                Some(enrichment_md) => prepend_enrichment(&enrichment_md, &prompt),
                None => prompt,
            }
        };

        // After the (enriched) visible text: the pointers, then the documents.
        let prompt = format!("{prompt}{}", turn.model_tail);

        // Events are persisted in Neo4j — the WebSocket replay handles late-joining clients.

        // Variables that persist across retry iterations (needed after the retry loop)
        let session_uuid = Uuid::parse_str(&session_id).ok();
        let mut assistant_text_parts: Vec<String> = Vec::new();
        let mut events_to_persist: Vec<ChatEventRecord> = Vec::new();

        // What the CLI is given: the prompt as one string, or — with images — the
        // text and image blocks, checked by nexus (`agent_runtime::CliInput`). An
        // image that cannot be read or that nexus refuses: `images_refused` on the
        // wire (persisted), NOTHING written on stdin, no retry; the session stays
        // usable for the next message.
        let cli_input = match images
            .map_err(|reason| {
                warn!(session_id = %session_id, %reason, "an attached image could not be read");
                Box::new(super::agent_runtime::images_refused(
                    "unreadable",
                    format!("Error: {reason}: the message was not sent."),
                ))
            })
            .and_then(|images| super::agent_runtime::CliInput::of(prompt, &images))
        {
            Ok(input) => Some(input),
            Err(refusal) => {
                info!(session_id = %session_id, "the attached images were refused: nothing sent to the CLI");
                emit_chat((*refusal).clone(), &events_tx, &nats, &session_id);
                if let Some(uuid) = session_uuid {
                    events_to_persist.push(ChatEventRecord {
                        id: Uuid::new_v4(),
                        session_id: uuid,
                        seq: next_seq.fetch_add(1, Ordering::SeqCst),
                        event_type: refusal.event_type().to_string(),
                        data: serde_json::to_string(&refusal).unwrap_or_default(),
                        created_at: chrono::Utc::now(),
                    });
                }
                None
            }
        };
        let mut emitted_tool_use_ids: std::collections::HashMap<String, Option<usize>> =
            std::collections::HashMap::new();
        let mut pending_tool_calls: std::collections::HashMap<String, Option<String>> =
            std::collections::HashMap::new();

        // Temporarily take the SDK control receiver from the shared slot.
        // This allows us to listen for control protocol messages (e.g., `can_use_tool`
        // permission requests) in parallel with the message stream.
        // The receiver is put back into the shared slot when this stream ends,
        // so the next stream_response invocation can reuse it.
        let mut sdk_control_rx = shared_sdk_control_rx.lock().await.take();

        // Get a handle to the session's pending permission inputs map so we can
        // store the original tool input when a permission request arrives.
        // send_permission_response() reads from this same Arc to retrieve the input.
        //
        // Also clone the stdin_tx sender for auto-allowing AskUserQuestion control
        // requests inline (without going through send_permission_response).
        let (pending_perm_inputs, stdin_tx_for_auto_allow, work_log) = {
            let guard = active_sessions.read().await;
            match guard.get(&session_id) {
                Some(s) => (
                    s.pending_permission_inputs.clone(),
                    s.stdin_tx.clone(),
                    s.work_log.clone(),
                ),
                None => (
                    Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
                    None,
                    Arc::new(Mutex::new(SessionWorkLog::default())),
                ),
            }
        };

        // Extract hook callbacks registry BEFORE taking the long-lived client lock.
        // This Arc<RwLock<>> allows dispatching hook_callbacks inside the select! loop
        // without needing to re-lock the client (which is held by the stream).
        let hook_callbacks_registry = {
            let c = client.lock().await;
            c.hook_callbacks()
        };

        // ===== RETRY LOOP =====
        // Wraps the entire streaming block. On retryable API errors (500, 529) with
        // 0 tokens emitted, we retry with exponential backoff instead of propagating.
        let mut retry_attempt = 0u32;

        'retry_loop: loop {
            // A refused turn is never sent (and so never retried).
            let Some(input) = cli_input.clone() else {
                break 'retry_loop;
            };
            // Per-attempt state — reset on each retry iteration
            let mut should_retry = false;
            let mut last_retry_error = String::new();

            // Reset per-attempt accumulators (they may have partial data from a failed attempt)
            if retry_attempt > 0 {
                assistant_text_parts.clear();
                events_to_persist.clear();
                emitted_tool_use_ids.clear();
                pending_tool_calls.clear();
                streaming_text.lock().await.clear();
                streaming_events.lock().await.clear();
            }

            {
                let mut c = client.lock().await;

                // Try to start the stream. If it fails with a retryable error,
                // set should_retry and break to the retry decision block.
                // Every attempt writes the same input: a retry of a turn with
                // images sends the images again, never the text alone.
                let stream_start = input.send(&mut c).await;
                let stream_ok = match stream_start {
                    Ok(s) => Some(s),
                    Err(e) => {
                        let err_str = format!("{}", e);
                        let kind = classify_api_error(&err_str);

                        if kind.is_retryable()
                            && retry_attempt < retry_config.max_attempts
                            && !interrupt_flag.load(Ordering::SeqCst)
                        {
                            // Retryable — set flag, will handle after client lock release
                            should_retry = true;
                            last_retry_error = err_str;
                        } else {
                            // Not retryable or max retries exceeded — propagate error
                            error!("Error starting stream for session {}: {}", session_id, e);

                            // If this is a ChannelSendError (CLI is dead), remove the
                            // session from active_sessions so the next user message
                            // routes to resume_session instead of hitting the same dead CLI.
                            let is_channel_dead = err_str.contains("channel")
                                || err_str.contains("Channel")
                                || err_str.contains("Not connected");
                            if is_channel_dead {
                                warn!(
                                    "CLI appears dead for session {} ({}), removing from active_sessions \
                                     so next message triggers resume_session",
                                    session_id, err_str
                                );
                                active_sessions.write().await.remove(&session_id);
                                notify_attention(
                                    &event_emitter,
                                    AttentionSubject::Session(session_id.to_string()),
                                    AttentionReason::SessionInactive,
                                );
                            }

                            emit_chat(
                                ChatEvent::Error {
                                    message: format!("Error: {}", e),
                                    parent_tool_use_id: None,
                                    code: None,
                                    reason: None,
                                    index: None,
                                },
                                &events_tx,
                                &nats,
                                &session_id,
                            );
                            is_streaming.store(false, Ordering::SeqCst);
                            if let Some(ref emitter) = event_emitter {
                                emitter.emit_updated(
                                    crate::events::EntityType::ChatSession,
                                    &session_id,
                                    serde_json::json!({ "is_streaming": false }),
                                    None,
                                );
                            }
                            streaming_text.lock().await.clear();
                            streaming_events.lock().await.clear();
                            return;
                        }
                        None
                    }
                };

                if let Some(s) = stream_ok {
                    let mut stream = std::pin::pin!(s);

                    // Track the current parent_tool_use_id from stream events.
                    // When a sub-agent is active, its stream Messages carry parent_tool_use_id.
                    // We capture this so permission requests (which arrive via a separate control
                    // channel without parent info) can be attributed to the correct agent.
                    let mut current_parent_tool_use_id: Option<String> = None;

                    // Helper closure: process an SDK control message (permission or AskUserQuestion).
                    // Returns Some(ChatEvent) if a can_use_tool was parsed, None otherwise.
                    // The caller is responsible for adding the event to streaming_events (async)
                    // and for auto-allowing AskUserQuestion events via stdin_tx.
                    let handle_control_msg = |control_msg: serde_json::Value,
                                              events_to_persist: &mut Vec<ChatEventRecord>,
                                              next_seq: &std::sync::atomic::AtomicI64,
                                              current_parent: Option<String>|
                     -> Option<ChatEvent> {
                        let event = parse_permission_control_msg(&control_msg, current_parent)?;

                        match &event {
                            ChatEvent::PermissionRequest { id, tool, .. } => {
                                info!(
                                    session_id = %session_id,
                                    tool = %tool,
                                    request_id = %id,
                                    "Permission request from CLI"
                                );
                            }
                            ChatEvent::AskUserQuestion { id, .. } => {
                                info!(
                                    session_id = %session_id,
                                    request_id = %id,
                                    "AskUserQuestion from CLI (will auto-allow)"
                                );
                            }
                            _ => {}
                        }

                        // Persist the event
                        if let Some(uuid) = session_uuid {
                            let seq = next_seq.fetch_add(1, Ordering::SeqCst);
                            events_to_persist.push(ChatEventRecord {
                                id: Uuid::new_v4(),
                                session_id: uuid,
                                seq,
                                event_type: event.event_type().to_string(),
                                data: serde_json::to_string(&event).unwrap_or_default(),
                                created_at: chrono::Utc::now(),
                            });
                        }

                        // Broadcast to WebSocket clients
                        emit_chat(event.clone(), &events_tx, &nats, &session_id);
                        // Light signal on /ws/events (ids only, never the command/question)
                        notify_attention_for_chat_event(&event_emitter, &session_id, &event);
                        Some(event)
                    };

                    // Main stream loop — uses tokio::select! to listen for BOTH stream
                    // events AND SDK control messages (permission requests) concurrently.
                    //
                    // BUG FIX: Previously used `stream.next().await` followed by
                    // `rx.try_recv()`. When the CLI blocks waiting for permission approval,
                    // it stops sending stream events, so `stream.next()` would never yield
                    // and `try_recv()` was never reached — permission requests were lost.
                    //
                    // Now we select! between the two sources so control messages are
                    // processed even when the stream is idle (waiting for permission).
                    loop {
                        let result = if let Some(ref mut rx) = sdk_control_rx {
                            tokio::select! {
                                biased;  // Prioritize interrupt > control messages > stream events

                                // Interrupt token: immediately unblocks the select! when cancelled,
                                // even if stream.next() is blocked waiting for CLI output (e.g., sleep 60).
                                // This is the core fix for the interrupt-during-long-tool bug.
                                _ = interrupt_token.cancelled() => {
                                    info!(
                                        "Interrupt token cancelled during stream for session {}",
                                        session_id
                                    );
                                    break;
                                }

                                control_msg = rx.recv() => {
                                    match control_msg {
                                        Some(msg) => {
                                            // Hook callbacks are answered by the session-lifetime control
                                            // pump (`control_pump.rs`) before they reach this channel. This
                                            // branch only remains as a defensive fallback.
                                            if is_hook_callback(&msg) {
                                                let request_id = msg.get("request_id")
                                                    .or_else(|| msg.get("request").and_then(|r| r.get("request_id")))
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string();
                                                if let Some(result) = dispatch_hook_from_registry(&msg, &hook_callbacks_registry).await {
                                                    info!(session_id = %session_id, request_id = %request_id, "Hook callback dispatched");
                                                    if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                        let response_json = build_hook_response_json(&request_id, &result);
                                                        let _ = tx.send(response_json).await;
                                                        debug!(request_id = %request_id, "Hook response sent to CLI");
                                                    }
                                                }
                                                continue;
                                            }

                                            if let Some(evt) = handle_control_msg(msg, &mut events_to_persist, &next_seq, current_parent_tool_use_id.clone()) {
                                                // AskUserQuestion: auto-allow the control request so the CLI
                                                // waits for the tool_result (user's answer) instead of blocking.
                                                // Do NOT store in pending_perm_inputs (it's not a permission).
                                                if let ChatEvent::AskUserQuestion { ref id, ref input, .. } = evt {
                                                    if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                        let control_response = serde_json::json!({
                                                            "type": "control_response",
                                                            "response": {
                                                                "subtype": "success",
                                                                "request_id": id,
                                                                "response": {
                                                                    "behavior": "allow",
                                                                    "updatedInput": input
                                                                }
                                                            }
                                                        });
                                                        if let Ok(json) = serde_json::to_string(&control_response) {
                                                            let _ = tx.send(json).await;
                                                            info!(request_id = %id, "Auto-allowed AskUserQuestion control request");
                                                        }
                                                    }
                                                } else if let ChatEvent::PermissionRequest { ref id, ref input, .. } = evt {
                                                    // Regular permission: store original input for later response
                                                    if !id.is_empty() {
                                                        store_pending_perm_input(&pending_perm_inputs, id, input).await;
                                                    }
                                                }
                                                streaming_events.lock().await.push(evt);
                                            }
                                            continue; // Go back to select! for next event
                                        }
                                        None => {
                                            debug!("SDK control channel closed");
                                            sdk_control_rx = None;
                                            continue;
                                        }
                                    }
                                }

                                stream_item = stream.next() => {
                                    match stream_item {
                                        Some(result) => {
                                            // Also drain any buffered control messages
                                            while let Ok(msg) = rx.try_recv() {
                                                // Hook callbacks: dispatch inline (same as select! branch)
                                                if is_hook_callback(&msg) {
                                                    let request_id = msg.get("request_id")
                                                        .or_else(|| msg.get("request").and_then(|r| r.get("request_id")))
                                                        .and_then(|v| v.as_str())
                                                        .unwrap_or("")
                                                        .to_string();
                                                    if let Some(result) = dispatch_hook_from_registry(&msg, &hook_callbacks_registry).await {
                                                        info!(session_id = %session_id, request_id = %request_id, "Hook callback dispatched (drain)");
                                                        if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                            let response_json = build_hook_response_json(&request_id, &result);
                                                            let _ = tx.try_send(response_json);
                                                        }
                                                    }
                                                    continue;
                                                }

                                                if let Some(evt) = handle_control_msg(msg, &mut events_to_persist, &next_seq, current_parent_tool_use_id.clone()) {
                                                    // AskUserQuestion: auto-allow (same logic as above)
                                                    if let ChatEvent::AskUserQuestion { ref id, ref input, .. } = evt {
                                                        if let Some(ref tx) = stdin_tx_for_auto_allow {
                                                            let control_response = serde_json::json!({
                                                                "type": "control_response",
                                                                "response": {
                                                                    "subtype": "success",
                                                                    "request_id": id,
                                                                    "response": {
                                                                        "behavior": "allow",
                                                                        "updatedInput": input
                                                                    }
                                                                }
                                                            });
                                                            if let Ok(json) = serde_json::to_string(&control_response) {
                                                                // try_send here because we're in a sync-ish drain loop
                                                                let _ = tx.try_send(json);
                                                                info!(request_id = %id, "Auto-allowed AskUserQuestion control request (drain)");
                                                            }
                                                        }
                                                    } else if let ChatEvent::PermissionRequest { ref id, ref input, .. } = evt {
                                                        if !id.is_empty() {
                                                            store_pending_perm_input(&pending_perm_inputs, id, input).await;
                                                        }
                                                    }
                                                    streaming_events.lock().await.push(evt);
                                                }
                                            }
                                            result
                                        }
                                        None => break, // Stream ended
                                    }
                                }
                            }
                        } else {
                            // No control channel — use select! with interrupt token + stream
                            tokio::select! {
                                biased;

                                _ = interrupt_token.cancelled() => {
                                    info!(
                                        "Interrupt token cancelled during stream for session {} (no control channel)",
                                        session_id
                                    );
                                    break;
                                }

                                stream_item = stream.next() => {
                                    match stream_item {
                                        Some(result) => result,
                                        None => break,
                                    }
                                }
                            }
                        };

                        // Secrets first: everything below (deltas, events,
                        // persistence, NATS, memory) reads the masked message.
                        let result = result.map(Self::mask_cli_message);
                        match result {
                            Ok(ref msg) => {
                                // Track the current parent_tool_use_id from every stream message.
                                // This is used by handle_control_msg to attribute permission
                                // requests to the correct sub-agent. We update on every message
                                // (including top-level ones where parent is None) so the tracker
                                // resets correctly when switching between agents.
                                current_parent_tool_use_id =
                                    msg.parent_tool_use_id().map(|s| s.to_string());

                                // Handle StreamEvent — emit StreamDelta for text tokens directly
                                // stream_delta are NOT persisted (too many writes)
                                if let Message::StreamEvent {
                                    event:
                                        StreamEventData::ContentBlockDelta {
                                            delta: StreamDelta::TextDelta { ref text },
                                            ..
                                        },
                                    ..
                                } = msg
                                {
                                    let parent = msg.parent_tool_use_id().map(|s| s.to_string());
                                    // Accumulate for mid-stream join snapshot
                                    streaming_text.lock().await.push_str(text);
                                    emit_chat(
                                        ChatEvent::StreamDelta {
                                            text: text.clone(),
                                            parent_tool_use_id: parent,
                                        },
                                        &events_tx,
                                        &nats,
                                        &session_id,
                                    );
                                    continue;
                                }

                                // Extract cli_session_id from Result message
                                if let Message::Result {
                                    session_id: ref cli_sid,
                                    total_cost_usd: ref cost,
                                    ..
                                } = msg
                                {
                                    // Update Neo4j with cli_session_id and cost. Never the
                                    // message count: each user message bumps it (send_message,
                                    // drain, the NATS listener), a `result` is not a message.
                                    if let Some(uuid) = session_uuid {
                                        let _ = graph
                                            .update_chat_session(
                                                uuid,
                                                Some(cli_sid.clone()),
                                                None,
                                                None,
                                                *cost,
                                                None,
                                                None,
                                            )
                                            .await;
                                    }

                                    // Update active session's cli_session_id
                                    let mut sessions = active_sessions.write().await;
                                    if let Some(active) = sessions.get_mut(&session_id) {
                                        active.cli_session_id = Some(cli_sid.clone());
                                        active.last_activity = Instant::now();
                                    }
                                }

                                // Extract cli_session_id and model from System init message
                                if let Message::System { subtype, ref data } = msg {
                                    if subtype == "init" {
                                        let cli_sid = data
                                            .get("session_id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        if let Some(uuid) = session_uuid {
                                            let _ = graph
                                                .update_chat_session(
                                                    uuid,
                                                    Some(cli_sid.clone()),
                                                    None,
                                                    None,
                                                    None,
                                                    None,
                                                    None,
                                                )
                                                .await;
                                        }
                                        // Update active session
                                        let mut sessions = active_sessions.write().await;
                                        if let Some(active) = sessions.get_mut(&session_id) {
                                            active.cli_session_id = Some(cli_sid);
                                            active.last_activity = Instant::now();
                                        }
                                    }
                                }

                                // Check for retryable API error in Result message.
                                // If retryable and no tokens were emitted, skip event emission
                                // and break to retry the entire stream.
                                if let Message::Result {
                                    is_error: true,
                                    result: Some(ref text),
                                    ..
                                } = msg
                                {
                                    let kind = classify_api_error(text);
                                    let text_empty = streaming_text.lock().await.is_empty();
                                    if kind.is_retryable()
                                        && text_empty
                                        && retry_attempt < retry_config.max_attempts
                                        && !interrupt_flag.load(Ordering::SeqCst)
                                    {
                                        should_retry = true;
                                        last_retry_error = text.clone();
                                        break; // break inner stream loop
                                    }
                                }

                                // Collect assistant text for memory
                                if let Message::Assistant {
                                    message: ref am, ..
                                } = msg
                                {
                                    for block in &am.content {
                                        if let ContentBlock::Text(t) = block {
                                            assistant_text_parts.push(t.text.clone());
                                        }
                                    }
                                }

                                // Convert to ChatEvent(s) and emit + persist structured events
                                let events = Self::message_to_events(msg);
                                for event in events {
                                    // Deduplicate ToolUse events — ContentBlockStart and
                                    // AssistantMessage can both produce the same tool_use.
                                    //
                                    // ContentBlockStart arrives first with input: {} (empty),
                                    // AssistantMessage arrives later with the FULL input params.
                                    //
                                    // Strategy:
                                    // 1. First occurrence (ContentBlockStart): emit + persist normally
                                    // 2. Second occurrence (AssistantMessage): DON'T re-emit to broadcast
                                    //    (clients already have the tool_use), but UPDATE the persisted
                                    //    record and streaming_events with the full input.
                                    if let ChatEvent::ToolUse {
                                        ref id,
                                        ref tool,
                                        ref input,
                                        ref parent_tool_use_id,
                                        ..
                                    } = event
                                    {
                                        // Plan 754a1379, T3 — INSERT side of the
                                        // background-task lifecycle. Idempotent across
                                        // both ContentBlockStart (empty input) and
                                        // AssistantMessage (full input) passes; for
                                        // Monitor the first call inserts and the second
                                        // refreshes the description, for Bash with
                                        // run_in_background=true only the second call
                                        // succeeds (the flag isn't visible on the first
                                        // pass).
                                        let _ = Self::track_background_task_start(
                                            &session_id,
                                            &active_sessions,
                                            &events_tx,
                                            &nats,
                                            id,
                                            tool,
                                            input,
                                            parent_tool_use_id.as_deref(),
                                        )
                                        .await;

                                        if let Some(persist_idx) = emitted_tool_use_ids.get(id) {
                                            // Duplicate — update persisted record with full input
                                            let has_real_input = input.is_object()
                                                && input.as_object().is_some_and(|o| !o.is_empty());
                                            if has_real_input {
                                                if let Some(idx) = persist_idx {
                                                    if let Some(record) =
                                                        events_to_persist.get_mut(*idx)
                                                    {
                                                        record.data = serde_json::to_string(&event)
                                                            .unwrap_or_default();
                                                        debug!(
                                                    "Updated persisted ToolUse input for id={}",
                                                    id
                                                );
                                                    }
                                                }
                                                // Also update in streaming_events snapshot
                                                let mut se = streaming_events.lock().await;
                                                if let Some(existing) = se.iter_mut().find(|e| {
                                            matches!(e, ChatEvent::ToolUse { id: ref eid, .. } if eid == id)
                                        }) {
                                            *existing = event.clone();
                                        }
                                                // Emit ToolUseInputResolved so the frontend can
                                                // update the existing tool_use block's input
                                                emit_chat(
                                                    ChatEvent::ToolUseInputResolved {
                                                        id: id.clone(),
                                                        input: input.clone(),
                                                        parent_tool_use_id: parent_tool_use_id
                                                            .clone(),
                                                    },
                                                    &events_tx,
                                                    &nats,
                                                    &session_id,
                                                );
                                            }
                                            debug!("Skipping duplicate ToolUse broadcast (id={}), sent input_resolved", id);
                                            continue;
                                        }
                                        // First occurrence — record it
                                        emitted_tool_use_ids.insert(id.clone(), None);
                                        // Track as pending (no ToolResult yet)
                                        pending_tool_calls
                                            .insert(id.clone(), parent_tool_use_id.clone());
                                    }

                                    // When a ToolResult arrives, the tool is no longer pending
                                    if let ChatEvent::ToolResult { ref id, .. } = event {
                                        pending_tool_calls.remove(id);
                                    }

                                    // Persist structured events (skip transient: stream_delta, streaming_status)
                                    if !matches!(
                                        event,
                                        ChatEvent::StreamDelta { .. }
                                            | ChatEvent::StreamingStatus { .. }
                                    ) {
                                        if let Some(uuid) = session_uuid {
                                            let seq = next_seq.fetch_add(1, Ordering::SeqCst);
                                            let persist_idx = events_to_persist.len();
                                            events_to_persist.push(ChatEventRecord {
                                                id: Uuid::new_v4(),
                                                session_id: uuid,
                                                seq,
                                                event_type: event.event_type().to_string(),
                                                data: serde_json::to_string(&event)
                                                    .unwrap_or_default(),
                                                created_at: chrono::Utc::now(),
                                            });
                                            // Track persist index for ToolUse so we can update later
                                            if let ChatEvent::ToolUse { ref id, .. } = event {
                                                emitted_tool_use_ids
                                                    .insert(id.clone(), Some(persist_idx));
                                            }
                                        }
                                    }

                                    // Accumulate structured events for mid-stream join snapshot.
                                    // Excluded:
                                    // - StreamDelta: text is in streaming_text (sent as partial_text)
                                    // - StreamingStatus: transient, sent explicitly in Phase 1.5
                                    // - AssistantText: duplicates streaming_text content (sent as partial_text)
                                    if !matches!(
                                        event,
                                        ChatEvent::StreamDelta { .. }
                                            | ChatEvent::StreamingStatus { .. }
                                            | ChatEvent::AssistantText { .. }
                                    ) {
                                        // Flush accumulated text before non-text events (ToolUse,
                                        // ToolResult, etc.) so the snapshot preserves correct ordering.
                                        // Without this, partial_text would contain "Text A + Text B"
                                        // with no way to know Text A came before ToolUse.
                                        // After flushing, partial_text only contains text streamed
                                        // AFTER the last structured event.
                                        {
                                            let mut st = streaming_text.lock().await;
                                            if !st.is_empty() {
                                                // Note: streaming_text is a flat buffer that doesn't track
                                                // per-agent text. The parent_tool_use_id is set to None here.
                                                // The frontend uses individual streaming_events (which carry
                                                // parent_tool_use_id) for agent grouping, not partial_text.
                                                streaming_events.lock().await.push(
                                                    ChatEvent::AssistantText {
                                                        content: st.clone(),
                                                        parent_tool_use_id: None,
                                                    },
                                                );
                                                st.clear();
                                            }
                                        }
                                        streaming_events.lock().await.push(event.clone());
                                    }

                                    // Track tool_use for objective tracking + work_log
                                    if let ChatEvent::ToolUse {
                                        ref tool,
                                        ref input,
                                        ..
                                    } = event
                                    {
                                        if is_conclusive_tool(tool, input) {
                                            had_conclusive_tool_use = true;
                                        } else {
                                            had_productive_tool_use = true;
                                        }
                                        work_log.lock().await.record_tool_use(tool, input);
                                    }

                                    // Detect error_max_turns for auto-continue
                                    if let ChatEvent::Result { ref subtype, .. } = event {
                                        if subtype == "error_max_turns" {
                                            hit_error_max_turns = true;
                                        }
                                    }

                                    // Detect CompactBoundary for post-compaction context re-injection
                                    if matches!(event, ChatEvent::CompactBoundary { .. }) {
                                        info!(
                                            "CompactBoundary detected in session {}, will inject context after stream",
                                            session_id
                                        );
                                        needs_post_compaction_injection = true;
                                    }

                                    emit_chat(event, &events_tx, &nats, &session_id);
                                }
                            }
                            Err(e) => {
                                let err_str = format!("{}", e);
                                let kind = classify_api_error(&err_str);
                                let text_empty = streaming_text.lock().await.is_empty();

                                if kind.is_retryable()
                                    && text_empty
                                    && retry_attempt < retry_config.max_attempts
                                    && !interrupt_flag.load(Ordering::SeqCst)
                                {
                                    // Retryable stream error with 0 tokens — will retry
                                    should_retry = true;
                                    last_retry_error = err_str;
                                } else {
                                    // Not retryable — propagate error
                                    error!("Stream error for session {}: {}", session_id, e);
                                    let cli_gone = err_str.contains("ended before a Result");
                                    if cli_gone {
                                        warn!(
                                            "CLI left before answering for session {}, removing from \
                                             active_sessions so the next message resumes it",
                                            session_id
                                        );
                                        active_sessions.write().await.remove(&session_id);
                                        notify_attention(
                                            &event_emitter,
                                            AttentionSubject::Session(session_id.to_string()),
                                            AttentionReason::SessionInactive,
                                        );
                                    }
                                    emit_chat(
                                        ChatEvent::Error {
                                            message: if cli_gone {
                                                "The Claude Code process stopped before answering. Your message was not processed: send it again, the session will resume (or use Restart session).".to_string()
                                            } else {
                                                format!("Error: {}", e)
                                            },
                                            parent_tool_use_id: None,
                                            code: None,
                                            reason: None,
                                            index: None,
                                        },
                                        &events_tx,
                                        &nats,
                                        &session_id,
                                    );
                                }
                                break;
                            }
                        }
                    }
                } // end if let Some(s) = stream_ok
            } // client lock released here

            // ===== RETRY DECISION =====
            // If should_retry is set (from stream item error or Result is_error),
            // emit Retrying event, sleep with backoff, and continue the retry loop.
            if should_retry {
                retry_attempt += 1;
                let delay = retry_config.delay_for_attempt(retry_attempt);
                warn!(
                    session_id = %session_id,
                    attempt = retry_attempt,
                    max_attempts = retry_config.max_attempts,
                    delay_ms = delay,
                    error = %last_retry_error,
                    "Retryable stream error (0 tokens emitted), will retry"
                );

                // Emit Retrying event so frontend can show indicator
                let retry_event = ChatEvent::Retrying {
                    attempt: retry_attempt,
                    max_attempts: retry_config.max_attempts,
                    delay_ms: delay,
                    error_message: last_retry_error.clone(),
                };
                emit_chat(retry_event.clone(), &events_tx, &nats, &session_id);

                // Persist the Retrying event
                if let Some(uuid) = session_uuid {
                    let seq = next_seq.fetch_add(1, Ordering::SeqCst);
                    let record = ChatEventRecord {
                        id: Uuid::new_v4(),
                        session_id: uuid,
                        seq,
                        event_type: retry_event.event_type().to_string(),
                        data: serde_json::to_string(&retry_event).unwrap_or_default(),
                        created_at: chrono::Utc::now(),
                    };
                    let _ = graph.store_chat_events(uuid, vec![record]).await;
                }

                // Reset streaming buffers for the retry
                streaming_text.lock().await.clear();
                streaming_events.lock().await.clear();

                // Interruptible backoff sleep
                let cancelled = tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_millis(delay)) => false,
                    _ = interrupt_token.cancelled() => true,
                };
                if cancelled {
                    info!(
                        session_id = %session_id,
                        "Retry cancelled by interrupt during backoff"
                    );
                    break 'retry_loop;
                }

                continue 'retry_loop;
            }

            break 'retry_loop;
        } // end 'retry_loop

        // Put the SDK control receiver back into the shared slot so the next
        // stream_response invocation can reuse it (fixes permission requests
        // being silently lost after the first message in a session).
        if sdk_control_rx.is_some() {
            *shared_sdk_control_rx.lock().await = sdk_control_rx;
        }

        // ===== POST-STREAM PROCESSING =====
        // All post-stream logic delegated to PostStreamHandler (see post_stream.rs)
        let post_ctx = super::post_stream::PostStreamContext::build(&graph, session_uuid).await;
        let post_handler = super::post_stream::PostStreamHandler {
            graph: graph.clone(),
            pending_messages: pending_messages.clone(),
            session_id: session_id.clone(),
            session_uuid,
            active_sessions: active_sessions.clone(),
            events_tx: events_tx.clone(),
            nats: nats.clone(),
            next_seq: next_seq.clone(),
            interrupt_flag: interrupt_flag.clone(),
            interrupt_token: interrupt_token.clone(),
            ctx: post_ctx,
            is_streaming: is_streaming.clone(),
            streaming_text: streaming_text.clone(),
            streaming_events: streaming_events.clone(),
            event_emitter: event_emitter.clone(),
            search: search.clone(),
            auto_continue: auto_continue.clone(),
            work_log: work_log.clone(),
        };

        // 1. Post-compaction context re-injection
        post_handler
            .handle_post_compaction(needs_post_compaction_injection)
            .await;

        // 2. Lock-free interrupt + ToolCancelled
        post_handler
            .handle_interrupt_cleanup(&stdin_tx_for_auto_allow, &pending_tool_calls)
            .await;

        // 3. Auto-continue
        let auto_continue_allowed = post_handler.handle_auto_continue(hit_error_max_turns).await;

        // 4. Objective tracking — uses had_productive_tool_use (not had_tool_use)
        // so that conclusive-only turns (git commit/push) still trigger reminders.
        // Also passes had_conclusive_tool_use so that mixed turns (Edit + git commit)
        // still check for pending objectives.
        // A turn refused before it was sent (its images) did not stop the agent:
        // no objective reminder starts a turn without the user's picture.
        if cli_input.is_some() {
            post_handler
                .handle_objective_tracking(
                    had_productive_tool_use,
                    had_conclusive_tool_use,
                    auto_continue_allowed,
                    hit_error_max_turns,
                )
                .await;
        }

        // 5. Streaming status update
        let has_pending = post_handler.finalize_streaming_status().await;

        // 6. Batch-persist events to Neo4j
        post_handler.persist_events(events_to_persist).await;

        // 7. Memory / feedback / RFC
        post_handler
            .handle_feedback(&assistant_text_parts, &memory_manager, &context_injector)
            .await;

        // 8. Drain pending messages queue
        super::drain::drain_pending_messages(
            has_pending,
            client,
            events_tx,
            session_id,
            session_uuid,
            graph,
            active_sessions,
            interrupt_flag,
            memory_manager,
            context_injector,
            next_seq,
            pending_messages,
            is_streaming,
            streaming_text,
            streaming_events,
            event_emitter,
            nats,
            shared_sdk_control_rx,
            auto_continue,
            retry_config,
            enrichment_pipeline,
            search,
            documents,
        )
        .await;
    }

    /// Try to send a message to a session via NATS RPC (cross-instance proxy).
    ///
    /// Returns `Ok(true)` if the message was successfully proxied to the owning instance.
    /// Returns `Ok(false)` if NATS is not configured, no instance responded (timeout),
    /// or the remote instance reported the session is not active there.
    ///
    /// This method has NO side-effects on the local ChatManager — it only communicates
    /// with remote instances via NATS request/reply.
    ///
    /// Callers should fall back to `resume_session()` when this returns `Ok(false)`.
    pub async fn try_remote_send(
        &self,
        session_id: &str,
        message: &str,
        message_type: &str,
    ) -> Result<bool> {
        let Some(ref nats) = self.nats else {
            debug!(
                session_id = %session_id,
                "No NATS configured, skipping remote send"
            );
            return Ok(false);
        };

        info!(
            session_id = %session_id,
            "Attempting NATS RPC send to remote instance"
        );

        match nats
            .request_send_message(session_id, message, message_type)
            .await
        {
            Some(response) if response.success => {
                info!(
                    session_id = %session_id,
                    "Message proxied to remote instance via NATS RPC"
                );
                Ok(true)
            }
            Some(response) => {
                debug!(
                    session_id = %session_id,
                    error = ?response.error,
                    "Remote instance rejected message (session not active there)"
                );
                Ok(false)
            }
            None => {
                debug!(
                    session_id = %session_id,
                    "No NATS RPC reply (timeout) — no instance owns this session"
                );
                Ok(false)
            }
        }
    }

    /// Send a follow-up message to an existing session
    pub async fn send_message(&self, session_id: &str, message: &str) -> Result<()> {
        // Mode `full`: the turn may belong to another provider. A move sends the message
        // to the new session, so this one starts no turn.
        if self.move_provider_before_turn(session_id, message).await {
            return Ok(());
        }
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            // The turn router learns the text when the turn is prepared
            // (`ManagerTurnServices::prepare`): a queued message must not overwrite it.
            return handle.send_message(message).await;
        }
        // Check is_streaming with read lock first — if streaming, queue the message
        // AND trigger an interrupt so the stream breaks and processes it sooner (T4 fix, Gap 8).
        {
            let sessions = self.active_sessions.read().await;
            let session = sessions
                .get(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;

            if session.is_streaming.load(Ordering::SeqCst) {
                info!(
                    "Stream in progress for session {}, queuing message and interrupting",
                    session_id
                );
                // Queue the message (real user message)
                let mut queue = session.pending_messages.lock().await;
                queue.push_back(PendingMessage::user(message.to_string()));

                // Interrupt the stream so the message is processed sooner
                session.interrupt_flag.store(true, Ordering::SeqCst);
                session.interrupt_token.cancel();

                // Send interrupt to CLI immediately via stdin_tx (lock-free)
                if let Some(ref tx) = session.stdin_tx {
                    let json = InteractiveClient::build_interrupt_json();
                    let _ = tx.try_send(json);
                }

                return Ok(());
            }
        }

        // The model of the turn (`full` mode), before the message is written.
        self.apply_turn_directive(session_id, message).await;

        // Not streaming — get session state for persist + stream.
        // DON'T create new interrupt_token here — stream_response will do it.
        let (
            client,
            events_tx,
            interrupt_flag,
            memory_manager,
            next_seq,
            pending_messages,
            is_streaming,
            streaming_text,
            streaming_events,
            sdk_control_rx,
            auto_continue,
        ) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;

            session.last_activity = Instant::now();

            (
                session.client.clone(),
                session.events_tx.clone(),
                session.interrupt_flag.clone(),
                session.memory_manager.clone(),
                session.next_seq.clone(),
                session.pending_messages.clone(),
                session.is_streaming.clone(),
                session.streaming_text.clone(),
                session.streaming_events.clone(),
                session.sdk_control_rx.clone(),
                session.auto_continue.clone(),
            )
        };

        // No stream in progress — persist and broadcast immediately

        // Update message count in Neo4j
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Ok(Some(node)) = self.graph.get_chat_session(uuid).await {
                let _ = self
                    .graph
                    .update_chat_session(
                        uuid,
                        None,
                        None,
                        Some(node.message_count + 1),
                        None,
                        None,
                        None,
                    )
                    .await;
            }
        }

        // Persist the user_message event
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            let user_event = ChatEventRecord {
                id: Uuid::new_v4(),
                session_id: uuid,
                seq: next_seq.fetch_add(1, Ordering::SeqCst),
                event_type: "user_message".to_string(),
                data: serde_json::to_string(&ChatEvent::UserMessage {
                    content: message.to_string(),
                })
                .unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            let _ = self.graph.store_chat_events(uuid, vec![user_event]).await;
        }

        // Emit user_message on local broadcast + NATS (visible to all clients)
        let user_msg_event = ChatEvent::UserMessage {
            content: message.to_string(),
        };
        let _ = events_tx.send(user_msg_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, user_msg_event);
        }
        self.notify_attention(session_id, AttentionReason::UserMessage);

        // Start streaming directly
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();
        let injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();
        let retry_config = self.config.retry.clone();
        let enrichment_pipeline = self.enrichment_pipeline.clone();
        let search = self.search.clone();
        let documents = self.document_store.clone();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx,
                prompt,
                session_id_str,
                graph,
                active_sessions,
                interrupt_flag,
                memory_manager,
                injector,
                next_seq,
                pending_messages,
                is_streaming,
                streaming_text,
                streaming_events,
                event_emitter,
                nats,
                sdk_control_rx,
                auto_continue,
                retry_config,
                enrichment_pipeline,
                search,
                documents,
            )
            .await;
        });

        Ok(())
    }

    /// Inject a hint message into the agent session WITHOUT interrupting.
    ///
    /// Unlike `send_message()`, this does NOT:
    /// - Set `interrupt_flag`
    /// - Cancel `interrupt_token`
    /// - Send SIGINT via `stdin_tx`
    ///
    /// The message is queued in `pending_messages` and will be processed
    /// after the current stream turn ends naturally (via the drain loop).
    ///
    /// If the session is NOT currently streaming, falls back to `send_message()`
    /// since there's no stream to preserve.
    ///
    /// Used by the AgentGuard for soft reminders (idle detection, loop detection,
    /// post-compaction context re-injection) without killing child processes
    /// (cargo build, npm install, etc.).
    pub async fn inject_hint(&self, session_id: &str, message: &str) -> Result<()> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle.inject_hint(message).await;
        }
        let sessions = self.active_sessions.read().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;

        if session.is_streaming.load(Ordering::SeqCst) {
            // Queue the message WITHOUT interrupt — the drain loop will pick it up
            // after the current stream turn ends naturally.
            info!("Injecting hint into session {} (no interrupt)", session_id);
            let mut queue = session.pending_messages.lock().await;
            queue.push_back(PendingMessage::system_hint(message.to_string()));
            Ok(())
        } else {
            // Not streaming — just send normally (will start a new stream)
            drop(sessions); // Release read lock before calling send_message
            self.send_message(session_id, message).await
        }
    }

    /// Hold a user message until the running turn ends — it interrupts nothing.
    ///
    /// `send_message` mid-stream queues the message AND interrupts the turn so
    /// it is read sooner. This is the other choice: the message waits in
    /// `pending_messages` and the drain delivers it when the turn ends, as its
    /// own turn (persisted and broadcast as a `user_message` at that moment).
    /// The session delivers it, not the client: it leaves whether or not a
    /// client is still looking at this conversation.
    ///
    /// `message` is the stored form (attachment block included).
    ///
    /// Returns `true` when the message was held, `false` when the session was
    /// idle and the message was simply sent.
    pub async fn queue_user_message(&self, session_id: &str, message: &str) -> Result<bool> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle.queue_message(message).await;
        }
        {
            let sessions = self.active_sessions.read().await;
            let session = sessions
                .get(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;

            // `is_streaming` is read under the queue lock, the same lock the end
            // of a turn holds to decide there is nothing left to drain
            // (`PostStreamHandler::finalize_streaming_status`). Either the turn
            // is still running and will see this entry, or it is over and the
            // message goes out directly below — never queued behind a turn
            // that has already finished.
            let mut queue = session.pending_messages.lock().await;
            if session.is_streaming.load(Ordering::SeqCst) {
                queue.push_back(PendingMessage::held_user(message.to_string()));
                let messages = super::pending_queue::snapshot(&queue);
                drop(queue);
                info!(
                    "Holding user message for session {} (no interrupt)",
                    session_id
                );
                self.publish_pending_queue(session_id, &session.events_tx, messages);
                return Ok(true);
            }
        }
        self.send_message(session_id, message).await?;
        Ok(false)
    }

    /// Broadcast the held messages of a session to its clients, here and on
    /// the other instances.
    fn publish_pending_queue(
        &self,
        session_id: &str,
        events_tx: &broadcast::Sender<ChatEvent>,
        messages: Vec<super::types::PendingQueueEntry>,
    ) {
        let event = ChatEvent::PendingQueue { messages };
        let _ = events_tx.send(event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, event);
        }
    }

    /// The held messages of a session active on THIS instance (`None` when it
    /// is not — idle, or running elsewhere).
    pub async fn pending_queue_snapshot(
        &self,
        session_id: &str,
    ) -> Option<Vec<super::types::PendingQueueEntry>> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return Some(handle.queue_snapshot().await);
        }
        let sessions = self.active_sessions.read().await;
        let session = sessions.get(session_id)?;
        let queue = session.pending_messages.lock().await;
        Some(super::pending_queue::snapshot(&queue))
    }

    /// Edit, drop, move to the front or send now one held message — or just
    /// publish the list again.
    ///
    /// Returns whether the operation reached a session: `false` means no
    /// instance holds this session any more, so there is no queue to act on.
    /// An id that no longer names a held message is not an error (it left in
    /// the meantime): the list is published again and the client catches up.
    pub async fn pending_queue_op(
        &self,
        session_id: &str,
        op: &super::pending_queue::QueueOp,
    ) -> Result<bool> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            handle.queue_op(op).await;
            return Ok(true);
        }
        {
            let sessions = self.active_sessions.read().await;
            if let Some(session) = sessions.get(session_id) {
                let (outcome, messages) = {
                    let mut queue = session.pending_messages.lock().await;
                    let outcome = super::pending_queue::apply(&mut queue, op);
                    (outcome, super::pending_queue::snapshot(&queue))
                };
                self.publish_pending_queue(session_id, &session.events_tx, messages);
                if outcome.interrupt && session.is_streaming.load(Ordering::SeqCst) {
                    // Same three steps as a message sent mid-stream.
                    session.interrupt_flag.store(true, Ordering::SeqCst);
                    session.interrupt_token.cancel();
                    if let Some(ref tx) = session.stdin_tx {
                        let _ = tx.try_send(InteractiveClient::build_interrupt_json());
                    }
                }
                return Ok(true);
            }
        }
        // Not here: the session may run on another instance.
        let payload = serde_json::to_string(op)?;
        self.try_remote_send(session_id, &payload, "queue_op").await
    }

    /// `route_user_message` for a message that must WAIT for the running turn:
    /// same three routes (this instance, another instance, resume), but a
    /// streaming session holds the message instead of being interrupted.
    /// A session that is not streaming — or not active anywhere — has no turn
    /// to wait for: the message is sent, or the session resumed, as usual.
    pub async fn route_queued_user_message(
        &self,
        session_id: &str,
        content: &str,
        claims: Option<&crate::auth::jwt::Claims>,
    ) -> std::result::Result<DeliveryRoute, MessageDeliveryError> {
        if self.is_session_active(session_id).await {
            match self.queue_user_message(session_id, content).await {
                Ok(_) => return Ok(DeliveryRoute::Local),
                Err(send_err) => {
                    warn!(
                        session_id = %session_id,
                        error = %send_err,
                        "queue_user_message failed, attempting resume_session as fallback"
                    );
                    return match self.resume_session(session_id, content, claims).await {
                        Ok(()) => Ok(DeliveryRoute::ResumedAfterSendFailure),
                        Err(resume) => Err(MessageDeliveryError::SendAndResume {
                            send: send_err,
                            resume,
                        }),
                    };
                }
            }
        }
        if self
            .try_remote_send(session_id, content, "queued_user_message")
            .await
            .unwrap_or(false)
        {
            return Ok(DeliveryRoute::Remote);
        }
        self.resume_session(session_id, content, claims)
            .await
            .map(|()| DeliveryRoute::Resumed)
            .map_err(MessageDeliveryError::Resume)
    }

    /// Send a permission response (allow/deny) to the Claude CLI subprocess.
    ///
    /// Unlike `send_message`, this does NOT:
    /// - Persist as a user_message event
    /// - Broadcast to WebSocket subscribers
    /// - Trigger a new stream_response
    ///
    /// It sends a JSON control response directly to the CLI subprocess via the
    /// cloned `stdin_tx` sender — **without** taking the `client` Mutex lock.
    ///
    /// This is critical because `stream_response` holds the client lock for the
    /// entire duration of streaming. If we tried to lock the client here, we'd
    /// deadlock: stream_response waits for the permission response to continue,
    /// but we'd wait for stream_response to release the lock.
    ///
    /// The control response format uses the SDK control protocol envelope,
    /// with the inner `response` matching the SDK's `internal_query.rs` format:
    /// ```json
    /// {
    ///   "type": "control_response",
    ///   "response": {
    ///     "subtype": "success",
    ///     "request_id": "<requestId from the control_request>",
    ///     "response": { "allow": true }
    ///   }
    /// }
    /// ```
    /// For deny:
    /// ```json
    /// {
    ///   "type": "control_response",
    ///   "response": {
    ///     "subtype": "success",
    ///     "request_id": "<requestId from the control_request>",
    ///     "response": { "allow": false, "reason": "User denied" }
    ///   }
    /// }
    /// ```
    ///
    /// **IMPORTANT**: Do NOT include `"updatedInput": {}` — the CLI uses that field
    /// to REPLACE the original tool input. An empty `{}` erases the command/input,
    /// causing `"undefined is not an object"` when the CLI tries to execute.
    pub async fn send_permission_response(
        &self,
        session_id: &str,
        request_id: &str,
        allow: bool,
    ) -> Result<()> {
        self.send_permission_response_inner(session_id, request_id, allow, false)
            .await
            .map_err(|e| match e {
                PermissionDeliveryError::Failed(e) => e,
                other => anyhow!(other.to_string()),
            })
    }

    /// Shared body of the permission answer. With `require_pending`, the
    /// request must still be waiting in the session's pending map: the entry
    /// is CLAIMED atomically (removed under the map lock), so two concurrent
    /// answers to the same request cannot both reach the CLI — the second one
    /// gets [`PermissionDeliveryError::NotPending`]. The WS path passes
    /// `false` and keeps its historical lenient behaviour.
    async fn send_permission_response_inner(
        &self,
        session_id: &str,
        request_id: &str,
        allow: bool,
        require_pending: bool,
    ) -> std::result::Result<(), PermissionDeliveryError> {
        let (stdin_tx, pending_perm_inputs, events_tx, session_uuid, next_seq) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| PermissionDeliveryError::SessionDead(session_id.to_string()))?;
            session.last_activity = Instant::now();
            let tx = session.stdin_tx.clone().ok_or_else(|| {
                PermissionDeliveryError::Failed(anyhow!(
                    "No stdin sender for session {} (CLI may not be connected)",
                    session_id
                ))
            })?;
            (
                tx,
                session.pending_permission_inputs.clone(),
                session.events_tx.clone(),
                uuid::Uuid::parse_str(session_id).ok(),
                session.next_seq.clone(),
            )
        };

        // Retrieve the original tool input stored when the permission_request arrived.
        // This is needed because the CLI's Zod schema for permission responses requires:
        //   Allow: { behavior: "allow", updatedInput: <record> }
        //   Deny:  { behavior: "deny",  message: <string> }
        // The `updatedInput` field REPLACES the original tool input in the CLI, so we
        // MUST pass back the original input — an empty {} would erase command/file_path/etc.
        let claimed = pending_perm_inputs.lock().await.remove(request_id);
        if require_pending && claimed.is_none() {
            return Err(PermissionDeliveryError::NotPending);
        }
        let was_claimed = claimed.is_some();
        let original_input = claimed.unwrap_or_else(|| serde_json::json!({}));
        // The entry is only really consumed once the decision reached the CLI:
        // on a failed send it is put back so a retry is still `pending`.
        let restore_claim = |input: serde_json::Value| {
            let pending = pending_perm_inputs.clone();
            let request_id = request_id.to_string();
            async move {
                if was_claimed {
                    pending.lock().await.insert(request_id, input);
                }
            }
        };

        let permission_response = if allow {
            serde_json::json!({
                "behavior": "allow",
                "updatedInput": original_input
            })
        } else {
            serde_json::json!({
                "behavior": "deny",
                "message": "User denied the permission request"
            })
        };

        // Wrap in the control_response envelope with subtype + request_id:
        let control_response = serde_json::json!({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": permission_response
            }
        });

        let json = match serde_json::to_string(&control_response) {
            Ok(j) => j,
            Err(e) => {
                restore_claim(original_input).await;
                return Err(PermissionDeliveryError::Failed(anyhow!(
                    "Failed to serialize control response: {}",
                    e
                )));
            }
        };

        info!(
            session_id = %session_id,
            request_id = %request_id,
            allow,
            "Sending permission control response to CLI (via stdin_tx, lock-free)"
        );

        if let Err(e) = stdin_tx.send(json).await {
            restore_claim(original_input).await;
            return Err(PermissionDeliveryError::Failed(anyhow!(
                "Failed to send permission control response: {}",
                e
            )));
        }

        // Persist and broadcast the permission decision so it survives session reload.
        let decision_event = ChatEvent::PermissionDecision {
            id: request_id.to_string(),
            allow,
        };

        // Broadcast to all connected WebSocket clients
        let _ = events_tx.send(decision_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, decision_event.clone());
        }
        self.notify_attention(session_id, AttentionReason::PermissionDecision);

        // Persist to Neo4j
        if let Some(uuid) = session_uuid {
            let seq = next_seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let record = crate::neo4j::models::ChatEventRecord {
                id: uuid::Uuid::new_v4(),
                session_id: uuid,
                seq,
                event_type: "permission_decision".to_string(),
                data: serde_json::to_string(&decision_event).unwrap_or_default(),
                created_at: chrono::Utc::now(),
            };
            if let Err(e) = self.graph.store_chat_events(uuid, vec![record]).await {
                warn!(
                    session_id = %session_id,
                    request_id = %request_id,
                    error = %e,
                    "Failed to persist permission_decision event"
                );
            }
        }

        Ok(())
    }

    /// Answer a permission request: local CLI first, then the instance that
    /// owns the session (NATS RPC). The ONE routing used by the WS
    /// `permission_response` frame and by the REST route.
    ///
    /// `require_pending` makes the answer strict for a LOCAL session (see
    /// [`Self::send_permission_response_inner`]); a remote owner does its own
    /// bookkeeping. No instance holds the session -> `SessionDead`.
    pub async fn route_permission_response(
        &self,
        session_id: &str,
        request_id: &str,
        allow: bool,
        require_pending: bool,
    ) -> std::result::Result<DeliveryRoute, PermissionDeliveryError> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle
                .answer_permission(request_id, allow)
                .await
                .map(|()| DeliveryRoute::Local)
                .map_err(PermissionDeliveryError::Failed);
        }
        if self.is_session_active(session_id).await {
            return self
                .send_permission_response_inner(session_id, request_id, allow, require_pending)
                .await
                .map(|()| DeliveryRoute::Local);
        }
        // The message_type "control_response" tells the receiving instance to
        // use send_permission_response instead of send_message.
        // `request_id`: what an agent-engine owner answers with (the CLI does not need it).
        let payload = serde_json::json!({ "allow": allow, "request_id": request_id }).to_string();
        if self
            .try_remote_send(session_id, &payload, "control_response")
            .await
            .unwrap_or(false)
        {
            Ok(DeliveryRoute::Remote)
        } else {
            Err(PermissionDeliveryError::SessionDead(session_id.to_string()))
        }
    }

    /// Whether a freshly spawned CLI is still there after `grace`. A CLI given a
    /// `--resume` target it does not know prints an error and exits, but not
    /// instantly: the probe must outlast that.
    async fn stays_alive(client: &InteractiveClient, grace: std::time::Duration) -> bool {
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            if !client.is_alive().await {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        client.is_alive().await
    }

    /// Removes a legacy session from `active_sessions` when its CLI process is
    /// gone (stdout at EOF). A session being streamed holds the client lock and
    /// is left alone: its own stream reports a death.
    pub(crate) async fn evict_if_cli_dead(&self, session_id: &str) -> bool {
        let client = match self.active_sessions.read().await.get(session_id) {
            Some(s) => s.client.clone(),
            None => return false,
        };
        let dead = match client.try_lock() {
            Ok(c) => !c.is_alive().await,
            Err(_) => false,
        };
        if dead {
            warn!(
                session_id = %session_id,
                "CLI process is gone, dropping the session from active_sessions so the message resumes it"
            );
            self.active_sessions.write().await.remove(session_id);
        }
        dead
    }

    /// Deliver a user message: local CLI -> owning instance (NATS) ->
    /// `resume_session` (respawns the CLI, keeping the session's identity and
    /// links). A failed local send falls back to `resume_session` too (dead
    /// CLI). The ONE routing used by the WS `user_message` frame and by the
    /// REST route; it is also the only way to answer an orphan question.
    pub async fn route_user_message(
        &self,
        session_id: &str,
        content: &str,
        claims: Option<&crate::auth::jwt::Claims>,
    ) -> std::result::Result<DeliveryRoute, MessageDeliveryError> {
        // A registered session whose CLI has left would swallow the message (the
        // write goes to a closed stdin and the turn ends empty): drop it so the
        // message takes the resume path below.
        self.evict_if_cli_dead(session_id).await;
        if self.is_session_active(session_id).await {
            match self.send_message(session_id, content).await {
                Ok(()) => return Ok(DeliveryRoute::Local),
                Err(send_err) => {
                    warn!(
                        session_id = %session_id,
                        error = %send_err,
                        "send_message failed, attempting resume_session as fallback"
                    );
                    return match self.resume_session(session_id, content, claims).await {
                        Ok(()) => Ok(DeliveryRoute::ResumedAfterSendFailure),
                        Err(resume) => Err(MessageDeliveryError::SendAndResume {
                            send: send_err,
                            resume,
                        }),
                    };
                }
            }
        }
        if self
            .try_remote_send(session_id, content, "user_message")
            .await
            .unwrap_or(false)
        {
            return Ok(DeliveryRoute::Remote);
        }
        self.resume_session(session_id, content, claims)
            .await
            .map(|()| DeliveryRoute::Resumed)
            .map_err(MessageDeliveryError::Resume)
    }

    /// Change the permission mode of an active CLI session mid-conversation.
    ///
    /// Sends a `set_permission_mode` control request to the Claude CLI subprocess,
    /// updates the in-memory `ActiveSession`, and persists the change to Neo4j.
    ///
    /// **IMPORTANT**: This method sends the control request via the cloned `stdin_tx`
    /// sender — **without** taking the `client` Mutex lock. This is critical because
    /// `stream_response` holds the client lock for the entire duration of streaming.
    /// If we tried to lock the client here, the WS event loop would deadlock:
    /// it awaits the lock while stream_response holds it, so broadcast events
    /// can no longer be forwarded to the frontend.
    pub async fn set_session_permission_mode(&self, session_id: &str, mode: &str) -> Result<()> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            let pair = super::provider::policy::parse_mode(mode)
                .ok_or_else(|| anyhow!("Unknown permission mode '{mode}'"))?;
            // Refused for a mode the provider's ceiling does not allow: the
            // session's token was signed with that ceiling at open.
            let native = super::provider::policy::to_legacy(mode);
            handle.set_policy_mode(pair.neutral, native).await?;
            // A third party's `full` profile holds in trust only (H6): its token
            // cannot change, the REST boundary reads this.
            crate::auth::agent_tokens::set_out_of_trust(
                session_id,
                pair.neutral != nexus_claude::agent::PolicyMode::Trust,
            );
            return Ok(());
        }
        // Validate mode: the Claude strings or the neutral names (A43). The CLI
        // only understands its own strings, so a neutral name is translated.
        let Some(mode) = super::provider::policy::to_legacy(mode) else {
            bail!(
                "Invalid permission mode '{}'. Valid modes: {}",
                mode,
                super::config::PermissionConfig::valid_modes().join(", ")
            );
        };

        // Get session state and stdin_tx — do NOT extract client (avoids Mutex deadlock)
        let (stdin_tx, old_mode, events_tx) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            session.last_activity = Instant::now();
            let old_mode = session.permission_mode.clone();
            session.permission_mode = Some(mode.to_string());
            let tx = session.stdin_tx.clone().ok_or_else(|| {
                anyhow!(
                    "No stdin sender for session {} (CLI may not be connected)",
                    session_id
                )
            })?;
            (tx, old_mode, session.events_tx.clone())
        };

        // Build the control_request JSON — same format as InteractiveClient::set_permission_mode
        let control_request = serde_json::json!({
            "type": "control_request",
            "request_id": Uuid::new_v4().to_string(),
            "request": {
                "subtype": "set_permission_mode",
                "mode": mode
            }
        });

        let json = serde_json::to_string(&control_request)
            .map_err(|e| anyhow!("Failed to serialize set_permission_mode request: {}", e))?;

        // Send via stdin_tx (lock-free — bypasses the client Mutex entirely)
        stdin_tx
            .send(json)
            .await
            .map_err(|e| anyhow!("Failed to send set_permission_mode to CLI: {}", e))?;

        info!(
            session_id = %session_id,
            old_mode = ?old_mode,
            new_mode = %mode,
            "Permission mode changed for session (via stdin_tx, lock-free)"
        );

        // Persist to Neo4j
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Err(e) = self
                .graph
                .update_chat_session_permission_mode(uuid, mode)
                .await
            {
                warn!(
                    session_id = %session_id,
                    error = %e,
                    "Failed to persist permission mode change to Neo4j (non-fatal)"
                );
            }
        }

        // Broadcast event to WebSocket clients
        let _ = events_tx.send(ChatEvent::PermissionModeChanged {
            mode: mode.to_string(),
            policy_mode: None,
        });

        Ok(())
    }

    /// Moves a conversation to ANOTHER provider (B-SW).
    ///
    /// Providers do not share a session format, so this opens a NEW session on
    /// `provider`, in the same project and directory, and closes the old one only
    /// once the new one is open. The new session is sent the earlier conversation
    /// as a relay (see [`super::relay`]) in front of `message`; the conversation
    /// itself shows `message` alone. The target is resolved like any explicit
    /// choice: the project's consent, the endpoint guard and the security gate all
    /// apply, and a refusal leaves the old session untouched.
    ///
    /// Moving to the provider the session is already on is refused: that is
    /// `set_session_model`'s job.
    pub async fn switch_session_provider(
        &self,
        session_id: &str,
        provider: &str,
        model: Option<&str>,
        message: &str,
        claims: Option<crate::auth::jwt::Claims>,
    ) -> Result<super::types::SwitchProviderResponse> {
        self.relay_conversation(
            session_id,
            provider,
            model,
            message,
            claims,
            super::relay::MOVED_BY_USER,
        )
        .await
    }

    /// [`Self::switch_session_provider`], saying who moves the conversation. A move by the
    /// router (`moved_by: auto`) keeps the conversation's routing (its mode, its pool) on
    /// the new session, whose pair is then not imposed: the router goes on choosing there.
    /// A move by the user imposes its target (`routed_by: request`).
    pub(crate) async fn relay_conversation(
        &self,
        session_id: &str,
        provider: &str,
        model: Option<&str>,
        message: &str,
        claims: Option<crate::auth::jwt::Claims>,
        moved_by: &str,
    ) -> Result<super::types::SwitchProviderResponse> {
        use super::relay;
        use super::types::SwitchProviderError;

        let previous = Uuid::parse_str(session_id)
            .map_err(|_| anyhow::Error::new(SwitchProviderError::InvalidSession))?;
        if message.trim().is_empty() {
            return Err(anyhow::Error::new(SwitchProviderError::EmptyMessage));
        }
        let node = self
            .graph
            .get_chat_session(previous)
            .await?
            .ok_or_else(|| anyhow::Error::new(SwitchProviderError::NotFound))?;
        let current = node
            .provider_id
            .clone()
            .unwrap_or_else(|| super::provider::resolver::CLAUDE_CODE.to_string());
        if current == provider {
            return Err(anyhow::Error::new(SwitchProviderError::SameProvider(
                current,
            )));
        }

        // The conversation so far, from what was stored.
        let records = self.graph.get_chat_events(previous, 0, 5_000).await?;
        let events: Vec<ChatEvent> = records
            .iter()
            .filter_map(|record| serde_json::from_str(&record.data).ok())
            .collect();
        // A relay may use 40 % of the target's window when it is known.
        let window = match self.provider_for(provider).await {
            Ok(target) => target
                .capabilities(model.or(node.model.as_str().into()))
                .context_window
                .map(|w| w.value),
            Err(_) => None,
        };
        let relayed = relay::RelayedFrom {
            relay: relay::render_relay(
                &events,
                &current,
                provider,
                relay::budget_for_window(window),
            ),
            from_session_id: session_id.to_string(),
            from_provider: current.clone(),
            moved_by: moved_by.to_string(),
            // The memory conversation goes on in the new session.
            conversation_id: node.conversation_id.clone(),
        };
        let rendered = &relayed.relay;

        // A neutral session continues in a neutral place of its own (a new session, a new directory).
        let relay_cwd = if node.execution_place == super::neutral_place::ExecutionPlace::Neutral {
            String::new()
        } else {
            node.cwd.clone()
        };
        let by_router = moved_by == relay::MOVED_BY_AUTO;
        let request = ChatRequest {
            routing_pool: node
                .routing_pool
                .as_deref()
                .filter(|_| by_router)
                .and_then(|json| serde_json::from_str(json).ok()),
            routing_mode: node
                .routing_mode
                .as_deref()
                .filter(|_| by_router)
                .and_then(|m| serde_json::from_value(serde_json::Value::String(m.to_owned())).ok()),
            attachments: Vec::new(),
            refs: Vec::new(),
            message: message.to_string(),
            session_id: None,
            cwd: relay_cwd,
            project_slug: node.project_slug.clone(),
            model: model.map(str::to_string),
            provider: Some(provider.to_string()),
            task_alias: None,
            persona_alias: None,
            run_provider: None,
            run_model: None,
            max_tokens: None,
            task_class: None,
            permission_mode: node.permission_mode.clone(),
            add_dirs: node.add_dirs.clone(),
            workspace_slug: node.workspace_slug.clone(),
            user_claims: claims,
            spawned_by: None,
            task_context: None,
            scaffolding_override: None,
            // A switch never widens: the new session keeps the access of the old one.
            access: Some(node.access),
            runner_context: None,
            routing_decision_id: None,
        };
        // Opening can be refused (consent, endpoint, gate, no model): the old
        // session is then left exactly as it was.
        let created = self
            .create_session_relayed(&request, Some(&relayed))
            .await?;

        // The thread it leaves learns where the conversation went, before it closes.
        self.emit_on_session(session_id, relayed.event(&created.session_id, provider))
            .await;

        let note = serde_json::json!({
            "from_session": session_id,
            "from_provider": current,
            "relayed_entries": rendered.included,
            "omitted_entries": rendered.omitted,
        });
        if let Err(error) = self
            .graph
            .put_llm_setting(
                &format!("handoff:{}", created.session_id),
                "note",
                &note.to_string(),
            )
            .await
        {
            warn!(session_id = %created.session_id, %error, "recording the provider handoff failed (non-fatal)");
        }
        // The conversation lives on in the new session; the old one ends.
        if let Err(error) = self.close_session(session_id).await {
            debug!(%session_id, %error, "closing the previous session after a provider switch");
        }
        Ok(super::types::SwitchProviderResponse {
            session_id: created.session_id,
            stream_url: created.stream_url,
            previous_session_id: session_id.to_string(),
            relayed_entries: rendered.included,
            omitted_entries: rendered.omitted,
            conversation_id: relayed.conversation_id.clone(),
        })
    }

    /// Persists and broadcasts `event` on the thread of a session this instance may or
    /// may not run live: through the agent handle when it owns it, through the legacy
    /// session's channel when it is active, else on the stored thread alone (a dormant
    /// session still has a thread, replayed when it is opened again).
    pub(crate) async fn emit_on_session(&self, session_id: &str, event: ChatEvent) {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            handle.emit(event).await;
            return;
        }
        let Ok(uuid) = Uuid::parse_str(session_id) else {
            return;
        };
        let live = {
            let sessions = self.active_sessions.read().await;
            sessions
                .get(session_id)
                .map(|s| (Arc::clone(&s.next_seq), s.events_tx.clone()))
        };
        let seq = match &live {
            Some((next_seq, _)) => next_seq.fetch_add(1, Ordering::SeqCst),
            None => {
                self.graph
                    .get_latest_chat_event_seq(uuid)
                    .await
                    .unwrap_or(0)
                    + 1
            }
        };
        let record = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq,
            event_type: event.event_type().to_string(),
            data: serde_json::to_string(&event).unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        if let Err(error) = self.graph.store_chat_events(uuid, vec![record]).await {
            warn!(%session_id, %error, event = event.event_type(), "event not stored on the thread");
        }
        if let Some((_, events_tx)) = live {
            let _ = events_tx.send(event.clone());
        }
        if let Some(nats) = &self.nats {
            nats.publish_chat_event(session_id, event);
        }
    }

    /// Change the model of an active CLI session mid-conversation.
    ///
    /// Sends a `set_model` control request to the Claude CLI subprocess,
    /// updates the in-memory `ActiveSession`, and broadcasts the change.
    ///
    /// **IMPORTANT**: This method sends the control request via the cloned `stdin_tx`
    /// sender — **without** taking the `client` Mutex lock. This is critical because
    /// `stream_response` holds the client lock for the entire duration of streaming.
    /// If we tried to lock the client here, the WS event loop would deadlock.
    ///
    /// Returns `true` when `ChatEvent::ModelChanged` was broadcast to the
    /// session's subscribers — EVERY connected client of the session, the one
    /// that asked included, already receives the confirmation that way. The
    /// caller must then send no confirmation of its own: the asking device used
    /// to get it twice (one direct, one broadcast) and show "Model changed" twice.
    /// Returns `false` for a dormant session, which has no subscribers: the
    /// caller confirms to the asker directly.
    pub async fn set_session_model(&self, session_id: &str, model: &str) -> Result<bool> {
        self.set_session_model_inner(session_id, model, true).await
    }

    /// Changes the routing of ONE existing conversation (`PUT /api/chat/sessions/{id}/routing`),
    /// from its next turn on. Stored on the session node only: never in the global or project
    /// settings. A refusal is a [`super::types::SessionRoutingError`] and changes nothing.
    ///
    /// * `auto`: PO routes (`full`); a model imposed before is released (`routed_by` `auto`).
    /// * one model ticked: strict (`primary`); that model is imposed (set now if it differs,
    ///   `routed_by` `request`) and never substituted.
    /// * two or more: `mixed`; the per-turn decision chooses among the ticked models of the
    ///   session's provider only (one session = one provider; another provider's ticks are
    ///   stored, and only `switch-provider` moves the conversation there).
    pub async fn set_session_routing(
        &self,
        session_id: &str,
        body: &super::types::SessionRoutingRequest,
    ) -> Result<ChatSessionNode> {
        use super::provider::cognitive::ProviderRoutingMode as Mode;
        use super::types::SessionRoutingError as Refused;
        let uuid =
            Uuid::parse_str(session_id).map_err(|_| anyhow::Error::new(Refused::NotFound))?;
        let node = self
            .graph
            .get_chat_session(uuid)
            .await?
            .ok_or_else(|| anyhow::Error::new(Refused::NotFound))?;
        let provider = node
            .provider_id
            .clone()
            .unwrap_or_else(|| super::provider::resolver::CLAUDE_CODE.to_owned());
        let router = self.turn_routing.get(session_id);
        let (mode, pool, routed_by) = if body.auto {
            self.unpin_session_model(session_id).await;
            if let Some(router) = &router {
                router.reroute(Mode::Full, None, false);
            }
            (Mode::Full, None, "auto")
        } else {
            let pool = super::types::settle_routing_pool(&body.routing_pool)
                .map_err(|index| anyhow::Error::new(Refused::BlankEntry(index)))?;
            match pool.as_slice() {
                [] => return Err(anyhow::Error::new(Refused::EmptyPool)),
                [only] => {
                    if only.provider != provider {
                        return Err(anyhow::Error::new(Refused::OtherProvider(provider)));
                    }
                    if node.model != only.model {
                        // Pins it, ends the automatic changes and tells the clients.
                        self.set_session_model(session_id, &only.model).await?;
                    } else {
                        self.pin_session_model(session_id).await;
                    }
                    if let Some(router) = &router {
                        router.reroute(Mode::Primary, None, true);
                    }
                    (Mode::Primary, None, "request")
                }
                _ => {
                    let allowed = Self::allowed_models_of(Some(&pool), &provider)
                        .filter(|models| !models.is_empty())
                        .ok_or_else(|| anyhow::Error::new(Refused::OtherProvider(provider)))?;
                    self.unpin_session_model(session_id).await;
                    if let Some(router) = &router {
                        router.reroute(Mode::Mixed, Some(allowed), false);
                    }
                    (Mode::Mixed, Some(serde_json::to_string(&pool)?), "auto")
                }
            }
        };
        self.graph
            .update_chat_session_routing(
                uuid,
                Some(mode.as_str()),
                pool.as_deref(),
                Some(routed_by),
            )
            .await?;
        self.graph
            .get_chat_session(uuid)
            .await?
            .ok_or_else(|| anyhow::Error::new(Refused::NotFound))
    }

    /// The model of the session is the router's again (Auto, or models ticked).
    async fn unpin_session_model(&self, session_id: &str) {
        if let Err(error) = self
            .graph
            .delete_llm_setting(&Self::pin_scope(session_id), Self::MODEL_PIN_KEY)
            .await
        {
            warn!(session_id, %error, "model pin not removed: a resume may keep it pinned");
        }
    }

    /// `manual`: the user asked for it, which ends the automatic model routing of the
    /// session. The router's own changes pass `false`.
    async fn set_session_model_inner(
        &self,
        session_id: &str,
        model: &str,
        manual: bool,
    ) -> Result<bool> {
        if manual {
            if let Some(router) = self.turn_routing.get(session_id) {
                router.mark_manual();
            }
            self.pin_session_model(session_id).await;
            self.flag_routing_override(session_id).await;
        }
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            handle.set_model(model).await?;
            if let Ok(uuid) = Uuid::parse_str(session_id) {
                let _ = self.graph.update_chat_session_model(uuid, model).await;
            }
            return Ok(true);
        }
        // Capture the live session's stdin_tx + events_tx IF the CLI subprocess is still
        // attached. A dormant session (idle-cleaned) won't be present in `active_sessions`
        // — that is NOT an error: we still persist the chosen model to Neo4j below so the
        // next resume/spawn launches with it. Do NOT extract `client` (avoids Mutex deadlock).
        let live = {
            let mut sessions = self.active_sessions.write().await;
            match sessions.get_mut(session_id) {
                Some(session) => {
                    session.last_activity = Instant::now();
                    let old_model = session.model.clone();
                    session.model = Some(model.to_string());
                    Some((
                        session.stdin_tx.clone(),
                        old_model,
                        session.events_tx.clone(),
                    ))
                }
                None => None,
            }
        };

        // Persist to Neo4j ALWAYS. This is the core fix: previously the model was updated
        // in-memory only, so it was silently lost on idle-cleanup → resume respawned on the
        // create-time model. `resume_session` reads `s.model` back from Neo4j (manager line
        // ~5271 → build_options.model), so persisting here makes the switch durable.
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Err(e) = self.graph.update_chat_session_model(uuid, model).await {
                warn!(
                    session_id = %session_id,
                    error = %e,
                    "Failed to persist model change to Neo4j (non-fatal)"
                );
            }
        }

        match live {
            // Active session with a live CLI: switch the running model mid-conversation.
            Some((Some(stdin_tx), old_model, events_tx)) => {
                // Build the control_request JSON — same format as InteractiveClient::set_model
                let control_request = serde_json::json!({
                    "type": "control_request",
                    "request_id": Uuid::new_v4().to_string(),
                    "request": {
                        "subtype": "set_model",
                        "model": model
                    }
                });

                let json = serde_json::to_string(&control_request)
                    .map_err(|e| anyhow!("Failed to serialize set_model request: {}", e))?;

                // Send via stdin_tx (lock-free — bypasses the client Mutex entirely)
                stdin_tx
                    .send(json)
                    .await
                    .map_err(|e| anyhow!("Failed to send set_model to CLI: {}", e))?;

                info!(
                    session_id = %session_id,
                    old_model = ?old_model,
                    new_model = %model,
                    "Model changed for active session (via stdin_tx, lock-free)"
                );

                let _ = events_tx.send(ChatEvent::ModelChanged {
                    model: model.to_string(),
                });
                Ok(true)
            }
            // Session present but CLI not connected yet — persisted; applies on next spawn.
            Some((None, _, events_tx)) => {
                info!(
                    session_id = %session_id,
                    new_model = %model,
                    "Model persisted (no live CLI stdin); applies on next spawn"
                );
                let _ = events_tx.send(ChatEvent::ModelChanged {
                    model: model.to_string(),
                });
                Ok(true)
            }
            // Dormant session (idle-cleaned) — persisted to Neo4j only; applies on resume.
            None => {
                info!(
                    session_id = %session_id,
                    new_model = %model,
                    "Model persisted for dormant session; applies on resume"
                );
                Ok(false)
            }
        }
    }

    /// Toggle auto-continue for an active session.
    ///
    /// Updates the in-memory `ActiveSession.auto_continue` AtomicBool,
    /// persists the change to Neo4j, and broadcasts a `ChatEvent::AutoContinueStateChanged`
    /// so that all connected frontends can sync their toggle UI.
    ///
    /// Unlike `set_session_permission_mode`, this does NOT send a control request to the
    /// CLI subprocess — auto-continue is purely a backend-side concern.
    ///
    /// Works whether the session is active (CLI spawned) or idle (just viewing).
    /// - Active: updates in-memory AtomicBool + broadcasts locally + NATS
    /// - Idle: persists to Neo4j + broadcasts via NATS only (no local broadcast channel)
    pub async fn set_auto_continue(&self, session_id: &str, enabled: bool) -> Result<()> {
        // Try to update in-memory state if session is active
        let maybe_events_tx = {
            if let Some(handle) = self.agent_runtime.get(session_id).await {
                handle.auto_continue.store(enabled, Ordering::Relaxed);
                Some(handle.events_tx.clone())
            } else {
                let sessions = self.active_sessions.read().await;
                if let Some(session) = sessions.get(session_id) {
                    session
                        .auto_continue
                        .store(enabled, std::sync::atomic::Ordering::Relaxed);
                    Some(session.events_tx.clone())
                } else {
                    None
                }
            }
        };

        // Persist to Neo4j (always — whether active or idle)
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Err(e) = self.graph.set_session_auto_continue(uuid, enabled).await {
                warn!(
                    session_id = %session_id,
                    error = %e,
                    "Failed to persist auto_continue change to Neo4j (non-fatal)"
                );
            }
        }

        let is_local = maybe_events_tx.is_some();
        info!(
            session_id = %session_id,
            enabled = %enabled,
            is_local = %is_local,
            "Auto-continue toggled for session"
        );

        // Broadcast event to WebSocket clients
        let event = ChatEvent::AutoContinueStateChanged {
            session_id: session_id.to_string(),
            enabled,
        };

        // Local broadcast (only if session is active on this instance)
        if let Some(events_tx) = maybe_events_tx {
            let _ = events_tx.send(event.clone());
        }

        // NATS broadcast (always — for cross-instance AND idle-session propagation)
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, event);
        }

        Ok(())
    }

    /// Get the current auto-continue state for a session.
    ///
    /// Reads from the in-memory ActiveSession if local, otherwise falls back to Neo4j.
    pub async fn get_auto_continue_state(&self, session_id: &str) -> Result<bool> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return Ok(handle.auto_continue.load(Ordering::Relaxed));
        }
        // Try local active session first
        let sessions = self.active_sessions.read().await;
        if let Some(session) = sessions.get(session_id) {
            return Ok(session
                .auto_continue
                .load(std::sync::atomic::Ordering::Relaxed));
        }
        drop(sessions);

        // Fallback to Neo4j
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            return self.graph.get_session_auto_continue(uuid).await;
        }

        Ok(self.config.auto_continue)
    }

    /// Resume a previously inactive session by creating a new InteractiveClient.
    ///
    /// The knowledge-graph hooks of a session, as the table the Claude CLI takes
    /// (`event → matchers`). The agent engine serves the SAME table to every
    /// provider through [`super::agent_hooks::GraphSessionHooks`]: one logic, two
    /// doors.
    pub(crate) fn graph_hook_table(&self, i: GraphHookInput) -> super::agent_hooks::HookTable {
        let GraphHookInput {
            session_id,
            context_source,
            work_log,
            tool_knowledge,
            announce,
        } = i;
        let mut hooks = std::collections::HashMap::new();

        // PreCompact → CompactionNotifier builds custom_instructions from the task /
        // project context, and (when `announce` carries a sender) broadcasts
        // ChatEvent::CompactionStarted. The agent engine announces compactions from the
        // provider's own `compaction` event, so it passes none: a second announcement
        // would show the spinner twice.
        let (events_tx, nats) = match announce {
            Some(tx) => (tx, self.nats.clone()),
            None => (broadcast::channel(1).0, None),
        };
        let notifier = CompactionNotifier::new(events_tx, nats, session_id.clone())
            .with_context(self.graph.clone(), context_source)
            .with_work_log(work_log);
        hooks.insert(
            "PreCompact".to_string(),
            vec![nexus_claude::HookMatcher {
                matcher: None,
                hooks: vec![std::sync::Arc::new(notifier)],
            }],
        );

        if tool_knowledge {
            // PreToolUse → SkillActivationHook injects skill context as additionalContext
            let session_project = self.hook_session_project(&session_id);
            Self::register_skill_hook(&mut hooks, self.graph.clone(), session_project.clone());

            // PostToolUse → PostToolUseRedirectHook suggests MCP alternatives after noisy Grep
            let post_hook = Self::redirect_hook(self.graph.clone(), session_project);
            hooks.insert(
                "PostToolUse".to_string(),
                vec![nexus_claude::HookMatcher {
                    matcher: None,
                    hooks: vec![std::sync::Arc::new(post_hook)],
                }],
            );
        }
        hooks
    }

    /// If the session has a `cli_session_id`, resumes with `--resume`.
    /// If not (first message or previous spawn failed), starts fresh without `--resume`.
    /// The resolved project the per-tool hooks of a session work for: only in mode
    /// `on` (`off` and `shadow` keep the historical project-from-tool-cwd).
    fn hook_session_project(
        &self,
        session_id: &str,
    ) -> Option<Arc<super::anchor_resolver::SessionProject>> {
        if self.anchor_mode != super::anchor_resolver::AnchorContextMode::On {
            return None;
        }
        let id = Uuid::parse_str(session_id).ok()?;
        Some(Arc::new(
            super::anchor_resolver::SessionProject::new(self.graph.clone(), id)
                .with_anchor_cache(self.anchor_cache.clone()),
        ))
    }

    fn redirect_hook(
        graph: Arc<dyn GraphStore>,
        session_project: Option<Arc<super::anchor_resolver::SessionProject>>,
    ) -> post_tool_hook::PostToolUseRedirectHook {
        let hook = post_tool_hook::PostToolUseRedirectHook::new(graph);
        match session_project {
            Some(sp) => hook.with_session_project(sp),
            None => hook,
        }
    }

    /// Register the PreToolUse knowledge hook, plus a PreCompact companion that
    /// resets its injection ledger — after a compaction, knowledge injected
    /// earlier is no longer in context and may be shown again.
    fn register_skill_hook(
        hooks: &mut HashMap<String, Vec<nexus_claude::HookMatcher>>,
        graph: Arc<dyn GraphStore>,
        session_project: Option<Arc<super::anchor_resolver::SessionProject>>,
    ) {
        let ledger = Arc::new(super::hook_ledger::HookLedger::new());
        let mut skill_hook = skill_hook::SkillActivationHook::with_ledger(graph, ledger.clone());
        if let Some(sp) = session_project {
            skill_hook = skill_hook.with_session_project(sp);
        }
        hooks.insert(
            "PreToolUse".to_string(),
            vec![nexus_claude::HookMatcher {
                matcher: None, // Match all tools — filtering is done inside the hook
                hooks: vec![Arc::new(skill_hook)],
            }],
        );
        hooks
            .entry("PreCompact".to_string())
            .or_default()
            .push(nexus_claude::HookMatcher {
                matcher: None,
                hooks: vec![Arc::new(super::hook_ledger::HookLedgerReset::new(ledger))],
            });
    }

    /// Refuses a request whose `provider` differs from the one the session was
    /// opened on (409 `provider_conflict`). A session that does not exist yet,
    /// or a request that names no provider, passes.
    pub async fn check_provider_binding(
        &self,
        session_id: &str,
        requested: Option<&str>,
    ) -> Result<()> {
        let Some(requested) = requested.filter(|p| !p.is_empty()) else {
            return Ok(());
        };
        let Ok(uuid) = Uuid::parse_str(session_id) else {
            return Ok(());
        };
        if let Some(node) = self.graph.get_chat_session(uuid).await? {
            super::provider::resolver::resolve_for_open(
                node.provider_id.as_deref(),
                true,
                Some(requested),
                &super::provider::resolver::BuiltinCatalog,
            )
            .map_err(anyhow::Error::new)?;
        }
        Ok(())
    }

    pub async fn resume_session(
        &self,
        session_id: &str,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
    ) -> Result<()> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;

        // Load session from Neo4j
        let mut session_node = self
            .graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {} not found in database", session_id))?;
        // Sessions stored without a slug (all-projects mode) resume with the
        // project inferred from their cwd, like create_session does.
        if session_node.project_slug.is_none() {
            session_node.project_slug = infer_session_project(
                self.graph.as_ref(),
                self.anchor_mode,
                Some(uuid),
                session_node.execution_place,
                &session_node.cwd,
            )
            .await;
        }

        // The provider is frozen at open (A16): a resume never re-resolves. The
        // legacy path only drives Claude Code, so a session bound to another
        // provider cannot be resumed here.
        let frozen = super::provider::resolver::resolve_for_open(
            session_node.provider_id.as_deref(),
            true,
            None,
            &super::provider::resolver::BuiltinCatalog,
        )
        .map_err(anyhow::Error::new)?;
        if frozen.provider_id != super::provider::resolver::CLAUDE_CODE
            && session_node.capabilities.is_none()
        {
            return Err(anyhow::Error::new(
                super::provider::resolver::ResolveError::Unavailable {
                    provider_id: frozen.provider_id,
                    role: super::provider::resolver::Role::Pilot,
                },
            ));
        }

        // A session opened by the agent engine carries its capability snapshot:
        // it resumes on that engine, and only when that engine is switched on.
        if session_node.capabilities.is_some() {
            // A third-party session always resumes on the agent engine. A Claude
            // Code session that was forced onto it needs the flag still on:
            // otherwise a typed, explicit error (not a mute `provider_unavailable`).
            if !self.engine_is_agent(&frozen.provider_id) {
                return Err(anyhow::Error::new(
                    super::provider::resolver::ResolveError::EngineUnavailable {
                        provider_id: frozen.provider_id,
                    },
                ));
            }
            return self
                .resume_agent_session(&session_node, message, user_claims)
                .await;
        }

        let cli_session_id = session_node.cli_session_id.as_deref();

        if let Some(cli_id) = cli_session_id {
            info!("Resuming session {} with CLI ID {}", session_id, cli_id);
        } else {
            info!(
                "Starting fresh CLI for session {} (no previous cli_session_id)",
                session_id
            );
        }

        // Build options - with resume flag only if we have a cli_session_id
        let (system_prompt, _included_note_ids) = self
            .build_system_prompt_anchored(
                session_node.execution_place,
                &session_node.cwd,
                session_node.project_slug.as_deref(),
                message,
                Some(&session_node.model),
                session_id,
                None,
            )
            .await;

        // Create broadcast channel early so CompactionNotifier can use the sender
        let (events_tx, _) = broadcast::channel(BROADCAST_BUFFER);

        // Create work_log early so CompactionNotifier can reference it
        let work_log = Arc::new(Mutex::new(SessionWorkLog::default()));

        // Build session hooks: PreCompact (compaction notifier) + PreToolUse (skill activation)
        let session_hooks = {
            let mut hooks = std::collections::HashMap::new();

            // PreCompact → CompactionNotifier broadcasts ChatEvent::CompactionStarted
            // + builds custom_instructions from session context
            let context_source = match session_node.project_slug.as_deref() {
                Some(slug) => CompactionContextSource::Session(slug.to_string()),
                None => CompactionContextSource::None,
            };
            let notifier = CompactionNotifier::new(
                events_tx.clone(),
                self.nats.clone(),
                session_id.to_string(),
            )
            .with_context(self.graph.clone(), context_source)
            .with_work_log(work_log.clone());
            hooks.insert(
                "PreCompact".to_string(),
                vec![nexus_claude::HookMatcher {
                    matcher: None,
                    hooks: vec![std::sync::Arc::new(notifier)],
                }],
            );

            // PreToolUse → SkillActivationHook injects skill context as additionalContext.
            // Same runner exclusion as create_session: resuming a runner session
            // used to re-enable the per-tool-call injection that creation
            // deliberately skips.
            let session_project = self.hook_session_project(session_id);
            if !is_runner_spawned(session_node.spawned_by.as_deref()) {
                Self::register_skill_hook(&mut hooks, self.graph.clone(), session_project.clone());
            }

            // PostToolUse → PostToolUseRedirectHook suggests MCP alternatives after noisy Grep
            let post_hook = Self::redirect_hook(self.graph.clone(), session_project);
            hooks.insert(
                "PostToolUse".to_string(),
                vec![nexus_claude::HookMatcher {
                    matcher: None,
                    hooks: vec![std::sync::Arc::new(post_hook)],
                }],
            );

            hooks
        };

        // A resume keeps the access the session was opened with and can only narrow it.
        let access = super::provider::policy::SessionAccess::for_resume(session_node.access, None);
        let resume_add_dirs = session_node.add_dirs.clone().unwrap_or_default();

        // `--resume <cli_session_id>` can name a conversation the CLI no longer
        // has (expired, purged, stale): the CLI then exits at once and the turn
        // used to end empty (task 1ff0e2e4). The PO session — identity, links,
        // context — lives in the graph, not in the CLI: when the resumed CLI is
        // not alive after its handshake, fall back to a fresh CLI for the SAME
        // PO session instead of handing back a dead one.
        let mut resume_with = cli_session_id;
        let client = loop {
            let options = self
                .build_options_with_access(
                    &session_node.cwd,
                    &session_node.model,
                    &system_prompt,
                    resume_with,
                    session_node.permission_mode.as_deref(),
                    Some(session_hooks.clone()),
                    &resume_add_dirs,
                    user_claims,
                    Some(session_id),
                    access,
                )
                .await;

            let mut candidate = InteractiveClient::new(options).map_err(|e| {
                super::provider::errors::sdk_open_error(
                    "Failed to create InteractiveClient for resume",
                    e,
                )
            })?;

            let connected = candidate.connect().await;
            if let Err(e) = connected {
                if resume_with.is_some() {
                    warn!(
                        session_id = %session_id,
                        error = %e,
                        "Resumed CLI failed to start, falling back to a fresh CLI (same PO session)"
                    );
                    resume_with = None;
                    continue;
                }
                return Err(super::provider::errors::sdk_open_error(
                    "Failed to connect resumed InteractiveClient",
                    e,
                ));
            }

            // Initialize hooks with the CLI (sends PreCompact, etc. registrations).
            // Must be called AFTER connect() and BEFORE take_sdk_control_receiver().
            // Graceful: warn on failure but don't abort the session.
            if let Err(e) = candidate.initialize_hooks().await {
                warn!(
                    session_id = %session_id,
                    "Failed to initialize hooks on resume (non-fatal): {}",
                    e
                );
            }

            if resume_with.is_some() && !Self::stays_alive(&candidate, RESUME_GRACE).await {
                warn!(
                    session_id = %session_id,
                    "Resumed CLI exited right after its handshake, falling back to a fresh CLI \
                     (same PO session)"
                );
                let _ = candidate.disconnect().await;
                resume_with = None;
                continue;
            }
            break candidate;
        };

        // Clone stdin sender for lock-free permission responses (see create_session).
        let stdin_tx = client.clone_stdin_sender().await;

        // Take the SDK control receiver ONCE at session resume and hand it to the
        // session-lifetime control pump (see create_session and `control_pump.rs`).
        let sdk_control_rx = client.take_sdk_control_receiver().await.map(|raw_rx| {
            super::control_pump::spawn(
                session_id.to_string(),
                raw_rx,
                client.hook_callbacks(),
                stdin_tx.clone(),
            )
        });
        let sdk_control_rx = Arc::new(tokio::sync::Mutex::new(sdk_control_rx));

        // CLI subprocess PID for descendant SIGINT cascade (T1+T2 of
        // plan 28e9afe3). Same as create_session.
        let child_pid: Option<u32> = client.child_pid().await;

        let client = Arc::new(Mutex::new(client));

        // Re-create ConversationMemoryManager for resumed session
        // (uses existing conversation_id if available)
        let memory_manager = if let Some(ref mem_config) = self.memory_config {
            let mm = if let Some(ref conv_id) = session_node.conversation_id {
                ConversationMemoryManager::new(mem_config.clone())
                    .with_conversation_id(conv_id.clone())
            } else {
                let mm = ConversationMemoryManager::new(mem_config.clone());
                // Persist the new conversation_id
                let _ = self
                    .graph
                    .update_chat_session(
                        uuid,
                        None,
                        None,
                        None,
                        None,
                        Some(mm.conversation_id().to_string()),
                        None,
                    )
                    .await;
                mm
            };
            Some(Arc::new(Mutex::new(mm)))
        } else {
            None
        };

        // Initialize next_seq from Neo4j (resume existing event history)
        let latest_seq = self
            .graph
            .get_latest_chat_event_seq(uuid)
            .await
            .unwrap_or(0);
        let next_seq = Arc::new(AtomicI64::new(latest_seq + 1));
        let pending_messages = Arc::new(Mutex::new(VecDeque::<PendingMessage>::new()));
        let is_streaming = Arc::new(AtomicBool::new(false));
        let streaming_text = Arc::new(Mutex::new(String::new()));
        let streaming_events = Arc::new(Mutex::new(Vec::new()));

        // Register as active — cancel old NATS listeners if session was previously active
        let nats_cancel = CancellationToken::new();
        let interrupt_token = CancellationToken::new();
        let auto_continue = Arc::new(AtomicBool::new(
            self.graph
                .get_session_auto_continue(uuid)
                .await
                .unwrap_or(self.config.auto_continue),
        ));
        let interrupt_flag = {
            let mut sessions = self.active_sessions.write().await;
            // Cancel stale NATS listeners from a previous resume/create of this session.
            // Without this, each resume_session() spawns 3 new NATS listeners (interrupt,
            // snapshot, RPC) that accumulate — the old ones never stop because the session
            // key still exists in the HashMap (insert replaces the value, not the key).
            // This causes N duplicate stream_response spawns per message, where N is the
            // number of times the session was resumed.
            if let Some(old_session) = sessions.get(session_id) {
                info!(
                    session_id = %session_id,
                    "Cancelling stale NATS listeners for session (resume replacing)"
                );
                old_session.nats_cancel.cancel();
            }
            let interrupt_flag = Arc::new(AtomicBool::new(false));
            sessions.insert(
                session_id.to_string(),
                ActiveSession {
                    anchor: self.anchor_session(),
                    events_tx: events_tx.clone(),
                    last_activity: Instant::now(),
                    cli_session_id: cli_session_id.map(|s| s.to_string()),
                    client: client.clone(),
                    interrupt_flag: interrupt_flag.clone(),
                    memory_manager: memory_manager.clone(),
                    next_seq: next_seq.clone(),
                    pending_messages: pending_messages.clone(),
                    is_streaming: is_streaming.clone(),
                    streaming_text: streaming_text.clone(),
                    streaming_events: streaming_events.clone(),
                    permission_mode: session_node.permission_mode.clone(),
                    model: Some(session_node.model.clone()),
                    sdk_control_rx: sdk_control_rx.clone(),
                    stdin_tx,
                    child_pid,
                    nats_cancel: nats_cancel.clone(),
                    interrupt_token: interrupt_token.clone(),
                    pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                        std::collections::HashMap::new(),
                    )),
                    auto_continue: auto_continue.clone(),
                    auto_continue_count: Arc::new(AtomicU32::new(0)),
                    max_auto_continues: 0, // resumed sessions = interactive, unlimited
                    rfc_accumulator: Arc::new(Mutex::new(
                        super::observation_detector::RfcAccumulator::new(),
                    )),
                    protocol_run_id: None,
                    protocol_state: None,
                    reasoning_path_tracker: super::feedback::ReasoningPathTracker::new(),
                    objective_tracking: true,
                    objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
                    objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
                    work_log: work_log.clone(),
                    // Resumed sessions = interactive: use the generous cap (50/5min).
                    // T7 of plan 9a1684b2.
                    oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
                    oob_trigger_cap: OOB_TRIGGER_CAP_INTERACTIVE,
                    oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
                    oob_capped_warned: Arc::new(AtomicBool::new(false)),
                    cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
                    cancel_tools_cap: CANCEL_TOOLS_CAP,
                    cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
                    active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
                    cli_background_tasks: Arc::new(AtomicUsize::new(0)),
                    cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
                    cancel_task_cap: CANCEL_TASK_CAP,
                    cancel_task_window: Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
                },
            );
            interrupt_flag
        };
        self.notify_attention(session_id, AttentionReason::SessionActive);

        // Spawn NATS interrupt listener for cross-instance interrupt support
        self.spawn_nats_interrupt_listener(
            session_id,
            interrupt_flag.clone(),
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS snapshot responder for cross-instance mid-stream join
        self.spawn_nats_snapshot_responder(
            session_id,
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS RPC send listener for cross-instance message routing
        self.spawn_nats_rpc_listener(
            session_id,
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn NATS cancel_tools listener (T2 of plan 28e9afe3) — same
        // as create_session. The old listener was cancelled above via
        // old_session.nats_cancel.cancel().
        self.spawn_nats_cancel_tools_listener(
            session_id,
            self.active_sessions.clone(),
            nats_cancel.clone(),
        );

        // Spawn the per-session background-tasks poller (T12 of plan
        // 754a1379). Mirror of `create_session` — without this spawn,
        // resumed sessions never run the grace-period purge, so
        // `pending_removal_at` set by `cancel_task` is never honoured
        // and entries linger forever in `active_background_tasks`.
        // (Bug discovered post-V2 cancel_task: a stuck "stopping…"
        // entry in the frontend with PIDs already dead — the poller
        // wasn't running for resumed sessions to physically purge.)
        Self::spawn_background_tasks_poller(
            session_id.to_string(),
            self.active_sessions.clone(),
            events_tx.clone(),
            self.nats.clone(),
            nats_cancel.clone(),
        );

        // Spawn the permanent out-of-band SDK message listener (T4+T5 of
        // plan 9a1684b2). Same as `create_session` — the resumed session
        // needs its own listener bound to the new InteractiveClient. The
        // old listener (if any) was already cancelled above via
        // `old_session.nats_cancel.cancel()`.
        super::oob_listener::spawn_oob_listener(
            session_id.to_string(),
            client.clone(),
            events_tx.clone(),
            is_streaming.clone(),
            next_seq.clone(),
            nats_cancel,
            super::oob_listener::OobListenerDeps {
                graph: self.graph.clone(),
                active_sessions: self.active_sessions.clone(),
                context_injector: self.context_injector.clone(),
                event_emitter: self.event_emitter.clone(),
                retry_config: self.config.retry.clone(),
                enrichment_pipeline: self.enrichment_pipeline.clone(),
                search: self.search.clone(),
                documents: self.document_store.clone(),
                nats: self.nats.clone(),
            },
        );

        // Persist the user_message event
        let user_event = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: uuid,
            seq: next_seq.fetch_add(1, Ordering::SeqCst),
            event_type: "user_message".to_string(),
            data: serde_json::to_string(&ChatEvent::UserMessage {
                content: message.to_string(),
            })
            .unwrap_or_default(),
            created_at: chrono::Utc::now(),
        };
        let _ = self.graph.store_chat_events(uuid, vec![user_event]).await;

        // Emit user_message on local broadcast + NATS
        let user_msg_event = ChatEvent::UserMessage {
            content: message.to_string(),
        };
        let _ = events_tx.send(user_msg_event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, user_msg_event);
        }
        self.notify_attention(session_id, AttentionReason::UserMessage);

        // Stream in background
        let session_id_str = session_id.to_string();
        let graph = self.graph.clone();
        let active_sessions = self.active_sessions.clone();
        let prompt = message.to_string();
        let injector = self.context_injector.clone();
        let event_emitter = self.event_emitter.clone();
        let nats = self.nats.clone();
        let retry_config = self.config.retry.clone();
        let enrichment_pipeline = self.enrichment_pipeline.clone();
        let search = self.search.clone();
        let documents = self.document_store.clone();

        tokio::spawn(async move {
            Self::stream_response(
                client,
                events_tx,
                prompt,
                session_id_str,
                graph,
                active_sessions,
                interrupt_flag,
                memory_manager,
                injector,
                next_seq,
                pending_messages,
                is_streaming,
                streaming_text,
                streaming_events,
                event_emitter,
                nats,
                sdk_control_rx,
                auto_continue,
                retry_config,
                enrichment_pipeline,
                search,
                documents,
            )
            .await;
        });

        Ok(())
    }

    /// Retrieve full event history for a session (including tool_use, tool_result, etc.).
    ///
    /// Reads structured `ChatEventRecord` from Neo4j (the same data the WebSocket
    /// replay uses), so every event type is preserved — not just text.
    ///
    /// Falls back to Meilisearch `nexus_messages` for pre-migration sessions that
    /// have no ChatEvent nodes yet (text only, no tool_use).
    pub async fn get_session_messages(
        &self,
        session_id: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<ChatEventPage> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;

        // Verify session exists
        self.graph
            .get_chat_session(uuid)
            .await
            .context("Failed to fetch session from Neo4j")?
            .ok_or_else(|| anyhow!("Session {} not found", session_id))?;

        let limit_val = limit.unwrap_or(200) as i64;
        let offset_val = offset.unwrap_or(0) as i64;

        // Try Neo4j ChatEvent nodes first (new format — has tool_use, etc.)
        let total_count = self.graph.count_chat_events(uuid).await?;

        if total_count > 0 {
            let events = self
                .graph
                .get_chat_events_paginated(uuid, offset_val, limit_val)
                .await?;

            let has_more = (offset_val + events.len() as i64) < total_count;

            return Ok(ChatEventPage {
                events,
                total_count: total_count as usize,
                has_more,
                offset: offset_val as usize,
                limit: limit_val as usize,
            });
        }

        // Fallback: pre-migration sessions — read from Meilisearch (text only)
        let session_node = self.graph.get_chat_session(uuid).await?.unwrap();
        if let Some(conversation_id) = session_node.conversation_id {
            let meili_client = meilisearch_sdk::client::Client::new(
                &self.config.meilisearch_url,
                Some(&self.config.meilisearch_key),
            )
            .map_err(|e| anyhow!("Failed to create Meilisearch client: {}", e))?;

            let index = meili_client.index("nexus_messages");
            let filter = legacy_messages_filter(&conversation_id);

            let results: meilisearch_sdk::search::SearchResults<
                nexus_claude::memory::MessageDocument,
            > = index
                .search()
                .with_query("")
                .with_filter(&filter)
                .with_sort(&["created_at:asc"])
                .with_limit(limit_val as usize)
                .with_offset(offset_val as usize)
                .execute()
                .await
                .map_err(|e| anyhow!("Meilisearch query failed: {}", e))?;

            let meili_total = results.estimated_total_hits.unwrap_or(0);

            // Convert MessageDocument → ChatEventRecord (text-only approximation)
            let events: Vec<ChatEventRecord> = results
                .hits
                .into_iter()
                .enumerate()
                .map(|(i, hit)| {
                    let msg = hit.result;
                    let event_type = if msg.role == "user" {
                        "user_message"
                    } else {
                        "assistant_text"
                    };
                    let chat_event = if msg.role == "user" {
                        ChatEvent::UserMessage {
                            content: msg.content.clone(),
                        }
                    } else {
                        ChatEvent::AssistantText {
                            content: msg.content.clone(),
                            parent_tool_use_id: None,
                        }
                    };
                    ChatEventRecord {
                        id: Uuid::new_v4(),
                        session_id: uuid,
                        seq: (offset_val + i as i64 + 1),
                        event_type: event_type.to_string(),
                        data: serde_json::to_string(&chat_event).unwrap_or_default(),
                        created_at: chrono::Utc::now(),
                    }
                })
                .collect();

            let has_more = offset_val as usize + events.len() < meili_total;

            return Ok(ChatEventPage {
                events,
                total_count: meili_total,
                has_more,
                offset: offset_val as usize,
                limit: limit_val as usize,
            });
        }

        // No events anywhere
        Ok(ChatEventPage {
            events: vec![],
            total_count: 0,
            has_more: false,
            offset: offset_val as usize,
            limit: limit_val as usize,
        })
    }

    /// Backfill title/preview for sessions that have a conversation_id but no title.
    /// Uses Meilisearch to fetch the first user message from each conversation.
    pub async fn backfill_previews_from_meilisearch(&self) -> Result<usize> {
        let injector = match self.context_injector.as_ref() {
            Some(i) => i,
            None => return Ok(0),
        };

        // Get all sessions without title
        let sessions = self
            .graph
            .list_chat_sessions(None, None, 200, 0, true)
            .await
            .context("Failed to list sessions")?;

        let mut count = 0;
        for session in &sessions.0 {
            // Skip sessions that already have a title
            if session.title.is_some() {
                continue;
            }
            // Need a conversation_id to look up messages
            let conv_id = match &session.conversation_id {
                Some(c) => c.clone(),
                None => continue,
            };

            // Load the first user message from Meilisearch
            let loaded = match injector.load_conversation(&conv_id, Some(1), Some(0)).await {
                Ok(l) => l,
                Err(_) => continue,
            };

            let first_user_msg = loaded
                .messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.clone());

            if let Some(content) = first_user_msg {
                let chars: Vec<char> = content.chars().collect();
                let title = if chars.len() > 80 {
                    format!("{}...", chars[..77].iter().collect::<String>().trim_end())
                } else {
                    content.clone()
                };
                let preview = if chars.len() > 200 {
                    format!("{}...", chars[..197].iter().collect::<String>().trim_end())
                } else {
                    content
                };
                let _ = self
                    .graph
                    .update_chat_session(
                        session.id,
                        None,
                        Some(title),
                        None,
                        None,
                        None,
                        Some(preview),
                    )
                    .await;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Search messages across all sessions via Meilisearch full-text search.
    ///
    /// Queries the `nexus_messages` index directly (bypassing the nexus SDK's
    /// `retrieve_context` which applies token_budget / max_context_items limits
    /// designed for LLM context injection, not UI search).
    pub async fn search_messages(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<MessageSearchResult>> {
        use nexus_claude::memory::MessageDocument;
        use std::collections::HashMap as StdHashMap;

        // Build a direct Meilisearch client from our config
        let meili_client = meilisearch_sdk::client::Client::new(
            &self.config.meilisearch_url,
            Some(&self.config.meilisearch_key),
        )
        .map_err(|e| anyhow!("Failed to create Meilisearch client: {}", e))?;

        let index = meili_client.index("nexus_messages");

        // Query Meilisearch directly — no token budget, no max_context_items
        let search_results: meilisearch_sdk::search::SearchResults<MessageDocument> = index
            .search()
            .with_query(query)
            .with_limit(limit * 5) // Fetch extra to allow grouping by session
            .with_show_ranking_score(true)
            .execute()
            .await
            .map_err(|e| anyhow!("Meilisearch search failed: {}", e))?;

        if search_results.hits.is_empty() {
            return Ok(vec![]);
        }

        // Group results by conversation_id
        let mut by_conversation: StdHashMap<String, Vec<MessageSearchHit>> = StdHashMap::new();
        for hit in &search_results.hits {
            let doc = &hit.result;
            let score = hit.ranking_score.unwrap_or(0.0);
            let search_hit = MessageSearchHit {
                message_id: doc.id.clone(),
                role: doc.role.clone(),
                content_snippet: truncate_snippet(&doc.content, 300),
                turn_index: doc.turn_index,
                created_at: doc.created_at,
                score,
            };
            by_conversation
                .entry(doc.conversation_id.clone())
                .or_default()
                .push(search_hit);
        }

        // Resolve conversation_id → session metadata from Neo4j
        let (all_sessions, _) = self
            .graph
            .list_chat_sessions(None, None, 200, 0, true)
            .await?;
        let session_lookup: StdHashMap<String, &crate::neo4j::models::ChatSessionNode> =
            all_sessions
                .iter()
                .filter_map(|s| s.conversation_id.as_ref().map(|cid| (cid.clone(), s)))
                .collect();

        // Build grouped results
        let mut results: Vec<MessageSearchResult> = Vec::new();
        for (conv_id, hits) in by_conversation {
            let session_info = session_lookup.get(&conv_id);

            // Filter by project_slug if specified
            if let Some(filter_slug) = project_slug {
                if let Some(session) = session_info {
                    if session.project_slug.as_deref() != Some(filter_slug) {
                        continue;
                    }
                } else {
                    continue; // No session found, skip
                }
            }

            let best_score = hits.iter().map(|h| h.score).fold(0.0_f64, f64::max);

            results.push(MessageSearchResult {
                session_id: session_info.map(|s| s.id.to_string()).unwrap_or_default(),
                session_title: session_info.and_then(|s| s.title.clone()),
                session_preview: session_info.and_then(|s| s.preview.clone()),
                project_slug: session_info.and_then(|s| s.project_slug.clone()),
                workspace_slug: session_info.and_then(|s| s.workspace_slug.clone()),
                conversation_id: conv_id,
                hits,
                best_score,
            });
        }

        // Sort by best_score descending
        results.sort_by(|a, b| {
            b.best_score
                .partial_cmp(&a.best_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to requested number of sessions
        results.truncate(limit);

        Ok(results)
    }

    /// Subscribe to a session's broadcast channel (used by WebSocket handler)
    pub async fn subscribe(&self, session_id: &str) -> Result<broadcast::Receiver<ChatEvent>> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return Ok(handle.events_tx.subscribe());
        }
        let sessions = self.active_sessions.read().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
        Ok(session.events_tx.subscribe())
    }

    /// Get the broadcast sender for a session (used by AgentGuard to emit events).
    pub async fn get_events_tx(&self, session_id: &str) -> Result<broadcast::Sender<ChatEvent>> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return Ok(handle.events_tx.clone());
        }
        let sessions = self.active_sessions.read().await;
        let session = sessions
            .get(session_id)
            .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
        Ok(session.events_tx.clone())
    }

    /// Get persisted events since a given sequence number (for WebSocket replay)
    pub async fn get_events_since(
        &self,
        session_id: &str,
        after_seq: i64,
    ) -> Result<Vec<ChatEventRecord>> {
        let uuid = Uuid::parse_str(session_id).context("Invalid session ID")?;
        self.graph
            .get_chat_events(uuid, after_seq, 10000) // generous limit for replay
            .await
    }

    /// Check if a session is currently streaming
    pub async fn is_session_streaming(&self, session_id: &str) -> bool {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle.is_streaming.load(Ordering::SeqCst);
        }
        let sessions = self.active_sessions.read().await;
        sessions
            .get(session_id)
            .map(|s| s.is_streaming.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    /// Get a snapshot of the current streaming state for mid-stream join.
    ///
    /// Returns `(is_streaming, accumulated_text, streaming_events)`.
    /// - `accumulated_text`: all stream_delta text since the current stream started
    /// - `streaming_events`: all structured events (ToolUse, ToolResult, AssistantText, etc.)
    ///   since the current stream started, excluding StreamDelta (which is in accumulated_text)
    ///
    /// This allows a newly connected WebSocket client to fully reconstruct the
    /// in-progress assistant turn, including tool calls.
    pub async fn get_streaming_snapshot(&self, session_id: &str) -> (bool, String, Vec<ChatEvent>) {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return (
                handle.is_streaming.load(Ordering::SeqCst),
                handle.streaming_text.lock().await.clone(),
                handle.streaming_events.lock().await.clone(),
            );
        }
        let sessions = self.active_sessions.read().await;
        match sessions.get(session_id) {
            Some(session) => {
                let is_streaming = session.is_streaming.load(Ordering::SeqCst);
                let text = session.streaming_text.lock().await.clone();
                let events = session.streaming_events.lock().await.clone();
                (is_streaming, text, events)
            }
            None => (false, String::new(), Vec::new()),
        }
    }

    /// Interrupt the current operation in a session.
    ///
    /// Sets the interrupt flag, which causes the stream loop to break and release the
    /// client lock. The stream loop then sends the actual interrupt signal to the CLI.
    /// This is instantaneous — no waiting for the Mutex.
    /// Interrupt the current turn of a session: end the LLM turn **and**
    /// SIGINT every descendant of the CLI (the tools it is running).
    ///
    /// Thin wrapper over [`Self::interrupt_scoped`], kept for the many
    /// callers that need neither the outcome nor a narrower scope. It
    /// preserves the historical `Result<()>` contract — including the
    /// "succeeds silently when the session is not local" semantics the
    /// tests assert.
    pub async fn interrupt(&self, session_id: &str) -> Result<()> {
        self.interrupt_scoped(session_id, true).await.map(|_| ())
    }

    /// Interrupt the current turn of a session, choosing whether the
    /// running tool subprocesses go down with it.
    ///
    /// ## Scope
    ///
    /// - `kill_tools = true` — historical behaviour. The turn ends and
    ///   every descendant of the CLI receives `SIGINT` (find, cargo,
    ///   sleep…). This is what the composer's Stop button wants: stop
    ///   everything, now.
    /// - `kill_tools = false` — the turn ends but no descendant is
    ///   signalled. The flag, the token and the SDK `control_request`
    ///   still go out, so the CLI stops generating; a `Bash` or
    ///   `Monitor` subprocess started in the background survives.
    ///
    /// ## What `kill_tools = false` does NOT preserve
    ///
    /// In-process `Task` sub-agents are **not** subprocesses: they live
    /// inside the CLI and are torn down by the same
    /// `control_request: interrupt` that ends the turn (it aborts the
    /// shared `QueryEngine.abortController`). There is therefore no way,
    /// at this layer, to end a turn while letting its in-process
    /// sub-agents run on — only detached *subprocesses* can be spared.
    ///
    /// ## Outcome
    ///
    /// Unlike [`Self::interrupt`], this reports what actually happened.
    /// `delivered: false` means the session was not in `active_sessions`
    /// and nothing local was interrupted — the caller can surface that
    /// instead of spinning on "Stopping…" forever.
    pub async fn interrupt_scoped(
        &self,
        session_id: &str,
        kill_tools: bool,
    ) -> Result<InterruptOutcome> {
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            let scope = if kill_tools {
                nexus_claude::agent::InterruptScope::TurnAndTools
            } else {
                nexus_claude::agent::InterruptScope::TurnOnly
            };
            let outcome = handle.interrupt_scoped(scope).await?;
            let diagnostic = outcome.diagnostic;
            return Ok(InterruptOutcome {
                delivered: true,
                routed: "local".to_string(),
                cli_pid: diagnostic.as_ref().and_then(|d| d.pid),
                killed_pids: diagnostic.map(|d| d.killed_pids).unwrap_or_default(),
            });
        }
        let (interrupt_flag, interrupt_token, stdin_tx, child_pid) = {
            let sessions = self.active_sessions.read().await;
            match sessions.get(session_id) {
                Some(s) => (
                    Some(s.interrupt_flag.clone()),
                    Some(s.interrupt_token.clone()),
                    s.stdin_tx.clone(),
                    s.child_pid,
                ),
                None => (None, None, None, None),
            }
        };

        let mut outcome = InterruptOutcome {
            delivered: false,
            routed: "none".to_string(),
            cli_pid: child_pid,
            killed_pids: Vec::new(),
        };

        if let Some(flag) = interrupt_flag {
            // Session is local — set the flag AND cancel the token so the stream loop breaks
            // immediately, even if blocked on stream.next() (e.g., CLI executing sleep 60)
            flag.store(true, Ordering::SeqCst);
            if let Some(token) = interrupt_token {
                token.cancel();
            }

            // Send the interrupt control_request to the CLI IMMEDIATELY via stdin_tx.
            // Don't wait for stream_response post-loop — it may take time to break out.
            if let Some(ref tx) = stdin_tx {
                let json = InteractiveClient::build_interrupt_json();
                if let Err(e) = tx.try_send(json) {
                    warn!(
                        session_id = %session_id,
                        "Failed to send interrupt control_request via stdin_tx: {}",
                        e
                    );
                }
            }

            // Kill descendant processes of the CLI (find, sleep, cargo, etc.)
            // WITHOUT killing the CLI itself. Delegated to `kill_descendants`
            // which is also used by `cancel_running_tools` (T1 of plan
            // 28e9afe3 — see decision d2bf0e7b for why these two paths share
            // the SIGINT primitive but interrupt() additionally sets the
            // flag/token + sends the control_request to end the turn,
            // whereas cancel_running_tools deliberately does NEITHER).
            //
            // Skipped entirely when `kill_tools` is false: the turn still
            // ends, but background subprocesses are left running.
            if kill_tools {
                let killed = Self::kill_descendants(child_pid);
                if !killed.is_empty() {
                    info!(
                        session_id = %session_id,
                        cli_pid = ?child_pid,
                        descendant_count = killed.len(),
                        descendant_pids = ?killed,
                        "Sent SIGINT to CLI descendant processes (not the CLI itself)"
                    );
                } else if let Some(pid) = child_pid {
                    debug!(
                        session_id = %session_id,
                        cli_pid = pid,
                        "No descendant processes found to kill"
                    );
                }
                outcome.killed_pids = killed;
            }

            outcome.delivered = true;
            outcome.routed = "local".to_string();

            info!(
                session_id = %session_id,
                kill_tools,
                descendants_killed = outcome.killed_pids.len(),
                "Interrupt: flag set, token cancelled, control_request sent"
            );
        } else {
            debug!(
                session_id = %session_id,
                "Session not active locally, interrupt will be routed via NATS only"
            );
        }

        // Always publish interrupt to NATS so the owning instance (if remote) also stops.
        // This is fire-and-forget — no-op if NATS is not configured.
        if let Some(ref nats) = self.nats {
            nats.publish_interrupt(session_id);
            if !outcome.delivered {
                outcome.routed = "nats".to_string();
            }
            debug!(
                session_id = %session_id,
                "Interrupt published to NATS"
            );
        }

        Ok(outcome)
    }

    /// Cancel the currently-running tool subprocess(es) of a session
    /// **without** ending the LLM turn (T2 of plan 28e9afe3).
    ///
    /// ## Semantics — what this does, and what it deliberately does NOT
    ///
    /// Sends `SIGINT` to every descendant of the CLI process via the
    /// `kill_descendants` helper extracted in T1. The descendant shell
    /// (running `find`, `npm install`, `cargo build`, …) exits with code
    /// 130, BashTool's awaited `exec()` reports a normal `tool_result`
    /// with `isError: true`, and the agent's turn continues — the LLM
    /// can then decide to call another tool, abandon, retry, etc.
    ///
    /// **Critical invariants** (cf decision `d2bf0e7b` on T1):
    /// - **Does NOT touch** `interrupt_flag` or `interrupt_token` —
    ///   touching them would break the PO stream loop and end the turn,
    ///   defeating the purpose of this method.
    /// - **Does NOT send** the SDK control_request `interrupt` — that
    ///   triggers `QueryEngine.abortController.abort()` in the CLI which
    ///   ends the turn entirely (the AbortController is shared by the
    ///   in-flight Anthropic API fetch and never reset).
    ///
    /// Both invariants have dedicated regression tests in T5.
    ///
    /// ## Rate limiting
    ///
    /// Sliding-window cap of `cancel_tools_cap` invocations per
    /// `cancel_tools_window` (default 10/60s) prevents click-spam from
    /// saturating `pgrep -P` and the CLI. When the cap is hit, the call
    /// returns `capped: true` with `killed_pids: []` and **no SIGINT is
    /// sent** — caller maps to HTTP 429 or a "slow down" toast.
    ///
    /// ## Cross-instance routing
    ///
    /// If the session is not local, the request is propagated via NATS
    /// `chat.{session_id}.cancel_tools` so the owning instance executes
    /// the SIGINT. The local rate cap still applies to prevent flooding
    /// NATS itself.
    pub async fn cancel_running_tools(&self, session_id: &str) -> Result<CancelToolsResult> {
        // The agent engine: the session handle stops the tools through the
        // provider (`AgentSession::cancel_tools`, scope all), under the same
        // per-session cap, and announces `tools_cancelled` itself — stored and
        // published to the other instances like every event of the session.
        // The session is local: the signal is not sent over NATS, which would
        // only come back to our own listener (`spawn_agent_nats_listeners`)
        // and cancel twice.
        if let Some(handle) = self.agent_runtime.get(session_id).await {
            return handle.cancel_tools().await;
        }

        // Look up session-local state in a single read lock.
        let session_state = {
            let sessions = self.active_sessions.read().await;
            sessions.get(session_id).map(|s| {
                (
                    s.child_pid,
                    s.cancel_tools_history.clone(),
                    s.cancel_tools_cap,
                    s.cancel_tools_window,
                    s.events_tx.clone(),
                )
            })
        };

        // Apply the rate cap — even when the session is remote (NATS
        // path), so a cross-instance click-spam can't loop forever.
        let (cli_pid, history, cap, window, events_tx) = match session_state {
            Some(state) => state,
            None => {
                // Session not local — still enforce a "soft" cap by
                // doing nothing locally; remote owner has its own cap.
                debug!(
                    session_id = %session_id,
                    "cancel_running_tools: session not active locally; routing via NATS only"
                );
                if let Some(ref nats) = self.nats {
                    nats.publish_cancel_tools(session_id);
                }
                return Ok(CancelToolsResult {
                    cli_pid: None,
                    killed_pids: Vec::new(),
                    capped: false,
                });
            }
        };

        // Check cap. The `false` return path is the happy path that
        // also records the timestamp atomically.
        let capped = !Self::check_and_record_cancel_cap(&history, cap, window).await;
        if capped {
            warn!(
                session_id = %session_id,
                cap = cap,
                window_secs = window.as_secs(),
                "cancel_running_tools: rate cap hit, refusing"
            );
            return Ok(CancelToolsResult {
                cli_pid,
                killed_pids: Vec::new(),
                capped: true,
            });
        }

        // SIGINT-only — never the control_request, never flag/token.
        let killed_pids = Self::kill_descendants(cli_pid);
        info!(
            session_id = %session_id,
            cli_pid = ?cli_pid,
            descendant_count = killed_pids.len(),
            descendant_pids = ?killed_pids,
            "cancel_running_tools: SIGINT sent to descendants (turn preserved)"
        );

        // Broadcast a typed event so all clients of this session
        // (multi-tab, sidebar, frontend logs) observe the cancel
        // immediately — even if no `ToolResult` cancelled is emitted
        // (e.g., agent was thinking, no tool was running).
        let event = ChatEvent::ToolsCancelled {
            cli_pid,
            killed_count: killed_pids.len(),
            requested_by: "user".to_string(),
        };
        let _ = events_tx.send(event.clone());

        // Cross-instance fan-out — both the cancel SIGNAL (so the
        // owning instance executes the SIGINT if remote) and the
        // ChatEvent (so remote clients of this session see the cancel
        // in their feed).
        if let Some(ref nats) = self.nats {
            nats.publish_cancel_tools(session_id);
            nats.publish_chat_event(session_id, event);
        }

        Ok(CancelToolsResult {
            cli_pid,
            killed_pids,
            capped: false,
        })
    }

    /// Cancel a single tracked background task (Monitor, Bash bg) by id.
    ///
    /// Plan 754a1379 (T7) — granular companion to `cancel_running_tools`.
    /// Instead of killing every descendant of the CLI, this targets a
    /// single tracked task by its `tool_use_id` (= map key).
    ///
    /// ## V2 semantics — PID-targeted kill (plan fc35b25e, T4)
    ///
    /// 1. Atomically extract the task's `pid` and mark
    ///    `pending_removal_at = Some(now)` under a single tasks-lock op
    ///    (the actual physical purge happens after `GRACE_PERIOD` in T12).
    /// 2. If `pid.is_some()`: call `kill_subtree(pid)` — sends SIGINT to
    ///    the root subprocess **and** all its descendants (`tail -F`,
    ///    `find`, etc.) and returns the list of PIDs that received the
    ///    signal. Populates `CancelTaskResult.killed_pids`.
    /// 3. If `pid.is_none()` (claim race — `track_background_task_start`
    ///    spawned the async claim but it hasn't fired yet, OR the
    ///    subprocess crashed before pgrep saw it): logs a warning and
    ///    falls back to V1 map-side-only cancel. `killed_pids` is empty
    ///    in this edge case — the user can fall back to the global
    ///    `cancel_running_tools` if needed.
    /// 4. Broadcasts a fresh `ChatEvent::ActiveTasksUpdate` so the
    ///    frontend immediately reflects the cancelled state.
    ///
    /// The 5 s grace period from T12 (plan 754a1379) is preserved — it
    /// absorbs `BackgroundOutput` ticks that may be in flight between
    /// the SIGINT and the subprocess actually exiting.
    ///
    /// ## Cap
    ///
    /// Reuses `check_and_record_cancel_cap` with the per-session
    /// `cancel_task_history` (T9 fields). Default 30 invocations / 5
    /// minutes. The cap is more generous than `cancel_running_tools`
    /// (10/60s) because the per-task popover can naturally produce more
    /// clicks (one per tracked task).
    ///
    /// ## NATS routing
    ///
    /// Cross-instance: if the session is owned by another instance,
    /// publishes a `cancel_task` event over NATS so the owning instance
    /// performs the local map mutation. (Mirror of
    /// `publish_cancel_tools` — added in a follow-up alongside the
    /// listener wiring.)
    ///
    /// ## Returns
    ///
    /// - `CancelTaskResult { task_id, killed_pids, capped: false }` on
    ///   success — `killed_pids` is the list of PIDs (root + descendants)
    ///   that received SIGINT in V2. Empty in the claim-race fallback
    ///   path described above.
    /// - `capped: true` if the rate cap was hit (no broadcast, no map
    ///   mutation, no kill).
    /// - Returns `Ok(...)` even if the session is unknown or the
    ///   task_id isn't in the map (idempotent — clicking Stop twice on
    ///   the same task is fine).
    pub async fn cancel_task(&self, session_id: &str, task_id: &str) -> Result<CancelTaskResult> {
        let session_state = {
            let sessions = self.active_sessions.read().await;
            sessions.get(session_id).map(|s| {
                (
                    s.cancel_task_history.clone(),
                    s.cancel_task_cap,
                    s.cancel_task_window,
                    s.events_tx.clone(),
                    s.active_background_tasks.clone(),
                )
            })
        };

        let (history, cap, window, events_tx, tasks_arc) = match session_state {
            Some(state) => state,
            None => {
                debug!(
                    session_id = %session_id,
                    task_id = %task_id,
                    "cancel_task: session not active locally; idempotent no-op"
                );
                if let Some(ref _nats) = self.nats {
                    // TODO follow-up: nats.publish_cancel_task(session_id, task_id)
                    // once the cross-instance listener exists.
                }
                return Ok(CancelTaskResult {
                    task_id: task_id.to_string(),
                    killed_pids: Vec::new(),
                    capped: false,
                });
            }
        };

        // Rate cap.
        let capped = !Self::check_and_record_cancel_cap(&history, cap, window).await;
        if capped {
            warn!(
                session_id = %session_id,
                task_id = %task_id,
                cap = cap,
                window_secs = window.as_secs(),
                "cancel_task: rate cap hit, refusing"
            );
            return Ok(CancelTaskResult {
                task_id: task_id.to_string(),
                killed_pids: Vec::new(),
                capped: true,
            });
        }

        // V2 (plan fc35b25e, T4): atomically extract the task's pid AND
        // mark for removal under a single tasks-lock op, then capture the
        // snapshot for broadcast. Doing both in the same critical section
        // ensures a concurrent claim-update from `async_pid_claim` cannot
        // populate the pid AFTER we read it.
        let (snapshot, task_pid) = {
            let mut tasks = tasks_arc.lock().await;
            let pid = if let Some(entry) = tasks.get_mut(task_id) {
                entry.pending_removal_at = Some(std::time::Instant::now());
                let captured = entry.pid;
                info!(
                    session_id = %session_id,
                    task_id = %task_id,
                    kind = ?entry.kind,
                    pid = ?captured,
                    "cancel_task: marked for removal (grace period via T12)"
                );
                captured
            } else {
                debug!(
                    session_id = %session_id,
                    task_id = %task_id,
                    "cancel_task: task_id unknown in map; idempotent no-op"
                );
                None
            };
            (
                tasks.values().cloned().collect::<Vec<BackgroundTaskInfo>>(),
                pid,
            )
        };

        // V2: SIGINT the subtree if the async claim populated a pid;
        // otherwise log a warning and fall back to map-side-only cancel.
        let killed_pids = match task_pid {
            Some(root_pid) => {
                let killed = Self::kill_subtree(root_pid);
                info!(
                    session_id = %session_id,
                    task_id = %task_id,
                    root_pid = root_pid,
                    killed_count = killed.len(),
                    "cancel_task: SIGINT'd subtree (plan fc35b25e, T4)"
                );
                killed
            }
            None => {
                warn!(
                    session_id = %session_id,
                    task_id = %task_id,
                    "cancel_task: no PID stored (claim race or subprocess crashed before discovery), map-side only"
                );
                Vec::new()
            }
        };

        let event = ChatEvent::ActiveTasksUpdate { tasks: snapshot };
        let _ = events_tx.send(event.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, event);
            // TODO follow-up: nats.publish_cancel_task(session_id, task_id);
        }

        Ok(CancelTaskResult {
            task_id: task_id.to_string(),
            killed_pids,
            capped: false,
        })
    }

    /// Snapshot the current background task tracking map for a session.
    ///
    /// Returns an empty `Vec` if the session is unknown locally — the
    /// caller (typically the REST snapshot endpoint, T6) treats this as
    /// "no tasks active" rather than 404, since a fresh session legitimately
    /// has none. Plan 754a1379 (T6).
    ///
    /// Used by the frontend on WebSocket reconnect to re-hydrate the
    /// toolbar indicator and any in-progress MonitorCards before the
    /// next `ChatEvent::ActiveTasksUpdate` lands.
    pub async fn get_active_background_tasks(&self, session_id: &str) -> Vec<BackgroundTaskInfo> {
        // Drop the read lock on `active_sessions` before taking the
        // per-session mutex to avoid holding two locks simultaneously
        // and keep the borrow-checker happy (the inner Arc clone moves
        // out of the read guard's borrow scope).
        let tasks_arc = {
            let sessions = self.active_sessions.read().await;
            match sessions.get(session_id) {
                Some(active) => active.active_background_tasks.clone(),
                None => return Vec::new(),
            }
        };
        let result: Vec<BackgroundTaskInfo> = tasks_arc.lock().await.values().cloned().collect();
        result
    }

    /// Sliding-window rate cap helper for `cancel_running_tools`.
    /// Returns `true` when the call is allowed (records the timestamp
    /// atomically), `false` when the cap is hit (no record).
    ///
    /// Mirror of `chat::oob_listener::check_and_record_trigger_cap` but
    /// without the warning-emission gate — cancel_tools is user-driven
    /// and the caller surfaces the cap hit directly via the HTTP 429
    /// or `capped: true` response.
    pub(crate) async fn check_and_record_cancel_cap(
        history: &Arc<Mutex<VecDeque<Instant>>>,
        cap: u32,
        window: Duration,
    ) -> bool {
        let mut hist = history.lock().await;
        let now = Instant::now();

        // Drain entries older than `window` from the front.
        while let Some(front) = hist.front() {
            if now.duration_since(*front) > window {
                hist.pop_front();
            } else {
                break;
            }
        }

        if (hist.len() as u32) >= cap {
            return false;
        }
        hist.push_back(now);
        true
    }

    /// Spawn the per-session NATS listener for `cancel_tools` signals
    /// (T2 of plan 28e9afe3). Mirror of `spawn_nats_interrupt_listener`.
    ///
    /// On message, fetches the session's `child_pid` + cap state and
    /// runs `kill_descendants` directly — does NOT re-enter
    /// `cancel_running_tools` to avoid the routing loop where every
    /// SIGINT would re-publish to NATS and trigger our own listener.
    ///
    /// Applies the rate cap **here** as well, so a remote instance
    /// that publishes the cancel signal at high frequency cannot
    /// bypass the local cap on the SIGINT actually executed. Without
    /// this, spam-publishing on NATS would translate to spam SIGINT
    /// on the owning instance.
    fn spawn_nats_cancel_tools_listener(
        &self,
        session_id: &str,
        active_sessions: Arc<RwLock<HashMap<String, ActiveSession>>>,
        cancel: CancellationToken,
    ) {
        let Some(ref nats) = self.nats else {
            return;
        };

        let nats = nats.clone();
        let session_id = session_id.to_string();

        tokio::spawn(async move {
            let mut subscriber = match nats.subscribe_cancel_tools(&session_id).await {
                Ok(sub) => sub,
                Err(e) => {
                    warn!(
                        "Failed to subscribe to NATS cancel_tools for session {}: {}",
                        session_id, e
                    );
                    return;
                }
            };

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        debug!(
                            "NATS cancel_tools listener cancelled for session {} (session replaced)",
                            session_id
                        );
                        break;
                    }
                    msg = subscriber.next() => {
                        let Some(_msg) = msg else { break; };

                        // Pull the session's cli_pid + cap state (or stop
                        // if the session was removed locally).
                        let session_state = {
                            let sessions = active_sessions.read().await;
                            sessions.get(&session_id).map(|s| {
                                (
                                    s.child_pid,
                                    s.cancel_tools_history.clone(),
                                    s.cancel_tools_cap,
                                    s.cancel_tools_window,
                                )
                            })
                        };
                        let Some((cli_pid, history, cap, window)) = session_state else {
                            debug!(
                                "Session {} no longer active, stopping NATS cancel_tools listener",
                                session_id
                            );
                            break;
                        };

                        // Apply the cap on the SIGINT side too — protects
                        // against a remote instance flooding the NATS
                        // subject. Mirrors the cap applied by the
                        // local-path of `cancel_running_tools`.
                        if !Self::check_and_record_cancel_cap(&history, cap, window).await {
                            warn!(
                                session_id = %session_id,
                                cap = cap,
                                window_secs = window.as_secs(),
                                "NATS cancel_tools listener: rate cap hit, dropping signal"
                            );
                            continue;
                        }

                        let killed = Self::kill_descendants(cli_pid);
                        info!(
                            session_id = %session_id,
                            cli_pid = ?cli_pid,
                            descendant_count = killed.len(),
                            "NATS cancel_tools received, SIGINT sent to descendants"
                        );
                    }
                }
            }
        });
    }

    /// Recursively enumerate all descendant PIDs of a given process.
    ///
    /// Uses `pgrep -P <pid>` to find direct children, then recurses.
    /// Returns an empty Vec if the process has no children or pgrep fails.
    /// This is used by `interrupt()` to kill tool subprocesses (find, sleep, etc.)
    /// without killing the CLI itself (which would break the session).
    #[cfg(unix)]
    fn get_descendant_pids(pid: u32) -> Vec<u32> {
        fn get_children(pid: u32) -> Vec<u32> {
            std::process::Command::new("pgrep")
                .args(["-P", &pid.to_string()])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.lines().filter_map(|l| l.trim().parse().ok()).collect())
                .unwrap_or_default()
        }

        let mut result = Vec::new();
        let mut stack = get_children(pid);
        while let Some(child) = stack.pop() {
            result.push(child);
            stack.extend(get_children(child));
        }
        result
    }

    /// Windows stub for `get_descendant_pids` — descendant tracking relies
    /// on `pgrep -P <pid>` (POSIX-specific). Windows would need a different
    /// implementation (e.g. `wmic process` or the `Toolhelp32` API). The
    /// runtime feature this powers (V2 cancel-tools subtree termination)
    /// is currently Unix-only, so we no-op cleanly on Windows: callers
    /// receive an empty Vec and skip the descendant-aware code paths.
    #[cfg(windows)]
    fn get_descendant_pids(_pid: u32) -> Vec<u32> {
        Vec::new()
    }

    /// Parse the `[[dd-]hh:]mm:ss` format returned by `ps -o etime=` into a
    /// total elapsed-seconds count. Plan fc35b25e (T1, V2 cancel_task).
    ///
    /// `ps -o etime=` is the only locale-independent process-age field that
    /// works identically on Linux and macOS — `etimes` (with `s`) is
    /// Linux-only, and `lstart` is locale-formatted. See audit observation
    /// `7dbde66b` for the full cross-platform investigation.
    ///
    /// Format examples:
    /// - `"00:01"`        → 1 second
    /// - `"12:34"`        → 754 seconds (12 min 34 s)
    /// - `"02:12:34"`     → 7954 seconds
    /// - `"01-02:12:34"`  → 94354 seconds (1 day 2 h …)
    ///
    /// Returns `None` on any parse failure rather than panicking — callers
    /// fall back to skipping the PID in the diff sort.
    #[allow(dead_code)] // wired by T2 of plan fc35b25e
    pub(crate) fn parse_etime(s: &str) -> Option<u64> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        // Optional "DD-" days prefix.
        let (days, rest) = match s.find('-') {
            Some(idx) => (s[..idx].parse::<u64>().ok()?, &s[idx + 1..]),
            None => (0u64, s),
        };
        let parts: Vec<&str> = rest.split(':').collect();
        let (hours, minutes, seconds) = match parts.as_slice() {
            [sec] => (0u64, 0u64, sec.parse::<u64>().ok()?),
            [m, sec] => (0u64, m.parse::<u64>().ok()?, sec.parse::<u64>().ok()?),
            [h, m, sec] => (
                h.parse::<u64>().ok()?,
                m.parse::<u64>().ok()?,
                sec.parse::<u64>().ok()?,
            ),
            _ => return None,
        };
        Some(days * 86_400 + hours * 3_600 + minutes * 60 + seconds)
    }

    /// Return the elapsed seconds since `pid` started, or `None` if the
    /// process is gone / `ps` failed / the format couldn't be parsed.
    /// Plan fc35b25e (T1).
    ///
    /// Cross-platform via `ps -p <pid> -o etime=`. Single code path for
    /// Linux + macOS — see audit observation `7dbde66b` for why we
    /// abandoned the original `/proc/<pid>/stat` + boot_time approach.
    #[cfg(unix)]
    #[allow(dead_code)] // wired by T2 of plan fc35b25e
    pub(crate) fn process_etime_seconds(pid: u32) -> Option<u64> {
        let output = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "etime="])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let raw = String::from_utf8(output.stdout).ok()?;
        Self::parse_etime(&raw)
    }

    /// Windows stub for `process_etime_seconds` — `ps -o etime=` doesn't
    /// exist on Windows. Caller `pid_discovery_diff` uses this only as a
    /// sort key and substitutes `u64::MAX` on `None`, so returning `None`
    /// here keeps PIDs in their original (unsorted) order on Windows.
    #[cfg(windows)]
    #[allow(dead_code)]
    pub(crate) fn process_etime_seconds(_pid: u32) -> Option<u64> {
        None
    }

    /// Compute `after \ before` (set difference, treating each `Vec` as a
    /// set) and return the new PIDs sorted by elapsed time **ascending**
    /// (smallest etime first → most recently spawned first). PIDs whose
    /// etime can't be read are placed at the end.
    ///
    /// Plan fc35b25e (T1, T2). The first element of the returned Vec is
    /// the heuristic "claim" candidate for a freshly-observed
    /// `ToolUse(Monitor | Bash run_in_background)` event — the assumption
    /// being that the most recently spawned descendant of the CLI is the
    /// subprocess this ToolUse just created.
    ///
    /// ## Race-tolerance caveat
    ///
    /// When two ToolUses fire within the diff window (1s), this function
    /// returns multiple new PIDs and the first-PID claim heuristic
    /// becomes ambiguous. V2 accepts this as documented degraded
    /// behaviour — see plan constraint `e1dfeaa5`.
    #[allow(dead_code)] // wired by T2 of plan fc35b25e
    pub(crate) fn pid_discovery_diff(before: &[u32], after: &[u32]) -> Vec<u32> {
        let before_set: std::collections::HashSet<u32> = before.iter().copied().collect();
        let mut new_pids: Vec<u32> = after
            .iter()
            .copied()
            .filter(|p| !before_set.contains(p))
            .collect();
        // Sort by etime ascending (smallest = most recent). Unreadable
        // etimes (process gone, ps failed) sort last via u64::MAX.
        new_pids.sort_by_key(|pid| Self::process_etime_seconds(*pid).unwrap_or(u64::MAX));
        new_pids
    }

    /// Send `SIGINT` to every descendant of `cli_pid` **without** sending the
    /// SDK `interrupt` control_request and **without** touching the
    /// `interrupt_flag`/`interrupt_token`. Returns the PIDs that received the
    /// signal (empty if `cli_pid` is `None` or no descendants exist).
    ///
    /// ## Why this helper exists separately from `interrupt()`
    ///
    /// Decision `d2bf0e7b` (T1 of plan 28e9afe3) — empirical analysis of the
    /// Claude Code CLI source (`bridge/bridgeMessaging.ts:362` →
    /// `QueryEngine.ts:1158`) showed that the SDK control_request `interrupt`
    /// triggers `QueryEngine.abortController.abort()`, which propagates to
    /// **every** consumer of the AbortController (including the in-flight
    /// fetch to the Anthropic API and the BashTool's `exec()`). The
    /// AbortController is **never reset** within a `QueryEngine` instance, so
    /// `abort()` ends the entire turn — the LLM cannot continue afterwards.
    ///
    /// In contrast, sending `SIGINT` directly to the descendant shell
    /// (without touching the AbortController) makes BashTool's awaited
    /// `exec()` see its child exit with code 130. BashTool reports a normal
    /// `tool_result` with `isError: true` and the agent's turn continues
    /// normally — exactly the behaviour required by `cancel_running_tools`.
    ///
    /// Therefore: `interrupt()` keeps using the control_request +
    /// flag/token (turn-ending behaviour, by design), while
    /// `cancel_running_tools()` calls **only** this helper.
    ///
    /// On non-unix platforms this is a no-op that returns an empty Vec.
    fn kill_descendants(cli_pid: Option<u32>) -> Vec<u32> {
        #[cfg(unix)]
        {
            let Some(pid) = cli_pid else {
                return Vec::new();
            };
            let descendants = Self::get_descendant_pids(pid);
            for &desc_pid in &descendants {
                // SAFETY: libc::kill is FFI; SIGINT to a non-existent PID
                // returns ESRCH which we silently ignore (idempotent —
                // the tool may have finished naturally between snapshot
                // and signal).
                unsafe {
                    libc::kill(desc_pid as i32, libc::SIGINT);
                }
            }
            descendants
        }
        #[cfg(not(unix))]
        {
            let _ = cli_pid;
            Vec::new()
        }
    }

    /// Send `SIGINT` to a single PID **and** all its descendants. Returns the
    /// PIDs that received the signal (filters out ESRCH for already-dead
    /// processes).
    ///
    /// Plan fc35b25e (T3, V2 cancel_task). Companion to `kill_descendants`
    /// but with two key differences:
    ///
    /// | Aspect | `kill_descendants(cli_pid)` | `kill_subtree(root_pid)` |
    /// |--------|------------------------------|---------------------------|
    /// | Targets the root | **No** (preserves CLI) | **Yes** (root + descendants) |
    /// | Use case | Global "Stop" — kill every running tool | Per-task cancel — kill one Monitor / Bash bg subtree |
    /// | Caller | `cancel_running_tools` | `cancel_task` |
    /// | Plan | `28e9afe3` (T2) | `fc35b25e` (T3) |
    ///
    /// The reason `kill_descendants` doesn't kill the root is that the root
    /// is the Claude Code CLI subprocess — killing it would tear down the
    /// whole session (cf decision `d2bf0e7b`). For `kill_subtree` the root
    /// is a Monitor / Bash bg subprocess that we **do** want to terminate
    /// (it has no protected status), so we include it in the SIGINT batch.
    ///
    /// On non-unix platforms this is a no-op that returns an empty Vec.
    fn kill_subtree(root_pid: u32) -> Vec<u32> {
        #[cfg(unix)]
        {
            let mut all_pids = vec![root_pid];
            all_pids.extend(Self::get_descendant_pids(root_pid));

            let mut killed = Vec::with_capacity(all_pids.len());
            for &pid in &all_pids {
                // SAFETY: libc::kill is FFI; SIGINT to a non-existent PID
                // returns ESRCH which we silently ignore (idempotent —
                // the subprocess may have finished naturally between
                // pgrep snapshot and signal).
                let rc = unsafe { libc::kill(pid as i32, libc::SIGINT) };
                if rc == 0 {
                    killed.push(pid);
                } else {
                    let errno = std::io::Error::last_os_error();
                    if errno.raw_os_error() != Some(libc::ESRCH) {
                        warn!(pid = pid, error = ?errno, "kill_subtree: SIGINT failed");
                    }
                }
            }
            killed
        }
        #[cfg(not(unix))]
        {
            let _ = root_pid;
            Vec::new()
        }
    }

    /// Whether a provider is served by the agent engine: every provider but
    /// Claude Code, and Claude Code itself only when `CHAT_PROVIDER_PATH=agent`.
    pub(crate) fn engine_is_agent(&self, provider_id: &str) -> bool {
        provider_id != super::provider::resolver::CLAUDE_CODE
            || self.config.provider_path == super::config::ProviderPath::Agent
    }

    /// The visible warning when Claude Code is FORCED onto the agent engine
    /// (`CHAT_PROVIDER_PATH=agent`): the server logs what that session loses.
    fn warn_if_forced(&self, provider_id: &str, session: &dyn nexus_claude::agent::AgentSession) {
        if provider_id == super::provider::resolver::CLAUDE_CODE {
            warn!(
                features = ?super::agent_runtime::degraded_features(session.capabilities()),
                "Claude Code is running on the AGENT engine (CHAT_PROVIDER_PATH=agent): \
                 these features are NOT available for this session"
            );
        }
    }

    /// Decides which provider instance serves a session being opened (A16).
    ///
    /// Reads the stored instances, the consent of the project (tied to the
    /// instance's current origin), the roles (project before global) and the
    /// aliases. The legacy engine can only drive Claude Code: a choice of any
    /// other instance is `provider_unavailable` there.
    #[cfg(test)]
    pub(crate) async fn resolve_provider_choice(
        &self,
        request: &ChatRequest,
        project_slug: Option<&str>,
    ) -> Result<super::provider::resolver::ProviderChoice> {
        self.resolve_provider_choice_for(request, project_slug, None)
            .await
    }

    /// The models ticked for the conversation being opened, `None` when nothing restricts it.
    fn routing_pool_of(request: &ChatRequest) -> Option<&[super::types::RoutingPoolEntry]> {
        request.routing_pool.as_deref().filter(|p| !p.is_empty())
    }

    /// Same, tying the stored cognitive decision to the session being opened.
    pub(crate) async fn resolve_provider_choice_for(
        &self,
        request: &ChatRequest,
        project_slug: Option<&str>,
        session_id: Option<Uuid>,
    ) -> Result<super::provider::resolver::ProviderChoice> {
        use super::provider::{catalog, resolver, settings, store};
        let instances = store::instances(self.graph.as_ref()).await?;
        let consents = match project_slug {
            Some(slug) => store::consents(self.graph.as_ref(), slug).await?,
            None => Vec::new(),
        };
        let global = store::roles(self.graph.as_ref(), settings::GLOBAL).await?;
        let project = match project_slug {
            Some(slug) => store::roles(self.graph.as_ref(), &settings::project_scope(slug)).await?,
            None => Default::default(),
        };
        let aliases = store::aliases(self.graph.as_ref()).await?;
        let role = if request.spawned_by.is_some() || request.runner_context.is_some() {
            resolver::Role::Executor
        } else {
            resolver::Role::Pilot
        };
        let routing_instances = self.cognitive_routing.as_ref().map(|_| instances.clone());
        let store_catalog =
            catalog::StoreCatalog::new(instances, &consents, project_slug.is_some());
        let policy: settings::ModelPolicy = self
            .graph
            .get_llm_setting(settings::GLOBAL, settings::POLICY_KEY)
            .await?
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default();
        let pick = catalog::policy_pick(&policy, role, request.task_class.as_deref(), &aliases);
        let mut input = catalog::resolve_input(
            role,
            request.provider.as_deref(),
            request.task_alias.as_deref(),
            request
                .run_provider
                .as_deref()
                .map(|p| (p, request.run_model.as_deref())),
            &global,
            &project,
            &aliases,
        );
        // The persona level (A16): the persona's preference goes through the alias
        // table like the task alias; one the table does not know is no level.
        input.persona = catalog::alias_candidate(request.persona_alias.as_deref(), &aliases);
        // The cognitive router (R2): a decision is always taken and stored; it
        // fills the project-rule level only when the mode and the stage say so.
        // Never fatal: any failure leaves the declarative resolution untouched.
        let cognitive = match (&self.cognitive_routing, routing_instances) {
            (Some(routing), Some(instances)) => {
                match self
                    .cognitive_decision(
                        routing,
                        request,
                        project_slug,
                        role,
                        &instances,
                        &store_catalog,
                        &aliases,
                        session_id,
                    )
                    .await
                {
                    Ok(decision) => decision,
                    Err(error) => {
                        warn!(%error, "cognitive routing skipped: the declared rules apply");
                        None
                    }
                }
            }
            _ => None,
        };
        let auto_pick = cognitive
            .as_ref()
            .filter(|d| d.applied)
            .and_then(|d| d.chosen.clone());
        if let Some(pick) = &auto_pick {
            input.project_rule = Some(resolver::Candidate::new(
                pick.provider_id.clone(),
                Some(pick.model.clone()),
            ));
        }
        // Models ticked, none named and no applied pick among them: the first ticked model
        // opens the conversation, never a default the user did not tick.
        if auto_pick.is_none() && input.request.is_none() {
            if let Some(first) = Self::routing_pool_of(request).and_then(<[_]>::first) {
                input.request = Some(resolver::Candidate::new(
                    first.provider.clone(),
                    Some(first.model.clone()),
                ));
            }
        }
        // `enforce` puts the policy's candidate where the global rule would be;
        // `shadow` changes nothing and is only recorded (A19).
        if let Some(p) = pick.as_ref().filter(|p| p.enforced) {
            input.global_rule = Some(p.candidate.clone());
        }
        let mut choice = resolver::resolve(&input, &store_catalog).map_err(anyhow::Error::new)?;
        choice.decision_id = cognitive.as_ref().map(|d| d.id);
        if let Some(p) = &pick {
            if p.enforced && choice.routed_by == resolver::RoutedBy::GlobalRule {
                choice.route_rule = Some(p.rule.clone());
            } else if !p.enforced {
                choice.shadow = Some((p.candidate.provider_id.clone(), p.candidate.model.clone()));
                choice.route_rule = Some(p.rule.clone());
            }
        }
        if let (Some(pick), Some(decision)) = (&auto_pick, &cognitive) {
            // Our candidate won the project-rule level: it was chosen by the router.
            if choice.routed_by == resolver::RoutedBy::ProjectRule
                && choice.provider_id == pick.provider_id
                && choice.model.as_deref() == Some(pick.model.as_str())
            {
                choice.routed_by = resolver::RoutedBy::Auto;
                choice.route_rule = Some(format!("auto:{}", decision.signature.arm_key()));
                choice.reason = Some(decision.reason.clone());
            }
        } else if let Some(pick) = cognitive.as_ref().and_then(|d| d.chosen.as_ref()) {
            // Not applied: what the router would have chosen, recorded only.
            if choice.shadow.is_none() {
                choice.shadow = Some((pick.provider_id.clone(), Some(pick.model.clone())));
            }
        }
        Ok(choice)
    }

    /// The pool for a project (`None` = none): the routing pool built from what is
    /// stored. Empty when the cognitive router is not wired.
    pub(crate) async fn routing_pool_for(
        &self,
        project_slug: Option<&str>,
    ) -> Vec<super::provider::cognitive::candidates::ModelFacts> {
        use super::provider::{catalog, store};
        let Some(routing) = self.cognitive_routing.as_ref() else {
            return Vec::new();
        };
        let Ok(instances) = store::instances(self.graph.as_ref()).await else {
            return Vec::new();
        };
        let consents = match project_slug {
            Some(slug) => store::consents(self.graph.as_ref(), slug)
                .await
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let store_catalog =
            catalog::StoreCatalog::new(instances.clone(), &consents, project_slug.is_some());
        self.routing_pool(routing, &instances, &store_catalog).await
    }

    /// Links the cognitive decision taken for a session to that session, then keeps
    /// it open until the session closes. Best effort: a failure loses a link, never
    /// a session.
    async fn remember_routing_decision(&self, session_id: &str, decision_id: Option<Uuid>) {
        let (Some(decision_id), Ok(session)) = (decision_id, Uuid::parse_str(session_id)) else {
            return;
        };
        let Some(store) = self
            .cognitive_routing
            .as_ref()
            .and_then(|routing| routing.store.clone())
        else {
            return;
        };
        match store.decision(decision_id).await {
            Ok(Some(mut decision)) => {
                decision.session_id = Some(session);
                if let Err(error) = store.put_decision(&decision).await {
                    warn!(%session_id, %decision_id, %error, "linking the routing decision to its session failed");
                    return;
                }
            }
            Ok(None) => return,
            Err(error) => {
                warn!(%session_id, %decision_id, %error, "routing decision unreadable: not linked");
                return;
            }
        }
        self.open_decisions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(session_id.to_string(), decision_id);
    }

    /// The user changed the model of the session by hand: the open decision, if
    /// any, counts that as an override when it closes.
    async fn flag_routing_override(&self, session_id: &str) {
        let decision_id = self
            .open_decisions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session_id)
            .copied();
        let (Some(decision_id), Some(store)) = (
            decision_id,
            self.cognitive_routing
                .as_ref()
                .and_then(|routing| routing.store.clone()),
        ) else {
            return;
        };
        if let Err(error) =
            super::provider::cognitive::feedback::mark_override(store.as_ref(), decision_id).await
        {
            warn!(%session_id, %decision_id, %error, "flagging the routing override failed");
        }
    }

    /// Closes the decision of a session that ends. A chat cannot tell whether its
    /// work succeeded, so an unknown outcome is NOT fed to the arm (it would read
    /// as a failure and bias every chat arm downward): cost and duration are
    /// recorded, nothing is learned. A model switched by hand is the one signal a
    /// chat gives: that closes the decision with the override penalty.
    async fn close_routing_decision(&self, session_id: &str) {
        use super::provider::cognitive::{
            demotion::{apply_demotion, should_demote},
            feedback::{close_decision_with, CollectorSink, Outcome, TrajectorySink},
        };
        let decision_id = self
            .open_decisions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(session_id);
        let (Some(decision_id), Some(store)) = (
            decision_id,
            self.cognitive_routing
                .as_ref()
                .and_then(|routing| routing.store.clone()),
        ) else {
            return;
        };
        let (cost_usd, duration_ms) = match Uuid::parse_str(session_id) {
            Ok(id) => match self.graph.get_chat_session(id).await {
                Ok(Some(session)) => (
                    session.total_cost_usd,
                    u64::try_from((chrono::Utc::now() - session.created_at).num_milliseconds())
                        .ok(),
                ),
                _ => (None, None),
            },
            Err(_) => (None, None),
        };
        let decision = match store.decision(decision_id).await {
            Ok(Some(decision)) => decision,
            Ok(None) => return,
            Err(error) => {
                warn!(%session_id, %decision_id, %error, "routing decision unreadable: left open");
                return;
            }
        };
        let overridden = decision.outcome.as_ref().is_some_and(|o| o.overridden);
        if overridden {
            let project = decision.signature.project_slug.clone();
            let settings = match super::provider::cognitive::load_routing(
                self.graph.as_ref(),
                project.as_deref(),
            )
            .await
            {
                Ok((settings, _scope)) => settings,
                Err(error) => {
                    warn!(%session_id, %decision_id, %error, "routing settings unreadable: decision left open");
                    return;
                }
            };
            let outcome = Outcome {
                attempts: 1,
                cost_usd,
                duration_ms,
                user_overrode_model: true,
                ..Outcome::default()
            };
            // The closed decision reaches the trajectory collector when one is wired.
            let collector = self
                .trajectory_collector
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let sink = collector.map(CollectorSink::new);
            let sink: Option<&dyn TrajectorySink> = sink.as_ref().map(|s| s as &dyn TrajectorySink);
            if let Err(error) =
                close_decision_with(store.as_ref(), &settings, sink, decision_id, &outcome).await
            {
                warn!(%session_id, %decision_id, %error, "closing the routing decision failed");
                return;
            }
            // The override is the only close that moves an arm: check its class for demotion.
            let class = decision.signature.arm_key();
            match should_demote(store.as_ref(), &class, project.as_deref(), &settings).await {
                Ok(Some(reason)) => {
                    if let Err(error) = apply_demotion(
                        self.graph.as_ref(),
                        project.as_deref(),
                        &reason,
                        self.event_emitter.as_deref(),
                    )
                    .await
                    {
                        warn!(%session_id, class, %error, "the demotion could not be applied");
                    }
                }
                Ok(None) => {}
                Err(error) => warn!(%session_id, class, %error, "demotion check failed"),
            }
            return;
        }
        let mut outcome = decision.outcome.unwrap_or_default();
        outcome.cost_usd = cost_usd;
        outcome.duration_ms = duration_ms;
        if let Err(error) = store.set_outcome(decision_id, outcome).await {
            warn!(%session_id, %decision_id, %error, "recording the session cost on its routing decision failed");
        }
    }

    /// The pool the cognitive router chooses from: every model of every
    /// reachable instance, with the capabilities and the price nexus reports,
    /// the health (one probe per instance per window) and the project's consent.
    async fn routing_pool(
        &self,
        routing: &super::provider::cognitive::decider::CognitiveRouting,
        instances: &[super::provider::settings::InstanceRecord],
        store_catalog: &super::provider::catalog::StoreCatalog,
    ) -> Vec<super::provider::cognitive::candidates::ModelFacts> {
        use super::provider::cognitive::candidates::ModelFacts;
        use super::provider::resolver::{self, InstanceCatalog};
        use std::time::{Duration, Instant};

        let mut ids = vec![resolver::CLAUDE_CODE.to_string()];
        ids.extend(
            instances
                .iter()
                .filter(|i| !resolver::is_remote_instance(&i.id))
                .map(|i| i.id.clone()),
        );
        let mut pool = Vec::new();
        for id in ids {
            // The legacy engine can only drive Claude Code.
            if !self.engine_is_agent(&id) && id != resolver::CLAUDE_CODE {
                continue;
            }
            let Ok(provider) = self.provider_for(&id).await else {
                continue;
            };
            let now = Instant::now();
            let healthy = match routing.health.get(&id, now) {
                Some(known) => known,
                None => {
                    let ok = provider.health().await.status
                        != nexus_claude::agent::HealthStatus::Unavailable;
                    routing.health.put(&id, ok, now);
                    ok
                }
            };
            let default_model = instances
                .iter()
                .find(|i| i.id == id)
                .and_then(|i| i.default_model.clone());
            let models = if healthy {
                tokio::time::timeout(Duration::from_secs(2), provider.catalog())
                    .await
                    .ok()
                    .and_then(Result::ok)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let allowed = store_catalog.is_allowed_for_project(&id);
            let mut seen = std::collections::HashSet::new();
            let mut entries: Vec<(String, Option<nexus_claude::agent::ModelPrice>)> =
                models.into_iter().map(|m| (m.id, m.pricing)).collect();
            if let Some(model) = default_model {
                if !entries.iter().any(|(m, _)| *m == model) {
                    entries.push((model, None));
                }
            }
            // A native model nobody probed reports no tools and no window, so the
            // hard constraints would drop it for good. Probe the instance's default
            // model once (bounded; a failure is remembered for ten minutes).
            if id != resolver::CLAUDE_CODE && healthy {
                if let Some(model) = instances
                    .iter()
                    .find(|i| i.id == id)
                    .and_then(|i| i.default_model.clone())
                {
                    let unknown = {
                        let caps = provider.capabilities(Some(&model));
                        // `tools` stays false until a probe has run; a window can be
                        // known from the preset without one.
                        !caps.tools
                    };
                    let key = (id.clone(), model.clone());
                    let recent = routing
                        .probed
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .get(&key)
                        .is_some_and(|at| now.duration_since(*at) < Duration::from_secs(600));
                    let concrete = self.native_probers.read().await.get(&id).cloned();
                    if let (true, false, Some(native)) = (unknown, recent, concrete) {
                        let outcome = tokio::time::timeout(
                            Duration::from_secs(10),
                            native.refresh_capabilities(&model),
                        )
                        .await;
                        if !matches!(outcome, Ok(Ok(ref caps)) if caps.tools) {
                            routing
                                .probed
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .insert(key, now);
                        }
                    }
                }
            }
            for (model, price) in entries {
                if !seen.insert(model.clone()) {
                    continue;
                }
                let caps = provider.capabilities(Some(&model));
                pool.push(ModelFacts::from_capabilities(
                    id.clone(),
                    model,
                    &caps,
                    price,
                    Some(healthy),
                    allowed,
                ));
            }
        }
        pool
    }

    /// Asks the cognitive router about the session being opened. The decision is
    /// stored by the decider whether it is applied or not.
    #[allow(clippy::too_many_arguments)]
    async fn cognitive_decision(
        &self,
        routing: &super::provider::cognitive::decider::CognitiveRouting,
        request: &ChatRequest,
        project_slug: Option<&str>,
        role: super::provider::resolver::Role,
        instances: &[super::provider::settings::InstanceRecord],
        store_catalog: &super::provider::catalog::StoreCatalog,
        aliases: &[super::provider::settings::ModelAlias],
        session_id: Option<Uuid>,
    ) -> Result<Option<super::provider::cognitive::decision::CognitiveDecision>> {
        use super::provider::cognitive::{
            candidates::Slot,
            decision::DecideRequest,
            scorer::PriorHints,
            signature::{ContextHints, TaskSignature},
        };
        let (settings, _) =
            super::provider::cognitive::load_routing(self.graph.as_ref(), project_slug).await?;
        let signature = match role {
            super::provider::resolver::Role::Pilot => TaskSignature::from_chat_request(
                &crate::refs::turn::visible_text(&request.message),
                !request.attachments.is_empty(),
                project_slug,
                ContextHints::default(),
            ),
            super::provider::resolver::Role::Executor => TaskSignature::from_delegation(
                request.task_class.as_deref(),
                None,
                1,
                project_slug,
                ContextHints::default(),
            ),
        };
        routing.set_hints(PriorHints::from_aliases(aliases));
        let mut pool = self.routing_pool(routing, instances, store_catalog).await;
        // A provider, model, alias or run the caller named is never substituted.
        let named = request.provider.is_some()
            || request.model.as_deref().is_some_and(|m| !m.is_empty())
            || request.task_alias.is_some()
            || request.persona_alias.is_some()
            || request.run_provider.is_some();
        // The conversation's own mode replaces the settings' (the chat menu: Auto = full).
        let mut settings = settings;
        if let Some(mode) = request.routing_mode {
            settings.mode = mode;
        }
        // The models ticked in the menu are the only candidates, and are routed like `full`
        // inside them, whatever the settings' mode.
        if let Some(ticked) = Self::routing_pool_of(request) {
            pool.retain(|facts| {
                ticked
                    .iter()
                    .any(|e| e.provider == facts.provider_id && e.model == facts.model)
            });
            settings.mode = super::provider::cognitive::ProviderRoutingMode::Full;
        }
        let mut decide = DecideRequest::new(signature, settings, pool);
        decide.slot = if named {
            Slot::Explicit
        } else {
            Slot::Automatic
        };
        decide.trust = request.permission_mode.as_deref() == Some("bypassPermissions");
        decide.session_id = session_id;
        Ok(Some(routing.decider.decide(&decide).await?))
    }

    /// The provider instance, built-in or stored. A stored instance becomes a
    /// native harness over its OpenAI-compatible endpoint; the result is kept
    /// until the stored record changes.
    pub(crate) async fn provider_for(
        &self,
        provider_id: &str,
    ) -> Result<Arc<dyn nexus_claude::agent::AgentProvider>> {
        use super::provider::{native_factory, resolver, store};
        if let Some(provider) = self.provider_source.get(provider_id) {
            return Ok(provider);
        }
        let record = store::instance(self.graph.as_ref(), provider_id)
            .await?
            .ok_or_else(|| {
                anyhow::Error::new(resolver::ResolveError::UnknownProvider(
                    provider_id.to_string(),
                ))
            })?;
        {
            let cache = self.native_cache.read().await;
            if let Some((cached, provider)) = cache.get(provider_id) {
                if *cached == record {
                    return Ok(Arc::clone(provider));
                }
            }
        }
        let (provider, concrete) = native_factory::build_provider_with_handle(
            &record,
            self.vault.clone(),
            self.native_transcripts.as_deref(),
        )
        .map_err(anyhow::Error::new)?;
        {
            let mut probers = self.native_probers.write().await;
            match concrete {
                Some(native) => probers.insert(provider_id.to_string(), native),
                None => probers.remove(provider_id),
            };
        }
        self.native_cache
            .write()
            .await
            .insert(provider_id.to_string(), (record, Arc::clone(&provider)));
        Ok(provider)
    }

    /// The working directory on the machine of a `claude_code_remote` instance;
    /// `None` for every other provider. A remote instance without one is a
    /// stored record that cannot work: refused, never run in a local path.
    pub(crate) async fn remote_cwd_of(&self, provider_id: &str) -> Result<Option<String>> {
        if !super::provider::resolver::is_remote_instance(provider_id) {
            return Ok(None);
        }
        let record = super::provider::store::instance(self.graph.as_ref(), provider_id)
            .await?
            .ok_or_else(|| {
                anyhow::Error::new(super::provider::resolver::ResolveError::UnknownProvider(
                    provider_id.to_string(),
                ))
            })?;
        if record.kind != super::provider::settings::KIND_CLAUDE_CODE_REMOTE {
            return Ok(None);
        }
        record.remote_cwd.map(Some).ok_or_else(|| {
            anyhow::Error::new(nexus_claude::agent::ProviderError::invalid(
                "this remote instance has no working directory",
            ))
        })
    }

    /// Everything that must hold before a session's content is sent to a
    /// provider other than Claude Code. Claude Code (the historical path) is
    /// not subject to it.
    ///
    /// 1. the security gate: bound session tokens need a signing key (A32);
    /// 2. the project's consent, tied to the instance's CURRENT origin (A28);
    /// 3. the endpoint guard (A36), before any connection;
    /// 4. `Trust` is refused only for a remote machine whose record does not allow it:
    ///    every other provider treats it like Claude Code does (decision of 2026-10-07,
    ///    which replaces A35 — the sandbox level informs the user, it gates nothing);
    /// 5. the sending is journaled; a failed write refuses the opening (A37).
    pub(crate) async fn authorize_provider_use(&self, u: ProviderUse<'_>) -> Result<()> {
        let ProviderUse {
            provider_id,
            model,
            mode,
            project_slug,
            claims,
            session_id,
        } = u;
        use super::provider::{endpoint_guard, resolver, store};
        use nexus_claude::agent::{PolicyMode, ProviderError};
        if provider_id == resolver::CLAUDE_CODE {
            return Ok(());
        }
        let refuse = |e: resolver::ResolveError| anyhow::Error::new(e);
        if self.config.jwt_secret.is_none() {
            return Err(anyhow::Error::new(ProviderError::unsupported(
                "security_gate",
            )));
        }
        let record = store::instance(self.graph.as_ref(), provider_id)
            .await?
            .ok_or_else(|| refuse(resolver::ResolveError::UnknownProvider(provider_id.into())))?;
        let slug = project_slug
            .ok_or_else(|| refuse(resolver::ResolveError::NotAllowed(provider_id.into())))?;
        let consented = store::consents(self.graph.as_ref(), slug)
            .await?
            .iter()
            .any(|c| super::provider::settings::consent_holds(c, &record));
        if !consented {
            return Err(refuse(resolver::ResolveError::NotAllowed(
                provider_id.into(),
            )));
        }
        // A codex / acp instance is a local process: there is no endpoint to guard.
        let guard = if super::provider::settings::is_process_kind(&record.kind) {
            Ok(())
        } else {
            endpoint_guard::validate_endpoint(
                &record.base_url,
                &endpoint_guard::EndpointPolicy::default(),
            )
            .await
            .map(|_| ())
        };
        if let Err(refusal) = guard {
            warn!(
                provider = provider_id,
                code = refusal.code(),
                "endpoint refused by the guard"
            );
            return Err(refuse(resolver::ResolveError::NotAllowed(
                provider_id.into(),
            )));
        }
        // A remote machine may run `Trust` only when its record says so explicitly (per
        // machine): the tools run where nobody is watching. Any other provider behaves like
        // Claude Code: `trust` opens, whatever its sandbox.
        if mode == PolicyMode::Trust && Self::trust_needs_opt_in(&record) {
            return Err(anyhow::Error::new(ProviderError::unsupported("sandbox")));
        }
        let entry = serde_json::json!({
            "session_id": session_id,
            "project": slug,
            "provider": provider_id,
            "origin": record.origin,
            "model": model,
            "by": claims.map(|c| c.email.as_str()),
            "at": chrono::Utc::now().to_rfc3339(),
        });
        let key = format!(
            "send:{}:{}",
            chrono::Utc::now().timestamp_millis(),
            session_id
        );
        if let Err(e) = self
            .graph
            .put_llm_setting("journal", &key, &entry.to_string())
            .await
        {
            error!(error = %e, "send journal write failed: refusing to open");
            return Err(refuse(resolver::ResolveError::Unavailable {
                provider_id: provider_id.into(),
                role: resolver::Role::Pilot,
            }));
        }
        Ok(())
    }

    /// Whether this instance only runs `Trust` when its own record allows it: a Claude Code on
    /// another machine, whose tools run where nobody is watching. No other provider is held to it.
    pub(crate) fn trust_needs_opt_in(record: &super::provider::settings::InstanceRecord) -> bool {
        record.kind == super::provider::settings::KIND_CLAUDE_CODE_REMOTE && !record.allow_trust
    }

    /// Names of the server variables handed to an agent besides the base
    /// allowlist: the tooling list and the operator's, never a server secret.
    fn child_env_inherit_names() -> Vec<String> {
        let operator = std::env::var(CHILD_ENV_INHERIT_VAR).unwrap_or_default();
        CHILD_ENV_TOOLING
            .iter()
            .map(|name| (*name).to_string())
            .chain(
                operator
                    .split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty() && !SERVER_ONLY_SECRETS.contains(name))
                    .map(str::to_string),
            )
            .collect()
    }

    /// The `SessionSpec` of a session of the agent path: working directory,
    /// model, system prompt, the neutral tool policy of the permission mode,
    /// the project-orchestrator MCP server (same environment as the Claude
    /// path, session-bound token included) and the clean child environment.
    #[cfg(test)]
    pub(crate) async fn build_agent_spec(
        &self,
        i: AgentSpecInput<'_>,
    ) -> Result<nexus_claude::agent::SessionSpec> {
        self.build_agent_spec_with_access(i, super::provider::policy::SessionAccess::Normal)
            .await
    }

    /// [`Self::build_agent_spec`] for a session of the given access: the read-only deny list
    /// is part of the neutral policy, so it wins over every mode, `trust` included.
    pub(crate) async fn build_agent_spec_with_access(
        &self,
        i: AgentSpecInput<'_>,
        access: super::provider::policy::SessionAccess,
    ) -> Result<nexus_claude::agent::SessionSpec> {
        let AgentSpecInput {
            cwd,
            model,
            system_prompt,
            permission_mode,
            add_dirs,
            user_claims,
            session_id,
            third_party,
            max_tokens,
            kind,
            remote_cwd,
            hooks,
        } = i;
        // A neutral directory of the host is made on demand (new session or resume).
        if remote_cwd.is_none() {
            if let Err(e) = super::neutral_place::ensure(cwd) {
                warn!(cwd = %cwd, error = %e, "could not create the neutral chat directory");
            }
        }
        use nexus_claude::agent::{
            EnvSpec, McpServerSpec, SessionSpec, SystemPromptMode, SystemPromptSpec,
        };
        let (mode, allowed, disallowed) = {
            let perm = self.permission_config.read().await;
            (
                permission_mode
                    .map(str::to_string)
                    .unwrap_or_else(|| perm.mode.clone()),
                perm.allowed_tools.clone(),
                perm.disallowed_tools.clone(),
            )
        };
        let policy =
            super::provider::policy::tool_policy_with_access(&mode, &allowed, &disallowed, access)
                .ok_or_else(|| {
                    anyhow::Error::new(nexus_claude::agent::ProviderError::invalid(
                        "unknown permission mode or malformed tool pattern",
                    ))
                })?;
        // The project is read before the hooks scope is consumed: it is what the consent of
        // the network tools is tied to (A28).
        let project_slug = hooks.as_ref().and_then(|scope| scope.project_slug.clone());
        let env = self
            .po_mcp_env(
                Some(&mode),
                user_claims,
                Some(session_id),
                access
                    .tool_profile()
                    .or_else(|| third_party_tool_profile(third_party, policy.mode)),
                third_party,
            )
            .await;
        // A remote session runs where `remote_cwd` says; the local path of the
        // project (and a local `~`) means nothing on that machine.
        let mut spec = SessionSpec::new(match remote_cwd {
            Some(remote) => remote.to_string(),
            None => expand_tilde(cwd),
        });
        spec.model = Some(model.to_string());
        // What a provider kind refuses, it is not given (a refusal here would be a
        // typed `unsupported` at open): ACP has no system prompt and no extra
        // dirs; Codex and ACP have no turn limits.
        let is_acp = kind == nexus_claude::agent::ProviderKind::Acp;
        let has_turn_limits = matches!(
            kind,
            nexus_claude::agent::ProviderKind::ClaudeCode
                | nexus_claude::agent::ProviderKind::Native
        );
        if !is_acp {
            let text = if remote_cwd.is_some() {
                format!("{system_prompt}\n\n{REMOTE_SESSION_NOTICE}")
            } else {
                system_prompt.to_string()
            };
            spec.system_prompt = Some(SystemPromptSpec {
                text,
                mode: SystemPromptMode::Replace,
            });
        }
        spec.policy = policy;
        // A remote Claude Code cannot carry an MCP server (its configuration holds
        // the session token, which must not reach another machine's command line):
        // the PO tools are NOT given to it, and `system_init.degraded_features`
        // says so (`project_orchestrator_tools`).
        if remote_cwd.is_none() {
            spec.mcp_servers.insert(
                "project-orchestrator".to_string(),
                McpServerSpec::Stdio {
                    command: self.config.mcp_server_path.to_string_lossy().to_string(),
                    args: Vec::new(),
                    env: env.into_iter().collect(),
                },
            );
        }
        if !is_acp && remote_cwd.is_none() {
            spec.extra_dirs = add_dirs.iter().map(std::path::PathBuf::from).collect();
        }
        // The native harness has no tool of its own: its files, shell and web are `nexus-tools`,
        // one process per session attached as the `nexus` server (B40). Not for Claude Code,
        // Codex or ACP (they bring their own), nor for a remote session (its paths are another
        // machine's). The executable is launched by its absolute path, checked at every
        // opening (#598); its profile is signed into a per-session token, its network tools
        // follow the project's consent by origin and the browser its authorisation (#596).
        let mut gate: Option<Vec<String>> = None;
        if kind == nexus_claude::agent::ProviderKind::Native && remote_cwd.is_none() {
            use super::provider::native_factory::runnable_nexus_tools;
            use super::provider::nexus_tools as nt;
            match self
                .config
                .nexus_tools_path
                .as_deref()
                .and_then(runnable_nexus_tools)
            {
                Some(program) => {
                    // The browser is held to the same rule: an absolute, executable path.
                    let browser = self
                        .config
                        .nexus_browser_path
                        .as_deref()
                        .and_then(runnable_nexus_tools)
                        .map(nexus_claude::providers::native::BrowserTools::new);
                    let attachment = nt::attach(
                        &nt::NexusToolsConfig { program, browser },
                        &nt::GraphConsents(self.graph.clone()),
                        self.vault.as_deref(),
                        nt::AttachInput {
                            session_id,
                            cwd: &spec.cwd,
                            extra_dirs: &spec.extra_dirs,
                            policy: &spec.policy,
                            ceiling: spec.policy_ceiling.as_ref(),
                            project: project_slug.as_deref(),
                            ttl_secs: self.config.session_token_expiry_secs,
                            now: chrono::Utc::now(),
                        },
                    )
                    .await;
                    for (engine, why) in &attachment.skipped_engines {
                        warn!(
                            session_id,
                            engine = %engine,
                            code = why.code(),
                            "search engine left out of the session"
                        );
                    }
                    for (name, server) in attachment.servers {
                        // A session that already names its own server keeps it.
                        spec.mcp_servers.entry(name).or_insert(server);
                    }
                    gate = Some(attachment.search_origins);
                }
                None => tracing::warn!(
                    session_id,
                    configured = ?self.config.nexus_tools_path,
                    "nexus-tools not found or not executable (set NEXUS_TOOLS_PATH, or put it \
                     next to the server or on the PATH): this native session has no file, \
                     shell or web tool, and reports the `nexus_tools` feature as missing"
                ),
            }
        }
        if has_turn_limits {
            spec.max_turns = u32::try_from(self.config.max_turns).ok();
        }
        spec.limits.max_tokens = max_tokens;
        spec.env = EnvSpec {
            inherit: Self::child_env_inherit_names(),
            set: Default::default(),
        };
        // The knowledge graph's hooks, for the providers that run hooks in their loop. A
        // remote session has no local project to resolve a file against.
        let runs_hooks = remote_cwd.is_none()
            && matches!(
                kind,
                nexus_claude::agent::ProviderKind::Native
                    | nexus_claude::agent::ProviderKind::ClaudeCode
            );
        if let Some(scope) = hooks.filter(|_| runs_hooks) {
            let context_source = match (scope.task_id, scope.project_slug) {
                (Some(task_id), _) => CompactionContextSource::Task(task_id),
                (None, Some(slug)) => CompactionContextSource::Session(slug),
                (None, None) => CompactionContextSource::None,
            };
            let table = self.graph_hook_table(GraphHookInput {
                session_id: session_id.to_string(),
                context_source,
                work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
                tool_knowledge: !scope.runner,
                announce: None,
            });
            spec.hooks = Some(Arc::new(
                super::agent_hooks::GraphSessionHooks::new(
                    table,
                    session_id,
                    spec.cwd.display().to_string(),
                    Some(mode.clone()),
                )
                .with_turn_router(self.turn_routing.get(session_id)),
            ));
        }
        // The consent of the project and the revocation of the token are checked before ANY
        // other hook, and whether or not the session has the knowledge-graph hooks.
        if let Some(search_origins) = gate {
            spec.hooks = Some(Arc::new(
                super::provider::nexus_tools::ToolAccessHooks::new(
                    spec.hooks.take(),
                    Arc::new(super::provider::nexus_tools::GraphConsents(
                        self.graph.clone(),
                    )),
                    session_id,
                    project_slug,
                    search_origins,
                ),
            ));
        }
        Ok(spec)
    }

    /// Opens a session on the provider-neutral engine and sends the first
    /// message. The `ChatSession` node is already persisted by `create_session`.
    async fn open_agent_session(&self, o: AgentOpen<'_>) -> Result<CreateSessionResponse> {
        let AgentOpen {
            request,
            session_id,
            provider_id,
            model,
            system_prompt,
            add_dirs,
            project_slug,
            relay,
            access,
        } = o;
        let provider = self.provider_for(provider_id).await?;
        let remote_cwd = self.remote_cwd_of(provider_id).await?;
        let sid = session_id.to_string();
        // A run (an executor) on a remote machine that does not allow `Trust` runs under `ask`
        // instead (a pilot asking for it is refused in authorize_provider_use). Every other
        // provider keeps the mode it was asked: it behaves like Claude Code (decision of
        // 2026-10-07, which replaces A35).
        let permission_mode = match request.permission_mode.as_deref() {
            Some(mode)
                if request.spawned_by.is_some()
                    && super::provider::policy::parse_mode(mode)
                        .is_some_and(|p| p.neutral == nexus_claude::agent::PolicyMode::Trust)
                    && match super::provider::store::instance(self.graph.as_ref(), provider_id)
                        .await?
                    {
                        Some(record) => Self::trust_needs_opt_in(&record),
                        None => false,
                    } =>
            {
                Some("default")
            }
            other => other,
        };
        self.register_turn_router(
            &sid,
            provider_id,
            model,
            project_slug,
            OpeningTurn {
                explicit_model: request.model.is_some(),
                provider_imposed: request.provider.is_some(),
                moved_in: super::relay::moved_by_auto(relay),
                routing_mode: request.routing_mode,
                routing_pool: request.routing_pool.clone(),
                permission_mode: request.permission_mode.as_deref(),
                message: &crate::refs::turn::visible_text(&request.message),
                next_turn: 0,
            },
        )
        .await;
        let spec = self
            .build_agent_spec_with_access(
                AgentSpecInput {
                    cwd: &request.cwd,
                    model,
                    system_prompt,
                    permission_mode,
                    add_dirs,
                    user_claims: request.user_claims.as_ref(),
                    session_id: &sid,
                    third_party: provider_id != super::provider::resolver::CLAUDE_CODE,
                    max_tokens: request.max_tokens,
                    kind: provider.kind(),
                    remote_cwd: remote_cwd.as_deref(),
                    hooks: Some(AgentHookScope {
                        project_slug: project_slug.map(str::to_string),
                        task_id: request
                            .spawned_by
                            .as_deref()
                            .and_then(parse_spawned_by)
                            .and_then(|ctx| ctx.task_id),
                        runner: request.runner_context.is_some(),
                    }),
                },
                access,
            )
            .await?;
        let tool_policy = super::provider::policy::wire_tool_policy(&spec.policy);
        if let Err(e) = self
            .authorize_provider_use(ProviderUse {
                provider_id,
                model: spec.model.as_deref().unwrap_or(model),
                mode: spec.policy.mode,
                project_slug,
                claims: request.user_claims.as_ref(),
                session_id: &sid,
            })
            .await
        {
            crate::auth::agent_tokens::revoke_session(&sid);
            return Err(e);
        }
        // Preflight at EVERY opening (A30), not only when the instance was saved:
        // is the provider reachable and logged in now?
        if provider_id != super::provider::resolver::CLAUDE_CODE {
            let health = provider.health().await;
            if health.status == nexus_claude::agent::HealthStatus::Unavailable {
                crate::auth::agent_tokens::revoke_session(&sid);
                return Err(anyhow::Error::new(health.error.unwrap_or(
                    nexus_claude::agent::ProviderError::EndpointUnreachable {
                        detail: "the provider reports itself unavailable".to_string(),
                    },
                )));
            }
        }
        let tool_profile = spec_tool_profile(&spec);
        let nexus_missing = lacks_nexus_tools(
            provider.kind(),
            &spec,
            self.config.nexus_tools_path.as_deref(),
        );
        let session = provider.open(spec).await.map_err(|e| {
            // Nothing will ever use this session's token.
            crate::auth::agent_tokens::revoke_session(&sid);
            self.turn_routing.remove(&sid);
            anyhow::Error::new(e)
        })?;
        // ... and does the model's window hold the tool schemas the session was
        // given, with room left to work? (known only once the provider has probed)
        if provider_id != super::provider::resolver::CLAUDE_CODE {
            if let Err(e) = window_holds_the_tools(session.capabilities(), tool_profile) {
                let _ = session.close().await;
                crate::auth::agent_tokens::revoke_session(&sid);
                return Err(anyhow::Error::new(e));
            }
        }
        self.finish_agent_open(
            &sid,
            provider_id,
            provider.kind(),
            session,
            1,
            tool_policy,
            nexus_missing,
        )
        .await;
        if let Some(handle) = self.agent_runtime.get(&sid).await {
            // As on Claude Code: a runner always continues, at most five times; an
            // interactive session follows the configuration, with no limit.
            let runner = request.runner_context.is_some();
            handle.configure_auto_continue(
                runner || self.config.auto_continue,
                if runner { 5 } else { 0 },
            );
        }
        if let Some(handle) = self.agent_runtime.get(&sid).await {
            // A relayed conversation says so first, on the thread.
            if let Some(relay) = relay {
                handle.emit(relay.event(&sid, provider_id)).await;
            }
            if !request.message.is_empty() {
                handle
                    .send_message_relayed(
                        &request.message,
                        &super::relay::prefixed(relay.and_then(|r| r.text()), &request.message),
                    )
                    .await?;
            }
        }
        Ok(CreateSessionResponse {
            session_id: sid.clone(),
            stream_url: format!("/ws/chat/{sid}"),
            execution_place: Default::default(),
            access,
            notices: Vec::new(),
        })
    }

    /// What a session of the agent engine gets around its turns, built from the
    /// manager as it is configured NOW (the pipeline is replaced after construction).
    pub(crate) fn turn_services(&self) -> Arc<dyn super::agent_runtime::TurnServices> {
        Arc::new(ManagerTurnServices {
            graph: self.graph.clone(),
            enrichment_pipeline: self.enrichment_pipeline.clone(),
            turn_routing: Arc::clone(&self.turn_routing),
            documents: self.document_store.clone(),
            nats: self.nats.clone(),
            anchor: self.anchor_session(),
        })
    }

    /// Records what the provider reported (frozen capabilities, resume token)
    /// and registers the live session. `nexus_missing`: the session was opened without
    /// its `nexus-tools` server ([`lacks_nexus_tools`]), which it reports as a feature it
    /// does not have rather than leaving only a line in the server's log.
    #[allow(clippy::too_many_arguments)]
    async fn finish_agent_open(
        &self,
        session_id: &str,
        provider_id: &str,
        kind: nexus_claude::agent::ProviderKind,
        session: Arc<dyn nexus_claude::agent::AgentSession>,
        first_seq: i64,
        tool_policy: serde_json::Value,
        nexus_missing: bool,
    ) {
        self.warn_if_forced(provider_id, session.as_ref());
        if let Some(router) = self.turn_routing.get(session_id) {
            router.set_model_live(session.capabilities().set_model_live);
        }
        let capabilities = serde_json::to_string(session.capabilities()).unwrap_or_default();
        let token = session.resume_token().map(|t| t.to_wire());
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            if let Err(e) = self
                .graph
                .update_chat_session_harness(uuid, Some(&capabilities), token.as_deref())
                .await
            {
                warn!(session_id, error = %e, "Failed to persist the provider snapshot (non-fatal)");
            }
        }
        let kind_name = serde_json::to_value(kind)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| "claude_code".to_string());
        let extra_degraded = if nexus_missing {
            vec![super::agent_runtime::NEXUS_TOOLS_FEATURE.to_string()]
        } else {
            Vec::new()
        };
        let handle = self
            .agent_runtime
            .adopt_with(
                session_id,
                provider_id,
                session,
                first_seq,
                &kind_name,
                tool_policy,
                Some(self.turn_services()),
                extra_degraded,
            )
            .await;
        self.spawn_agent_nats_listeners(handle);
    }

    /// What the other instances can ask of a session of the agent engine this
    /// instance runs, as for a Claude Code session (`spawn_nats_*_listener`):
    /// an interrupt, a cancel of the running tools, a streaming snapshot (a
    /// client joining mid-turn there), and the RPC send (a message, a held
    /// message, a queue op, a permission answer, the auto-continue toggle).
    /// They end when the session closes.
    fn spawn_agent_nats_listeners(&self, handle: Arc<super::agent_runtime::AgentSessionHandle>) {
        let Some(nats) = self.nats.clone() else {
            return;
        };
        let sid = handle.session_id.clone();
        {
            let (nats, handle, sid) = (nats.clone(), Arc::clone(&handle), sid.clone());
            tokio::spawn(async move {
                let Ok(mut sub) = nats.subscribe_interrupt(&sid).await else {
                    warn!(session_id = %sid, "agent engine: NATS interrupt subscription failed");
                    return;
                };
                loop {
                    tokio::select! {
                        _ = handle.closed.cancelled() => break,
                        msg = sub.next() => {
                            if msg.is_none() { break; }
                            let _ = handle.interrupt().await;
                        }
                    }
                }
            });
        }
        {
            // `cancel_tools` asked on another instance (`cancel_running_tools`
            // there finds no local session and publishes): the handle applies
            // the cap and stops the tools, as `spawn_nats_cancel_tools_listener`
            // does for a Claude Code session.
            let (nats, handle, sid) = (nats.clone(), Arc::clone(&handle), sid.clone());
            tokio::spawn(async move {
                let Ok(mut sub) = nats.subscribe_cancel_tools(&sid).await else {
                    warn!(session_id = %sid, "agent engine: NATS cancel_tools subscription failed");
                    return;
                };
                loop {
                    tokio::select! {
                        _ = handle.closed.cancelled() => break,
                        msg = sub.next() => {
                            if msg.is_none() { break; }
                            if let Err(e) = handle.cancel_tools().await {
                                warn!(session_id = %sid, error = %e, "agent engine: NATS cancel_tools failed");
                            }
                        }
                    }
                }
            });
        }
        {
            let (nats, handle, sid) = (nats.clone(), Arc::clone(&handle), sid.clone());
            tokio::spawn(async move {
                let Ok(mut sub) = nats.subscribe_snapshot_requests(&sid).await else {
                    warn!(session_id = %sid, "agent engine: NATS snapshot subscription failed");
                    return;
                };
                loop {
                    tokio::select! {
                        _ = handle.closed.cancelled() => break,
                        msg = sub.next() => {
                            let Some(msg) = msg else { break };
                            let Some(reply_to) = msg.reply else { continue };
                            let snapshot = crate::events::StreamingSnapshot {
                                is_streaming: handle.is_streaming.load(Ordering::SeqCst),
                                partial_text: handle.streaming_text.lock().await.clone(),
                                events: handle.streaming_events.lock().await.clone(),
                            };
                            if let Ok(payload) = serde_json::to_vec(&snapshot) {
                                let _ = nats.client().publish(reply_to, payload.into()).await;
                            }
                        }
                    }
                }
            });
        }
        let graph = self.graph.clone();
        tokio::spawn(async move {
            let Ok(mut sub) = nats.subscribe_rpc_send(&sid).await else {
                warn!(session_id = %sid, "agent engine: NATS RPC subscription failed");
                return;
            };
            loop {
                tokio::select! {
                    _ = handle.closed.cancelled() => break,
                    msg = sub.next() => {
                        let Some(msg) = msg else { break };
                        let response = match serde_json::from_slice::<crate::events::ChatRpcRequest>(&msg.payload) {
                            Ok(request) => agent_rpc(&handle, &graph, &request).await,
                            Err(e) => crate::events::ChatRpcResponse {
                                success: false,
                                error: Some(format!("Invalid request: {e}")),
                            },
                        };
                        if let (Some(reply_to), Ok(payload)) = (msg.reply, serde_json::to_vec(&response)) {
                            let _ = nats.client().publish(reply_to, payload.into()).await;
                        }
                    }
                }
            }
        });
    }

    /// Reopens a session of the agent engine that is no longer live, from its
    /// persisted resume token (a new session when it never got one), then
    /// delivers `message`.
    async fn resume_agent_session(
        &self,
        node: &ChatSessionNode,
        message: &str,
        user_claims: Option<&crate::auth::jwt::Claims>,
    ) -> Result<()> {
        let provider_id = node
            .provider_id
            .clone()
            .unwrap_or_else(|| super::provider::resolver::CLAUDE_CODE.to_string());
        let provider = self.provider_for(&provider_id).await?;
        let remote_cwd = self.remote_cwd_of(&provider_id).await?;
        let sid = node.id.to_string();
        let (system_prompt, _) = self
            .build_system_prompt_anchored(
                node.execution_place,
                &node.cwd,
                node.project_slug.as_deref(),
                message,
                Some(&node.model),
                &sid,
                None,
            )
            .await;
        // A resumed session starts counting its turns again; the harness is the authority
        // on the index, and nothing says whether its model was named by the request.
        self.register_turn_router(
            &sid,
            &provider_id,
            &node.model,
            node.project_slug.as_deref(),
            OpeningTurn {
                // The mode the conversation was opened with stays its own across a resume.
                routing_mode: node.routing_mode.as_deref().and_then(|m| {
                    serde_json::from_value(serde_json::Value::String(m.to_owned())).ok()
                }),
                routing_pool: node
                    .routing_pool
                    .as_deref()
                    .and_then(|json| serde_json::from_str(json).ok()),
                explicit_model: false,
                permission_mode: node.permission_mode.as_deref(),
                // A provider the request named stays imposed across a resume (a conversation
                // opened with ticked models names its pilot without imposing it).
                provider_imposed: node.routed_by.as_deref() == Some("request")
                    && node.routing_pool.is_none(),
                moved_in: false,
                message,
                next_turn: 0,
            },
        )
        .await;
        // A resume keeps the access the session was opened with and can only narrow it.
        let access = super::provider::policy::SessionAccess::for_resume(node.access, None);
        let spec = self
            .build_agent_spec_with_access(
                AgentSpecInput {
                    cwd: &node.cwd,
                    model: &node.model,
                    system_prompt: &system_prompt,
                    permission_mode: node.permission_mode.as_deref(),
                    add_dirs: node.add_dirs.as_deref().unwrap_or(&[]),
                    user_claims,
                    session_id: &sid,
                    third_party: provider_id != super::provider::resolver::CLAUDE_CODE,
                    max_tokens: None,
                    kind: provider.kind(),
                    remote_cwd: remote_cwd.as_deref(),
                    hooks: Some(AgentHookScope {
                        project_slug: node.project_slug.clone(),
                        task_id: node
                            .spawned_by
                            .as_deref()
                            .and_then(parse_spawned_by)
                            .and_then(|ctx| ctx.task_id),
                        runner: false,
                    }),
                },
                access,
            )
            .await?;
        let tool_policy = super::provider::policy::wire_tool_policy(&spec.policy);
        // A resume is a new sending of the project's content: consent, guard
        // and journal apply again (a consent may have been revoked meanwhile).
        if let Err(e) = self
            .authorize_provider_use(ProviderUse {
                provider_id: &provider_id,
                model: &node.model,
                mode: spec.policy.mode,
                project_slug: node.project_slug.as_deref(),
                claims: user_claims,
                session_id: &sid,
            })
            .await
        {
            crate::auth::agent_tokens::revoke_session(&sid);
            return Err(e);
        }
        let token = node
            .resume_token
            .as_deref()
            .and_then(|raw| nexus_claude::agent::ResumeToken::from_wire(raw).ok());
        let nexus_missing = lacks_nexus_tools(
            provider.kind(),
            &spec,
            self.config.nexus_tools_path.as_deref(),
        );
        let session = match token {
            Some(token) => provider.resume(spec, token).await,
            None => provider.open(spec).await,
        }
        .map_err(anyhow::Error::new)?;
        let latest = self
            .graph
            .get_latest_chat_event_seq(node.id)
            .await
            .unwrap_or(0);
        self.finish_agent_open(
            &sid,
            &provider_id,
            provider.kind(),
            session,
            latest + 1,
            tool_policy,
            nexus_missing,
        )
        .await;
        let handle = self
            .agent_runtime
            .get(&sid)
            .await
            .ok_or_else(|| anyhow!("Session {sid} vanished while resuming"))?;
        // A resumed session is interactive: its persisted toggle, no limit (as on Claude Code).
        let auto_continue = self
            .graph
            .get_session_auto_continue(node.id)
            .await
            .unwrap_or(self.config.auto_continue);
        handle.configure_auto_continue(auto_continue, 0);
        handle.send_message(message).await
    }

    /// Close an active session: interrupt first, then disconnect and remove.
    ///
    /// T3 fix (Gap 5): call interrupt() BEFORE removing the session from
    /// active_sessions so the stream loop can observe the flag/token. Then
    /// disconnect with a 5s timeout — if the CLI hangs, drop the client to
    /// trigger SIGKILL via the Drop impl.
    pub async fn close_session(&self, session_id: &str) -> Result<()> {
        let closed = self.close_session_inner(session_id).await;
        // The neutral directory (if the session had one) goes with the session; a resume
        // makes it again.
        super::neutral_place::remove_for(session_id);
        closed
    }

    /// Delete a session: close it if it is live, remove what it keeps on disk (the
    /// transcript of a native session, P14), then the node. `Ok(false)`: no such
    /// session. A CLOSED session keeps its transcript (it stays resumable): only a
    /// deletion removes it, and before the node, which is what names it. A transcript
    /// that cannot be removed fails the deletion (the conversation would stay on disk
    /// with nothing left to find it); a retry removes both.
    pub async fn delete_session(&self, session_id: Uuid) -> Result<bool> {
        let _ = self.close_session(&session_id.to_string()).await;
        if let Some(node) = self.graph.get_chat_session(session_id).await? {
            if let (Some(root), Some(provider), Some(token)) = (
                self.native_transcripts.as_deref(),
                node.provider_id.as_deref(),
                node.resume_token.as_deref(),
            ) {
                super::provider::transcripts::remove_for_token(root, provider, token).map_err(
                    // The path may name a user; keep only the kind of failure.
                    |e| anyhow!("removing the session's transcript failed: {:?}", e.kind()),
                )?;
            }
        }
        self.graph.delete_chat_session(session_id).await
    }

    async fn close_session_inner(&self, session_id: &str) -> Result<()> {
        if let Ok(uuid) = Uuid::parse_str(session_id) {
            self.anchor_cache.forget(uuid);
        }
        self.turn_routing.remove(session_id);
        self.close_routing_decision(session_id).await;
        if self.agent_runtime.owns(session_id).await {
            crate::auth::agent_tokens::revoke_session(session_id);
            // Every instance learns the session is gone, as on the legacy path: the
            // `session_closed` the runtime emits is published like every event.
            self.agent_runtime.close(session_id).await?;
            return Ok(());
        }
        // 0. The session's MCP token dies with it — before anything that can
        //    fail, so a half-closed session never leaves a usable token behind.
        crate::auth::agent_tokens::revoke_session(session_id);

        // 1. Interrupt first (session still in map so interrupt() can find it)
        self.interrupt(session_id).await.ok();

        // 2. Brief wait for stream loop to break
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // 3. Remove session from active map
        let (client, protocol_run_id, protocol_state, events_tx) = {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .remove(session_id)
                .ok_or_else(|| anyhow!("Session {} not found or inactive", session_id))?;
            (
                session.client,
                session.protocol_run_id,
                session.protocol_state,
                session.events_tx,
            )
        };
        self.notify_attention(session_id, AttentionReason::SessionInactive);

        // A45: tell every client (and every instance) the session is gone.
        let closed = ChatEvent::SessionClosed {
            session_id: session_id.to_string(),
            reason: Some("closed".to_string()),
        };
        let _ = events_tx.send(closed.clone());
        if let Some(ref nats) = self.nats {
            nats.publish_chat_event(session_id, closed);
        }

        // 4. Finalize trajectory — fire-and-forget (non-blocking)
        //    Uses end_session_auto() so the collector computes the reward from
        //    actual buffered DecisionRecords (tool success rate, confidence, duration).
        //    SessionHints provides optional task metadata not available inside the loop.
        if let Some(ref collector) = *self.trajectory_collector.read().unwrap() {
            let hints = neural_routing_runtime::SessionHints {
                protocol_run_id,
                protocol_state,
                ..Default::default()
            };
            collector.end_session_auto(session_id.to_string(), hints);
            tracing::info!(
                session_id = %session_id,
                "Trajectory finalized with auto-computed reward on session close"
            );
        }

        // 5. Disconnect with 5s timeout — if it hangs, drop(client) triggers SIGKILL via Drop
        let disconnect_result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let mut c = client.lock().await;
            c.disconnect().await
        })
        .await;

        match disconnect_result {
            Ok(Ok(())) => {
                info!("Closed session {}", session_id);
            }
            Ok(Err(e)) => {
                warn!("Error disconnecting session {}: {}", session_id, e);
            }
            Err(_elapsed) => {
                warn!(
                    "Disconnect timed out for session {} (5s) — dropping client to force SIGKILL",
                    session_id
                );
                drop(client);
            }
        }

        Ok(())
    }

    /// Start a background task that cleans up timed-out sessions
    pub fn start_cleanup_task(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        let timeout = manager.config.session_timeout;
        let interval = timeout / 2; // Check at half the timeout interval

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                // Candidates by idle time first, under the read lock; the
                // busy checks below take other (async) locks, so they run
                // after it is released.
                let candidates: Vec<_> = {
                    let sessions = manager.active_sessions.read().await;
                    sessions
                        .iter()
                        .filter(|(_, s)| s.last_activity.elapsed() > timeout)
                        .map(|(id, s)| {
                            (
                                id.clone(),
                                s.last_activity.elapsed(),
                                s.is_streaming.clone(),
                                s.active_background_tasks.clone(),
                                s.cli_background_tasks.clone(),
                            )
                        })
                        .collect()
                };

                let mut expired = Vec::new();
                for (id, idle, streaming, background, cli_background) in candidates {
                    let is_streaming = streaming.load(Ordering::Relaxed);
                    // Whichever knows of more work wins: this server's own
                    // tracking, or what the CLI reports about itself.
                    let background_tasks = background
                        .lock()
                        .await
                        .len()
                        .max(cli_background.load(Ordering::Relaxed));
                    if session_is_expired(idle, timeout, is_streaming, background_tasks) {
                        expired.push(id);
                    } else {
                        debug!(
                            session_id = %id,
                            idle_secs = idle.as_secs(),
                            is_streaming,
                            background_tasks,
                            "Idle session kept alive: work still in progress"
                        );
                    }
                }

                for id in expired {
                    info!("Cleaning up timed-out session {}", id);
                    if let Err(e) = manager.close_session(&id).await {
                        warn!("Failed to close timed-out session {}: {}", id, e);
                    }
                }
            }
        });
    }

    /// Get the number of currently active sessions
    pub async fn active_session_count(&self) -> usize {
        self.active_sessions.read().await.len()
    }
}

/// Is the process `pid` still there? `None` when it cannot be told (no pid
/// was ever discovered for the task, or the platform has no cheap probe).
pub(crate) fn process_alive(pid: Option<u32>) -> Option<bool> {
    let pid = pid?;
    #[cfg(unix)]
    {
        // 0 and negative values address process GROUPS in kill(2); a pid
        // that does not fit a positive i32 is not a process we tracked.
        let Ok(pid) = i32::try_from(pid) else {
            return Some(false);
        };
        if pid <= 0 {
            return Some(false);
        }
        // SAFETY: signal 0 delivers nothing; it only checks that the pid
        // exists and that we may signal it.
        let rc = unsafe { libc::kill(pid, 0) };
        if rc == 0 {
            return Some(true);
        }
        // EPERM: the process exists but belongs to someone else.
        Some(std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM))
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        None
    }
}

/// Should a tracked background task be declared dead for having been silent?
///
/// Silence used to be enough: 30 minutes without output and the entry was
/// purged. But a build or a test run whose output goes to a file is silent
/// for its whole life. Purged while still running, it left the session with
/// "no background work", so the idle cleanup closed the session — killing the
/// CLI, the command with it, and showing the user "The CLI subprocess for this
/// session has exited". A silent task is now dead only when nothing says it
/// runs:
///
/// - its process is known → the process decides;
/// - its process is unknown (the pid claim often fails) → the CLI's own
///   report decides: while the CLI says it has background tasks, keep it.
///
/// A task that is kept wrongly is bounded by the session hard cap
/// (`IDLE_HARD_CAP_FACTOR`).
pub(crate) fn background_task_is_idle_dead(
    silent_for: chrono::Duration,
    idle_death: chrono::Duration,
    process_alive: Option<bool>,
    cli_reports_live_tasks: bool,
) -> bool {
    if silent_for < idle_death {
        return false;
    }
    match process_alive {
        Some(alive) => !alive,
        None => !cli_reports_live_tasks,
    }
}

/// Multiple of the idle timeout after which a session is closed even if it
/// still looks busy — a background task that never reports completion must not
/// pin a CLI subprocess forever.
const IDLE_HARD_CAP_FACTOR: u32 = 8;

/// Should an idle session be closed?
///
/// Idle time alone used to decide, and "activity" was only refreshed during
/// turns. A session whose agent had finished its turn but still had
/// background work running (sub-agents, background Bash, Monitor) looked idle
/// and was closed after `session_timeout` — killing the CLI subprocess, the
/// background work with it, and the completion notification that would have
/// woken the agent to report back. Busy sessions are now kept, up to a hard
/// cap.
pub(crate) fn session_is_expired(
    idle: Duration,
    timeout: Duration,
    is_streaming: bool,
    background_tasks: usize,
) -> bool {
    if idle <= timeout {
        return false;
    }
    if idle > timeout * IDLE_HARD_CAP_FACTOR {
        return true;
    }
    !is_streaming && background_tasks == 0
}

#[cfg(test)]
mod idle_expiry_tests {
    use super::{background_task_is_idle_dead, process_alive, session_is_expired};
    use std::time::Duration;

    const T: Duration = Duration::from_secs(1800);

    fn mins(m: i64) -> chrono::Duration {
        chrono::Duration::minutes(m)
    }

    #[test]
    fn a_task_that_emitted_recently_is_alive_whatever_else_is_known() {
        for process in [None, Some(true), Some(false)] {
            for cli in [false, true] {
                assert!(!background_task_is_idle_dead(
                    mins(29),
                    mins(30),
                    process,
                    cli
                ));
            }
        }
    }

    /// The failure this guards: a build that writes to a file says nothing for
    /// 40 minutes. Its process runs → it is not dead.
    #[test]
    fn a_silent_task_whose_process_runs_is_not_dead() {
        assert!(!background_task_is_idle_dead(
            mins(40),
            mins(30),
            Some(true),
            false
        ));
        // The process is the stronger evidence: it wins over a CLI report
        // that is empty (or that never came).
        assert!(!background_task_is_idle_dead(
            mins(600),
            mins(30),
            Some(true),
            false
        ));
    }

    #[test]
    fn a_silent_task_whose_process_is_gone_is_dead() {
        assert!(background_task_is_idle_dead(
            mins(30),
            mins(30),
            Some(false),
            false
        ));
        // Even if the CLI still reports work: that is some other task.
        assert!(background_task_is_idle_dead(
            mins(31),
            mins(30),
            Some(false),
            true
        ));
    }

    /// The pid claim often fails ("no new PID in diff"): then the CLI's own
    /// report is the only evidence.
    #[test]
    fn a_silent_task_without_a_pid_follows_the_cli_report() {
        assert!(!background_task_is_idle_dead(
            mins(40),
            mins(30),
            None,
            true
        ));
        // Nothing says it runs: the previous behaviour, unchanged.
        assert!(background_task_is_idle_dead(
            mins(40),
            mins(30),
            None,
            false
        ));
    }

    #[test]
    fn process_alive_tells_a_running_process_from_a_dead_one() {
        assert_eq!(process_alive(None), None);

        #[cfg(unix)]
        {
            assert_eq!(process_alive(Some(std::process::id())), Some(true));
            // kill(0, ..) would address our whole process group.
            assert_eq!(process_alive(Some(0)), Some(false));
            assert_eq!(process_alive(Some(u32::MAX)), Some(false));

            // A child that has exited and been reaped no longer exists.
            let mut child = std::process::Command::new("true")
                .spawn()
                .expect("spawn `true`");
            let pid = child.id();
            child.wait().expect("wait for `true`");
            assert_eq!(process_alive(Some(pid)), Some(false));
        }
    }

    /// What the cleanup feeds `session_is_expired`: the larger of this
    /// server's tracking and the CLI's report. With the tracking emptied (the
    /// old purge) but the CLI reporting one task, the session is kept.
    #[test]
    fn idle_session_is_kept_while_the_cli_reports_background_work() {
        let tracked = 0usize;
        let cli_reported = 1usize;
        assert!(!session_is_expired(
            T * 2,
            T,
            false,
            tracked.max(cli_reported)
        ));
        // Without the CLI's report, the emptied tracking alone closes it.
        assert!(session_is_expired(T * 2, T, false, tracked));
    }

    #[test]
    fn active_session_is_kept() {
        assert!(!session_is_expired(Duration::from_secs(60), T, false, 0));
    }

    #[test]
    fn idle_session_without_work_expires() {
        assert!(session_is_expired(T + Duration::from_secs(1), T, false, 0));
    }

    #[test]
    fn idle_session_with_background_work_is_kept() {
        // The failure mode: agent's turn over, sub-agents still running.
        assert!(!session_is_expired(T * 2, T, false, 3));
    }

    #[test]
    fn streaming_session_is_kept() {
        assert!(!session_is_expired(T * 2, T, true, 0));
    }

    #[test]
    fn stuck_background_work_is_bounded_by_the_hard_cap() {
        assert!(session_is_expired(T * 9, T, true, 1));
    }
}

/// Parse a raw SDK control message into a [`ChatEvent::PermissionRequest`] if it is
/// a `can_use_tool` request.  Returns `None` for any other subtype.
///
/// This is the pure-logic core extracted from the `handle_control_msg` closure inside
/// `stream_response`.  Having it as a standalone function makes it directly testable
/// Store a pending permission input in the shared map.
/// Extracted as a standalone async fn so that `tokio::select!` blocks don't
/// struggle with type inference on `Arc<Mutex<HashMap>>` inline locks.
async fn store_pending_perm_input(
    map: &Arc<tokio::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>>,
    id: &str,
    input: &serde_json::Value,
) {
    map.lock().await.insert(id.to_string(), input.clone());
}

/// Relay a permission request / question as a light `attention_changed` on
/// the general bus. Ids only: the command or question text never leaves the
/// session's own WebSocket.
fn notify_attention_for_chat_event(
    emitter: &Option<Arc<dyn crate::events::EventEmitter>>,
    session_id: &str,
    event: &ChatEvent,
) {
    let reason = match event {
        ChatEvent::PermissionRequest { .. } => AttentionReason::PermissionRequest,
        ChatEvent::AskUserQuestion { .. } => AttentionReason::AskUserQuestion,
        _ => return,
    };
    notify_attention(
        emitter,
        AttentionSubject::Session(session_id.to_string()),
        reason,
    );
}

/// without needing to spin up a full streaming session.
///
/// The caller is still responsible for:
/// - persisting the event to `events_to_persist`
/// - broadcasting via `emit_chat`
fn parse_permission_control_msg(
    control_msg: &serde_json::Value,
    current_parent: Option<String>,
) -> Option<ChatEvent> {
    // Extract the request data (may be nested under "request")
    let request_data = if control_msg.get("request").is_some() {
        control_msg
            .get("request")
            .cloned()
            .unwrap_or_else(|| control_msg.clone())
    } else {
        control_msg.clone()
    };

    if request_data.get("subtype").and_then(|v| v.as_str()) != Some("can_use_tool") {
        return None;
    }

    let tool_name = request_data
        .get("toolName")
        .or_else(|| request_data.get("tool_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let input = request_data
        .get("input")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    let request_id = control_msg
        .get("requestId")
        .or_else(|| control_msg.get("request_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // AskUserQuestion is NOT a permission — it's a user interaction tool.
    // Instead of emitting PermissionRequest (which shows the approval dialog),
    // emit a dedicated AskUserQuestion event so the frontend renders the
    // question widget directly. The caller will auto-allow this request.
    if tool_name == "AskUserQuestion" {
        let questions = input
            .get("questions")
            .cloned()
            .unwrap_or(serde_json::json!([]));
        // Extract the tool_call_id from the control message — this is the tool_use ID
        // that the frontend needs to send the tool_result response back.
        let tool_call_id = request_data
            .get("toolUseId")
            .or_else(|| request_data.get("tool_use_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return Some(ChatEvent::AskUserQuestion {
            id: request_id,
            tool_call_id,
            questions,
            input,
            parent_tool_use_id: current_parent,
            synthetic: None,
        });
    }

    Some(ChatEvent::PermissionRequest {
        id: request_id,
        tool: tool_name,
        input,
        parent_tool_use_id: current_parent,
        category: None,
        canonical: None,
    })
}

/// Whether a session's `spawned_by` marks it as a PlanRunner session
/// (`{"type":"runner",…}`). Parsed, not substring-matched, so a user session
/// whose metadata merely mentions "runner" is not mistaken for one.
pub(crate) fn is_runner_spawned(spawned_by: Option<&str>) -> bool {
    spawned_by
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| {
            v.get("type")
                .and_then(|t| t.as_str())
                .map(|t| t == "runner")
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod runner_spawn_tests {
    use super::is_runner_spawned;

    #[test]
    fn detects_runner_sessions() {
        assert!(is_runner_spawned(Some(
            r#"{"type":"runner","run_id":"x","plan_id":"y"}"#
        )));
    }

    #[test]
    fn user_and_malformed_sessions_are_not_runners() {
        assert!(!is_runner_spawned(None));
        assert!(!is_runner_spawned(Some("")));
        assert!(!is_runner_spawned(Some(
            r#"{"type":"user","note":"runner"}"#
        )));
        assert!(!is_runner_spawned(Some("runner")));
    }
}

/// Meilisearch filter selecting one conversation in `nexus_messages`.
/// `conversation_id` is stored data (not trusted): it is quoted/escaped so a
/// value containing `"` cannot widen the filter.
pub(crate) fn legacy_messages_filter(conversation_id: &str) -> String {
    crate::meilisearch::client::meili_filter_eq("conversation_id", conversation_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_messages_filter_escapes_the_conversation_id() {
        assert_eq!(legacy_messages_filter("abc"), "conversation_id = \"abc\"");
        let evil = legacy_messages_filter("x\" OR conversation_id != \"y");
        assert_eq!(
            evil,
            "conversation_id = \"x\\\" OR conversation_id != \\\"y\""
        );
    }

    #[tokio::test]
    async fn test_enrichment_project_id_resolves_the_session_slug() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let project = crate::test_helpers::test_project_named("Enriched");
        store.create_project(&project).await.unwrap();
        assert_eq!(
            enrichment_project_id(&store, Some(&project.slug)).await,
            Some(project.id)
        );
        assert_eq!(enrichment_project_id(&store, Some("unknown")).await, None);
        assert_eq!(enrichment_project_id(&store, None).await, None);
    }
    use crate::neo4j::models::ChatSessionNode;
    use crate::test_helpers::{
        mock_app_state, mock_app_state_with_graph, test_chat_session, test_project,
    };
    use nexus_claude::{
        AssistantMessage, ContentBlock, ContentValue, PermissionMode, TextContent, ThinkingContent,
        ToolResultContent, ToolUseContent,
    };
    use std::path::PathBuf;
    use std::time::Duration;

    fn test_config() -> ChatConfig {
        ChatConfig {
            provider_path: Default::default(),
            mcp_server_path: PathBuf::from("/usr/bin/mcp_server"),
            nexus_tools_path: None,
            nexus_browser_path: None,
            default_model: "claude-sonnet-4-6".into(),
            max_sessions: 10,
            session_timeout: Duration::from_secs(1800),
            neo4j_uri: "bolt://localhost:7687".into(),
            neo4j_user: "neo4j".into(),
            neo4j_password: "test".into(),
            meilisearch_url: "http://localhost:7700".into(),
            meilisearch_key: "key".into(),
            nats_url: None,
            max_turns: 10,
            permission: crate::chat::config::PermissionConfig::default(),
            auto_continue: false,
            retry: crate::chat::config::RetryConfig::default(),
            process_path: None,
            claude_cli_path: None,
            auto_update_cli: false,
            auto_update_app: true,
            jwt_secret: None,
            server_port: 8080,
            session_token_expiry_secs: 86400,
        }
    }

    #[test]
    fn test_resolve_model_with_override() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert_eq!(
            manager.resolve_model(Some("claude-sonnet-4-6")),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_resolve_model_default() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert_eq!(manager.resolve_model(None), "claude-sonnet-4-6");
    }

    #[tokio::test]
    async fn test_build_options_uses_config_permission_default() {
        // Default config uses "default" permission mode (safe-by-default)
        // and pre-approves MCP tools out of the box
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let opts = manager
            .build_options(
                "/tmp",
                "claude-opus-4-6",
                "test prompt",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));
        assert_eq!(opts.allowed_tools, vec!["mcp__project-orchestrator__*"]);
        assert!(opts.disallowed_tools.is_empty());
    }

    // ── read-only sessions: the harness refuses, the prompt is not involved ──

    #[tokio::test]
    async fn read_only_access_adds_the_deny_list_to_the_cli_options_and_keeps_the_mode() {
        use crate::chat::provider::policy::SessionAccess;
        let state = mock_app_state();
        let mut config = test_config();
        config.permission.disallowed_tools = vec!["Bash(rm -rf *)".into()];
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        for mode in [Some("bypassPermissions"), Some("plan"), None] {
            let opts = manager
                .build_options_with_access(
                    "/tmp",
                    "m",
                    "p",
                    Some("resume-id"),
                    mode,
                    None,
                    &[],
                    None,
                    None,
                    SessionAccess::ReadOnly,
                )
                .await;
            for tool in ["Edit", "Write", "MultiEdit", "NotebookEdit", "Bash", "Task"] {
                assert!(opts.disallowed_tools.iter().any(|d| d == tool), "{tool}");
            }
            assert!(opts.disallowed_tools.iter().any(|d| d == "Bash(rm -rf *)"));
            // The mode itself is not rewritten: the deny list is what protects.
            if mode == Some("bypassPermissions") {
                assert!(matches!(
                    opts.permission_mode,
                    PermissionMode::BypassPermissions
                ));
            }
        }
    }

    #[tokio::test]
    async fn normal_access_leaves_the_cli_options_as_build_options_makes_them() {
        use crate::chat::provider::policy::SessionAccess;
        let state = mock_app_state();
        let mut config = test_config();
        config.permission.disallowed_tools = vec!["Bash(rm -rf *)".into()];
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let plain = manager
            .build_options("/tmp", "m", "p", None, None, None, &[], None, None)
            .await;
        let with = manager
            .build_options_with_access(
                "/tmp",
                "m",
                "p",
                None,
                None,
                None,
                &[],
                None,
                None,
                SessionAccess::Normal,
            )
            .await;
        assert_eq!(plain.disallowed_tools, with.disallowed_tools);
        assert_eq!(plain.disallowed_tools, vec!["Bash(rm -rf *)"]);
    }

    #[tokio::test]
    async fn a_read_only_agent_spec_denies_writes_in_trust_and_a_normal_one_does_not() {
        use crate::chat::provider::policy::SessionAccess;
        use nexus_claude::agent::{PolicyDecision, ProviderKind, ToolCategory};
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let spec_for = |access: SessionAccess| {
            let manager = &manager;
            async move {
                manager
                    .build_agent_spec_with_access(
                        AgentSpecInput {
                            cwd: "/tmp",
                            model: "m",
                            system_prompt: "p",
                            permission_mode: Some("bypassPermissions"),
                            add_dirs: &[],
                            user_claims: None,
                            session_id: "ro-s1",
                            third_party: false,
                            max_tokens: None,
                            kind: ProviderKind::ClaudeCode,
                            remote_cwd: None,
                            hooks: None,
                        },
                        access,
                    )
                    .await
                    .unwrap()
            }
        };
        let read_only = spec_for(SessionAccess::ReadOnly).await;
        let normal = spec_for(SessionAccess::Normal).await;
        // A tool the prompt never mentions: the policy refuses it all the same.
        let write = |spec: &nexus_claude::agent::SessionSpec| {
            spec.policy
                .decide("Write", Some("/tmp/x"), ToolCategory::Edit)
        };
        assert_eq!(write(&read_only), PolicyDecision::Deny);
        assert_eq!(write(&normal), PolicyDecision::Allow);
        assert_eq!(
            read_only
                .policy
                .decide("Read", Some("/tmp/x"), ToolCategory::Read),
            PolicyDecision::Allow
        );
    }

    #[tokio::test]
    async fn test_build_options_uses_config_permission_custom() {
        let state = mock_app_state();
        let mut config = test_config();
        config.permission = crate::chat::config::PermissionConfig {
            mode: "default".into(),
            allowed_tools: vec!["Bash(git *)".into(), "Read".into()],
            disallowed_tools: vec!["Bash(rm -rf *)".into()],
        };
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        let opts = manager
            .build_options(
                "/tmp",
                "claude-opus-4-6",
                "test prompt",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));
        assert_eq!(opts.allowed_tools, vec!["Bash(git *)", "Read"]);
        assert_eq!(opts.disallowed_tools, vec!["Bash(rm -rf *)"]);
    }

    #[tokio::test]
    async fn test_build_options_with_permission_override() {
        // Global config is Default, session overrides to BypassPermissions
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let opts = manager
            .build_options(
                "/tmp",
                "claude-opus-4-6",
                "prompt",
                None,
                Some("bypassPermissions"),
                None,
                &[],
                None,
                None,
            )
            .await;
        assert!(matches!(
            opts.permission_mode,
            PermissionMode::BypassPermissions
        ));

        // Without override, falls back to global (Default)
        let opts = manager
            .build_options(
                "/tmp",
                "claude-opus-4-6",
                "prompt",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));
    }

    #[tokio::test]
    async fn test_get_permission_config_returns_defaults() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let config = manager.get_permission_config().await;
        assert_eq!(config.mode, "default");
        // MCP tools are pre-approved by default
        assert_eq!(config.allowed_tools, vec!["mcp__project-orchestrator__*"]);
        assert!(config.disallowed_tools.is_empty());
    }

    #[tokio::test]
    async fn test_update_permission_config_changes_mode() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let new_config = crate::chat::config::PermissionConfig {
            mode: "default".into(),
            allowed_tools: vec!["Read".into(), "Bash(git *)".into()],
            disallowed_tools: vec!["Bash(rm -rf *)".into()],
        };
        let updated = manager.update_permission_config(new_config).await.unwrap();
        assert_eq!(updated.mode, "default");
        assert_eq!(updated.allowed_tools, vec!["Read", "Bash(git *)"]);
        assert_eq!(updated.disallowed_tools, vec!["Bash(rm -rf *)"]);

        // Verify getter returns the updated config
        let fetched = manager.get_permission_config().await;
        assert_eq!(fetched.mode, "default");
        assert_eq!(fetched.allowed_tools, vec!["Read", "Bash(git *)"]);
    }

    #[tokio::test]
    async fn test_update_permission_config_rejects_invalid_mode() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let bad_config = crate::chat::config::PermissionConfig {
            mode: "yolo".into(),
            ..Default::default()
        };
        let result = manager.update_permission_config(bad_config).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid permission mode 'yolo'"));
    }

    #[tokio::test]
    async fn test_update_permission_config_affects_build_options() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // Initially Default (safe-by-default)
        let opts = manager
            .build_options(
                "/tmp",
                "claude-opus-4-6",
                "prompt",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;
        assert!(matches!(opts.permission_mode, PermissionMode::Default));

        // Update to BypassPermissions mode at runtime
        manager
            .update_permission_config(crate::chat::config::PermissionConfig {
                mode: "bypassPermissions".into(),
                allowed_tools: vec![],
                disallowed_tools: vec![],
            })
            .await
            .unwrap();

        // New build_options should reflect the update
        let opts = manager
            .build_options(
                "/tmp",
                "claude-opus-4-6",
                "prompt",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;
        assert!(matches!(
            opts.permission_mode,
            PermissionMode::BypassPermissions
        ));
        assert!(opts.allowed_tools.is_empty());
    }

    #[tokio::test]
    async fn test_build_system_prompt_no_project() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let (prompt, note_ids) = manager
            .build_system_prompt(None, "test", None, None, None)
            .await;
        assert!(prompt.contains("Project Orchestrator"));
        assert!(prompt.contains("EXCLUSIVELY the Project Orchestrator MCP tools"));
        assert!(!prompt.contains("Active Project"));
        assert!(note_ids.is_empty(), "No project → no included note IDs");
    }

    // ── Anchor context wiring (PO_ANCHOR_CONTEXT) ─────────────────────────

    use crate::chat::anchor_resolver::AnchorContextMode;

    fn anchor_in(mode: AnchorContextMode) -> crate::chat::anchor_resolver::AnchorSession {
        crate::chat::anchor_resolver::AnchorSession::new(mode, Default::default())
    }

    fn manager_in(mode: AnchorContextMode, state: &crate::AppState) -> ChatManager {
        ChatManager::new_without_memory(state.neo4j.clone(), state.meili.clone(), test_config())
            .with_anchor_context_mode(mode)
    }

    #[tokio::test]
    async fn shadow_and_off_prompts_are_byte_identical_to_the_historical_one() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();
        let sid = Uuid::new_v4().to_string();
        let plain = manager_in(AnchorContextMode::Off, &state)
            .build_system_prompt(Some(&project.slug), "plan", None, Some(&sid), None)
            .await;
        for mode in [AnchorContextMode::Off, AnchorContextMode::Shadow] {
            let m = manager_in(mode, &state);
            let anchored = m
                .build_system_prompt_anchored(
                    crate::chat::neutral_place::ExecutionPlace::Project,
                    "/tmp/test-project",
                    Some(&project.slug),
                    "plan",
                    None,
                    &sid,
                    None,
                )
                .await;
            assert_eq!(anchored.0.as_bytes(), plain.0.as_bytes(), "{mode:?}");
        }
    }

    #[tokio::test]
    async fn on_mode_puts_a_stable_anchor_map_in_the_system_prompt() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();
        let sid = Uuid::new_v4().to_string();
        let m = manager_in(AnchorContextMode::On, &state);
        let build = || {
            m.build_system_prompt_anchored(
                crate::chat::neutral_place::ExecutionPlace::Project,
                "/tmp/test-project",
                Some(&project.slug),
                "plan",
                None,
                &sid,
                None,
            )
        };
        let (first, _) = build().await;
        let (second, _) = build().await;
        let base = m
            .build_system_prompt(Some(&project.slug), "plan", None, Some(&sid), None)
            .await
            .0;
        assert!(
            first.starts_with(&base),
            "the historical prompt stays the prefix"
        );
        assert!(first.contains("<untrusted_data") && first.contains("anchor_map"));
        assert_eq!(first, second, "same anchors, same epochs: same bytes");
    }

    #[tokio::test]
    async fn on_mode_opens_the_enrichment_with_the_live_block_and_shadow_never_does() {
        let state = mock_app_state();
        let mut s = crate::test_helpers::test_chat_session(None);
        s.execution_place = crate::chat::neutral_place::ExecutionPlace::Neutral;
        s.cwd = crate::chat::neutral_place::root()
            .join("s")
            .display()
            .to_string();
        state.neo4j.create_chat_session(&s).await.unwrap();
        let graph: Arc<dyn GraphStore> = state.neo4j.clone();
        let pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
        let sid = s.id.to_string();
        let on = enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            "hello",
            Default::default(),
            Default::default(),
            &anchor_in(AnchorContextMode::On),
        )
        .await
        .expect("a neutral session without anchor gets the notice");
        assert!(
            on.starts_with(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT),
            "{on}"
        );
        for mode in [AnchorContextMode::Off, AnchorContextMode::Shadow] {
            let out = enrichment_for_turn(
                &graph,
                &pipeline,
                &sid,
                "hello",
                Default::default(),
                Default::default(),
                &anchor_in(mode),
            )
            .await;
            assert!(
                out.is_none_or(|o| !o.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT)),
                "{mode:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_neutral_session_is_not_given_a_project_from_its_cwd_in_on_mode() {
        let state = mock_app_state();
        let mut p = crate::test_helpers::test_project_named("rooted");
        p.root_path = "/definitely/not/neutral/rooted".into();
        state.neo4j.create_project(&p).await.unwrap();
        let graph: Arc<dyn GraphStore> = state.neo4j.clone();
        let neutral = crate::chat::neutral_place::root()
            .join("n")
            .display()
            .to_string();
        let slug = infer_session_project(
            graph.as_ref(),
            AnchorContextMode::On,
            None,
            crate::chat::neutral_place::ExecutionPlace::Neutral,
            &neutral,
        )
        .await;
        assert_eq!(slug, None);
        // even when a (wrong) cwd points inside a project: neutral stays neutral
        let slug = infer_session_project(
            graph.as_ref(),
            AnchorContextMode::On,
            None,
            crate::chat::neutral_place::ExecutionPlace::Neutral,
            "/definitely/not/neutral/rooted/src",
        )
        .await;
        assert_eq!(slug, None);
    }
    /// Records the project slug each enrichment is handed.
    struct SlugProbe(Arc<std::sync::Mutex<Vec<Option<String>>>>);

    #[async_trait::async_trait]
    impl crate::chat::enrichment::ParallelEnrichmentStage for SlugProbe {
        async fn execute(
            &self,
            input: &crate::chat::enrichment::EnrichmentInput,
        ) -> anyhow::Result<crate::chat::enrichment::StageOutput> {
            self.0.lock().unwrap().push(input.project_slug.clone());
            Ok(crate::chat::enrichment::StageOutput::new("slug-probe"))
        }
        fn name(&self) -> &str {
            "slug-probe"
        }
        fn is_enabled(&self, _config: &crate::chat::enrichment::EnrichmentConfig) -> bool {
            true
        }
    }

    fn neutral_session() -> crate::neo4j::models::ChatSessionNode {
        let mut s = crate::test_helpers::test_chat_session(None);
        s.execution_place = crate::chat::neutral_place::ExecutionPlace::Neutral;
        s.cwd = crate::chat::neutral_place::root()
            .join(s.id.to_string())
            .display()
            .to_string();
        s
    }

    async fn anchor_plan(graph: &Arc<dyn GraphStore>, session: Uuid, plan: String) {
        use crate::chat::anchor::{AnchorActor, AnchorOp, AnchorRole, AnchorTargetType, NewAnchor};
        graph
            .apply_anchor_op(
                session,
                AnchorOp::Add(NewAnchor::new(
                    AnchorTargetType::Plan,
                    plan,
                    [AnchorRole::Focus],
                    AnchorActor::User,
                    "t",
                )),
            )
            .await
            .unwrap();
    }

    async fn human_anchors_project(
        graph: &Arc<dyn GraphStore>,
        session: Uuid,
        project: Uuid,
        by: crate::chat::anchor::AnchorActor,
    ) {
        use crate::chat::anchor::{AnchorOp, AnchorRole, AnchorTargetType, NewAnchor};
        graph
            .apply_anchor_op(
                session,
                AnchorOp::Add(NewAnchor::new(
                    AnchorTargetType::Project,
                    project.to_string(),
                    [AnchorRole::Focus],
                    by,
                    "t",
                )),
            )
            .await
            .unwrap();
    }

    /// The environment is read when the manager is built, nowhere else: a turn
    /// takes the mode of its session (`ActiveSession::anchor`), which the manager
    /// fixed when the session was opened or resumed.
    #[tokio::test]
    async fn the_anchor_mode_is_fixed_per_session_and_the_environment_is_not_read_per_turn() {
        let state = mock_app_state();
        let m = manager_in(AnchorContextMode::On, &state);
        assert_eq!(m.anchor_session().mode, AnchorContextMode::On);
        // every other call site of the environment reader would be a per-turn read
        let source = include_str!("manager.rs");
        let needle = concat!("AnchorContextMode::", "from_env()");
        assert_eq!(
            source.matches(needle).count(),
            2,
            "only the two constructors read the environment"
        );
        // a live session keeps the mode it was opened with
        assert_eq!(
            manager_in(AnchorContextMode::Off, &state)
                .anchor_session()
                .mode,
            AnchorContextMode::Off
        );
    }

    #[tokio::test]
    async fn an_explicit_slug_that_resolves_to_no_project_is_kept_for_the_enrichment_in_on_mode() {
        let state = mock_app_state();
        let mut s = neutral_session();
        s.project_slug = Some("ghost".into());
        state.neo4j.create_chat_session(&s).await.unwrap();
        let graph: Arc<dyn GraphStore> = state.neo4j.clone();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
        pipeline.add_parallel_stage(Box::new(SlugProbe(seen.clone())));
        let anchor = anchor_in(AnchorContextMode::On);
        let sid = s.id.to_string();
        let live = enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            "hello",
            Default::default(),
            Default::default(),
            &anchor,
        )
        .await
        .expect("the notice");
        // as before the resolver: the slug reaches the stages as it is
        assert_eq!(
            seen.lock().unwrap().last().unwrap().as_deref(),
            Some("ghost")
        );
        // ... without widening the resolver: no project, the neutral notice
        assert!(
            live.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT),
            "{live}"
        );
        // the warning is given once per session
        assert!(!anchor.cache.first_time(s.id, "unresolved_explicit_slug"));
        enrichment_for_turn(
            &graph,
            &pipeline,
            &sid,
            "again",
            Default::default(),
            Default::default(),
            &anchor,
        )
        .await;
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn an_anchor_put_mid_session_is_in_the_next_turn_the_hooks_and_the_next_map() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let project = test_project();
        mock.create_project(&project).await.unwrap();
        let plan = crate::test_helpers::seed_plan_with_task(&mock, project.id).await;
        let s = neutral_session();
        mock.create_chat_session(&s).await.unwrap();
        let graph: Arc<dyn GraphStore> = mock.clone();
        let pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
        let m =
            ChatManager::new_without_memory(graph.clone(), mock_app_state().meili, test_config())
                .with_anchor_context_mode(AnchorContextMode::On);
        let anchor = m.anchor_session();
        let sid = s.id.to_string();
        let hooks = m.hook_session_project(&sid).expect("mode on");
        let turn = |text: &'static str| {
            enrichment_for_turn(
                &graph,
                &pipeline,
                &sid,
                text,
                Default::default(),
                Default::default(),
                &anchor,
            )
        };
        let system = || async {
            m.build_system_prompt_anchored(
                s.execution_place,
                &s.cwd,
                None,
                "hello",
                None,
                &sid,
                None,
            )
            .await
            .0
        };
        // turn 1: nothing anchored
        let first = turn("one").await.expect("the notice");
        assert!(first.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT));
        assert_eq!(hooks.project_id().await, None);
        let map_before = system().await;
        assert!(!map_before.contains("Plan mid"));

        human_anchors_project(
            &graph,
            s.id,
            project.id,
            crate::chat::anchor::AnchorActor::User,
        )
        .await;
        anchor_plan(&graph, s.id, plan).await;

        // the very next turn: live block with the new anchor, hooks on the new project
        let second = turn("two").await.expect("live block");
        assert!(second.contains("Task mid"), "{second}");
        assert!(!second.contains(crate::chat::anchor_resolver::NOTICE_NO_CONTEXT));
        assert_eq!(hooks.project_id().await, Some(project.id), "no 30 s wait");
        // the system prompt is rebuilt at the next prompt rebuild (resume,
        // compaction) with the current anchors
        let map_after = system().await;
        assert!(map_after.contains("Plan mid"), "{map_after}");
    }

    #[tokio::test]
    async fn on_mode_turn_cost_in_store_reads_with_and_without_the_cache() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let project = test_project();
        mock.create_project(&project).await.unwrap();
        let s = neutral_session();
        mock.create_chat_session(&s).await.unwrap();
        let graph: Arc<dyn GraphStore> = mock.clone();
        human_anchors_project(
            &graph,
            s.id,
            project.id,
            crate::chat::anchor::AnchorActor::User,
        )
        .await;
        let plan = crate::test_helpers::seed_plan_with_task(&mock, project.id).await;
        anchor_plan(&graph, s.id, plan).await;
        let pipeline = crate::chat::enrichment::EnrichmentPipeline::new(Default::default());
        let reads = || mock.store_reads.load(std::sync::atomic::Ordering::Relaxed);
        let sid = s.id.to_string();
        let mut per_turn = Vec::new();
        // a cache of its own for every turn = no cache
        for _ in 0..3 {
            let b = reads();
            enrichment_for_turn(
                &graph,
                &pipeline,
                &sid,
                "x",
                Default::default(),
                Default::default(),
                &anchor_in(AnchorContextMode::On),
            )
            .await;
            per_turn.push(reads() - b);
        }
        let shared = anchor_in(AnchorContextMode::On);
        let mut cached = Vec::new();
        for _ in 0..3 {
            let b = reads();
            enrichment_for_turn(
                &graph,
                &pipeline,
                &sid,
                "x",
                Default::default(),
                Default::default(),
                &shared,
            )
            .await;
            cached.push(reads() - b);
        }
        eprintln!("on-mode reads per turn: without cache {per_turn:?}, with cache {cached:?}");
        assert!(cached[1] < per_turn[1], "{cached:?} vs {per_turn:?}");
        assert_eq!(cached[1], cached[2], "a stable cost once cached");
        // the shadow, in the same conditions: only the first turn reads
        let shadow = anchor_in(AnchorContextMode::Shadow);
        let mut sh = Vec::new();
        for _ in 0..3 {
            let b = reads();
            enrichment_for_turn(
                &graph,
                &pipeline,
                &sid,
                "x",
                Default::default(),
                Default::default(),
                &shadow,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            sh.push(reads() - b);
        }
        eprintln!("shadow reads per turn (turn included): {sh:?}");
        assert_eq!(sh[1], sh[2], "the shadow adds nothing after the first run");
    }

    #[tokio::test]
    async fn test_build_system_prompt_with_project() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (prompt, _note_ids) = manager
            .build_system_prompt(Some(&project.slug), "help me plan", None, None, None)
            .await;

        // Contains the base prompt
        assert!(prompt.contains("EXCLUSIVELY the Project Orchestrator MCP tools"));
        // Contains dynamic context section (either oneshot or fallback)
        assert!(prompt.contains("---"));
        // The project name should appear somewhere in the dynamic context
        assert!(prompt.contains(&project.name));
    }

    #[tokio::test]
    async fn test_session_not_active_by_default() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert!(!manager.is_session_active("nonexistent").await);
    }

    // ====================================================================
    // build_options
    // ====================================================================

    #[tokio::test]
    async fn test_build_options_basic() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let options = manager
            .build_options(
                "/tmp/project",
                "claude-opus-4-6",
                "System prompt here",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;

        assert_eq!(options.model, Some("claude-opus-4-6".into()));
        assert_eq!(options.cwd, Some(PathBuf::from("/tmp/project")));
        assert!(options.resume.is_none());
        assert!(options.mcp_servers.contains_key("project-orchestrator"));
    }

    #[tokio::test]
    async fn test_build_options_with_resume() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let options = manager
            .build_options(
                "/tmp/project",
                "claude-opus-4-6",
                "System prompt",
                Some("cli-session-abc"),
                None,
                None,
                &[],
                None,
                None,
            )
            .await;

        assert_eq!(options.resume, Some("cli-session-abc".into()));
    }

    #[tokio::test]
    async fn test_build_options_mcp_server_config() {
        let state = mock_app_state();
        let config = test_config();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        let options = manager
            .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
            .await;

        let mcp = options.mcp_servers.get("project-orchestrator").unwrap();
        match mcp {
            McpServerConfig::Stdio { command, env, .. } => {
                assert_eq!(command, "/usr/bin/mcp_server");
                let env = env.as_ref().unwrap();
                assert!(env.contains_key("PO_SERVER_URL"));
            }
            _ => panic!("Expected Stdio MCP config"),
        }
    }

    // ── The knowledge graph's hooks on the agent engine ─────────────────────

    fn scope(runner: bool) -> Option<AgentHookScope> {
        Some(AgentHookScope {
            project_slug: Some("proj".into()),
            task_id: None,
            runner,
        })
    }

    #[tokio::test]
    async fn the_agent_engine_gives_the_graph_hooks_only_to_providers_that_run_hooks() {
        use nexus_claude::agent::ProviderKind;
        let state = mock_app_state();
        let mut config = test_config();
        config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let claims = crate::auth::jwt::Claims::service_account("hooks");
        let has_hooks = |kind: ProviderKind, remote: Option<&'static str>, hooks| {
            let manager = &manager;
            let claims = &claims;
            async move {
                manager
                    .build_agent_spec(AgentSpecInput {
                        cwd: "/tmp",
                        model: "m",
                        system_prompt: "p",
                        permission_mode: None,
                        add_dirs: &[],
                        user_claims: Some(claims),
                        session_id: "hooks-s1",
                        third_party: true,
                        max_tokens: None,
                        kind,
                        remote_cwd: remote,
                        hooks,
                    })
                    .await
                    .unwrap()
                    .hooks
                    .is_some()
            }
        };
        // Native runs them in its loop; so does Claude Code (its hook protocol).
        assert!(has_hooks(ProviderKind::Native, None, scope(false)).await);
        assert!(has_hooks(ProviderKind::ClaudeCode, None, scope(false)).await);
        // Codex and ACP ignore them (and say so): they are not handed any.
        assert!(!has_hooks(ProviderKind::Codex, None, scope(false)).await);
        assert!(!has_hooks(ProviderKind::Acp, None, scope(false)).await);
        // A remote machine has no local project to resolve a file against.
        assert!(!has_hooks(ProviderKind::ClaudeCode, Some("~/w"), scope(false)).await);
        // No scope, no hooks.
        assert!(!has_hooks(ProviderKind::Native, None, None).await);
    }

    #[tokio::test]
    async fn the_hook_table_has_the_per_tool_hooks_unless_the_session_is_a_runner() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let table = |tool_knowledge: bool| {
            let mut keys: Vec<String> = manager
                .graph_hook_table(GraphHookInput {
                    session_id: "s".into(),
                    context_source: CompactionContextSource::None,
                    work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
                    tool_knowledge,
                    announce: None,
                })
                .into_keys()
                .collect();
            keys.sort();
            keys
        };
        assert_eq!(table(true), ["PostToolUse", "PreCompact", "PreToolUse"]);
        // A runner has its task context in the prompt: compaction guidance only.
        assert_eq!(table(false), ["PreCompact"]);
    }

    #[tokio::test]
    async fn an_agent_engine_compaction_is_not_announced_twice() {
        // The provider emits its own `compaction` event; the notifier of the agent
        // path must stay silent on the chat channel (it only builds the instructions).
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let table = manager.graph_hook_table(GraphHookInput {
            session_id: "s".into(),
            context_source: CompactionContextSource::None,
            work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
            tool_knowledge: false,
            announce: None,
        });
        let (tx, mut rx) = broadcast::channel(8);
        // The same table, but the claude path's notifier would have announced on `tx`:
        let announcing = manager.graph_hook_table(GraphHookInput {
            session_id: "s".into(),
            context_source: CompactionContextSource::None,
            work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
            tool_knowledge: false,
            announce: Some(tx),
        });
        let input = nexus_claude::HookInput::PreCompact(nexus_claude::PreCompactHookInput {
            session_id: "s".into(),
            transcript_path: String::new(),
            cwd: "/tmp".into(),
            permission_mode: None,
            trigger: "auto".into(),
            custom_instructions: None,
        });
        let ctx = nexus_claude::HookContext { signal: None };
        for matcher in &table["PreCompact"] {
            for hook in &matcher.hooks {
                hook.execute(&input, None, &ctx).await.unwrap();
            }
        }
        assert!(
            rx.try_recv().is_err(),
            "the silent table announced something"
        );
        for matcher in &announcing["PreCompact"] {
            for hook in &matcher.hooks {
                hook.execute(&input, None, &ctx).await.unwrap();
            }
        }
        assert!(matches!(
            rx.try_recv(),
            Ok(ChatEvent::CompactionStarted { .. })
        ));
    }

    // ── Claude Code on another machine (claude_code_remote) ────────────────

    #[tokio::test]
    async fn a_remote_session_spec_runs_in_the_remote_directory_with_no_local_server_or_dirs() {
        use nexus_claude::agent::ProviderKind;
        let state = mock_app_state();
        let mut config = test_config();
        config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let claims = crate::auth::jwt::Claims::service_account("remote");
        let add_dirs = vec!["/local/extra".to_string()];
        let spec_for = |remote_cwd: Option<&'static str>| {
            let manager = &manager;
            let claims = &claims;
            let add_dirs = &add_dirs;
            async move {
                manager
                    .build_agent_spec(AgentSpecInput {
                        cwd: "/Users/me/project",
                        model: "sonnet",
                        system_prompt: "base prompt",
                        permission_mode: None,
                        add_dirs,
                        user_claims: Some(claims),
                        session_id: "remote-s1",
                        third_party: true,
                        max_tokens: None,
                        kind: ProviderKind::ClaudeCode,
                        remote_cwd,
                        hooks: None,
                    })
                    .await
                    .unwrap()
            }
        };
        let remote = spec_for(Some("~/work/app")).await;
        // The remote path, untouched (no local `~` expansion, no local project path).
        assert_eq!(remote.cwd, std::path::PathBuf::from("~/work/app"));
        assert!(remote.mcp_servers.is_empty(), "the PO server is not given");
        assert!(remote.extra_dirs.is_empty());
        let prompt = remote.system_prompt.as_ref().unwrap();
        assert!(prompt.text.starts_with("base prompt"));
        assert!(prompt.text.contains("remote machine"), "the model is told");
        // The same input, local: the server and the directories are there.
        let local = spec_for(None).await;
        assert!(local.mcp_servers.contains_key("project-orchestrator"));
        assert_eq!(local.extra_dirs.len(), 1);
    }

    // ── nexus-tools on a native session (B40) ───────────────────────────────

    /// A native session starts with the `nexus` server (`nexus-tools` over stdio),
    /// bounded by its policy: a denied tool and the web tools (no project, so no
    /// network consent) are not in its `--tools`. Claude Code has its own tools: no `nexus`.
    #[tokio::test]
    async fn a_native_session_gets_the_nexus_tools_server_bounded_by_its_policy() {
        use nexus_claude::agent::{McpServerSpec, ProviderKind};
        let state = mock_app_state();
        let mut config = test_config();
        config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
        let (_bin, program) = fake_nexus_tools();
        config.nexus_tools_path = Some(program.clone());
        config.permission.disallowed_tools = vec!["Monitor".into()];
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let claims = crate::auth::jwt::Claims::service_account("nexus");
        let add_dirs = vec!["/tmp/extra".to_string()];
        let spec_for = |kind: ProviderKind| {
            let manager = &manager;
            let claims = &claims;
            let add_dirs = &add_dirs;
            async move {
                manager
                    .build_agent_spec(AgentSpecInput {
                        cwd: "/tmp",
                        model: "m",
                        system_prompt: "p",
                        permission_mode: Some("default"),
                        add_dirs,
                        user_claims: Some(claims),
                        session_id: "nexus-s1",
                        third_party: true,
                        max_tokens: None,
                        kind,
                        remote_cwd: None,
                        hooks: None,
                    })
                    .await
                    .unwrap()
            }
        };
        let native = spec_for(ProviderKind::Native).await;
        let Some(McpServerSpec::Stdio { command, args, env }) = native.mcp_servers.get("nexus")
        else {
            panic!("a stdio `nexus` server expected");
        };
        assert_eq!(
            std::path::Path::new(command),
            program.canonicalize().unwrap()
        );
        // Nothing from the host's environment: only the session's own key and the profile
        // it signs (#596), never on argv.
        assert_eq!(
            env.keys().map(String::as_str).collect::<Vec<_>>(),
            ["NEXUS_TOOLS_KEY", "NEXUS_TOOLS_PROFILE"]
        );
        assert!(!args.contains(&env["NEXUS_TOOLS_PROFILE"]));
        assert!(!args.iter().any(|a| a == "--trust-harness"));
        let tools_at = args.iter().position(|a| a == "--tools").unwrap() + 1;
        let tools: Vec<&str> = args[tools_at].split(',').collect();
        for tool in ["Read", "Write", "Edit", "Glob", "Grep", "Bash"] {
            assert!(tools.contains(&tool), "{tool} in {tools:?}");
        }
        for tool in ["Monitor", "WebFetch", "WebSearch"] {
            assert!(
                !tools.contains(&tool),
                "{tool} must not be served: {tools:?}"
            );
        }
        assert!(args.windows(2).any(|w| w == ["--add-dir", "/tmp/extra"]));
        assert!(
            !args.iter().any(|a| a.contains("eyJ")),
            "no token on the command line"
        );
        // The PO server stays.
        assert!(native.mcp_servers.contains_key("project-orchestrator"));
        // Claude Code brings its own tools.
        assert!(!spec_for(ProviderKind::ClaudeCode)
            .await
            .mcp_servers
            .contains_key("nexus"));
    }

    /// `nexus-tools` is launched with the session's directory as its working
    /// directory: a relative program path (NEXUS_TOOLS_PATH=./nexus-tools, a `.` or
    /// empty entry of the PATH) would then name a file of the PROJECT, which the
    /// model can write. The server is always launched by its absolute path.
    #[tokio::test]
    async fn the_nexus_tools_server_is_launched_by_its_absolute_path() {
        use nexus_claude::agent::{McpServerSpec, ProviderKind};
        use std::os::unix::fs::PermissionsExt;
        let crate_root = std::env::current_dir().unwrap();
        let dir = tempfile::TempDir::new_in(&crate_root).unwrap();
        let program = dir.path().join("nexus-tools");
        std::fs::write(&program, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let relative = program.strip_prefix(&crate_root).unwrap().to_path_buf();
        assert!(relative.is_relative());
        let state = mock_app_state();
        let mut config = test_config();
        config.nexus_tools_path = Some(relative);
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let spec = manager
            .build_agent_spec(AgentSpecInput {
                cwd: "/tmp",
                model: "m",
                system_prompt: "p",
                permission_mode: Some("default"),
                add_dirs: &[],
                user_claims: None,
                session_id: "nexus-rel",
                third_party: true,
                max_tokens: None,
                kind: ProviderKind::Native,
                remote_cwd: None,
                hooks: None,
            })
            .await
            .unwrap();
        let Some(McpServerSpec::Stdio { command, .. }) = spec.mcp_servers.get("nexus") else {
            panic!("a stdio `nexus` server expected");
        };
        assert!(
            std::path::Path::new(command).is_absolute(),
            "launched as {command}"
        );
        assert_eq!(
            std::path::Path::new(command),
            program.canonicalize().unwrap()
        );
    }

    /// Without `nexus-tools`, a native session opens with the PO tools only.
    #[tokio::test]
    async fn a_native_session_without_nexus_tools_has_no_nexus_server() {
        use nexus_claude::agent::ProviderKind;
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let spec = manager
            .build_agent_spec(AgentSpecInput {
                cwd: "/tmp",
                model: "m",
                system_prompt: "p",
                permission_mode: None,
                add_dirs: &[],
                user_claims: None,
                session_id: "nexus-s2",
                third_party: true,
                max_tokens: None,
                kind: ProviderKind::Native,
                remote_cwd: None,
                hooks: None,
            })
            .await
            .unwrap();
        assert!(!spec.mcp_servers.contains_key("nexus"));
    }
    #[test]
    fn a_session_without_per_session_mcp_reports_the_missing_po_tools() {
        let mut caps = nexus_claude::agent::Capabilities::none();
        caps.per_session_mcp = false;
        assert!(super::super::agent_runtime::degraded_features(&caps)
            .iter()
            .any(|f| f == "project_orchestrator_tools"));
        caps.per_session_mcp = true;
        assert!(!super::super::agent_runtime::degraded_features(&caps)
            .iter()
            .any(|f| f == "project_orchestrator_tools"));
    }

    // ── agent environment and MCP secrets (decision A33) ───────────────────

    /// The PO session token a session of the agent path is given.
    async fn agent_spec_token(
        manager: &ChatManager,
        claims: &crate::auth::jwt::Claims,
        third_party: bool,
        mode: Option<&str>,
        sid: &str,
    ) -> String {
        use nexus_claude::agent::{McpServerSpec, ProviderKind};
        let spec = manager
            .build_agent_spec(AgentSpecInput {
                cwd: "/tmp",
                model: "m",
                system_prompt: "p",
                permission_mode: mode,
                add_dirs: &[],
                user_claims: Some(claims),
                session_id: sid,
                third_party,
                max_tokens: None,
                kind: ProviderKind::Native,
                remote_cwd: None,
                hooks: None,
            })
            .await
            .unwrap();
        let Some(McpServerSpec::Stdio { env, .. }) = spec.mcp_servers.get("project-orchestrator")
        else {
            panic!("stdio MCP server expected");
        };
        env.get("PO_AUTH_TOKEN").expect("a session token").clone()
    }

    /// The claims of a person signed in (a user JWT), not the server's service account.
    fn person_claims() -> crate::auth::jwt::Claims {
        crate::auth::jwt::Claims {
            sub: Uuid::new_v4().to_string(),
            email: "alice@example.com".into(),
            name: "Alice".into(),
            iat: 0,
            exp: 0,
            token_type: None,
            scope: None,
            jti: None,
        }
    }

    fn signed_manager(state: crate::AppState) -> ChatManager {
        let mut config = test_config();
        config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
        ChatManager::new_without_memory(state.neo4j, state.meili, config)
    }

    /// VERIFIER: the token minted for a provider other than Claude Code carries the
    /// restricted tool profile (A35) unless the session is in `trust` (H6); Claude
    /// Code keeps the full one.
    #[tokio::test]
    async fn verifier_a_third_party_session_token_carries_the_restricted_profile() {
        use crate::auth::tool_profile::ToolProfile;
        let manager = signed_manager(mock_app_state());
        let claims = crate::auth::jwt::Claims::service_account("verifier");
        let profile = |token: String| ToolProfile::from_unverified_token(&token);
        for mode in [None, Some("default"), Some("acceptEdits"), Some("plan")] {
            assert_eq!(
                profile(agent_spec_token(&manager, &claims, true, mode, "verifier-s1").await),
                ToolProfile::Restricted,
                "a third party in {mode:?} stays restricted"
            );
        }
        assert_eq!(
            profile(agent_spec_token(&manager, &claims, false, None, "verifier-s2").await),
            ToolProfile::Full
        );
    }

    /// H6: a third-party session a person opened in `trust` (Rock'n roll) gets the
    /// full profile: `plan.delegate_task` passes the MCP profile check and its REST
    /// route is open to the token.
    #[tokio::test]
    async fn a_third_party_session_in_trust_carries_the_full_profile() {
        use crate::auth::tool_profile::ToolProfile;
        use crate::mcp::protocol::ToolCallParams;
        let manager = signed_manager(mock_app_state());
        let claims = person_claims();
        for mode in ["bypassPermissions", "trust"] {
            let token = agent_spec_token(&manager, &claims, true, Some(mode), "trust-s1").await;
            let profile = ToolProfile::from_unverified_token(&token);
            assert_eq!(profile, ToolProfile::Full, "{mode}");
            let delegate = ToolCallParams {
                name: "plan".into(),
                arguments: Some(serde_json::json!({ "action": "delegate_task" })),
            };
            assert_eq!(
                crate::mcp::server::profile_refusal(profile, &delegate),
                None
            );
            assert!(!profile
                .route_forbidden(&axum::http::Method::POST, "/api/plans/p1/tasks/t1/delegate"));
            // The lineage is signed in: whatever this session opens is restricted.
            let decoded =
                jsonwebtoken::dangerous::insecure_decode::<crate::auth::jwt::Claims>(&token)
                    .unwrap()
                    .claims;
            assert!(
                crate::auth::jwt::agent_session_binding(&decoded)
                    .unwrap()
                    .third_party
            );
        }
    }

    /// H6 guard: a session opened BY a third-party session (its caller's token
    /// carries the lineage: delegation, `chat send_message`, `plan run`) is
    /// restricted, whatever its own provider and mode — no delegation loop.
    #[tokio::test]
    async fn a_session_opened_by_a_third_party_session_is_restricted() {
        use crate::auth::tool_profile::ToolProfile;
        let manager = signed_manager(mock_app_state());
        let person = person_claims();
        let parent_token =
            agent_spec_token(&manager, &person, true, Some("trust"), "parent-s1").await;
        let parent =
            crate::auth::jwt::decode_jwt(&parent_token, "test-secret-key-minimum-32-chars!!")
                .unwrap();
        for third_party in [true, false] {
            let token = agent_spec_token(
                &manager,
                &parent,
                third_party,
                Some("bypassPermissions"),
                "child-s1",
            )
            .await;
            let profile = ToolProfile::from_unverified_token(&token);
            assert_eq!(
                profile,
                ToolProfile::Restricted,
                "third_party={third_party}"
            );
            assert!(profile
                .route_forbidden(&axum::http::Method::POST, "/api/plans/p1/tasks/t1/delegate"));
        }
        // A Claude Code parent's children keep what their own provider grants.
        let cc_token = agent_spec_token(&manager, &person, false, None, "cc-parent").await;
        let cc_parent =
            crate::auth::jwt::decode_jwt(&cc_token, "test-secret-key-minimum-32-chars!!").unwrap();
        assert_eq!(
            ToolProfile::from_unverified_token(
                &agent_spec_token(&manager, &cc_parent, true, Some("trust"), "child-s2").await
            ),
            ToolProfile::Full
        );
    }

    /// H6 guard, from the graph: a delegated child resumed later by a person (no
    /// lineage in the person's token) is still restricted when its parent runs on
    /// a third-party provider.
    #[tokio::test]
    async fn a_resumed_child_of_a_third_party_session_is_restricted() {
        use crate::auth::tool_profile::ToolProfile;
        let state = mock_app_state();
        let graph = state.neo4j.clone();
        let manager = signed_manager(state);
        let node = |id: Uuid, provider: &str, spawned_by: Option<String>| {
            serde_json::from_value::<ChatSessionNode>(serde_json::json!({
                "id": id,
                "cwd": "/tmp",
                "model": "m",
                "created_at": "2026-10-09T00:00:00Z",
                "updated_at": "2026-10-09T00:00:00Z",
                "message_count": 0,
                "total_cost_usd": 0.0,
                "provider_id": provider,
                "spawned_by": spawned_by,
            }))
            .unwrap()
        };
        let (third, claude, child_of_third, child_of_claude) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        let delegated = |parent: Uuid| {
            Some(
                crate::chat::types::SpawnedBy::Delegation {
                    plan_id: Uuid::new_v4(),
                    task_id: Uuid::new_v4(),
                    parent_session_id: Some(parent),
                    scaffolding_level: None,
                }
                .to_json_string(),
            )
        };
        for n in [
            node(third, "deepseek", None),
            node(claude, "claude-code", None),
            node(child_of_third, "deepseek", delegated(third)),
            node(child_of_claude, "deepseek", delegated(claude)),
        ] {
            graph.create_chat_session(&n).await.unwrap();
        }
        let person = person_claims();
        let profile_of = |sid: Uuid| {
            let manager = &manager;
            let person = &person;
            async move {
                ToolProfile::from_unverified_token(
                    &agent_spec_token(manager, person, true, Some("trust"), &sid.to_string()).await,
                )
            }
        };
        assert_eq!(profile_of(child_of_third).await, ToolProfile::Restricted);
        assert_eq!(profile_of(child_of_claude).await, ToolProfile::Full);
        assert_eq!(profile_of(third).await, ToolProfile::Full);
    }

    /// H6 guard, origin unknown: a session opened under the server's own service
    /// account (a protocol run, a plan run resumed after a restart) carries no
    /// lineage and has no parent session. A third-party session in `trust` could
    /// otherwise start a protocol whose third-party agent is `full` again and starts
    /// the next one: the loop the guard is there to break. Such a session is
    /// restricted; a delegation whose parent is a Claude Code session keeps `full`.
    #[tokio::test]
    async fn a_third_party_session_opened_by_the_service_account_without_a_known_parent_is_restricted(
    ) {
        use crate::auth::tool_profile::ToolProfile;
        let state = mock_app_state();
        let graph = state.neo4j.clone();
        let manager = signed_manager(state);
        let node = |id: Uuid, provider: &str, spawned_by: Option<String>| {
            serde_json::from_value::<ChatSessionNode>(serde_json::json!({
                "id": id,
                "cwd": "/tmp",
                "model": "m",
                "created_at": "2026-10-09T00:00:00Z",
                "updated_at": "2026-10-09T00:00:00Z",
                "message_count": 0,
                "total_cost_usd": 0.0,
                "provider_id": provider,
                "spawned_by": spawned_by,
            }))
            .unwrap()
        };
        let (protocol_agent, recovered_runner_agent, claude_parent, delegated) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        let runner_spawn = serde_json::json!({
            "type": "runner",
            "run_id": Uuid::new_v4().to_string(),
            "plan_id": Uuid::new_v4().to_string(),
            "task_id": Uuid::new_v4().to_string(),
        })
        .to_string();
        let protocol_spawn = serde_json::json!({
            "type": "protocol_runner",
            "run_id": Uuid::new_v4().to_string(),
            "protocol_id": Uuid::new_v4().to_string(),
            "state_name": "s",
        })
        .to_string();
        let delegation = crate::chat::types::SpawnedBy::Delegation {
            plan_id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            parent_session_id: Some(claude_parent),
            scaffolding_level: None,
        }
        .to_json_string();
        for n in [
            node(protocol_agent, "deepseek", Some(protocol_spawn)),
            node(recovered_runner_agent, "deepseek", Some(runner_spawn)),
            node(claude_parent, "claude-code", None),
            node(delegated, "deepseek", Some(delegation)),
        ] {
            graph.create_chat_session(&n).await.unwrap();
        }
        let profile_of = |claims: crate::auth::jwt::Claims, sid: Uuid| {
            let manager = &manager;
            async move {
                ToolProfile::from_unverified_token(
                    &agent_spec_token(
                        manager,
                        &claims,
                        true,
                        Some("bypassPermissions"),
                        &sid.to_string(),
                    )
                    .await,
                )
            }
        };
        let service = |who: &str| crate::auth::jwt::Claims::service_account(who);
        assert_eq!(
            profile_of(service("protocol-agent:r1"), protocol_agent).await,
            ToolProfile::Restricted,
            "a protocol agent has no known origin"
        );
        assert_eq!(
            profile_of(service("runner-agent:r2"), recovered_runner_agent).await,
            ToolProfile::Restricted,
            "a run resumed after a restart lost its caller"
        );
        assert_eq!(
            profile_of(service("delegate-agent:t1"), delegated).await,
            ToolProfile::Full,
            "a delegation by a Claude Code session keeps what trust grants"
        );
        // A plan run a person started carries that person's claims.
        assert_eq!(
            profile_of(person_claims(), recovered_runner_agent).await,
            ToolProfile::Full
        );
    }

    /// An executable `nexus-tools` stand-in (it is never started here) and its directory.
    fn fake_nexus_tools() -> (tempfile::TempDir, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let bin = tempfile::TempDir::new().unwrap();
        let program = bin.path().join("nexus-tools");
        std::fs::write(&program, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        (bin, program)
    }

    // ── nexus-tools on a native session (B40) ───────────────────────────────

    async fn nexus_tools_spec(
        kind: nexus_claude::agent::ProviderKind,
        remote_cwd: Option<&'static str>,
        configured: bool,
        graph: Arc<crate::neo4j::mock::MockGraphStore>,
        sid: &str,
        permission_mode: Option<&str>,
    ) -> nexus_claude::agent::SessionSpec {
        let state = mock_app_state_with_graph(graph);
        let mut config = test_config();
        config.jwt_secret = Some("test-secret-key-minimum-32-chars!!".into());
        // The stand-in lives until the spec is built: its path is resolved at the opening.
        let (_bin, program) = fake_nexus_tools();
        config.nexus_tools_path = configured.then_some(program);
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let claims = crate::auth::jwt::Claims::service_account("b40");
        manager
            .build_agent_spec(AgentSpecInput {
                cwd: "/work/app",
                model: "m",
                system_prompt: "p",
                permission_mode,
                add_dirs: &[],
                user_claims: Some(&claims),
                session_id: sid,
                third_party: true,
                max_tokens: None,
                kind,
                remote_cwd,
                hooks: Some(AgentHookScope {
                    project_slug: Some("proj".into()),
                    task_id: None,
                    runner: false,
                }),
            })
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn a_native_session_starts_with_nexus_tools_next_to_the_po_server_and_its_profile_in_the_token(
    ) {
        use nexus_claude::agent::{McpServerSpec, ProviderKind};
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let spec = nexus_tools_spec(ProviderKind::Native, None, true, graph, "b40-s1", None).await;
        assert!(spec.mcp_servers.contains_key("project-orchestrator"));
        let Some(McpServerSpec::Stdio { command, args, env }) = spec.mcp_servers.get("nexus")
        else {
            panic!("nexus-tools must be attached to a native session");
        };
        let command = std::path::Path::new(command);
        assert!(command.is_absolute() && command.ends_with("nexus-tools"));
        assert!(env.contains_key("NEXUS_TOOLS_PROFILE") && env.contains_key("NEXUS_TOOLS_KEY"));
        assert!(args.windows(2).any(|w| w == ["--cwd", "/work/app"]));
        assert!(
            args.iter().all(|a| !a.starts_with("v1.")),
            "no token on argv"
        );
        assert!(!format!("{spec:?}").contains(&env["NEXUS_TOOLS_PROFILE"]));
        assert!(spec.hooks.is_some(), "the consent gate is on the session");
        crate::auth::agent_tokens::revoke_session("b40-s1");
    }

    #[tokio::test]
    async fn nexus_tools_is_not_attached_to_other_harnesses_to_a_remote_session_or_without_the_binary(
    ) {
        use nexus_claude::agent::ProviderKind;
        let graph = || Arc::new(crate::neo4j::mock::MockGraphStore::new());
        for (kind, remote, configured, sid) in [
            (ProviderKind::ClaudeCode, None, true, "b40-cc"),
            (ProviderKind::Codex, None, true, "b40-codex"),
            (ProviderKind::Native, Some("~/w"), true, "b40-remote"),
            (ProviderKind::Native, None, false, "b40-nobin"),
        ] {
            let spec = nexus_tools_spec(kind, remote, configured, graph(), sid, None).await;
            assert!(!spec.mcp_servers.contains_key("nexus"), "{sid}");
            assert!(
                !crate::auth::agent_tokens::tools_live(sid),
                "{sid}: nothing minted"
            );
        }
    }

    #[tokio::test]
    async fn only_a_native_session_opened_without_its_nexus_server_is_reported_as_lacking_it() {
        use nexus_claude::agent::ProviderKind;
        let graph = || Arc::new(crate::neo4j::mock::MockGraphStore::new());
        // A runnable stand-in alive for the whole test (the spec's own is gone once the
        // spec is built).
        let (_bin, program) = fake_nexus_tools();
        // (kind, remote, binary configured, session, lacks the file/shell/web tools)
        for (kind, remote, configured, sid, lacks) in [
            (ProviderKind::Native, None, true, "b40-has", false),
            (ProviderKind::Native, None, false, "b40-lacks", true),
            // Claude Code and Codex bring their own tools: never reported.
            (ProviderKind::ClaudeCode, None, false, "b40-cc-own", false),
            (ProviderKind::Codex, None, false, "b40-codex-own", false),
        ] {
            let spec = nexus_tools_spec(kind, remote, configured, graph(), sid, None).await;
            let path = configured.then_some(program.as_path());
            assert_eq!(lacks_nexus_tools(kind, &spec, path), lacks, "{sid}");
            crate::auth::agent_tokens::revoke_session(sid);
        }
        // The executable is there but nothing was attached (a policy that leaves no
        // `nexus-tools` tool to offer): the installation lacks nothing, not reported.
        let bare =
            nexus_tools_spec(ProviderKind::Native, None, false, graph(), "b40-bare", None).await;
        crate::auth::agent_tokens::revoke_session("b40-bare");
        assert!(!bare
            .mcp_servers
            .contains_key(nexus_claude::providers::native::NEXUS_TOOLS_SERVER));
        assert!(!lacks_nexus_tools(
            ProviderKind::Native,
            &bare,
            Some(program.as_path())
        ));
        // A configured path that is not (or no longer) there is missing.
        let gone = program.with_file_name("gone");
        assert!(lacks_nexus_tools(
            ProviderKind::Native,
            &bare,
            Some(gone.as_path())
        ));
    }

    #[tokio::test]
    async fn the_session_hooks_refuse_webfetch_to_an_origin_the_project_did_not_consent_to() {
        use nexus_claude::agent::{HookVerdict, ProviderKind, ToolCallInfo};
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let spec = nexus_tools_spec(
            ProviderKind::Native,
            None,
            true,
            graph.clone(),
            "b40-gate",
            None,
        )
        .await;
        let hooks = spec.hooks.clone().expect("hooks");
        let fetch = ToolCallInfo {
            id: None,
            name: "mcp__nexus__WebFetch".into(),
            canonical: Some("WebFetch".into()),
            category: nexus_claude::agent::ToolCategory::Web,
            input: serde_json::json!({"url": "https://docs.rs/serde"}),
        };
        let HookVerdict::Deny { reason } = hooks.before_tool(&fetch).await else {
            panic!("an origin without consent must be refused");
        };
        assert!(reason.starts_with("tool_origin_not_allowed:"), "{reason}");
        // The project consents; the same call is no longer this gate's refusal.
        graph
            .put_llm_setting(
                "project:proj",
                "tool_origin:https://docs.rs",
                &serde_json::json!({
                    "origin": "https://docs.rs",
                    "consented_by": "me@example.com",
                    "consented_at": "2026-10-08T10:00:00Z"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(!matches!(
            hooks.before_tool(&fetch).await,
            HookVerdict::Deny { .. }
        ));
        // The session is closed: its token is revoked and its nexus tools stop.
        crate::auth::agent_tokens::revoke_session("b40-gate");
        let read = ToolCallInfo {
            name: "mcp__nexus__Read".into(),
            canonical: Some("Read".into()),
            input: serde_json::json!({"file_path": "/work/app/a"}),
            ..fetch
        };
        assert!(matches!(
            hooks.before_tool(&read).await,
            HookVerdict::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn mcp_config_carries_no_secret() {
        let state = mock_app_state();
        let config = test_config();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let options = manager
            .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
            .await;

        let Some(McpServerConfig::Stdio { env, .. }) =
            options.mcp_servers.get("project-orchestrator")
        else {
            panic!("Expected Stdio MCP config");
        };
        let env = env.as_ref().unwrap();
        for name in [
            "NEO4J_PASSWORD",
            "NEO4J_USER",
            "NEO4J_URI",
            "MEILISEARCH_KEY",
            "MEILISEARCH_URL",
        ] {
            assert!(
                !env.contains_key(name),
                "the MCP proxy does not need {name}; it must not be handed to the agent"
            );
        }
        let serialized = serde_json::to_string(&env).unwrap();
        assert!(
            !serialized.contains("\"test\"") && !serialized.contains("\"key\""),
            "database/search credentials leaked into the MCP config: {serialized}"
        );
        assert!(
            options.mcp_config_via_file,
            "the MCP config (session token) must go through a file, not argv"
        );
    }

    #[tokio::test]
    async fn child_env_only_carries_allowlisted_names() {
        let state = mock_app_state();
        let config = test_config();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        let options = manager
            .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
            .await;

        let policy = &options.env_policy;
        assert!(
            policy.is_isolated(),
            "the agent must start from a clean environment"
        );
        for secret in [
            "NEO4J_PASSWORD",
            "MEILISEARCH_KEY",
            "EMBEDDING_API_KEY",
            "PO_JWT_SECRET",
            "GOOGLE_CLIENT_SECRET",
            "AWS_SECRET_ACCESS_KEY",
            "SOME_UNKNOWN_SERVER_VARIABLE",
        ] {
            assert!(!policy.allows(secret), "{secret} must not reach the agent");
        }
        for needed in [
            "PATH",
            "HOME",
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "SSH_AUTH_SOCK",
        ] {
            assert!(policy.allows(needed), "{needed} must reach the agent");
        }
    }

    #[test]
    fn the_operator_list_extends_the_allowlist_but_never_with_server_secrets() {
        let policy = child_env_policy_with(" GH_TOKEN , PO_JWT_SECRET,,NEO4J_PASSWORD ");
        assert!(policy.allows("GH_TOKEN"));
        assert!(!policy.allows("PO_JWT_SECRET"));
        assert!(!policy.allows("NEO4J_PASSWORD"));
        assert!(!child_env_policy_with("").allows("GH_TOKEN"));
    }

    #[tokio::test]
    async fn test_build_options_with_hooks() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // Build hooks map with a CompactionNotifier
        let (tx, _rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx, None, "test-session".to_string());
        let mut hooks = std::collections::HashMap::new();
        hooks.insert(
            "PreCompact".to_string(),
            vec![nexus_claude::HookMatcher {
                matcher: None,
                hooks: vec![std::sync::Arc::new(notifier)],
            }],
        );

        let opts = manager
            .build_options(
                "/tmp",
                "model",
                "prompt",
                None,
                None,
                Some(hooks),
                &[],
                None,
                None,
            )
            .await;

        // Hooks should be configured
        let hook_map = opts.hooks.as_ref().expect("hooks should be Some");
        assert!(hook_map.contains_key("PreCompact"));
        let matchers = &hook_map["PreCompact"];
        assert_eq!(matchers.len(), 1);
        assert_eq!(matchers[0].hooks.len(), 1);
    }

    #[tokio::test]
    async fn test_build_options_without_hooks() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let opts = manager
            .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
            .await;

        // No hooks configured
        assert!(opts.hooks.is_none());
    }

    // ====================================================================
    // resolve_add_dirs & build_options with add_dirs
    // ====================================================================

    #[tokio::test]
    async fn test_resolve_add_dirs_explicit() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let dirs = vec!["/path/a".to_string(), "/path/b".to_string()];
        let result = manager
            .resolve_add_dirs("/tmp", Some(&dirs), None, None)
            .await;

        assert_eq!(result, vec!["/path/a", "/path/b"]);
    }

    #[tokio::test]
    async fn test_resolve_add_dirs_empty() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.resolve_add_dirs("/tmp", None, None, None).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_resolve_add_dirs_workspace() {
        let state = mock_app_state();

        // Create a workspace and projects
        let ws = crate::test_helpers::test_workspace();
        state.neo4j.create_workspace(&ws).await.unwrap();

        let mut p1 = crate::test_helpers::test_project();
        p1.root_path = "/home/user/proj-a".to_string();
        state.neo4j.create_project(&p1).await.unwrap();
        state
            .neo4j
            .add_project_to_workspace(ws.id, p1.id)
            .await
            .unwrap();

        let mut p2 = crate::test_helpers::test_project();
        p2.root_path = "/home/user/proj-b".to_string();
        state.neo4j.create_project(&p2).await.unwrap();
        state
            .neo4j
            .add_project_to_workspace(ws.id, p2.id)
            .await
            .unwrap();

        let mut p3 = crate::test_helpers::test_project();
        p3.root_path = "/home/user/proj-cwd".to_string();
        state.neo4j.create_project(&p3).await.unwrap();
        state
            .neo4j
            .add_project_to_workspace(ws.id, p3.id)
            .await
            .unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // cwd matches p3, so only p1 and p2 should be returned
        let result = manager
            .resolve_add_dirs("/home/user/proj-cwd", None, Some(&ws.slug), None)
            .await;

        assert_eq!(result.len(), 2);
        assert!(result.contains(&"/home/user/proj-a".to_string()));
        assert!(result.contains(&"/home/user/proj-b".to_string()));
        assert!(!result.contains(&"/home/user/proj-cwd".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_add_dirs_workspace_not_found() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager
            .resolve_add_dirs("/tmp", None, Some("nonexistent-ws"), None)
            .await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_build_options_with_add_dirs() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let dirs = ["/extra/dir1".to_string(), "/extra/dir2".to_string()];
        let opts = manager
            .build_options(
                "/tmp", "model", "prompt", None, None, None, &dirs, None, None,
            )
            .await;

        assert_eq!(opts.add_dirs.len(), 2);
        assert!(opts
            .add_dirs
            .contains(&std::path::PathBuf::from("/extra/dir1")));
        assert!(opts
            .add_dirs
            .contains(&std::path::PathBuf::from("/extra/dir2")));
    }

    #[tokio::test]
    async fn test_build_options_without_add_dirs() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let opts = manager
            .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
            .await;

        assert!(opts.add_dirs.is_empty());
    }

    #[tokio::test]
    async fn test_build_options_injects_session_id_env() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // With session_id → PO_SESSION_ID should be in MCP server env
        let opts = manager
            .build_options(
                "/tmp",
                "model",
                "prompt",
                None,
                None,
                None,
                &[],
                None,
                Some("abc-def-123"),
            )
            .await;

        // PO_SESSION_ID is injected into the MCP server config env, not ClaudeCodeOptions.env
        let mcp = opts.mcp_servers.get("project-orchestrator").unwrap();
        let mcp_env = match mcp {
            nexus_claude::McpServerConfig::Stdio { env, .. } => env.as_ref().unwrap(),
            _ => panic!("Expected Stdio config"),
        };
        assert_eq!(
            mcp_env.get("PO_SESSION_ID").map(|s| s.as_str()),
            Some("abc-def-123"),
            "PO_SESSION_ID should be injected into MCP server env when session_id is provided"
        );
    }

    #[tokio::test]
    async fn test_build_options_no_session_id_no_env() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // Without session_id → PO_SESSION_ID should NOT be in MCP server env
        let opts = manager
            .build_options("/tmp", "model", "prompt", None, None, None, &[], None, None)
            .await;

        let mcp = opts.mcp_servers.get("project-orchestrator").unwrap();
        let mcp_env = match mcp {
            nexus_claude::McpServerConfig::Stdio { env, .. } => env.as_ref().unwrap(),
            _ => panic!("Expected Stdio config"),
        };
        assert!(
            !mcp_env.contains_key("PO_SESSION_ID"),
            "PO_SESSION_ID should not be in MCP server env when session_id is None"
        );
    }

    // ====================================================================
    // message_to_events
    // ====================================================================

    #[test]
    fn test_message_to_events_assistant_text() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: "Hello!".into(),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], ChatEvent::AssistantText { content, .. } if content == "Hello!")
        );
    }

    #[test]
    fn test_message_to_events_thinking() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::Thinking(ThinkingContent {
                    thinking: "Let me think...".into(),
                    signature: "sig".into(),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], ChatEvent::Thinking { content, .. } if content == "Let me think...")
        );
    }

    #[test]
    fn test_message_to_events_tool_use() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolUse(ToolUseContent {
                    id: "tool-1".into(),
                    name: "create_plan".into(),
                    input: serde_json::json!({"title": "My Plan"}),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::ToolUse { id, tool, .. }
            if id == "tool-1" && tool == "create_plan"));
    }

    #[test]
    fn test_message_to_events_tool_result() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-1".into(),
                    content: Some(ContentValue::Text("Success".into())),
                    is_error: Some(false),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], ChatEvent::ToolResult { id, is_error, .. }
            if id == "tool-1" && !is_error)
        );
    }

    #[test]
    fn test_message_to_events_tool_result_error() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-2".into(),
                    content: Some(ContentValue::Text("Not found".into())),
                    is_error: Some(true),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::ToolResult { is_error, .. } if *is_error));
    }

    #[test]
    fn test_message_to_events_result() {
        let msg = Message::Result {
            subtype: "success".into(),
            duration_ms: 5000,
            duration_api_ms: 4500,
            is_error: false,
            num_turns: 3,
            session_id: "cli-abc-123".into(),
            total_cost_usd: Some(0.15),
            usage: None,
            result: None,
            structured_output: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::Result {
            session_id, duration_ms, cost_usd, subtype, is_error, num_turns, result_text, ..
        } if session_id == "cli-abc-123"
            && *duration_ms == 5000
            && *cost_usd == Some(0.15)
            && subtype == "success"
            && !is_error
            && *num_turns == Some(3)
            && result_text.is_none()
        ));
    }

    #[test]
    fn test_message_to_events_result_error_max_turns() {
        let msg = Message::Result {
            subtype: "error_max_turns".into(),
            duration_ms: 8000,
            duration_api_ms: 7500,
            is_error: true,
            num_turns: 15,
            session_id: "cli-max-turns".into(),
            total_cost_usd: Some(0.50),
            usage: None,
            result: None,
            structured_output: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::Result {
            subtype, is_error, num_turns, ..
        } if subtype == "error_max_turns" && *is_error && *num_turns == Some(15)));
    }

    #[test]
    fn test_message_to_events_result_error_during_execution() {
        let msg = Message::Result {
            subtype: "error_during_execution".into(),
            duration_ms: 2000,
            duration_api_ms: 1800,
            is_error: true,
            num_turns: 1,
            session_id: "cli-exec-err".into(),
            total_cost_usd: Some(0.02),
            usage: None,
            result: Some("Process exited with code 1".into()),
            structured_output: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::Result {
            subtype, is_error, result_text, ..
        } if subtype == "error_during_execution"
            && *is_error
            && result_text.as_deref() == Some("Process exited with code 1")
        ));
    }

    #[test]
    fn test_message_to_events_multiple_blocks() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![
                    ContentBlock::Thinking(ThinkingContent {
                        thinking: "hmm...".into(),
                        signature: "s".into(),
                    }),
                    ContentBlock::Text(TextContent {
                        text: "Here is my answer".into(),
                    }),
                    ContentBlock::ToolUse(ToolUseContent {
                        id: "t1".into(),
                        name: "list_plans".into(),
                        input: serde_json::json!({}),
                    }),
                ],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], ChatEvent::Thinking { .. }));
        assert!(matches!(&events[1], ChatEvent::AssistantText { .. }));
        assert!(matches!(&events[2], ChatEvent::ToolUse { .. }));
    }

    #[test]
    fn test_message_to_events_system_init() {
        let msg = Message::System {
            subtype: "init".into(),
            data: serde_json::json!({
                "session_id": "cli-sess-abc",
                "model": "claude-sonnet-4-6",
                "tools": ["Bash", "Read", "Write", "Edit"],
                "mcp_servers": [{"name": "po", "status": "connected"}],
                "permissionMode": "default"
            }),
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::SystemInit {
            cli_session_id, model, tools, mcp_servers, permission_mode, ..
        } if cli_session_id == "cli-sess-abc"
            && model.as_deref() == Some("claude-sonnet-4-6")
            && tools.len() == 4
            && mcp_servers.len() == 1
            && permission_mode.as_deref() == Some("default")
        ));
    }

    #[test]
    fn test_message_to_events_system_unknown() {
        // Unknown system subtypes should still be ignored
        let msg = Message::System {
            subtype: "unknown_future_type".into(),
            data: serde_json::json!({"version": "1.0"}),
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    // ── masking fails closed (decision A36) ────────────────────────────────

    #[test]
    fn a_message_that_cannot_be_masked_is_withheld_not_passed_in_clear() {
        // The secret value collides with a key of the message's own JSON form:
        // once masked, the message no longer reads back.
        let secret = "duration_api_ms";
        let masker = crate::vault::mask::Masker::from_values([("TOKEN", secret)]);
        let msg = Message::Result {
            subtype: "success".into(),
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: false,
            num_turns: 1,
            session_id: "cli-1".into(),
            total_cost_usd: None,
            usage: None,
            result: Some(format!("the value is {secret}")),
            structured_output: None,
        };

        let out = ChatManager::mask_cli_message_with(&masker, msg);
        assert!(
            matches!(&out, Message::System { subtype, .. } if subtype == MASKING_FAILED_SUBTYPE),
            "an unmaskable message must be replaced, got {out:?}"
        );
        let events = ChatManager::message_to_events(&out);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], ChatEvent::Error { message, .. } if message == MASKING_FAILED_MESSAGE)
        );
        let wire = serde_json::to_string(&events).unwrap();
        assert!(
            !wire.contains(secret),
            "the secret reached the wire: {wire}"
        );
    }

    #[test]
    fn a_maskable_message_is_masked_and_an_unrelated_one_is_untouched() {
        let masker = crate::vault::mask::Masker::from_values([("TOKEN", "s3cr3t-value")]);
        let with_secret = Message::System {
            subtype: "note".into(),
            data: serde_json::json!({"text": "key=s3cr3t-value"}),
        };
        let out = ChatManager::mask_cli_message_with(&masker, with_secret);
        let json = serde_json::to_string(&out).unwrap();
        assert!(!json.contains("s3cr3t-value") && json.contains("[secret:TOKEN]"));

        let plain = Message::System {
            subtype: "note".into(),
            data: serde_json::json!({"text": "nothing here"}),
        };
        let out = ChatManager::mask_cli_message_with(&masker, plain);
        assert!(matches!(&out, Message::System { subtype, .. } if subtype == "note"));
    }

    #[test]
    fn test_message_to_events_user_message() {
        let msg = Message::User {
            message: nexus_claude::UserMessage {
                content: "Hi".into(),
                content_blocks: None,
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_user_message_with_tool_result() {
        let msg = Message::User {
            message: nexus_claude::UserMessage {
                content: String::new(),
                content_blocks: Some(vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "toolu_abc123".into(),
                    content: Some(ContentValue::Text("fn main() {}".into())),
                    is_error: Some(false),
                })]),
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ChatEvent::ToolResult { id, is_error, .. }
            if id == "toolu_abc123" && !is_error
        ));
    }

    #[test]
    fn test_message_to_events_user_message_with_tool_result_error() {
        let msg = Message::User {
            message: nexus_claude::UserMessage {
                content: String::new(),
                content_blocks: Some(vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "toolu_err001".into(),
                    content: Some(ContentValue::Text("File not found".into())),
                    is_error: Some(true),
                })]),
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ChatEvent::ToolResult { id, is_error, .. }
            if id == "toolu_err001" && *is_error
        ));
    }

    // ====================================================================
    // message_to_events — StreamEvent
    // ====================================================================

    #[test]
    fn test_message_to_events_stream_text_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockDelta {
                index: 0,
                delta: StreamDelta::TextDelta {
                    text: "Hello".into(),
                },
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::StreamDelta { text, .. } if text == "Hello"));
    }

    #[test]
    fn test_message_to_events_stream_content_block_start_tool_use() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockStart {
                index: 1,
                content_block: serde_json::json!({
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "list_plans",
                    "input": {"status": "active"}
                }),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ChatEvent::ToolUse { id, tool, .. }
                if id == "toolu_abc123" && tool == "list_plans"));
    }

    #[test]
    fn test_message_to_events_stream_content_block_start_text_ignored() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockStart {
                index: 0,
                content_block: serde_json::json!({"type": "text", "text": ""}),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_thinking_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockDelta {
                index: 0,
                delta: StreamDelta::ThinkingDelta {
                    thinking: "hmm".into(),
                },
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_message_stop() {
        let msg = Message::StreamEvent {
            event: StreamEventData::MessageStop,
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_content_block_start() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockStart {
                index: 0,
                content_block: serde_json::json!({"type": "text", "text": ""}),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_input_json_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::ContentBlockDelta {
                index: 0,
                delta: StreamDelta::InputJsonDelta {
                    partial_json: r#"{"title":"#.into(),
                },
            },
            session_id: Some("sess-1".into()),
            parent_tool_use_id: None,
        };

        // InputJsonDelta is not TextDelta, so it should produce empty events
        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_message_start() {
        let msg = Message::StreamEvent {
            event: StreamEventData::MessageStart {
                message: serde_json::json!({"id": "msg_123"}),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_stream_message_delta() {
        let msg = Message::StreamEvent {
            event: StreamEventData::MessageDelta {
                delta: serde_json::json!({"stop_reason": "end_turn"}),
                usage: Some(serde_json::json!({"output_tokens": 50})),
            },
            session_id: None,
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert!(events.is_empty());
    }

    #[test]
    fn test_message_to_events_tool_result_structured() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-3".into(),
                    content: Some(ContentValue::Structured(vec![
                        serde_json::json!({"type": "text", "text": "result 1"}),
                        serde_json::json!({"type": "text", "text": "result 2"}),
                    ])),
                    is_error: None,
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::ToolResult {
                id,
                result,
                is_error,
                ..
            } => {
                assert_eq!(id, "tool-3");
                assert!(result.is_array());
                assert_eq!(result.as_array().unwrap().len(), 2);
                assert!(!is_error); // None defaults to false
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_message_to_events_tool_result_none_content() {
        let msg = Message::Assistant {
            message: AssistantMessage {
                content: vec![ContentBlock::ToolResult(ToolResultContent {
                    tool_use_id: "tool-4".into(),
                    content: None,
                    is_error: Some(false),
                })],
            },
            parent_tool_use_id: None,
        };

        let events = ChatManager::message_to_events(&msg);
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::ToolResult { result, .. } => {
                assert!(result.is_null());
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    // ====================================================================
    // build_system_prompt with active plans
    // ====================================================================

    #[tokio::test]
    async fn test_build_system_prompt_with_active_plans() {
        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        // Create an active plan linked to the project
        let plan = crate::test_helpers::test_plan();
        state.neo4j.create_plan(&plan).await.unwrap();
        state
            .neo4j
            .link_plan_to_project(plan.id, project.id)
            .await
            .unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (prompt, _note_ids) = manager
            .build_system_prompt(Some(&project.slug), "check the plan", None, None, None)
            .await;

        // Base prompt present
        assert!(prompt.contains("EXCLUSIVELY the Project Orchestrator MCP tools"));
        // Dynamic context section present (either oneshot or fallback)
        assert!(prompt.contains("---"));
        // Project name should appear in the dynamic context
        assert!(prompt.contains(&project.name));
    }

    // ====================================================================
    // active_session_count
    // ====================================================================

    #[tokio::test]
    async fn test_active_session_count_empty() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert_eq!(manager.active_session_count().await, 0);
    }

    // ====================================================================
    // subscribe / interrupt / close errors for missing sessions
    // ====================================================================

    #[tokio::test]
    async fn test_subscribe_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.subscribe("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_interrupt_nonexistent_session() {
        // interrupt() no longer errors for non-local sessions — it always succeeds
        // because it may need to publish to NATS for cross-instance interrupt routing.
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.interrupt("nonexistent").await;
        assert!(
            result.is_ok(),
            "interrupt should succeed even for non-local sessions"
        );
    }

    #[tokio::test]
    async fn test_close_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.close_session("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn closing_a_session_broadcasts_session_closed() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        test_support::insert_live_session_without_cli(&manager, "sess-close").await;
        let mut rx = manager
            .active_sessions
            .read()
            .await
            .get("sess-close")
            .expect("registered")
            .events_tx
            .subscribe();
        manager.close_session("sess-close").await.unwrap();
        let mut seen = None;
        while let Ok(ev) = rx.try_recv() {
            if let ChatEvent::SessionClosed { session_id, reason } = ev {
                seen = Some((session_id, reason));
            }
        }
        assert_eq!(
            seen,
            Some(("sess-close".to_string(), Some("closed".to_string())))
        );
    }

    #[tokio::test]
    async fn test_send_message_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.send_message("nonexistent", "hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_resume_session_invalid_uuid() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.resume_session("not-a-uuid", "hello", None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid session ID"));
    }

    #[tokio::test]
    async fn test_resume_session_not_in_db() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let id = Uuid::new_v4().to_string();
        let result = manager.resume_session(&id, "hello", None).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not found in database"));
    }

    #[tokio::test]
    async fn test_resume_session_no_cli_session_id_starts_fresh() {
        // When a session has no cli_session_id (first message or previous spawn failed),
        // resume_session should attempt to start a fresh CLI (not error immediately).
        // In CI without Claude CLI, this will fail at InteractiveClient creation,
        // but the error should NOT be "no CLI session ID".
        let state = mock_app_state();
        let session = test_chat_session(None); // no cli_session_id
        state.neo4j.create_chat_session(&session).await.unwrap();

        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager
            .resume_session(&session.id.to_string(), "hello", None)
            .await;
        // Should fail (CLI not available in test), but NOT with "no CLI session ID"
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            !err_msg.contains("no CLI session ID"),
            "Should not fail with 'no CLI session ID', got: {}",
            err_msg
        );
    }

    // ====================================================================
    // Session -> run/thread attachment (task 1.7)
    // ====================================================================

    fn runner_request(run_id: Uuid, plan_id: Uuid, task_id: Uuid) -> ChatRequest {
        ChatRequest {
            access: None,
            routing_pool: None,
            routing_mode: None,
            attachments: Vec::new(),
            refs: Vec::new(),
            message: "go".into(),
            session_id: None,
            cwd: "/tmp/test".into(),
            project_slug: None,
            model: None,
            provider: None,
            task_alias: None,
            persona_alias: None,
            run_provider: None,
            run_model: None,
            max_tokens: None,
            task_class: None,
            permission_mode: Some("bypassPermissions".into()),
            add_dirs: None,
            workspace_slug: None,
            user_claims: None,
            spawned_by: Some(
                serde_json::json!({
                    "type": "runner",
                    "run_id": run_id.to_string(),
                    "plan_id": plan_id.to_string(),
                    "task_id": task_id.to_string(),
                })
                .to_string(),
            ),
            task_context: None,
            scaffolding_override: None,
            runner_context: None,
            routing_decision_id: None,
        }
    }

    fn manager_with_mock() -> (ChatManager, Arc<crate::neo4j::mock::MockGraphStore>) {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let state = mock_app_state();
        (
            ChatManager::new_without_memory(dyn_graph, state.meili, test_config()),
            graph,
        )
    }

    #[tokio::test]
    async fn test_create_session_links_runner_session_to_its_run() {
        let (manager, graph) = manager_with_mock();
        let (run, plan, task) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        // The CLI is not available in tests: only the persisted side matters.
        let _ = manager
            .create_session(&runner_request(run, plan, task))
            .await;

        let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
        assert_eq!(sessions.len(), 1, "the runner session was persisted");
        let rows = graph.session_link_rows.read().await.clone();
        assert_eq!(rows.len(), 1, "exactly one run link: {rows:?}");
        assert_eq!(rows[0].session_id, sessions[0].id);
        assert_eq!(rows[0].run_id, Some(run));
        assert_eq!(rows[0].plan_id, Some(plan));
        assert_eq!(rows[0].task_id, Some(task));
        // and the JSON mechanism is there too: both mechanisms on the session
        let a = crate::chat::attachment::attach(&sessions, &rows);
        assert_eq!(a.by_plan[&plan][0].links.len(), 2);
    }

    #[tokio::test]
    async fn test_create_session_of_a_resumed_run_links_to_the_new_run() {
        let (manager, graph) = manager_with_mock();
        let (old_run, new_run, plan) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let _ = manager
            .create_session(&runner_request(old_run, plan, Uuid::new_v4()))
            .await;
        // free the single "active session" slot if the CLI did start
        manager.active_sessions.write().await.clear();
        let _ = manager
            .create_session(&runner_request(new_run, plan, Uuid::new_v4()))
            .await;
        let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
        let rows = graph.session_link_rows.read().await.clone();
        let a = crate::chat::attachment::attach(&sessions, &rows);
        assert_eq!(a.by_plan.len(), 1, "same plan, same thread");
        let relation_runs: std::collections::HashSet<_> =
            rows.iter().filter_map(|r| r.run_id).collect();
        assert_eq!(
            relation_runs,
            [old_run, new_run].into_iter().collect(),
            "each session is linked by relation to ITS run"
        );
        let runs: std::collections::HashSet<_> = a.by_plan[&plan]
            .iter()
            .flat_map(|s| s.links.iter().filter_map(|l| l.run_id))
            .collect();
        assert_eq!(runs, [old_run, new_run].into_iter().collect());
    }

    #[tokio::test]
    async fn test_create_session_free_chat_gets_no_run_link() {
        let (manager, graph) = manager_with_mock();
        let mut req = runner_request(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        req.spawned_by = None;
        let _ = manager.create_session(&req).await;
        assert!(graph.session_link_rows.read().await.is_empty());
        let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
        let a = crate::chat::attachment::attach(&sessions, &[]);
        assert_eq!(a.unattached.len(), sessions.len());
    }

    #[tokio::test]
    async fn create_session_persists_the_provider_and_how_it_was_routed() {
        let (manager, graph) = manager_with_mock();
        // The CLI is not available in tests: only the persisted side matters.
        let _ = manager
            .create_session(&runner_request(
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
            ))
            .await;
        let sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].provider_id.as_deref(), Some("claude-code"));
        assert_eq!(sessions[0].routed_by.as_deref(), Some("claude_code"));
    }

    #[tokio::test]
    async fn create_session_naming_an_unknown_provider_is_refused_before_persisting() {
        let (manager, graph) = manager_with_mock();
        let mut req = runner_request(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        req.provider = Some("deepseek".into());
        let err = manager.create_session(&req).await.unwrap_err();
        let failure =
            crate::chat::provider::errors::classify_open_error(&err, None).expect("typed failure");
        assert_eq!((failure.status, failure.code), (404, "provider_unknown"));
        assert!(
            graph.chat_sessions.read().await.is_empty(),
            "nothing persisted"
        );
    }

    #[tokio::test]
    async fn a_request_for_another_provider_on_an_existing_session_is_a_409() {
        let (manager, graph) = manager_with_mock();
        let mut s = test_chat_session(None);
        s.provider_id = Some("claude-code".into());
        graph.create_chat_session(&s).await.unwrap();
        let id = s.id.to_string();
        manager.check_provider_binding(&id, None).await.unwrap();
        manager
            .check_provider_binding(&id, Some("claude-code"))
            .await
            .unwrap();
        let err = manager
            .check_provider_binding(&id, Some("deepseek"))
            .await
            .unwrap_err();
        let failure = crate::chat::provider::errors::classify_open_error(&err, None).unwrap();
        assert_eq!((failure.status, failure.code), (409, "provider_conflict"));
    }

    /// Task 1ff0e2e4: a `--resume` target the CLI no longer has makes it exit at
    /// once. The message must still be taken: a fresh CLI for the SAME PO
    /// session, never a dead one handed back (silent 2 s turns).
    #[cfg(unix)]
    #[tokio::test]
    async fn resume_falls_back_to_a_fresh_cli_when_the_resumed_one_dies_at_once() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("args.log");
        let cli = dir.path().join("fake-claude");
        std::fs::write(
            &cli,
            format!(
                "#!/bin/sh\necho \"ARGS $*\" >> {log}\ncase \"$1\" in\n  --version) echo '2.1.287 (Claude Code)'; exit 0;;\nesac\nfor a in \"$@\"; do [ \"$a\" = \"--resume\" ] && exit 1; done\nexec cat > /dev/null\n",
                log = log.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();

        let (manager, graph) = manager_with_mock();
        manager
            .update_claude_cli_path(Some(cli.display().to_string()))
            .await;
        let mut s = test_chat_session(None);
        s.provider_id = Some("claude-code".into());
        s.cwd = dir.path().display().to_string();
        s.cli_session_id = Some("512a5305-16d1-4ad8-a02f-d2b6e845dbdf".into());
        graph.create_chat_session(&s).await.unwrap();
        let id = s.id.to_string();

        manager
            .resume_session(&id, "hello", None)
            .await
            .expect("the message is taken by a fresh CLI");

        assert!(manager.is_session_active(&id).await, "the session is live");
        let mut calls = String::new();
        for _ in 0..80 {
            calls = std::fs::read_to_string(&log).unwrap_or_default();
            if calls.matches("ARGS --output-format").count() >= 2 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let spawns: Vec<&str> = calls
            .split("ARGS ")
            .filter(|c| c.starts_with("--output-format"))
            .collect();
        let resumes = |c: &str| c.contains(" --resume ");
        assert!(
            spawns.len() >= 2,
            "resume then fresh: {:?}",
            calls
                .split("ARGS ")
                .map(|c| (
                    c.len(),
                    c.contains(" --resume "),
                    c.chars().take(60).collect::<String>()
                ))
                .collect::<Vec<_>>()
        );
        assert!(resumes(spawns[0]), "first try resumes");
        assert!(!resumes(spawns[1]), "the fallback does not resume");
        manager.evict_if_cli_dead(&id).await;
    }

    /// Task 1ff0e2e4: a registered session whose CLI has left is dropped, so the
    /// next message resumes instead of being written to a closed stdin.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_session_whose_cli_has_left_is_evicted_before_delivery() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let cli = dir.path().join("fake-claude");
        std::fs::write(
            &cli,
            "#!/bin/sh\ncase \"$1\" in\n  --version) echo '2.1.287 (Claude Code)'; exit 0;;\nesac\nexec cat > /dev/null\n",
        )
        .unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (manager, graph) = manager_with_mock();
        manager
            .update_claude_cli_path(Some(cli.display().to_string()))
            .await;
        let mut s = test_chat_session(None);
        s.provider_id = Some("claude-code".into());
        s.cwd = dir.path().display().to_string();
        graph.create_chat_session(&s).await.unwrap();
        let id = s.id.to_string();
        manager.resume_session(&id, "hello", None).await.unwrap();
        assert!(manager.is_session_active(&id).await);
        assert!(!manager.evict_if_cli_dead(&id).await, "alive: kept");

        // The CLI leaves: kill its process.
        let client = manager
            .active_sessions
            .read()
            .await
            .get(&id)
            .unwrap()
            .client
            .clone();
        let pid = client.lock().await.child_pid().await.expect("pid");
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        for _ in 0..50 {
            if manager.evict_if_cli_dead(&id).await {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(!manager.is_session_active(&id).await, "evicted");
    }

    #[tokio::test]
    async fn resuming_a_session_bound_to_a_non_claude_provider_is_unavailable_on_the_legacy_path() {
        let (manager, graph) = manager_with_mock();
        let mut s = test_chat_session(None);
        s.provider_id = Some("deepseek".into());
        graph.create_chat_session(&s).await.unwrap();
        let err = manager
            .resume_session(&s.id.to_string(), "hi", None)
            .await
            .unwrap_err();
        let failure = crate::chat::provider::errors::classify_open_error(&err, None).unwrap();
        assert_eq!(failure.code, "provider_unavailable");
    }

    // ---- the agent path (CHAT_PROVIDER_PATH=agent), against a fake provider ----

    fn agent_manager() -> (
        ChatManager,
        Arc<crate::neo4j::mock::MockGraphStore>,
        super::super::agent_runtime::fake::FakeProvider,
    ) {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let state = mock_app_state();
        let mut config = test_config();
        config.provider_path = crate::chat::config::ProviderPath::Agent;
        let fake = super::super::agent_runtime::fake::FakeProvider::new();
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_provider_source(Arc::new(fake.clone()));
        (manager, graph, fake)
    }

    fn agent_request(message: &str) -> ChatRequest {
        let mut req = runner_request(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        req.spawned_by = None;
        req.runner_context = None;
        req.message = message.to_string();
        req.permission_mode = Some("acceptEdits".into());
        req
    }

    async fn next_matching(
        rx: &mut broadcast::Receiver<ChatEvent>,
        pred: impl Fn(&ChatEvent) -> bool,
    ) -> ChatEvent {
        loop {
            let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("an event within 5 s")
                .expect("channel open");
            if pred(&ev) {
                return ev;
            }
        }
    }

    #[tokio::test]
    async fn agent_path_opens_a_session_persists_the_snapshot_and_streams_a_turn() {
        use nexus_claude::agent::{AgentEvent, StopReason};
        let (manager, graph, fake) = agent_manager();
        let created = manager
            .create_session(&agent_request("hello"))
            .await
            .unwrap();
        let sid = created.session_id.clone();
        let mut rx = manager.subscribe(&sid).await.unwrap();

        // The provider got the turn, a neutral policy and the PO MCP server.
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert_eq!(spec.policy.mode, nexus_claude::agent::PolicyMode::AutoEdits);
        assert!(spec.mcp_servers.contains_key("project-orchestrator"));
        assert_eq!(
            fake.state.turns_started.lock().unwrap().as_slice(),
            ["hello"]
        );

        // What the provider reported is on the persisted session.
        let node = graph
            .get_chat_session(Uuid::parse_str(&sid).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert!(node.capabilities.is_some(), "frozen capability snapshot");
        // Like the Claude Code façade, the provider has named no session yet.
        assert_eq!(
            node.resume_token, None,
            "no resume token before the provider names one"
        );
        assert_eq!(node.provider_id.as_deref(), Some("claude-code"));

        fake.state.push(AgentEvent::Text {
            text: "hi!".into(),
            seq: None,
            parent: None,
        });
        fake.state.push(AgentEvent::Done {
            stop_reason: StopReason::Completed,
            subtype: Some("success".into()),
            is_error: false,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 5,
            duration_api_ms: None,
            num_turns: 1,
            model: Some("m".into()),
            provider_session_id: Some("p-1".into()),
            structured_output: None,
            error: None,
        });
        let text = next_matching(&mut rx, |e| matches!(e, ChatEvent::AssistantText { .. })).await;
        assert!(matches!(text, ChatEvent::AssistantText { ref content, .. } if content == "hi!"));
        let result = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        assert!(
            matches!(result, ChatEvent::Result { ref stop_reason, .. } if stop_reason.as_deref() == Some("completed"))
        );
        next_matching(&mut rx, |e| {
            matches!(
                e,
                ChatEvent::StreamingStatus {
                    is_streaming: false
                }
            )
        })
        .await;
        assert!(!manager.is_session_streaming(&sid).await);

        // The session id the turn named is persisted, without waiting for a reopen.
        let node = graph
            .get_chat_session(Uuid::parse_str(&sid).unwrap())
            .await
            .unwrap()
            .unwrap();
        let token = nexus_claude::agent::ResumeToken::from_wire(
            node.resume_token
                .as_deref()
                .expect("resume token after the turn"),
        )
        .unwrap();
        assert_eq!(token.data()["session_id"], "p-1");

        // The events were persisted for replay (never the transient ones).
        let events = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 100)
            .await
            .unwrap();
        let kinds: Vec<_> = events.iter().map(|e| e.event_type.as_str()).collect();
        assert_eq!(kinds, ["user_message", "assistant_text", "result"]);
    }

    #[tokio::test]
    async fn agent_path_queues_a_second_turn_while_one_runs_and_answers_permissions() {
        use nexus_claude::agent::AgentEvent;
        let (manager, _graph, fake) = agent_manager();
        let sid = manager
            .create_session(&agent_request("first"))
            .await
            .unwrap()
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();

        // Queued, not refused: the running turn is interrupted and the message read next.
        manager.send_message(&sid, "second").await.unwrap();
        assert_eq!(fake.state.interrupts.lock().unwrap().len(), 1);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while fake.state.turns_started.lock().unwrap().len() < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "the queued turn starts"
            );
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(fake.state.turns_started.lock().unwrap()[1].ends_with("second"));

        fake.state.push(AgentEvent::PermissionAsk {
            request_id: "perm-1".into(),
            tool_name: "Bash".into(),
            input: serde_json::json!({"command": "ls"}),
            category: Default::default(),
            canonical: None,
            tool_call_id: None,
            scopes: vec![],
            parent: None,
        });
        next_matching(&mut rx, |e| {
            matches!(e, ChatEvent::PermissionRequest { .. })
        })
        .await;
        manager
            .route_permission_response(&sid, "perm-1", true, true)
            .await
            .unwrap();
        let answers = fake.state.permission_answers.lock().unwrap().clone();
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].0, "perm-1");
    }

    #[tokio::test]
    async fn agent_path_interrupt_model_mode_and_close_reach_the_provider() {
        let (manager, _graph, fake) = agent_manager();
        let sid = manager
            .create_session(&agent_request("go"))
            .await
            .unwrap()
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();

        manager.interrupt(&sid).await.unwrap();
        assert_eq!(fake.state.interrupts.lock().unwrap().len(), 1);

        assert!(manager
            .set_session_model(&sid, "other-model")
            .await
            .unwrap());
        assert_eq!(
            fake.state.models.lock().unwrap().as_slice(),
            ["other-model"]
        );
        manager
            .set_session_permission_mode(&sid, "plan_only")
            .await
            .unwrap();
        assert_eq!(
            fake.state.modes.lock().unwrap().as_slice(),
            [nexus_claude::agent::PolicyMode::PlanOnly]
        );
        let changed = next_matching(&mut rx, |e| {
            matches!(e, ChatEvent::PermissionModeChanged { .. })
        })
        .await;
        assert!(
            matches!(changed, ChatEvent::PermissionModeChanged { ref policy_mode, .. } if policy_mode.as_deref() == Some("plan_only"))
        );

        manager.close_session(&sid).await.unwrap();
        assert!(fake.state.closed.load(Ordering::SeqCst));
        next_matching(&mut rx, |e| matches!(e, ChatEvent::SessionClosed { .. })).await;
        assert!(!manager.is_session_active(&sid).await);
    }

    #[tokio::test]
    async fn agent_path_masks_a_vault_secret_before_persisting_and_broadcasting() {
        use nexus_claude::agent::{AgentEvent, StopReason, ToolOutput};
        let secret = "sk-vault-secret-in-a-tool-output-5521";
        crate::vault::mask::global().register("AGENT_PATH_TEST_KEY", secret);
        let (manager, graph, fake) = agent_manager();
        let sid = manager
            .create_session(&agent_request("go"))
            .await
            .unwrap()
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();
        fake.state.push(AgentEvent::ToolResult {
            id: "t1".into(),
            output: Some(ToolOutput::Text(format!("cat .env -> KEY={secret}"))),
            is_error: false,
            seq: None,
            parent: None,
        });
        fake.state.push(AgentEvent::Text {
            text: format!("the key is {secret}"),
            seq: None,
            parent: None,
        });
        fake.state.push(AgentEvent::Done {
            stop_reason: StopReason::Completed,
            subtype: Some("success".into()),
            is_error: false,
            result_text: Some(format!("done, {secret}")),
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 1,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: None,
            structured_output: None,
            error: None,
        });
        let mut seen = Vec::new();
        loop {
            let ev = next_matching(&mut rx, |_| true).await;
            let done = matches!(
                ev,
                ChatEvent::StreamingStatus {
                    is_streaming: false
                }
            );
            seen.push(serde_json::to_string(&ev).unwrap());
            if done {
                break;
            }
        }
        assert!(
            seen.iter().all(|e| !e.contains(secret)),
            "broadcast: {seen:?}"
        );
        assert!(
            seen.iter().any(|e| e.contains("cat .env")),
            "the rest of the output is kept"
        );
        let stored = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 100)
            .await
            .unwrap();
        assert!(stored.iter().all(|e| !e.data.contains(secret)), "persisted");
        crate::vault::mask::global().forget("AGENT_PATH_TEST_KEY");
    }

    #[tokio::test]
    async fn agent_path_open_failure_is_typed_and_revokes_nothing_live() {
        let (manager, _graph, fake) = agent_manager();
        *fake.fail_open.lock().unwrap() = Some(nexus_claude::agent::ProviderError::CliNotFound {
            program: "claude".into(),
        });
        let err = manager
            .create_session(&agent_request("x"))
            .await
            .unwrap_err();
        let failure =
            crate::chat::provider::errors::classify_open_error(&err, Some("claude-code")).unwrap();
        assert_eq!((failure.status, failure.code), (424, "cli_not_found"));
    }

    #[tokio::test]
    async fn a_session_without_cwd_runs_in_a_neutral_directory_of_the_host_and_resumes_in_it() {
        let (manager, graph, fake) = agent_manager();
        let mut req = agent_request("hello");
        req.cwd = String::new();
        let created = manager.create_session(&req).await.unwrap();
        let sid = created.session_id.clone();
        let id = Uuid::parse_str(&sid).unwrap();

        // The marker is on the response and on the node; the cwd is the host's directory.
        assert_eq!(
            created.execution_place,
            crate::chat::neutral_place::ExecutionPlace::Neutral
        );
        let expected = crate::chat::neutral_place::dir_for(id);
        let node = graph.get_chat_session(id).await.unwrap().unwrap();
        assert_eq!(
            node.execution_place,
            crate::chat::neutral_place::ExecutionPlace::Neutral
        );
        assert_eq!(node.cwd, expected.to_string_lossy());
        // Real, empty, and what the provider was given.
        assert!(expected.is_dir());
        assert_eq!(std::fs::read_dir(&expected).unwrap().count(), 0);
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert_eq!(spec.cwd, expected);
        // No project named: the lack of graph context is said, not silent.
        assert!(node.project_slug.is_none());
        assert_eq!(created.notices.len(), 1);
        assert!(created.notices[0].contains("project_slug"));

        // Closing removes the directory; a resume makes the same one again.
        fake.state.end_turn();
        manager.close_session(&sid).await.unwrap();
        assert!(!expected.exists());
        manager.resume_session(&sid, "again", None).await.unwrap();
        assert!(expected.is_dir());
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert_eq!(spec.cwd, expected);

        manager.close_session(&sid).await.unwrap();
        assert!(!expected.exists());
    }

    #[tokio::test]
    async fn a_session_with_a_cwd_stays_a_project_session_without_notice() {
        let (manager, graph, _fake) = agent_manager();
        let created = manager
            .create_session(&agent_request("hello"))
            .await
            .unwrap();
        assert_eq!(
            created.execution_place,
            crate::chat::neutral_place::ExecutionPlace::Project
        );
        assert!(created.notices.is_empty());
        let node = graph
            .get_chat_session(Uuid::parse_str(&created.session_id).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            node.execution_place,
            crate::chat::neutral_place::ExecutionPlace::Project
        );
        manager.close_session(&created.session_id).await.unwrap();
    }

    /// What the opened session's tool policy says about a write tool.
    fn spec_refuses_writes(spec: &nexus_claude::agent::SessionSpec) -> bool {
        spec.policy.decide(
            "Write",
            Some("/tmp/x"),
            nexus_claude::agent::ToolCategory::Edit,
        ) == nexus_claude::agent::PolicyDecision::Deny
    }

    /// The MCP server environment of an opened spec.
    fn spec_po_env(
        spec: &nexus_claude::agent::SessionSpec,
    ) -> std::collections::BTreeMap<String, String> {
        match spec.mcp_servers.get("project-orchestrator") {
            Some(nexus_claude::agent::McpServerSpec::Stdio { env, .. }) => env.clone(),
            _ => panic!("stdio MCP server expected"),
        }
    }

    #[tokio::test]
    async fn a_neutral_session_opens_read_only_persists_it_and_every_resume_keeps_it() {
        use crate::chat::provider::policy::SessionAccess;
        let (manager, graph, fake) = agent_manager();
        let mut req = agent_request("hello");
        req.cwd = String::new();
        let created = manager.create_session(&req).await.unwrap();
        let sid = created.session_id.clone();
        let id = Uuid::parse_str(&sid).unwrap();

        // Said on the response, stored on the node, enforced on the open.
        assert_eq!(created.access, SessionAccess::ReadOnly);
        let node = graph.get_chat_session(id).await.unwrap().unwrap();
        assert_eq!(node.access, SessionAccess::ReadOnly);
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert!(spec_refuses_writes(&spec), "the open denies writes");
        assert_eq!(
            spec_po_env(&spec).get(crate::auth::tool_profile::TOOL_PROFILE_ENV),
            Some(&crate::auth::tool_profile::READ_ONLY.to_string())
        );

        // A resume reads the access from the node and applies it again.
        fake.state.end_turn();
        manager.close_session(&sid).await.unwrap();
        manager.resume_session(&sid, "again", None).await.unwrap();
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert!(spec_refuses_writes(&spec), "the resume denies writes too");
        assert_eq!(
            graph.get_chat_session(id).await.unwrap().unwrap().access,
            SessionAccess::ReadOnly,
            "a resume does not rewrite the access"
        );
        manager.close_session(&sid).await.unwrap();
    }

    #[tokio::test]
    async fn a_project_session_is_normal_unless_the_client_asks_for_read_only_and_a_resume_cannot_widen_it(
    ) {
        use crate::chat::provider::policy::SessionAccess;
        let (manager, graph, fake) = agent_manager();

        // Project session, nothing asked: normal, left off the wire, writes allowed.
        let normal = manager
            .create_session(&agent_request("hello"))
            .await
            .unwrap();
        assert_eq!(normal.access, SessionAccess::Normal);
        let json = serde_json::to_value(&normal).unwrap();
        assert!(json.get("access").is_none(), "normal is left off the wire");
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert!(!spec_refuses_writes(&spec));
        assert!(!spec_po_env(&spec).contains_key(crate::auth::tool_profile::TOOL_PROFILE_ENV));
        fake.state.end_turn();
        manager.close_session(&normal.session_id).await.unwrap();

        // Project session, read_only asked explicitly.
        let mut req = agent_request("hello");
        req.access = Some(SessionAccess::ReadOnly);
        let created = manager.create_session(&req).await.unwrap();
        let sid = created.session_id.clone();
        let id = Uuid::parse_str(&sid).unwrap();
        assert_eq!(created.access, SessionAccess::ReadOnly);
        assert_eq!(
            graph
                .get_chat_session(id)
                .await
                .unwrap()
                .unwrap()
                .execution_place,
            crate::chat::neutral_place::ExecutionPlace::Project
        );
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert!(spec_refuses_writes(&spec));

        // Resume: nothing in the stored node or the call can bring `normal` back.
        fake.state.end_turn();
        manager.close_session(&sid).await.unwrap();
        manager.resume_session(&sid, "again", None).await.unwrap();
        let spec = fake.state.opened_specs.lock().unwrap().pop().unwrap();
        assert!(spec_refuses_writes(&spec), "read_only -> normal is refused");
        assert_eq!(
            graph.get_chat_session(id).await.unwrap().unwrap().access,
            SessionAccess::ReadOnly
        );
        manager.close_session(&sid).await.unwrap();
    }

    #[tokio::test]
    async fn a_neutral_session_can_be_opened_normal_on_explicit_request_and_a_legacy_node_reads_as_normal(
    ) {
        use crate::chat::provider::policy::SessionAccess;
        let (manager, graph, fake) = agent_manager();
        let mut req = agent_request("hello");
        req.cwd = String::new();
        req.access = Some(SessionAccess::Normal);
        let created = manager.create_session(&req).await.unwrap();
        assert_eq!(created.access, SessionAccess::Normal);
        let id = Uuid::parse_str(&created.session_id).unwrap();
        let node = graph.get_chat_session(id).await.unwrap().unwrap();
        assert_eq!(node.access, SessionAccess::Normal);
        assert!(!spec_refuses_writes(
            &fake.state.opened_specs.lock().unwrap().pop().unwrap()
        ));
        fake.state.end_turn();
        manager.close_session(&created.session_id).await.unwrap();

        // A node written before the field existed has no `access`: it reads as normal
        // (compatibility), on the wire and on a resume.
        let mut json = serde_json::to_value(&node).unwrap();
        assert!(json.get("access").is_none());
        json.as_object_mut().unwrap().remove("access");
        let legacy: ChatSessionNode = serde_json::from_value(json).unwrap();
        assert_eq!(legacy.access, SessionAccess::Normal);
        let request: ChatRequest =
            serde_json::from_value(serde_json::json!({"message": "m"})).unwrap();
        assert_eq!(request.access, None);
    }

    #[tokio::test]
    async fn a_provider_switch_keeps_a_read_only_session_read_only() {
        use crate::chat::provider::policy::SessionAccess;
        let (manager, graph, fake) = agent_manager();
        let mut req = agent_request("hello");
        req.access = Some(SessionAccess::ReadOnly);
        let created = manager.create_session(&req).await.unwrap();
        fake.state.end_turn();
        manager.close_session(&created.session_id).await.unwrap();
        // The request a switch builds carries the access of the node it leaves.
        let node = graph
            .get_chat_session(Uuid::parse_str(&created.session_id).unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(node.access, SessionAccess::ReadOnly);
    }

    /// The token minted for a read-only session carries the read-only profile, a
    /// third-party lineage included: the restricted profile it would otherwise get is
    /// WIDER (it runs the writes of the tools it keeps).
    #[tokio::test]
    async fn the_token_of_a_read_only_session_carries_the_read_only_profile() {
        use crate::auth::tool_profile::ToolProfile;
        use crate::chat::provider::policy::SessionAccess;
        use nexus_claude::agent::ProviderKind;
        let manager = signed_manager(mock_app_state());
        let claims = person_claims();
        for (third_party, mode) in [
            (false, "default"),
            (false, "bypassPermissions"),
            (true, "default"),
            (true, "bypassPermissions"),
        ] {
            let spec = manager
                .build_agent_spec_with_access(
                    AgentSpecInput {
                        cwd: "/tmp",
                        model: "m",
                        system_prompt: "p",
                        permission_mode: Some(mode),
                        add_dirs: &[],
                        user_claims: Some(&claims),
                        session_id: "ro-token",
                        third_party,
                        max_tokens: None,
                        kind: ProviderKind::Native,
                        remote_cwd: None,
                        hooks: None,
                    },
                    SessionAccess::ReadOnly,
                )
                .await
                .unwrap();
            let token = spec_po_env(&spec)
                .get("PO_AUTH_TOKEN")
                .cloned()
                .expect("a session token");
            let profile = ToolProfile::from_unverified_token(&token);
            assert_eq!(
                profile,
                ToolProfile::ReadOnly,
                "third_party={third_party} {mode}"
            );
            let call = |action: &str| -> crate::mcp::protocol::ToolCallParams {
                serde_json::from_value(serde_json::json!({
                    "name": "task", "arguments": {"action": action}
                }))
                .unwrap()
            };
            assert!(crate::mcp::server::profile_refusal(profile, &call("create")).is_some());
            assert_eq!(
                crate::mcp::server::profile_refusal(profile, &call("list")),
                None
            );
        }
    }

    /// VERIFIER (4): the Claude CLI really receives the deny list when the mode is the
    /// most permissive one. The CLI is the `fake_claude` of nexus, which records its
    /// argv; a real `claude` is not available here and nothing below runs one.
    #[tokio::test]
    async fn the_cli_gets_the_read_only_deny_list_under_bypass_permissions() {
        use crate::chat::provider::policy::SessionAccess;
        let fake = std::env::var_os("NEXUS_FAKES_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../.target-nexus-fakes/debug")
            })
            .join("fake_claude");
        assert!(
            fake.exists(),
            "{} is missing: build the nexus fakes (.github/actions/nexus-fakes builds fake_claude with the others)",
            fake.display()
        );
        let dir = tempfile::tempdir().unwrap();
        let argv_out = dir.path().join("argv.json");
        let transcript = dir.path().join("transcript.jsonl");
        std::fs::write(&transcript, "{\"op\":\"exit\",\"code\":0}\n").unwrap();

        let state = mock_app_state();
        let mut config = test_config();
        config.permission.disallowed_tools = vec!["Bash(rm -rf *)".into()];
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);
        for (access, expect_denied) in [
            (SessionAccess::ReadOnly, true),
            (SessionAccess::Normal, false),
        ] {
            let mut options = manager
                .build_options_with_access(
                    "/tmp",
                    "m",
                    "p",
                    None,
                    Some("bypassPermissions"),
                    None,
                    &[],
                    None,
                    Some("ro-cli"),
                    access,
                )
                .await;
            options.cli_path = Some(fake.clone());
            options.env.insert(
                "FAKE_CLAUDE_TRANSCRIPT".into(),
                transcript.display().to_string(),
            );
            options.env.insert(
                "FAKE_CLAUDE_ARGS_OUT".into(),
                argv_out.display().to_string(),
            );
            let _ = std::fs::remove_file(&argv_out);
            let mut client = nexus_claude::InteractiveClient::new(options).unwrap();
            // The fake exits at once; the connection may or may not survive that, the
            // recorded argv is what matters.
            let _ = client.connect().await;
            for _ in 0..100 {
                if argv_out.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            let recorded: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&argv_out).expect("argv recorded"))
                    .unwrap();
            let argv = recorded.to_string();
            assert!(
                argv.contains("bypassPermissions"),
                "the mode is the permissive one: {argv}"
            );
            assert!(
                argv.contains("Bash(rm -rf *)"),
                "configured entries stay: {argv}"
            );
            for tool in [
                "Write",
                "Edit",
                "Bash",
                "Task",
                "mcp__project-orchestrator__admin",
            ] {
                assert_eq!(
                    argv.contains(&format!("\"{tool}\""))
                        || argv.contains(&format!("{tool},"))
                        || argv.contains(&format!(",{tool}"))
                        || argv.contains(&format!("{tool}\"")),
                    expect_denied,
                    "{tool} in the CLI's disallowed tools ({access:?}): {argv}"
                );
            }
            let _ = client.disconnect().await;
        }
    }

    #[tokio::test]
    async fn agent_path_resumes_from_the_persisted_token_and_out_of_turn_output_regroups() {
        use nexus_claude::agent::AgentEvent;
        let (manager, graph, fake) = agent_manager();
        let sid = manager
            .create_session(&agent_request("one"))
            .await
            .unwrap()
            .session_id;
        // The provider names its session during the first turn, never at open.
        let mut rx = manager.subscribe(&sid).await.unwrap();
        let mut done = done_event(false, None);
        if let AgentEvent::Done {
            provider_session_id,
            ..
        } = &mut done
        {
            *provider_session_id = Some("fake-provider-session".into());
        }
        fake.state.push(done);
        next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        fake.state.end_turn();
        manager.agent_runtime.close(&sid).await.unwrap();
        assert!(!manager.is_session_active(&sid).await);

        manager.resume_session(&sid, "two", None).await.unwrap();
        let token = fake
            .state
            .resumed_with
            .lock()
            .unwrap()
            .pop()
            .expect("resumed with a token");
        assert_eq!(token.data()["session_id"], "fake-provider-session");
        assert_eq!(
            fake.state
                .turns_started
                .lock()
                .unwrap()
                .last()
                .map(String::as_str),
            Some("two")
        );
        // Event numbers continue after the persisted ones.
        let events = graph
            .get_chat_events(Uuid::parse_str(&sid).unwrap(), 0, 100)
            .await
            .unwrap();
        let seqs: Vec<i64> = events.iter().map(|e| e.seq).collect();
        let mut sorted = seqs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(seqs.len(), sorted.len(), "no event number reused: {seqs:?}");

        let mut rx = manager.subscribe(&sid).await.unwrap();
        fake.state.push_oob(AgentEvent::Text {
            text: "background line".into(),
            seq: Some(9),
            parent: None,
        });
        let bg = next_matching(&mut rx, |e| matches!(e, ChatEvent::BackgroundOutput { .. })).await;
        assert!(
            matches!(bg, ChatEvent::BackgroundOutput { ref source, ref content, .. } if source == "assistant" && content == "background line")
        );
    }

    #[tokio::test]
    async fn claude_code_stays_on_the_legacy_engine_by_default_hooks_included() {
        let graph = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let state = mock_app_state();
        let config = test_config();
        assert_eq!(
            config.provider_path,
            crate::chat::config::ProviderPath::Legacy
        );
        let fake = super::super::agent_runtime::fake::FakeProvider::new();
        let manager = ChatManager::new_without_memory(dyn_graph, state.meili, config)
            .with_provider_source(Arc::new(fake.clone()));
        // The Claude CLI is not there in tests: the call may fail, but it must
        // NOT have gone through the agent engine (which has no hooks, queue, retry...).
        let _ = manager.create_session(&agent_request("hello")).await;
        assert_eq!(manager.agent_runtime.len().await, 0);
        assert!(
            fake.state.opened_specs.lock().unwrap().is_empty(),
            "the agent provider was never asked"
        );
        let nodes: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
        assert_eq!(nodes.len(), 1);
        assert!(
            nodes[0].capabilities.is_none(),
            "no agent snapshot: it is a legacy session"
        );
    }

    #[tokio::test]
    async fn forcing_claude_code_onto_the_agent_engine_says_what_is_missing() {
        use nexus_claude::agent::AgentEvent;
        let (manager, _graph, fake) = agent_manager(); // CHAT_PROVIDER_PATH=agent
        let sid = manager
            .create_session(&agent_request("go"))
            .await
            .unwrap()
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();
        fake.state.push(AgentEvent::SessionStarted {
            provider_session_id: Some("p1".into()),
            model: Some("m".into()),
            policy_mode: None,
            native_mode: None,
            tools: vec![],
            mcp_servers: vec![],
            cwd: None,
        });
        let init = next_matching(&mut rx, |e| matches!(e, ChatEvent::SystemInit { .. })).await;
        let wire = serde_json::to_value(&init).unwrap();
        assert_eq!(wire["engine"], "agent", "{wire}");
        let degraded: Vec<String> = serde_json::from_value(wire["degraded_features"].clone())
            .unwrap_or_else(|_| panic!("degraded_features missing: {wire}"));
        for lost in ["hooks", "compaction", "images"] {
            assert!(
                degraded.iter().any(|d| d == lost),
                "{lost} must be listed: {degraded:?}"
            );
        }
    }

    /// The `system_init` a live agent session REALLY emits (not a fixture).
    async fn real_agent_system_init(
        manager: &ChatManager,
        fake: &super::super::agent_runtime::fake::FakeProvider,
    ) -> serde_json::Value {
        use nexus_claude::agent::AgentEvent;
        let sid = manager
            .create_session(&agent_request("go"))
            .await
            .unwrap()
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();
        fake.state.push(AgentEvent::SessionStarted {
            provider_session_id: Some("p1".into()),
            model: Some("m".into()),
            policy_mode: None,
            native_mode: None,
            tools: vec![],
            mcp_servers: vec![],
            cwd: None,
        });
        let init = next_matching(&mut rx, |e| matches!(e, ChatEvent::SystemInit { .. })).await;
        serde_json::to_value(&init).unwrap()
    }

    #[tokio::test]
    async fn the_agent_system_init_lists_what_the_session_capabilities_do_not_cover() {
        // A provider that declares images and a compaction signal: those two are not missing.
        let (manager, _graph, fake) = agent_manager();
        {
            let mut caps = fake.caps.lock().unwrap();
            caps.images = true;
            caps.compaction_signal = true;
        }
        let wire = real_agent_system_init(&manager, &fake).await;
        assert_eq!(wire["engine"], "agent", "{wire}");
        let degraded: Vec<String> =
            serde_json::from_value(wire["degraded_features"].clone()).expect("a list");
        // This provider runs no hooks in its loop: the graph hooks are listed...
        assert!(degraded.iter().any(|d| d == "hooks"), "{degraded:?}");
        // ...what the engine ported is not claimed missing...
        let ported = ["enrichment", "message_queue", "auto_continue", "nats"];
        assert!(
            !degraded.iter().any(|d| ported.contains(&d.as_str())),
            "{degraded:?}"
        );
        // ...and what the provider covers is not claimed missing.
        assert!(
            !degraded.iter().any(|d| d == "images" || d == "compaction"),
            "{degraded:?}"
        );

        // A provider that declares neither: both are listed.
        let (manager, _graph, fake) = agent_manager();
        let wire = real_agent_system_init(&manager, &fake).await;
        let degraded: Vec<String> =
            serde_json::from_value(wire["degraded_features"].clone()).unwrap();
        assert!(
            degraded.iter().any(|d| d == "images") && degraded.iter().any(|d| d == "compaction")
        );
    }

    #[test]
    fn the_legacy_system_init_says_legacy_with_nothing_missing() {
        // `message_to_events` is what the legacy stream loop emits for the CLI's init.
        let msg = Message::System {
            subtype: "init".into(),
            data: serde_json::json!({"session_id": "cli-1", "model": "m"}),
        };
        let events = ChatManager::message_to_events(&msg);
        let wire = serde_json::to_value(&events[0]).unwrap();
        assert_eq!(wire["engine"], "legacy", "{wire}");
        assert_eq!(wire["degraded_features"], serde_json::json!([]), "{wire}");
    }

    fn done_event(
        is_error: bool,
        error: Option<nexus_claude::agent::ProviderError>,
    ) -> nexus_claude::agent::AgentEvent {
        nexus_claude::agent::AgentEvent::Done {
            stop_reason: if is_error {
                nexus_claude::agent::StopReason::Error
            } else {
                nexus_claude::agent::StopReason::Completed
            },
            subtype: Some(
                if is_error {
                    "error_during_execution"
                } else {
                    "success"
                }
                .into(),
            ),
            is_error,
            result_text: None,
            usage: Default::default(),
            cost: Default::default(),
            duration_ms: 1,
            duration_api_ms: None,
            num_turns: 1,
            model: None,
            provider_session_id: Some("p".into()),
            structured_output: None,
            error,
        }
    }

    #[tokio::test]
    async fn a_retryable_done_error_before_any_output_is_retried_once_the_provider_says_when() {
        use nexus_claude::agent::{AgentEvent, ProviderError};
        let (manager, _graph, fake) = agent_manager();
        let sid = manager
            .create_session(&agent_request("go"))
            .await
            .unwrap()
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();
        fake.state.push(done_event(
            true,
            Some(ProviderError::RateLimited {
                retry_after_ms: Some(10),
            }),
        ));
        let retrying = next_matching(&mut rx, |e| matches!(e, ChatEvent::Retrying { .. })).await;
        assert!(matches!(
            retrying,
            ChatEvent::Retrying {
                attempt: 1,
                delay_ms: 10,
                ..
            }
        ));
        // The same turn is sent again.
        for _ in 0..200 {
            if fake.state.turns_started.lock().unwrap().len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            fake.state.turns_started.lock().unwrap().as_slice(),
            ["go", "go"]
        );
        fake.state.push(AgentEvent::Text {
            text: "ok".into(),
            seq: None,
            parent: None,
        });
        fake.state.push(done_event(false, None));
        let result = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        // The failed attempt never reached the user: the first Result is the good one.
        assert!(
            matches!(
                result,
                ChatEvent::Result {
                    is_error: false,
                    ..
                }
            ),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn a_turn_that_already_showed_output_or_failed_for_good_is_not_retried() {
        use nexus_claude::agent::{AgentEvent, ProviderError};
        let (manager, _graph, fake) = agent_manager();
        let sid = manager
            .create_session(&agent_request("go"))
            .await
            .unwrap()
            .session_id;
        let mut rx = manager.subscribe(&sid).await.unwrap();
        fake.state.push(AgentEvent::Text {
            text: "partial".into(),
            seq: None,
            parent: None,
        });
        fake.state
            .push(done_event(true, Some(ProviderError::Overloaded)));
        let result = next_matching(&mut rx, |e| matches!(e, ChatEvent::Result { .. })).await;
        assert!(matches!(result, ChatEvent::Result { is_error: true, .. }));
        assert_eq!(
            fake.state.turns_started.lock().unwrap().len(),
            1,
            "no replay after output"
        );

        // A non-retryable cause is shown at once.
        let sid2 = manager
            .create_session(&agent_request("again"))
            .await
            .unwrap()
            .session_id;
        let mut rx2 = manager.subscribe(&sid2).await.unwrap();
        fake.state
            .push(done_event(true, Some(ProviderError::Unauthorized)));
        let result = next_matching(&mut rx2, |e| matches!(e, ChatEvent::Result { .. })).await;
        assert!(matches!(result, ChatEvent::Result { is_error: true, .. }));
    }

    #[tokio::test]
    async fn an_agent_session_of_claude_code_resumed_without_the_engine_is_a_typed_engine_error() {
        let (manager, graph, _fake) = agent_manager();
        let sid = manager
            .create_session(&agent_request("one"))
            .await
            .unwrap()
            .session_id;
        manager.agent_runtime.close(&sid).await.unwrap();
        // The flag is back to its default: the engine of that session is not available.
        let state = mock_app_state();
        let dyn_graph: Arc<dyn GraphStore> = graph.clone();
        let legacy = ChatManager::new_without_memory(dyn_graph, state.meili, test_config());
        let err = legacy.resume_session(&sid, "two", None).await.unwrap_err();
        let failure =
            crate::chat::provider::errors::classify_open_error(&err, None).expect("typed");
        assert_eq!((failure.status, failure.code), (409, "engine_unavailable"));
        assert!(
            failure.message.contains("CHAT_PROVIDER_PATH"),
            "{}",
            failure.message
        );
    }

    #[tokio::test]
    async fn legacy_path_never_touches_the_agent_runtime() {
        let (manager, _graph) = {
            let (m, g) = manager_with_mock();
            (m, g)
        };
        let _ = manager
            .create_session(&runner_request(
                Uuid::new_v4(),
                Uuid::new_v4(),
                Uuid::new_v4(),
            ))
            .await;
        assert_eq!(manager.agent_runtime.len().await, 0);
    }

    #[tokio::test]
    async fn test_resume_session_keeps_identity_and_links() {
        let (manager, graph) = manager_with_mock();
        let (run, plan, task) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let mut s = test_chat_session(None);
        s.spawned_by = runner_request(run, plan, task).spawned_by;
        graph.create_chat_session(&s).await.unwrap();
        graph
            .link_session_to_run(&s.id.to_string(), run, Some(plan), Some(task))
            .await
            .unwrap();
        let before = crate::chat::attachment::attach(
            &[s.clone()],
            &graph.session_link_rows.read().await.clone(),
        );

        let rows_before = graph.session_link_rows.read().await.clone();

        // WHAT THIS PROVES: `resume_session` reads the stored session, never
        // creates another one and never touches its link rows, up to the CLI
        // spawn. In tests the spawn itself fails (or succeeds against whatever
        // `claude` is installed), so what happens to the links AFTER a real
        // spawn is NOT exercised here: `resume_session` has no call to
        // `link_session_to_run` / `create_spawned_by_relation` at all
        // (they live in `create_session` only), which is what the row-for-row
        // comparison below pins on the mock.
        let _ = manager
            .resume_session(&s.id.to_string(), "continue", None)
            .await;

        let after_sessions: Vec<_> = graph.chat_sessions.read().await.values().cloned().collect();
        assert_eq!(
            after_sessions.len(),
            1,
            "resume must not create another session"
        );
        assert_eq!(after_sessions[0].id, s.id, "same session id");
        assert_eq!(
            after_sessions[0].spawned_by, s.spawned_by,
            "spawned_by kept"
        );
        let rows = graph.session_link_rows.read().await.clone();
        assert_eq!(
            rows, rows_before,
            "the stored link rows are not rewritten, added to or removed by a resume"
        );
        let after = crate::chat::attachment::attach(&after_sessions, &rows);
        assert_eq!(after.by_plan, before.by_plan, "same links, same thread");
        assert!(after.unattached.is_empty(), "never a detached session");
    }

    #[tokio::test]
    #[ignore = "needs the claude CLI on PATH (run with: cargo test -- --ignored)"]
    async fn live_snapshot_reports_the_permissions_each_live_cli_still_holds() {
        let (manager, _graph) = manager_with_mock();
        let with = Uuid::new_v4();
        let without = Uuid::new_v4();
        let _a = super::test_support::insert_live_session(
            &manager,
            &with.to_string(),
            true,
            &["req-1", "req-2"],
        )
        .await
        .expect("the Claude CLI binary must be installed to run this test");
        let _b =
            super::test_support::insert_live_session(&manager, &without.to_string(), false, &[])
                .await
                .expect("the Claude CLI binary must be installed to run this test");
        let snap = manager.live_session_snapshot().await;
        assert_eq!(snap.live, [with, without].into_iter().collect());
        assert_eq!(snap.streaming, [with].into_iter().collect());
        assert_eq!(
            snap.pending_permissions.get(&with),
            Some(
                &["req-1".to_string(), "req-2".to_string()]
                    .into_iter()
                    .collect()
            )
        );
        assert!(!snap.pending_permissions.contains_key(&without));
    }

    /// `is_streaming` lives in an `AtomicBool` and is written to Neo4j
    /// nowhere, so turning the in-memory snapshot into the per-session
    /// activity the API reports is the whole mechanism behind a working
    /// indicator that survives a page reload. These tests need no CLI: the
    /// mapping and the counting are pure.
    mod activity {
        use super::super::{count_background_tasks, LiveSessionSnapshot};
        use crate::chat::types::{BackgroundTaskInfo, BackgroundTaskKind};
        use uuid::Uuid;

        fn task(kind: BackgroundTaskKind, dying: bool) -> BackgroundTaskInfo {
            let now = chrono::Utc::now();
            BackgroundTaskInfo {
                id: Uuid::new_v4().to_string(),
                kind,
                description: "watch the log".into(),
                started_at: now,
                last_seen_at: now,
                pid: Some(1234),
                parent_tool_use_id: None,
                pending_removal_at: dying.then(std::time::Instant::now),
            }
        }

        #[test]
        fn an_unknown_session_is_quiet_rather_than_unknown() {
            let snap = LiveSessionSnapshot::default();
            let a = snap.activity_for(Uuid::new_v4());
            assert!(
                a.is_quiet(),
                "absence from the map is an answer: nothing is running"
            );
            assert!(!a.live);
            assert_eq!(a.background_tasks(), 0);
        }

        #[test]
        fn a_live_session_reports_what_it_is_waiting_on() {
            let id = Uuid::new_v4();
            let other = Uuid::new_v4();
            let mut snap = LiveSessionSnapshot::default();
            snap.live.insert(id);
            snap.live.insert(other);
            snap.streaming.insert(id);
            snap.pending_permissions.insert(
                id,
                ["req-1".to_string(), "req-2".to_string()]
                    .into_iter()
                    .collect(),
            );
            snap.background_tasks.insert(id, (2, 1));

            let a = snap.activity_for(id);
            assert!(a.live && a.streaming);
            assert_eq!(a.pending_permissions, 2);
            assert_eq!(a.monitors, 2);
            assert_eq!(a.bash_tasks, 1);
            assert_eq!(a.background_tasks(), 3);
            assert!(!a.is_quiet());

            // A second live session must not inherit the first one's state —
            // the bug this guards against is one row's indicator leaking onto
            // every other row.
            let b = snap.activity_for(other);
            assert!(b.live);
            assert!(!b.streaming);
            assert_eq!(b.pending_permissions, 0);
            assert_eq!(b.background_tasks(), 0);
        }

        #[test]
        fn a_live_but_idle_session_is_not_quiet() {
            let id = Uuid::new_v4();
            let mut snap = LiveSessionSnapshot::default();
            snap.live.insert(id);
            // The distinction the UI needs: "the agent is up, waiting for you"
            // must be sayable, because showing nothing there is what made a
            // running conversation look dead.
            assert!(!snap.activity_for(id).is_quiet());
        }

        #[test]
        fn counting_separates_monitors_from_background_commands() {
            let tasks = [
                task(BackgroundTaskKind::Monitor, false),
                task(BackgroundTaskKind::Monitor, false),
                task(BackgroundTaskKind::BashBackground, false),
            ];
            assert_eq!(count_background_tasks(tasks.iter()), (2, 1));
        }

        #[test]
        fn a_cancelled_watch_stops_being_advertised_at_once() {
            // `pending_removal_at` entries linger for a 5 s grace period so
            // late ticks still route. Counting them would keep a cancelled
            // watch on screen for those five seconds.
            let tasks = [
                task(BackgroundTaskKind::Monitor, true),
                task(BackgroundTaskKind::BashBackground, true),
            ];
            assert_eq!(count_background_tasks(tasks.iter()), (0, 0));
        }

        #[test]
        fn nothing_tracked_counts_as_nothing() {
            assert_eq!(count_background_tasks(std::iter::empty()), (0, 0));
        }
    }

    #[tokio::test]
    #[ignore = "needs the claude CLI on PATH (run with: cargo test -- --ignored)"]
    async fn failed_stdin_send_keeps_the_permission_pending_for_a_retry() {
        let (manager, _graph) = manager_with_mock();
        let sid = Uuid::new_v4().to_string();
        let (stdin_rx, _q) =
            super::test_support::insert_live_session(&manager, &sid, false, &["req-1"])
                .await
                .expect("the Claude CLI binary must be installed to run this test");
        // The CLI is gone: the stdin channel is closed, the send fails.
        drop(stdin_rx);
        let mut events = {
            let sessions = manager.active_sessions.read().await;
            sessions[&sid].events_tx.subscribe()
        };

        for attempt in 1..=2 {
            let err = manager
                .send_permission_response_inner(&sid, "req-1", true, true)
                .await
                .unwrap_err();
            assert!(
                matches!(err, PermissionDeliveryError::Failed(_)),
                "attempt {attempt}: a failed send stays a delivery failure, not NotPending ({err})"
            );
        }
        let pending = manager.active_sessions.read().await[&sid]
            .pending_permission_inputs
            .clone();
        assert!(
            pending.lock().await.contains_key("req-1"),
            "the request is still pending"
        );
        assert!(
            events.try_recv().is_err(),
            "no permission_decision is broadcast for a decision that was not sent"
        );
    }

    // ====================================================================
    // ChatSession CRUD via GraphStore (mock)
    // ====================================================================

    #[tokio::test]
    async fn test_chat_session_crud_lifecycle() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        // Create
        let session = test_chat_session(Some("my-project"));
        graph.create_chat_session(&session).await.unwrap();

        // Get
        let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
        assert_eq!(fetched.id, session.id);
        assert_eq!(fetched.cwd, "/tmp/test");
        assert_eq!(fetched.model, "claude-opus-4-6");
        assert_eq!(fetched.project_slug.as_deref(), Some("my-project"));

        // Update
        let updated = graph
            .update_chat_session(
                session.id,
                Some("cli-abc-123".into()),
                Some("My Chat".into()),
                Some(5),
                Some(0.25),
                None,
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.cli_session_id.as_deref(), Some("cli-abc-123"));
        assert_eq!(updated.title.as_deref(), Some("My Chat"));
        assert_eq!(updated.message_count, 5);
        assert_eq!(updated.total_cost_usd, Some(0.25));

        // Delete
        let deleted = graph.delete_chat_session(session.id).await.unwrap();
        assert!(deleted);

        // Get after delete
        let gone = graph.get_chat_session(session.id).await.unwrap();
        assert!(gone.is_none());
    }

    #[tokio::test]
    async fn test_chat_session_list_with_filter() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let s1 = test_chat_session(Some("project-a"));
        let s2 = test_chat_session(Some("project-a"));
        let s3 = test_chat_session(Some("project-b"));
        let s4 = test_chat_session(None);

        graph.create_chat_session(&s1).await.unwrap();
        graph.create_chat_session(&s2).await.unwrap();
        graph.create_chat_session(&s3).await.unwrap();
        graph.create_chat_session(&s4).await.unwrap();

        // All sessions
        let (all, total) = graph
            .list_chat_sessions(None, None, 50, 0, false)
            .await
            .unwrap();
        assert_eq!(total, 4);
        assert_eq!(all.len(), 4);

        // Filter by project-a
        let (filtered, total) = graph
            .list_chat_sessions(Some("project-a"), None, 50, 0, false)
            .await
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(filtered.len(), 2);

        // Pagination
        let (page, total) = graph
            .list_chat_sessions(Some("project-a"), None, 1, 0, false)
            .await
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(page.len(), 1);
    }

    #[tokio::test]
    async fn test_chat_session_update_partial() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let session = test_chat_session(None);
        graph.create_chat_session(&session).await.unwrap();

        // Update only title
        let updated = graph
            .update_chat_session(
                session.id,
                None,
                Some("Title only".into()),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.title.as_deref(), Some("Title only"));
        assert!(updated.cli_session_id.is_none()); // unchanged
        assert_eq!(updated.message_count, 0); // unchanged
    }

    #[tokio::test]
    async fn test_chat_session_update_nonexistent() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let result = graph
            .update_chat_session(uuid::Uuid::new_v4(), None, None, None, None, None, None)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_chat_session_delete_nonexistent() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let deleted = graph
            .delete_chat_session(uuid::Uuid::new_v4())
            .await
            .unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_chat_session_node_serialization() {
        let session = ChatSessionNode {
            routing_pool: None,
            routing_mode: None,
            id: uuid::Uuid::new_v4(),
            cli_session_id: Some("cli-123".into()),
            project_slug: Some("test-proj".into()),
            workspace_slug: None,
            cwd: "/home/user/code".into(),
            title: Some("My session".into()),
            model: "claude-opus-4-6".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 10,
            total_cost_usd: Some(1.50),
            conversation_id: Some("conv-abc-123".into()),
            preview: Some("Hello, can you help me with this?".into()),
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
            provider_id: None,
            routed_by: None,
            capabilities: None,
            resume_token: None,
            execution_place: Default::default(),
            access: Default::default(),
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, session.id);
        assert_eq!(deserialized.cli_session_id, session.cli_session_id);
        assert_eq!(deserialized.message_count, 10);
        assert_eq!(deserialized.total_cost_usd, Some(1.50));
    }

    // ====================================================================
    // new_without_memory — fields
    // ====================================================================

    #[test]
    fn test_new_without_memory_has_no_injector() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        assert!(manager.context_injector.is_none());
        assert!(manager.memory_config.is_none());
    }

    // ====================================================================
    // get_session_messages — error paths
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_messages_no_injector() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // Session doesn't exist in mock store — should get "not found"
        let result = manager
            .get_session_messages(&Uuid::new_v4().to_string(), None, None)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_get_session_messages_invalid_uuid() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.get_session_messages("not-a-uuid", None, None).await;
        assert!(result.is_err());
        // Invalid UUID is rejected before any storage lookup
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid session ID"));
    }

    // ====================================================================
    // get_session_messages — returns structured events (tool_use, etc.)
    // ====================================================================

    #[tokio::test]
    async fn test_get_session_messages_returns_tool_use_events() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(
            state.neo4j.clone(),
            state.meili.clone(),
            test_config(),
        );

        // Create a session
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Store events including tool_use and tool_result
        let events = vec![
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 1,
                event_type: "user_message".into(),
                data: serde_json::to_string(&ChatEvent::UserMessage {
                    content: "List my plans".into(),
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 2,
                event_type: "tool_use".into(),
                data: serde_json::to_string(&ChatEvent::ToolUse {
                    id: "tu_1".into(),
                    tool: "list_plans".into(),
                    input: serde_json::json!({"status": "in_progress"}),
                    parent_tool_use_id: None,
                    category: None,
                    canonical: None,
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 3,
                event_type: "tool_result".into(),
                data: serde_json::to_string(&ChatEvent::ToolResult {
                    id: "tu_1".into(),
                    result: serde_json::json!({"plans": []}),
                    is_error: false,
                    parent_tool_use_id: None,
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
            ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: 4,
                event_type: "assistant_text".into(),
                data: serde_json::to_string(&ChatEvent::AssistantText {
                    content: "You have no in-progress plans.".into(),
                    parent_tool_use_id: None,
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            },
        ];
        state
            .neo4j
            .store_chat_events(session_id, events)
            .await
            .unwrap();

        // Retrieve via get_session_messages
        let page = manager
            .get_session_messages(&session_id.to_string(), None, None)
            .await
            .unwrap();

        assert_eq!(page.total_count, 4);
        assert_eq!(page.events.len(), 4);

        // Verify event types are preserved
        assert_eq!(page.events[0].event_type, "user_message");
        assert_eq!(page.events[1].event_type, "tool_use");
        assert_eq!(page.events[2].event_type, "tool_result");
        assert_eq!(page.events[3].event_type, "assistant_text");

        // Verify tool_use data is intact
        let tool_use: ChatEvent = serde_json::from_str(&page.events[1].data).unwrap();
        match tool_use {
            ChatEvent::ToolUse {
                id, tool, input, ..
            } => {
                assert_eq!(id, "tu_1");
                assert_eq!(tool, "list_plans");
                assert_eq!(input, serde_json::json!({"status": "in_progress"}));
            }
            _ => panic!("Expected ToolUse event"),
        }

        // Verify ordering by seq
        assert_eq!(page.events[0].seq, 1);
        assert_eq!(page.events[1].seq, 2);
        assert_eq!(page.events[2].seq, 3);
        assert_eq!(page.events[3].seq, 4);
    }

    #[tokio::test]
    async fn test_get_session_messages_pagination() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(
            state.neo4j.clone(),
            state.meili.clone(),
            test_config(),
        );

        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Store 5 events
        let events: Vec<ChatEventRecord> = (1..=5)
            .map(|i| ChatEventRecord {
                id: Uuid::new_v4(),
                session_id,
                seq: i,
                event_type: "assistant_text".into(),
                data: serde_json::to_string(&ChatEvent::AssistantText {
                    content: format!("Message {}", i),
                    parent_tool_use_id: None,
                })
                .unwrap(),
                created_at: chrono::Utc::now(),
            })
            .collect();
        state
            .neo4j
            .store_chat_events(session_id, events)
            .await
            .unwrap();

        // First page: offset=0, limit=2
        let page1 = manager
            .get_session_messages(&session_id.to_string(), Some(2), Some(0))
            .await
            .unwrap();
        assert_eq!(page1.events.len(), 2);
        assert_eq!(page1.total_count, 5);
        assert_eq!(page1.events[0].seq, 1);
        assert_eq!(page1.events[1].seq, 2);

        // Second page: offset=2, limit=2
        let page2 = manager
            .get_session_messages(&session_id.to_string(), Some(2), Some(2))
            .await
            .unwrap();
        assert_eq!(page2.events.len(), 2);
        assert_eq!(page2.events[0].seq, 3);
        assert_eq!(page2.events[1].seq, 4);

        // Last page: offset=4, limit=2
        let page3 = manager
            .get_session_messages(&session_id.to_string(), Some(2), Some(4))
            .await
            .unwrap();
        assert_eq!(page3.events.len(), 1);
        assert_eq!(page3.events[0].seq, 5);
    }

    #[tokio::test]
    async fn test_get_session_messages_empty_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(
            state.neo4j.clone(),
            state.meili.clone(),
            test_config(),
        );

        // Create session with no events
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let page = manager
            .get_session_messages(&session_id.to_string(), None, None)
            .await
            .unwrap();

        assert_eq!(page.total_count, 0);
        assert!(page.events.is_empty());
        assert!(!page.has_more);
    }

    // ====================================================================
    // conversation_id in mock GraphStore CRUD
    // ====================================================================

    #[tokio::test]
    async fn test_chat_session_update_conversation_id() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let session = test_chat_session(None);
        graph.create_chat_session(&session).await.unwrap();
        assert!(session.conversation_id.is_none());

        // Update conversation_id
        let updated = graph
            .update_chat_session(
                session.id,
                None,
                None,
                None,
                None,
                Some("conv-new-123".into()),
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.conversation_id.as_deref(), Some("conv-new-123"));

        // Fetch and verify persisted
        let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
        assert_eq!(fetched.conversation_id.as_deref(), Some("conv-new-123"));
    }

    #[tokio::test]
    async fn test_chat_session_create_with_conversation_id() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let mut session = test_chat_session(Some("proj"));
        session.conversation_id = Some("conv-init-456".into());
        graph.create_chat_session(&session).await.unwrap();

        let fetched = graph.get_chat_session(session.id).await.unwrap().unwrap();
        assert_eq!(fetched.conversation_id.as_deref(), Some("conv-init-456"));
    }

    #[tokio::test]
    async fn test_chat_session_conversation_id_survives_other_updates() {
        let state = mock_app_state();
        let graph = &state.neo4j;

        let session = test_chat_session(None);
        graph.create_chat_session(&session).await.unwrap();

        // Set conversation_id
        graph
            .update_chat_session(
                session.id,
                None,
                None,
                None,
                None,
                Some("conv-persist".into()),
                None,
            )
            .await
            .unwrap();

        // Update title only — conversation_id should be preserved
        let updated = graph
            .update_chat_session(
                session.id,
                None,
                Some("New Title".into()),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.title.as_deref(), Some("New Title"));
        assert_eq!(updated.conversation_id.as_deref(), Some("conv-persist"));
    }

    #[tokio::test]
    async fn test_chat_session_node_serialization_with_conversation_id() {
        let session = ChatSessionNode {
            routing_pool: None,
            routing_mode: None,
            id: uuid::Uuid::new_v4(),
            cli_session_id: None,
            project_slug: None,
            workspace_slug: None,
            cwd: "/tmp".into(),
            title: None,
            model: "model".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: Some("conv-serde-test".into()),
            preview: None,
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
            provider_id: None,
            routed_by: None,
            capabilities: None,
            resume_token: None,
            execution_place: Default::default(),
            access: Default::default(),
        };

        let json = serde_json::to_string(&session).unwrap();
        assert!(json.contains("conv-serde-test"));

        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.conversation_id.as_deref(),
            Some("conv-serde-test")
        );
    }

    #[tokio::test]
    async fn test_chat_session_node_serialization_without_conversation_id() {
        let session = ChatSessionNode {
            routing_pool: None,
            routing_mode: None,
            id: uuid::Uuid::new_v4(),
            cli_session_id: None,
            project_slug: None,
            workspace_slug: None,
            cwd: "/tmp".into(),
            title: None,
            model: "model".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: None,
            spawned_by: None,
            provider_id: None,
            routed_by: None,
            capabilities: None,
            resume_token: None,
            execution_place: Default::default(),
            access: Default::default(),
        };

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: ChatSessionNode = serde_json::from_str(&json).unwrap();
        assert!(deserialized.conversation_id.is_none());
    }

    // ====================================================================
    // get_streaming_snapshot
    // ====================================================================

    #[tokio::test]
    async fn test_get_streaming_snapshot_nonexistent_session() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let (is_streaming, text, events) = manager.get_streaming_snapshot("nonexistent").await;
        assert!(!is_streaming);
        assert!(text.is_empty());
        assert!(events.is_empty());
    }

    /// Helper: create a dummy InteractiveClient for tests.
    /// Returns None if the Claude CLI is not installed (e.g., in CI).
    /// A client that needs no Claude CLI: it sits on the SDK's in-memory
    /// transport. `InteractiveClient::new` looks the CLI up on disk, so the
    /// tests built on it were skipped wherever it is not installed — CI
    /// included — and passed without running.
    fn create_dummy_client() -> InteractiveClient {
        let (transport, _handle) = nexus_claude::transport::mock::MockTransport::pair();
        InteractiveClient::from_transport(transport)
    }

    /// Helper: create a dummy ActiveSession for testing (no CLI needed).
    fn create_dummy_session(
        is_streaming: bool,
        streaming_text: &str,
        streaming_events_data: Vec<ChatEvent>,
    ) -> (ActiveSession, Arc<Mutex<VecDeque<PendingMessage>>>) {
        let client = create_dummy_client();
        let (tx, _rx) = broadcast::channel(16);
        let pending_messages = Arc::new(Mutex::new(VecDeque::<PendingMessage>::new()));

        let session = ActiveSession {
            anchor: Default::default(),
            events_tx: tx,
            last_activity: Instant::now(),
            cli_session_id: None,
            client: Arc::new(Mutex::new(client)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            memory_manager: None,
            next_seq: Arc::new(AtomicI64::new(1)),
            pending_messages: pending_messages.clone(),
            is_streaming: Arc::new(AtomicBool::new(is_streaming)),
            streaming_text: Arc::new(Mutex::new(streaming_text.to_string())),
            streaming_events: Arc::new(Mutex::new(streaming_events_data)),
            permission_mode: None,
            model: None,
            sdk_control_rx: Arc::new(tokio::sync::Mutex::new(None)),
            stdin_tx: None,
            child_pid: None,
            nats_cancel: CancellationToken::new(),
            interrupt_token: CancellationToken::new(),
            pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            auto_continue: Arc::new(AtomicBool::new(false)),
            auto_continue_count: Arc::new(AtomicU32::new(0)),
            max_auto_continues: 0,
            rfc_accumulator: Arc::new(Mutex::new(
                crate::chat::observation_detector::RfcAccumulator::new(),
            )),
            protocol_run_id: None,
            protocol_state: None,
            reasoning_path_tracker: crate::chat::feedback::ReasoningPathTracker::new(),
            objective_tracking: false,
            objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
            objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
            work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
            oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
            oob_trigger_cap: OOB_TRIGGER_CAP_INTERACTIVE,
            oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
            oob_capped_warned: Arc::new(AtomicBool::new(false)),
            cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
            cancel_tools_cap: CANCEL_TOOLS_CAP,
            cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
            active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
            cli_background_tasks: Arc::new(AtomicUsize::new(0)),
            cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
            cancel_task_cap: CANCEL_TASK_CAP,
            cancel_task_window: Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
        };

        (session, pending_messages)
    }

    #[tokio::test]
    async fn test_get_streaming_snapshot_with_active_session_not_streaming() {
        let (session, _) = create_dummy_session(false, "", vec![]);

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        manager
            .active_sessions
            .write()
            .await
            .insert("test-session".into(), session);

        let (is_streaming, text, events) = manager.get_streaming_snapshot("test-session").await;
        assert!(!is_streaming);
        assert!(text.is_empty());
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn test_get_streaming_snapshot_with_active_streaming_session() {
        let events_data = vec![
            ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "list_plans".into(),
                input: serde_json::json!({}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
            ChatEvent::ToolResult {
                id: "t1".into(),
                result: serde_json::json!({"plans": []}),
                is_error: false,
                parent_tool_use_id: None,
            },
        ];

        let (session, _) = create_dummy_session(true, "Hello world", events_data);

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        manager
            .active_sessions
            .write()
            .await
            .insert("streaming-session".into(), session);

        let (is_streaming, text, events) =
            manager.get_streaming_snapshot("streaming-session").await;
        assert!(is_streaming);
        assert_eq!(text, "Hello world");
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], ChatEvent::ToolUse { tool, .. } if tool == "list_plans"));
        assert!(matches!(&events[1], ChatEvent::ToolResult { id, .. } if id == "t1"));
    }

    // ====================================================================
    // send_message — queuing when is_streaming=true
    // ====================================================================

    #[tokio::test]
    async fn test_send_message_queues_when_streaming() {
        let (session, pending_messages) = create_dummy_session(true, "", vec![]);

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let session_id = Uuid::new_v4().to_string();

        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.clone(), session);

        // Send a message while streaming — should be queued, NOT sent
        let result = manager.send_message(&session_id, "queued message").await;
        assert!(result.is_ok());

        // Verify the message was queued
        let queue = pending_messages.lock().await;
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0], "queued message");
    }

    // ── Held messages: queued WITHOUT interrupting (chat::pending_queue) ──

    /// A streaming session registered in a fresh manager, with a subscriber on
    /// its event channel.
    async fn manager_with_streaming_session() -> (
        ChatManager,
        String,
        Arc<Mutex<VecDeque<PendingMessage>>>,
        broadcast::Receiver<ChatEvent>,
        Arc<AtomicBool>,
    ) {
        let (session, pending_messages) = create_dummy_session(true, "", vec![]);
        let events_rx = session.events_tx.subscribe();
        let interrupt_flag = session.interrupt_flag.clone();
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let session_id = Uuid::new_v4().to_string();
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.clone(), session);
        (
            manager,
            session_id,
            pending_messages,
            events_rx,
            interrupt_flag,
        )
    }

    fn held_texts(event: ChatEvent) -> Vec<String> {
        match event {
            ChatEvent::PendingQueue { messages } => {
                messages.into_iter().map(|m| m.content).collect()
            }
            other => panic!("expected pending_queue, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn queue_user_message_holds_without_interrupting_and_publishes_the_list() {
        let (manager, session_id, pending, mut events, interrupt_flag) =
            manager_with_streaming_session().await;

        let held = manager
            .queue_user_message(&session_id, "after you finish")
            .await
            .unwrap();

        assert!(held, "a streaming session holds the message");
        assert!(
            !interrupt_flag.load(Ordering::SeqCst),
            "holding a message must not interrupt the running turn"
        );
        {
            let queue = pending.lock().await;
            assert_eq!(queue.len(), 1);
            assert!(queue[0].held);
            assert_eq!(queue[0].kind, crate::chat::types::PendingMessageKind::User);
        }
        assert_eq!(
            held_texts(events.try_recv().unwrap()),
            vec!["after you finish"]
        );
    }

    #[tokio::test]
    async fn every_device_of_the_session_sees_the_queue_and_its_changes() {
        // Two devices on one conversation: each holds its own subscription to the
        // session's event channel. A message queued from the phone must show on
        // the laptop, and an action from the laptop must show on the phone.
        let (manager, session_id, _pending, mut phone, _interrupt) =
            manager_with_streaming_session().await;
        let mut laptop = manager
            .active_sessions
            .read()
            .await
            .get(&session_id)
            .unwrap()
            .events_tx
            .subscribe();

        manager
            .queue_user_message(&session_id, "typed on the phone")
            .await
            .unwrap();
        assert_eq!(
            held_texts(phone.try_recv().unwrap()),
            vec!["typed on the phone"]
        );
        assert_eq!(
            held_texts(laptop.try_recv().unwrap()),
            vec!["typed on the phone"]
        );

        let id = manager.pending_queue_snapshot(&session_id).await.unwrap()[0].id;
        manager
            .pending_queue_op(
                &session_id,
                &crate::chat::pending_queue::QueueOp::Edit {
                    id,
                    content: "fixed on the laptop".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            held_texts(phone.try_recv().unwrap()),
            vec!["fixed on the laptop"]
        );
        assert_eq!(
            held_texts(laptop.try_recv().unwrap()),
            vec!["fixed on the laptop"]
        );
    }

    #[tokio::test]
    async fn send_message_mid_stream_still_interrupts_and_is_not_listed_as_held() {
        // The historical path must keep its meaning: it is what "send now" relies on.
        let (manager, session_id, _pending, _events, interrupt_flag) =
            manager_with_streaming_session().await;

        manager
            .send_message(&session_id, "right now")
            .await
            .unwrap();

        assert!(interrupt_flag.load(Ordering::SeqCst));
        assert_eq!(
            manager.pending_queue_snapshot(&session_id).await,
            Some(vec![]),
            "a message already on its way is not a held message"
        );
    }

    #[tokio::test]
    async fn pending_queue_ops_edit_remove_and_prioritize_without_interrupting() {
        let (manager, session_id, _pending, mut events, interrupt_flag) =
            manager_with_streaming_session().await;
        for text in ["a", "b", "c"] {
            manager.queue_user_message(&session_id, text).await.unwrap();
            let _ = events.try_recv();
        }
        let ids: Vec<Uuid> = manager
            .pending_queue_snapshot(&session_id)
            .await
            .unwrap()
            .iter()
            .map(|m| m.id)
            .collect();

        use crate::chat::pending_queue::QueueOp;
        assert!(manager
            .pending_queue_op(&session_id, &QueueOp::Prioritize { id: ids[2] })
            .await
            .unwrap());
        assert_eq!(held_texts(events.try_recv().unwrap()), vec!["c", "a", "b"]);

        manager
            .pending_queue_op(
                &session_id,
                &QueueOp::Edit {
                    id: ids[0],
                    content: "A".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(held_texts(events.try_recv().unwrap()), vec!["c", "A", "b"]);

        manager
            .pending_queue_op(&session_id, &QueueOp::Remove { id: ids[1] })
            .await
            .unwrap();
        assert_eq!(held_texts(events.try_recv().unwrap()), vec!["c", "A"]);

        assert!(
            !interrupt_flag.load(Ordering::SeqCst),
            "none of these may cut the running response short"
        );
    }

    #[tokio::test]
    async fn send_now_puts_the_message_first_and_interrupts() {
        let (manager, session_id, pending, mut events, interrupt_flag) =
            manager_with_streaming_session().await;
        for text in ["a", "b"] {
            manager.queue_user_message(&session_id, text).await.unwrap();
            let _ = events.try_recv();
        }
        let id = manager.pending_queue_snapshot(&session_id).await.unwrap()[1].id;

        manager
            .pending_queue_op(
                &session_id,
                &crate::chat::pending_queue::QueueOp::SendNow { id },
            )
            .await
            .unwrap();

        assert!(interrupt_flag.load(Ordering::SeqCst));
        assert_eq!(pending.lock().await[0].content, "b", "delivered next");
        assert_eq!(
            held_texts(events.try_recv().unwrap()),
            vec!["a"],
            "no longer listed: it is on its way"
        );
    }

    #[tokio::test]
    async fn a_queue_op_on_a_session_nobody_holds_reaches_nothing() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let reached = manager
            .pending_queue_op(
                &Uuid::new_v4().to_string(),
                &crate::chat::pending_queue::QueueOp::Snapshot,
            )
            .await
            .unwrap();
        assert!(!reached, "no local session and no NATS: there is no queue");
    }

    #[tokio::test]
    async fn test_send_message_queues_multiple_when_streaming() {
        let (session, pending_messages) = create_dummy_session(true, "", vec![]);

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let session_id = Uuid::new_v4().to_string();

        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.clone(), session);

        // Queue multiple messages
        manager.send_message(&session_id, "first").await.unwrap();
        manager.send_message(&session_id, "second").await.unwrap();
        manager.send_message(&session_id, "third").await.unwrap();

        let queue = pending_messages.lock().await;
        assert_eq!(queue.len(), 3);
        assert_eq!(queue[0], "first");
        assert_eq!(queue[1], "second");
        assert_eq!(queue[2], "third");
    }

    // ====================================================================
    // streaming_events filtering logic
    // ====================================================================

    #[test]
    fn test_streaming_events_filter_excludes_transient_events() {
        // This tests the filtering logic used in stream_response():
        // StreamDelta, StreamingStatus, and AssistantText should be EXCLUDED
        // from streaming_events buffer (they are handled via streaming_text or
        // sent explicitly in Phase 1.5)

        let events_to_exclude = vec![
            ChatEvent::StreamDelta {
                text: "Hello".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::StreamingStatus { is_streaming: true },
            ChatEvent::StreamingStatus {
                is_streaming: false,
            },
            ChatEvent::AssistantText {
                content: "Hello world".into(),
                parent_tool_use_id: None,
            },
        ];

        for event in &events_to_exclude {
            let should_add = !matches!(
                event,
                ChatEvent::StreamDelta { .. }
                    | ChatEvent::StreamingStatus { .. }
                    | ChatEvent::AssistantText { .. }
            );
            assert!(
                !should_add,
                "Event {:?} should be excluded from streaming_events",
                event.event_type()
            );
        }
    }

    #[test]
    fn test_streaming_events_filter_includes_structured_events() {
        // These events SHOULD be included in streaming_events buffer
        // so mid-stream joiners can reconstruct tool calls

        let events_to_include = vec![
            ChatEvent::ToolUse {
                id: "t1".into(),
                tool: "create_plan".into(),
                input: serde_json::json!({"title": "Plan"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
            ChatEvent::ToolResult {
                id: "t1".into(),
                result: serde_json::json!({"id": "abc"}),
                is_error: false,
                parent_tool_use_id: None,
            },
            ChatEvent::Thinking {
                content: "Let me think...".into(),
                parent_tool_use_id: None,
            },
            ChatEvent::PermissionRequest {
                id: "p1".into(),
                tool: "bash".into(),
                input: serde_json::json!({"command": "ls"}),
                parent_tool_use_id: None,
                category: None,
                canonical: None,
            },
            ChatEvent::Error {
                message: "Something went wrong".into(),
                parent_tool_use_id: None,
                code: None,
                reason: None,
                index: None,
            },
            ChatEvent::Result {
                session_id: "cli-123".into(),
                duration_ms: 5000,
                cost_usd: Some(0.15),
                subtype: "success".into(),
                is_error: false,
                num_turns: None,
                result_text: None,
                cost: None,
                usage: None,
                model: None,
                stop_reason: None,
            },
            ChatEvent::UserMessage {
                content: "Hello".into(),
            },
        ];

        for event in &events_to_include {
            let should_add = !matches!(
                event,
                ChatEvent::StreamDelta { .. }
                    | ChatEvent::StreamingStatus { .. }
                    | ChatEvent::AssistantText { .. }
            );
            assert!(
                should_add,
                "Event {:?} should be included in streaming_events",
                event.event_type()
            );
        }
    }

    // ====================================================================
    // ActiveSession with streaming_events field
    // ====================================================================

    #[tokio::test]
    async fn test_active_session_streaming_events_field() {
        let (session, _) = create_dummy_session(false, "", vec![]);

        // Verify streaming_events starts empty
        assert!(session.streaming_events.lock().await.is_empty());

        // Push an event and verify it's accessible
        session
            .streaming_events
            .lock()
            .await
            .push(ChatEvent::Thinking {
                content: "test".into(),
                parent_tool_use_id: None,
            });
        assert_eq!(session.streaming_events.lock().await.len(), 1);

        // Clear and verify
        session.streaming_events.lock().await.clear();
        assert!(session.streaming_events.lock().await.is_empty());
    }

    // ====================================================================
    // ChatEvent event_type() — covers the StreamingStatus variant
    // ====================================================================

    #[test]
    fn test_chat_event_streaming_status_type() {
        let event = ChatEvent::StreamingStatus { is_streaming: true };
        assert_eq!(event.event_type(), "streaming_status");

        let event = ChatEvent::StreamingStatus {
            is_streaming: false,
        };
        assert_eq!(event.event_type(), "streaming_status");
    }

    #[test]
    fn test_chat_event_user_message_type() {
        let event = ChatEvent::UserMessage {
            content: "Hello".into(),
        };
        assert_eq!(event.event_type(), "user_message");
    }

    #[test]
    fn test_chat_event_streaming_status_serde_roundtrip() {
        let event = ChatEvent::StreamingStatus { is_streaming: true };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"streaming_status\""));
        assert!(json.contains("\"is_streaming\":true"));

        let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            deserialized,
            ChatEvent::StreamingStatus { is_streaming: true }
        ));
    }

    #[test]
    fn test_chat_event_user_message_serde_roundtrip() {
        let event = ChatEvent::UserMessage {
            content: "Hello world".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"user_message\""));

        let deserialized: ChatEvent = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(deserialized, ChatEvent::UserMessage { ref content } if content == "Hello world")
        );
    }

    // ====================================================================
    // with_event_emitter builder
    // ====================================================================

    #[test]
    fn test_with_event_emitter_sets_emitter() {
        use crate::events::{CrudEvent, EventEmitter};

        struct DummyEmitter;
        impl EventEmitter for DummyEmitter {
            fn emit(&self, _event: CrudEvent) {}
        }

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        assert!(manager.event_emitter.is_none());

        let manager = manager.with_event_emitter(Arc::new(DummyEmitter));
        assert!(manager.event_emitter.is_some());
    }

    // ====================================================================
    // backfill_previews_from_meilisearch — early return when no injector
    // ====================================================================

    #[tokio::test]
    async fn test_backfill_previews_no_injector_returns_zero() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        // new_without_memory => context_injector is None => should return Ok(0)
        let result = manager.backfill_previews_from_meilisearch().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    // ====================================================================
    // update_chat_session via mock — title, preview, conversation_id
    // ====================================================================

    #[tokio::test]
    async fn test_update_session_title_and_preview() {
        let state = mock_app_state();
        let session = test_chat_session(Some("test-proj"));
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Update title
        let updated = state
            .neo4j
            .update_chat_session(
                session_id,
                None,
                Some("My new title".into()),
                None,
                None,
                None,
                Some("Preview of the conversation".into()),
            )
            .await
            .unwrap();
        assert!(updated.is_some());
        let s = updated.unwrap();
        assert_eq!(s.title.as_deref(), Some("My new title"));
        assert_eq!(s.preview.as_deref(), Some("Preview of the conversation"));
    }

    #[tokio::test]
    async fn test_update_session_conversation_id() {
        let state = mock_app_state();
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let updated = state
            .neo4j
            .update_chat_session(
                session_id,
                None,
                None,
                None,
                None,
                Some("conv-abc-123".into()),
                None,
            )
            .await
            .unwrap();
        assert!(updated.is_some());
        assert_eq!(
            updated.unwrap().conversation_id.as_deref(),
            Some("conv-abc-123")
        );
    }

    #[tokio::test]
    async fn test_update_session_cost_and_message_count() {
        let state = mock_app_state();
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let updated = state
            .neo4j
            .update_chat_session(session_id, None, None, Some(42), Some(1.50), None, None)
            .await
            .unwrap();
        assert!(updated.is_some());
        let s = updated.unwrap();
        assert_eq!(s.message_count, 42);
        assert!((s.total_cost_usd.unwrap() - 1.50).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_update_session_not_found() {
        let state = mock_app_state();
        let fake_id = Uuid::new_v4();
        let updated = state
            .neo4j
            .update_chat_session(fake_id, None, None, None, None, None, None)
            .await
            .unwrap();
        assert!(updated.is_none());
    }

    #[tokio::test]
    async fn test_update_session_cli_session_id() {
        let state = mock_app_state();
        let session = test_chat_session(None);
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        let updated = state
            .neo4j
            .update_chat_session(
                session_id,
                Some("cli-session-xyz".into()),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(updated.is_some());
        assert_eq!(
            updated.unwrap().cli_session_id.as_deref(),
            Some("cli-session-xyz")
        );
    }

    #[tokio::test]
    async fn test_update_session_partial_preserves_existing() {
        let state = mock_app_state();
        let mut session = test_chat_session(Some("my-proj"));
        session.title = Some("Original title".into());
        session.preview = Some("Original preview".into());
        let session_id = session.id;
        state.neo4j.create_chat_session(&session).await.unwrap();

        // Update only message_count — should preserve title and preview
        let updated = state
            .neo4j
            .update_chat_session(session_id, None, None, Some(10), None, None, None)
            .await
            .unwrap();
        assert!(updated.is_some());
        let s = updated.unwrap();
        assert_eq!(s.title.as_deref(), Some("Original title"));
        assert_eq!(s.preview.as_deref(), Some("Original preview"));
        assert_eq!(s.message_count, 10);
    }

    // ====================================================================
    // backfill_chat_session_previews via mock — returns 0 (mock has no events)
    // ====================================================================

    #[tokio::test]
    async fn test_backfill_session_previews_mock_returns_zero() {
        let state = mock_app_state();
        let result = state.neo4j.backfill_chat_session_previews().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    // ====================================================================
    // Title truncation logic (mirrors send_message_internal behavior)
    // ====================================================================

    #[test]
    fn test_title_generation_short_message() {
        let msg = "Hello, help me with my project";
        // < 80 chars → title == message
        let title = if msg.chars().count() > 80 {
            let truncated: String = msg.chars().take(77).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(title, "Hello, help me with my project");
    }

    #[test]
    fn test_title_generation_long_message() {
        let msg = "a".repeat(100);
        // > 80 chars → truncated to 77 + "..."
        let title = if msg.chars().count() > 80 {
            let truncated: String = msg.chars().take(77).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(title.chars().count(), 80);
        assert!(title.ends_with("..."));
    }

    #[test]
    fn test_preview_generation_short_message() {
        let msg = "Short message";
        let preview = if msg.chars().count() > 200 {
            let truncated: String = msg.chars().take(197).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(preview, "Short message");
    }

    #[test]
    fn test_preview_generation_long_message() {
        let msg = "b".repeat(300);
        let preview = if msg.chars().count() > 200 {
            let truncated: String = msg.chars().take(197).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(preview.chars().count(), 200);
        assert!(preview.ends_with("..."));
    }

    #[test]
    fn test_title_generation_utf8_multibyte() {
        // 90 chars with accented characters
        let msg: String = "é".repeat(90);
        let title = if msg.chars().count() > 80 {
            let truncated: String = msg.chars().take(77).collect();
            format!("{}...", truncated.trim_end())
        } else {
            msg.to_string()
        };
        assert_eq!(title.chars().count(), 80);
        assert!(title.ends_with("..."));
    }

    // ========================================================================
    // NATS RPC routing tests
    // ========================================================================

    #[tokio::test]
    async fn test_try_remote_send_no_nats() {
        // ChatManager without NATS → try_remote_send should return Ok(false) immediately
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // No NATS configured → should return false (fallback needed)
        assert!(manager.nats.is_none());
        let result = manager
            .try_remote_send("any-session-id", "hello", "user_message")
            .await;
        assert!(result.is_ok());
        assert!(!result.unwrap(), "Should return false without NATS");
    }

    #[tokio::test]
    async fn test_try_remote_send_all_message_types_no_nats() {
        // Verify all message types return Ok(false) without NATS
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        for msg_type in &["user_message", "permission_response", "input_response"] {
            let result = manager
                .try_remote_send("session-123", "test", msg_type)
                .await
                .unwrap();
            assert!(!result, "Should return false for {} without NATS", msg_type);
        }
    }

    #[tokio::test]
    async fn test_interrupt_no_local_session_no_nats() {
        // interrupt() should succeed silently when session is not local and no NATS
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // No session active, no NATS → should return Ok (no error)
        let result = manager.interrupt("nonexistent-session").await;
        assert!(
            result.is_ok(),
            "interrupt should not error when session is not local"
        );
    }

    /// `interrupt()` returning `Ok(())` for a session it never found is the
    /// reason a lost Stop was invisible: the UI could not tell "turn ended"
    /// from "nothing happened". `interrupt_scoped` says which it was.
    #[tokio::test]
    async fn test_interrupt_scoped_reports_a_no_op_as_undelivered() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let outcome = manager
            .interrupt_scoped("nonexistent-session", true)
            .await
            .expect("interrupt_scoped must not error on an unknown session");

        assert!(
            !outcome.delivered,
            "nothing was interrupted — delivered must be false"
        );
        assert_eq!(
            outcome.routed, "none",
            "no local session and no NATS means the interrupt went nowhere"
        );
        assert!(outcome.killed_pids.is_empty());
    }

    /// The narrow scope still ends the turn: flag set, token cancelled.
    #[tokio::test]
    async fn test_interrupt_scoped_still_ends_the_turn_without_tools() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(true);
        let flag = session.interrupt_flag.clone();
        let token = session.interrupt_token.clone();

        let session_id = "test-session-scoped-turn";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        let outcome = manager.interrupt_scoped(session_id, false).await.unwrap();

        assert!(outcome.delivered, "a local session was interrupted");
        assert_eq!(outcome.routed, "local");
        assert!(
            flag.load(Ordering::SeqCst),
            "the turn must still be interrupted with kill_tools = false"
        );
        assert!(
            token.is_cancelled(),
            "the stream token must still be cancelled"
        );
        assert!(outcome.killed_pids.is_empty());
    }

    /// The point of the narrow scope, proved on real processes rather than
    /// on an empty `killed_pids`: a subprocess the agent launched survives
    /// an interrupt scoped to the turn, and dies under the wide one.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_interrupt_scoped_spares_then_reaps_a_running_tool() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (mut session, _handle) = mock_active_session(true);

        // Stand-in for a tool the agent is running: a shell that stays alive
        // as the parent of a long sleep, so the sleep is a *descendant* of
        // the pretend-CLI, exactly like a real Bash tool child. The trailing
        // `:` stops `sh` from exec'ing sleep in its own process.
        let mut shell = std::process::Command::new("sh")
            .arg("-c")
            .arg("sleep 30; :")
            .spawn()
            .expect("failed to spawn the stand-in tool");
        session.child_pid = Some(shell.id());

        // Wait for the grandchild to actually exist — without it the test
        // would pass for the wrong reason.
        let mut descendants = Vec::new();
        for _ in 0..40 {
            descendants = ChatManager::get_descendant_pids(shell.id());
            if !descendants.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            !descendants.is_empty(),
            "no descendant ever appeared under the stand-in tool — the test cannot conclude"
        );

        let session_id = "test-session-scoped-tools";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        // Narrow scope: the turn ends, the tool keeps running.
        let outcome = manager.interrupt_scoped(session_id, false).await.unwrap();
        assert!(outcome.delivered);
        assert!(
            outcome.killed_pids.is_empty(),
            "kill_tools = false must signal nobody"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !ChatManager::get_descendant_pids(shell.id()).is_empty(),
            "the tool subprocess must survive an interrupt scoped to the turn"
        );

        // Wide scope: same session, the tool is reaped.
        let outcome = manager.interrupt_scoped(session_id, true).await.unwrap();
        assert!(
            !outcome.killed_pids.is_empty(),
            "kill_tools = true must SIGINT the descendant"
        );

        let _ = shell.kill();
        let _ = shell.wait();
    }

    #[tokio::test]
    async fn test_spawn_nats_rpc_listener_noop_without_nats() {
        // Verifies that spawn_nats_rpc_listener is a no-op without NATS (no panic, no error)
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // This should be a complete no-op — no NATS means early return
        manager.spawn_nats_rpc_listener(
            "test-session",
            manager.active_sessions.clone(),
            CancellationToken::new(),
        );
        // If we get here without panic, the test passes
    }

    #[tokio::test]
    async fn test_routing_decision_local_active_session() {
        // When a session is active locally, is_session_active returns true
        // → the routing logic should choose send_message (local path)
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // No active sessions → is_session_active should return false
        assert!(!manager.is_session_active("some-session").await);

        // try_remote_send without NATS → false
        let remote = manager
            .try_remote_send("some-session", "hello", "user_message")
            .await
            .unwrap();
        assert!(!remote);

        // This confirms the routing: not local + not remote = resume_session fallback
    }

    // ====================================================================
    // YAML persistence tests
    // ====================================================================

    #[test]
    fn test_persist_permission_to_yaml_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // Write an existing config.yaml with other sections
        std::fs::write(
            &yaml_path,
            "server:\n  port: 9090\nneo4j:\n  uri: bolt://db:7687\nchat:\n  default_model: claude-sonnet\n",
        )
        .unwrap();

        let perm = super::super::config::PermissionConfig {
            mode: "default".into(),
            allowed_tools: vec!["Bash(git *)".into(), "Read".into()],
            disallowed_tools: vec!["Bash(rm -rf *)".into()],
        };

        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // Re-read and verify
        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();

        // Other sections preserved
        assert_eq!(
            doc["server"]["port"].as_u64().unwrap(),
            9090,
            "server.port should be preserved"
        );
        assert_eq!(
            doc["neo4j"]["uri"].as_str().unwrap(),
            "bolt://db:7687",
            "neo4j.uri should be preserved"
        );
        // Existing chat fields preserved
        assert_eq!(
            doc["chat"]["default_model"].as_str().unwrap(),
            "claude-sonnet",
            "chat.default_model should be preserved"
        );
        // Permissions written correctly
        assert_eq!(
            doc["chat"]["permissions"]["mode"].as_str().unwrap(),
            "default"
        );
        let allowed = doc["chat"]["permissions"]["allowed_tools"]
            .as_sequence()
            .unwrap();
        assert_eq!(allowed.len(), 2);
        assert_eq!(allowed[0].as_str().unwrap(), "Bash(git *)");
        assert_eq!(allowed[1].as_str().unwrap(), "Read");
        let disallowed = doc["chat"]["permissions"]["disallowed_tools"]
            .as_sequence()
            .unwrap();
        assert_eq!(disallowed.len(), 1);
        assert_eq!(disallowed[0].as_str().unwrap(), "Bash(rm -rf *)");
    }

    #[test]
    fn test_persist_permission_to_yaml_no_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // File does not exist yet
        assert!(!yaml_path.exists());

        let perm = super::super::config::PermissionConfig {
            mode: "acceptEdits".into(),
            allowed_tools: vec![],
            disallowed_tools: vec!["Bash(sudo *)".into()],
        };

        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // File should now exist
        assert!(yaml_path.exists());
        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();

        assert_eq!(
            doc["chat"]["permissions"]["mode"].as_str().unwrap(),
            "acceptEdits"
        );
        // Empty allowed_tools should be present as empty sequence
        assert!(doc["chat"]["permissions"]["allowed_tools"]
            .as_sequence()
            .unwrap()
            .is_empty());
        assert_eq!(
            doc["chat"]["permissions"]["disallowed_tools"]
                .as_sequence()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_persist_permission_roundtrip_with_config_load() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // Start with a config that has no permissions
        std::fs::write(
            &yaml_path,
            "server:\n  port: 8080\nchat:\n  default_model: claude-opus-4-6\n",
        )
        .unwrap();

        // Persist permissions
        let perm = super::super::config::PermissionConfig {
            mode: "plan".into(),
            allowed_tools: vec!["mcp__project-orchestrator__*".into()],
            disallowed_tools: vec![],
        };
        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // Reload via Config::from_yaml_and_env
        // Clear env vars that would override
        std::env::remove_var("CHAT_PERMISSION_MODE");
        std::env::remove_var("CHAT_ALLOWED_TOOLS");
        std::env::remove_var("CHAT_DISALLOWED_TOOLS");

        let config = crate::Config::from_yaml_and_env(Some(&yaml_path)).unwrap();
        let loaded_perm = config
            .chat_permissions
            .expect("chat_permissions should be Some");
        assert_eq!(loaded_perm.mode, "plan");
        assert_eq!(
            loaded_perm.allowed_tools,
            vec!["mcp__project-orchestrator__*"]
        );
        assert!(loaded_perm.disallowed_tools.is_empty());

        // config_yaml_path should be set
        assert_eq!(
            config.config_yaml_path.as_deref(),
            Some(yaml_path.as_path())
        );
    }

    #[test]
    fn test_persist_permission_atomic_no_tmp_leftover() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");
        let tmp_path = dir.path().join("config.yaml.tmp");

        std::fs::write(&yaml_path, "chat: {}\n").unwrap();

        let perm = super::super::config::PermissionConfig::default();
        ChatManager::persist_permission_to_yaml(&yaml_path, &perm).unwrap();

        // .tmp file should not exist after successful rename
        assert!(
            !tmp_path.exists(),
            "Temporary file should be cleaned up after atomic rename"
        );
        // Original file should still exist with updated content
        assert!(yaml_path.exists());
    }

    #[tokio::test]
    async fn test_update_permission_config_persists_to_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");
        std::fs::write(&yaml_path, "chat:\n  default_model: test\n").unwrap();

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
            .with_config_yaml_path(yaml_path.clone());

        let new_perm = super::super::config::PermissionConfig {
            mode: "acceptEdits".into(),
            allowed_tools: vec!["Read".into()],
            disallowed_tools: vec![],
        };

        let result = manager.update_permission_config(new_perm).await.unwrap();
        assert_eq!(result.mode, "acceptEdits");

        // Verify it was persisted to disk
        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();
        assert_eq!(
            doc["chat"]["permissions"]["mode"].as_str().unwrap(),
            "acceptEdits"
        );
        // Other chat fields preserved
        assert_eq!(doc["chat"]["default_model"].as_str().unwrap(), "test");
    }

    // ── helpers for ActiveSession tests (no Claude CLI required) ────────

    /// Create an ActiveSession backed by a MockTransport.
    /// Does NOT require the Claude CLI to be installed.
    fn mock_active_session(
        is_streaming: bool,
    ) -> (
        ActiveSession,
        nexus_claude::transport::mock::MockTransportHandle,
    ) {
        let (transport, handle) = nexus_claude::transport::mock::MockTransport::pair();
        let client = InteractiveClient::from_transport(transport);
        let (tx, _rx) = broadcast::channel(16);

        let session = ActiveSession {
            anchor: Default::default(),
            events_tx: tx,
            last_activity: Instant::now(),
            cli_session_id: None,
            client: Arc::new(Mutex::new(client)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            memory_manager: None,
            next_seq: Arc::new(AtomicI64::new(1)),
            pending_messages: Arc::new(Mutex::new(VecDeque::<PendingMessage>::new())),
            is_streaming: Arc::new(AtomicBool::new(is_streaming)),
            streaming_text: Arc::new(Mutex::new(String::new())),
            streaming_events: Arc::new(Mutex::new(Vec::new())),
            permission_mode: None,
            model: None,
            sdk_control_rx: Arc::new(tokio::sync::Mutex::new(None)),
            stdin_tx: None,
            child_pid: None,
            nats_cancel: CancellationToken::new(),
            interrupt_token: CancellationToken::new(),
            pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            auto_continue: Arc::new(AtomicBool::new(false)),
            auto_continue_count: Arc::new(AtomicU32::new(0)),
            max_auto_continues: 0,
            rfc_accumulator: Arc::new(Mutex::new(
                crate::chat::observation_detector::RfcAccumulator::new(),
            )),
            protocol_run_id: None,
            protocol_state: None,
            reasoning_path_tracker: crate::chat::feedback::ReasoningPathTracker::new(),
            objective_tracking: false,
            objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
            objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
            work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
            oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
            oob_trigger_cap: OOB_TRIGGER_CAP_INTERACTIVE,
            oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
            oob_capped_warned: Arc::new(AtomicBool::new(false)),
            cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
            cancel_tools_cap: CANCEL_TOOLS_CAP,
            cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
            active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
            cli_background_tasks: Arc::new(AtomicUsize::new(0)),
            cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
            cancel_task_cap: CANCEL_TASK_CAP,
            cancel_task_window: Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
        };

        (session, handle)
    }

    // ── parse_permission_control_msg tests ──────────────────────────────

    #[test]
    fn test_parse_permission_control_msg_can_use_tool() {
        let msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req_abc",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "ls -la"}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some(), "should parse can_use_tool");

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id,
                tool,
                input,
                parent_tool_use_id,
                ..
            } => {
                assert_eq!(id, "req_abc");
                assert_eq!(tool, "Bash");
                assert_eq!(input["command"], "ls -la");
                assert!(parent_tool_use_id.is_none());
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_with_parent_tool_use_id() {
        let msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req_xyz",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "Read",
                "input": {"file_path": "/tmp/test.rs"}
            }
        });

        let event = parse_permission_control_msg(&msg, Some("toolu_parent_123".to_string()));
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id,
                tool,
                input,
                parent_tool_use_id,
                ..
            } => {
                assert_eq!(id, "req_xyz");
                assert_eq!(tool, "Read");
                assert_eq!(input["file_path"], "/tmp/test.rs");
                assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_parent_123"));
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_ignores_non_permission() {
        // subtype != "can_use_tool"
        let msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req_other",
            "request": {
                "subtype": "server_info",
                "data": {"model": "claude-sonnet-4-6"}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_none(), "non-permission should return None");
    }

    #[test]
    fn test_parse_permission_control_msg_flat_format() {
        // No nested "request" — all fields at top level
        let msg = serde_json::json!({
            "subtype": "can_use_tool",
            "requestId": "flat_001",
            "tool_name": "Write",
            "input": {"file_path": "/tmp/out.txt", "content": "hello"}
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some(), "flat format should parse");

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id, tool, input, ..
            } => {
                assert_eq!(id, "flat_001");
                assert_eq!(tool, "Write");
                assert_eq!(input["content"], "hello");
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_missing_fields_defaults() {
        // Minimal message with just the subtype
        let msg = serde_json::json!({
            "request": {
                "subtype": "can_use_tool"
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::PermissionRequest {
                id, tool, input, ..
            } => {
                assert_eq!(id, "", "missing request_id defaults to empty");
                assert_eq!(tool, "unknown", "missing tool defaults to 'unknown'");
                assert_eq!(
                    input,
                    serde_json::json!({}),
                    "missing input defaults to empty object"
                );
            }
            other => panic!("expected PermissionRequest, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_camel_vs_snake_tool_name() {
        // toolName (camelCase) should be preferred
        let msg_camel = serde_json::json!({
            "request_id": "r1",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "CamelTool",
                "input": {}
            }
        });
        let evt = parse_permission_control_msg(&msg_camel, None).unwrap();
        if let ChatEvent::PermissionRequest { tool, .. } = evt {
            assert_eq!(tool, "CamelTool");
        }

        // tool_name (snake_case) fallback
        let msg_snake = serde_json::json!({
            "request_id": "r2",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "SnakeTool",
                "input": {}
            }
        });
        let evt = parse_permission_control_msg(&msg_snake, None).unwrap();
        if let ChatEvent::PermissionRequest { tool, .. } = evt {
            assert_eq!(tool, "SnakeTool");
        }
    }

    #[test]
    fn test_parse_permission_control_msg_request_id_variants() {
        // requestId (camelCase)
        let msg = serde_json::json!({
            "requestId": "camel_id_001",
            "request": { "subtype": "can_use_tool", "tool_name": "T", "input": {} }
        });
        if let Some(ChatEvent::PermissionRequest { id, .. }) =
            parse_permission_control_msg(&msg, None)
        {
            assert_eq!(id, "camel_id_001");
        }

        // request_id (snake_case)
        let msg = serde_json::json!({
            "request_id": "snake_id_002",
            "request": { "subtype": "can_use_tool", "tool_name": "T", "input": {} }
        });
        if let Some(ChatEvent::PermissionRequest { id, .. }) =
            parse_permission_control_msg(&msg, None)
        {
            assert_eq!(id, "snake_id_002");
        }
    }

    // ── parse_permission_control_msg: AskUserQuestion tests ─────────────

    #[test]
    fn test_parse_permission_control_msg_ask_user_question_returns_dedicated_event() {
        let msg = serde_json::json!({
            "type": "control_request",
            "requestId": "req_ask_001",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "AskUserQuestion",
                "toolUseId": "toolu_ask_123",
                "input": {
                    "questions": [{
                        "question": "Which framework?",
                        "header": "Framework",
                        "options": [
                            {"label": "React", "description": "Popular UI library"},
                            {"label": "Vue", "description": "Progressive framework"}
                        ],
                        "multiSelect": false
                    }]
                }
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some(), "AskUserQuestion should parse");

        match event.unwrap() {
            ChatEvent::AskUserQuestion {
                id,
                tool_call_id,
                questions,
                input,
                parent_tool_use_id,
                ..
            } => {
                assert_eq!(id, "req_ask_001");
                assert_eq!(tool_call_id, "toolu_ask_123");
                assert!(questions.is_array(), "questions should be an array");
                assert_eq!(questions.as_array().unwrap().len(), 1);
                assert_eq!(questions[0]["header"], "Framework");
                assert!(input.get("questions").is_some());
                assert!(parent_tool_use_id.is_none());
            }
            other => panic!("expected AskUserQuestion, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_ask_user_question_with_parent() {
        let msg = serde_json::json!({
            "requestId": "req_ask_002",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "AskUserQuestion",
                "toolUseId": "toolu_ask_456",
                "input": {
                    "questions": [{"question": "Choose:", "header": "Q", "options": [{"label": "A"}, {"label": "B"}], "multiSelect": false}]
                }
            }
        });

        let event = parse_permission_control_msg(&msg, Some("toolu_parent_789".to_string()));
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::AskUserQuestion {
                id,
                parent_tool_use_id,
                ..
            } => {
                assert_eq!(id, "req_ask_002");
                assert_eq!(parent_tool_use_id.as_deref(), Some("toolu_parent_789"));
            }
            other => panic!("expected AskUserQuestion, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_bash_still_returns_permission_request() {
        // Regression: Bash (and all other tools) must still return PermissionRequest
        let msg = serde_json::json!({
            "requestId": "req_bash_001",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "Bash",
                "input": {"command": "echo hello"}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::PermissionRequest { tool, .. } => {
                assert_eq!(tool, "Bash");
            }
            other => panic!("expected PermissionRequest for Bash, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_permission_control_msg_ask_user_question_missing_questions() {
        // If input doesn't have "questions", should default to empty array
        let msg = serde_json::json!({
            "requestId": "req_ask_003",
            "request": {
                "subtype": "can_use_tool",
                "toolName": "AskUserQuestion",
                "input": {}
            }
        });

        let event = parse_permission_control_msg(&msg, None);
        assert!(event.is_some());

        match event.unwrap() {
            ChatEvent::AskUserQuestion {
                questions,
                tool_call_id,
                ..
            } => {
                assert!(questions.is_array());
                assert_eq!(questions.as_array().unwrap().len(), 0);
                assert_eq!(tool_call_id, "", "missing toolUseId defaults to empty");
            }
            other => panic!("expected AskUserQuestion, got: {:?}", other),
        }
    }

    // ── send_permission_response tests ──────────────────────────────────

    #[tokio::test]
    async fn test_send_permission_response_no_session_errors() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager
            .send_permission_response("nonexistent-session-id", "req-001", true)
            .await;
        assert!(result.is_err(), "should fail for non-existent session");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found"),
            "error should mention 'not found': {err_msg}"
        );
    }

    // ── interrupt tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_interrupt_no_session_errors() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let result = manager.interrupt("nonexistent-session-id").await;
        // The manager falls back to NATS publish when no local session exists.
        // Without NATS configured, it returns an error or succeeds silently.
        // Either way, it should not panic.
        assert!(
            result.is_ok() || result.is_err(),
            "interrupt on non-existent session should not panic"
        );
    }

    // ── interrupt flag atomicity & latency tests ────────────────────────

    /// Verify that `interrupt()` sets the atomic flag immediately (< 1ms)
    /// without acquiring the InteractiveClient Mutex.
    #[tokio::test]
    async fn test_interrupt_sets_flag_atomically() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(false);
        let flag = session.interrupt_flag.clone();

        // Insert session into manager
        let session_id = "test-session-atomic";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        assert!(!flag.load(Ordering::SeqCst), "flag should start false");

        // Measure interrupt latency
        let start = Instant::now();
        manager.interrupt(session_id).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            flag.load(Ordering::SeqCst),
            "flag should be true after interrupt"
        );
        assert!(
            elapsed < Duration::from_millis(1),
            "interrupt() took {:?}, expected < 1ms (atomic flag only)",
            elapsed
        );
    }

    /// Verify that `interrupt()` works even when the client Mutex is held
    /// (simulating an active stream_response).
    #[tokio::test]
    async fn test_interrupt_does_not_wait_for_client_lock() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(true);
        let flag = session.interrupt_flag.clone();
        let client_lock = session.client.clone();

        let session_id = "test-session-lock";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        // Hold the client Mutex (simulating stream_response owning it)
        let _guard = client_lock.lock().await;

        // interrupt() should still complete instantly because it only
        // touches the AtomicBool, not the Mutex
        let start = Instant::now();
        manager.interrupt(session_id).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            flag.load(Ordering::SeqCst),
            "flag should be set even while Mutex is held"
        );
        assert!(
            elapsed < Duration::from_millis(1),
            "interrupt() took {:?} even though client Mutex is held — should be < 1ms",
            elapsed
        );
    }

    /// Verify that interrupt works when streaming is active
    /// (is_streaming flag set to true).
    #[tokio::test]
    async fn test_interrupt_during_active_stream() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(true);
        let interrupt_flag = session.interrupt_flag.clone();
        let is_streaming = session.is_streaming.clone();

        let session_id = "test-session-streaming";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        assert!(
            is_streaming.load(Ordering::SeqCst),
            "session should be streaming"
        );
        assert!(
            !interrupt_flag.load(Ordering::SeqCst),
            "interrupt should start false"
        );

        manager.interrupt(session_id).await.unwrap();

        assert!(
            interrupt_flag.load(Ordering::SeqCst),
            "interrupt flag should be set during active stream"
        );
    }

    /// Benchmark: interrupt latency over 100 sessions, each must be < 1ms.
    #[tokio::test]
    async fn test_interrupt_latency_benchmark() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        let iterations = 100u32;
        let mut flags = Vec::with_capacity(iterations as usize);

        // Create N sessions
        for i in 0..iterations {
            let (session, _handle) = mock_active_session(true);
            let flag = session.interrupt_flag.clone();
            flags.push(flag);
            manager
                .active_sessions
                .write()
                .await
                .insert(format!("bench-{i}"), session);
        }

        // Interrupt all sessions and measure total time
        let start = Instant::now();
        for i in 0..iterations {
            manager.interrupt(&format!("bench-{i}")).await.unwrap();
        }
        let total = start.elapsed();
        let avg = total / iterations;

        // Verify all flags were set
        for (i, flag) in flags.iter().enumerate() {
            assert!(
                flag.load(Ordering::SeqCst),
                "flag for session bench-{i} should be set"
            );
        }

        assert!(
            avg < Duration::from_millis(1),
            "Average interrupt latency {:?} exceeds 1ms (total {:?} for {iterations})",
            avg,
            total
        );
    }

    /// Verify that double-interrupt doesn't panic and is idempotent.
    #[tokio::test]
    async fn test_double_interrupt_is_idempotent() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(false);
        let flag = session.interrupt_flag.clone();

        let session_id = "test-double-interrupt";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        // First interrupt
        manager.interrupt(session_id).await.unwrap();
        assert!(flag.load(Ordering::SeqCst));

        // Second interrupt — should not panic or error
        let result = manager.interrupt(session_id).await;
        assert!(result.is_ok(), "second interrupt should succeed");
        assert!(flag.load(Ordering::SeqCst), "flag should remain true");
    }

    // ── attention_changed relay ─────────────────────────────────────────

    fn attention_hybrid() -> (
        Arc<crate::events::HybridEmitter>,
        tokio::sync::broadcast::Receiver<crate::events::CrudEvent>,
    ) {
        let h = Arc::new(crate::events::HybridEmitter::new(Arc::new(
            crate::events::EventBus::default(),
        )));
        let rx = h.subscribe();
        (h, rx)
    }

    #[tokio::test]
    async fn test_permission_request_emits_attention_changed_without_the_command() {
        let (h, mut rx) = attention_hybrid();
        let emitter: Option<Arc<dyn crate::events::EventEmitter>> = Some(h);
        let msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req-1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "rm -rf /secret-command-text"},
                "tool_use_id": "tu-1"
            }
        });
        let event = parse_permission_control_msg(&msg, None).expect("permission request");
        assert!(matches!(event, ChatEvent::PermissionRequest { .. }));
        notify_attention_for_chat_event(&emitter, "sess-1", &event);
        tokio::time::sleep(Duration::from_millis(400)).await;
        let ev = rx.try_recv().expect("attention_changed on the bus");
        assert_eq!(ev.entity_type, crate::events::EntityType::AttentionChanged);
        assert_eq!(ev.payload["session_id"], "sess-1");
        assert_eq!(
            ev.payload["reasons"],
            serde_json::json!(["permission_request"])
        );
        assert!(
            !serde_json::to_string(&ev)
                .unwrap()
                .contains("secret-command-text"),
            "the command text must not transit on the general bus"
        );
    }

    #[tokio::test]
    async fn test_permission_decision_emits_attention_changed() {
        let state = mock_app_state();
        let (h, mut rx) = attention_hybrid();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config())
            .with_event_emitter(h);
        let (mut session, _handle) = mock_active_session(false);
        let (stdin_tx, _stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
        session.stdin_tx = Some(stdin_tx);
        session
            .pending_permission_inputs
            .lock()
            .await
            .insert("req-d".to_string(), serde_json::json!({}));
        manager
            .active_sessions
            .write()
            .await
            .insert("sess-d".to_string(), session);
        manager
            .send_permission_response("sess-d", "req-d", true)
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
        let ev = rx.try_recv().expect("attention_changed on the bus");
        assert_eq!(ev.payload["session_id"], "sess-d");
        assert_eq!(
            ev.payload["reasons"],
            serde_json::json!(["permission_decision"])
        );
    }

    // ── send_permission_response with mock transport ────────────────────

    #[tokio::test]
    async fn test_send_permission_response_allow_via_mock() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (mut session, _handle) = mock_active_session(false);

        // Create a stdin channel to capture the control response
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
        session.stdin_tx = Some(stdin_tx);

        // Pre-populate pending_permission_inputs with the original tool input
        let original_input = serde_json::json!({
            "command": "echo \"hello\"",
            "description": "Print hello"
        });
        session
            .pending_permission_inputs
            .lock()
            .await
            .insert("req-allow-001".to_string(), original_input.clone());

        let session_id = "test-perm-allow";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        manager
            .send_permission_response(session_id, "req-allow-001", true)
            .await
            .unwrap();

        // Verify the control response was sent through stdin_tx
        let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should be open");

        let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();

        // Verify full envelope: {"type": "control_response", "response": {"subtype": "success", "request_id": ..., "response": {...}}}
        assert_eq!(sent["type"], "control_response");
        let outer_response = &sent["response"];
        assert_eq!(outer_response["subtype"], "success");
        assert_eq!(outer_response["request_id"], "req-allow-001");
        let inner_response = &outer_response["response"];
        assert_eq!(
            inner_response["behavior"], "allow",
            "inner response should contain behavior: allow"
        );
        assert_eq!(
            inner_response["updatedInput"], original_input,
            "updatedInput should contain the original tool input"
        );
    }

    #[tokio::test]
    async fn test_send_permission_response_deny_via_mock() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (mut session, _handle) = mock_active_session(false);

        // Create a stdin channel to capture the control response
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
        session.stdin_tx = Some(stdin_tx);

        let session_id = "test-perm-deny";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        manager
            .send_permission_response(session_id, "req-deny-002", false)
            .await
            .unwrap();

        let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should be open");

        let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();
        assert_eq!(sent["type"], "control_response");
        let outer_response = &sent["response"];
        assert_eq!(outer_response["subtype"], "success");
        assert_eq!(outer_response["request_id"], "req-deny-002");
        let inner_response = &outer_response["response"];
        assert_eq!(
            inner_response["behavior"], "deny",
            "inner response should contain behavior: deny"
        );
        assert!(
            inner_response["message"]
                .as_str()
                .unwrap()
                .contains("denied"),
            "inner response should contain a denial message"
        );
    }

    #[tokio::test]
    async fn test_set_session_model_sends_control_request_via_stdin() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (mut session, _handle) = mock_active_session(false);

        // Create a stdin channel to capture the control request
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<String>(16);
        session.stdin_tx = Some(stdin_tx);

        // Subscribe to events BEFORE inserting the session
        let mut events_rx = session.events_tx.subscribe();

        let session_id = "test-model-change";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        // Change model
        let broadcast = manager
            .set_session_model(session_id, "claude-opus-4-20250514")
            .await
            .unwrap();
        assert!(
            broadcast,
            "a live session broadcasts model_changed itself: the caller must not confirm again"
        );

        // Verify the control request was sent through stdin_tx
        let sent_json = tokio::time::timeout(Duration::from_millis(100), stdin_rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should be open");

        let sent: serde_json::Value = serde_json::from_str(&sent_json).unwrap();
        assert_eq!(sent["type"], "control_request");
        assert!(
            sent["request_id"].as_str().is_some(),
            "should have a request_id"
        );
        let request = &sent["request"];
        assert_eq!(request["subtype"], "set_model");
        assert_eq!(request["model"], "claude-opus-4-20250514");

        // Verify ModelChanged event was broadcast — exactly once
        let event = tokio::time::timeout(Duration::from_millis(100), events_rx.recv())
            .await
            .expect("should receive event within timeout")
            .expect("channel should be open");
        assert!(
            events_rx.try_recv().is_err(),
            "one change, one model_changed event: two would show the confirmation twice"
        );

        assert_eq!(event.event_type(), "model_changed");
        if let ChatEvent::ModelChanged { model } = event {
            assert_eq!(model, "claude-opus-4-20250514");
        } else {
            panic!("Expected ModelChanged event, got {:?}", event.event_type());
        }

        // Verify the in-memory session was updated
        let sessions = manager.active_sessions.read().await;
        let session = sessions.get(session_id).unwrap();
        assert_eq!(session.model.as_deref(), Some("claude-opus-4-20250514"));
    }

    #[tokio::test]
    async fn test_set_session_model_without_stdin_persists_and_broadcasts() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        let (session, _handle) = mock_active_session(false);
        // stdin_tx is None — no live CLI to receive a control_request. This must NOT error:
        // the model is updated in-memory + persisted so it applies on the next spawn.

        let mut events_rx = session.events_tx.subscribe();

        let session_id = "test-model-no-stdin";
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);

        let broadcast = manager
            .set_session_model(session_id, "claude-opus-4-20250514")
            .await
            .expect("model change without live stdin should succeed (persist + broadcast)");
        assert!(broadcast);

        // In-memory model is updated so a respawn from this session uses the new model.
        {
            let sessions = manager.active_sessions.read().await;
            assert_eq!(
                sessions.get(session_id).unwrap().model.as_deref(),
                Some("claude-opus-4-20250514")
            );
        }

        // ModelChanged is still broadcast so the frontend reflects the choice.
        let event = tokio::time::timeout(Duration::from_millis(100), events_rx.recv())
            .await
            .expect("should receive event within timeout")
            .expect("channel should be open");
        assert_eq!(event.event_type(), "model_changed");
    }

    #[tokio::test]
    async fn test_set_session_model_dormant_session_persists() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // A session absent from active_sessions is dormant (idle-cleaned). set_model must
        // still succeed by persisting to Neo4j so the next resume launches with the new
        // model — rather than rejecting the change and respawning on the create-time model.
        let session_id = uuid::Uuid::new_v4().to_string();
        let result = manager
            .set_session_model(&session_id, "claude-opus-4-20250514")
            .await;

        // `false`: nothing was broadcast (no subscribers), so the WebSocket
        // handler confirms to the asker directly.
        assert!(!result.expect("dormant session model change should persist without error"));
    }

    // ====================================================================
    // CompactionNotifier
    // ====================================================================

    #[tokio::test]
    async fn test_compaction_notifier_emits_event_on_pre_compact() {
        use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

        let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx, None, "test-session".to_string());

        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: "test-session".into(),
            transcript_path: "/tmp/transcript".into(),
            cwd: "/test".into(),
            permission_mode: None,
            trigger: "auto".into(),
            custom_instructions: None,
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, None, &context).await;
        assert!(result.is_ok());

        // Verify event was broadcast
        let event = rx.try_recv().unwrap();
        assert!(matches!(
            event,
            ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"
        ));
    }

    #[tokio::test]
    async fn test_compaction_notifier_manual_trigger() {
        use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

        let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx, None, "sess-2".to_string());

        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: "sess-2".into(),
            transcript_path: "/tmp/t".into(),
            cwd: "/".into(),
            permission_mode: Some("default".into()),
            trigger: "manual".into(),
            custom_instructions: Some("Keep API context".into()),
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, None, &context).await.unwrap();

        // Must always return continue=true
        if let nexus_claude::HookJSONOutput::Sync(sync) = result {
            assert_eq!(sync.continue_, Some(true));
        } else {
            panic!("Expected Sync output");
        }

        let event = rx.try_recv().unwrap();
        assert!(matches!(
            event,
            ChatEvent::CompactionStarted { ref trigger } if trigger == "manual"
        ));
    }

    #[tokio::test]
    async fn test_compaction_notifier_noop_on_other_hooks() {
        use nexus_claude::{HookCallback, HookContext, HookInput, PreToolUseHookInput};

        let (tx, mut rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx, None, "test-session".to_string());

        let input = HookInput::PreToolUse(PreToolUseHookInput {
            // Main thread, not a sub-agent — the CLI omits both fields there.
            agent_id: None,
            agent_type: None,
            session_id: "s".into(),
            transcript_path: "/t".into(),
            cwd: "/c".into(),
            permission_mode: None,
            tool_name: "Bash".into(),
            tool_input: serde_json::json!({"command": "ls"}),
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, Some("tu1"), &context).await;
        assert!(result.is_ok());

        // No event should be broadcast for non-PreCompact hooks
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_compaction_notifier_returns_custom_instructions() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::test_helpers::{test_plan, test_step, test_task};
        use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

        // 1. Setup: create a plan + task + steps in MockGraphStore
        let mock = Arc::new(MockGraphStore::new());
        let plan = test_plan();
        let plan_id = plan.id;
        mock.create_plan(&plan).await.unwrap();

        let mut task = test_task();
        task.affected_files = vec![
            "src/chat/manager.rs".to_string(),
            "src/chat/compaction_context.rs".to_string(),
        ];
        task.title = Some("Implement compaction context".to_string());
        let task_id = task.id;
        mock.create_task(plan_id, &task).await.unwrap();

        let step1 = test_step(1, "Create builder struct");
        let step2 = test_step(2, "Add custom_instructions output");
        mock.create_step(task_id, &step1).await.unwrap();
        mock.create_step(task_id, &step2).await.unwrap();

        // 2. Create CompactionNotifier with task context
        let (tx, _rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx, None, "test-ctx-session".to_string())
            .with_context(
                mock.clone() as Arc<dyn GraphStore>,
                CompactionContextSource::Task(task_id),
            );

        // 3. Execute with a PreCompact input
        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: "test-ctx-session".into(),
            transcript_path: "/tmp/t".into(),
            cwd: "/test".into(),
            permission_mode: None,
            trigger: "auto".into(),
            custom_instructions: None,
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, None, &context).await.unwrap();

        // 4. Verify: output should have custom_instructions (via reason field)
        if let nexus_claude::HookJSONOutput::Sync(sync) = result {
            assert_eq!(sync.continue_, Some(true), "Must always continue");
            let reason = sync
                .reason
                .expect("reason (custom_instructions) should be set");
            assert!(
                reason.contains("src/chat/manager.rs"),
                "Should mention affected files, got: {reason}"
            );
            assert!(
                reason.contains("Implement compaction context"),
                "Should mention task title, got: {reason}"
            );
        } else {
            panic!("Expected Sync output");
        }
    }

    #[tokio::test]
    async fn test_compaction_notifier_no_context_returns_no_instructions() {
        use nexus_claude::{HookCallback, HookContext, HookInput, PreCompactHookInput};

        // CompactionNotifier without context (backward-compatible)
        let (tx, _rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(tx, None, "no-ctx".to_string());

        let input = HookInput::PreCompact(PreCompactHookInput {
            session_id: "no-ctx".into(),
            transcript_path: "/tmp/t".into(),
            cwd: "/".into(),
            permission_mode: None,
            trigger: "auto".into(),
            custom_instructions: None,
        });

        let context = HookContext { signal: None };
        let result = notifier.execute(&input, None, &context).await.unwrap();

        if let nexus_claude::HookJSONOutput::Sync(sync) = result {
            assert_eq!(sync.continue_, Some(true));
            assert!(
                sync.reason.is_none(),
                "No context source → no custom instructions"
            );
        } else {
            panic!("Expected Sync output");
        }
    }

    // ====================================================================
    // Hook dispatch e2e (CompactionNotifier + dispatch_hook_from_registry)
    // ====================================================================

    /// E2E test: simulates the full hook_callback flow as it happens in stream_response.
    ///
    /// 1. Create a CompactionNotifier with a broadcast channel
    /// 2. Register it via InteractiveClient::initialize_hooks()
    /// 3. Clone the hook_callbacks registry (like stream_response does)
    /// 4. Build a hook_callback JSON message (like the CLI would send)
    /// 5. Dispatch via dispatch_hook_from_registry (lock-free)
    /// 6. Verify CompactionStarted was broadcast
    /// 7. Verify the hook output is continue: true
    #[tokio::test]
    async fn test_compaction_hook_e2e_dispatch_broadcasts_event() {
        use nexus_claude::transport::mock::MockTransport;
        use nexus_claude::{dispatch_hook_from_registry, HookMatcher, InteractiveClient};

        // 1. Create CompactionNotifier backed by a broadcast channel
        let (events_tx, mut events_rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(events_tx, None, "test-session-e2e".to_string());

        // 2. Build hooks map and initialize
        let mut hooks = std::collections::HashMap::new();
        hooks.insert(
            "PreCompact".to_string(),
            vec![HookMatcher {
                matcher: None,
                hooks: vec![std::sync::Arc::new(notifier)],
            }],
        );

        let (transport, _handle) = MockTransport::pair();
        let client = InteractiveClient::from_transport_with_hooks(transport, hooks);
        client.initialize_hooks().await.unwrap();

        // 3. Clone registry (this is what stream_response does before the select! loop)
        let registry = client.hook_callbacks();

        // 4. Get the callback_id that was generated
        let callback_id = {
            let cbs = registry.read().await;
            assert_eq!(cbs.len(), 1, "Should have exactly one registered callback");
            cbs.keys().next().unwrap().clone()
        };

        // 5. Build a hook_callback control message (simulating what the CLI sends)
        let control_msg = serde_json::json!({
            "type": "control_request",
            "request_id": "req-e2e-001",
            "request": {
                "subtype": "hook_callback",
                "callback_id": callback_id,
                "input": {
                    "hook_event_name": "PreCompact",
                    "session_id": "test-session-e2e",
                    "transcript_path": "/tmp/transcript.json",
                    "cwd": "/home/user/project",
                    "trigger": "auto"
                }
            }
        });

        // 6. Dispatch (lock-free, just like stream_response does)
        let result = dispatch_hook_from_registry(&control_msg, &registry).await;
        assert!(result.is_some(), "dispatch should find the callback");
        let output = result.unwrap();
        assert!(output.is_ok(), "callback should succeed");

        // 7. Verify CompactionStarted was broadcast on the events channel
        let event = events_rx
            .try_recv()
            .expect("Should have received CompactionStarted event");
        assert!(
            matches!(event, ChatEvent::CompactionStarted { ref trigger } if trigger == "auto"),
            "Event should be CompactionStarted with trigger=auto, got: {:?}",
            event
        );

        // 8. Verify output is Sync with continue: true
        match output.unwrap() {
            nexus_claude::HookJSONOutput::Sync(sync) => {
                assert_eq!(
                    sync.continue_,
                    Some(true),
                    "Hook should return continue=true"
                );
            }
            other => panic!("Expected Sync output, got: {:?}", other),
        }
    }

    /// Test that build_hook_response_json produces valid JSON that would be sent to CLI.
    /// Verifies the format matches what the CLI expects as a control_response.
    #[tokio::test]
    async fn test_hook_response_sent_to_cli_format() {
        use nexus_claude::transport::mock::MockTransport;
        use nexus_claude::{
            build_hook_response_json, dispatch_hook_from_registry, HookMatcher, InteractiveClient,
        };

        // Setup: CompactionNotifier + initialize + dispatch
        let (events_tx, _events_rx) = broadcast::channel::<ChatEvent>(16);
        let notifier = CompactionNotifier::new(events_tx, None, "sess-resp".to_string());

        let mut hooks = std::collections::HashMap::new();
        hooks.insert(
            "PreCompact".to_string(),
            vec![HookMatcher {
                matcher: None,
                hooks: vec![std::sync::Arc::new(notifier)],
            }],
        );

        let (transport, _handle) = MockTransport::pair();
        let client = InteractiveClient::from_transport_with_hooks(transport, hooks);
        client.initialize_hooks().await.unwrap();

        let registry = client.hook_callbacks();
        let callback_id = {
            let cbs = registry.read().await;
            cbs.keys().next().unwrap().clone()
        };

        let request_id = "req-response-test-001";
        let control_msg = serde_json::json!({
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "hook_callback",
                "callback_id": callback_id,
                "input": {
                    "hook_event_name": "PreCompact",
                    "session_id": "sess-resp",
                    "transcript_path": "/tmp/t.json",
                    "cwd": "/home",
                    "trigger": "manual"
                }
            }
        });

        let result = dispatch_hook_from_registry(&control_msg, &registry)
            .await
            .expect("Should dispatch");

        // Build the response JSON (this is what stream_response sends via stdin_tx)
        let response_json_str = build_hook_response_json(request_id, &result);

        // Parse and verify structure
        let response: serde_json::Value =
            serde_json::from_str(&response_json_str).expect("Should be valid JSON");

        assert_eq!(
            response["type"], "control_response",
            "type must be control_response"
        );

        let resp = &response["response"];
        assert_eq!(resp["subtype"], "success", "subtype must be success");
        assert_eq!(
            resp["request_id"], request_id,
            "request_id must match the original"
        );

        // The inner response should contain the hook output (continue: true)
        let inner = &resp["response"];
        assert_eq!(
            inner["continue"], true,
            "Hook output should have continue=true"
        );
    }

    #[tokio::test]
    async fn test_build_options_injects_path_and_cli_path() {
        let state = mock_app_state();
        let mut config = test_config();
        config.process_path = Some("/test/path:/usr/bin:/bin".into());
        config.claude_cli_path = Some("/test/claude".into());
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        let options = manager
            .build_options(
                "/tmp",
                "claude-sonnet-4-6",
                "test prompt",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;

        // Verify PATH was injected into the env HashMap
        assert_eq!(
            options.env.get("PATH").map(|s| s.as_str()),
            Some("/test/path:/usr/bin:/bin"),
            "process_path should be injected as PATH env var"
        );
        // Verify cli_path was set
        assert_eq!(
            options
                .cli_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            Some("/test/claude".to_string()),
            "claude_cli_path should be set on options"
        );
    }

    #[tokio::test]
    async fn test_build_options_without_path_no_injection() {
        let state = mock_app_state();
        let config = test_config(); // process_path=None, claude_cli_path=None
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        let options = manager
            .build_options(
                "/tmp",
                "claude-sonnet-4-6",
                "test prompt",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;

        // PATH should NOT be in env when process_path is None
        assert!(
            !options.env.contains_key("PATH"),
            "PATH should not be injected when process_path is None"
        );
        // cli_path should be None
        assert!(
            options.cli_path.is_none(),
            "cli_path should be None when claude_cli_path is not configured"
        );
    }

    // ================================================================
    // Runtime env config update + persistence tests
    // ================================================================

    #[tokio::test]
    async fn test_runtime_env_config_update_methods() {
        let state = mock_app_state();
        let config = test_config();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        // Initial state: all None / false
        let env = manager.get_env_config().await;
        assert!(env.process_path.is_none());
        assert!(env.claude_cli_path.is_none());
        assert!(!env.auto_update_cli);

        // Update process_path
        manager
            .update_process_path(Some("/custom/path:/usr/bin".into()))
            .await;
        let env = manager.get_env_config().await;
        assert_eq!(env.process_path.as_deref(), Some("/custom/path:/usr/bin"));

        // Update claude_cli_path
        manager
            .update_claude_cli_path(Some("/opt/claude".into()))
            .await;
        let env = manager.get_env_config().await;
        assert_eq!(env.claude_cli_path.as_deref(), Some("/opt/claude"));

        // Update auto_update_cli
        manager.update_auto_update_cli(true).await;
        let env = manager.get_env_config().await;
        assert!(env.auto_update_cli);

        // Clear process_path
        manager.update_process_path(None).await;
        let env = manager.get_env_config().await;
        assert!(env.process_path.is_none());
    }

    #[tokio::test]
    async fn test_runtime_env_config_affects_build_options() {
        let state = mock_app_state();
        let config = test_config();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, config);

        // Initially no PATH injection
        let options = manager
            .build_options(
                "/tmp",
                "claude-sonnet-4-6",
                "test",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;
        assert!(!options.env.contains_key("PATH"));

        // Update at runtime via env_config
        manager
            .update_process_path(Some("/runtime/path:/usr/bin".into()))
            .await;
        manager
            .update_claude_cli_path(Some("/runtime/claude".into()))
            .await;

        // Now build_options should pick up the runtime values
        let options = manager
            .build_options(
                "/tmp",
                "claude-sonnet-4-6",
                "test",
                None,
                None,
                None,
                &[],
                None,
                None,
            )
            .await;
        assert_eq!(
            options.env.get("PATH").map(|s| s.as_str()),
            Some("/runtime/path:/usr/bin")
        );
        assert_eq!(
            options
                .cli_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            Some("/runtime/claude".to_string())
        );
    }

    #[test]
    fn test_persist_env_config_to_yaml_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        let env = RuntimeEnvConfig {
            process_path: Some("/custom/path:/usr/bin".into()),
            claude_cli_path: Some("/opt/claude".into()),
            auto_update_cli: true,
            auto_update_app: true,
        };

        ChatManager::persist_env_config_to_yaml(&yaml_path, &env).unwrap();

        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();
        let chat = doc["chat"].as_mapping().unwrap();
        assert_eq!(
            chat[&serde_yaml::Value::String("process_path".into())]
                .as_str()
                .unwrap(),
            "/custom/path:/usr/bin"
        );
        assert_eq!(
            chat[&serde_yaml::Value::String("claude_cli_path".into())]
                .as_str()
                .unwrap(),
            "/opt/claude"
        );
        assert!(chat[&serde_yaml::Value::String("auto_update_cli".into())]
            .as_bool()
            .unwrap());
    }

    #[test]
    fn test_persist_env_config_preserves_existing_yaml() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // Write an existing config with other fields
        std::fs::write(
            &yaml_path,
            "server:\n  port: 6600\nchat:\n  default_model: claude-sonnet-4-6\n  permissions:\n    mode: default\n",
        )
        .unwrap();

        let env = RuntimeEnvConfig {
            process_path: Some("/my/path".into()),
            claude_cli_path: None,
            auto_update_cli: false,
            auto_update_app: true,
        };

        ChatManager::persist_env_config_to_yaml(&yaml_path, &env).unwrap();

        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();

        // server section preserved
        assert_eq!(doc["server"]["port"].as_u64().unwrap(), 6600);
        // chat.default_model preserved
        assert_eq!(
            doc["chat"]["default_model"].as_str().unwrap(),
            "claude-sonnet-4-6"
        );
        // chat.permissions preserved
        assert_eq!(
            doc["chat"]["permissions"]["mode"].as_str().unwrap(),
            "default"
        );
        // process_path set
        assert_eq!(doc["chat"]["process_path"].as_str().unwrap(), "/my/path");
        // claude_cli_path removed (None)
        assert!(doc["chat"]["claude_cli_path"].is_null());
        // auto_update_cli removed (false → omitted)
        assert!(doc["chat"]["auto_update_cli"].is_null());
    }

    #[test]
    fn test_persist_env_config_clears_fields() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("config.yaml");

        // First persist with values
        let env1 = RuntimeEnvConfig {
            process_path: Some("/my/path".into()),
            claude_cli_path: Some("/my/claude".into()),
            auto_update_cli: true,
            auto_update_app: true,
        };
        ChatManager::persist_env_config_to_yaml(&yaml_path, &env1).unwrap();

        // Then persist with cleared values
        let env2 = RuntimeEnvConfig {
            process_path: None,
            claude_cli_path: None,
            auto_update_cli: false,
            auto_update_app: true,
        };
        ChatManager::persist_env_config_to_yaml(&yaml_path, &env2).unwrap();

        let contents = std::fs::read_to_string(&yaml_path).unwrap();
        let doc: serde_yaml::Value = serde_yaml::from_str(&contents).unwrap();

        // All fields should be removed
        let chat = doc["chat"].as_mapping().unwrap();
        assert!(!chat.contains_key(serde_yaml::Value::String("process_path".into())));
        assert!(!chat.contains_key(serde_yaml::Value::String("claude_cli_path".into())));
        assert!(!chat.contains_key(serde_yaml::Value::String("auto_update_cli".into())));
    }

    // ================================================================
    // parse_spawned_by tests
    // ================================================================

    #[test]
    fn test_parse_spawned_by_full_context() {
        let run_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();
        let proto_run_id = Uuid::new_v4();
        let json = serde_json::json!({
            "parent_session_id": "sess-123",
            "type": "wave-runner",
            "run_id": run_id.to_string(),
            "task_id": task_id.to_string(),
            "protocol_run_id": proto_run_id.to_string(),
            "protocol_state": "implement"
        })
        .to_string();

        let ctx = parse_spawned_by(&json).unwrap();
        assert_eq!(ctx.parent_session_id, Some("sess-123".to_string()));
        assert_eq!(ctx.spawn_type, "wave-runner");
        assert_eq!(ctx.run_id, Some(run_id));
        assert_eq!(ctx.task_id, Some(task_id));
        assert_eq!(ctx.protocol_run_id, Some(proto_run_id));
        assert_eq!(ctx.protocol_state, Some("implement".to_string()));
    }

    #[test]
    fn test_parse_spawned_by_minimal() {
        let json = r#"{"parent_session_id": "sess-456"}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.parent_session_id, Some("sess-456".to_string()));
        assert_eq!(ctx.spawn_type, "runner"); // default
        assert_eq!(ctx.run_id, None);
        assert_eq!(ctx.task_id, None);
        assert_eq!(ctx.protocol_run_id, None);
        assert_eq!(ctx.protocol_state, None);
    }

    #[test]
    fn test_parse_spawned_by_protocol_only() {
        let proto_run_id = Uuid::new_v4();
        let json = serde_json::json!({
            "protocol_run_id": proto_run_id.to_string(),
            "protocol_state": "review"
        })
        .to_string();

        let ctx = parse_spawned_by(&json).unwrap();
        assert_eq!(ctx.parent_session_id, None);
        assert_eq!(ctx.protocol_run_id, Some(proto_run_id));
        assert_eq!(ctx.protocol_state, Some("review".to_string()));
    }

    // ---------------------------------------------------------------
    // parse_spawned_by — scaffolding_level tests
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_spawned_by_scaffolding_level_zero() {
        let json = r#"{"scaffolding_level": 0}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.scaffolding_level, Some(0));
    }

    #[test]
    fn test_parse_spawned_by_scaffolding_level_mid() {
        let json = r#"{"scaffolding_level": 2}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.scaffolding_level, Some(2));
    }

    #[test]
    fn test_parse_spawned_by_scaffolding_level_max() {
        let json = r#"{"scaffolding_level": 4}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.scaffolding_level, Some(4));
    }

    #[test]
    fn test_parse_spawned_by_scaffolding_level_capped_at_4() {
        let json = r#"{"scaffolding_level": 5}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.scaffolding_level, Some(4));
    }

    #[test]
    fn test_parse_spawned_by_scaffolding_level_large_capped() {
        let json = r#"{"scaffolding_level": 255}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.scaffolding_level, Some(4));
    }

    #[test]
    fn test_parse_spawned_by_scaffolding_level_null() {
        let json = r#"{"scaffolding_level": null}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.scaffolding_level, None);
    }

    #[test]
    fn test_parse_spawned_by_no_scaffolding_level_backward_compat() {
        let json = r#"{"type": "runner", "parent_session_id": "abc"}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.scaffolding_level, None);
        assert_eq!(ctx.parent_session_id, Some("abc".to_string()));
        assert_eq!(ctx.spawn_type, "runner");
    }

    #[test]
    fn test_parse_spawned_by_scaffolding_with_other_fields() {
        let run_id = Uuid::new_v4();
        let task_id = Uuid::new_v4();
        let json = format!(
            r#"{{"type":"runner","run_id":"{}","task_id":"{}","scaffolding_level":3}}"#,
            run_id, task_id
        );
        let ctx = parse_spawned_by(&json).unwrap();
        assert_eq!(ctx.scaffolding_level, Some(3));
        assert_eq!(ctx.run_id, Some(run_id));
        assert_eq!(ctx.task_id, Some(task_id));
        assert_eq!(ctx.spawn_type, "runner");
    }

    #[test]
    fn test_parse_spawned_by_empty_json() {
        let ctx = parse_spawned_by("{}").unwrap();
        assert_eq!(ctx.parent_session_id, None);
        assert_eq!(ctx.spawn_type, "runner");
        assert_eq!(ctx.protocol_run_id, None);
    }

    #[test]
    fn test_parse_spawned_by_invalid_json() {
        assert!(parse_spawned_by("not json").is_none());
    }

    #[test]
    fn test_parse_spawned_by_invalid_uuid() {
        let json = r#"{"protocol_run_id": "not-a-uuid", "run_id": "also-bad"}"#;
        let ctx = parse_spawned_by(json).unwrap();
        assert_eq!(ctx.protocol_run_id, None);
        assert_eq!(ctx.run_id, None);
    }

    // ── is_conclusive_tool tests ────────────────────────────────────────

    #[test]
    fn test_git_commit_is_conclusive() {
        let input = serde_json::json!({"command": "git commit -m 'feat: add stuff'"});
        assert!(is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_git_push_is_conclusive() {
        let input = serde_json::json!({"command": "git push origin feature-branch"});
        assert!(is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_git_status_is_conclusive() {
        let input = serde_json::json!({"command": "git status"});
        assert!(is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_git_log_is_conclusive() {
        let input = serde_json::json!({"command": "git log --oneline -5"});
        assert!(is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_git_tag_is_conclusive() {
        let input = serde_json::json!({"command": "git tag v1.0.0"});
        assert!(is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_chained_git_commit_push_is_conclusive() {
        let input = serde_json::json!({"command": "git commit -m 'fix' && git push"});
        assert!(is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_cargo_build_is_not_conclusive() {
        let input = serde_json::json!({"command": "cargo build"});
        assert!(!is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_npm_test_is_not_conclusive() {
        let input = serde_json::json!({"command": "npm test"});
        assert!(!is_conclusive_tool("Bash", &input));
    }

    #[test]
    fn test_edit_tool_is_not_conclusive() {
        let input =
            serde_json::json!({"file_path": "src/main.rs", "old_string": "a", "new_string": "b"});
        assert!(!is_conclusive_tool("Edit", &input));
    }

    #[test]
    fn test_read_tool_is_not_conclusive() {
        let input = serde_json::json!({"file_path": "src/main.rs"});
        assert!(!is_conclusive_tool("Read", &input));
    }

    #[test]
    fn test_write_tool_is_not_conclusive() {
        let input = serde_json::json!({"file_path": "src/main.rs", "content": "fn main() {}"});
        assert!(!is_conclusive_tool("Write", &input));
    }

    #[test]
    fn test_bash_without_command_is_not_conclusive() {
        let input = serde_json::json!({});
        assert!(!is_conclusive_tool("Bash", &input));
    }

    // ── T5 of plan 28e9afe3: cancel_running_tools tests ──────────────────
    //
    // Most invariants of `cancel_running_tools` are enforced by the type
    // system: `kill_descendants` takes only `Option<u32>` (no
    // `AtomicBool`/`CancellationToken` — cannot touch flag/token), and
    // `check_and_record_cancel_cap` takes only `&Arc<Mutex<VecDeque>>`,
    // `u32`, `Duration`. Runtime invariant (cancel doesn't break the
    // stream) is covered by the live e2e test in T6.
    //
    // Unit tests below cover: rate cap (basics + recovery), kill helper
    // edge cases, and CancelToolsResult serde shape.

    #[tokio::test]
    async fn test_cancel_cap_allows_up_to_cap_then_refuses() {
        let history = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
        let cap: u32 = 10;
        let window = Duration::from_secs(60);

        for i in 0..cap {
            assert!(
                ChatManager::check_and_record_cancel_cap(&history, cap, window).await,
                "call #{i} below cap must be allowed"
            );
        }
        // 11th refused, history len stays at cap.
        assert!(!ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
        assert_eq!(history.lock().await.len() as u32, cap);
        // Subsequent refused calls don't add to history either.
        assert!(!ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
        assert_eq!(history.lock().await.len() as u32, cap);
    }

    #[tokio::test]
    async fn test_cancel_cap_recovers_after_window_expires() {
        let history = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
        let cap: u32 = 2;
        // Tiny window so we can age entries out within test time.
        let window = Duration::from_millis(50);

        assert!(ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
        assert!(ChatManager::check_and_record_cancel_cap(&history, cap, window).await);
        assert!(!ChatManager::check_and_record_cancel_cap(&history, cap, window).await);

        // Wait past the window, history should drain.
        tokio::time::sleep(Duration::from_millis(80)).await;

        assert!(
            ChatManager::check_and_record_cancel_cap(&history, cap, window).await,
            "after window expiry, calls should be allowed again"
        );
    }

    /// P5: through `cancel_running_tools` (what the WS `cancel_tools` frame and
    /// the REST route call), a Claude Code session takes `CANCEL_TOOLS_CAP`
    /// cancels per window, then refuses with `capped` — the agent engine
    /// applies the same cap (`agent_e2e_tests::cancel_tools`).
    #[tokio::test]
    async fn test_cancel_running_tools_caps_a_claude_code_session() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        manager
            .active_sessions
            .write()
            .await
            .insert("legacy".into(), session);

        for i in 0..CANCEL_TOOLS_CAP {
            let result = manager.cancel_running_tools("legacy").await.unwrap();
            assert!(!result.capped, "call {i} is within the cap");
        }
        let past = manager.cancel_running_tools("legacy").await.unwrap();
        assert!(past.capped, "the call past the cap is refused");
        assert!(past.killed_pids.is_empty());
    }

    #[test]
    fn test_kill_descendants_with_no_pid_is_noop() {
        // No PID → nothing to kill, returns empty Vec, no panic, no SIGINT.
        let killed = ChatManager::kill_descendants(None);
        assert!(killed.is_empty());
    }

    // ========================================================================
    // T3 of plan fc35b25e — kill_subtree(pid) helper
    // ========================================================================

    #[test]
    #[cfg(unix)]
    fn test_kill_subtree_nonexistent_pid_is_noop() {
        // PID 999_999 is almost certainly not a live process — kill returns
        // ESRCH which we silently filter, so killed list is empty and the
        // call must not panic.
        let killed = ChatManager::kill_subtree(999_999);
        assert!(
            killed.is_empty(),
            "expected empty killed list, got {killed:?}"
        );
    }

    /// Spawn a real `sleep 30`, call `kill_subtree`, verify the subprocess
    /// is reaped within a reasonable timeout. Marked `#[ignore]` because it
    /// touches real processes — run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    #[cfg(unix)]
    fn test_kill_subtree_kills_real_sleep_subprocess() {
        let mut child = match std::process::Command::new("sleep").arg("30").spawn() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: sleep not available");
                return;
            }
        };
        let pid = child.id();

        // Give the OS a moment to register the process.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let killed = ChatManager::kill_subtree(pid);
        assert!(
            killed.contains(&pid),
            "expected {pid} in killed list, got {killed:?}"
        );

        // Reap the zombie and confirm the process actually died (sleep was
        // 30s — if it's already dead now, kill_subtree did its job).
        let exit = child.wait().expect("waitpid");
        assert!(
            !exit.success(),
            "subprocess should have been signalled, not exited cleanly"
        );
    }

    #[test]
    fn test_cancel_tools_result_serde_roundtrip() {
        let result = CancelToolsResult {
            cli_pid: Some(12345),
            killed_pids: vec![67890, 67891],
            capped: false,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        // Spot-check the wire format the frontend will consume.
        assert!(json.contains("\"cli_pid\":12345"));
        assert!(json.contains("\"killed_pids\":[67890,67891]"));
        assert!(json.contains("\"capped\":false"));

        let parsed: CancelToolsResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.cli_pid, Some(12345));
        assert_eq!(parsed.killed_pids, vec![67890, 67891]);
        assert!(!parsed.capped);
    }

    #[test]
    fn test_cancel_tools_result_serde_capped_state() {
        let result = CancelToolsResult {
            cli_pid: Some(1234),
            killed_pids: vec![],
            capped: true,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"capped\":true"));
        assert!(json.contains("\"killed_pids\":[]"));
    }

    // ========================================================================
    // T3 of plan 754a1379 — track_background_task_start (INSERT lifecycle hook)
    // ========================================================================

    /// Helper: insert a dummy session under the given id and return the
    /// active_sessions map ready for the helper. No CLI needed.
    async fn build_sessions_map_for_track_test(
        session_id: &str,
    ) -> (
        Arc<RwLock<HashMap<String, ActiveSession>>>,
        broadcast::Sender<ChatEvent>,
        broadcast::Receiver<ChatEvent>,
    ) {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let events_tx = session.events_tx.clone();
        let events_rx = events_tx.subscribe();
        let map = Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert(session_id.to_string(), session);
        (map, events_tx, events_rx)
    }

    #[tokio::test]
    async fn test_track_background_task_inserts_monitor_on_first_pass() {
        let (sessions, events_tx, mut events_rx) = build_sessions_map_for_track_test("s-mon").await;

        // First pass: ContentBlockStart with empty input.
        let inserted = ChatManager::track_background_task_start(
            "s-mon",
            &sessions,
            &events_tx,
            &None,
            "tool_M1",
            "Monitor",
            &serde_json::json!({}),
            None,
        )
        .await;
        assert!(inserted, "Monitor should be tracked on first pass");

        // Map should now have one entry.
        let session_guard = sessions.read().await;
        let session = session_guard.get("s-mon").unwrap();
        let tasks = session.active_background_tasks.lock().await;
        assert_eq!(tasks.len(), 1);
        let entry = tasks.get("tool_M1").unwrap();
        assert_eq!(entry.kind, BackgroundTaskKind::Monitor);
        assert_eq!(entry.description, "(no description)");

        // Should have broadcast an ActiveTasksUpdate.
        let event = events_rx.try_recv().expect("expected one broadcast");
        match event {
            ChatEvent::ActiveTasksUpdate { tasks } => assert_eq!(tasks.len(), 1),
            other => panic!("expected ActiveTasksUpdate, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_track_background_task_updates_description_on_second_pass() {
        let (sessions, events_tx, mut events_rx) =
            build_sessions_map_for_track_test("s-mon2").await;

        // First pass — empty input, placeholder description.
        ChatManager::track_background_task_start(
            "s-mon2",
            &sessions,
            &events_tx,
            &None,
            "tool_M2",
            "Monitor",
            &serde_json::json!({}),
            None,
        )
        .await;
        // Drain first broadcast.
        let _ = events_rx.try_recv();

        // Second pass — full input with description.
        ChatManager::track_background_task_start(
            "s-mon2",
            &sessions,
            &events_tx,
            &None,
            "tool_M2",
            "Monitor",
            &serde_json::json!({
                "command": "tail -F /tmp/build.log",
                "description": "watch build log",
                "persistent": false,
                "timeout_ms": 300000,
            }),
            None,
        )
        .await;

        let session_guard = sessions.read().await;
        let tasks = session_guard
            .get("s-mon2")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        // Still one entry (idempotent), description refreshed.
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks.get("tool_M2").unwrap().description, "watch build log");

        // Two broadcasts total (one per pass).
        events_rx.try_recv().expect("second broadcast missing");
    }

    #[tokio::test]
    async fn test_track_background_task_bash_run_in_background_first_pass_skipped() {
        let (sessions, events_tx, mut events_rx) =
            build_sessions_map_for_track_test("s-bash").await;

        // First pass: empty input, run_in_background flag absent → not tracked.
        let inserted = ChatManager::track_background_task_start(
            "s-bash",
            &sessions,
            &events_tx,
            &None,
            "tool_B1",
            "Bash",
            &serde_json::json!({}),
            None,
        )
        .await;
        assert!(!inserted, "Bash with empty input must not be tracked yet");

        // Second pass: full input with run_in_background=true → tracked.
        let inserted = ChatManager::track_background_task_start(
            "s-bash",
            &sessions,
            &events_tx,
            &None,
            "tool_B1",
            "Bash",
            &serde_json::json!({
                "command": "cargo watch -x test",
                "description": "run tests on change",
                "run_in_background": true,
            }),
            None,
        )
        .await;
        assert!(inserted, "Bash with run_in_background=true must be tracked");

        let session_guard = sessions.read().await;
        let tasks = session_guard
            .get("s-bash")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(tasks.len(), 1);
        let entry = tasks.get("tool_B1").unwrap();
        assert_eq!(entry.kind, BackgroundTaskKind::BashBackground);
        assert_eq!(entry.description, "run tests on change");

        // Only ONE broadcast (the second pass) since the first returned false.
        events_rx.try_recv().expect("expected the second broadcast");
        assert!(events_rx.try_recv().is_err(), "no extra broadcast");
    }

    #[tokio::test]
    async fn test_track_background_task_bash_synchronous_not_tracked() {
        let (sessions, events_tx, mut events_rx) =
            build_sessions_map_for_track_test("s-bash-sync").await;

        // Synchronous Bash (run_in_background absent or false) — never tracked.
        let inserted_a = ChatManager::track_background_task_start(
            "s-bash-sync",
            &sessions,
            &events_tx,
            &None,
            "tool_B2",
            "Bash",
            &serde_json::json!({ "command": "ls" }),
            None,
        )
        .await;
        let inserted_b = ChatManager::track_background_task_start(
            "s-bash-sync",
            &sessions,
            &events_tx,
            &None,
            "tool_B3",
            "Bash",
            &serde_json::json!({ "command": "pwd", "run_in_background": false }),
            None,
        )
        .await;
        assert!(!inserted_a);
        assert!(!inserted_b);

        let tasks = sessions
            .read()
            .await
            .get("s-bash-sync")
            .unwrap()
            .active_background_tasks
            .lock()
            .await
            .len();
        assert_eq!(tasks, 0, "no Bash should be tracked");
        assert!(
            events_rx.try_recv().is_err(),
            "no broadcast expected for untracked tools"
        );
    }

    #[tokio::test]
    async fn test_track_background_task_other_tools_ignored() {
        let (sessions, events_tx, _events_rx) = build_sessions_map_for_track_test("s-other").await;

        for tool in ["Read", "Write", "Edit", "Glob", "Task"] {
            let inserted = ChatManager::track_background_task_start(
                "s-other",
                &sessions,
                &events_tx,
                &None,
                "tool_X",
                tool,
                &serde_json::json!({"description": "should be ignored"}),
                None,
            )
            .await;
            assert!(!inserted, "{} must not be tracked", tool);
        }

        let count = sessions
            .read()
            .await
            .get("s-other")
            .unwrap()
            .active_background_tasks
            .lock()
            .await
            .len();
        assert_eq!(count, 0);
    }

    // ========================================================================
    // T2 of plan fc35b25e — async PID claim from track_background_task_start
    // ========================================================================

    /// Trigger `track_background_task_start` against a session whose
    /// `child_pid` points at a real `bash` subprocess that periodically
    /// forks new descendants. The async claim must:
    /// 1. snapshot descendants at insert time (`before`),
    /// 2. sleep 1 s,
    /// 3. snapshot again (`after`) and diff,
    /// 4. claim the newest new PID,
    /// 5. update the entry's `pid` and broadcast a fresh `ActiveTasksUpdate`.
    ///
    /// Marked `#[ignore]` because it spawns real subprocesses and depends
    /// on timing — run with `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    #[cfg(unix)]
    async fn test_pid_claim_async_updates_map_after_1s() {
        use std::process::Stdio;

        // Spawn a bash that chains short sleeps so its descendant set
        // changes during the 1 s claim window. At t=0 the descendant is
        // a 0.3 s sleep; at t=1.0 it's the long-lived 30 s sleep.
        //
        // The trailing `& wait` is critical: without it, bash would
        // optimise the final command via exec(2) and **replace itself**
        // with `sleep 30`, leaving no descendants for pgrep to find.
        // Backgrounding with `&` and using `wait` forces bash to fork
        // and stay alive as the PPID.
        let mut bash = match std::process::Command::new("bash")
            .arg("-c")
            .arg("sleep 0.3; sleep 0.5; sleep 30 & wait")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: bash not available");
                return;
            }
        };
        let bash_pid = bash.id();
        // Give the OS a moment so pgrep can see the first descendant.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let (mut session, _) = create_dummy_session(false, "", vec![]);
        session.child_pid = Some(bash_pid);
        let events_tx = session.events_tx.clone();
        let mut events_rx = events_tx.subscribe();

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-claim".into(), session);

        let inserted = ChatManager::track_background_task_start(
            "s-claim",
            &map,
            &events_tx,
            &None,
            "tu-claim",
            "Monitor",
            &serde_json::json!({}),
            None,
        )
        .await;
        assert!(inserted, "Monitor first-pass should insert");

        // Drain the first broadcast (insert with pid=None).
        match tokio::time::timeout(Duration::from_secs(1), events_rx.recv()).await {
            Ok(Ok(ChatEvent::ActiveTasksUpdate { tasks })) => {
                assert_eq!(tasks.len(), 1);
                assert!(tasks[0].pid.is_none(), "first broadcast pid must be None");
            }
            other => {
                let _ = bash.kill();
                let _ = bash.wait();
                panic!("expected ActiveTasksUpdate broadcast, got {other:?}");
            }
        }

        // Wait past the 1 s sleep + a margin for the async claim to land.
        tokio::time::sleep(Duration::from_millis(1500)).await;

        // The entry should now carry pid=Some(...).
        let claimed_pid = {
            let sessions = map.read().await;
            let active = sessions.get("s-claim").expect("session present");
            let tasks = active.active_background_tasks.lock().await;
            let entry = tasks.get("tu-claim").expect("entry present");
            entry.pid
        };

        // Cleanup before assertions so a panic doesn't leak the bash.
        let _ = bash.kill();
        let _ = bash.wait();

        assert!(
            claimed_pid.is_some(),
            "expected pid populated by async claim, got {claimed_pid:?}"
        );
    }

    /// Cancel the entry between the synchronous track call and the 1 s
    /// async wake-up. The claim must no-op without re-inserting the entry
    /// or panicking.
    #[tokio::test]
    async fn test_pid_claim_noop_when_entry_removed_before_wakeup() {
        let (mut session, _) = create_dummy_session(false, "", vec![]);
        // Set a non-None child_pid so the claim task is actually spawned
        // (else it short-circuits in the synchronous track call). We use
        // a guaranteed-nonexistent PID so `pgrep -P` returns instantly
        // with no descendants → the diff is empty and the claim takes
        // the warn-and-return path, which is what we want to exercise.
        // (Avoid PID 1 — on macOS that's launchd whose subtree is the
        // entire process table, making `get_descendant_pids` very slow.)
        session.child_pid = Some(999_999);
        let events_tx = session.events_tx.clone();
        let _events_rx = events_tx.subscribe();

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-race".into(), session);

        let inserted = ChatManager::track_background_task_start(
            "s-race",
            &map,
            &events_tx,
            &None,
            "tu-race",
            "Monitor",
            &serde_json::json!({}),
            None,
        )
        .await;
        assert!(inserted);

        // Immediately remove the entry — simulates a rapid cancel that
        // beats the async claim.
        {
            let sessions = map.read().await;
            let active = sessions.get("s-race").expect("session present");
            active
                .active_background_tasks
                .lock()
                .await
                .remove("tu-race");
        }

        // Wait past the 1 s sleep + a margin.
        tokio::time::sleep(Duration::from_millis(1300)).await;

        // The entry must NOT have been re-inserted by the claim, and the
        // process must not have panicked (if it did, the test runtime
        // would have aborted).
        let count = map
            .read()
            .await
            .get("s-race")
            .unwrap()
            .active_background_tasks
            .lock()
            .await
            .len();
        assert_eq!(count, 0, "claim must not re-insert removed entries");
    }

    // ========================================================================
    // T7 of plan 754a1379 — cancel_task (map-side cancel V1)
    // ========================================================================

    #[tokio::test]
    async fn test_cancel_task_marks_for_removal_and_broadcasts() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        // Pre-seed a task in the map.
        session.active_background_tasks.lock().await.insert(
            "tool_T1".into(),
            BackgroundTaskInfo {
                id: "tool_T1".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "tail log".into(),
                started_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                pid: None,
                parent_tool_use_id: Some("tool_T1".into()),
                pending_removal_at: None,
            },
        );

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        manager
            .active_sessions
            .write()
            .await
            .insert("s-cancel".into(), session);

        let result = manager.cancel_task("s-cancel", "tool_T1").await.unwrap();
        assert_eq!(result.task_id, "tool_T1");
        // V2 (T4): entry has pid=None → fallback path, killed_pids empty.
        assert!(
            result.killed_pids.is_empty(),
            "expected empty killed_pids for pid=None entry"
        );
        assert!(!result.capped);

        // Entry should be marked for removal but still in the map (T12 grace).
        let sessions = manager.active_sessions.read().await;
        let tasks = sessions
            .get("s-cancel")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(tasks.len(), 1);
        assert!(tasks.get("tool_T1").unwrap().pending_removal_at.is_some());

        // Broadcast: ActiveTasksUpdate carrying the snapshot (the entry is
        // still visible — it's the grace-period state).
        let event = events_rx.try_recv().expect("expected broadcast");
        match event {
            ChatEvent::ActiveTasksUpdate { tasks } => assert_eq!(tasks.len(), 1),
            other => panic!("expected ActiveTasksUpdate, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cancel_task_unknown_task_id_idempotent() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        manager
            .active_sessions
            .write()
            .await
            .insert("s-empty".into(), session);

        // No tasks in the map — cancel still succeeds, just broadcasts an
        // empty snapshot (idempotent: cancelling something that was never
        // there is OK).
        let result = manager.cancel_task("s-empty", "unknown_id").await.unwrap();
        assert!(!result.capped);
        assert!(result.killed_pids.is_empty());

        let event = events_rx.try_recv().expect("broadcast still emitted");
        match event {
            ChatEvent::ActiveTasksUpdate { tasks } => assert!(tasks.is_empty()),
            other => panic!("expected empty ActiveTasksUpdate, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cancel_task_unknown_session_idempotent() {
        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());

        // No session inserted at all.
        let result = manager
            .cancel_task("ghost-session", "tool_X")
            .await
            .unwrap();
        assert_eq!(result.task_id, "tool_X");
        assert!(!result.capped);
        assert!(result.killed_pids.is_empty());
    }

    #[tokio::test]
    async fn test_cancel_task_rate_cap_enforced() {
        let (mut session, _) = create_dummy_session(false, "", vec![]);
        // Tighten the cap so the test runs in milliseconds: 2 cancels /
        // 60s window. Reusing the same window const would make the test
        // either very slow or flaky.
        session.cancel_task_cap = 2;
        session.cancel_task_window = Duration::from_secs(60);

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        manager
            .active_sessions
            .write()
            .await
            .insert("s-cap".into(), session);

        // 2 calls allowed.
        for _ in 0..2 {
            let r = manager.cancel_task("s-cap", "tool_W").await.unwrap();
            assert!(!r.capped);
        }
        // 3rd call should be capped.
        let r = manager.cancel_task("s-cap", "tool_W").await.unwrap();
        assert!(r.capped, "3rd call must be capped");
        assert!(r.killed_pids.is_empty());
    }

    // ========================================================================
    // T4 of plan fc35b25e — cancel_task wires `kill_subtree` for V2 kill
    // ========================================================================

    /// When the task entry has `pid: None` (claim race or subprocess
    /// crashed before discovery), `cancel_task` must:
    /// - return `killed_pids: []` (no kill happened),
    /// - still mark the entry for removal (V1 fallback preserved),
    /// - still broadcast an `ActiveTasksUpdate` (frontend feedback).
    #[tokio::test]
    async fn test_cancel_task_without_pid_falls_back_to_v1() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        // Pre-seed a task with pid=None to exercise the fallback branch.
        session.active_background_tasks.lock().await.insert(
            "tool_NoPid".into(),
            BackgroundTaskInfo {
                id: "tool_NoPid".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "no pid yet".into(),
                started_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                pid: None,
                parent_tool_use_id: Some("tool_NoPid".into()),
                pending_removal_at: None,
            },
        );

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        manager
            .active_sessions
            .write()
            .await
            .insert("s-nopid".into(), session);

        let result = manager.cancel_task("s-nopid", "tool_NoPid").await.unwrap();
        assert_eq!(result.task_id, "tool_NoPid");
        assert!(
            result.killed_pids.is_empty(),
            "fallback path must return empty killed_pids, got {:?}",
            result.killed_pids
        );
        assert!(!result.capped);

        // Mark for removal still applied (V1 fallback preserved).
        let sessions = manager.active_sessions.read().await;
        let tasks = sessions
            .get("s-nopid")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert!(tasks
            .get("tool_NoPid")
            .unwrap()
            .pending_removal_at
            .is_some());

        // Broadcast still happens.
        let event = events_rx.try_recv().expect("expected broadcast");
        assert!(matches!(event, ChatEvent::ActiveTasksUpdate { .. }));
    }

    /// Spawn a real `sleep 30`, seed a task entry with its PID, call
    /// `cancel_task`, and verify:
    /// - `killed_pids` contains the sleep's PID,
    /// - the subprocess actually dies (exit status non-success).
    ///
    /// Marked `#[ignore]` because it touches real processes — run with
    /// `cargo test -- --ignored`.
    #[tokio::test]
    #[ignore]
    #[cfg(unix)]
    async fn test_cancel_task_with_pid_calls_kill_subtree() {
        let mut child = match std::process::Command::new("sleep").arg("30").spawn() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: sleep not available");
                return;
            }
        };
        let real_pid = child.id();

        let (session, _) = create_dummy_session(false, "", vec![]);
        // Pre-seed a task entry pointing at the real subprocess.
        session.active_background_tasks.lock().await.insert(
            "tool_Real".into(),
            BackgroundTaskInfo {
                id: "tool_Real".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "real sleep".into(),
                started_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                pid: Some(real_pid),
                parent_tool_use_id: Some("tool_Real".into()),
                pending_removal_at: None,
            },
        );

        let state = mock_app_state();
        let manager = ChatManager::new_without_memory(state.neo4j, state.meili, test_config());
        manager
            .active_sessions
            .write()
            .await
            .insert("s-real".into(), session);

        let result = manager.cancel_task("s-real", "tool_Real").await.unwrap();
        assert!(!result.capped);
        assert!(
            result.killed_pids.contains(&real_pid),
            "expected real_pid {real_pid} in killed_pids, got {:?}",
            result.killed_pids
        );

        // The subprocess should have received SIGINT and be dead now.
        let exit = child.wait().expect("waitpid");
        assert!(
            !exit.success(),
            "subprocess should have been signalled, not exited cleanly"
        );
    }

    // ========================================================================
    // T1 of plan fc35b25e — parse_etime + pid_discovery_diff helpers
    // ========================================================================

    #[test]
    fn test_parse_etime_seconds_only() {
        assert_eq!(ChatManager::parse_etime("00:01"), Some(1));
        assert_eq!(ChatManager::parse_etime("00:42"), Some(42));
        assert_eq!(ChatManager::parse_etime("00:00"), Some(0));
    }

    #[test]
    fn test_parse_etime_minutes_seconds() {
        assert_eq!(ChatManager::parse_etime("12:34"), Some(12 * 60 + 34));
        assert_eq!(ChatManager::parse_etime("00:30"), Some(30));
        assert_eq!(ChatManager::parse_etime("59:59"), Some(59 * 60 + 59));
    }

    #[test]
    fn test_parse_etime_hours_minutes_seconds() {
        assert_eq!(
            ChatManager::parse_etime("02:12:34"),
            Some(2 * 3600 + 12 * 60 + 34)
        );
        assert_eq!(ChatManager::parse_etime("01:00:00"), Some(3600));
    }

    #[test]
    fn test_parse_etime_with_days() {
        assert_eq!(
            ChatManager::parse_etime("01-02:12:34"),
            Some(86_400 + 2 * 3600 + 12 * 60 + 34)
        );
        assert_eq!(ChatManager::parse_etime("07-00:00:00"), Some(7 * 86_400));
    }

    #[test]
    fn test_parse_etime_handles_whitespace() {
        // `ps -o etime=` may pad with leading spaces.
        assert_eq!(ChatManager::parse_etime("  00:42  "), Some(42));
        assert_eq!(ChatManager::parse_etime("\n12:34\n"), Some(12 * 60 + 34));
    }

    #[test]
    fn test_parse_etime_rejects_garbage() {
        assert_eq!(ChatManager::parse_etime(""), None);
        assert_eq!(ChatManager::parse_etime("hello"), None);
        assert_eq!(ChatManager::parse_etime("1:2:3:4"), None); // too many parts
        assert_eq!(ChatManager::parse_etime("aa:bb"), None);
        assert_eq!(ChatManager::parse_etime("-1:00"), None);
    }

    #[test]
    fn test_pid_discovery_diff_finds_new_pids() {
        // before = [100, 200], after = [100, 200, 300] → new = [300]
        let before = vec![100u32, 200];
        let after = vec![100u32, 200, 300];
        let diff = ChatManager::pid_discovery_diff(&before, &after);
        // PID 300 doesn't exist in /proc, etime returns None → sorted last,
        // but it's the only new entry so it's at index 0.
        assert_eq!(diff, vec![300]);
    }

    #[test]
    fn test_pid_discovery_diff_no_new_pids() {
        let before = vec![100u32, 200, 300];
        let after = vec![100u32, 200];
        let diff = ChatManager::pid_discovery_diff(&before, &after);
        // No PIDs in `after` that aren't in `before`.
        assert!(diff.is_empty());
    }

    #[test]
    fn test_pid_discovery_diff_empty_before() {
        let before = Vec::<u32>::new();
        let after = vec![1u32, 2, 3];
        let diff = ChatManager::pid_discovery_diff(&before, &after);
        assert_eq!(diff.len(), 3);
        // Order is stable (etime returns None for fake PIDs → all u64::MAX,
        // sort_by_key preserves insertion order on ties).
    }

    #[test]
    fn test_pid_discovery_diff_with_real_subprocess() {
        // Spawn a real `sleep 30` subprocess so we have a PID with a
        // readable etime. Diff should put it before fake PIDs (which
        // have unreadable etime → sorted last).
        let mut child = match std::process::Command::new("sleep").arg("30").spawn() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: sleep not available");
                return;
            }
        };
        let real_pid = child.id();

        // Give the OS a moment to register the process so `ps` sees it.
        std::thread::sleep(std::time::Duration::from_millis(100));

        let before = Vec::<u32>::new();
        let after = vec![999_999u32, real_pid, 999_998u32];
        let diff = ChatManager::pid_discovery_diff(&before, &after);

        // Real PID has etime ~0s (just spawned), fake PIDs have etime
        // None → u64::MAX → sorted last. So real PID should be first.
        assert_eq!(diff[0], real_pid, "real subprocess should sort first");

        let _ = child.kill();
        let _ = child.wait();
    }

    // ========================================================================
    // T13 of plan 754a1379 — track_background_task_recovery_if_orphan
    // (lazy crash-recovery via BackgroundOutput orphan correlation_id)
    // ========================================================================

    #[tokio::test]
    async fn test_recovery_inserts_orphan_monitor() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        let events_tx = session.events_tx.clone();

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-rec".into(), session);

        let inserted = ChatManager::track_background_task_recovery_if_orphan(
            "s-rec",
            &map,
            &events_tx,
            &None,
            Some("toolu_orphan_M"),
            "Monitor",
        )
        .await;
        assert!(inserted);

        // Map now has the recovered entry.
        let sessions_guard = map.read().await;
        let tasks = sessions_guard
            .get("s-rec")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(tasks.len(), 1);
        let entry = tasks.get("toolu_orphan_M").unwrap();
        assert_eq!(entry.kind, BackgroundTaskKind::Monitor);
        assert_eq!(entry.description, "(recovered after restart)");
        assert_eq!(
            entry.parent_tool_use_id.as_deref(),
            Some("toolu_orphan_M"),
            "parent_tool_use_id mirrors the recovered id"
        );
        drop(tasks);
        drop(sessions_guard);

        // Broadcast carries the snapshot.
        let event = events_rx.try_recv().expect("expected broadcast");
        match event {
            ChatEvent::ActiveTasksUpdate { tasks } => {
                assert_eq!(tasks.len(), 1);
                assert_eq!(tasks[0].id, "toolu_orphan_M");
            }
            other => panic!("expected ActiveTasksUpdate, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_recovery_inserts_orphan_bash_background() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let events_tx = session.events_tx.clone();

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-rec-b".into(), session);

        let inserted = ChatManager::track_background_task_recovery_if_orphan(
            "s-rec-b",
            &map,
            &events_tx,
            &None,
            Some("toolu_orphan_B"),
            "BashOutput",
        )
        .await;
        assert!(inserted);

        let sessions_guard = map.read().await;
        let tasks = sessions_guard
            .get("s-rec-b")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(
            tasks.get("toolu_orphan_B").unwrap().kind,
            BackgroundTaskKind::BashBackground
        );
    }

    #[tokio::test]
    async fn test_recovery_skips_already_tracked() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        let events_tx = session.events_tx.clone();

        // Pre-seed an entry under the same id.
        session.active_background_tasks.lock().await.insert(
            "toolu_existing".into(),
            BackgroundTaskInfo {
                id: "toolu_existing".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "original".into(),
                started_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                pid: None,
                parent_tool_use_id: Some("toolu_existing".into()),
                pending_removal_at: None,
            },
        );

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-rec-skip".into(), session);

        let inserted = ChatManager::track_background_task_recovery_if_orphan(
            "s-rec-skip",
            &map,
            &events_tx,
            &None,
            Some("toolu_existing"),
            "Monitor",
        )
        .await;
        assert!(!inserted, "already-tracked id must not trigger insertion");

        // Description preserved (we did NOT overwrite).
        let sessions_guard = map.read().await;
        let tasks = sessions_guard
            .get("s-rec-skip")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(tasks.get("toolu_existing").unwrap().description, "original");
        drop(tasks);
        drop(sessions_guard);

        // No broadcast on no-op.
        assert!(events_rx.try_recv().is_err(), "no broadcast for no-op");
    }

    #[tokio::test]
    async fn test_recovery_ignores_unknown_source() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let events_tx = session.events_tx.clone();

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-rec-unk".into(), session);

        for source in ["system", "WebFetch", "", "foobar"] {
            let inserted = ChatManager::track_background_task_recovery_if_orphan(
                "s-rec-unk",
                &map,
                &events_tx,
                &None,
                Some("toolu_X"),
                source,
            )
            .await;
            assert!(!inserted, "source `{}` should not trigger recovery", source);
        }

        let count = map
            .read()
            .await
            .get("s-rec-unk")
            .unwrap()
            .active_background_tasks
            .lock()
            .await
            .len();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_recovery_touches_last_seen_at_on_known_correlation_id() {
        // Path 1: known correlation_id → refresh last_seen_at, no broadcast,
        // returns false (no insertion, but the refresh is the silent
        // side-effect that drives idle-death detection).
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        let events_tx = session.events_tx.clone();

        // Pre-seed an old entry — last_seen_at deep in the past.
        let old = chrono::Utc::now() - chrono::Duration::hours(2);
        session.active_background_tasks.lock().await.insert(
            "tool_known".into(),
            BackgroundTaskInfo {
                id: "tool_known".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "watched a long time".into(),
                started_at: old,
                last_seen_at: old,
                pid: None,
                parent_tool_use_id: Some("tool_known".into()),
                pending_removal_at: None,
            },
        );

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-touch".into(), session);

        let inserted = ChatManager::track_background_task_recovery_if_orphan(
            "s-touch",
            &map,
            &events_tx,
            &None,
            Some("tool_known"),
            "Monitor",
        )
        .await;
        assert!(!inserted, "no insertion when id is already known");

        // last_seen_at refreshed.
        let sessions_guard = map.read().await;
        let tasks = sessions_guard
            .get("s-touch")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        let entry = tasks.get("tool_known").unwrap();
        assert!(
            entry.last_seen_at > old,
            "last_seen_at must be refreshed (was {:?}, now {:?})",
            old,
            entry.last_seen_at
        );
        drop(tasks);
        drop(sessions_guard);

        // No broadcast (path 1 is silent — tick-by-tick refreshes would
        // saturate the WebSocket on a chatty Monitor).
        assert!(events_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_recovery_no_correlation_id_is_noop() {
        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let (events_tx, _rx) = broadcast::channel(8);
        let inserted = ChatManager::track_background_task_recovery_if_orphan(
            "s-anywhere",
            &map,
            &events_tx,
            &None,
            None,
            "Monitor",
        )
        .await;
        assert!(!inserted);
    }

    // ========================================================================
    // A silent background task is not a dead one (idle-death vs. liveness)
    // ========================================================================

    /// One silent `Bash run_in_background` entry, last seen well past the
    /// idle-death threshold, in a session inserted under `sid`.
    async fn session_with_silent_background_task(
        sid: &str,
        pid: Option<u32>,
        cli_reported: usize,
    ) -> (
        Arc<RwLock<HashMap<String, ActiveSession>>>,
        broadcast::Sender<ChatEvent>,
    ) {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let events_tx = session.events_tx.clone();
        let now = chrono::Utc::now();
        let silent_since =
            now - chrono::Duration::seconds((BACKGROUND_TASK_IDLE_DEATH_SECS + 60) as i64);
        session.active_background_tasks.lock().await.insert(
            "tool_silent".into(),
            BackgroundTaskInfo {
                id: "tool_silent".into(),
                kind: BackgroundTaskKind::BashBackground,
                description: "cargo test > log 2>&1".into(),
                started_at: silent_since,
                last_seen_at: silent_since,
                pid,
                parent_tool_use_id: None,
                pending_removal_at: None,
            },
        );
        session
            .cli_background_tasks
            .store(cli_reported, Ordering::Relaxed);
        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert(sid.into(), session);
        (map, events_tx)
    }

    async fn silent_task_is_marked(
        map: &Arc<RwLock<HashMap<String, ActiveSession>>>,
        sid: &str,
    ) -> bool {
        let sessions = map.read().await;
        let tasks = sessions
            .get(sid)
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        tasks
            .get("tool_silent")
            .expect("the entry is never removed within one tick")
            .pending_removal_at
            .is_some()
    }

    /// The bug: 30 minutes without output and the task was purged although
    /// its process was still running; the session then had "no background
    /// work" and the idle cleanup closed it, killing the CLI and the command.
    /// A task whose process runs is kept, and nothing is broadcast.
    #[cfg(unix)]
    #[tokio::test]
    async fn test_tick_purge_keeps_a_silent_task_whose_process_is_running() {
        let (map, events_tx) =
            session_with_silent_background_task("s-silent-pid", Some(std::process::id()), 0).await;
        let mut events_rx = events_tx.subscribe();

        ChatManager::tick_purge_background_tasks(
            "s-silent-pid",
            &map,
            &events_tx,
            &None,
            Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
        )
        .await;

        assert!(
            !silent_task_is_marked(&map, "s-silent-pid").await,
            "a task whose process is running must not be marked for removal"
        );
        assert!(
            events_rx.try_recv().is_err(),
            "a quiet tick broadcasts nothing"
        );
    }

    /// No pid was discovered (the claim often fails), but the CLI reports one
    /// running background task: the entry is kept.
    #[tokio::test]
    async fn test_tick_purge_keeps_a_silent_task_while_the_cli_reports_work() {
        let (map, events_tx) = session_with_silent_background_task("s-silent-cli", None, 1).await;

        ChatManager::tick_purge_background_tasks(
            "s-silent-cli",
            &map,
            &events_tx,
            &None,
            Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
        )
        .await;
        assert!(!silent_task_is_marked(&map, "s-silent-cli").await);

        // The CLI then reports that nothing runs any more: the next tick
        // marks the entry, as before.
        map.read()
            .await
            .get("s-silent-cli")
            .unwrap()
            .cli_background_tasks
            .store(0, Ordering::Relaxed);
        ChatManager::tick_purge_background_tasks(
            "s-silent-cli",
            &map,
            &events_tx,
            &None,
            Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
        )
        .await;
        assert!(silent_task_is_marked(&map, "s-silent-cli").await);
    }

    /// A known pid whose process is gone is dead, whatever the CLI reports
    /// (its report is about some other task).
    #[cfg(unix)]
    #[tokio::test]
    async fn test_tick_purge_marks_a_silent_task_whose_process_is_gone() {
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn `true`");
        let dead_pid = child.id();
        child.wait().expect("wait for `true`");

        let (map, events_tx) =
            session_with_silent_background_task("s-silent-dead", Some(dead_pid), 1).await;

        ChatManager::tick_purge_background_tasks(
            "s-silent-dead",
            &map,
            &events_tx,
            &None,
            Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
        )
        .await;
        assert!(silent_task_is_marked(&map, "s-silent-dead").await);
    }

    // ========================================================================
    // T12 of plan 754a1379 — tick_purge_background_tasks (grace period purge)
    // ========================================================================

    #[tokio::test]
    async fn test_tick_purge_removes_stale_pending_entries_and_broadcasts() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        let events_tx = session.events_tx.clone();

        // Insert: one expired (pending_removal_at far in the past),
        // one fresh (no pending_removal_at).
        let now = std::time::Instant::now();
        let stale_at = now.checked_sub(Duration::from_secs(10)).unwrap_or(now);
        {
            let mut tasks = session.active_background_tasks.lock().await;
            tasks.insert(
                "tool_stale".into(),
                BackgroundTaskInfo {
                    id: "tool_stale".into(),
                    kind: BackgroundTaskKind::Monitor,
                    description: "stale".into(),
                    started_at: chrono::Utc::now(),
                    last_seen_at: chrono::Utc::now(),
                    pid: None,
                    parent_tool_use_id: None,
                    pending_removal_at: Some(stale_at),
                },
            );
            tasks.insert(
                "tool_alive".into(),
                BackgroundTaskInfo {
                    id: "tool_alive".into(),
                    kind: BackgroundTaskKind::BashBackground,
                    description: "alive".into(),
                    started_at: chrono::Utc::now(),
                    last_seen_at: chrono::Utc::now(),
                    pid: None,
                    parent_tool_use_id: None,
                    pending_removal_at: None,
                },
            );
        }

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-purge".into(), session);

        ChatManager::tick_purge_background_tasks(
            "s-purge",
            &map,
            &events_tx,
            &None,
            Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
        )
        .await;

        // Stale entry purged, alive entry kept.
        let sessions_guard = map.read().await;
        let tasks = sessions_guard
            .get("s-purge")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(tasks.len(), 1);
        assert!(tasks.contains_key("tool_alive"));
        assert!(!tasks.contains_key("tool_stale"));
        drop(tasks);
        drop(sessions_guard);

        // A single broadcast carrying the post-purge snapshot.
        let event = events_rx
            .try_recv()
            .expect("expected broadcast after purge");
        match event {
            ChatEvent::ActiveTasksUpdate { tasks } => {
                assert_eq!(tasks.len(), 1);
                assert_eq!(tasks[0].id, "tool_alive");
            }
            other => panic!("expected ActiveTasksUpdate, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_tick_purge_marks_idle_entries_as_pending_removal() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        let events_tx = session.events_tx.clone();

        // Pre-seed: one ALIVE entry (last_seen_at = now), one IDLE
        // entry (last_seen_at well past the death threshold).
        let now = chrono::Utc::now();
        let stale_seen =
            now - chrono::Duration::seconds((BACKGROUND_TASK_IDLE_DEATH_SECS + 60) as i64);
        {
            let mut tasks = session.active_background_tasks.lock().await;
            tasks.insert(
                "tool_alive".into(),
                BackgroundTaskInfo {
                    id: "tool_alive".into(),
                    kind: BackgroundTaskKind::Monitor,
                    description: "live".into(),
                    started_at: now,
                    last_seen_at: now,
                    pid: None,
                    parent_tool_use_id: None,
                    pending_removal_at: None,
                },
            );
            tasks.insert(
                "tool_idle".into(),
                BackgroundTaskInfo {
                    id: "tool_idle".into(),
                    kind: BackgroundTaskKind::BashBackground,
                    description: "silently dead".into(),
                    started_at: stale_seen,
                    last_seen_at: stale_seen,
                    pid: None,
                    parent_tool_use_id: None,
                    pending_removal_at: None,
                },
            );
        }

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-idle".into(), session);

        ChatManager::tick_purge_background_tasks(
            "s-idle",
            &map,
            &events_tx,
            &None,
            Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
        )
        .await;

        // Idle entry must now have pending_removal_at set.
        let sessions_guard = map.read().await;
        let tasks = sessions_guard
            .get("s-idle")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(tasks.len(), 2, "first tick only marks, doesn't purge yet");
        assert!(
            tasks
                .get("tool_alive")
                .unwrap()
                .pending_removal_at
                .is_none(),
            "alive entry must NOT be marked"
        );
        assert!(
            tasks.get("tool_idle").unwrap().pending_removal_at.is_some(),
            "idle entry MUST be marked"
        );
        drop(tasks);
        drop(sessions_guard);

        // Single broadcast carrying the marked-but-still-present snapshot.
        let event = events_rx.try_recv().expect("expected broadcast");
        match event {
            ChatEvent::ActiveTasksUpdate { tasks } => assert_eq!(tasks.len(), 2),
            other => panic!("expected ActiveTasksUpdate, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_tick_purge_within_grace_does_nothing() {
        let (session, _) = create_dummy_session(false, "", vec![]);
        let mut events_rx = session.events_tx.subscribe();
        let events_tx = session.events_tx.clone();

        // Mark just now → still within grace.
        let now = std::time::Instant::now();
        session.active_background_tasks.lock().await.insert(
            "tool_recent".into(),
            BackgroundTaskInfo {
                id: "tool_recent".into(),
                kind: BackgroundTaskKind::Monitor,
                description: "recent".into(),
                started_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                pid: None,
                parent_tool_use_id: None,
                pending_removal_at: Some(now),
            },
        );

        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        map.write().await.insert("s-grace".into(), session);

        ChatManager::tick_purge_background_tasks(
            "s-grace",
            &map,
            &events_tx,
            &None,
            Duration::from_secs(BACKGROUND_TASK_PURGE_GRACE_SECS),
        )
        .await;

        // Entry still present.
        let sessions_guard = map.read().await;
        let tasks = sessions_guard
            .get("s-grace")
            .unwrap()
            .active_background_tasks
            .lock()
            .await;
        assert_eq!(tasks.len(), 1);
        drop(tasks);
        drop(sessions_guard);

        // No broadcast — purge was a no-op.
        assert!(
            events_rx.try_recv().is_err(),
            "no broadcast expected when nothing was purged"
        );
    }

    #[tokio::test]
    async fn test_cancel_task_result_serde_round_trip() {
        let r = CancelTaskResult {
            task_id: "tool_abc".into(),
            killed_pids: vec![],
            capped: false,
        };
        let json = serde_json::to_string(&r).expect("serialize");
        assert!(json.contains("\"task_id\":\"tool_abc\""));
        assert!(json.contains("\"killed_pids\":[]"));
        assert!(json.contains("\"capped\":false"));
        let back: CancelTaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.task_id, "tool_abc");
        assert!(back.killed_pids.is_empty());
        assert!(!back.capped);
    }

    #[tokio::test]
    async fn test_track_background_task_unknown_session_returns_false() {
        let map: Arc<RwLock<HashMap<String, ActiveSession>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let (events_tx, _) = broadcast::channel(8);

        let inserted = ChatManager::track_background_task_start(
            "ghost-session",
            &map,
            &events_tx,
            &None,
            "tool_X",
            "Monitor",
            &serde_json::json!({}),
            None,
        )
        .await;
        assert!(!inserted);
    }
}

/// Fixtures for handler tests that need a LIVE session (a registered
/// `ActiveSession` whose CLI stdin is a channel the test can read).
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn chat_config() -> ChatConfig {
        ChatConfig::default()
    }

    /// Register a live session. Returns the receiver of what would be written
    /// to the CLI's stdin and the queue of messages received while streaming,
    /// or `None` when the Claude CLI binary is not installed (the dummy client
    /// needs it). Callers that assert something must `.expect` it: a test must
    /// never pass without verifying anything.
    pub(crate) async fn insert_live_session(
        manager: &ChatManager,
        session_id: &str,
        is_streaming: bool,
        pending_permissions: &[&str],
    ) -> Option<(
        tokio::sync::mpsc::Receiver<String>,
        Arc<Mutex<VecDeque<PendingMessage>>>,
    )> {
        let client = InteractiveClient::new(nexus_claude::ClaudeCodeOptions {
            model: Some("test".into()),
            ..Default::default()
        })
        .ok()?;
        insert_session_with_client(
            manager,
            session_id,
            client,
            is_streaming,
            pending_permissions,
        )
        .await
    }

    /// A live session whose "CLI" is an in-memory transport: the handle shows
    /// what the SDK writes to the CLI (`sent_input_rx`) and lets the test answer
    /// (`inbound_message_tx`). Needs no Claude CLI installed.
    pub(crate) async fn insert_mock_cli_session(
        manager: &ChatManager,
        session_id: &str,
    ) -> nexus_claude::transport::mock::MockTransportHandle {
        let (transport, handle) = nexus_claude::transport::mock::MockTransport::pair();
        let mut client = InteractiveClient::from_transport(transport);
        client.connect().await.expect("mock client connects");
        insert_session_with_client(manager, session_id, client, false, &[]).await;
        handle
    }

    /// Like `insert_live_session`, but needs no Claude CLI installed: the
    /// client is given an explicit (never spawned) binary path, so building
    /// it does not search the machine. For tests that only care that the
    /// session is registered as live.
    pub(crate) async fn insert_live_session_without_cli(manager: &ChatManager, session_id: &str) {
        let client = InteractiveClient::new(nexus_claude::ClaudeCodeOptions {
            model: Some("test".into()),
            cli_path: Some("/nonexistent/claude".into()),
            ..Default::default()
        })
        .expect("an explicit CLI path is never searched for");
        insert_session_with_client(manager, session_id, client, false, &[]).await;
    }

    pub(crate) async fn insert_session_with_client(
        manager: &ChatManager,
        session_id: &str,
        client: InteractiveClient,
        is_streaming: bool,
        pending_permissions: &[&str],
    ) -> Option<(
        tokio::sync::mpsc::Receiver<String>,
        Arc<Mutex<VecDeque<PendingMessage>>>,
    )> {
        let (events_tx, _rx) = broadcast::channel(16);
        let (stdin_tx, stdin_rx) = tokio::sync::mpsc::channel(16);
        let pending_messages = Arc::new(Mutex::new(VecDeque::<PendingMessage>::new()));
        let mut pending = HashMap::new();
        for id in pending_permissions {
            pending.insert((*id).to_string(), serde_json::json!({ "command": "ls" }));
        }
        let session = ActiveSession {
            anchor: Default::default(),
            events_tx,
            last_activity: Instant::now(),
            cli_session_id: None,
            client: Arc::new(Mutex::new(client)),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            memory_manager: None,
            next_seq: Arc::new(AtomicI64::new(1)),
            pending_messages: pending_messages.clone(),
            is_streaming: Arc::new(AtomicBool::new(is_streaming)),
            streaming_text: Arc::new(Mutex::new(String::new())),
            streaming_events: Arc::new(Mutex::new(Vec::new())),
            permission_mode: None,
            model: None,
            sdk_control_rx: Arc::new(tokio::sync::Mutex::new(None)),
            stdin_tx: Some(stdin_tx),
            child_pid: None,
            nats_cancel: CancellationToken::new(),
            interrupt_token: CancellationToken::new(),
            pending_permission_inputs: Arc::new(tokio::sync::Mutex::new(pending)),
            auto_continue: Arc::new(AtomicBool::new(false)),
            auto_continue_count: Arc::new(AtomicU32::new(0)),
            max_auto_continues: 0,
            rfc_accumulator: Arc::new(Mutex::new(
                crate::chat::observation_detector::RfcAccumulator::new(),
            )),
            protocol_run_id: None,
            protocol_state: None,
            reasoning_path_tracker: crate::chat::feedback::ReasoningPathTracker::new(),
            objective_tracking: false,
            objective_reminder_turns_since: Arc::new(AtomicU32::new(0)),
            objective_reminders_in_a_row: Arc::new(AtomicU32::new(0)),
            work_log: Arc::new(Mutex::new(SessionWorkLog::default())),
            oob_trigger_history: Arc::new(Mutex::new(VecDeque::new())),
            oob_trigger_cap: OOB_TRIGGER_CAP_INTERACTIVE,
            oob_trigger_window: Duration::from_secs(OOB_TRIGGER_WINDOW_SECS),
            oob_capped_warned: Arc::new(AtomicBool::new(false)),
            cancel_tools_history: Arc::new(Mutex::new(VecDeque::new())),
            cancel_tools_cap: CANCEL_TOOLS_CAP,
            cancel_tools_window: Duration::from_secs(CANCEL_TOOLS_WINDOW_SECS),
            active_background_tasks: Arc::new(Mutex::new(HashMap::new())),
            cli_background_tasks: Arc::new(AtomicUsize::new(0)),
            cancel_task_history: Arc::new(Mutex::new(VecDeque::new())),
            cancel_task_cap: CANCEL_TASK_CAP,
            cancel_task_window: Duration::from_secs(CANCEL_TASK_WINDOW_SECS),
        };
        manager
            .active_sessions
            .write()
            .await
            .insert(session_id.to_string(), session);
        Some((stdin_rx, pending_messages))
    }
}

#[cfg(test)]
mod agent_env_tests {
    use super::server_secrets_to_hide;

    #[test]
    fn only_the_server_secrets_that_exist_are_hidden() {
        let present = |k: &str| k == "NEO4J_PASSWORD" || k == "PO_JWT_SECRET";
        assert_eq!(
            server_secrets_to_hide(present),
            vec!["NEO4J_PASSWORD", "PO_JWT_SECRET"]
        );
        assert!(server_secrets_to_hide(|_| false).is_empty());
    }

    #[test]
    fn the_cli_keeps_the_credentials_it_needs_to_authenticate() {
        let hidden = server_secrets_to_hide(|_| true);
        for needed in [
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "PATH",
            "HOME",
        ] {
            assert!(
                !hidden.contains(&needed),
                "{needed} must stay visible to the CLI"
            );
        }
        for secret in [
            "NEO4J_PASSWORD",
            "MEILISEARCH_KEY",
            "EMBEDDING_API_KEY",
            "PO_JWT_SECRET",
        ] {
            assert!(
                hidden.contains(&secret),
                "{secret} must be hidden from agents"
            );
        }
    }
}

/// `ManagerTurnServices::prepare` × `refs::turn`: what the agent engine sends for
/// a turn whose message carries `#` references (H3 enrichment + refs_v1).
#[cfg(test)]
mod refs_turn_services_tests {
    use super::*;
    use crate::chat::agent_runtime::TurnServices;
    use crate::chat::enrichment::{
        EnrichmentConfig, EnrichmentInput, EnrichmentPipeline, EnrichmentSource,
        ParallelEnrichmentStage, StageOutput,
    };
    use crate::refs::types::{EntityRef, RefKind};
    use std::sync::Mutex as StdMutex;

    /// What one enrichment read: the message and the notes it must not inject.
    type Seen = (String, std::collections::HashSet<String>);

    /// Records what each enrichment reads, and answers one section.
    #[derive(Default)]
    struct Recording {
        seen: Arc<StdMutex<Vec<Seen>>>,
    }

    #[async_trait::async_trait]
    impl ParallelEnrichmentStage for Recording {
        async fn execute(&self, input: &EnrichmentInput) -> anyhow::Result<StageOutput> {
            self.seen
                .lock()
                .unwrap()
                .push((input.message.clone(), input.excluded_note_ids.clone()));
            let mut out = StageOutput::new("recording");
            out.add_section("CTX", "graph context", "recording", EnrichmentSource::Other);
            Ok(out)
        }
        fn name(&self) -> &str {
            "recording"
        }
        fn is_enabled(&self, _config: &EnrichmentConfig) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn prepare_enriches_the_typed_text_once_and_sends_the_pointers_behind_the_relay() {
        let mock = Arc::new(crate::neo4j::mock::MockGraphStore::new());
        let session = crate::test_helpers::test_chat_session(None);
        mock.create_chat_session(&session).await.unwrap();
        let graph: Arc<dyn GraphStore> = mock;
        let stage = Recording::default();
        let seen = Arc::clone(&stage.seen);
        let mut pipeline = EnrichmentPipeline::new(EnrichmentConfig::default());
        pipeline.add_parallel_stage(Box::new(stage));
        let services = ManagerTurnServices {
            graph: graph.clone(),
            enrichment_pipeline: Arc::new(pipeline),
            turn_routing: Arc::default(),
            nats: None,
            documents: crate::documents::store::DocumentStore::new(std::env::temp_dir()),
            anchor: Default::default(),
        };

        let note = Uuid::new_v4();
        let typed = "look at #note:x";
        let stored = crate::refs::block::encode(typed, &[EntityRef::new(RefKind::Note, note)]);
        let sent = format!("RELAY HISTORY\n\n{stored}");
        let turn = crate::refs::turn::expand_user_turn_if(&graph, &stored, true).await;
        let out = services
            .prepare(&session.id.to_string(), &stored, &sent, &turn)
            .await;

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1, "one enrichment per turn: {seen:?}");
        assert_eq!(seen[0].0, typed, "the enrichment reads the typed text");
        assert!(
            seen[0].1.contains(&note.to_string()),
            "the note pointed at is not injected again: {seen:?}"
        );
        assert!(
            out.starts_with("## CTX\ngraph context\n\n---\n\nRELAY HISTORY\n\nlook at #note:x\n\n<po-context nonce=\""),
            "{out}"
        );
        assert!(!out.contains("<po-refs>"), "{out}");
    }
}
